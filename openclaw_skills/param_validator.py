"""
Parameter Validator for OpenClaw Level 3 Skills
================================================
A security-focused parameter validation system that prevents
malicious or accidental resource exhaustion attacks.

This module validates all parameters before they reach the underlying
quant models, enforcing hard limits on computation.

Usage:
    from param_validator import validate_params, ValidationError
    
    try:
        validated = validate_params("monte_carlo", {"n_paths": 50000})
    except ValidationError as e:
        print(f"Invalid parameters: {e}")
"""

import re
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings


class ValidationError(Exception):
    """Raised when parameter validation fails."""
    
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


class WarningLevel(Enum):
    """Level of validation warning."""
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@dataclass
class ParamRule:
    """A single validation rule for a parameter."""
    name: str
    param_type: type
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None  # Regex pattern for strings
    required: bool = False
    default: Any = None
    description: str = ""
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a value against this rule.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Handle None values
        if value is None:
            if self.required:
                return False, f"Parameter '{self.name}' is required"
            return True, None
        
        # Type check
        if not isinstance(value, self.param_type):
            return False, f"Parameter '{self.name}' must be {self.param_type.__name__}, got {type(value).__name__}"
        
        # Numeric validations
        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                return False, f"Parameter '{self.name}' must be >= {self.min_value}, got {value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Parameter '{self.name}' must be <= {self.max_value}, got {value}"
        
        # String/list length validations
        if isinstance(value, (str, list, dict)):
            length = len(value)
            if self.min_length is not None and length < self.min_length:
                return False, f"Parameter '{self.name}' must have length >= {self.min_length}, got {length}"
            if self.max_length is not None and length > self.max_length:
                return False, f"Parameter '{self.name}' must have length <= {self.max_length}, got {length}"
        
        # Allowed values check
        if self.allowed_values is not None and value not in self.allowed_values:
            return False, f"Parameter '{self.name}' must be one of {self.allowed_values}, got {value}"
        
        # Pattern matching for strings
        if isinstance(value, str) and self.pattern is not None:
            if not re.match(self.pattern, value):
                return False, f"Parameter '{self.name}' must match pattern '{self.pattern}', got '{value}'"
        
        return True, None


# =============================================================================
# Skill-specific parameter rules
# =============================================================================

SKILL_PARAM_RULES: Dict[str, Dict[str, ParamRule]] = {
    "hmm": {
        "symbol": ParamRule(
            name="symbol",
            param_type=str,
            required=True,
            pattern=r"^[A-Z]{2,10}(USDT|USD|BTC|ETH)?$",
            description="Trading symbol (e.g., BTCUSDT, ETH)"
        ),
        "n_states": ParamRule(
            name="n_states",
            param_type=int,
            min_value=2,
            max_value=5,
            default=3,
            description="Number of hidden states (regimes)"
        ),
        "lookback_days": ParamRule(
            name="lookback_days",
            param_type=int,
            min_value=30,
            max_value=365,
            default=90,
            description="Days of historical data to use"
        ),
        "confidence_threshold": ParamRule(
            name="confidence_threshold",
            param_type=float,
            min_value=0.0,
            max_value=1.0,
            default=0.5,
            description="Minimum confidence for regime classification"
        )
    },
    
    "monte_carlo": {
        "initial_price": ParamRule(
            name="initial_price",
            param_type=(int, float),
            min_value=0.01,
            max_value=1_000_000,
            required=True,
            description="Starting price for simulation"
        ),
        "expected_return": ParamRule(
            name="expected_return",
            param_type=float,
            min_value=-1.0,  # -100%
            max_value=1.0,   # +100%
            default=0.0,
            description="Expected daily return (as decimal)"
        ),
        "volatility": ParamRule(
            name="volatility",
            param_type=float,
            min_value=0.001,
            max_value=2.0,  # 200% volatility
            required=True,
            description="Daily volatility (as decimal)"
        ),
        "n_paths": ParamRule(
            name="n_paths",
            param_type=int,
            min_value=100,
            max_value=20_000,  # Hard limit to prevent DoS
            default=5000,
            description="Number of simulation paths"
        ),
        "days_ahead": ParamRule(
            name="days_ahead",
            param_type=int,
            min_value=1,
            max_value=365,
            default=30,
            description="Forecast horizon in days"
        ),
        "random_seed": ParamRule(
            name="random_seed",
            param_type=int,
            min_value=0,
            max_value=999999999,
            description="Random seed for reproducibility"
        )
    },
    
    "garch": {
        "symbol": ParamRule(
            name="symbol",
            param_type=str,
            required=True,
            pattern=r"^[A-Z]{2,10}(USDT|USD|BTC|ETH)?$",
            description="Trading symbol"
        ),
        "p": ParamRule(
            name="p",
            param_type=int,
            min_value=1,
            max_value=3,
            default=1,
            description="GARCH lag order"
        ),
        "q": ParamRule(
            name="q",
            param_type=int,
            min_value=1,
            max_value=3,
            default=1,
            description="ARCH lag order"
        ),
        "forecast_horizon": ParamRule(
            name="forecast_horizon",
            param_type=int,
            min_value=1,
            max_value=30,
            default=5,
            description="Days to forecast volatility"
        ),
        "confidence_level": ParamRule(
            name="confidence_level",
            param_type=float,
            min_value=0.8,
            max_value=0.99,
            default=0.95,
            description="Confidence level for VaR"
        )
    },
    
    "portfolio_optimizer": {
        "assets": ParamRule(
            name="assets",
            param_type=list,
            min_length=2,
            max_length=50,  # Limit to prevent expensive optimization
            required=True,
            description="List of asset symbols"
        ),
        "objective": ParamRule(
            name="objective",
            param_type=str,
            allowed_values=["max_sharpe", "min_volatility", "max_return", "risk_parity"],
            default="max_sharpe",
            description="Optimization objective"
        ),
        "risk_free_rate": ParamRule(
            name="risk_free_rate",
            param_type=float,
            min_value=0.0,
            max_value=0.2,
            default=0.02,
            description="Risk-free rate for Sharpe calculation"
        ),
        "constraints": ParamRule(
            name="constraints",
            param_type=dict,
            max_length=10,
            description="Additional optimization constraints"
        )
    },
    
    "risk_engine": {
        "portfolio_value": ParamRule(
            name="portfolio_value",
            param_type=(int, float),
            min_value=0.01,
            max_value=1_000_000_000,
            required=True,
            description="Total portfolio value"
        ),
        "positions": ParamRule(
            name="positions",
            param_type=list,
            max_length=100,
            required=True,
            description="List of positions"
        ),
        "var_confidence": ParamRule(
            name="var_confidence",
            param_type=float,
            min_value=0.9,
            max_value=0.99,
            default=0.95,
            description="VaR confidence level"
        ),
        "var_horizon": ParamRule(
            name="var_horizon",
            param_type=int,
            min_value=1,
            max_value=30,
            default=1,
            description="VaR horizon in days"
        )
    }
}


# =============================================================================
# Global resource limits
# =============================================================================

GLOBAL_LIMITS = {
    "max_total_memory_mb": 512,
    "max_execution_time_ms": 5000,
    "max_iterations": 1_000_000,
    "max_api_calls_per_minute": 60,
    "max_concurrent_skills": 5,
    "max_cache_size_mb": 100
}


@dataclass
class ValidationContext:
    """Context for validation decisions."""
    user_tier: str = "standard"  # standard, premium, institutional
    ip_whitelist: List[str] = field(default_factory=list)
    audit_enabled: bool = True


# =============================================================================
# Main validation functions
# =============================================================================

def validate_params(
    skill: str,
    params: Dict[str, Any],
    context: Optional[ValidationContext] = None
) -> Dict[str, Any]:
    """
    Validate parameters for a skill.
    
    Args:
        skill: Skill identifier
        params: Parameters to validate
        context: Optional validation context
        
    Returns:
        Validated and normalized parameters
        
    Raises:
        ValidationError: If any parameter is invalid
    """
    errors: List[str] = []
    warnings: List[str] = []
    validated: Dict[str, Any] = {}
    
    # Get rules for this skill
    rules = SKILL_PARAM_RULES.get(skill, {})
    
    # Check for unknown parameters
    unknown_params = set(params.keys()) - set(rules.keys())
    if unknown_params:
        errors.append(f"Unknown parameters: {unknown_params}")
    
    # Validate each parameter
    for param_name, rule in rules.items():
        value = params.get(param_name)
        
        # Check if required and missing
        if value is None and rule.required:
            if rule.default is not None:
                # Use default value
                validated[param_name] = rule.default
                warnings.append(f"Using default for '{param_name}': {rule.default}")
            else:
                errors.append(f"Required parameter '{param_name}' is missing")
            continue
        
        # Validate the value
        if value is not None:
            is_valid, error_msg = rule.validate(value)
            if not is_valid:
                errors.append(error_msg)
            else:
                validated[param_name] = value
    
    # Apply tier-based limits
    if context and context.user_tier == "standard":
        # Apply stricter limits for standard users
        if skill == "monte_carlo":
            max_paths = 10000
            if validated.get("n_paths", 0) > max_paths:
                errors.append(f"Standard tier limited to {max_paths} paths")
    
    # Raise if there are errors
    if errors:
        raise ValidationError(errors)
    
    # Issue warnings if any
    if warnings:
        for warning in warnings:
            warnings.warn(f"Validation warning: {warning}")
    
    return validated


def get_skill_limits(skill: str) -> Dict[str, Any]:
    """
    Get the resource limits for a skill.
    
    Args:
        skill: Skill identifier
        
    Returns:
        Dictionary of limit settings
    """
    base_limits = {
        "max_runtime_ms": 2000,
        "max_memory_mb": 200,
        "max_iterations": 100000,
        "cache_ttl_seconds": 3600
    }
    
    # Skill-specific overrides
    skill_limits = {
        "hmm": {
            "max_runtime_ms": 5000,
            "max_memory_mb": 256,
            "cache_ttl_seconds": 3600  # 1 hour
        },
        "monte_carlo": {
            "max_runtime_ms": 3000,
            "max_memory_mb": 512,
            "max_iterations": 500000,
            "cache_ttl_seconds": 600  # 10 minutes
        },
        "garch": {
            "max_runtime_ms": 4000,
            "max_memory_mb": 256,
            "cache_ttl_seconds": 3600
        },
        "portfolio_optimizer": {
            "max_runtime_ms": 5000,
            "max_memory_mb": 512,
            "cache_ttl_seconds": 1800  # 30 minutes
        }
    }
    
    return {**base_limits, **skill_limits.get(skill, {})}


def check_resource_availability(
    skill: str,
    params: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Check if resources are available for a skill execution.
    
    Args:
        skill: Skill to check
        params: Parameters that affect resource usage
        
    Returns:
        Tuple of (is_available, reasons)
    """
    reasons: List[str] = []
    
    # Estimate memory usage
    if skill == "monte_carlo":
        n_paths = params.get("n_paths", 5000)
        days_ahead = params.get("days_ahead", 30)
        
        # Rough memory estimate: n_paths * days_ahead * 8 bytes * 2 (arrays)
        estimated_memory_mb = (n_paths * days_ahead * 8 * 2) / (1024 * 1024)
        
        if estimated_memory_mb > GLOBAL_LIMITS["max_total_memory_mb"]:
            reasons.append(
                f"Estimated memory {estimated_memory_mb:.1f}MB exceeds limit {GLOBAL_LIMITS['max_total_memory_mb']}MB"
            )
    
    # Check iteration limits
    if skill == "portfolio_optimizer":
        n_assets = len(params.get("assets", []))
        # Rough estimate: quadratic in number of assets
        estimated_iterations = n_assets ** 2 * 1000
        
        if estimated_iterations > GLOBAL_LIMITS["max_iterations"]:
            reasons.append(
                f"Estimated iterations {estimated_iterations} exceeds limit"
            )
    
    return len(reasons) == 0, reasons


def sanitize_input(value: Any, param_type: type) -> Any:
    """
    Sanitize input value to prevent injection attacks.
    
    Args:
        value: Input value
        param_type: Expected type
        
    Returns:
        Sanitized value
    """
    if value is None:
        return None
    
    # Strip whitespace from strings
    if isinstance(value, str):
        value = value.strip()
        
        # Remove potentially dangerous characters
        dangerous_chars = [';', '&', '|', '`', '$', '(', ')']
        for char in dangerous_chars:
            if char in value:
                value = value.replace(char, '')
    
    # Recursively sanitize dicts and lists
    if isinstance(value, dict):
        return {k: sanitize_input(v, type(v)) for k, v in value.items()}
    
    if isinstance(value, list):
        return [sanitize_input(item, type(item)) for item in value]
    
    return value


# =============================================================================
# Decorator for easy validation
# =============================================================================

def validated(skill: str):
    """
    Decorator to automatically validate parameters for a skill function.
    
    Usage:
        @validated("monte_carlo")
        def run_monte_carlo(params):
            # params is already validated
            pass
    """
    def decorator(func):
        def wrapper(params: Dict[str, Any], *args, **kwargs):
            validated_params = validate_params(skill, params)
            return func(validated_params, *args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test valid parameters
    print("=== Testing Valid Parameters ===")
    try:
        result = validate_params("monte_carlo", {
            "initial_price": 50000,
            "volatility": 0.03,
            "n_paths": 5000,
            "days_ahead": 30
        })
        print(f"Valid: {result}")
    except ValidationError as e:
        print(f"Error: {e.errors}")
    
    print("\n=== Testing Invalid Parameters ===")
    try:
        result = validate_params("monte_carlo", {
            "initial_price": 50000,
            "volatility": 0.03,
            "n_paths": 100000,  # Too many paths
            "days_ahead": 30
        })
        print(f"Valid: {result}")
    except ValidationError as e:
        print(f"Error: {e.errors}")
    
    print("\n=== Testing Missing Required Parameter ===")
    try:
        result = validate_params("hmm", {
            "n_states": 3
            # Missing required 'symbol'
        })
        print(f"Valid: {result}")
    except ValidationError as e:
        print(f"Error: {e.errors}")
    
    print("\n=== Testing Skill Limits ===")
    limits = get_skill_limits("monte_carlo")
    print(f"Monte Carlo limits: {limits}")
