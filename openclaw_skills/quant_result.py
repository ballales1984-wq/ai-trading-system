"""
QuantResult Standard Format
===========================
A unified output format for all OpenClaw Level 3 quant skills.

This module provides the standard QuantResult class that all quant skills
must return, ensuring consistency, security, and proper formatting.

Usage:
    from quant_result import QuantResult, create_result
    
    result = create_result(
        skill="hmm",
        metrics={"regime": "bull", "confidence": 0.87},
        recommendations=["Increase exposure", "Momentum strategy"]
    )
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import hashlib


class SkillType(Enum):
    """Enumeration of available quant skills."""
    HMM = "hmm"
    MONTE_CARLO = "monte_carlo"
    GARCH = "garch"
    PORTFOLIO_OPTIMIZER = "portfolio_optimizer"
    RISK_ENGINE = "risk_engine"
    REGIME_DETECTION = "regime_detection"
    VOLATILITY_FORECAST = "volatility_forecast"
    UNKNOWN = "unknown"


class RiskLevel(Enum):
    """Risk level classification for results."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ExecutionStatus(Enum):
    """Status of skill execution."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"


@dataclass
class QuantResult:
    """
    Standardized result format for all OpenClaw quant skills.
    
    Attributes:
        skill: The skill that generated this result
        timestamp: When the result was generated
        status: Execution status (success/failed/partial/timeout)
        execution_time_ms: How long the skill took to execute
        metadata: Additional context (inputs, config, etc.)
        metrics: Quantitative results (regime, VaR, weights, etc.)
        recommendations: Actionable advice based on results
        risk_level: Risk assessment (low/medium/high/critical)
        warnings: Any warnings or caveats
        errors: Any errors that occurred
        raw: Raw data from the underlying model
        cache_key: Key for caching this result
        version: Format version for compatibility
    """
    
    skill: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: str = ExecutionStatus.SUCCESS.value
    execution_time_ms: float = 0.0
    
    # Core data
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    # Risk and warnings
    risk_level: str = RiskLevel.LOW.value
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Raw data for advanced users
    raw: Dict[str, Any] = field(default_factory=dict)
    
    # Caching
    cache_key: Optional[str] = None
    cache_ttl_seconds: int = 3600  # Default 1 hour
    
    # Versioning
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Validate and normalize fields after initialization."""
        # Ensure timestamp is ISO format string for JSON serialization
        if isinstance(self.timestamp, datetime):
            self.timestamp = self.timestamp.isoformat()
        
        # Validate status
        valid_statuses = [s.value for s in ExecutionStatus]
        if self.status not in valid_statuses:
            self.status = ExecutionStatus.FAILED.value
            self.errors.append(f"Invalid status: {self.status}")
        
        # Validate risk level
        valid_risks = [r.value for r in RiskLevel]
        if self.risk_level not in valid_risks:
            self.risk_level = RiskLevel.MEDIUM.value
        
        # Generate cache key if not provided
        if not self.cache_key:
            self.cache_key = self._generate_cache_key()
    
    def _generate_cache_key(self) -> str:
        """Generate a unique cache key based on skill and metadata."""
        key_data = f"{self.skill}:{self.timestamp}:{json.dumps(self.metadata, sort_keys=True)}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def format_for_chat(self) -> str:
        """
        Format the result as a readable chat message.
        
        This creates the output that users see in the conversation.
        """
        # Header with skill and status
        status_emoji = {
            ExecutionStatus.SUCCESS.value: "✅",
            ExecutionStatus.FAILED.value: "❌",
            ExecutionStatus.PARTIAL.value: "⚠️",
            ExecutionStatus.TIMEOUT.value: "⏱️",
            ExecutionStatus.VALIDATION_ERROR.value: "🔴"
        }
        
        emoji = status_emoji.get(self.status, "❓")
        header = f"{emoji} **{self.skill.upper()} Result**"
        
        lines = [header, ""]
        
        # Execution time
        if self.execution_time_ms > 0:
            lines.append(f"⏱️ Execution time: {self.execution_time_ms:.1f}ms")
            lines.append("")
        
        # Risk level indicator
        risk_emoji = {
            RiskLevel.LOW.value: "🟢",
            RiskLevel.MEDIUM.value: "🟡",
            RiskLevel.HIGH.value: "🟠",
            RiskLevel.CRITICAL.value: "🔴"
        }
        risk = risk_emoji.get(self.risk_level, "⚪")
        lines.append(f"{risk} Risk Level: {self.risk_level.upper()}")
        lines.append("")
        
        # Metrics section
        if self.metrics:
            lines.append("📊 **Metrics:**")
            for key, value in self.metrics.items():
                formatted_value = self._format_value(value)
                lines.append(f"  - {key}: {formatted_value}")
            lines.append("")
        
        # Recommendations section
        if self.recommendations:
            lines.append("📋 **Recommendations:**")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")
            lines.append("")
        
        # Warnings section
        if self.warnings:
            lines.append("⚠️ **Warnings:**")
            for warning in self.warnings:
                lines.append(f"  • {warning}")
            lines.append("")
        
        # Errors section
        if self.errors:
            lines.append("❌ **Errors:**")
            for error in self.errors:
                lines.append(f"  • {error}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_value(self, value: Any) -> str:
        """Format a value for display."""
        if isinstance(value, float):
            if abs(value) < 1:
                return f"{value:.2%}"
            elif abs(value) < 100:
                return f"{value:.2f}"
            else:
                return f"{value:,.2f}"
        elif isinstance(value, dict):
            return json.dumps(value, default=str)
        elif isinstance(value, list):
            if len(value) > 5:
                return f"[{len(value)} items]"
            return str(value)
        else:
            return str(value)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantResult":
        """Create a QuantResult from a dictionary."""
        # Handle datetime conversion
        if "timestamp" in data and isinstance(data["timestamp"], str):
            try:
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
            except (ValueError, TypeError):
                data["timestamp"] = datetime.utcnow()
        
        return cls(**data)
    
    @classmethod
    def create_error(
        cls,
        skill: str,
        error_message: str,
        execution_time_ms: float = 0.0
    ) -> "QuantResult":
        """
        Factory method to create an error result.
        
        Args:
            skill: The skill that failed
            error_message: Description of the error
            execution_time_ms: Time taken before failure
            
        Returns:
            QuantResult with error status
        """
        return cls(
            skill=skill,
            status=ExecutionStatus.FAILED.value,
            execution_time_ms=execution_time_ms,
            errors=[error_message],
            risk_level=RiskLevel.HIGH.value
        )
    
    @classmethod
    def create_validation_error(
        cls,
        skill: str,
        validation_errors: List[str]
    ) -> "QuantResult":
        """
        Factory method to create a validation error result.
        
        Args:
            skill: The skill with invalid parameters
            validation_errors: List of validation error messages
            
        Returns:
            QuantResult with validation error status
        """
        return cls(
            skill=skill,
            status=ExecutionStatus.VALIDATION_ERROR.value,
            errors=validation_errors,
            risk_level=RiskLevel.MEDIUM.value
        )


def create_result(
    skill: str,
    metrics: Optional[Dict[str, Any]] = None,
    recommendations: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    status: str = ExecutionStatus.SUCCESS.value,
    risk_level: str = RiskLevel.LOW.value,
    execution_time_ms: float = 0.0,
    warnings: Optional[List[str]] = None,
    raw: Optional[Dict[str, Any]] = None,
    cache_ttl_seconds: int = 3600
) -> QuantResult:
    """
    Convenience function to create a QuantResult.
    
    Args:
        skill: Skill identifier
        metrics: Quantitative results
        recommendations: Actionable advice
        metadata: Additional context
        status: Execution status
        risk_level: Risk assessment
        execution_time_ms: Execution time
        warnings: Warning messages
        raw: Raw model output
        cache_ttl_seconds: Cache duration
        
    Returns:
        Standardized QuantResult
    """
    return QuantResult(
        skill=skill,
        metrics=metrics or {},
        recommendations=recommendations or [],
        metadata=metadata or {},
        status=status,
        risk_level=risk_level,
        execution_time_ms=execution_time_ms,
        warnings=warnings or [],
        raw=raw or {},
        cache_ttl_seconds=cache_ttl_seconds
    )


# =============================================================================
# Integration helpers for existing skills
# =============================================================================

def wrap_skill_result(
    skill_name: str,
    skill_output: Dict[str, Any],
    execution_time_ms: float
) -> QuantResult:
    """
    Wrap an existing skill's output into a QuantResult.
    
    This allows gradual migration of existing skills to the new format.
    
    Args:
        skill_name: Name of the skill
        skill_output: Original skill output dictionary
        execution_time_ms: How long the skill took
        
    Returns:
        QuantResult with standardized format
    """
    # Extract known fields from skill output
    metrics = skill_output.get("metrics", skill_output)
    recommendations = skill_output.get("recommendations", [])
    warnings = skill_output.get("warnings", [])
    errors = skill_output.get("errors", [])
    
    # Determine risk level from metrics if not explicitly set
    risk_level = skill_output.get("risk_level", RiskLevel.LOW.value)
    
    return QuantResult(
        skill=skill_name,
        execution_time_ms=execution_time_ms,
        metrics=metrics,
        recommendations=recommendations if isinstance(recommendations, list) else [recommendations],
        warnings=warnings if isinstance(warnings, list) else [warnings],
        errors=errors if isinstance(errors, list) else [errors],
        risk_level=risk_level,
        metadata=skill_output.get("metadata", {}),
        raw=skill_output.get("raw", {})
    )


# =============================================================================
# Example usage and testing
# =============================================================================

if __name__ == "__main__":
    # Example 1: Create a basic result
    result = create_result(
        skill="hmm",
        metrics={
            "regime": "bull",
            "confidence": 0.87,
            "volatility": 0.025
        },
        recommendations=[
            "Increase exposure to risk assets",
            "Use momentum strategy",
            "Set stop-loss at 5%"
        ],
        risk_level="low"
    )
    
    print("=== Basic Result ===")
    print(result.format_for_chat())
    print()
    
    # Example 2: Create an error result
    error_result = QuantResult.create_error(
        skill="garch",
        error_message="Insufficient data: need at least 100 data points",
        execution_time_ms=45.2
    )
    
    print("=== Error Result ===")
    print(error_result.format_for_chat())
    print()
    
    # Example 3: JSON output
    print("=== JSON Output ===")
    print(result.to_json())
