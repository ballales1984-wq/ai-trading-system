"""
Skill Orchestrator
==================
Coordinates execution of multiple quant skills in pipelines.

This module provides:
- Pipeline definition and execution
- Data passing between skills
- Result aggregation and formatting
- Error handling and recovery
- Execution monitoring

Usage:
    from skill_orchestrator import SkillOrchestrator
    
    orchestrator = SkillOrchestrator()
    result = orchestrator.execute_pipeline("full_risk_analysis", {
        "symbol": "BTC",
        "portfolio_value": 100000
    })
"""

import time
import yaml
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging

from quant_result import QuantResult, create_result, ExecutionStatus, RiskLevel
from param_validator import validate_params, get_skill_limits, ValidationError


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Status of pipeline execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class PipelineStep:
    """A single step in a pipeline."""
    skill: str
    input_from: str = "user"  # user, previous_step, or named output
    output_to: str = "result"
    combine: bool = False  # If true, combine with previous output
    condition: Optional[str] = None  # Conditional execution


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""
    pipeline_id: str
    status: PipelineStatus
    steps: List[Dict[str, Any]] = field(default_factory=list)
    aggregated_result: Optional[QuantResult] = None
    execution_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Skill Registry Loader
# =============================================================================

class SkillRegistry:
    """Loads and manages skill registry."""
    
    def __init__(self, registry_path: Optional[str] = None):
        self.registry_path = registry_path
        self.skills: Dict[str, Dict[str, Any]] = {}
        self.pipelines: Dict[str, Dict[str, Any]] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load the skill registry from YAML."""
        if not self.registry_path:
            self.registry_path = "openclaw_skills/skill_registry.yaml"
        
        try:
            with open(self.registry_path, 'r') as f:
                registry = yaml.safe_load(f)
                self.skills = registry.get('skills', {})
                self.pipelines = registry.get('pipelines', {})
                logger.info(f"Loaded {len(self.skills)} skills and {len(self.pipelines)} pipelines")
        except FileNotFoundError:
            logger.warning(f"Registry not found at {self.registry_path}, using defaults")
            self._load_defaults()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing registry: {e}")
            self._load_defaults()
    
    def _load_defaults(self):
        """Load default skills and pipelines."""
        self.skills = {
            'hmm': {'module': 'hmm_regime_detect', 'function': 'detect_regimes'},
            'monte_carlo': {'module': 'monte_carlo_paths', 'function': 'generate_price_paths'},
            'garch': {'module': 'garch_volatility', 'function': 'fit_garch'},
            'portfolio_optimizer': {'module': 'portfolio_optimizer', 'function': 'optimize_portfolio'}
        }
        self.pipelines = {}
    
    def get_skill_config(self, skill: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a skill."""
        return self.skills.get(skill)
    
    def get_pipeline_config(self, pipeline: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a pipeline."""
        return self.pipelines.get(pipeline)


# =============================================================================
# Result Cache
# =============================================================================

class ResultCache:
    """Simple in-memory cache for skill results."""
    
    def __init__(self, max_size_mb: int = 100):
        self.max_size_mb = max_size_mb
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.timestamps: Dict[str, datetime] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get a cached result."""
        if key in self.cache:
            # Check if expired
            timestamp = self.timestamps.get(key)
            if timestamp:
                age = (datetime.utcnow() - timestamp).total_seconds()
                if age > 3600:  # 1 hour default
                    del self.cache[key]
                    del self.timestamps[key]
                    return None
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Cache a result."""
        self.cache[key] = value
        self.timestamps[key] = datetime.utcnow()
    
    def clear(self):
        """Clear all cached results."""
        self.cache.clear()
        self.timestamps.clear()


# =============================================================================
# Skill Orchestrator
# =============================================================================

class SkillOrchestrator:
    """
    Orchestrates execution of skills and pipelines.
    
    This is the main entry point for executing quant skills
    with proper validation, caching, and error handling.
    """
    
    def __init__(
        self,
        registry_path: Optional[str] = None,
        enable_cache: bool = True
    ):
        """
        Initialize the orchestrator.
        
        Args:
            registry_path: Path to skill registry YAML
            enable_cache: Whether to enable result caching
        """
        self.registry = SkillRegistry(registry_path)
        self.cache = ResultCache() if enable_cache else None
        self.skill_executors: Dict[str, Callable] = {}
        self.execution_history: List[Dict[str, Any]] = []
    
    def execute(
        self,
        skill: str,
        params: Dict[str, Any],
        use_cache: bool = True
    ) -> QuantResult:
        """
        Execute a single skill.
        
        Args:
            skill: Skill identifier
            params: Skill parameters
            use_cache: Whether to use cached results
            
        Returns:
            QuantResult from skill execution
        """
        start_time = time.time()
        
        # Check cache
        if use_cache and self.cache:
            cache_key = f"{skill}:{json.dumps(params, sort_keys=True)}"
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Cache hit for {skill}")
                return QuantResult.from_dict(cached)
        
        # Validate parameters
        try:
            validated_params = validate_params(skill, params)
        except ValidationError as e:
            return QuantResult.create_validation_error(skill, e.errors)
        
        # Get skill limits
        limits = get_skill_limits(skill)
        
        # Execute skill
        try:
            result = self._execute_skill(skill, validated_params)
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Wrap result in QuantResult if needed
            if isinstance(result, dict):
                from quant_result import wrap_skill_result
                result = wrap_skill_result(skill, result, execution_time_ms)
            
            # Check if we exceeded limits
            if execution_time_ms > limits.get('max_runtime_ms', 5000):
                result.warnings.append(
                    f"Execution time {execution_time_ms:.0f}ms exceeded limit"
                )
            
            # Cache the result
            if use_cache and self.cache and result.status == ExecutionStatus.SUCCESS.value:
                self.cache.set(cache_key, result.to_dict(), limits.get('cache_ttl_seconds', 3600))
            
            return result
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return QuantResult.create_error(skill, str(e), execution_time_ms)
    
    def _execute_skill(self, skill: str, params: Dict[str, Any]) -> Any:
        """Execute a skill by name."""
        # Try to use registered executor first
        if skill in self.skill_executors:
            return self.skill_executors[skill](params)
        
        # Otherwise, try to import and execute
        skill_config = self.registry.get_skill_config(skill)
        if not skill_config:
            raise ValueError(f"Unknown skill: {skill}")
        
        module_name = skill_config.get('module')
        function_name = skill_config.get('function')
        
        # Import module
        try:
            import importlib
            module = importlib.import_module(module_name)
            function = getattr(module, function_name)
            return function(**params)
        except ImportError as e:
            raise RuntimeError(f"Missing dependency for {skill}: {e}")
        except AttributeError as e:
            raise RuntimeError(f"Function not found in {module_name}: {e}")
    
    def execute_pipeline(
        self,
        pipeline_id: str,
        params: Dict[str, Any]
    ) -> PipelineResult:
        """
        Execute a pipeline of skills.
        
        Args:
            pipeline_id: Pipeline identifier
            params: Initial parameters
            
        Returns:
            PipelineResult with aggregated outputs
        """
        start_time = time.time()
        
        # Get pipeline config
        pipeline_config = self.registry.get_pipeline_config(pipeline_id)
        if not pipeline_config:
            return PipelineResult(
                pipeline_id=pipeline_id,
                status=PipelineStatus.FAILED,
                errors=[f"Unknown pipeline: {pipeline_id}"]
            )
        
        steps_config = pipeline_config.get('steps', [])
        
        # Execute each step
        step_results = {}
        step_errors = []
        step_warnings = []
        
        current_params = params.copy()
        
        for i, step_config in enumerate(steps_config):
            skill = step_config.get('skill')
            output_to = step_config.get('output_to', f'step_{i}')
            combine = step_config.get('combine', False)
            
            # Determine input parameters
            input_from = step_config.get('input_from', 'user')
            
            # Handle case where input_from is a list (multiple inputs)
            if isinstance(input_from, list):
                step_params = params.copy()
                for src in input_from:
                    if src in step_results:
                        extracted = self._extract_params_from_result(step_results[src], params)
                        step_params.update(extracted)
            elif input_from == 'user':
                step_params = params.copy()
            elif input_from in step_results:
                # Use output from previous step
                prev_result = step_results[input_from]
                step_params = self._extract_params_from_result(prev_result, params)

                if combine:
                    # Merge with original params
                    step_params = {**params, **step_params}
            else:
                step_params = params.copy()
            
            # Execute skill
            try:
                result = self.execute(skill, step_params)
                step_results[output_to] = result
                
                if result.status != ExecutionStatus.SUCCESS.value:
                    step_errors.append(f"Step {i} ({skill}): {result.errors}")
                
                if result.warnings:
                    step_warnings.extend(result.warnings)
                    
            except Exception as e:
                step_errors.append(f"Step {i} ({skill}): {str(e)}")
        
        # Aggregate results
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Determine status
        if step_errors:
            status = PipelineStatus.PARTIAL if step_results else PipelineStatus.FAILED
        else:
            status = PipelineStatus.COMPLETED
        
        # Create aggregated result
        aggregated = self._aggregate_results(step_results)
        
        return PipelineResult(
            pipeline_id=pipeline_id,
            status=status,
            steps=[
                {'config': s, 'result': r.to_dict() if isinstance(r, QuantResult) else r}
                for s, r in zip(steps_config, list(step_results.values()))
            ],
            aggregated_result=aggregated,
            execution_time_ms=execution_time_ms,
            errors=step_errors,
            warnings=step_warnings,
            metadata={'total_steps': len(steps_config)}
        )
    
    def _extract_params_from_result(
        self,
        result: QuantResult,
        base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract relevant parameters from a result for the next step."""
        extracted = {}
        
        # Extract regime info for Monte Carlo
        if result.skill == 'hmm':
            if 'regime' in result.metrics:
                extracted['regime'] = result.metrics['regime']
            if 'volatility' in result.metrics:
                extracted['volatility'] = result.metrics['volatility'] / 100  # Convert from percentage
        
        # Extract volatility for Monte Carlo
        elif result.skill == 'garch':
            if 'current_volatility' in result.metrics:
                extracted['volatility'] = result.metrics['current_volatility']
        
        # Extract weights for Monte Carlo
        elif result.skill == 'portfolio_optimizer':
            if 'weights' in result.metrics:
                extracted['weights'] = result.metrics['weights']
        
        return extracted
    
    def _aggregate_results(
        self,
        step_results: Dict[str, QuantResult]
    ) -> Optional[QuantResult]:
        """Aggregate multiple step results into one."""
        if not step_results:
            return None
        
        # Combine all metrics
        all_metrics = {}
        all_recommendations = []
        all_warnings = []
        all_errors = []
        max_risk_level = RiskLevel.LOW
        
        for name, result in step_results.items():
            # Prefix metrics with step name
            for key, value in result.metrics.items():
                all_metrics[f"{name}_{key}"] = value
            
            all_recommendations.extend(result.recommendations)
            all_warnings.extend(result.warnings)
            all_errors.extend(result.errors)
            
            # Update max risk level
            result_risk = RiskLevel(result.risk_level)
            if result_risk.value > max_risk_level.value:
                max_risk_level = result_risk
        
        return create_result(
            skill="pipeline",
            metrics=all_metrics,
            recommendations=all_recommendations,
            warnings=all_warnings,
            risk_level=max_risk_level.value,
            metadata={'steps': list(step_results.keys()), 'errors': all_errors}
        )
    
    def register_executor(self, skill: str, executor: Callable):
        """Register a custom skill executor."""
        self.skill_executors[skill] = executor
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self.execution_history


# =============================================================================
# Convenience functions
# =============================================================================

def run_full_risk_analysis(symbol: str, portfolio_value: float = 100000) -> PipelineResult:
    """
    Run a complete risk analysis pipeline.
    
    This combines HMM regime detection, GARCH volatility, and Monte Carlo simulation.
    
    Args:
        symbol: Trading symbol
        portfolio_value: Portfolio value for risk calculations
        
    Returns:
        Aggregated risk analysis results
    """
    orchestrator = SkillOrchestrator()
    return orchestrator.execute_pipeline("full_risk_analysis", {
        "symbol": symbol,
        "portfolio_value": portfolio_value
    })


def run_smart_portfolio(
    assets: List[str],
    portfolio_value: float = 100000
) -> PipelineResult:
    """
    Run a smart portfolio optimization with regime awareness.
    
    Args:
        assets: List of asset symbols
        portfolio_value: Portfolio value
        
    Returns:
        Optimized portfolio with risk analysis
    """
    orchestrator = SkillOrchestrator()
    return orchestrator.execute_pipeline("smart_portfolio", {
        "assets": assets,
        "portfolio_value": portfolio_value
    })


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=== Skill Orchestrator Test ===\n")
    
    # Test single skill execution
    orchestrator = SkillOrchestrator()
    
    print("1. Testing single skill (HMM)...")
    result = orchestrator.execute("hmm", {
        "symbol": "BTCUSDT",
        "n_states": 3
    })
    print(f"   Status: {result.status}")
    print(f"   Metrics: {result.metrics}")
    print()
    
    print("2. Testing single skill (Monte Carlo)...")
    result = orchestrator.execute("monte_carlo", {
        "initial_price": 50000,
        "volatility": 0.03,
        "n_paths": 1000,
        "days_ahead": 30
    })
    print(f"   Status: {result.status}")
    print(f"   Metrics: {result.metrics}")
    print()
    
    print("3. Testing parameter validation...")
    try:
        result = orchestrator.execute("monte_carlo", {
            "initial_price": 50000,
            "volatility": 0.03,
            "n_paths": 100000  # Too many paths
        })
        print(f"   Status: {result.status}")
        print(f"   Errors: {result.errors}")
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    print("4. Testing pipeline execution...")
    pipeline_result = orchestrator.execute_pipeline("full_risk_analysis", {
        "symbol": "BTCUSDT"
    })
    print(f"   Pipeline Status: {pipeline_result.status.value}")
    print(f"   Execution Time: {pipeline_result.execution_time_ms:.1f}ms")
    print(f"   Steps: {len(pipeline_result.steps)}")
    print(f"   Errors: {pipeline_result.errors}")
    print()
    
    print("=== Tests Complete ===")
