# OpenClaw Skills Package
# Institutional-grade quantitative trading skills

__version__ = "3.0.0"

# Core modules
from .skill_registry import SkillRegistry, get_registry, list_skills, get_skill, is_skill_enabled
from .intent_router import IntentRouter, Intent, IntentType, SkillExecutor, route_message, route_intent
from .sandbox import SandboxError, SandboxResult, run_sandboxed, run_sandboxed_safe, validate_code_safety
from .composed_strategies import (
    regime_aware_mc_portfolio,
    full_risk_analysis,
    smart_portfolio_optimization,
    get_available_compositions,
    describe_composition,
    RegimeAwareParams,
    RiskAnalysisParams,
    PortfolioOptimizationParams,
)

__all__ = [
    "__version__",
    "SkillRegistry",
    "get_registry",
    "list_skills",
    "get_skill",
    "is_skill_enabled",
    "IntentRouter",
    "Intent",
    "IntentType",
    "SkillExecutor",
    "route_message",
    "route_intent",
    "SandboxError",
    "SandboxResult",
    "run_sandboxed",
    "run_sandboxed_safe",
    "validate_code_safety",
    "regime_aware_mc_portfolio",
    "full_risk_analysis",
    "smart_portfolio_optimization",
    "get_available_compositions",
    "describe_composition",
    "RegimeAwareParams",
    "RiskAnalysisParams",
    "PortfolioOptimizationParams",
]
