"""
Intent Parser and Skill Router
===============================
Routes user messages to the appropriate quant skill using NLP-like matching.

This module provides:
- Intent classification using keyword matching
- Entity extraction (symbols, numbers, dates)
- Parameter inference from natural language
- Multi-skill detection for complex queries

Usage:
    from intent_router import IntentRouter, Intent
    
    router = IntentRouter()
    intent = router.parse("What regime is BTC in?")
    print(intent.skill)  # "hmm"
    print(intent.params)  # {"symbol": "BTC"}
"""

import re
import yaml
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from difflib import SequenceMatcher


class IntentType(Enum):
    """Types of intents the router can handle."""
    REGIME_DETECTION = "regime_detection"
    MONTE_CARLO = "monte_carlo"
    VOLATILITY = "volatility"
    PORTFOLIO = "portfolio"
    RISK_ANALYSIS = "risk_analysis"
    GENERAL = "general"
    UNKNOWN = "unknown"


@dataclass
class ExtractedEntity:
    """An entity extracted from user input."""
    type: str  # symbol, number, date, etc.
    value: Any
    confidence: float


@dataclass
class Intent:
    """
    Parsed intent from user input.
    
    Attributes:
        type: The type of intent
        skill: The skill to execute
        params: Extracted parameters
        entities: Extracted entities
        confidence: Confidence score (0-1)
        original_text: Original user message
        needs_clarification: Whether more info is needed
        clarification_questions: Questions to ask user
    """
    type: IntentType = IntentType.UNKNOWN
    skill: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    entities: List[ExtractedEntity] = field(default_factory=list)
    confidence: float = 0.0
    original_text: str = ""
    needs_clarification: bool = False
    clarification_questions: List[str] = field(default_factory=list)
    pipeline: Optional[str] = None  # For multi-skill queries


# =============================================================================
# Common patterns for entity extraction
# =============================================================================

SYMBOL_PATTERNS = [
    # Crypto pairs
    r'\b(BTC|ETH|SOL|ADA|XRP|DOGE|AVAX|MATIC|LINK|UNI|ATOM|LTC|BCH|XLM|ALGO|VET|FTM|ARB|OP|SHIB|PEPE)[/\s]?(USDT|USD|BTC|ETH)?\b',
    # Traditional assets
    r'\b(SPY|QQQ|IWM|GLD|SLV|TLT|AGG|VCIT|LQD|HYG|EMB|BND|VCLO|REIT)\b',
    # Stocks
    r'\b(AAPL|MSFT|GOOGL|AMZN|NVDA|META|TSLA|JPM|BAC|GS|MS|WFC|C|JNUG|DIRECT|ARKK|QQQM|VOO|IVV|SPY)\b',
]

NUMBER_PATTERNS = [
    # Percentages
    r'(\d+(?:\.\d+)?)\s*%',
    # Dollar amounts
    r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)',
    # Plain numbers
    r'\b(\d+(?:\.\d+)?)\b',
]

DATE_PATTERNS = [
    # Relative dates
    r'(today|tomorrow|yesterday|next week|next month|last week|last month)',
    # Days ahead
    r'(\d+)\s*(days?|weeks?|months?|years?)\s*(ahead|from now)?',
    # Specific dates
    r'(\d{4}-\d{2}-\d{2})',
]


# =============================================================================
# Intent Router Class
# =============================================================================

class IntentRouter:
    """
    Routes user messages to appropriate quant skills.
    
    Uses keyword matching and entity extraction to determine
    the user's intent and extract relevant parameters.
    """
    
    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize the router.
        
        Args:
            registry_path: Path to skill_registry.yaml (optional)
        """
        self.registry_path = registry_path
        self.intent_mappings: Dict[str, Dict[str, Any]] = {}
        self._load_intent_mappings()
    
    def _load_intent_mappings(self):
        """Load intent mappings from registry or use defaults."""
        if self.registry_path:
            try:
                with open(self.registry_path, 'r') as f:
                    registry = yaml.safe_load(f)
                    self.intent_mappings = registry.get('intents', {})
                return
            except (FileNotFoundError, yaml.YAMLError):
                pass
        
        # Default intent mappings
        self.intent_mappings = {
            'regime_detection': {
                'keywords': [
                    'regime', 'market regime', 'bull', 'bear', 'sideways',
                    'trend', 'market state', 'market condition', 'detect regime',
                    'what regime', 'is the market'
                ],
                'skill': 'hmm',
                'priority': 1
            },
            'monte_carlo': {
                'keywords': [
                    'monte carlo', 'simulate', 'simulation', 'price path',
                    'probability of profit', 'var', 'cvar', 'value at risk',
                    'worst case', 'best case', 'percentile'
                ],
                'skill': 'monte_carlo',
                'priority': 1
            },
            'volatility': {
                'keywords': [
                    'volatility', 'garch', 'forecast volatility',
                    'expected volatility', 'vol', 'implied volatility',
                    'historical volatility', 'realized volatility'
                ],
                'skill': 'garch',
                'priority': 1
            },
            'portfolio': {
                'keywords': [
                    'portfolio', 'optimize', 'optimal', 'allocation',
                    'weights', 'sharpe', 'efficient frontier',
                    'risk parity', 'mean variance', 'MPT'
                ],
                'skill': 'portfolio_optimizer',
                'priority': 1
            },
            'risk_analysis': {
                'keywords': [
                    'risk', 'risk analysis', 'risk assessment',
                    ' VaR', ' cvar', 'stress test', 'drawdown'
                ],
                'skill': 'risk_engine',
                'priority': 2
            }
        }
    
    def parse(self, message: str) -> Intent:
        """
        Parse a user message into an Intent.
        
        Args:
            message: User's natural language message
            
        Returns:
            Intent object with parsed information
        """
        message = message.strip()
        
        # Extract entities first
        entities = self._extract_entities(message)
        
        # Determine intent type
        intent_type, skill, confidence = self._classify_intent(message)
        
        # Extract parameters
        params = self._extract_params(message, skill, entities)
        
        # Check if we need clarification
        needs_clarification, questions = self._check_clarification_needed(
            skill, params, entities
        )
        
        # Check for multi-skill queries
        pipeline = self._detect_pipeline(message)
        
        return Intent(
            type=intent_type,
            skill=skill,
            params=params,
            entities=entities,
            confidence=confidence,
            original_text=message,
            needs_clarification=needs_clarification,
            clarification_questions=questions,
            pipeline=pipeline
        )
    
    def _extract_entities(self, message: str) -> List[ExtractedEntity]:
        """Extract entities from the message."""
        entities = []
        
        # Extract symbols
        for pattern in SYMBOL_PATTERNS:
            for match in re.finditer(pattern, message, re.IGNORECASE):
                symbol = match.group(1).upper()
                if len(symbol) >= 2:
                    entities.append(ExtractedEntity(
                        type='symbol',
                        value=symbol,
                        confidence=0.95
                    ))
        
        # Extract numbers
        for pattern in NUMBER_PATTERNS:
            for match in re.finditer(pattern, message):
                try:
                    value = float(match.group(1).replace(',', ''))
                    entities.append(ExtractedEntity(
                        type='number',
                        value=value,
                        confidence=0.9
                    ))
                except (ValueError, AttributeError):
                    pass
        
        # Extract time-related entities
        for pattern in DATE_PATTERNS:
            for match in re.finditer(pattern, message, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    type='date',
                    value=match.group(0),
                    confidence=0.8
                ))
        
        return entities
    
    def _classify_intent(self, message: str) -> Tuple[IntentType, str, float]:
        """
        Classify the intent based on keywords.
        
        Returns:
            Tuple of (IntentType, skill_name, confidence)
        """
        message_lower = message.lower()
        
        best_match = None
        best_score = 0.0
        
        for intent_name, mapping in self.intent_mappings.items():
            keywords = mapping.get('keywords', [])
            
            for keyword in keywords:
                # Check for exact match
                if keyword.lower() in message_lower:
                    score = len(keyword) / len(message) * 2
                    if score > best_score:
                        best_score = score
                        best_match = mapping.get('skill', '')
            
            # Fuzzy matching
            for keyword in keywords:
                ratio = SequenceMatcher(
                    None, keyword.lower(), message_lower
                ).ratio()
                if ratio > best_score:
                    best_score = ratio
                    best_match = mapping.get('skill', '')
        
        # Map skill to intent type
        if best_match == 'hmm':
            return IntentType.REGIME_DETECTION, 'hmm', min(best_score, 1.0)
        elif best_match == 'monte_carlo':
            return IntentType.MONTE_CARLO, 'monte_carlo', min(best_score, 1.0)
        elif best_match == 'garch':
            return IntentType.VOLATILITY, 'garch', min(best_score, 1.0)
        elif best_match == 'portfolio_optimizer':
            return IntentType.PORTFOLIO, 'portfolio_optimizer', min(best_score, 1.0)
        
        return IntentType.UNKNOWN, '', 0.0
    
    def _extract_params(
        self,
        message: str,
        skill: str,
        entities: List[ExtractedEntity]
    ) -> Dict[str, Any]:
        """Extract parameters based on skill and entities."""
        params = {}
        
        # Get symbols from entities
        symbols = [e.value for e in entities if e.type == 'symbol']
        if symbols:
            params['symbol'] = symbols[0]
            if len(symbols) > 1:
                params['assets'] = symbols
        
        # Get numbers from entities
        numbers = [e.value for e in entities if e.type == 'number']
        
        # Skill-specific parameter extraction
        if skill == 'monte_carlo':
            # Extract number of paths
            path_match = re.search(r'(\d+)\s*(paths?|simulations?)', message, re.IGNORECASE)
            if path_match:
                params['n_paths'] = int(path_match.group(1))
            elif numbers:
                # Use first number as n_paths if present
                if 'simulation' in message.lower() or 'path' in message.lower():
                    params['n_paths'] = int(numbers[0])
            
            # Extract days ahead
            days_match = re.search(r'(\d+)\s*days?', message, re.IGNORECASE)
            if days_match:
                params['days_ahead'] = int(days_match.group(1))
            
            # Extract volatility
            vol_match = re.search(r'volatility[:\s]+(\d+(?:\.\d+)?)', message, re.IGNORECASE)
            if vol_match:
                params['volatility'] = float(vol_match.group(1)) / 100
            
            # Extract initial price
            if numbers:
                price_match = re.search(r'\$?(\d+(?:,\d{3})*(?:\.\d+)?)', message)
                if price_match:
                    params['initial_price'] = float(price_match.group(1).replace(',', ''))
        
        elif skill == 'hmm':
            # Number of states
            state_match = re.search(r'(\d+)\s*states?', message, re.IGNORECASE)
            if state_match:
                params['n_states'] = int(state_match.group(1))
            
            # Lookback days
            lookback_match = re.search(r'(\d+)\s*days?', message, re.IGNORECASE)
            if lookback_match:
                params['lookback_days'] = int(lookback_match.group(1))
        
        elif skill == 'garch':
            # GARCH parameters
            p_match = re.search(r'GARCH\((\d+)[,\)]', message, re.IGNORECASE)
            if p_match:
                params['p'] = int(p_match.group(1))
            
            q_match = re.search(r'GARCH\(\d+,(\d+)\)', message, re.IGNORECASE)
            if q_match:
                params['q'] = int(q_match.group(1))
            
            # Forecast horizon
            horizon_match = re.search(r'forecast.*?(\d+)\s*days?', message, re.IGNORECASE)
            if horizon_match:
                params['forecast_horizon'] = int(horizon_match.group(1))
        
        elif skill == 'portfolio_optimizer':
            # Extract objective
            if 'max sharpe' in message.lower():
                params['objective'] = 'max_sharpe'
            elif 'min vol' in message.lower() or 'minimum volatility' in message.lower():
                params['objective'] = 'min_volatility'
            elif 'max return' in message.lower():
                params['objective'] = 'max_return'
            elif 'risk parity' in message.lower():
                params['objective'] = 'risk_parity'
            else:
                params['objective'] = 'max_sharpe'  # Default
            
            # Extract risk-free rate
            rf_match = re.search(r'risk[- ]?free.*?(\d+(?:\.\d+)?)\s*%?', message, re.IGNORECASE)
            if rf_match:
                params['risk_free_rate'] = float(rf_match.group(1)) / 100
        
        return params
    
    def _check_clarification_needed(
        self,
        skill: str,
        params: Dict[str, Any],
        entities: List[ExtractedEntity]
    ) -> Tuple[bool, List[str]]:
        """Check if we need more information from the user."""
        questions = []
        
        # Check required parameters for each skill
        if skill == 'hmm':
            if 'symbol' not in params:
                questions.append("Which symbol or market would you like to analyze?")
        
        elif skill == 'monte_carlo':
            missing = []
            if 'initial_price' not in params:
                missing.append("initial price")
            if 'volatility' not in params:
                missing.append("volatility")
            
            if missing:
                questions.append(f"Please provide: {', '.join(missing)}")
        
        elif skill == 'garch':
            if 'symbol' not in params:
                questions.append("Which symbol's volatility would you like to forecast?")
        
        elif skill == 'portfolio_optimizer':
            if 'assets' not in params or len(params.get('assets', [])) < 2:
                questions.append("Which assets would you like to include in the portfolio?")
        
        return len(questions) > 0, questions
    
    def _detect_pipeline(self, message: str) -> Optional[str]:
        """Detect if this is a multi-skill query."""
        message_lower = message.lower()
        
        # Check for full risk analysis pattern
        if all(kw in message_lower for kw in ['risk', 'regime', 'volatility', 'monte carlo']):
            return 'full_risk_analysis'
        
        # Check for smart portfolio pattern
        if all(kw in message_lower for kw in ['portfolio', 'regime']):
            return 'smart_portfolio'
        
        return None
    
    def route(self, message: str) -> Intent:
        """
        Route a message to the appropriate skill.
        
        This is the main entry point for the router.
        
        Args:
            message: User's message
            
        Returns:
            Intent with skill and parameters
        """
        return self.parse(message)


# =============================================================================
# Skill Executor
# =============================================================================

class SkillExecutor:
    """
    Executes skills based on parsed intents.
    
    This class bridges the intent router with the actual skill implementations.
    """
    
    def __init__(self):
        """Initialize the executor."""
        self.skill_modules: Dict[str, Any] = {}
    
    def execute(self, intent: Intent) -> Any:
        """
        Execute the skill for the given intent.
        
        Args:
            intent: Parsed intent
            
        Returns:
            Skill execution result
        """
        if intent.pipeline:
            return self._execute_pipeline(intent)
        
        if not intent.skill:
            return {"error": "No skill identified", "intent": intent}
        
        # Lazy import of skill modules
        return self._execute_skill(intent.skill, intent.params)
    
    def _execute_skill(self, skill: str, params: Dict[str, Any]) -> Any:
        """Execute a single skill."""
        try:
            if skill == 'hmm':
                from hmm_regime_detect import detect_regimes, format_output
                result = detect_regimes(**params)
                return format_output(result)
            
            elif skill == 'monte_carlo':
                from monte_carlo_paths import generate_price_paths, format_output
                result = generate_price_paths(**params)
                return format_output(result)
            
            elif skill == 'garch':
                from garch_volatility import fit_garch, format_output
                result = fit_garch(**params)
                return format_output(result)
            
            elif skill == 'portfolio_optimizer':
                from portfolio_optimizer import optimize_portfolio, format_output
                result = optimize_portfolio(**params)
                return format_output(result)
            
            else:
                return {"error": f"Unknown skill: {skill}"}
        
        except ImportError as e:
            return {"error": f"Missing dependency: {e}"}
        except Exception as e:
            return {"error": str(e)}
    
    def _execute_pipeline(self, intent: Intent) -> List[Any]:
        """Execute a pipeline of skills."""
        # This would be implemented in skill_orchestrator.py
        return []


# =============================================================================
# Intent to Skill Mapping (for route_intent function)
# =============================================================================

_INTENT_MAP = {
    "regime_analysis": "hmm_regime_detect",
    "hmm_regime": "hmm_regime_detect",
    "detect_regime": "hmm_regime_detect",
    "volatility_analysis": "garch_volatility",
    "garch_vol": "garch_volatility",
    "volatility_forecast": "garch_volatility",
    "simulate_paths": "monte_carlo_paths",
    "monte_carlo": "monte_carlo_paths",
    "price_simulation": "monte_carlo_paths",
    "portfolio_optimization": "portfolio_optimizer",
    "optimize_portfolio": "portfolio_optimizer",
    "portfolio_allocation": "portfolio_optimizer",
}


def _load_skill_callable(skill_name: str):
    """
    Load a skill's callable function based on registry configuration.
    
    Args:
        skill_name: Name of the skill to load
        
    Returns:
        Tuple of (function, format_function)
    """
    # Import the registry
    try:
        from skill_registry import get_registry
        registry = get_registry()
        cfg = registry.get(skill_name)
    except (ImportError, FileNotFoundError):
        # Fallback to direct imports if registry not available
        cfg = {}
    
    # Map skill names to module functions
    skill_modules = {
        "hmm_regime_detect": ("hmm_regime_detect", "detect_regimes"),
        "garch_volatility": ("garch_volatility", "fit_garch"),
        "monte_carlo_paths": ("monte_carlo_paths", "generate_price_paths"),
        "portfolio_optimizer": ("portfolio_optimizer", "optimize_portfolio"),
    }
    
    if skill_name not in skill_modules:
        raise ValueError(f"Unknown skill: {skill_name}")
    
    module_name, func_name = skill_modules[skill_name]
    
    # Import the module dynamically
    import importlib
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    
    # Try to get format_output function if it exists
    format_func = getattr(module, "format_output", None)
    
    return func, format_func


def route_intent(intent: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Route an intent to the appropriate skill and execute it.
    
    This is the main entry point for executing skills based on intent.
    Used by composed_strategies.py for multi-skill orchestration.
    
    Args:
        intent: The intent identifier (e.g., "regime_analysis", "simulate_paths")
        params: Dictionary of parameters to pass to the skill
        
    Returns:
        Dictionary containing the skill's results
        
    Raises:
        ValueError: If the intent is not recognized
        RuntimeError: If the skill is disabled
    """
    # Map intent to skill name
    if intent not in _INTENT_MAP:
        raise ValueError(f"Unknown intent: {intent}")
    
    skill_name = _INTENT_MAP[intent]
    
    # Check if skill is enabled (if registry available)
    try:
        from skill_registry import get_registry
        registry = get_registry()
        if not registry.is_enabled(skill_name):
            raise RuntimeError(f"Skill disabled: {skill_name}")
    except (ImportError, FileNotFoundError):
        pass  # Continue if registry not available
    
    # Load and execute the skill
    func, format_func = _load_skill_callable(skill_name)
    
    try:
        result = func(**params)
        
        # Apply formatting if available
        if format_func and hasattr(result, '__dict__'):
            return format_func(result)
        
        return result
    
    except TypeError as e:
        # Handle missing or invalid parameters
        return {
            "error": f"Invalid parameters for {skill_name}: {str(e)}",
            "skill": skill_name,
            "intent": intent,
        }
    except Exception as e:
        return {
            "error": f"Skill execution failed: {str(e)}",
            "skill": skill_name,
            "intent": intent,
        }


# =============================================================================
# Main entry point
# =============================================================================

def route_message(message: str) -> Intent:
    """
    Convenience function to route a message.
    
    Args:
        message: User's message
        
    Returns:
        Parsed intent
    """
    router = IntentRouter()
    return router.route(message)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    router = IntentRouter()
    
    test_messages = [
        "What regime is BTC in?",
        "Simulate 5000 paths for BTC with 3% volatility",
        "Optimize my portfolio: BTC, ETH, SOL, SPY",
        "What's the GARCH volatility forecast for ETH?",
        "Calculate VaR for my position",
        "Simulate price paths for the next 30 days"
    ]
    
    print("=== Intent Routing Tests ===\n")
    
    for msg in test_messages:
        intent = router.parse(msg)
        print(f"Message: {msg}")
        print(f"  Skill: {intent.skill}")
        print(f"  Confidence: {intent.confidence:.2f}")
        print(f"  Params: {intent.params}")
        if intent.needs_clarification:
            print(f"  Questions: {intent.clarification_questions}")
        if intent.pipeline:
            print(f"  Pipeline: {intent.pipeline}")
        print()
