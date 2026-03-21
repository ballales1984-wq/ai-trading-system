"""
MiniMax API Connector for OpenClaw
==================================
This module provides integration with MiniMax-M2 API for AI-powered
quantitative analysis and natural language understanding.

MiniMax API Documentation: https://platform.minimax.io/docs

Usage:
    from minimax_connector import MiniMaxClient, MiniMaxConfig
    
    config = MiniMaxConfig(
        api_key="your_api_key",
        group_id="your_group_id"
    )
    client = MiniMaxClient(config)
    
    # Chat completion
    response = client.chat_completion(
        messages=[{"role": "user", "content": "Analyze BTC market regime"}]
    )
    
    # Streaming chat
    for chunk in client.stream_chat(messages=[...]):
        print(chunk)

Environment Variables:
    MINIMAX_API_KEY: Your MiniMax API key (starts with 'ey')
    MINIMAX_GROUP_ID: Your MiniMax Group ID
    MINIMAX_API_BASE: API base URL (default: https://api.minimax.chat/v1)

Supported Models:
    - MiniMax-M2: General purpose chat model (recommended)
    - MiniMax-Text-01: Text-specific model
    - abab6.5s-chat: Fast chat model
    - abab6.5g-chat: General chat model
"""

import os
import json
import time
from typing import Any, Dict, Generator, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import requests


class MiniMaxModel(str, Enum):
    """Available MiniMax models.
    
    Based on official MiniMax API documentation:
    https://platform.minimax.io/docs/llms.txt
    """
    # Latest models (recommended)
    MINIMAX_M2_7 = "MiniMax-M2.7"
    MINIMAX_M2_7_HIGH_SPEED = "MiniMax-M2.7-highspeed"
    MINIMAX_M2_5 = "MiniMax-M2.5"
    MINIMAX_M2_5_HIGH_SPEED = "MiniMax-M2.5-highspeed"
    MINIMAX_M2_1 = "MiniMax-M2.1"
    MINIMAX_M2_1_HIGH_SPEED = "MiniMax-M2.1-highspeed"
    
    # Original M2 model
    MINIMAX_M2 = "MiniMax-M2"
    
    # Legacy models
    MINIMAX_TEXT_01 = "MiniMax-Text-01"
    ABAB65S_CHAT = "abab6.5s-chat"
    ABAB65G_CHAT = "abab6.5g-chat"


@dataclass
class MiniMaxConfig:
    """
    Configuration for MiniMax API connection.
    
    Attributes:
        api_key: MiniMax API key (starts with 'ey')
        group_id: MiniMax Group ID from dashboard
        api_base: Base URL for API (default: https://api.minimax.chat/v1)
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
    """
    api_key: str = ""
    group_id: str = ""
    api_base: str = "https://api.minimax.chat/v1"
    timeout: int = 60
    max_retries: int = 3
    
    def __post_init__(self):
        """Load config from environment variables if not provided."""
        self.api_key = self.api_key or os.getenv("MINIMAX_API_KEY", "")
        self.group_id = self.group_id or os.getenv("MINIMAX_GROUP_ID", "")
        self.api_base = self.api_base or os.getenv("MINIMAX_API_BASE", "https://api.minimax.chat/v1")
    
    def is_configured(self) -> bool:
        """Check if the configuration is complete."""
        return bool(self.api_key and self.group_id)
    
    def get_headers(self) -> Dict[str, str]:
        """Get common headers for API requests."""
        # MiniMax API uses Bearer token with the API key
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }


@dataclass
class ChatMessage:
    """A chat message for the API."""
    role: str = "user"  # system, user, assistant
    content: str = ""
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        return result


@dataclass
class ChatCompletionRequest:
    """Request payload for chat completion."""
    # Default to latest M2.7 model for best performance
    model: str = MiniMaxModel.MINIMAX_M2_7
    messages: List[Dict[str, str]] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 8192  # Increased for longer context
    top_p: float = 0.95
    stream: bool = False
    stop: Optional[List[str]] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    # Additional parameters from official API
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[str] = None
    reasoning_effort: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        result = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": self.stream,
        }
        if self.stop:
            result["stop"] = self.stop
        if self.frequency_penalty != 0.0:
            result["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty != 0.0:
            result["presence_penalty"] = self.presence_penalty
        if self.tools:
            result["tools"] = self.tools
        if self.tool_choice:
            result["tool_choice"] = self.tool_choice
        if self.reasoning_effort:
            result["reasoning_effort"] = self.reasoning_effort
        return result


class MiniMaxAPIError(Exception):
    """Exception raised for MiniMax API errors."""
    def __init__(self, message: str, code: Optional[int] = None, response: Optional[Dict] = None):
        self.message = message
        self.code = code
        self.response = response
        super().__init__(self.message)


class MiniMaxClient:
    """
    Client for MiniMax API.
    
    Provides methods for chat completions, embeddings, and other AI tasks.
    """
    
    # Endpoint paths
    CHAT_COMPLETION_PATH = "/text/chatcompletion_v2"
    EMBEDDINGS_PATH = "/text/embeddings"
    
    def __init__(self, config: Optional[MiniMaxConfig] = None):
        """
        Initialize the MiniMax client.
        
        Args:
            config: MiniMax configuration. If None, loads from environment.
        """
        self.config = config or MiniMaxConfig()
        self._session = requests.Session()
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """
        Make an API request with retry logic.
        
        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint path
            payload: Request payload
            stream: Whether to stream the response
            
        Returns:
            Response data or stream generator
            
        Raises:
            MiniMaxAPIError: If the API request fails
        """
        url = f"{self.config.api_base}{endpoint}"
        
        if not self.config.is_configured():
            raise MiniMaxAPIError(
                "MiniMax API is not configured. "
                "Please provide api_key and group_id."
            )
        
        # Add group_id to payload for chat completions
        if endpoint == self.CHAT_COMPLETION_PATH and payload:
            payload["group_id"] = self.config.group_id
        
        headers = self.config.get_headers()
        
        for attempt in range(self.config.max_retries):
            try:
                if method == "POST":
                    if stream:
                        response = self._session.post(
                            url,
                            headers=headers,
                            json=payload,
                            stream=True,
                            timeout=self.config.timeout
                        )
                        return self._handle_stream_response(response)
                    else:
                        response = self._session.post(
                            url,
                            headers=headers,
                            json=payload,
                            timeout=self.config.timeout
                        )
                else:
                    response = self._session.get(
                        url,
                        headers=headers,
                        params=payload,
                        timeout=self.config.timeout
                    )
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Parse response
                if stream:
                    return self._handle_stream_response(response)
                return response.json()
                
            except requests.exceptions.Timeout:
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise MiniMaxAPIError("Request timed out")
                
            except requests.exceptions.RequestException as e:
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise MiniMaxAPIError(f"Request failed: {str(e)}")
        
        raise MiniMaxAPIError("Max retries exceeded")
    
    def _handle_stream_response(self, response: requests.Response) -> Generator[str, None, None]:
        """Handle streaming response."""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        return
                    try:
                        json_data = json.loads(data)
                        if 'choices' in json_data and len(json_data['choices']) > 0:
                            delta = json_data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                yield delta['content']
                    except json.JSONDecodeError:
                        continue
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = MiniMaxModel.MINIMAX_M2,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model name (default: MiniMax-M2)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters (top_p, stop, etc.)
            
        Returns:
            API response with generated text
            
        Example:
            >>> client = MiniMaxClient(config)
            >>> response = client.chat_completion([
            ...     {"role": "system", "content": "You are a trading assistant"},
            ...     {"role": "user", "content": "Analyze BTC trend"}
            ... ])
            >>> print(response['choices'][0]['message']['content'])
        """
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return self._make_request("POST", self.CHAT_COMPLETION_PATH, request.to_dict())
    
    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = MiniMaxModel.MINIMAX_M2,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Send a streaming chat completion request.
        
        Args:
            messages: List of message dictionaries
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Yields:
            Content chunks as they arrive
            
        Example:
            >>> for chunk in client.stream_chat(messages):
            ...     print(chunk, end='')
        """
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )
        
        yield from self._make_request(
            "POST",
            self.CHAT_COMPLETION_PATH,
            request.to_dict(),
            stream=True
        )
    
    def analyze_market(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        analysis_type: str = "regime"
    ) -> str:
        """
        Analyze market data using MiniMax-M2.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            market_data: Market data dictionary
            analysis_type: Type of analysis ("regime", "sentiment", "prediction")
            
        Returns:
            Analysis text from the model
            
        Example:
            >>> analysis = client.analyze_market(
            ...     "BTCUSDT",
            ...     {"price": 50000, "volume": 1000000},
            ...     "regime"
            ... )
        """
        system_prompt = self._get_analysis_prompt(analysis_type)
        
        user_message = f"""
Analyze {symbol} market:

Current Data:
{json.dumps(market_data, indent=2)}

Provide a detailed analysis focusing on:
1. Current market regime (bull/bear/sideways)
2. Key support and resistance levels
3. Risk assessment
4. Trading recommendations
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        response = self.chat_completion(messages)
        return response['choices'][0]['message']['content']
    
    def _get_analysis_prompt(self, analysis_type: str) -> str:
        """Get the system prompt for different analysis types."""
        prompts = {
            "regime": """You are an expert quantitative analyst specializing in market regime detection.
You analyze market data to identify current market conditions (bull, bear, sideways).
Provide clear, actionable insights based on technical indicators and price action.""",
            
            "sentiment": """You are a market sentiment analyst.
Analyze news, social media, and on-chain data to determine overall market sentiment.
Provide sentiment scores and key factors influencing market mood.""",
            
            "prediction": """You are a quantitative trading specialist.
Analyze historical data and patterns to provide price predictions.
Always include risk assessment and probability estimates.""",
        }
        return prompts.get(analysis_type, prompts["regime"])
    
    def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate a text response to a prompt.
        
        Args:
            prompt: User prompt
            context: Optional context/information to reference
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        if context:
            full_prompt = f"""Context:
{context}

Question: {prompt}

Please provide a detailed answer based on the context above."""
        else:
            full_prompt = prompt
        
        messages = [{"role": "user", "content": full_prompt}]
        
        response = self.chat_completion(messages, max_tokens=max_tokens)
        return response['choices'][0]['message']['content']


# =============================================================================
# Trading-Specific Functions
# =============================================================================

def create_trading_system_prompt(
    portfolio_value: float = 0,
    positions: List[Dict] = None,
    risk_tolerance: str = "moderate"
) -> str:
    """
    Create a system prompt for trading analysis.
    
    Args:
        portfolio_value: Total portfolio value in USD
        positions: List of current positions
        risk_tolerance: Risk tolerance level (conservative, moderate, aggressive)
        
    Returns:
        System prompt for the AI assistant
    """
    positions_text = ""
    if positions:
        positions_text = "\n".join([
            f"- {p.get('symbol', 'N/A')}: {p.get('quantity', 0)} @ ${p.get('entry_price', 0):.2f}"
            for p in positions
        ])
    
    risk_guidelines = {
        "conservative": "Prioritize capital preservation. Use tight stop losses (2-3%). Only take trades with >2:1 risk-reward.",
        "moderate": "Balance between growth and risk. Use standard stop losses (5-7%). Accept some drawdown.",
        "aggressive": "Focus on maximum returns. Allow larger drawdowns (10-15%). Take higher-risk setups."
    }
    
    return f"""You are an expert quantitative trading assistant for a crypto trading system.

Current Portfolio:
- Total Value: ${portfolio_value:,.2f}
- Positions:
{positions_text or "  (no open positions)"}

Risk Tolerance: {risk_tolerance}
{risk_guidelines.get(risk_tolerance, risk_guidelines['moderate'])}

Guidelines:
1. Always consider risk/reward ratio before suggesting trades
2. Provide probabilistic assessments, not certain predictions
3. Consider market regime and correlation with existing positions
4. Suggest position sizing based on current portfolio risk
5. Include stop-loss and take-profit levels in recommendations

When analyzing trades, always provide:
- Entry price range
- Stop loss level
- Take profit target
- Risk/reward ratio
- Probability estimate
- Position size recommendation (% of portfolio)
"""


# =============================================================================
# Integration with OpenClaw
# =============================================================================

class OpenClawMiniMaxBridge:
    """
    Bridge between OpenClaw agents and MiniMax API.
    
    This class integrates MiniMax AI capabilities with the OpenClaw
    multi-agent system for enhanced natural language understanding.
    """
    
    def __init__(self, config: Optional[MiniMaxConfig] = None):
        """Initialize the bridge."""
        self.client = MiniMaxClient(config)
        self.config = config or MiniMaxConfig()
    
    def process_agent_message(
        self,
        agent_name: str,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process a message from an OpenClaw agent using MiniMax.
        
        Args:
            agent_name: Name of the agent (e.g., "research_agent")
            message: User message
            context: Additional context (positions, market data, etc.)
            
        Returns:
            AI-generated response
        """
        system_prompt = self._get_agent_system_prompt(agent_name)
        
        # Add context to message if available
        full_message = message
        if context:
            context_str = f"\n\nContext:\n{json.dumps(context, indent=2)}"
            full_message += context_str
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_message}
        ]
        
        response = self.client.chat_completion(messages)
        return response['choices'][0]['message']['content']
    
    def _get_agent_system_prompt(self, agent_name: str) -> str:
        """Get the system prompt for a specific OpenClaw agent."""
        prompts = {
            "research_agent": """You are a Market Research Specialist for a quantitative trading system.
Your role is to gather and analyze market intelligence, including:
- Current market prices and trends
- News sentiment analysis
- On-chain metrics
- Social media trends

Provide concise, data-driven insights.""",
            
            "quant_analyst_agent": """You are a Quantitative Analyst specializing in:
- Hidden Markov Models for regime detection
- Monte Carlo simulations
- GARCH volatility modeling
- Portfolio optimization

Use only quantitative metrics and provide probabilistic forecasts.""",
            
            "risk_gate_agent": """You are a Risk Gatekeeper protecting the portfolio.
Your responsibilities:
- Validate trades against risk limits
- Calculate position-level risk contributions
- Block trades that exceed VaR/CVaR limits
- Provide actionable risk reduction suggestions

Always cite specific metrics when making decisions.""",
            
            "executor_agent": """You are an Order Execution Specialist.
Your role is to:
- Execute approved trades efficiently
- Use optimal execution strategies (TWAP, limit orders)
- Confirm order fills and log executions
- Never execute without proper risk approval
"""
        }
        return prompts.get(agent_name, "You are a helpful trading assistant.")


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI for testing MiniMax API connection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MiniMax API CLI")
    parser.add_argument("--api-key", help="MiniMax API key")
    parser.add_argument("--group-id", help="MiniMax Group ID")
    parser.add_argument("--model", default="MiniMax-M2", help="Model to use")
    parser.add_argument("prompt", nargs="*", help="Prompt to send")
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.getenv("MINIMAX_API_KEY")
    group_id = args.group_id or os.getenv("MINIMAX_GROUP_ID")
    
    if not api_key or not group_id:
        print("Error: API key and Group ID required.")
        print("Set them via --api-key/--group-id or MINIMAX_API_KEY/MINIMAX_GROUP_ID env vars")
        return
    
    config = MiniMaxConfig(api_key=api_key, group_id=group_id)
    client = MiniMaxClient(config)
    
    prompt = " ".join(args.prompt) if args.prompt else "Hello, how are you?"
    
    print(f"\nSending prompt: {prompt}\n")
    print("Response:")
    print("-" * 40)
    
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=args.model
    )
    
    print(response['choices'][0]['message']['content'])


if __name__ == "__main__":
    main()
