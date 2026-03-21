#!/usr/bin/env python3
"""
MiniMax API Connection Test
===========================
This script tests the connection to MiniMax API and verifies
that the integration with OpenClaw is working correctly.

Usage:
    # Set environment variables first:
    export MINIMAX_API_KEY="your_api_key"
    export MINIMAX_GROUP_ID="your_group_id"
    
    # Run the test
    python openclaw_skills/test_minimax_connection.py

Or with inline credentials:
    python openclaw_skills/test_minimax_connection.py --api-key "your_key" --group-id "your_group"
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openclaw_skills.minimax_connector import (
    MiniMaxClient,
    MiniMaxConfig,
    MiniMaxModel,  # Latest models: M2.7, M2.5, M2.1, M2
    OpenClawMiniMaxBridge,
    create_trading_system_prompt
)


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = ''
    RED = ''
    YELLOW = ''
    BLUE = ''
    BOLD = ''
    END = ''


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_success(text: str):
    """Print success message."""
    print(f"[OK] {text}")


def print_error(text: str):
    """Print error message."""
    print(f"[ERROR] {text}")


def print_warning(text: str):
    """Print warning message."""
    print(f"[WARNING] {text}")


def test_config_loading(config: MiniMaxConfig) -> bool:
    """Test that configuration is loaded correctly."""
    print_header("Testing Configuration")
    
    tests_passed = 0
    tests_total = 3
    
    # Test API key
    if config.api_key:
        print_success(f"API Key loaded: {config.api_key[:10]}...")
        tests_passed += 1
    else:
        print_error("API Key not loaded")
    
    # Test Group ID
    if config.group_id:
        print_success(f"Group ID loaded: {config.group_id}")
        tests_passed += 1
    else:
        print_error("Group ID not loaded")
    
    # Test API base URL
    print_success(f"API Base URL: {config.api_base}")
    tests_passed += 1
    
    print(f"\nConfiguration tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def test_api_connection(client: MiniMaxClient) -> bool:
    """Test basic API connection."""
    print_header("Testing API Connection")
    
    try:
        # Simple test message
        response = client.chat_completion(
            messages=[
                {"role": "user", "content": "Say 'API connection successful' in exactly those words if you can hear me."}
            ],
            model=MiniMaxModel.MINIMAX_M2,
            max_tokens=100
        )
        
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            print_success("API Connection successful!")
            print(f"\nResponse: {content}")
            return True
        else:
            print_error("Unexpected response format")
            print(f"Response: {json.dumps(response, indent=2)}")
            return False
            
    except Exception as e:
        print_error(f"API Connection failed: {str(e)}")
        return False


def test_trading_analysis(client: MiniMaxClient) -> bool:
    """Test trading-specific analysis."""
    print_header("Testing Trading Analysis")
    
    market_data = {
        "symbol": "BTCUSDT",
        "price": 67500.00,
        "volume_24h": 28500000000,
        "price_change_24h": 2.5,
        "market_cap": 1320000000000,
        "rsi": 58.5,
        "macd": {
            "histogram": 125.5,
            "signal": "bullish"
        },
        "regime": "bull_market"
    }
    
    try:
        response = client.analyze_market(
            symbol="BTCUSDT",
            market_data=market_data,
            analysis_type="regime"
        )
        
        print_success("Trading analysis completed!")
        print(f"\nAnalysis:\n{response[:500]}...")
        return True
        
    except Exception as e:
        print_error(f"Trading analysis failed: {str(e)}")
        return False


def test_streaming(client: MiniMaxClient) -> bool:
    """Test streaming responses."""
    print_header("Testing Streaming Response")
    
    try:
        print("Streaming response (first 100 chars): ")
        chunk_count = 0
        full_response = ""
        
        for chunk in client.stream_chat(
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            max_tokens=50
        ):
            print(chunk, end="", flush=True)
            full_response += chunk
            chunk_count += 1
        
        print(f"\n\nSuccess! Received {chunk_count} chunks")
        return True
        
    except Exception as e:
        print_error(f"Streaming failed: {str(e)}")
        return False


def test_openclaw_bridge(bridge: OpenClawMiniMaxBridge) -> bool:
    """Test OpenClaw bridge integration."""
    print_header("Testing OpenClaw Bridge")
    
    # Test research agent
    try:
        response = bridge.process_agent_message(
            agent_name="research_agent",
            message="What is the current market sentiment for Bitcoin?",
            context={
                "symbol": "BTCUSDT",
                "price": 67500,
                "sentiment_score": 0.65
            }
        )
        
        print_success("Research agent response:")
        print(f"{response[:300]}...")
        return True
        
    except Exception as e:
        print_error(f"OpenClaw bridge failed: {str(e)}")
        return False


def test_system_prompt() -> bool:
    """Test trading system prompt generation."""
    print_header("Testing System Prompt Generation")
    
    try:
        prompt = create_trading_system_prompt(
            portfolio_value=50000,
            positions=[
                {"symbol": "BTCUSDT", "quantity": 0.5, "entry_price": 65000},
                {"symbol": "ETHUSDT", "quantity": 5, "entry_price": 3200}
            ],
            risk_tolerance="moderate"
        )
        
        print_success("System prompt generated!")
        print(f"\nPrompt preview:\n{prompt[:300]}...")
        return True
        
    except Exception as e:
        print_error(f"System prompt generation failed: {str(e)}")
        return False


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test MiniMax API connection")
    parser.add_argument("--api-key", help="MiniMax API key")
    parser.add_argument("--group-id", help="MiniMax Group ID")
    parser.add_argument("--skip-streaming", action="store_true", help="Skip streaming test")
    parser.add_argument("--test-only-config", action="store_true", help="Only test config loading")
    args = parser.parse_args()
    
    print_header("MiniMax API Integration Test")
    print("This test verifies the connection between OpenClaw and MiniMax API")
    
    # Get credentials from args or environment
    api_key = args.api_key or os.getenv("MINIMAX_API_KEY")
    group_id = args.group_id or os.getenv("MINIMAX_GROUP_ID")
    
    if not api_key or not group_id:
        print_error("Missing credentials!")
        print("\nPlease provide credentials via:")
        print("  1. Environment variables:")
        print("     export MINIMAX_API_KEY='your_api_key'")
        print("     export MINIMAX_GROUP_ID='your_group_id'")
        print("\n  2. Command line arguments:")
        print("     python test_minimax_connection.py --api-key 'your_key' --group-id 'your_group'")
        print("\nTo get your free API key, visit: https://platform.minimax.io/")
        sys.exit(1)
    
    # Create configuration
    config = MiniMaxConfig(api_key=api_key, group_id=group_id)
    
    # Run tests
    results = {}
    
    # Test 1: Configuration
    results["config"] = test_config_loading(config)
    if args.test_only_config:
        sys.exit(0 if results["config"] else 1)
    
    # Create client
    client = MiniMaxClient(config)
    bridge = OpenClawMiniMaxBridge(config)
    
    # Test 2: API Connection
    results["connection"] = test_api_connection(client)
    
    if not results["connection"]:
        print_error("\nAPI connection failed. Please check your credentials.")
        sys.exit(1)
    
    # Test 3: Trading Analysis
    results["trading_analysis"] = test_trading_analysis(client)
    
    # Test 4: System Prompt
    results["system_prompt"] = test_system_prompt()
    
    # Test 5: OpenClaw Bridge
    results["openclaw_bridge"] = test_openclaw_bridge(bridge)
    
    # Test 6: Streaming (optional)
    if not args.skip_streaming:
        results["streaming"] = test_streaming(client)
    
    # Summary
    print_header("Test Summary")
    total = sum(1 for v in results.values() if v)
    passed = sum(1 for v in results.values() if v)
    
    for test_name, passed_test in results.items():
        if passed_test:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")
    
    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print_success("All tests passed! MiniMax API is ready to use with OpenClaw.")
        sys.exit(0)
    else:
        print_error("Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
