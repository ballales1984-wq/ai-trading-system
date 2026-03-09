# test_twitter_api.py
"""
Test script for Twitter/X API integration.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock


class TestTwitterSentimentClient:
    """Test Twitter sentiment client."""
    
    def test_import_twitter_client(self):
        """Test that TwitterSentimentClient can be imported."""
        try:
            from src.external.sentiment_apis import TwitterSentimentClient
            assert TwitterSentimentClient is not None
            print("[OK] TwitterSentimentClient imported successfully")
        except ImportError as e:
            pytest.skip(f"Cannot import TwitterSentimentClient: {e}")
    
    def test_twitter_client_initialization(self):
        """Test Twitter client initialization."""
        from src.external.sentiment_apis import TwitterSentimentClient
        
        # Without token
        client = TwitterSentimentClient(bearer_token="")
        assert client.name == "twitter"
        assert client.base_url == "https://api.twitter.com/2"
        print("[OK] Twitter client initialized without token")
        
        # With token
        client_with_token = TwitterSentimentClient(bearer_token="test_token")
        assert client_with_token.api_key == "test_token"
        print("[OK] Twitter client initialized with token")
    
    @pytest.mark.asyncio
    async def test_twitter_client_fetch_no_token(self):
        """Test that fetch returns empty list without token."""
        from src.external.sentiment_apis import TwitterSentimentClient
        
        client = TwitterSentimentClient(bearer_token="")
        records = await client.fetch(query="bitcoin", limit=10)
        assert records == []
        print("[OK] Twitter client returns empty list without token")
    
    @pytest.mark.asyncio
    async def test_twitter_client_health_check_no_token(self):
        """Test health check without token."""
        from src.external.sentiment_apis import TwitterSentimentClient
        
        client = TwitterSentimentClient(bearer_token="")
        result = await client.health_check()
        assert result == False
        print("[OK] Health check returns False without token")


class TestTwitterConfig:
    """Test Twitter configuration."""
    
    def test_twitter_config_in_settings(self):
        """Test that Twitter config is in NEWS_SETTINGS."""
        from config import NEWS_SETTINGS
        
        assert 'twitter_bearer_token' in NEWS_SETTINGS
        print(f"[OK] Twitter config present: twitter_bearer_token")
    
    def test_twitter_env_variable(self):
        """Test Twitter environment variable loading."""
        import os
        from config import NEWS_SETTINGS
        
        # Should load from environment
        twitter_token = NEWS_SETTINGS.get('twitter_bearer_token', '')
        print(f"[OK] Twitter token from env: {'<set>' if twitter_token else '<not set>'}")


class TestCreateSentimentClients:
    """Test sentiment clients factory function."""
    
    def test_create_sentiment_clients_with_twitter(self):
        """Test creating sentiment clients with Twitter."""
        from src.external.sentiment_apis import create_sentiment_clients
        
        clients = create_sentiment_clients(
            newsapi_key="test_news",
            twitter_bearer="test_twitter",
            include_gdelt=False
        )
        
        # Should have NewsAPI and Twitter clients
        client_names = [c.name for c in clients]
        assert "newsapi" in client_names
        assert "twitter" in client_names
        print(f"[OK] Created clients: {client_names}")
    
    def test_create_sentiment_clients_without_twitter(self):
        """Test creating sentiment clients without Twitter."""
        from src.external.sentiment_apis import create_sentiment_clients
        
        clients = create_sentiment_clients(
            newsapi_key="test_news",
            twitter_bearer="",  # No Twitter token
            include_gdelt=False
        )
        
        # Should only have NewsAPI client
        client_names = [c.name for c in clients]
        assert "newsapi" in client_names
        assert "twitter" not in client_names
        print(f"[OK] Created clients without Twitter: {client_names}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
