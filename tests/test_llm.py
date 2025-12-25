"""
Tests for LLM client (GitHub Models + Ollama fallback).

These tests verify:
- LLMClient and SyncLLMClient initialization
- GitHub Models integration
- Ollama fallback mechanism
- Automatic switching on rate limits
- Provider status reporting
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os
import httpx

from llm.client import LLMClient, SyncLLMClient


# =============================================================================
# LLMClient Initialization Tests
# =============================================================================

class TestLLMClientInit:
    """Tests for LLMClient initialization."""
    
    def test_init_with_github_token(self):
        """Should initialize with GitHub token."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "test_token_123"}):
            client = LLMClient()
            
            assert client.github_api_key == "test_token_123"
            assert client.github_client is not None
    
    def test_init_without_token(self):
        """Should handle missing GitHub token."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GITHUB_TOKEN", None)
            client = LLMClient(api_key=None)
            
            assert client.github_api_key is None
            assert client.github_client is None
    
    def test_init_custom_model(self):
        """Should accept custom model name."""
        client = LLMClient(api_key="token", model="gpt-4")
        
        assert client.github_model == "gpt-4"
    
    def test_init_custom_ollama_model(self):
        """Should accept custom Ollama model."""
        client = LLMClient(api_key="token", ollama_model="llama2")
        
        assert client.ollama_model == "llama2"
    
    def test_init_prefer_ollama(self):
        """Should set Ollama as preferred provider."""
        client = LLMClient(api_key="token", prefer_ollama=True)
        
        assert client.prefer_ollama is True
    
    def test_init_from_environment(self):
        """Should read settings from environment."""
        env = {
            "GITHUB_TOKEN": "env_token",
            "OLLAMA_MODEL": "mistral",
            "PREFER_OLLAMA": "true"
        }
        with patch.dict(os.environ, env):
            client = LLMClient()
            
            assert client.github_api_key == "env_token"
            assert client.ollama_model == "mistral"
            assert client.prefer_ollama is True
    
    def test_init_with_custom_ollama_endpoint(self):
        """Should accept custom Ollama endpoint."""
        with patch.dict(os.environ, {"OLLAMA_ENDPOINT": "http://custom:11434"}, clear=True):
            client = LLMClient()
            
            assert client.ollama_endpoint == "http://custom:11434"
    
    def test_init_defaults(self):
        """Should have correct defaults."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GITHUB_TOKEN", None)
            os.environ.pop("OLLAMA_ENDPOINT", None)
            os.environ.pop("OLLAMA_MODEL", None)
            
            client = LLMClient()
            
            assert client.ollama_endpoint == LLMClient.OLLAMA_ENDPOINT
            assert client.ollama_model == LLMClient.OLLAMA_DEFAULT_MODEL


# =============================================================================
# Ollama Availability Tests
# =============================================================================

class TestLLMClientOllamaCheck:
    """Tests for Ollama availability checking."""
    
    @pytest.fixture
    def client(self):
        """Create LLM client."""
        return LLMClient(api_key="test_token")
    
    @pytest.mark.asyncio
    async def test_ollama_available(self, client):
        """Should detect when Ollama is available."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "mistral:latest"}
            ]
        }
        
        with patch('httpx.AsyncClient.get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            
            result = await client._check_ollama_available()
            
            assert result is True
            assert client._ollama_available is True
    
    @pytest.mark.asyncio
    async def test_ollama_model_not_found(self, client):
        """Should detect when model is not available."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "mistral:latest"}  # Not llama3.2
            ]
        }
        
        with patch('httpx.AsyncClient.get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            
            result = await client._check_ollama_available()
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_ollama_not_running(self, client):
        """Should detect when Ollama is not running."""
        with patch('httpx.AsyncClient.get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")
            
            result = await client._check_ollama_available()
            
            assert result is False
            assert client._ollama_available is False
    
    @pytest.mark.asyncio
    async def test_ollama_cached_result(self, client):
        """Should cache Ollama availability check."""
        client._ollama_available = True
        
        # Should return cached value without making a request
        result = await client._check_ollama_available()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_check_ollama_available_failure(self):
        """Should handle Ollama unavailability."""
        client = LLMClient()
        
        with patch('httpx.AsyncClient') as MockClient:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client
            
            result = await client._check_ollama_available()
            
            assert result is False


# =============================================================================
# GitHub Models API Tests
# =============================================================================

class TestLLMClientGitHubModels:
    """Tests for GitHub Models API calls."""
    
    @pytest.fixture
    def client(self):
        """Create LLM client with GitHub token."""
        return LLMClient(api_key="test_token")
    
    @pytest.mark.asyncio
    async def test_github_models_success(self, client):
        """Should call GitHub Models successfully."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        
        with patch.object(client.github_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await client._call_github_models(
                [{"role": "user", "content": "Hello"}],
                temperature=0.7,
                max_tokens=100
            )
            
            assert result == "Test response"
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_github_models_with_params(self, client):
        """Should pass correct parameters to API."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        
        with patch.object(client.github_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            await client._call_github_models(
                [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Hi"}],
                temperature=0.5,
                max_tokens=200
            )
            
            mock_create.assert_called_with(
                model=client.github_model,
                messages=[{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Hi"}],
                temperature=0.5,
                max_tokens=200
            )
    
    @pytest.mark.asyncio
    async def test_call_github_models_no_client(self):
        """Should raise error when GitHub client not initialized."""
        client = LLMClient()
        client.github_client = None
        
        with pytest.raises(RuntimeError, match="GitHub Models client not initialized"):
            await client._call_github_models(
                [{"role": "user", "content": "test"}],
                temperature=0.7,
                max_tokens=100
            )


# =============================================================================
# Ollama API Tests
# =============================================================================

class TestLLMClientOllama:
    """Tests for Ollama API calls."""
    
    @pytest.fixture
    def client(self):
        """Create LLM client."""
        return LLMClient(api_key="test_token")
    
    @pytest.mark.asyncio
    async def test_ollama_success(self, client):
        """Should call Ollama successfully."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Ollama response"}
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            result = await client._call_ollama(
                [{"role": "user", "content": "Hello"}],
                temperature=0.7,
                max_tokens=100
            )
            
            assert result == "Ollama response"
    
    @pytest.mark.asyncio
    async def test_ollama_with_params(self, client):
        """Should pass correct parameters to Ollama."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Response"}}
        mock_response.raise_for_status = MagicMock()
        
        with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            await client._call_ollama(
                [{"role": "user", "content": "Test"}],
                temperature=0.3,
                max_tokens=150
            )
            
            call_args = mock_post.call_args
            body = call_args.kwargs.get('json', call_args[1].get('json', {}))
            
            assert body["model"] == client.ollama_model
            assert body["options"]["temperature"] == 0.3
            assert body["options"]["num_predict"] == 150
    
    @pytest.mark.asyncio
    async def test_call_ollama_api(self):
        """Should call Ollama API correctly."""
        client = LLMClient()
        
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {"content": "Ollama test response"}
            }
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client
            
            result = await client._call_ollama(
                [{"role": "user", "content": "test"}],
                temperature=0.7,
                max_tokens=100
            )
            
            assert result == "Ollama test response"


# =============================================================================
# Fallback Behavior Tests
# =============================================================================

class TestLLMClientFallback:
    """Test fallback behavior between providers."""
    
    @pytest.fixture
    def client(self):
        """Create LLM client with GitHub token."""
        return LLMClient(api_key="test_token")
    
    @pytest.mark.asyncio
    async def test_github_rate_limit_fallback_to_ollama(self):
        """Should fallback to Ollama on GitHub rate limit."""
        with patch("llm.client.AsyncOpenAI") as MockOpenAI:
            mock_github = MagicMock()
            mock_chat = MagicMock()
            mock_chat.completions.create = AsyncMock(
                side_effect=Exception("429 RateLimitReached")
            )
            mock_github.chat = mock_chat
            MockOpenAI.return_value = mock_github
            
            client = LLMClient()
            client._github_available = True
            client.github_client = mock_github
            
            with patch.object(client, "_check_ollama_available", new_callable=AsyncMock) as mock_check:
                mock_check.return_value = True
                
                with patch.object(client, "_call_ollama", new_callable=AsyncMock) as mock_ollama:
                    mock_ollama.return_value = "Ollama response"
                    
                    result = await client.chat([{"role": "user", "content": "test"}])
                    
                    assert result == "Ollama response"
                    assert client._github_available is False
    
    @pytest.mark.asyncio
    async def test_github_error_fallback_to_ollama(self):
        """Should fallback to Ollama on GitHub general error."""
        with patch("llm.client.AsyncOpenAI") as MockOpenAI:
            mock_github = MagicMock()
            mock_chat = MagicMock()
            mock_chat.completions.create = AsyncMock(
                side_effect=Exception("Connection error")
            )
            mock_github.chat = mock_chat
            MockOpenAI.return_value = mock_github
            
            client = LLMClient()
            client._github_available = True
            client.github_client = mock_github
            
            with patch.object(client, "_check_ollama_available", new_callable=AsyncMock) as mock_check:
                mock_check.return_value = True
                
                with patch.object(client, "_call_ollama", new_callable=AsyncMock) as mock_ollama:
                    mock_ollama.return_value = "Ollama fallback"
                    
                    result = await client.chat([{"role": "user", "content": "test"}])
                    
                    assert result == "Ollama fallback"
    
    @pytest.mark.asyncio
    async def test_prefer_ollama_first(self):
        """Should try Ollama first when prefer_ollama is True."""
        client = LLMClient(prefer_ollama=True)
        
        with patch.object(client, "_check_ollama_available", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = True
            
            with patch.object(client, "_call_ollama", new_callable=AsyncMock) as mock_ollama:
                mock_ollama.return_value = "Ollama preferred"
                
                result = await client.chat([{"role": "user", "content": "test"}])
                
                assert result == "Ollama preferred"
    
    @pytest.mark.asyncio
    async def test_prefer_ollama_fallback_to_github(self):
        """Should fallback to GitHub when prefer_ollama but Ollama fails."""
        with patch("llm.client.AsyncOpenAI") as MockOpenAI:
            mock_github = MagicMock()
            mock_chat = MagicMock()
            mock_completions = MagicMock()
            mock_completions.create = AsyncMock(return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="GitHub response"))]
            ))
            mock_chat.completions = mock_completions
            mock_github.chat = mock_chat
            MockOpenAI.return_value = mock_github
            
            client = LLMClient(prefer_ollama=True)
            client._github_available = True
            client.github_client = mock_github
            
            with patch.object(client, "_check_ollama_available", new_callable=AsyncMock) as mock_check:
                mock_check.return_value = True
                
                with patch.object(client, "_call_ollama", new_callable=AsyncMock) as mock_ollama:
                    mock_ollama.side_effect = Exception("Ollama error")
                    
                    result = await client.chat([{"role": "user", "content": "test"}])
                    
                    assert result == "GitHub response"


# =============================================================================
# Main Chat Method Tests
# =============================================================================

class TestLLMClientChat:
    """Tests for main chat method with fallback logic."""
    
    @pytest.fixture
    def client(self):
        """Create LLM client with GitHub token."""
        return LLMClient(api_key="test_token")
    
    @pytest.mark.asyncio
    async def test_chat_github_primary(self, client):
        """Should use GitHub Models as primary."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="GitHub response"))]
        
        with patch.object(client.github_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await client.chat(
                [{"role": "user", "content": "Hello"}]
            )
            
            assert result == "GitHub response"
    
    @pytest.mark.asyncio
    async def test_chat_fallback_to_ollama(self, client):
        """Should fall back to Ollama on GitHub rate limit."""
        client._ollama_available = True
        
        mock_ollama_response = MagicMock()
        mock_ollama_response.json.return_value = {"message": {"content": "Ollama fallback"}}
        mock_ollama_response.raise_for_status = MagicMock()
        
        with patch.object(client.github_client.chat.completions, 'create', new_callable=AsyncMock) as mock_github:
            mock_github.side_effect = Exception("429 RateLimitReached")
            
            with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_ollama:
                mock_ollama.return_value = mock_ollama_response
                
                result = await client.chat([{"role": "user", "content": "Hello"}])
                
                assert result == "Ollama fallback"
    
    @pytest.mark.asyncio
    async def test_chat_prefer_ollama(self):
        """Should use Ollama as primary when preferred."""
        client = LLMClient(api_key="test_token", prefer_ollama=True)
        client._ollama_available = True
        
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Ollama primary"}}
        mock_response.raise_for_status = MagicMock()
        
        with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_ollama:
            mock_ollama.return_value = mock_response
            
            result = await client.chat([{"role": "user", "content": "Hello"}])
            
            assert result == "Ollama primary"
    
    @pytest.mark.asyncio
    async def test_chat_no_providers_available(self):
        """Should raise error when no providers available."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GITHUB_TOKEN", None)
            client = LLMClient(api_key=None)
            client._ollama_available = False
            client._github_available = False
            
            with patch.object(client, '_check_ollama_available', new_callable=AsyncMock, return_value=False):
                with pytest.raises(RuntimeError, match="No LLM provider available"):
                    await client.chat([{"role": "user", "content": "Hello"}])


# =============================================================================
# Generate Guidance Tests
# =============================================================================

class TestLLMClientGenerateGuidance:
    """Tests for generate_guidance method."""
    
    @pytest.fixture
    def client(self):
        """Create LLM client."""
        return LLMClient(api_key="test_token")
    
    @pytest.mark.asyncio
    async def test_generate_guidance_success(self, client):
        """Should generate guidance successfully."""
        with patch.object(client, 'chat', new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = "✅ Safe to exercise outdoors!"
            
            result = await client.generate_guidance(
                system_prompt="You are an advisor.",
                user_prompt="Is it safe to run?"
            )
            
            assert result == "✅ Safe to exercise outdoors!"
            mock_chat.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_guidance_with_prompts(self, client):
        """Should pass system and user prompts correctly."""
        with patch.object(client, 'chat', new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = "Guidance"
            
            await client.generate_guidance(
                system_prompt="System instruction",
                user_prompt="User question"
            )
            
            call_args = mock_chat.call_args
            messages = call_args[0][0]
            
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "System instruction"
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "User question"


# =============================================================================
# Provider Status Tests
# =============================================================================

class TestLLMClientProviderStatus:
    """Tests for provider status reporting."""
    
    @pytest.fixture
    def client(self):
        """Create LLM client."""
        return LLMClient(api_key="test_token")
    
    def test_get_provider_status(self, client):
        """Should return provider status dict."""
        client._github_available = True
        client._ollama_available = False
        
        status = client.get_provider_status()
        
        assert "github_models" in status
        assert "ollama" in status
        assert status["github_models"]["available"] is True
        assert status["ollama"]["available"] is False
    
    def test_provider_status_with_models(self, client):
        """Should include model names in status."""
        client._github_available = True
        client._ollama_available = True
        
        status = client.get_provider_status()
        
        assert status["github_models"]["model"] == client.github_model
        assert status["ollama"]["model"] == client.ollama_model
    
    def test_provider_status_all_fields(self):
        """Should return status of all providers."""
        client = LLMClient()
        
        status = client.get_provider_status()
        
        assert "github_models" in status
        assert "ollama" in status
        assert "prefer_ollama" in status
    
    @pytest.mark.asyncio
    async def test_get_provider_status_async(self):
        """Should return provider status."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "test-key"}, clear=True):
            client = LLMClient(api_key="test-key")
            client._ollama_available = True
            
            status = client.get_provider_status()
            
            assert "github_models" in status
            assert "ollama" in status


# =============================================================================
# SyncLLMClient Tests
# =============================================================================

class TestSyncLLMClient:
    """Test SyncLLMClient class."""
    
    def test_init_with_api_key(self):
        """Should initialize with provided API key."""
        with patch.dict(os.environ, {}, clear=True):
            client = SyncLLMClient(api_key="test-key")
            
            assert client.api_key == "test-key"
            assert client._use_ollama is False
    
    def test_init_without_api_key_uses_ollama(self):
        """Should use Ollama when no API key provided."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GITHUB_TOKEN", None)
            
            client = SyncLLMClient(api_key=None)
            
            assert client._use_ollama is True
    
    def test_init_with_env_token(self):
        """Should use GITHUB_TOKEN from environment."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "env-token"}, clear=True):
            client = SyncLLMClient()
            
            assert client.api_key == "env-token"
    
    def test_init_custom_model(self):
        """Should accept custom model."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "test-key"}, clear=True):
            client = SyncLLMClient(model="gpt-4")
            
            assert client.model == "gpt-4"
    
    def test_chat_with_github(self):
        """Should use GitHub client for chat."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "test-key"}, clear=True):
            client = SyncLLMClient(api_key="test-key")
            
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Hello!"
            
            client.github_client = MagicMock()
            client.github_client.chat.completions.create.return_value = mock_response
            
            result = client.chat([{"role": "user", "content": "Hi"}])
            
            assert result == "Hello!"
    
    def test_chat_github_rate_limit_fallback(self):
        """Should fallback to Ollama on 429 rate limit."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "test-key"}, clear=True):
            client = SyncLLMClient(api_key="test-key")
            
            client.github_client = MagicMock()
            client.github_client.chat.completions.create.side_effect = Exception("429 Too Many Requests")
            
            mock_response = MagicMock()
            mock_response.json.return_value = {"message": {"content": "Ollama response"}}
            mock_response.raise_for_status = MagicMock()
            
            with patch('httpx.Client') as MockHttpClient:
                mock_httpx = MagicMock()
                mock_httpx.post.return_value = mock_response
                mock_httpx.__enter__ = MagicMock(return_value=mock_httpx)
                mock_httpx.__exit__ = MagicMock(return_value=None)
                MockHttpClient.return_value = mock_httpx
                
                result = client.chat([{"role": "user", "content": "Hi"}])
                
                assert result == "Ollama response"
                assert client._use_ollama is True
    
    def test_chat_github_other_error_raises(self):
        """Should raise non-rate-limit errors."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "test-key"}, clear=True):
            client = SyncLLMClient(api_key="test-key")
            
            client.github_client = MagicMock()
            client.github_client.chat.completions.create.side_effect = Exception("Connection error")
            
            with pytest.raises(Exception, match="Connection error"):
                client.chat([{"role": "user", "content": "Hi"}])
    
    def test_chat_ollama_direct(self):
        """Should use Ollama directly when no GitHub key."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GITHUB_TOKEN", None)
            
            client = SyncLLMClient(api_key=None)
            
            mock_response = MagicMock()
            mock_response.json.return_value = {"message": {"content": "Ollama response"}}
            mock_response.raise_for_status = MagicMock()
            
            with patch('httpx.Client') as MockHttpClient:
                mock_httpx = MagicMock()
                mock_httpx.post.return_value = mock_response
                mock_httpx.__enter__ = MagicMock(return_value=mock_httpx)
                mock_httpx.__exit__ = MagicMock(return_value=None)
                MockHttpClient.return_value = mock_httpx
                
                result = client.chat([{"role": "user", "content": "Hi"}])
                
                assert result == "Ollama response"
    
    def test_sync_client_initialization_no_token(self):
        """Should initialize without GitHub token."""
        with patch.dict("os.environ", {}, clear=True):
            client = SyncLLMClient()
            
            assert client._use_ollama is True
    
    def test_sync_client_chat_ollama(self):
        """Should use Ollama when no GitHub token."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("httpx.Client") as MockClient:
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.json.return_value = {"message": {"content": "Ollama response"}}
                mock_response.raise_for_status = MagicMock()
                mock_client.post.return_value = mock_response
                mock_client.__enter__ = MagicMock(return_value=mock_client)
                mock_client.__exit__ = MagicMock(return_value=None)
                MockClient.return_value = mock_client
                
                client = SyncLLMClient()
                result = client.chat([{"role": "user", "content": "test"}])
                
                assert result == "Ollama response"
