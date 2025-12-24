"""
LLM Client with GitHub Models (primary) and Ollama (fallback).

This client supports two providers:
1. GitHub Models (free) - Primary, uses OpenAI-compatible API
2. Ollama (local) - Fallback when GitHub Models is rate limited

Setup:
- GitHub Models: Set GITHUB_TOKEN environment variable
- Ollama: Install Ollama and pull a model (e.g., llama3.2)

The client automatically falls back to Ollama when GitHub Models fails.
"""

import logging
import os
from typing import Optional, Literal
import httpx

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Async LLM client with automatic fallback.
    
    Primary: GitHub Models (gpt-4o-mini)
    Fallback: Ollama (llama3.2 or configured model)
    
    Automatically switches to Ollama when GitHub Models:
    - Returns 429 (rate limited)
    - Is unavailable
    - Times out
    """
    
    # GitHub Models configuration
    GITHUB_MODELS_ENDPOINT = "https://models.inference.ai.azure.com"
    GITHUB_DEFAULT_MODEL = "gpt-4o-mini"
    
    # Ollama configuration
    OLLAMA_ENDPOINT = "http://localhost:11434"
    OLLAMA_DEFAULT_MODEL = "llama3.2"  # Fast, good for parsing
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        ollama_model: Optional[str] = None,
        prefer_ollama: bool = False,
    ):
        """
        Initialize the LLM client.
        
        Args:
            api_key: GitHub Personal Access Token (defaults to GITHUB_TOKEN env)
            model: GitHub Models model to use
            ollama_model: Ollama model to use as fallback
            prefer_ollama: If True, use Ollama as primary (for testing)
        """
        self.github_api_key = api_key or os.getenv("GITHUB_TOKEN")
        self.github_model = model or self.GITHUB_DEFAULT_MODEL
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", self.OLLAMA_DEFAULT_MODEL)
        self.ollama_endpoint = os.getenv("OLLAMA_ENDPOINT", self.OLLAMA_ENDPOINT)
        self.prefer_ollama = prefer_ollama or os.getenv("PREFER_OLLAMA", "").lower() == "true"
        
        # Track which provider to use
        self._github_available = bool(self.github_api_key) and not self.prefer_ollama
        self._ollama_available: Optional[bool] = None  # Checked on first use
        
        # Initialize GitHub Models client if available
        self.github_client: Optional[AsyncOpenAI] = None
        if self.github_api_key:
            self.github_client = AsyncOpenAI(
                base_url=self.GITHUB_MODELS_ENDPOINT,
                api_key=self.github_api_key,
                max_retries=2,  # Reduce retries (default is 2, but wait time is long)
                timeout=10.0,   # 10 second timeout
            )
        
        provider = "Ollama (preferred)" if self.prefer_ollama else f"GitHub Models ({self.github_model})"
        logger.info(f"LLMClient initialized with model={self.github_model}, primary={provider}")
    
    async def _check_ollama_available(self) -> bool:
        """Check if Ollama is running and has the required model."""
        if self._ollama_available is not None:
            return self._ollama_available
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Check if Ollama is running
                response = await client.get(f"{self.ollama_endpoint}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [m.get("name", "").split(":")[0] for m in models]
                    self._ollama_available = self.ollama_model in model_names
                    if self._ollama_available:
                        logger.info(f"Ollama available with model: {self.ollama_model}")
                    else:
                        logger.warning(f"Ollama running but model '{self.ollama_model}' not found. Available: {model_names}")
                else:
                    self._ollama_available = False
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            self._ollama_available = False
        
        return self._ollama_available
    
    async def _call_github_models(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call GitHub Models API."""
        if not self.github_client:
            raise RuntimeError("GitHub Models client not initialized")
        
        response = await self.github_client.chat.completions.create(
            model=self.github_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    
    async def _call_ollama(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call Ollama API."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.ollama_endpoint}/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                }
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
    
    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        """
        Send a chat completion request with automatic fallback.
        
        Tries GitHub Models first, falls back to Ollama if:
        - Rate limited (429)
        - Server error (5xx)
        - Connection error
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Creativity (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum response length
            
        Returns:
            Generated text response
        """
        # If prefer_ollama, try Ollama first
        if self.prefer_ollama:
            if await self._check_ollama_available():
                try:
                    logger.debug("Using Ollama (preferred)")
                    return await self._call_ollama(messages, temperature, max_tokens)
                except Exception as e:
                    logger.warning(f"Ollama failed: {e}, trying GitHub Models")
        
        # Try GitHub Models
        if self._github_available and self.github_client:
            try:
                return await self._call_github_models(messages, temperature, max_tokens)
            except Exception as e:
                error_str = str(e)
                # Check for rate limiting
                if "429" in error_str or "RateLimitReached" in error_str:
                    logger.warning(f"GitHub Models rate limited, falling back to Ollama")
                    self._github_available = False  # Temporarily disable
                else:
                    logger.error(f"GitHub Models error: {e}")
                    # For other errors, also try fallback
        
        # Fallback to Ollama
        if await self._check_ollama_available():
            try:
                logger.info("Using Ollama fallback")
                return await self._call_ollama(messages, temperature, max_tokens)
            except Exception as e:
                logger.error(f"Ollama fallback failed: {e}")
                raise RuntimeError(f"All LLM providers failed. Last error: {e}")
        
        raise RuntimeError(
            "No LLM provider available. "
            "GitHub Models may be rate limited and Ollama is not running. "
            "Install Ollama: https://ollama.ai and run: ollama pull llama3.2"
        )
    
    async def generate_guidance(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.5,
    ) -> str:
        """
        Generate guidance using system and user prompts.
        
        Args:
            system_prompt: System instructions and persona
            user_prompt: User query with data context
            temperature: Creativity level
            
        Returns:
            Generated guidance text
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self.chat(messages, temperature=temperature)
    
    async def simple_query(self, prompt: str) -> str:
        """Simple single-turn query without system prompt."""
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages)
    
    def get_provider_status(self) -> dict:
        """Get current status of LLM providers."""
        return {
            "github_models": {
                "available": self._github_available,
                "model": self.github_model,
            },
            "ollama": {
                "available": self._ollama_available,
                "model": self.ollama_model,
                "endpoint": self.ollama_endpoint,
            },
            "prefer_ollama": self.prefer_ollama,
        }


# Synchronous wrapper for testing/CLI
class SyncLLMClient:
    """Synchronous wrapper for LLMClient (for testing/CLI use)."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        import httpx
        
        self.api_key = api_key or os.getenv("GITHUB_TOKEN")
        self.model = model or LLMClient.GITHUB_DEFAULT_MODEL
        self.ollama_model = os.getenv("OLLAMA_MODEL", LLMClient.OLLAMA_DEFAULT_MODEL)
        self.ollama_endpoint = os.getenv("OLLAMA_ENDPOINT", LLMClient.OLLAMA_ENDPOINT)
        
        # Try GitHub Models first
        self._use_ollama = not self.api_key
        
        if self.api_key:
            from openai import OpenAI
            self.github_client = OpenAI(
                base_url=LLMClient.GITHUB_MODELS_ENDPOINT,
                api_key=self.api_key,
            )
    
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        """Synchronous chat completion with fallback."""
        import httpx
        
        if not self._use_ollama and hasattr(self, 'github_client'):
            try:
                response = self.github_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                if "429" in str(e):
                    self._use_ollama = True
                else:
                    raise
        
        # Use Ollama
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{self.ollama_endpoint}/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens}
                }
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
