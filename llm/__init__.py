# LLM package
"""
LLM client for GitHub Models (OpenAI-compatible) inference.

GitHub Models provides free inference with:
- gpt-4o-mini (recommended for this use case)
- gpt-4o (more capable, also free)

Authentication: GitHub Personal Access Token
Endpoint: https://models.inference.ai.azure.com
"""

from .client import LLMClient
from .prompts import PromptLibrary, AirQualityPrompts, APODPrompts

__all__ = [
    "LLMClient",
    "PromptLibrary",
    "AirQualityPrompts",
    "APODPrompts",
]
