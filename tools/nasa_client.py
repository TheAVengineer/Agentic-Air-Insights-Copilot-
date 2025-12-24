"""
NASA APOD (Astronomy Picture of the Day) API client.

NASA API: https://api.nasa.gov/planetary/apod
- Free to use with DEMO_KEY or personal API key
- Rate limits: DEMO_KEY = 30 requests/hour, Personal key = 1000/hour

Attribution: "Image from NASA Astronomy Picture of the Day" is required.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional

import httpx

from policies import SAFETY_RULES

logger = logging.getLogger(__name__)


@dataclass
class APODData:
    """NASA Astronomy Picture of the Day data."""
    
    title: str
    url: str
    explanation: str
    date: str
    media_type: str = "image"
    hdurl: Optional[str] = None
    copyright: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "title": self.title,
            "url": self.url,
            "explanation": self.explanation,
            "date": self.date,
            "media_type": self.media_type,
            "hdurl": self.hdurl,
            "copyright": self.copyright,
        }


class NASAClient:
    """
    Async client for NASA APOD API.
    
    Implements:
    - API key authentication (DEMO_KEY fallback)
    - Retry logic with exponential backoff
    - Data validation and error handling
    """
    
    APOD_BASE_URL = "https://api.nasa.gov/planetary/apod"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NASA client.
        
        Args:
            api_key: NASA API key. If not provided, uses NASA_API_KEY 
                    environment variable or falls back to DEMO_KEY.
        """
        self.api_key = api_key or os.getenv("NASA_API_KEY", "DEMO_KEY")
        
        agent_config = SAFETY_RULES.get("agent", {})
        self.max_retries = agent_config.get("max_retries", 3)
        self.backoff_base = agent_config.get("retry_backoff_base", 1.0)
        self.backoff_max = agent_config.get("retry_backoff_max", 10.0)
        self.timeout = agent_config.get("request_timeout_seconds", 10)
        
        if self.api_key == "DEMO_KEY":
            logger.warning(
                "Using NASA DEMO_KEY - limited to 30 requests/hour. "
                "Get a free key at https://api.nasa.gov/"
            )
    
    async def _request_with_retry(
        self,
        url: str,
        params: dict,
        operation_name: str
    ) -> dict:
        """
        Make HTTP request with exponential backoff retry logic.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            operation_name: Name for logging purposes
            
        Returns:
            JSON response as dict
            
        Raises:
            httpx.HTTPError: After all retries exhausted
        """
        last_exception = None
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    logger.info(
                        f"{operation_name}: Attempt {attempt + 1}/{self.max_retries}"
                    )
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    
                    logger.info(f"{operation_name}: Success")
                    return response.json()
                    
                except httpx.HTTPStatusError as e:
                    last_exception = e
                    logger.warning(
                        f"{operation_name}: HTTP {e.response.status_code} - "
                        f"{e.response.text[:200]}"
                    )
                    # Don't retry client errors (4xx) except rate limiting
                    if 400 <= e.response.status_code < 500:
                        if e.response.status_code != 429:  # Rate limit
                            raise
                        
                except httpx.RequestError as e:
                    last_exception = e
                    logger.warning(f"{operation_name}: Request error - {str(e)}")
                
                # Calculate backoff time
                if attempt < self.max_retries - 1:
                    backoff = min(
                        self.backoff_base * (2 ** attempt),
                        self.backoff_max
                    )
                    logger.info(f"{operation_name}: Retrying in {backoff}s...")
                    await asyncio.sleep(backoff)
        
        # All retries exhausted
        logger.error(f"{operation_name}: All {self.max_retries} attempts failed")
        raise last_exception or httpx.RequestError("All retries failed")
    
    async def get_apod(self, apod_date: Optional[date] = None) -> APODData:
        """
        Fetch Astronomy Picture of the Day.
        
        Args:
            apod_date: Specific date to fetch (default: today)
            
        Returns:
            APODData with title, URL, and explanation
        """
        params = {
            "api_key": self.api_key,
        }
        
        if apod_date:
            params["date"] = apod_date.isoformat()
        
        try:
            data = await self._request_with_retry(
                self.APOD_BASE_URL,
                params,
                "NASA_APOD"
            )
            
            return APODData(
                title=data.get("title", "Unknown"),
                url=data.get("url", ""),
                explanation=data.get("explanation", ""),
                date=data.get("date", ""),
                media_type=data.get("media_type", "image"),
                hdurl=data.get("hdurl"),
                copyright=data.get("copyright"),
            )
            
        except Exception as e:
            logger.error(f"Failed to fetch APOD: {str(e)}")
            raise
    
    async def get_today(self) -> APODData:
        """
        Convenience method to fetch today's APOD.
        
        Returns:
            APODData for today's astronomy picture
        """
        return await self.get_apod(None)
