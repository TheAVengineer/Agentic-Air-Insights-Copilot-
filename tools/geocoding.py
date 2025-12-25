"""
Geocoding Service - Location name to coordinates conversion.

Uses OpenStreetMap Nominatim API (free, no API key required).
Handles country detection and suggests cities via LLM.
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple

import httpx

from llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class GeocodingResult:
    """Result from geocoding a location name."""
    coords: Optional[Tuple[float, float]]  # (latitude, longitude)
    location_name: str  # Display name
    is_country: bool  # True if this is a country, not a city
    country: Optional[str]  # Country name if available


class GeocodingService:
    """
    Geocoding service using OpenStreetMap Nominatim.
    
    Features:
    - Free geocoding without API key
    - Country detection (to ask for specific city)
    - LLM-powered city suggestions for countries
    """
    
    NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
    USER_AGENT = "AirInsightsAgent/1.0"
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
    
    async def geocode(self, location_name: str) -> Optional[GeocodingResult]:
        """
        Convert a location name to coordinates.
        
        Args:
            location_name: City, region, or country name
        
        Returns:
            GeocodingResult with coordinates and metadata, or None if not found
        """
        # Validate input - reject too short or invalid queries
        # Validate input - require minimum 2 characters
        clean_name = location_name.strip() if location_name else ""
        
        if len(clean_name) < 2:
            logger.warning(f"Location name too short (min 2 chars): '{location_name}'")
            return None
        
        # Reject purely numeric queries or those without letters
        if clean_name.isdigit() or not any(c.isalpha() for c in clean_name):
            logger.warning(f"Invalid location name (no letters): '{location_name}'")
            return None
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.NOMINATIM_URL,
                    params={
                        "q": location_name,
                        "format": "json",
                        "limit": 1,
                        "addressdetails": 1,
                        "accept-language": "en",  # Force English names
                    },
                    headers={"User-Agent": self.USER_AGENT},
                    timeout=10.0,
                )
                response.raise_for_status()
                results = response.json()
                
                if not results:
                    return None
                
                result = results[0]
                lat = float(result["lat"])
                lon = float(result["lon"])
                display_name = result.get("display_name", location_name)
                
                # The "addresstype" field is the most reliable indicator
                # It can be: country, state, city, town, village, suburb, etc.
                address_type = result.get("addresstype", "unknown")
                address = result.get("address", {})
                
                # Check if this is a country (not a city/town/village)
                # A location is a country ONLY if:
                # 1. addresstype is "country" OR
                # 2. addresstype is missing but type is "country"
                is_country = address_type == "country" or \
                             (address_type == "unknown" and result.get("type") == "country")
                
                # Get short location name based on address type
                if is_country:
                    short_name = address.get("country", location_name)
                else:
                    # Use addresstype to find the right name in address
                    short_name = address.get(address_type) or \
                                 address.get("city") or \
                                 address.get("town") or \
                                 address.get("village") or \
                                 display_name.split(",")[0]
                
                return GeocodingResult(
                    coords=(lat, lon),
                    location_name=short_name,
                    is_country=is_country,
                    country=address.get("country")
                )
                
        except Exception as e:
            logger.error(f"Geocoding failed for '{location_name}': {e}")
            return None
    
    async def get_country_cities(self, country_name: str) -> Tuple[List[str], str]:
        """
        Get major cities for a country using LLM.
        
        Args:
            country_name: Name of the country
        
        Returns:
            Tuple of (list of city names, brief country info)
        """
        try:
            response = await self.llm.chat([
                {"role": "system", "content": (
                    "You provide brief, factual information about countries. "
                    "Respond in JSON format only."
                )},
                {"role": "user", "content": (
                    f"For {country_name}, provide:\n"
                    "1. List of 5 major cities (largest/most important)\n"
                    "2. One sentence about the country\n\n"
                    "JSON format: {\"cities\": [...], \"info\": \"...\"}"
                )}
            ], temperature=0.3, max_tokens=200)
            
            # Parse response
            import json
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            data = json.loads(response.strip())
            cities = data.get("cities", [])[:5]
            info = data.get("info", "")
            
            return cities, info
            
        except Exception as e:
            logger.warning(f"Failed to get cities for {country_name}: {e}")
            return [], ""
