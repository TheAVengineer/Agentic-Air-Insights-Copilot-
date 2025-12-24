"""
Pydantic models for API request/response validation.

These models define the contract for the Air & Insights API
and are used to auto-generate OpenAPI documentation.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SafetyLevel(str, Enum):
    """Air quality safety levels for exercise."""
    SAFE = "SAFE"
    MODERATE = "MODERATE"
    UNHEALTHY_SENSITIVE = "UNHEALTHY_SENSITIVE"
    UNHEALTHY = "UNHEALTHY"
    VERY_UNHEALTHY = "VERY_UNHEALTHY"
    HAZARDOUS = "HAZARDOUS"
    UNKNOWN = "UNKNOWN"


class DataQuality(str, Enum):
    """Data quality assessment."""
    HIGH = "HIGH"  # >90% data points available
    MEDIUM = "MEDIUM"  # 80-90% data points
    LOW = "LOW"  # <80% data points
    INSUFFICIENT = "INSUFFICIENT"  # <50% data points


class AnalyzeRequest(BaseModel):
    """Request model for /analyze endpoint."""
    
    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Latitude coordinate (-90 to 90)",
        examples=[42.6977]
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Longitude coordinate (-180 to 180)",
        examples=[23.3219]
    )
    hours: int = Field(
        default=6,
        ge=1,
        le=384,
        description="Number of hours to forecast (1-384, up to 16 days)",
        examples=[6]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "latitude": 42.6977,
                    "longitude": 23.3219,
                    "hours": 6
                }
            ]
        }
    }


class AnalyzeResponse(BaseModel):
    """Response model for /analyze endpoint."""
    
    pm25_avg: float = Field(
        ...,
        description="Average PM2.5 concentration in µg/m³",
        examples=[15.5]
    )
    pm10_avg: float = Field(
        ...,
        description="Average PM10 concentration in µg/m³",
        examples=[28.3]
    )
    temp_avg: float = Field(
        ...,
        description="Average temperature in °C",
        examples=[18.2]
    )
    guidance_text: str = Field(
        ...,
        description="LLM-generated actionable guidance for outdoor exercise",
        examples=["✅ Great conditions for outdoor exercise! PM2.5 (15.5 µg/m³) and PM10 (28.3 µg/m³) are within safe limits. Temperature at 18.2°C is optimal. Enjoy your run!"]
    )
    
    # Additional metadata
    safety_level: SafetyLevel = Field(
        default=SafetyLevel.UNKNOWN,
        description="Overall safety level for outdoor exercise"
    )
    data_quality: DataQuality = Field(
        default=DataQuality.HIGH,
        description="Quality assessment of the underlying data"
    )
    location: Optional[str] = Field(
        default=None,
        description="Human-readable location description"
    )
    forecast_hours: int = Field(
        ...,
        description="Number of hours included in forecast"
    )
    attribution: str = Field(
        default="Weather data by Open-Meteo.com",
        description="Data source attribution (required)"
    )
    cached: bool = Field(
        default=False,
        description="Whether response was served from cache"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response generation timestamp (UTC)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "pm25_avg": 15.5,
                    "pm10_avg": 28.3,
                    "temp_avg": 18.2,
                    "guidance_text": "✅ Great conditions for outdoor exercise! PM2.5 (15.5 µg/m³) and PM10 (28.3 µg/m³) are within safe limits. Temperature at 18.2°C is optimal. Enjoy your run!",
                    "safety_level": "SAFE",
                    "data_quality": "HIGH",
                    "location": "Sofia, Bulgaria",
                    "forecast_hours": 6,
                    "attribution": "Weather data by Open-Meteo.com",
                    "cached": False,
                    "timestamp": "2024-12-20T10:30:00Z"
                }
            ]
        }
    }


class APODResponse(BaseModel):
    """Response model for /apod/today endpoint."""
    
    title: str = Field(
        ...,
        description="Title of the Astronomy Picture of the Day",
        examples=["The Orion Nebula in Infrared"]
    )
    url: str = Field(
        ...,
        description="URL to the image or video",
        examples=["https://apod.nasa.gov/apod/image/2412/orion_nebula.jpg"]
    )
    explanation: str = Field(
        ...,
        description="Original NASA explanation of the image"
    )
    summary: Optional[str] = Field(
        default=None,
        description="LLM-generated 2-line summary"
    )
    date: str = Field(
        ...,
        description="Date of the APOD (YYYY-MM-DD format)",
        examples=["2024-12-20"]
    )
    media_type: str = Field(
        default="image",
        description="Type of media (image or video)"
    )
    hdurl: Optional[str] = Field(
        default=None,
        description="URL to high-resolution image (if available)"
    )
    attribution: str = Field(
        default="Image from NASA Astronomy Picture of the Day",
        description="NASA attribution"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "The Orion Nebula in Infrared",
                    "url": "https://apod.nasa.gov/apod/image/2412/orion_nebula.jpg",
                    "explanation": "The Great Nebula in Orion...",
                    "summary": "A stunning infrared view of the Orion Nebula reveals hidden stars and dust structures. This image showcases the birthplace of new stars 1,344 light-years away.",
                    "date": "2024-12-20",
                    "media_type": "image",
                    "hdurl": "https://apod.nasa.gov/apod/image/2412/orion_nebula_hd.jpg",
                    "attribution": "Image from NASA Astronomy Picture of the Day"
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    
    status: str = Field(
        default="healthy",
        description="Service health status"
    )
    version: str = Field(
        default="1.0.0",
        description="API version"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error: str = Field(
        ...,
        description="Error type/code"
    )
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    details: Optional[dict] = Field(
        default=None,
        description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": "VALIDATION_ERROR",
                    "message": "Latitude must be between -90 and 90",
                    "details": {"field": "latitude", "value": 100},
                    "timestamp": "2024-12-20T10:30:00Z"
                }
            ]
        }
    }


# Chat-related models for the web UI
class ChatMessage(BaseModel):
    """Chat message model for web UI."""
    
    role: str = Field(
        ...,
        description="Message role (user or assistant)"
    )
    content: str = Field(
        ...,
        description="Message content"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Message timestamp"
    )


class ChatRequest(BaseModel):
    """Request model for /chat endpoint."""
    
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's natural language query",
        examples=["What's the air quality in Sofia for the next 6 hours?"]
    )


class ChatResponse(BaseModel):
    """Response model for /chat endpoint."""
    
    response: str = Field(
        ...,
        description="Assistant's response"
    )
    intent: Optional[str] = Field(
        default=None,
        description="Detected user intent (analyze, apod, help, unknown)"
    )
    data: Optional[dict] = Field(
        default=None,
        description="Structured data from the query (if any)"
    )
