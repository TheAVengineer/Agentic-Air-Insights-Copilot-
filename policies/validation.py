"""
Input validation module with policy-driven rules.

Validation rules are loaded from policies/safety_rules.json for:
- Auditability: Changes to validation rules are tracked
- Flexibility: Business users can adjust bounds without code changes
- Consistency: All components use the same validation logic
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from policies import SAFETY_RULES

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0
    
    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class InputValidator:
    """
    Policy-driven input validator for API requests.
    
    Validates:
    - Latitude bounds (-90 to 90)
    - Longitude bounds (-180 to 180)
    - Hours range (1 to 72)
    - Data quality thresholds
    """
    
    def __init__(self):
        """Load validation rules from policy configuration."""
        self.rules = SAFETY_RULES.get("validation", {})
        self.lat_bounds = self.rules.get("latitude", {"min": -90, "max": 90})
        self.lon_bounds = self.rules.get("longitude", {"min": -180, "max": 180})
        self.hours_bounds = self.rules.get("hours", {"min": 1, "max": 72})
    
    def validate_latitude(self, lat: float) -> Tuple[bool, Optional[str]]:
        """
        Validate latitude value.
        
        Args:
            lat: Latitude to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        min_val = self.lat_bounds.get("min", -90)
        max_val = self.lat_bounds.get("max", 90)
        
        if lat < min_val or lat > max_val:
            return False, f"Latitude must be between {min_val} and {max_val}, got {lat}"
        
        return True, None
    
    def validate_longitude(self, lon: float) -> Tuple[bool, Optional[str]]:
        """
        Validate longitude value.
        
        Args:
            lon: Longitude to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        min_val = self.lon_bounds.get("min", -180)
        max_val = self.lon_bounds.get("max", 180)
        
        if lon < min_val or lon > max_val:
            return False, f"Longitude must be between {min_val} and {max_val}, got {lon}"
        
        return True, None
    
    def validate_hours(self, hours: int) -> Tuple[bool, Optional[str]]:
        """
        Validate forecast hours.
        
        Args:
            hours: Number of hours to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        min_val = self.hours_bounds.get("min", 1)
        max_val = self.hours_bounds.get("max", 72)
        
        if hours < min_val or hours > max_val:
            return False, f"Hours must be between {min_val} and {max_val}, got {hours}"
        
        return True, None
    
    def validate_analyze_request(
        self,
        latitude: float,
        longitude: float,
        hours: int
    ) -> ValidationResult:
        """
        Validate all parameters for an analyze request.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            hours: Forecast hours
            
        Returns:
            ValidationResult with any errors or warnings
        """
        errors = []
        warnings = []
        
        # Validate latitude
        lat_valid, lat_error = self.validate_latitude(latitude)
        if not lat_valid:
            errors.append(lat_error)
        
        # Validate longitude
        lon_valid, lon_error = self.validate_longitude(longitude)
        if not lon_valid:
            errors.append(lon_error)
        
        # Validate hours
        hours_valid, hours_error = self.validate_hours(hours)
        if not hours_valid:
            errors.append(hours_error)
        
        # Add warnings for edge cases
        if -90 <= latitude <= -85 or 85 <= latitude <= 90:
            warnings.append("Location is near polar regions - weather data may be limited")
        
        if hours > 48:
            warnings.append("Forecasts beyond 48 hours have reduced accuracy")
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.warning(f"Validation failed: {errors}")
        elif warnings:
            logger.info(f"Validation passed with warnings: {warnings}")
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
        )


def validate_data_quality(
    data: list,
    threshold: float = 0.8,
    field_name: str = "data"
) -> Tuple[float, list[str]]:
    """
    Validate data quality by checking for null/missing values.
    
    Args:
        data: List of values to check
        threshold: Minimum acceptable quality (0.0 to 1.0)
        field_name: Name for error messages
        
    Returns:
        Tuple of (quality_score, warnings)
    """
    if not data:
        return 0.0, [f"No {field_name} available"]
    
    valid_count = sum(1 for v in data if v is not None)
    quality = valid_count / len(data)
    
    warnings = []
    if quality < threshold:
        pct = int(quality * 100)
        warnings.append(
            f"{field_name} quality is low ({pct}% valid) - "
            f"results may be less reliable"
        )
    
    return quality, warnings


def calculate_safe_average(values: list[float]) -> float:
    """
    Calculate average excluding None values.
    
    Args:
        values: List of numeric values (may contain None)
        
    Returns:
        Average of valid values, or 0.0 if no valid values
    """
    valid = [v for v in values if v is not None]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


def get_data_quality_level(quality: float) -> str:
    """
    Convert quality score to human-readable level.
    
    Args:
        quality: Quality score (0.0 to 1.0)
        
    Returns:
        Quality level string (HIGH, MEDIUM, LOW, INSUFFICIENT)
    """
    if quality >= 0.9:
        return "HIGH"
    elif quality >= 0.8:
        return "MEDIUM"
    elif quality >= 0.5:
        return "LOW"
    else:
        return "INSUFFICIENT"


# Create singleton instance for convenience
validator = InputValidator()
