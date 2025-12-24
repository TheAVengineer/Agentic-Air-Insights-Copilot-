"""
Tests for input validation and data quality functions.

These tests verify:
- Latitude/longitude bounds validation
- Hours range validation  
- Data quality scoring
- Safe average calculations
"""

import pytest
from policies.validation import (
    InputValidator,
    ValidationResult,
    validate_data_quality,
    calculate_safe_average,
    get_data_quality_level,
    validator,
)


class TestInputValidator:
    """Tests for InputValidator class."""
    
    def test_validate_latitude_valid(self):
        """Valid latitudes should pass."""
        valid_cases = [0.0, 45.0, -45.0, 90.0, -90.0, 42.6977]
        
        for lat in valid_cases:
            is_valid, error = validator.validate_latitude(lat)
            assert is_valid, f"Latitude {lat} should be valid"
            assert error is None
    
    def test_validate_latitude_invalid(self):
        """Invalid latitudes should fail."""
        invalid_cases = [91.0, -91.0, 100.0, -180.0, 500.0]
        
        for lat in invalid_cases:
            is_valid, error = validator.validate_latitude(lat)
            assert not is_valid, f"Latitude {lat} should be invalid"
            assert error is not None
            assert "Latitude" in error
    
    def test_validate_longitude_valid(self):
        """Valid longitudes should pass."""
        valid_cases = [0.0, 90.0, -90.0, 180.0, -180.0, 23.3219]
        
        for lon in valid_cases:
            is_valid, error = validator.validate_longitude(lon)
            assert is_valid, f"Longitude {lon} should be valid"
            assert error is None
    
    def test_validate_longitude_invalid(self):
        """Invalid longitudes should fail."""
        invalid_cases = [181.0, -181.0, 200.0, 360.0]
        
        for lon in invalid_cases:
            is_valid, error = validator.validate_longitude(lon)
            assert not is_valid, f"Longitude {lon} should be invalid"
            assert error is not None
            assert "Longitude" in error
    
    def test_validate_hours_valid(self):
        """Valid hours should pass."""
        valid_cases = [1, 6, 24, 48, 72, 168, 384]  # Extended to support 16-day forecasts
        
        for hours in valid_cases:
            is_valid, error = validator.validate_hours(hours)
            assert is_valid, f"Hours {hours} should be valid"
            assert error is None
    
    def test_validate_hours_invalid(self):
        """Invalid hours should fail."""
        invalid_cases = [0, -1, 385, 500, 1000]  # Now max is 384 (16 days)
        
        for hours in invalid_cases:
            is_valid, error = validator.validate_hours(hours)
            assert not is_valid, f"Hours {hours} should be invalid"
            assert error is not None
            assert "Hours" in error
    
    def test_validate_analyze_request_valid(self):
        """Valid analyze request should pass validation."""
        result = validator.validate_analyze_request(
            latitude=42.6977,
            longitude=23.3219,
            hours=6,
        )
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_analyze_request_invalid_all(self):
        """Request with all invalid params should collect all errors."""
        result = validator.validate_analyze_request(
            latitude=100.0,  # Invalid
            longitude=200.0,  # Invalid
            hours=500,  # Invalid (now max is 384)
        )
        
        assert not result.is_valid
        assert len(result.errors) == 3
    
    def test_validate_analyze_request_polar_warning(self):
        """Polar locations should trigger a warning."""
        result = validator.validate_analyze_request(
            latitude=89.0,  # Near pole
            longitude=0.0,
            hours=6,
        )
        
        assert result.is_valid
        assert result.has_warnings
        assert any("polar" in w.lower() for w in result.warnings)
    
    def test_validate_analyze_request_long_forecast_warning(self):
        """Long forecasts should trigger a warning."""
        result = validator.validate_analyze_request(
            latitude=42.6977,
            longitude=23.3219,
            hours=60,  # > 48 hours
        )
        
        assert result.is_valid
        assert result.has_warnings
        assert any("accuracy" in w.lower() for w in result.warnings)


class TestDataQualityValidation:
    """Tests for data quality validation functions."""
    
    def test_validate_data_quality_full(self):
        """100% data quality should pass."""
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        quality, warnings = validate_data_quality(data)
        
        assert quality == 1.0
        assert len(warnings) == 0
    
    def test_validate_data_quality_with_nulls(self):
        """Data with nulls should have reduced quality."""
        data = [10.0, None, 30.0, None, 50.0]
        quality, warnings = validate_data_quality(data)
        
        assert quality == 0.6  # 3/5 valid
        assert len(warnings) > 0
    
    def test_validate_data_quality_empty(self):
        """Empty data should have 0 quality."""
        quality, warnings = validate_data_quality([])
        
        assert quality == 0.0
        assert len(warnings) > 0
    
    def test_validate_data_quality_all_nulls(self):
        """All null data should have 0 quality."""
        data = [None, None, None]
        quality, warnings = validate_data_quality(data)
        
        assert quality == 0.0
    
    def test_validate_data_quality_custom_threshold(self):
        """Custom threshold should affect warnings."""
        data = [10.0, 20.0, None]  # 66% quality
        
        # Should warn with 0.8 threshold
        quality, warnings = validate_data_quality(data, threshold=0.8)
        assert len(warnings) > 0
        
        # Should not warn with 0.5 threshold
        quality, warnings = validate_data_quality(data, threshold=0.5)
        assert len(warnings) == 0


class TestSafeAverageCalculation:
    """Tests for safe average calculation."""
    
    def test_calculate_safe_average_all_valid(self):
        """Average of all valid values."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        avg = calculate_safe_average(values)
        
        assert avg == 30.0
    
    def test_calculate_safe_average_with_nulls(self):
        """Average should exclude None values."""
        values = [10.0, None, 30.0, None, 50.0]
        avg = calculate_safe_average(values)
        
        assert avg == 30.0  # (10 + 30 + 50) / 3
    
    def test_calculate_safe_average_empty(self):
        """Empty list should return 0."""
        avg = calculate_safe_average([])
        
        assert avg == 0.0
    
    def test_calculate_safe_average_all_nulls(self):
        """All None should return 0."""
        avg = calculate_safe_average([None, None, None])
        
        assert avg == 0.0
    
    def test_calculate_safe_average_single_value(self):
        """Single value should return that value."""
        avg = calculate_safe_average([42.5])
        
        assert avg == 42.5
    
    def test_calculate_safe_average_negative_values(self):
        """Negative values should be handled correctly."""
        values = [-10.0, -5.0, 0.0, 5.0, 10.0]
        avg = calculate_safe_average(values)
        
        assert avg == 0.0


class TestDataQualityLevel:
    """Tests for data quality level classification."""
    
    def test_get_data_quality_level_high(self):
        """>= 90% should be HIGH."""
        assert get_data_quality_level(1.0) == "HIGH"
        assert get_data_quality_level(0.95) == "HIGH"
        assert get_data_quality_level(0.90) == "HIGH"
    
    def test_get_data_quality_level_medium(self):
        """80-90% should be MEDIUM."""
        assert get_data_quality_level(0.89) == "MEDIUM"
        assert get_data_quality_level(0.85) == "MEDIUM"
        assert get_data_quality_level(0.80) == "MEDIUM"
    
    def test_get_data_quality_level_low(self):
        """50-80% should be LOW."""
        assert get_data_quality_level(0.79) == "LOW"
        assert get_data_quality_level(0.65) == "LOW"
        assert get_data_quality_level(0.50) == "LOW"
    
    def test_get_data_quality_level_insufficient(self):
        """< 50% should be INSUFFICIENT."""
        assert get_data_quality_level(0.49) == "INSUFFICIENT"
        assert get_data_quality_level(0.25) == "INSUFFICIENT"
        assert get_data_quality_level(0.0) == "INSUFFICIENT"


class TestValidationResult:
    """Tests for ValidationResult dataclass."""
    
    def test_validation_result_valid(self):
        """Valid result properties."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
        )
        
        assert result.is_valid
        assert not result.has_errors
        assert not result.has_warnings
    
    def test_validation_result_with_errors(self):
        """Result with errors."""
        result = ValidationResult(
            is_valid=False,
            errors=["Error 1", "Error 2"],
            warnings=[],
        )
        
        assert not result.is_valid
        assert result.has_errors
        assert not result.has_warnings
    
    def test_validation_result_with_warnings(self):
        """Result with warnings but valid."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Warning 1"],
        )
        
        assert result.is_valid
        assert not result.has_errors
        assert result.has_warnings
    
    def test_validation_result_to_dict(self):
        """Result should convert to dict."""
        result = ValidationResult(
            is_valid=True,
            errors=["err"],
            warnings=["warn"],
        )
        
        d = result.to_dict()
        assert d["is_valid"] == True
        assert d["errors"] == ["err"]
        assert d["warnings"] == ["warn"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
