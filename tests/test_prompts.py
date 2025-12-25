"""
Tests for LLM prompt templates.

These tests verify:
- Air quality prompt formatting
- APOD prompt formatting
- Data quality warnings
- Proper variable substitution
"""

import pytest

from llm.prompts import AirQualityPrompts, APODPrompts


class TestAirQualityPrompts:
    """Tests for air quality prompt templates."""
    
    def test_system_prompt_exists(self):
        """Should have system prompt defined."""
        assert AirQualityPrompts.SYSTEM_PROMPT is not None
        assert len(AirQualityPrompts.SYSTEM_PROMPT) > 100
    
    def test_system_prompt_contains_guidelines(self):
        """Should contain WHO guidelines."""
        prompt = AirQualityPrompts.SYSTEM_PROMPT
        
        assert "PM2.5" in prompt
        assert "PM10" in prompt
        assert "WHO" in prompt or "guidelines" in prompt.lower()
    
    def test_system_prompt_contains_temperature(self):
        """Should contain temperature guidelines."""
        prompt = AirQualityPrompts.SYSTEM_PROMPT
        
        assert "Temperature" in prompt or "temperature" in prompt
        assert "°C" in prompt or "Celsius" in prompt
    
    def test_user_template_has_placeholders(self):
        """Should have all required placeholders."""
        template = AirQualityPrompts.USER_TEMPLATE
        
        placeholders = [
            "latitude", "longitude", "hours",
            "pm25_avg", "pm10_avg",
            "temp_avg", "temp_min", "temp_max",
            "air_quality_score", "weather_quality_score"
        ]
        
        for placeholder in placeholders:
            assert f"{{{placeholder}" in template, f"Missing placeholder: {placeholder}"
    
    def test_format_user_prompt_good_quality(self):
        """Should format prompt with good data quality."""
        prompt = AirQualityPrompts.format_user_prompt(
            latitude=42.6977,
            longitude=23.3219,
            hours=6,
            pm25_avg=15.5,
            pm10_avg=28.3,
            temp_avg=18.2,
            temp_min=15.0,
            temp_max=22.0,
            air_quality_score=0.95,
            weather_quality_score=0.98
        )
        
        assert "42.6977" in prompt
        assert "23.3219" in prompt
        assert "15.5" in prompt
        assert "WARNING" not in prompt
    
    def test_format_user_prompt_low_quality(self):
        """Should include warning for low data quality."""
        prompt = AirQualityPrompts.format_user_prompt(
            latitude=42.6977,
            longitude=23.3219,
            hours=6,
            pm25_avg=15.5,
            pm10_avg=28.3,
            temp_avg=18.2,
            temp_min=15.0,
            temp_max=22.0,
            air_quality_score=0.5,  # Low quality
            weather_quality_score=0.6
        )
        
        assert "WARNING" in prompt or "caution" in prompt.lower()
    
    def test_format_user_prompt_short_period(self):
        """Should indicate short-term forecast."""
        prompt = AirQualityPrompts.format_user_prompt(
            latitude=42.6977,
            longitude=23.3219,
            hours=3,
            pm25_avg=15.5,
            pm10_avg=28.3,
            temp_avg=18.2,
            temp_min=15.0,
            temp_max=22.0,
            air_quality_score=0.95,
            weather_quality_score=0.98
        )
        
        assert "short-term" in prompt.lower() or "3" in prompt
    
    def test_format_user_prompt_long_period(self):
        """Should add context for long forecasts."""
        prompt = AirQualityPrompts.format_user_prompt(
            latitude=42.6977,
            longitude=23.3219,
            hours=72,
            pm25_avg=15.5,
            pm10_avg=28.3,
            temp_avg=18.2,
            temp_min=15.0,
            temp_max=22.0,
            air_quality_score=0.95,
            weather_quality_score=0.98
        )
        
        assert "3 days" in prompt or "recheck" in prompt.lower()
    
    def test_low_quality_warning_format(self):
        """Should format low quality warning correctly."""
        warning = AirQualityPrompts.LOW_QUALITY_WARNING.format(
            quality_score=0.65
        )
        
        assert "65%" in warning or "0.65" in warning
        assert "WARNING" in warning


class TestAPODPrompts:
    """Tests for APOD prompt templates."""
    
    def test_system_prompt_exists(self):
        """Should have system prompt defined."""
        assert APODPrompts.SYSTEM_PROMPT is not None
        assert len(APODPrompts.SYSTEM_PROMPT) > 50
    
    def test_system_prompt_mentions_2_lines(self):
        """Should mention 2-line summary requirement."""
        prompt = APODPrompts.SYSTEM_PROMPT
        
        assert "2" in prompt
        assert "line" in prompt.lower() or "sentence" in prompt.lower()
    
    def test_user_template_has_placeholders(self):
        """Should have all required placeholders."""
        template = APODPrompts.USER_TEMPLATE
        
        assert "{title}" in template
        assert "{date}" in template
        assert "{explanation}" in template
    
    def test_format_user_prompt(self):
        """Should format APOD prompt correctly."""
        prompt = APODPrompts.format_user_prompt(
            title="Orion Nebula",
            date="2024-12-24",
            explanation="The Orion Nebula is a diffuse nebula situated in the constellation Orion."
        )
        
        assert "Orion Nebula" in prompt
        assert "2024-12-24" in prompt
        assert "diffuse nebula" in prompt
    
    def test_format_user_prompt_long_explanation(self):
        """Should truncate very long explanations."""
        long_explanation = "A" * 2000
        
        prompt = APODPrompts.format_user_prompt(
            title="Test",
            date="2024-12-24",
            explanation=long_explanation
        )
        
        # Should be truncated
        assert len(prompt) < 2000 + 500  # Some buffer for template
        assert "..." in prompt
    
    def test_format_user_prompt_short_explanation(self):
        """Should not truncate short explanations."""
        prompt = APODPrompts.format_user_prompt(
            title="Test Image",
            date="2024-12-24",
            explanation="Short explanation."
        )
        
        assert "Short explanation." in prompt
        # Should not have truncation marker at end of explanation
        lines = prompt.split("\n")
        explanation_part = [l for l in lines if "Short explanation" in l]
        if explanation_part:
            assert not explanation_part[0].endswith("...")


class TestPromptTemplateConsistency:
    """Tests for prompt template consistency."""
    
    def test_air_quality_prompt_is_complete(self):
        """Should produce complete prompts without missing values."""
        prompt = AirQualityPrompts.format_user_prompt(
            latitude=0.0,
            longitude=0.0,
            hours=1,
            pm25_avg=0.0,
            pm10_avg=0.0,
            temp_avg=0.0,
            temp_min=0.0,
            temp_max=0.0,
            air_quality_score=1.0,
            weather_quality_score=1.0
        )
        
        # Should not have unfilled placeholders
        assert "{" not in prompt or "}" not in prompt.split("{")[0]
    
    def test_apod_prompt_is_complete(self):
        """Should produce complete prompts without missing values."""
        prompt = APODPrompts.format_user_prompt(
            title="Test",
            date="2024-01-01",
            explanation="Test"
        )
        
        # Should not have unfilled placeholders
        assert "{title}" not in prompt
        assert "{date}" not in prompt
        assert "{explanation}" not in prompt


class TestPromptEdgeCases:
    """Edge case tests for prompts."""
    
    def test_air_quality_negative_values(self):
        """Should handle negative temperature."""
        prompt = AirQualityPrompts.format_user_prompt(
            latitude=-90.0,  # South pole
            longitude=0.0,
            hours=6,
            pm25_avg=5.0,
            pm10_avg=10.0,
            temp_avg=-40.0,  # Very cold
            temp_min=-50.0,
            temp_max=-30.0,
            air_quality_score=0.9,
            weather_quality_score=0.9
        )
        
        assert "-40" in prompt or "-40.0" in prompt
        assert "-90" in prompt
    
    def test_air_quality_extreme_values(self):
        """Should handle extreme pollution values."""
        prompt = AirQualityPrompts.format_user_prompt(
            latitude=42.6977,
            longitude=23.3219,
            hours=6,
            pm25_avg=500.0,  # Extremely high
            pm10_avg=800.0,  # Extremely high
            temp_avg=20.0,
            temp_min=15.0,
            temp_max=25.0,
            air_quality_score=0.95,
            weather_quality_score=0.95
        )
        
        assert "500" in prompt
        assert "800" in prompt
    
    def test_apod_special_characters(self):
        """Should handle special characters in title/explanation."""
        prompt = APODPrompts.format_user_prompt(
            title="M31: The Andromeda Galaxy",
            date="2024-12-24",
            explanation="The galaxy known as M31, or NGC 224, is approximately 2.5 million light-years away."
        )
        
        assert "M31" in prompt
        assert "NGC 224" in prompt
    
    def test_apod_unicode_characters(self):
        """Should handle unicode characters."""
        prompt = APODPrompts.format_user_prompt(
            title="星空 (Starry Sky)",
            date="2024-12-24",
            explanation="A beautiful 星空 view with α Centauri."
        )
        
        assert "星空" in prompt
        assert "α Centauri" in prompt
