"""
Tests for weather client (Open-Meteo API).

These tests verify:
- Air quality data fetching
- Weather data fetching
- Combined data retrieval
- Retry logic with exponential backoff
- Data quality calculations
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import asyncio
import httpx

from tools.weather_client import (
    WeatherClient,
    AirQualityData,
    WeatherData,
    DailyForecast,
)


class TestAirQualityData:
    """Tests for AirQualityData dataclass."""
    
    def test_pm25_avg_with_values(self):
        """Should calculate PM2.5 average correctly."""
        data = AirQualityData(
            pm25=[10.0, 20.0, 30.0, 40.0, 50.0],
            pm10=[],
            timestamps=[]
        )
        
        assert data.pm25_avg == 30.0
    
    def test_pm25_avg_with_none_values(self):
        """Should exclude None values from average."""
        data = AirQualityData(
            pm25=[10.0, None, 30.0, None, 50.0],
            pm10=[],
            timestamps=[]
        )
        
        assert data.pm25_avg == 30.0  # (10 + 30 + 50) / 3
    
    def test_pm25_avg_empty(self):
        """Should return 0 for empty list."""
        data = AirQualityData(pm25=[], pm10=[], timestamps=[])
        
        assert data.pm25_avg == 0.0
    
    def test_pm10_avg_with_values(self):
        """Should calculate PM10 average correctly."""
        data = AirQualityData(
            pm25=[],
            pm10=[20.0, 40.0, 60.0],
            timestamps=[]
        )
        
        assert data.pm10_avg == 40.0
    
    def test_data_quality_full(self):
        """Should return 1.0 for complete data."""
        data = AirQualityData(
            pm25=[10.0, 20.0, 30.0],
            pm10=[40.0, 50.0, 60.0],
            timestamps=[]
        )
        
        assert data.data_quality == 1.0
    
    def test_data_quality_with_nulls(self):
        """Should calculate quality based on non-null values."""
        data = AirQualityData(
            pm25=[10.0, None, 30.0],  # 2/3 valid
            pm10=[40.0, 50.0, None],  # 2/3 valid
            timestamps=[]
        )
        
        # 4 valid out of 6 total = 0.666...
        assert abs(data.data_quality - 0.667) < 0.01
    
    def test_data_quality_empty(self):
        """Should return 0 for empty data."""
        data = AirQualityData(pm25=[], pm10=[], timestamps=[])
        
        assert data.data_quality == 0.0


class TestWeatherData:
    """Tests for WeatherData dataclass."""
    
    def test_temp_avg_with_values(self):
        """Should calculate temperature average correctly."""
        data = WeatherData(
            temperature=[15.0, 20.0, 25.0],
            timestamps=[]
        )
        
        assert data.temp_avg == 20.0
    
    def test_temp_avg_with_none_values(self):
        """Should exclude None values from average."""
        data = WeatherData(
            temperature=[15.0, None, 25.0],
            timestamps=[]
        )
        
        assert data.temp_avg == 20.0
    
    def test_temp_min(self):
        """Should return minimum temperature."""
        data = WeatherData(
            temperature=[15.0, 10.0, 25.0, 20.0],
            timestamps=[]
        )
        
        assert data.temp_min == 10.0
    
    def test_temp_max(self):
        """Should return maximum temperature."""
        data = WeatherData(
            temperature=[15.0, 10.0, 25.0, 20.0],
            timestamps=[]
        )
        
        assert data.temp_max == 25.0
    
    def test_temp_empty(self):
        """Should return 0 for empty data."""
        data = WeatherData(temperature=[], timestamps=[])
        
        assert data.temp_avg == 0.0
        assert data.temp_min == 0.0
        assert data.temp_max == 0.0
    
    def test_data_quality(self):
        """Should calculate data quality correctly."""
        data = WeatherData(
            temperature=[15.0, None, 25.0, None],
            timestamps=[]
        )
        
        assert data.data_quality == 0.5


class TestDailyForecast:
    """Tests for DailyForecast dataclass."""
    
    def test_weather_code_clear(self):
        """Should return clear sky description."""
        desc, emoji = DailyForecast.weather_code_to_description(0)
        
        assert desc == "Clear sky"
        assert emoji == "☀️"
    
    def test_weather_code_rain(self):
        """Should return rain description."""
        desc, emoji = DailyForecast.weather_code_to_description(63)
        
        assert desc == "Moderate rain"
        assert emoji == "🌧️"
    
    def test_weather_code_snow(self):
        """Should return snow description."""
        desc, emoji = DailyForecast.weather_code_to_description(75)
        
        assert desc == "Heavy snow"
        assert emoji == "❄️"
    
    def test_weather_code_thunderstorm(self):
        """Should return thunderstorm description."""
        desc, emoji = DailyForecast.weather_code_to_description(95)
        
        assert desc == "Thunderstorm"
        assert emoji == "⛈️"
    
    def test_weather_code_unknown(self):
        """Should return unknown for invalid codes."""
        desc, emoji = DailyForecast.weather_code_to_description(999)
        
        assert desc == "Unknown"
        assert emoji == "❓"
    
    def test_get_day_summary(self):
        """Should return day summary dict."""
        forecast = DailyForecast(
            dates=["2024-12-24"],
            temp_max=[25.0],
            temp_min=[15.0],
            precipitation_sum=[0.0],
            precipitation_probability=[10],
            weather_code=[0],
            sunrise=["06:30"],
            sunset=["17:30"]
        )
        
        summary = forecast.get_day_summary(0)
        
        assert summary["date"] == "2024-12-24"
        assert summary["temp_max"] == 25.0
        assert summary["temp_min"] == 15.0
        assert summary["weather"] == "Clear sky"
        assert summary["emoji"] == "☀️"
    
    def test_get_day_summary_out_of_range(self):
        """Should return empty dict for invalid index."""
        forecast = DailyForecast(dates=["2024-12-24"])
        
        summary = forecast.get_day_summary(5)
        
        assert summary == {}


class TestWeatherClient:
    """Tests for WeatherClient class."""
    
    @pytest.fixture
    def client(self):
        """Create weather client instance."""
        return WeatherClient()
    
    @pytest.mark.asyncio
    async def test_get_air_quality_success(self, client):
        """Should fetch air quality data successfully."""
        mock_response = {
            "hourly": {
                "time": ["2024-12-24T00:00", "2024-12-24T01:00"],
                "pm2_5": [15.0, 18.0],
                "pm10": [25.0, 30.0]
            }
        }
        
        with patch.object(client, '_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.get_air_quality(42.6977, 23.3219, hours=2)
            
            assert result.pm25 == [15.0, 18.0]
            assert result.pm10 == [25.0, 30.0]
            assert result.pm25_avg == 16.5
    
    @pytest.mark.asyncio
    async def test_get_weather_success(self, client):
        """Should fetch weather data successfully."""
        mock_response = {
            "hourly": {
                "time": ["2024-12-24T00:00", "2024-12-24T01:00"],
                "temperature_2m": [20.0, 22.0]
            }
        }
        
        with patch.object(client, '_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.get_weather(42.6977, 23.3219, hours=2)
            
            assert result.temperature == [20.0, 22.0]
            assert result.temp_avg == 21.0
    
    @pytest.mark.asyncio
    async def test_get_combined_data(self, client):
        """Should fetch both air quality and weather."""
        mock_aq_response = {
            "hourly": {
                "time": ["2024-12-24T00:00"],
                "pm2_5": [15.0],
                "pm10": [25.0]
            }
        }
        mock_weather_response = {
            "hourly": {
                "time": ["2024-12-24T00:00"],
                "temperature_2m": [20.0]
            }
        }
        
        with patch.object(client, '_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = [mock_aq_response, mock_weather_response]
            
            air_quality, weather = await client.get_combined_data(42.6977, 23.3219, hours=1)
            
            assert air_quality.pm25_avg == 15.0
            assert weather.temp_avg == 20.0
    
    @pytest.mark.asyncio
    async def test_retry_on_server_error(self, client):
        """Should retry on server errors."""
        with patch('httpx.AsyncClient.get', new_callable=AsyncMock) as mock_get:
            # First two calls fail, third succeeds
            mock_get.side_effect = [
                httpx.HTTPStatusError("Server Error", request=MagicMock(), response=MagicMock(status_code=500)),
                httpx.HTTPStatusError("Server Error", request=MagicMock(), response=MagicMock(status_code=500)),
                MagicMock(
                    status_code=200,
                    json=lambda: {"hourly": {"time": [], "pm2_5": [], "pm10": []}},
                    raise_for_status=lambda: None
                )
            ]
            
            # This should eventually succeed
            with patch.object(client, 'max_retries', 3):
                with patch('asyncio.sleep', new_callable=AsyncMock):
                    result = await client._request_with_retry(
                        "https://api.test.com",
                        {},
                        "test"
                    )
                    
                    assert result is not None
    
    @pytest.mark.asyncio
    async def test_get_weekly_forecast(self, client):
        """Should fetch daily forecast data."""
        mock_response = {
            "daily": {
                "time": ["2024-12-24", "2024-12-25"],
                "temperature_2m_max": [25.0, 26.0],
                "temperature_2m_min": [15.0, 16.0],
                "precipitation_sum": [0.0, 2.0],
                "precipitation_probability_max": [10, 60],
                "weather_code": [0, 61],
                "sunrise": ["06:30", "06:31"],
                "sunset": ["17:30", "17:29"]
            }
        }
        
        with patch.object(client, '_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.get_weekly_forecast(42.6977, 23.3219, days=2)
            
            assert len(result.dates) == 2
            assert result.temp_max == [25.0, 26.0]
            assert result.weather_code == [0, 61]


class TestWeatherClientEdgeCases:
    """Edge case tests for weather client."""
    
    @pytest.fixture
    def client(self):
        """Create weather client instance."""
        return WeatherClient()
    
    @pytest.mark.asyncio
    async def test_empty_response_handling(self, client):
        """Should handle empty API response."""
        mock_response = {
            "hourly": {
                "time": [],
                "pm2_5": [],
                "pm10": []
            }
        }
        
        with patch.object(client, '_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.get_air_quality(42.6977, 23.3219, hours=6)
            
            assert result.pm25 == []
            assert result.data_quality == 0.0
    
    @pytest.mark.asyncio
    async def test_null_values_in_response(self, client):
        """Should handle null values in API response."""
        mock_response = {
            "hourly": {
                "time": ["2024-12-24T00:00", "2024-12-24T01:00"],
                "pm2_5": [15.0, None],
                "pm10": [None, 30.0]
            }
        }
        
        with patch.object(client, '_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.get_air_quality(42.6977, 23.3219, hours=2)
            
            assert result.pm25_avg == 15.0  # Only valid value
            assert result.pm10_avg == 30.0  # Only valid value
    
    @pytest.mark.asyncio
    async def test_latitude_longitude_bounds(self, client):
        """Should work with extreme coordinates."""
        mock_response = {
            "hourly": {
                "time": ["2024-12-24T00:00"],
                "temperature_2m": [-50.0]  # Antarctica cold
            }
        }
        
        with patch.object(client, '_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            # South pole
            result = await client.get_weather(-90.0, 0.0, hours=1)
            
            assert result.temp_avg == -50.0



class TestGetHistoricalDaily:
    """Test get_historical_daily method."""
    
    @pytest.mark.asyncio
    async def test_historical_daily_success(self):
        """Should fetch historical daily data successfully."""
        mock_response = {
            "daily": {
                "time": ["2025-12-22", "2025-12-23"],
                "temperature_2m_max": [10.0, 12.0],
                "temperature_2m_min": [2.0, 4.0],
                "precipitation_sum": [0.0, 5.0],
                "weather_code": [0, 61],
                "sunrise": ["07:00", "07:01"],
                "sunset": ["16:30", "16:31"]
            }
        }
        
        with patch.object(WeatherClient, '_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            client = WeatherClient()
            result = await client.get_historical_daily(42.69, 23.32, past_days=2)
            
            assert isinstance(result, DailyForecast)
            assert len(result.dates) == 2
            assert result.temp_max == [10.0, 12.0]
    
    @pytest.mark.asyncio
    async def test_historical_daily_caps_at_92_days(self):
        """Should cap past_days at 92."""
        with patch.object(WeatherClient, '_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"daily": {}}
            
            client = WeatherClient()
            await client.get_historical_daily(42.69, 23.32, past_days=200)
            
            # Check that past_days was capped in the request
            call_args = mock_request.call_args
            params = call_args[0][1]
            assert params["past_days"] == 92
    
    @pytest.mark.asyncio
    async def test_historical_daily_empty_response(self):
        """Should handle empty response gracefully."""
        mock_response = {"daily": {}}
        
        with patch.object(WeatherClient, '_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            client = WeatherClient()
            result = await client.get_historical_daily(42.69, 23.32, past_days=7)
            
            assert isinstance(result, DailyForecast)
            assert result.dates == []


class TestGetWeeklyForecastExtended:
    """Extended tests for get_weekly_forecast."""
    
    @pytest.mark.asyncio
    async def test_weekly_forecast_full_response(self):
        """Should handle full response with all fields."""
        mock_response = {
            "daily": {
                "time": ["2025-12-24", "2025-12-25", "2025-12-26"],
                "temperature_2m_max": [10, 12, 8],
                "temperature_2m_min": [2, 4, 1],
                "precipitation_sum": [0, 5, 0],
                "precipitation_probability_max": [10, 80, 20],
                "weather_code": [0, 61, 3],
                "sunrise": ["07:00"] * 3,
                "sunset": ["16:30"] * 3
            }
        }
        
        with patch.object(WeatherClient, '_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            client = WeatherClient()
            result = await client.get_weekly_forecast(42.69, 23.32, days=3)
            
            assert len(result.dates) == 3
            assert len(result.temp_max) == 3
    
    @pytest.mark.asyncio
    async def test_weekly_forecast_caps_at_16_days(self):
        """Should cap days at 16."""
        with patch.object(WeatherClient, '_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"daily": {}}
            
            client = WeatherClient()
            await client.get_weekly_forecast(42.69, 23.32, days=30)
            
            # Check that forecast_days was capped
            call_args = mock_request.call_args
            params = call_args[0][1]
            assert params["forecast_days"] == 16


class TestDailyForecastExtended:
    """Extended tests for DailyForecast dataclass."""
    
    def test_get_day_summary_with_all_fields(self):
        """Should return complete day summary."""
        forecast = DailyForecast(
            dates=["2025-12-24"],
            temp_max=[12.0],
            temp_min=[5.0],
            precipitation_sum=[2.5],
            precipitation_probability=[80],
            weather_code=[61],  # Rain
            sunrise=["07:00"],
            sunset=["16:30"]
        )
        
        summary = forecast.get_day_summary(0)
        
        assert summary["date"] == "2025-12-24"
        assert summary["temp_max"] == 12.0
        assert summary["temp_min"] == 5.0
    
    def test_get_day_summary_clear_weather(self):
        """Should identify clear weather."""
        forecast = DailyForecast(
            dates=["2025-12-24"],
            temp_max=[20.0],
            temp_min=[10.0],
            precipitation_sum=[0.0],
            precipitation_probability=[5],
            weather_code=[0],  # Clear
            sunrise=["06:00"],
            sunset=["18:00"]
        )
        
        summary = forecast.get_day_summary(0)
        
        assert "Clear" in summary["weather"]
        assert "☀" in summary["emoji"]
    
    def test_get_day_summary_rain(self):
        """Should identify rain weather."""
        forecast = DailyForecast(
            dates=["2025-12-24"],
            temp_max=[15.0],
            temp_min=[8.0],
            precipitation_sum=[10.0],
            precipitation_probability=[90],
            weather_code=[61],  # Rain
            sunrise=["07:00"],
            sunset=["16:30"]
        )
        
        summary = forecast.get_day_summary(0)
        
        assert "rain" in summary["weather"].lower() or "Rain" in summary["weather"]
    
    def test_get_day_summary_out_of_range(self):
        """Should return empty dict for out of range index."""
        forecast = DailyForecast(
            dates=["2025-12-24"],
            temp_max=[12.0],
            temp_min=[5.0],
            precipitation_sum=[0.0],
            precipitation_probability=[10],
            weather_code=[0],
            sunrise=["07:00"],
            sunset=["16:30"]
        )
        
        summary = forecast.get_day_summary(5)  # Out of range
        
        assert summary == {}


class TestAirQualityDataExtended:
    """Extended tests for AirQualityData."""
    
    def test_pm25_avg_with_mixed_values(self):
        """Should calculate PM2.5 average excluding nulls."""
        data = AirQualityData(
            pm25=[10.0, None, 20.0, None],
            pm10=[15.0, 25.0, None, None],
            timestamps=["t1", "t2", "t3", "t4"]
        )
        
        assert data.pm25_avg == 15.0  # (10 + 20) / 2
    
    def test_pm10_avg_with_mixed_values(self):
        """Should calculate PM10 average excluding nulls."""
        data = AirQualityData(
            pm25=[10.0, 20.0],
            pm10=[15.0, None, 25.0],
            timestamps=["t1", "t2", "t3"]
        )
        
        assert data.pm10_avg == 20.0  # (15 + 25) / 2
    
    def test_data_quality_full(self):
        """Should return 1.0 for all valid data."""
        data = AirQualityData(
            pm25=[10.0, 20.0],
            pm10=[15.0, 25.0],
            timestamps=["t1", "t2"]
        )
        
        assert data.data_quality == 1.0
    
    def test_data_quality_empty(self):
        """Should return 0.0 for empty data."""
        data = AirQualityData(
            pm25=[],
            pm10=[],
            timestamps=[]
        )
        
        assert data.data_quality == 0.0


class TestWeatherDataExtended:
    """Extended tests for WeatherData."""
    
    def test_temp_avg_with_values(self):
        """Should calculate temperature average."""
        data = WeatherData(
            temperature=[10.0, 15.0, 20.0],
            timestamps=["t1", "t2", "t3"]
        )
        
        assert data.temp_avg == 15.0
    
    def test_temp_avg_with_nulls(self):
        """Should handle null values in temperature."""
        data = WeatherData(
            temperature=[10.0, None, 20.0],
            timestamps=["t1", "t2", "t3"]
        )
        
        assert data.temp_avg == 15.0  # (10 + 20) / 2
    
    def test_temp_min_max(self):
        """Should calculate min and max temperature."""
        data = WeatherData(
            temperature=[10.0, 15.0, 20.0],
            timestamps=["t1", "t2", "t3"]
        )
        
        assert data.temp_min == 10.0
        assert data.temp_max == 20.0
    
    def test_temp_empty(self):
        """Should handle empty temperature list."""
        data = WeatherData(
            temperature=[],
            timestamps=[]
        )
        
        assert data.temp_avg == 0.0
        assert data.temp_min == 0.0
        assert data.temp_max == 0.0



class TestWeatherClientRetryLogic:
    """Test retry logic and error handling."""
    
    @pytest.mark.asyncio
    async def test_network_timeout_handling(self):
        """Should handle timeout errors gracefully."""
        client = WeatherClient()
        
        with patch.object(client, "_request_with_retry", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = httpx.TimeoutException("Timeout")
            
            with pytest.raises(httpx.TimeoutException):
                await client.get_weather(42.69, 23.32, 1)


class TestWeatherClientAirQuality:
    """Test air quality specific handling."""
    
    @pytest.mark.asyncio
    async def test_air_quality_with_missing_fields(self):
        """Should handle missing optional fields."""
        client = WeatherClient()
        
        mock_data = {
            "hourly": {
                "time": ["2025-12-24T00:00", "2025-12-24T01:00"],
                "pm2_5": [10.0, 12.0],
                "pm10": [20.0, 22.0],
            }
        }
        
        with patch.object(client, "_request_with_retry", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_data
            
            result = await client.get_air_quality(42.69, 23.32, 2)
            
            assert isinstance(result, AirQualityData)
            assert result.pm25 == [10.0, 12.0]
            assert result.pm10 == [20.0, 22.0]
    
    @pytest.mark.asyncio
    async def test_air_quality_with_all_fields(self):
        """Should parse all air quality fields."""
        client = WeatherClient()
        
        mock_data = {
            "hourly": {
                "time": ["2025-12-24T00:00"],
                "pm2_5": [15.0],
                "pm10": [25.0],
            }
        }
        
        with patch.object(client, "_request_with_retry", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_data
            
            result = await client.get_air_quality(42.69, 23.32, 1)
            
            assert result.pm25[0] == 15.0
            assert result.pm10[0] == 25.0


class TestWeatherClientHistorical:
    """Test historical data retrieval."""
    
    @pytest.mark.asyncio
    async def test_historical_daily_single_day(self):
        """Should get historical data for single day."""
        client = WeatherClient()
        
        mock_data = {
            "daily": {
                "time": ["2025-12-23"],
                "temperature_2m_max": [10.0],
                "temperature_2m_min": [2.0],
                "precipitation_sum": [0.0],
                "weather_code": [0],
                "sunrise": ["2025-12-23T07:00"],
                "sunset": ["2025-12-23T16:30"]
            }
        }
        
        with patch.object(client, "_request_with_retry", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_data
            
            result = await client.get_historical_daily(42.69, 23.32, 1)
            
            assert isinstance(result, DailyForecast)
            assert len(result.dates) == 1
            assert result.temp_max[0] == 10.0
    
    @pytest.mark.asyncio
    async def test_historical_daily_week(self):
        """Should get historical data for a week."""
        client = WeatherClient()
        
        dates = [f"2025-12-{17+i}" for i in range(7)]
        mock_data = {
            "daily": {
                "time": dates,
                "temperature_2m_max": [8.0, 9.0, 10.0, 11.0, 10.0, 9.0, 8.0],
                "temperature_2m_min": [0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0],
                "precipitation_sum": [0.0, 2.0, 0.0, 0.0, 5.0, 0.0, 0.0],
                "weather_code": [0, 61, 0, 0, 61, 0, 0],
                "sunrise": ["07:00"] * 7,
                "sunset": ["16:30"] * 7
            }
        }
        
        with patch.object(client, "_request_with_retry", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_data
            
            result = await client.get_historical_daily(42.69, 23.32, 7)
            
            assert len(result.dates) == 7
            assert sum(result.precipitation_sum) == 7.0


class TestWeatherClientWeeklyForecast:
    """Test weekly forecast retrieval."""
    
    @pytest.mark.asyncio
    async def test_weekly_forecast_3_days(self):
        """Should get 3 day forecast."""
        client = WeatherClient()
        
        mock_data = {
            "daily": {
                "time": ["2025-12-24", "2025-12-25", "2025-12-26"],
                "temperature_2m_max": [10.0, 12.0, 8.0],
                "temperature_2m_min": [2.0, 4.0, 1.0],
                "precipitation_sum": [0.0, 5.0, 0.0],
                "precipitation_probability_max": [10, 80, 20],
                "weather_code": [0, 61, 3],
                "sunrise": ["07:00", "07:01", "07:02"],
                "sunset": ["16:30", "16:31", "16:32"]
            }
        }
        
        with patch.object(client, "_request_with_retry", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_data
            
            result = await client.get_weekly_forecast(42.69, 23.32, 3)
            
            assert isinstance(result, DailyForecast)
            assert len(result.dates) == 3
            assert result.precipitation_probability == [10, 80, 20]
    
    @pytest.mark.asyncio
    async def test_weekly_forecast_full_week(self):
        """Should get full week forecast."""
        client = WeatherClient()
        
        dates = [f"2025-12-{24+i}" for i in range(7)]
        mock_data = {
            "daily": {
                "time": dates,
                "temperature_2m_max": [10.0] * 7,
                "temperature_2m_min": [2.0] * 7,
                "precipitation_sum": [0.0] * 7,
                "precipitation_probability_max": [10] * 7,
                "weather_code": [0] * 7,
                "sunrise": ["07:00"] * 7,
                "sunset": ["16:30"] * 7
            }
        }
        
        with patch.object(client, "_request_with_retry", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_data
            
            result = await client.get_weekly_forecast(42.69, 23.32, 7)
            
            assert len(result.dates) == 7


class TestWeatherClientEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.asyncio
    async def test_get_weather_basic(self):
        """Should get basic weather data."""
        client = WeatherClient()
        
        mock_data = {
            "hourly": {
                "time": ["2025-12-24T00:00"],
                "temperature_2m": [10.0],
            }
        }
        
        with patch.object(client, "_request_with_retry", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_data
            
            result = await client.get_weather(42.69, 23.32, 1)
            
            assert isinstance(result, WeatherData)
            assert len(result.temperature) == 1
    
    @pytest.mark.asyncio
    async def test_extreme_temperature(self):
        """Should handle extreme temperatures."""
        client = WeatherClient()
        
        mock_data = {
            "hourly": {
                "time": ["2025-12-24T00:00"],
                "temperature_2m": [-40.0],  # Extreme cold
            }
        }
        
        with patch.object(client, "_request_with_retry", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_data
            
            result = await client.get_weather(90.0, 0.0, 1)
            
            assert result.temperature[0] == -40.0


class TestWeatherDataClasses:
    """Test weather data class initialization and properties."""
    
    def test_weather_data_initialization(self):
        """Should initialize WeatherData correctly."""
        weather = WeatherData(
            temperature=[10.0, 12.0, 11.0],
            timestamps=["2025-12-24T00:00", "2025-12-24T01:00", "2025-12-24T02:00"]
        )
        
        assert weather.temperature[0] == 10.0
        assert len(weather.timestamps) == 3
    
    def test_weather_data_temp_avg(self):
        """Should calculate temperature average."""
        weather = WeatherData(
            temperature=[10.0, 20.0, 30.0],
            timestamps=[]
        )
        
        assert weather.temp_avg == 20.0
    
    def test_weather_data_temp_min(self):
        """Should get minimum temperature."""
        weather = WeatherData(
            temperature=[10.0, 5.0, 15.0],
            timestamps=[]
        )
        
        assert weather.temp_min == 5.0
    
    def test_weather_data_temp_max(self):
        """Should get maximum temperature."""
        weather = WeatherData(
            temperature=[10.0, 5.0, 15.0],
            timestamps=[]
        )
        
        assert weather.temp_max == 15.0
    
    def test_weather_data_empty(self):
        """Should handle empty weather data."""
        weather = WeatherData(
            temperature=[],
            timestamps=[]
        )
        
        assert weather.temp_avg == 0.0
        assert weather.temp_min == 0.0
        assert weather.temp_max == 0.0
    
    def test_weather_data_quality(self):
        """Should calculate data quality."""
        weather = WeatherData(
            temperature=[10.0, None, 15.0],  # type: ignore
            timestamps=[]
        )
        
        # 2 valid out of 3
        assert weather.data_quality == pytest.approx(2/3)
    
    def test_air_quality_data_initialization(self):
        """Should initialize AirQualityData correctly."""
        aq = AirQualityData(
            pm25=[15.0],
            pm10=[25.0],
            timestamps=["2025-12-24T00:00"]
        )
        
        assert aq.pm25[0] == 15.0
        assert aq.pm10[0] == 25.0
    
    def test_air_quality_pm25_avg(self):
        """Should calculate PM2.5 average."""
        aq = AirQualityData(
            pm25=[10.0, 20.0, 30.0],
            pm10=[],
            timestamps=[]
        )
        
        assert aq.pm25_avg == 20.0
    
    def test_air_quality_pm10_avg(self):
        """Should calculate PM10 average."""
        aq = AirQualityData(
            pm25=[],
            pm10=[10.0, 20.0, 30.0],
            timestamps=[]
        )
        
        assert aq.pm10_avg == 20.0
    
    def test_air_quality_empty(self):
        """Should handle empty air quality data."""
        aq = AirQualityData(
            pm25=[],
            pm10=[],
            timestamps=[]
        )
        
        assert aq.pm25_avg == 0.0
        assert aq.pm10_avg == 0.0
    
    def test_air_quality_data_quality(self):
        """Should calculate data quality for air quality."""
        aq = AirQualityData(
            pm25=[10.0, None, 15.0],  # type: ignore
            pm10=[20.0, 25.0, None],  # type: ignore
            timestamps=[]
        )
        
        # 4 valid out of 6
        assert aq.data_quality == pytest.approx(4/6)
    
    def test_daily_forecast_initialization(self):
        """Should initialize DailyForecast correctly."""
        forecast = DailyForecast(
            dates=["2025-12-24"],
            temp_max=[10.0],
            temp_min=[2.0],
            precipitation_sum=[0.0],
            precipitation_probability=[10],
            weather_code=[0],
            sunrise=["07:00"],
            sunset=["16:30"]
        )
        
        assert forecast.dates[0] == "2025-12-24"
        assert forecast.temp_max[0] == 10.0
    
    def test_daily_forecast_weather_code_to_description(self):
        """Should convert weather codes to descriptions."""
        desc, emoji = DailyForecast.weather_code_to_description(0)
        assert desc == "Clear sky"
        assert emoji == "☀️"
        
        desc, emoji = DailyForecast.weather_code_to_description(61)
        assert desc == "Slight rain"
        assert emoji == "🌧️"
        
        desc, emoji = DailyForecast.weather_code_to_description(999)
        assert desc == "Unknown"
        assert emoji == "❓"
    
    def test_daily_forecast_get_day_summary(self):
        """Should get day summary."""
        forecast = DailyForecast(
            dates=["2025-12-24"],
            temp_max=[10.0],
            temp_min=[2.0],
            precipitation_sum=[5.0],
            precipitation_probability=[80],
            weather_code=[61],
            sunrise=["07:00"],
            sunset=["16:30"]
        )
        
        summary = forecast.get_day_summary(0)
        
        assert summary["date"] == "2025-12-24"
        assert summary["temp_max"] == 10.0
        assert summary["temp_min"] == 2.0
        assert summary["weather"] == "Slight rain"
        assert summary["emoji"] == "🌧️"
    
    def test_daily_forecast_get_day_summary_out_of_range(self):
        """Should return empty dict for out of range index."""
        forecast = DailyForecast(
            dates=["2025-12-24"],
            temp_max=[10.0],
            temp_min=[2.0],
            precipitation_sum=[0.0],
            precipitation_probability=[10],
            weather_code=[0],
            sunrise=["07:00"],
            sunset=["16:30"]
        )
        
        summary = forecast.get_day_summary(5)
        
        assert summary == {}

class TestWeatherClientExtendedForecast:
    """Test extended forecast method."""
    
    @pytest.mark.asyncio
    async def test_get_extended_forecast(self):
        """Should get extended forecast with all data types."""
        client = WeatherClient()
        
        mock_aq = AirQualityData(
            pm25=[10.0, 12.0],
            pm10=[20.0, 22.0],
            timestamps=["2025-12-24T00:00", "2025-12-24T01:00"]
        )
        
        mock_weather = WeatherData(
            temperature=[10.0, 11.0],
            timestamps=["2025-12-24T00:00", "2025-12-24T01:00"]
        )
        
        mock_daily = DailyForecast(
            dates=["2025-12-24", "2025-12-25"],
            temp_max=[10.0, 12.0],
            temp_min=[2.0, 4.0],
            precipitation_sum=[0.0, 0.0],
            precipitation_probability=[10, 20],
            weather_code=[0, 1],
            sunrise=["07:00", "07:01"],
            sunset=["16:30", "16:31"]
        )
        
        with patch.object(client, "get_air_quality", new_callable=AsyncMock) as mock_get_aq:
            mock_get_aq.return_value = mock_aq
            
            with patch.object(client, "get_weather", new_callable=AsyncMock) as mock_get_weather:
                mock_get_weather.return_value = mock_weather
                
                with patch.object(client, "get_weekly_forecast", new_callable=AsyncMock) as mock_get_weekly:
                    mock_get_weekly.return_value = mock_daily
                    
                    aq, weather, daily = await client.get_extended_forecast(42.69, 23.32, 72)
                    
                    assert isinstance(aq, AirQualityData)
                    assert isinstance(weather, WeatherData)
                    assert isinstance(daily, DailyForecast)


class TestWeatherClientCombinedData:
    """Test combined data fetching."""
    
    @pytest.mark.asyncio
    async def test_get_combined_data(self):
        """Should get both air quality and weather concurrently."""
        client = WeatherClient()
        
        mock_aq = AirQualityData(
            pm25=[10.0],
            pm10=[20.0],
            timestamps=["2025-12-24T00:00"]
        )
        
        mock_weather = WeatherData(
            temperature=[10.0],
            timestamps=["2025-12-24T00:00"]
        )
        
        with patch.object(client, "get_air_quality", new_callable=AsyncMock) as mock_get_aq:
            mock_get_aq.return_value = mock_aq
            
            with patch.object(client, "get_weather", new_callable=AsyncMock) as mock_get_weather:
                mock_get_weather.return_value = mock_weather
                
                aq, weather = await client.get_combined_data(42.69, 23.32, 6)
                
                assert isinstance(aq, AirQualityData)
                assert isinstance(weather, WeatherData)


class TestWeatherClientErrorHandling:
    """Test error handling in weather client."""
    
    @pytest.mark.asyncio
    async def test_get_air_quality_error(self):
        """Should raise exception on API error."""
        client = WeatherClient()
        
        with patch.object(client, "_request_with_retry", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = httpx.HTTPStatusError(
                "Server Error",
                request=MagicMock(),
                response=MagicMock(status_code=500)
            )
            
            with pytest.raises(httpx.HTTPStatusError):
                await client.get_air_quality(42.69, 23.32, 6)
    
    @pytest.mark.asyncio
    async def test_get_weather_error(self):
        """Should raise exception on API error."""
        client = WeatherClient()
        
        with patch.object(client, "_request_with_retry", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = httpx.RequestError("Network error")
            
            with pytest.raises(httpx.RequestError):
                await client.get_weather(42.69, 23.32, 6)
    
    @pytest.mark.asyncio
    async def test_get_historical_daily_error(self):
        """Should raise exception on API error."""
        client = WeatherClient()
        
        with patch.object(client, "_request_with_retry", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = Exception("API Error")
            
            with pytest.raises(Exception):
                await client.get_historical_daily(42.69, 23.32, 7)


class TestWeatherClientPastDays:
    """Test historical data with past_days parameter."""
    
    @pytest.mark.asyncio
    async def test_get_air_quality_with_past_days(self):
        """Should fetch air quality with past days."""
        client = WeatherClient()
        
        mock_data = {
            "hourly": {
                "time": ["2025-12-20T00:00", "2025-12-21T00:00"],
                "pm2_5": [15.0, 18.0],
                "pm10": [25.0, 28.0],
            }
        }
        
        with patch.object(client, "_request_with_retry", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_data
            
            result = await client.get_air_quality(42.69, 23.32, 24, past_days=3)
            
            assert isinstance(result, AirQualityData)
            assert len(result.pm25) == 2
    
    @pytest.mark.asyncio
    async def test_get_weather_with_past_days(self):
        """Should fetch weather with past days."""
        client = WeatherClient()
        
        mock_data = {
            "hourly": {
                "time": ["2025-12-20T00:00", "2025-12-21T00:00"],
                "temperature_2m": [5.0, 8.0],
            }
        }
        
        with patch.object(client, "_request_with_retry", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_data
            
            result = await client.get_weather(42.69, 23.32, 24, past_days=3)
            
            assert isinstance(result, WeatherData)
            assert len(result.temperature) == 2


class TestWeatherClientRetryLogic:
    """Test retry logic in _request_with_retry."""
    
    @pytest.mark.asyncio
    async def test_request_with_retry_success(self):
        """Should succeed on first try."""
        client = WeatherClient()
        
        mock_data = {"hourly": {"time": [], "temperature_2m": []}}
        
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = mock_data
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client
            
            result = await client._request_with_retry(
                "https://api.example.com/test",
                {"param": "value"},
                "TestOperation"
            )
            
            assert result == mock_data
    
    @pytest.mark.asyncio
    async def test_request_with_retry_client_error(self):
        """Should not retry on 4xx errors."""
        client = WeatherClient()
        
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = "Bad Request"
            mock_response.raise_for_status = MagicMock(
                side_effect=httpx.HTTPStatusError(
                    "Bad Request",
                    request=MagicMock(),
                    response=mock_response
                )
            )
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client
            
            with pytest.raises(httpx.HTTPStatusError):
                await client._request_with_retry(
                    "https://api.example.com/test",
                    {"param": "value"},
                    "TestOperation"
                )
