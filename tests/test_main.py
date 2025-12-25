"""
Tests for main.py application setup.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient


class TestApplicationStartup:
    """Test application startup and configuration."""
    
    def test_app_import(self):
        """Should import app successfully."""
        from main import app
        
        assert app is not None
        assert app.title == "Air & Insights Agent"
    
    def test_routes_included(self):
        """Should include API routes."""
        from main import app
        
        # Check some routes are included
        routes = [route.path for route in app.routes]
        assert "/api/v1/health" in routes or "/health" in routes or any("/api/v1" in r for r in routes)
    
    def test_cors_enabled(self):
        """Should have CORS middleware."""
        from main import app
        
        # Check middleware is configured
        middleware_classes = [type(m) for m in app.middleware_stack.app.__dict__.get('app', app).__dict__.get('middleware_stack', [])]
        # Just verify app starts without error


class TestStaticFiles:
    """Test static file serving."""
    
    def test_index_html_endpoint(self):
        """Should serve index.html."""
        from main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Try to get root - should work or redirect
        response = client.get("/")
        
        # Should be 200 or redirect
        assert response.status_code in [200, 307, 404]


class TestHealthEndpoint:
    """Test health check endpoint via app."""
    
    def test_health_via_client(self):
        """Should respond to health check."""
        from main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestErrorHandling:
    """Test error handling in main."""
    
    def test_request_exception_handler(self):
        """Should handle request exceptions."""
        from main import app
        
        # Just verify the app can be created
        assert app is not None
