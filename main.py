"""
Air & Insights Agent - FastAPI Application Entry Point

This is the main entry point for the Air & Insights Agent API.
Run with: python main.py or uvicorn main:app --reload

Features:
- REST API with OpenAPI documentation
- CORS support for web UI
- Static file serving for web interface
- Structured logging
- Environment-based configuration
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Import routes after logging is configured
from api.routes import router, set_agent
from agent.orchestrator import AirInsightsAgent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events:
    - Initialize agent on startup
    - Cleanup resources on shutdown
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Air & Insights Agent Starting...")
    logger.info("=" * 60)
    
    try:
        # Initialize the global agent
        agent = AirInsightsAgent()
        set_agent(agent)
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {str(e)}")
        logger.warning("API will start but LLM features may be unavailable")
    
    logger.info(f"Log level: {log_level}")
    logger.info(f"API Documentation: http://localhost:{os.getenv('PORT', 8000)}/docs")
    logger.info(f"Web UI: http://localhost:{os.getenv('PORT', 8000)}/")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("Air & Insights Agent Shutting down...")


# Create FastAPI application
app = FastAPI(
    title="Air & Insights Agent",
    description="""
## 🌤️ Air & Insights Copilot

An agentic assistant that fetches weather & air quality data and provides 
actionable guidance for outdoor activities.

### Features
- **Air Quality Analysis**: PM2.5, PM10, and temperature data with exercise safety guidance
- **NASA APOD**: Astronomy Picture of the Day with AI-generated summaries
- **Smart Caching**: 10-minute TTL to reduce API calls
- **LLM Reasoning**: GitHub Models (GPT-4o-mini) for natural language guidance

### Data Sources
- [Open-Meteo](https://open-meteo.com/) - Weather & Air Quality (no API key required)
- [NASA APOD](https://api.nasa.gov/) - Astronomy Picture of the Day

### Attribution
- Weather data by Open-Meteo.com
- Image from NASA Astronomy Picture of the Day
""",
    version="1.0.0",
    contact={
        "name": "Air & Insights Agent",
        "url": "https://github.com/yourusername/air-insights-agent",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan,
)

# Configure CORS for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = datetime.utcnow()
    
    response = await call_next(request)
    
    duration = (datetime.utcnow() - start_time).total_seconds() * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"- {response.status_code} "
        f"- {duration:.0f}ms"
    )
    
    return response


# Include API routes
app.include_router(router, prefix="/api/v1")

# Also mount at root for convenience (Copilot Studio compatibility)
app.include_router(router)


# Mount static files for web UI
ui_path = Path(__file__).parent / "ui"
if ui_path.exists():
    app.mount("/static", StaticFiles(directory=str(ui_path)), name="static")
    
    @app.get("/", include_in_schema=False)
    async def serve_ui():
        """Serve the web UI."""
        return FileResponse(str(ui_path / "index.html"))
else:
    @app.get("/", include_in_schema=False)
    async def no_ui():
        """Fallback when UI is not available."""
        return JSONResponse({
            "message": "Air & Insights Agent API",
            "docs": "/docs",
            "openapi": "/openapi.json",
        })


# Export OpenAPI schema endpoint
@app.get("/openapi-export.json", include_in_schema=False)
async def export_openapi():
    """
    Export OpenAPI schema for Copilot Studio import.
    
    Use this endpoint to get the OpenAPI specification that can be
    imported into Microsoft Copilot Studio as a Tool/Action.
    """
    return JSONResponse(app.openapi())


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level=log_level.lower(),
    )
