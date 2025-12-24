# 🌤️ Air & Insights Agent

An **agentic AI assistant** that fetches weather & air quality data from free public APIs, reasons about it with a free LLM (GitHub Models), and provides actionable guidance for outdoor activities.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![License](https://img.shields.io/badge/license-MIT-gray)

## ✨ Features

- **🤖 LLM-First Design**: Natural language understanding powered by GitHub Models (GPT-4o-mini)
- **🔄 Automatic Fallback**: Ollama (llama3.2) as local backup when GitHub Models is rate-limited
- **🌡️ Air Quality Analysis**: PM2.5, PM10, and temperature data with exercise safety guidance
- **� 16-Day Forecasts**: Weather predictions up to 16 days ahead
- **📊 Historical Data**: Access up to 92 days of past weather/air quality data
- **� Conversation Context**: Follow-up queries like "what about tomorrow?" or "how about Paris?"
- **🌍 Smart Geocoding**: Understands cities, countries, and coordinates (lat/lon)
- **🌟 NASA APOD**: Astronomy Picture of the Day with AI summaries
- **💾 Smart Caching**: 10-minute TTL to reduce API calls
- **📄 OpenAPI Spec**: Ready for Microsoft Copilot Studio integration

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- GitHub Personal Access Token (for LLM - [get one here](https://github.com/settings/tokens))

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/air-insights-agent.git
cd air-insights-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GITHUB_TOKEN
```

### Running the Application

```bash
# Start the server
python main.py

# Or with uvicorn for development
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Access Points

- **Web UI**: http://localhost:8000/
- **API Docs (Swagger)**: http://localhost:8000/docs
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 📖 API Reference

### POST /analyze

Analyze air quality and get exercise guidance.

**Request:**
```json
{
    "latitude": 42.6977,
    "longitude": 23.3219,
    "hours": 6
}
```

**Response:**
```json
{
    "pm25_avg": 15.5,
    "pm10_avg": 28.3,
    "temp_avg": 18.2,
    "guidance_text": "✅ Great conditions for outdoor exercise!...",
    "safety_level": "SAFE",
    "data_quality": "HIGH",
    "forecast_hours": 6,
    "attribution": "Weather data by Open-Meteo.com",
    "cached": false,
    "timestamp": "2024-12-20T10:30:00Z"
}
```

### GET /apod/today

Get NASA's Astronomy Picture of the Day.

**Response:**
```json
{
    "title": "The Orion Nebula in Infrared",
    "url": "https://apod.nasa.gov/apod/image/...",
    "explanation": "...",
    "summary": "A stunning infrared view of...",
    "date": "2024-12-20",
    "media_type": "image",
    "attribution": "Image from NASA Astronomy Picture of the Day"
}
```

### POST /chat

Natural language chat interface.

**Request:**
```json
{
    "message": "Is it safe to run at 42.6977, 23.3219 for 6 hours?"
}
```

## 🏗️ Architecture

```
air-insights-agent/
├── main.py                 # FastAPI entry point
├── agent/
│   ├── orchestrator.py     # Main agent brain
│   ├── planner.py          # Execution planning
│   └── memory.py           # TTL-based cache
├── api/
│   ├── models.py           # Pydantic schemas
│   └── routes.py           # API endpoints
├── tools/
│   ├── weather_client.py   # Open-Meteo API
│   └── nasa_client.py      # NASA APOD API
├── llm/
│   ├── client.py           # GitHub Models client
│   └── prompts.py          # Prompt templates
├── policies/
│   ├── safety_rules.json   # Business rules (thresholds)
│   └── validation.py       # Input validation
├── tests/                  # Unit & integration tests
├── ui/                     # Web chat interface
└── docs/                   # Documentation
```

### Agentic Flow

```
User Request → Plan → Validate → Cache Check → Fetch APIs → 
Validate Data → LLM Reasoning → Cache Store → Response
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GITHUB_TOKEN` | GitHub PAT for Models API | Yes |
| `NASA_API_KEY` | NASA API key (DEMO_KEY works) | No |
| `LOG_LEVEL` | Logging level (DEBUG/INFO/WARNING) | No |
| `HOST` | Server host | No (default: 0.0.0.0) |
| `PORT` | Server port | No (default: 8000) |

### Policy Configuration

Edit `policies/safety_rules.json` to customize:
- Air quality thresholds (PM2.5, PM10)
- Temperature limits for exercise
- Cache TTL
- Retry behavior

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_validation.py -v
```

## 🔌 Copilot Studio Integration

### Track A: Microsoft Copilot Studio

1. Export OpenAPI spec: `GET /openapi-export.json`
2. In Copilot Studio, create a new Agent
3. Add Tool → REST API → Import OpenAPI
4. Use this tool description:

```
Air Quality Analysis Tool - Analyzes air quality (PM2.5, PM10) and 
weather data for any location and provides exercise safety guidance. 
Call this when users ask about outdoor exercise safety, air quality, 
or running conditions for a specific location.
```

### Track B: Web UI

The included web chat UI (`/`) provides a standalone interface that can be deployed anywhere.

## 📊 Demo Prompts

Try these prompts to see the agent in action:

### Weather & Air Quality
```
What's the weather in London for the next 3 days?
Is it safe to run outside in Tokyo?
Air quality in Sofia for the next week
What about Berlin? (follow-up)
```

### Historical Data (up to 92 days)
```
How was the weather yesterday in Paris?
Air quality in New York last week
Weather in Sydney for the past 5 days
```

### Coordinates Support
```
Weather at 42.6977, 23.3219 for the next 6 hours
Is it safe to exercise at lat 51.5, lon -0.1?
```

### NASA APOD
```
Show today's NASA picture
What's the astronomy picture of the day?
NASA APOD
```

## 🔌 Microsoft Copilot Studio Integration

This project follows **Track A: Microsoft Stack**:

| Component | Implementation |
|-----------|----------------|
| **Service** | Python FastAPI |
| **Orchestration** | Custom planner (`agent/orchestrator.py` + `query_parser.py`) |
| **LLM** | GitHub Models (GPT-4o-mini) + Ollama fallback |
| **OpenAPI** | Auto-generated at `/docs` with Copilot Studio-ready tool descriptions |

### Integration Steps

1. Export OpenAPI: `curl http://localhost:8081/openapi.json > openapi.json`
2. In Copilot Studio → Create Agent → Add Tool → REST API → Import OpenAPI
3. The tool descriptions are optimized for **Generative Orchestration** auto-selection

See [`docs/COPILOT_STUDIO_INTEGRATION.md`](docs/COPILOT_STUDIO_INTEGRATION.md) for detailed instructions.

> **Note**: Copilot Studio requires a Microsoft 365 work/school account. The API is fully prepared for integration and can be demonstrated with workspace access.

## 📝 Attribution

- **Weather Data**: [Open-Meteo.com](https://open-meteo.com/) (free, no API key)
- **Air Quality Data**: [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api)
- **NASA Images**: [NASA Astronomy Picture of the Day](https://apod.nasa.gov/)
- **LLM**: [GitHub Models](https://github.blog/2024-07-25-introducing-github-models/) (free tier)
- **Fallback LLM**: [Ollama](https://ollama.ai/) with llama3.2

## 🎯 Quality Checklist

- [x] LLM-first natural language understanding (no hardcoded intent matching)
- [x] Automatic LLM fallback (GitHub Models → Ollama)
- [x] Input validation (lat/lon bounds, hours 1-384)
- [x] Historical data support (up to 92 days past)
- [x] Future forecasts (up to 16 days)
- [x] Conversation context for follow-up queries
- [x] Error handling with retry/backoff
- [x] 10-minute caching for API responses
- [x] Off-topic query handling with polite redirects
- [x] Type hints throughout
- [x] Unit tests
- [x] Structured logging
- [x] OpenAPI documentation with Copilot Studio-ready descriptions
- [x] Attribution included in responses

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

Built with ❤️ for the AI Adoption Specialist role assessment.
