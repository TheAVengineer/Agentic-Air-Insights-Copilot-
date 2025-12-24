# 🏗️ Air & Insights Agent - Design Document

## Overview

This document explains the architecture decisions, design patterns, and implementation details of the Air & Insights Agent.

## Design Philosophy

### 1. Policy-Driven Architecture

**Why**: In enterprise AI, controllability > accuracy. Business rules should be:
- Auditable: Changes tracked and explainable
- Configurable: No code changes for threshold updates
- Transparent: Clear decision paths

**Implementation**: All thresholds and rules in `policies/safety_rules.json`:
```json
{
  "air_quality_thresholds": {
    "pm25": {"safe": 25, "moderate": 50, ...}
  }
}
```

### 2. Agentic Flow

The agent follows a structured decision flow:

```
┌─────────────┐
│   Request   │
└──────┬──────┘
       ▼
┌─────────────┐
│   Validate  │ ← Policy-driven validation
└──────┬──────┘
       ▼
┌─────────────┐
│ Cache Check │ ← 10-min TTL
└──────┬──────┘
       │
   ┌───┴───┐
   │ Hit?  │
   └───┬───┘
       │ No
       ▼
┌─────────────┐
│  Fetch APIs │ ← Parallel calls, retry/backoff
└──────┬──────┘
       ▼
┌─────────────┐
│  Validate   │ ← Data quality check
│    Data     │
└──────┬──────┘
       ▼
┌─────────────┐
│ LLM Reason  │ ← GitHub Models
└──────┬──────┘
       ▼
┌─────────────┐
│ Cache Store │
└──────┬──────┘
       ▼
┌─────────────┐
│  Response   │
└─────────────┘
```

### 3. Graceful Degradation

The system continues operating even when components fail:

| Failure | Fallback |
|---------|----------|
| LLM unavailable | Policy-based static guidance |
| Cache miss | Fresh API call |
| Partial data | Calculate with available data + warning |
| API timeout | Retry with exponential backoff |

## Component Design

### Agent Orchestrator (`agent/orchestrator.py`)

The brain of the system. Coordinates:
1. Input validation
2. Cache management
3. API orchestration
4. LLM reasoning
5. Response assembly

**Key Methods:**
- `analyze()`: Main entry point for air quality analysis
- `get_apod()`: NASA APOD with summarization
- `_determine_safety_level()`: Policy-based safety classification

### Agent Planner (`agent/planner.py`)

Creates execution plans for requests. Not fully utilized in current implementation but designed for:
- Complex multi-step workflows
- Parallel execution tracking
- Dependency management

### Memory/Cache (`agent/memory.py`)

TTL-based in-memory cache:
- Key generation from parameters
- Automatic expiration
- Hit/miss statistics

**Cache Key Format**: `{prefix}_{sorted_params}`

### Tools (`tools/`)

External API clients with:
- Async HTTP (httpx)
- Exponential backoff retry
- Data models for responses

### LLM Client (`llm/client.py`)

GitHub Models integration:
- OpenAI-compatible SDK
- Async operations
- Fallback support

### Prompts (`llm/prompts.py`)

Structured prompt templates:
- System prompts define persona and rules
- User templates accept data injection
- Designed for consistent, controllable output

## API Design

### OpenAPI Compatibility

Designed for Copilot Studio import:
- Clear operation descriptions
- Example values
- Proper error responses

### Endpoint Structure

```
/analyze      POST  - Main analysis endpoint
/apod/today   GET   - NASA APOD
/chat         POST  - Natural language interface
/health       GET   - Health check
/cache/stats  GET   - Monitoring
```

## Data Models

All models use Pydantic for:
- Type validation
- Automatic documentation
- JSON serialization

### Key Models

- `AnalyzeRequest/Response`: Air quality analysis
- `APODResponse`: NASA APOD data
- `SafetyLevel`: Enum for safety classification
- `DataQuality`: Enum for data quality assessment

## Error Handling

### Validation Errors (400)
- Invalid coordinates
- Hours out of range
- Missing required fields

### Internal Errors (500)
- API failures after retries
- LLM unavailable
- Unexpected exceptions

### Error Response Format
```json
{
    "error": "ERROR_CODE",
    "message": "Human-readable message",
    "details": { /* optional */ }
}
```

## Performance Considerations

### Target: p95 < 2s (cached)

Optimizations:
1. **Caching**: 10-minute TTL reduces API calls
2. **Parallel Fetching**: Air quality + weather fetched concurrently
3. **Async I/O**: All external calls are non-blocking

### Bottlenecks

1. **LLM Latency**: ~500-1500ms (uncached)
2. **External APIs**: ~200-500ms each
3. **First Request**: Cold start includes all API calls

## Security Considerations

1. **API Keys**: Environment variables only, never in code
2. **Input Validation**: Strict bounds checking
3. **Rate Limiting**: Rely on external API limits
4. **CORS**: Configured for web UI (restrict in production)

## Scalability Story

While built for the air quality use case, the architecture supports:

> "Weather is a template. Tomorrow the same system can reason about supplier delays, data quality issues, or operational risks."

Extension points:
1. Add new tools in `tools/`
2. Add new prompts in `llm/prompts.py`
3. Add new policies in `policies/`
4. New endpoints in `api/routes.py`

## Testing Strategy

### Unit Tests
- Validation logic
- Average calculations
- Cache operations

### Integration Tests
- API endpoints
- Request/response contracts
- Error handling

### Manual Testing
- Demo prompts
- Edge cases
- LLM quality

## Future Enhancements

1. **Location Geocoding**: Convert city names to coordinates
2. **Historical Data**: Track air quality over time
3. **Alerts**: Notify when conditions change
4. **Multi-location**: Compare multiple locations
5. **Semantic Kernel**: Full agent framework integration

## Conclusion

This architecture balances simplicity with extensibility. It's not an enterprise platform, but a focused demonstration of:
- Agentic AI patterns
- Policy-driven design
- Clean code practices
- Production-ready features

The same patterns can scale to more complex domains while maintaining controllability and auditability.
