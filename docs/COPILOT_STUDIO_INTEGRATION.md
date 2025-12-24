# Copilot Studio Integration Guide

This guide explains how to integrate the Air & Insights Agent with **Microsoft Copilot Studio** using the REST API connector.

## Prerequisites

1. **API Running**: Ensure the Air & Insights Agent is deployed and accessible
2. **Copilot Studio Access**: Microsoft 365 license with Copilot Studio access
3. **OpenAPI Spec**: Available at `http://your-api-url/openapi.json`

## Step 1: Export OpenAPI Specification

The API automatically generates an OpenAPI 3.0 specification:

```bash
# Get the OpenAPI spec
curl http://localhost:8081/openapi.json > air-insights-openapi.json
```

Or visit `http://localhost:8081/docs` to see the interactive documentation.

## Step 2: Create Agent in Copilot Studio

1. Go to [Copilot Studio](https://copilotstudio.microsoft.com/)
2. Click **Create** → **New Agent**
3. Name it: "Air & Weather Insights Assistant"
4. Description: "An assistant that provides weather forecasts, air quality analysis, and NASA's Astronomy Picture of the Day"

## Step 3: Add REST API Tool

1. In your agent, go to **Actions** → **Add an action**
2. Select **REST API (Custom connector)**
3. Choose **Import from OpenAPI**
4. Upload the `air-insights-openapi.json` file

### Tool Descriptions for Generative Orchestration

The API includes clear descriptions that help Copilot Studio's **Generative Orchestration** automatically select the right tool:

| Endpoint | When to Use |
|----------|-------------|
| `POST /chat` | Any natural language query about weather, air quality, or NASA APOD |
| `POST /analyze` | Direct air quality analysis with coordinates |
| `GET /apod/today` | NASA Astronomy Picture of the Day |

### Recommended: Use the `/chat` Endpoint

For the best user experience, configure the `/chat` endpoint as the primary tool. It handles:

- **Weather queries**: "What's the weather in London?"
- **Air quality**: "Is it safe to run in Tokyo?"
- **NASA APOD**: "Show me the astronomy picture"
- **Follow-ups**: "What about tomorrow?" or "How about Paris?"
- **Historical data**: "How was the weather yesterday?"

Example request:
```json
{
  "message": "Is it safe to exercise outside in Sofia today?"
}
```

Example response:
```json
{
  "message": "### Air Quality in Sofia\n\n✅ **Great conditions for outdoor exercise!**\n\n- PM2.5: 12.3 µg/m³ (Good)\n- PM10: 18.5 µg/m³ (Good)\n- Temperature: 15.2°C\n\nEnjoy your workout!",
  "intent": "analyze",
  "location": "Sofia",
  "data_source": "open-meteo"
}
```

## Step 4: Configure Tool Descriptions

In Copilot Studio, edit the tool description to help Generative Orchestration:

```
Use this tool when the user asks about:
- Weather forecasts (current, today, tomorrow, next week)
- Air quality and pollution levels
- Whether it's safe to exercise or run outside
- NASA Astronomy Picture of the Day
- Historical weather (yesterday, last week)

This tool understands natural language and supports conversation context.
```

## Step 5: Test the Integration

1. Open the **Test** panel in Copilot Studio
2. Try these queries:
   - "What's the weather in New York for the next 3 days?"
   - "Is the air quality good for jogging in Berlin?"
   - "Show me today's NASA picture"
   - "What about Tokyo?" (follow-up)

## Step 6: Publish

1. Review the agent settings
2. Click **Publish**
3. Share the agent link or embed in Teams

---

## Optional: Teams Message Extension

To run as a **Copilot plugin in Teams**:

1. In Copilot Studio, go to **Channels** → **Microsoft Teams**
2. Enable the Teams channel
3. Click **Open in Teams**
4. The agent will appear as a message extension

### Plugin Manifest

For advanced Teams integration, create a manifest:

```json
{
  "$schema": "https://developer.microsoft.com/json-schemas/teams/v1.16/MicrosoftTeams.schema.json",
  "manifestVersion": "1.16",
  "version": "1.0.0",
  "id": "air-insights-agent",
  "name": {
    "short": "Air & Weather Insights",
    "full": "Air & Weather Insights Copilot"
  },
  "description": {
    "short": "Weather, air quality, and NASA APOD",
    "full": "Get weather forecasts, air quality analysis for outdoor exercise, and NASA's Astronomy Picture of the Day"
  },
  "composeExtensions": [
    {
      "botId": "YOUR_BOT_ID",
      "commands": [
        {
          "id": "weather",
          "type": "query",
          "title": "Weather",
          "description": "Get weather forecast for a location",
          "parameters": [
            {
              "name": "location",
              "title": "Location",
              "description": "City name or coordinates"
            }
          ]
        }
      ]
    }
  ]
}
```

---

## API Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Natural language interface (recommended) |
| POST | `/analyze` | Air quality analysis with coordinates |
| GET | `/apod/today` | NASA Astronomy Picture of the Day |
| GET | `/health` | Health check |
| GET | `/status/llm` | LLM provider status |

## Screenshots

After integration, take screenshots of:
1. Agent configuration in Copilot Studio
2. Tool/Action import dialog
3. Test conversation showing the agent responding
4. (Optional) Teams plugin invocation

Store screenshots in `/docs/screenshots/copilot-studio/`
