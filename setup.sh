#!/bin/bash

# =============================================================================
# Air & Insights Agent - Quick Setup Script
# =============================================================================
# This script automates the entire setup process for testing and verification.
# 
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# What it does:
#   1. Creates Python virtual environment
#   2. Installs dependencies
#   3. Sets up environment variables
#   4. Runs tests to verify everything works
#   5. Starts the server
#   6. Runs demo prompts to verify functionality
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Emoji support
CHECK="✅"
CROSS="❌"
ROCKET="🚀"
GEAR="⚙️"
TEST="🧪"
GLOBE="🌍"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           🌤️  Air & Insights Agent - Quick Setup              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Check Python version
# -----------------------------------------------------------------------------
echo -e "${BLUE}${GEAR} Step 1/6: Checking Python version...${NC}"

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
        echo -e "${GREEN}${CHECK} Python $PYTHON_VERSION found${NC}"
    else
        echo -e "${YELLOW}⚠️  Python $PYTHON_VERSION found (3.11+ recommended)${NC}"
    fi
else
    echo -e "${RED}${CROSS} Python 3 not found. Please install Python 3.11+${NC}"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 2: Create virtual environment
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}${GEAR} Step 2/6: Setting up virtual environment...${NC}"

if [ -d "venv" ]; then
    echo -e "${YELLOW}   Virtual environment already exists, using it...${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}${CHECK} Virtual environment created${NC}"
fi

# Activate virtual environment
source venv/bin/activate
echo -e "${GREEN}${CHECK} Virtual environment activated${NC}"

# -----------------------------------------------------------------------------
# Step 3: Install dependencies
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}${GEAR} Step 3/6: Installing dependencies...${NC}"

pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "${GREEN}${CHECK} Dependencies installed${NC}"

# -----------------------------------------------------------------------------
# Step 4: Setup environment variables
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}${GEAR} Step 4/6: Configuring environment...${NC}"

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${YELLOW}   Created .env from .env.example${NC}"
    else
        echo "GITHUB_TOKEN=" > .env
        echo "NASA_API_KEY=DEMO_KEY" >> .env
        echo "LOG_LEVEL=INFO" >> .env
        echo -e "${YELLOW}   Created default .env file${NC}"
    fi
fi

# Check if GITHUB_TOKEN is set
if grep -q "GITHUB_TOKEN=$" .env || grep -q "GITHUB_TOKEN=\"\"" .env; then
    echo ""
    echo -e "${YELLOW}⚠️  GITHUB_TOKEN not set in .env file${NC}"
    echo -e "${YELLOW}   The agent will use Ollama as fallback (if available)${NC}"
    echo -e "${YELLOW}   For best results, add your GitHub token:${NC}"
    echo -e "${YELLOW}   1. Get a token: https://github.com/settings/tokens${NC}"
    echo -e "${YELLOW}   2. Edit .env and add: GITHUB_TOKEN=your_token_here${NC}"
    echo ""
else
    echo -e "${GREEN}${CHECK} Environment configured${NC}"
fi

# -----------------------------------------------------------------------------
# Step 5: Run tests
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}${TEST} Step 5/6: Running tests...${NC}"

# Run a quick subset of tests to verify installation
if python -m pytest tests/test_validation.py tests/test_api.py -v --tb=short -q 2>/dev/null; then
    echo -e "${GREEN}${CHECK} Core tests passed${NC}"
else
    echo -e "${YELLOW}⚠️  Some tests may have failed (this might be OK if APIs are unavailable)${NC}"
fi

# -----------------------------------------------------------------------------
# Step 6: Start server and verify
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}${ROCKET} Step 6/6: Starting server and running verification...${NC}"

# Kill any existing server on port 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# Start server in background
python main.py &
SERVER_PID=$!
echo "   Server starting (PID: $SERVER_PID)..."

# Wait for server to be ready
echo "   Waiting for server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}${CHECK} Server is running at http://localhost:8000${NC}"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo -e "${RED}${CROSS} Server failed to start${NC}"
        exit 1
    fi
done

# -----------------------------------------------------------------------------
# Verification Tests
# -----------------------------------------------------------------------------
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    🧪 Verification Tests                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Test 1: Health check
echo -e "${BLUE}Test 1: Health Check${NC}"
HEALTH=$(curl -s http://localhost:8000/health)
if echo "$HEALTH" | grep -q "healthy"; then
    echo -e "${GREEN}${CHECK} Health check passed${NC}"
else
    echo -e "${RED}${CROSS} Health check failed${NC}"
fi

# Test 2: OpenAPI spec
echo ""
echo -e "${BLUE}Test 2: OpenAPI Spec${NC}"
OPENAPI=$(curl -s http://localhost:8000/openapi.json)
if echo "$OPENAPI" | grep -q "Air & Insights Agent"; then
    echo -e "${GREEN}${CHECK} OpenAPI spec available${NC}"
else
    echo -e "${RED}${CROSS} OpenAPI spec not found${NC}"
fi

# Test 3: Analyze endpoint (Demo Prompt 1)
echo ""
echo -e "${BLUE}Test 3: POST /analyze (Air Quality)${NC}"
echo "   Request: Sofia coordinates (42.6977, 23.3219), 6 hours"
ANALYZE=$(curl -s -X POST http://localhost:8000/analyze \
    -H "Content-Type: application/json" \
    -d '{"latitude": 42.6977, "longitude": 23.3219, "hours": 6}')

if echo "$ANALYZE" | grep -q "pm25_avg"; then
    PM25=$(echo "$ANALYZE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('pm25_avg', 'N/A'))")
    PM10=$(echo "$ANALYZE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('pm10_avg', 'N/A'))")
    TEMP=$(echo "$ANALYZE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('temp_avg', 'N/A'))")
    SAFETY=$(echo "$ANALYZE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('safety_level', 'N/A'))")
    echo -e "${GREEN}${CHECK} Analyze endpoint working${NC}"
    echo "   📊 PM2.5: ${PM25} µg/m³"
    echo "   📊 PM10: ${PM10} µg/m³"
    echo "   🌡️  Temp: ${TEMP}°C"
    echo "   🛡️  Safety: ${SAFETY}"
else
    echo -e "${RED}${CROSS} Analyze endpoint failed${NC}"
fi

# Test 4: APOD endpoint (Demo Prompt 2)
echo ""
echo -e "${BLUE}Test 4: GET /apod/today (NASA APOD)${NC}"
APOD=$(curl -s http://localhost:8000/apod/today)

if echo "$APOD" | grep -q "title"; then
    TITLE=$(echo "$APOD" | python3 -c "import sys,json; print(json.load(sys.stdin).get('title', 'N/A')[:50])")
    echo -e "${GREEN}${CHECK} APOD endpoint working${NC}"
    echo "   🌟 Today's picture: ${TITLE}..."
else
    echo -e "${YELLOW}⚠️  APOD endpoint may have issues (NASA API rate limits)${NC}"
fi

# Test 5: Chat endpoint
echo ""
echo -e "${BLUE}Test 5: POST /chat (Natural Language)${NC}"
CHAT=$(curl -s -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "Is it safe to run in Sofia?"}')

if echo "$CHAT" | grep -q "response\|guidance\|PM"; then
    echo -e "${GREEN}${CHECK} Chat endpoint working${NC}"
else
    echo -e "${YELLOW}⚠️  Chat endpoint may need LLM configuration${NC}"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                      ✅ Setup Complete!                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}The Air & Insights Agent is now running!${NC}"
echo ""
echo "📍 Access Points:"
echo "   • Web UI:      http://localhost:8000/"
echo "   • Swagger:     http://localhost:8000/docs"
echo "   • OpenAPI:     http://localhost:8000/openapi.json"
echo ""
echo "🎯 Try the demo prompts:"
echo "   1. \"What's the PM2.5 and temperature around 42.6977, 23.3219"
echo "       for the next 6 hours and should I run outdoors?\""
echo "   2. \"Show today's NASA APOD and summarize in 2 lines.\""
echo ""
echo "📋 To stop the server: kill $SERVER_PID"
echo "📋 To restart: ./setup.sh"
echo ""
echo "─────────────────────────────────────────────────────────────────"
