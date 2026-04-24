#!/bin/bash
# start.sh — Start the AI/ML Job Market Pipeline
# Run this script to start PostgreSQL + API server

PROJECT_DIR="/home/safdarayub/Desktop/software house projects_3/ai-ml-job-market-pipeline"
UVICORN="$PROJECT_DIR/venv/bin/uvicorn"

cd "$PROJECT_DIR"

# Step 1: Ensure Docker engine is running
echo "[1/3] Checking Docker..."
if ! sudo systemctl is-active --quiet docker; then
    echo "    Starting Docker engine..."
    sudo systemctl start docker
    sleep 2
fi
docker context use default > /dev/null 2>&1
echo "    Docker OK."

# Step 2: Start PostgreSQL container
echo "[2/3] Starting PostgreSQL..."
docker compose up -d db
sleep 3
echo "    PostgreSQL OK."

# Step 3: Start API server
echo "[3/3] Starting API server on http://localhost:8000"
echo "    Press Ctrl+C to stop."
echo ""
"$UVICORN" api.main:app --port 8000
