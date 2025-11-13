#!/bin/bash

# Navigate to the project root
cd "$(dirname "$0")"

echo "Starting FastAPI server from project root..."
echo "Server will be available at: http://localhost:8000"
echo "Docs available at: http://localhost:8000/docs"
echo ""

# Option 1: Run from project root (recommended)
python -m uvicorn api.controller:app --reload --host 0.0.0.0 --port 8000

# Option 2: If the above doesn't work, uncomment this:
# cd api && uvicorn controller:app --reload --host 0.0.0.0 --port 8000

