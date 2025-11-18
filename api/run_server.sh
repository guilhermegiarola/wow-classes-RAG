#!/bin/bash

# Script to run the backend server from within the api folder
# This script runs the FastAPI backend server

echo "Starting WoW Classes RAG Backend Server..."
echo "Backend will be available at http://localhost:8000"
echo ""

# Navigate to the project root (parent directory)
cd "$(dirname "$0")/.."

# Run uvicorn from the project root, pointing to the api module
python -m uvicorn api.controller:app --reload --host 0.0.0.0 --port 8000
