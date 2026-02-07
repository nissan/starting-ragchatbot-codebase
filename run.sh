#!/bin/bash

# Create necessary directories
mkdir -p docs 

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "Error: backend directory not found"
    exit 1
fi

echo "Starting Course Materials RAG System..."
echo "If ANTHROPIC_API_KEY is set in .env, Claude will be used."
echo "Otherwise, falling back to local Ollama (llama3.2:3b)."

# Change to backend directory and start the server
cd backend && uv run uvicorn app:app --reload --port 8000