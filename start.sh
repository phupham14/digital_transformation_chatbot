#!/bin/bash
set -e

echo "Starting Rasa Action Server..."
rasa run actions --port 5055 &

echo "Starting Rasa Server..."
rasa run \
  --enable-api \
  --cors "*" \
  --port ${PORT:-10000} \
  --model models