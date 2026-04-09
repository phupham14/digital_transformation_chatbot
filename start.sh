#!/bin/bash

echo "Starting dummy server to open port..."
python -m http.server 10000 &

echo "Starting Rasa Action Server..."
rasa run actions --port 5055 &

echo "Starting Rasa Server..."
rasa run \
  --enable-api \
  --cors "*" \
  --port 10000 \
  --model models/20260328-133345-daring-resource.tar.gz