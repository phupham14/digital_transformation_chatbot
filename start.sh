#!/bin/bash

echo "Starting Rasa Action Server..."
rasa run actions --port 5055 &

echo "Starting Rasa Server..."
rasa run \
  --enable-api \
  --cors "*" \
  --port $PORT \
  --model models/20260328-133345-daring-resource.tar.gz