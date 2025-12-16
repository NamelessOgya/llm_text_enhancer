#!/bin/bash
# build.sh

IMAGE_NAME="llm_text_enhancer"

echo "Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME .
