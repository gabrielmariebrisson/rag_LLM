#!/bin/bash
# Script pour démarrer Qdrant sans Docker

cd "$(dirname "$0")"
./qdrant --config-path ./qdrant_config.yaml

