#!/bin/bash
# Script pour démarrer Qdrant sans Docker

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"
./qdrant --config-path "$PROJECT_ROOT/config/qdrant_config.yaml"

