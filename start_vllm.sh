#!/bin/bash
# Script pour lancer vLLM localement sans Docker

set -e

# Activer l'environnement virtuel
source venv/bin/activate

# Vérifier si vLLM est installé
if ! python -c "import vllm" 2>/dev/null; then
    echo "Installation de vLLM..."
    pip install vllm
fi

# Variables d'environnement
export CUDA_VISIBLE_DEVICES=0
export HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-""}
export HF_HUB_ENABLE_HF_TRANSFER=0  # Désactiver hf_transfer pour éviter les erreurs

# Paramètres vLLM (identique à docker-compose.yml)
MODEL="casperhansen/llama-3-8b-instruct-awq"
QUANTIZATION="awq"
DTYPE="float16"
MAX_MODEL_LEN="4096"
GPU_MEMORY_UTIL="0.95"
PORT="8002"
HOST="0.0.0.0"

echo "🚀 Démarrage de vLLM..."
echo "Modèle: $MODEL"
echo "Port: $PORT"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Lancer vLLM
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --quantization "$QUANTIZATION" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --enforce-eager \
    --port "$PORT" \
    --host "$HOST"

