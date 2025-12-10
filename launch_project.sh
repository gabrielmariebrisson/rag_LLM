#!/bin/bash
# Script pour lancer le projet complet avec vLLM sans Docker

set -e

cd /workspace/rag_LLM

# Activer l'environnement virtuel
source venv/bin/activate

# Désactiver hf_transfer pour éviter les erreurs avec sentence-transformers
export HF_HUB_ENABLE_HF_TRANSFER=0

# Créer le fichier .env s'il n'existe pas
if [ ! -f .env ]; then
    cat > .env << 'EOF'
# Configuration LLM - vLLM Local
LLM_BASE_URL=http://localhost:8002/v1
LLM_MODEL_NAME=casperhansen/llama-3-8b-instruct-awq
LLM_API_KEY=dummy-key
HUGGING_FACE_HUB_TOKEN=

# Configuration Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=squad_collection

# Configuration Reranker
USE_RERANKER=true
RERANKER_TOP_K=10  # Optimal: même recall que 50, 7x plus rapide

# Configuration Embeddings (utilise sentence-transformers pour BGE-M3)
DENSE_MODEL=BAAI/bge-large-en-v1.5
SPARSE_MODEL=prithivida/Splade_PP_en_v1
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
DENSE_DIM=1024

# Configuration GPU (utiliser GPU 1 si GPU 0 est saturé)
CUDA_DEVICE_ID=1

# Backend URL (pour le frontend)
BACKEND_URL=http://localhost:8000
EOF
    echo "✅ Fichier .env créé"
fi

# Fonction pour vérifier si un port est utilisé
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0  # Port utilisé
    else
        return 1  # Port libre
    fi
}

# Vérifier et lancer vLLM
if check_port 8002; then
    echo "⚠️  vLLM semble déjà être en cours d'exécution sur le port 8002"
else
    echo "🚀 Démarrage de vLLM..."
    ./start_vllm.sh > /tmp/vllm.log 2>&1 &
    VLLM_PID=$!
    echo "vLLM démarré (PID: $VLLM_PID)"
    
    # Attendre que vLLM soit prêt
    echo "⏳ Attente du démarrage de vLLM..."
    for i in {1..60}; do
        if curl -s http://localhost:8002/health > /dev/null 2>&1; then
            echo "✅ vLLM est prêt!"
            break
        fi
        if [ $i -eq 60 ]; then
            echo "❌ vLLM n'a pas démarré dans les délais"
            exit 1
        fi
        sleep 3
    done
fi

# Vérifier et lancer le backend
if check_port 8000; then
    echo "⚠️  Le backend semble déjà être en cours d'exécution sur le port 8000"
else
    echo "🚀 Démarrage du backend FastAPI..."
    uvicorn app.main:app --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &
    BACKEND_PID=$!
    echo "Backend démarré (PID: $BACKEND_PID)"
    sleep 3
fi

# Vérifier et lancer le frontend
if check_port 8501; then
    echo "⚠️  Le frontend semble déjà être en cours d'exécution sur le port 8501"
else
    echo "🚀 Démarrage du frontend Streamlit..."
    streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 > /tmp/frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo "Frontend démarré (PID: $FRONTEND_PID)"
    sleep 3
fi

echo ""
echo "=========================================="
echo "✅ Projet lancé avec succès!"
echo "=========================================="
echo "📊 Services disponibles:"
echo "  - vLLM:      http://localhost:8002"
echo "  - Backend:   http://localhost:8000"
echo "  - Frontend:  http://localhost:8501"
echo "  - API Docs:  http://localhost:8000/docs"
echo ""
echo "📝 Logs:"
echo "  - vLLM:    tail -f /tmp/vllm.log"
echo "  - Backend: tail -f /tmp/backend.log"
echo "  - Frontend: tail -f /tmp/frontend.log"
echo ""
echo "Pour arrêter les services, utilisez: pkill -f 'vllm|uvicorn|streamlit'"
echo "=========================================="

