#!/bin/bash
# Script de vérification des services et configuration GPU

echo "🔍 Vérification des services et configuration"
echo "=========================================="
echo ""

# 1. Vérification Qdrant
echo "📊 Qdrant:"
if curl -s http://localhost:6333/health > /dev/null 2>&1; then
    echo "  ✅ Qdrant est accessible sur http://localhost:6333"
    collections=$(curl -s http://localhost:6333/collections 2>/dev/null | grep -o '"name":"[^"]*"' | wc -l)
    echo "  📦 Collections: $collections"
else
    echo "  ❌ Qdrant n'est pas accessible"
fi
echo ""

# 2. Vérification vLLM
echo "📊 vLLM:"
if curl -s http://localhost:8002/health > /dev/null 2>&1; then
    echo "  ✅ vLLM est accessible sur http://localhost:8002"
elif curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo "  ⚠️  vLLM est accessible sur http://localhost:8001 (mais config attend 8002)"
else
    echo "  ❌ vLLM n'est pas accessible"
fi
echo ""

# 3. Vérification Backend
echo "📊 Backend:"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "  ✅ Backend est accessible sur http://localhost:8000"
else
    echo "  ❌ Backend n'est pas accessible"
fi
echo ""

# 4. Vérification Frontend
echo "📊 Frontend:"
if curl -s http://localhost:8501 > /dev/null 2>&1; then
    echo "  ✅ Frontend est accessible sur http://localhost:8501"
else
    echo "  ❌ Frontend n'est pas accessible"
fi
echo ""

# 5. Configuration GPU
echo "📊 Configuration GPU:"
if [ -f .env ]; then
    cuda_device=$(grep "^CUDA_DEVICE_ID" .env | cut -d'=' -f2 | tr -d ' ')
    if [ -n "$cuda_device" ]; then
        echo "  ✅ CUDA_DEVICE_ID=$cuda_device (dans .env)"
    else
        echo "  ⚠️  CUDA_DEVICE_ID non configuré (utilisera GPU 0 par défaut)"
    fi
    
    llm_url=$(grep "^LLM_BASE_URL" .env | cut -d'=' -f2 | tr -d ' ')
    if [ -n "$llm_url" ]; then
        echo "  ✅ LLM_BASE_URL=$llm_url"
    fi
else
    echo "  ⚠️  Fichier .env non trouvé"
fi
echo ""

# 6. Vérification utilisation GPU
echo "📊 Utilisation GPU:"
if command -v nvidia-smi > /dev/null 2>&1; then
    echo "  GPU 0:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' 'NR==1 {printf "    - Mémoire: %s/%s MB (%d%%)\n    - Utilisation: %s%%\n", $3, $4, ($3/$4)*100, $5}'
    if [ $(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l) -gt 1 ]; then
        echo "  GPU 1:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' 'NR==2 {printf "    - Mémoire: %s/%s MB (%d%%)\n    - Utilisation: %s%%\n", $3, $4, ($3/$4)*100, $5}'
    fi
else
    echo "  ⚠️  nvidia-smi non disponible"
fi
echo ""

echo "=========================================="
echo "✅ Vérification terminée"
