#!/bin/bash
# Script de rapport détaillé sur les processus et leur utilisation mémoire/GPU

echo "═══════════════════════════════════════════════════════════════"
echo "📊 RAPPORT DÉTAILLÉ DES PROCESSUS RAG"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 1. PROCESSUS vLLM
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  PROCESSUS vLLM (LLM Service)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
vllm_pids=$(pgrep -f "vllm.entrypoints.openai.api_server" || echo "")
if [ -n "$vllm_pids" ]; then
    for pid in $vllm_pids; do
        if ps -p $pid > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Processus actif (PID: $pid)${NC}"
            
            # Mémoire RAM
            ram_kb=$(ps -p $pid -o rss= 2>/dev/null | awk '{print $1}')
            if [ -n "$ram_kb" ]; then
                ram_mb=$((ram_kb / 1024))
                ram_gb=$(awk "BEGIN {printf \"%.2f\", $ram_mb / 1024}")
                echo "   📦 Mémoire RAM: ${ram_mb} MB (${ram_gb} GB)"
            fi
            
            # GPU utilisé
            if command -v nvidia-smi > /dev/null 2>&1; then
                gpu_info=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | grep "^$pid," || echo "")
                if [ -n "$gpu_info" ]; then
                    gpu_id=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -n "^$pid$" | cut -d: -f1)
                    gpu_id=$((gpu_id - 1))
                    vram_mb=$(echo "$gpu_info" | cut -d',' -f3 | tr -d ' ')
                    vram_gb=$(awk "BEGIN {printf \"%.2f\", $vram_mb / 1024}")
                    echo -e "   🎮 GPU: ${BLUE}GPU $gpu_id${NC} (VRAM: ${vram_mb} MB / ${vram_gb} GB)"
                else
                    echo "   🎮 GPU: Non détecté dans nvidia-smi"
                fi
            fi
            
            # Port
            port=$(netstat -tlnp 2>/dev/null | grep "$pid" | grep LISTEN | awk '{print $4}' | cut -d: -f2 | head -1)
            if [ -n "$port" ]; then
                echo "   🌐 Port: $port"
            fi
            
            # Command line
            cmd=$(ps -p $pid -o cmd= 2>/dev/null | head -c 100)
            echo "   📝 Commande: ${cmd}..."
        fi
    done
else
    echo -e "${RED}❌ Aucun processus vLLM trouvé${NC}"
fi
echo ""

# 2. PROCESSUS BACKEND (FastAPI + Embeddings)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  PROCESSUS BACKEND (FastAPI + Embeddings + Reranker)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
backend_pids=$(pgrep -f "uvicorn app.main:app" || echo "")
if [ -n "$backend_pids" ]; then
    for pid in $backend_pids; do
        if ps -p $pid > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Processus actif (PID: $pid)${NC}"
            
            # Mémoire RAM
            ram_kb=$(ps -p $pid -o rss= 2>/dev/null | awk '{print $1}')
            if [ -n "$ram_kb" ]; then
                ram_mb=$((ram_kb / 1024))
                ram_gb=$(awk "BEGIN {printf \"%.2f\", $ram_mb / 1024}")
                echo "   📦 Mémoire RAM: ${ram_mb} MB (${ram_gb} GB)"
            fi
            
            # GPU utilisé (pour les modèles d'embeddings)
            if command -v nvidia-smi > /dev/null 2>&1; then
                gpu_info=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | grep "^$pid," || echo "")
                if [ -n "$gpu_info" ]; then
                    gpu_id=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -n "^$pid$" | cut -d: -f1)
                    gpu_id=$((gpu_id - 1))
                    vram_mb=$(echo "$gpu_info" | cut -d',' -f3 | tr -d ' ')
                    vram_gb=$(awk "BEGIN {printf \"%.2f\", $vram_mb / 1024}")
                    echo -e "   🎮 GPU: ${BLUE}GPU $gpu_id${NC} (VRAM: ${vram_mb} MB / ${vram_gb} GB)"
                else
                    # Vérifier les processus Python qui utilisent GPU
                    python_pids=$(pgrep -P $pid || echo "")
                    total_vram=0
                    gpus_used=""
                    for py_pid in $python_pids; do
                        py_gpu_info=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | grep "^$py_pid," || echo "")
                        if [ -n "$py_gpu_info" ]; then
                            py_gpu_id=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -n "^$py_pid$" | cut -d: -f1)
                            py_gpu_id=$((py_gpu_id - 1))
                            py_vram=$(echo "$py_gpu_info" | cut -d',' -f2 | tr -d ' ')
                            total_vram=$((total_vram + py_vram))
                            if [ -z "$gpus_used" ]; then
                                gpus_used="GPU $py_gpu_id"
                            else
                                gpus_used="$gpus_used, GPU $py_gpu_id"
                            fi
                        fi
                    done
                    if [ $total_vram -gt 0 ]; then
                        vram_gb=$(echo "scale=2; $total_vram / 1024" | bc)
                        echo -e "   🎮 GPU: ${BLUE}$gpus_used${NC} (VRAM total: ${total_vram} MB / ${vram_gb} GB)"
                    else
                        echo "   🎮 GPU: Modèles pas encore chargés ou sur CPU"
                    fi
                fi
            fi
            
            # Port
            port=$(netstat -tlnp 2>/dev/null | grep "$pid" | grep LISTEN | awk '{print $4}' | cut -d: -f2 | head -1)
            if [ -n "$port" ]; then
                echo "   🌐 Port: $port"
            fi
        fi
    done
    
    # Vérifier les modèles chargés via Python
    echo ""
    echo "   📚 Modèles chargés:"
    if [ -f /workspace/rag_LLM/.env ]; then
        cuda_device=$(grep "^CUDA_DEVICE_ID" /workspace/rag_LLM/.env | cut -d'=' -f2 | tr -d ' ')
        if [ -n "$cuda_device" ]; then
            echo -e "      - Device configuré: ${BLUE}GPU $cuda_device${NC}"
        else
            echo "      - Device configuré: GPU 0 (défaut)"
        fi
    fi
else
    echo -e "${RED}❌ Aucun processus Backend trouvé${NC}"
fi
echo ""

# 3. PROCESSUS QDRANT
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣  PROCESSUS QDRANT (Vector Database)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
qdrant_pids=$(pgrep -f "qdrant" || echo "")
if [ -n "$qdrant_pids" ]; then
    for pid in $qdrant_pids; do
        if ps -p $pid > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Processus actif (PID: $pid)${NC}"
            
            # Mémoire RAM
            ram_kb=$(ps -p $pid -o rss= 2>/dev/null | awk '{print $1}')
            if [ -n "$ram_kb" ]; then
                ram_mb=$((ram_kb / 1024))
                ram_gb=$(awk "BEGIN {printf \"%.2f\", $ram_mb / 1024}")
                echo "   📦 Mémoire RAM: ${ram_mb} MB (${ram_gb} GB)"
            fi
            
            # GPU (Qdrant n'utilise généralement pas GPU)
            echo "   🎮 GPU: Non utilisé (Qdrant utilise CPU)"
            
            # Port
            port=$(netstat -tlnp 2>/dev/null | grep "$pid" | grep LISTEN | awk '{print $4}' | cut -d: -f2 | head -1)
            if [ -n "$port" ]; then
                echo "   🌐 Port: $port"
            fi
        fi
    done
else
    echo -e "${RED}❌ Aucun processus Qdrant trouvé${NC}"
fi
echo ""

# 4. PROCESSUS FRONTEND (Streamlit)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4️⃣  PROCESSUS FRONTEND (Streamlit)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
frontend_pids=$(pgrep -f "streamlit run" || echo "")
if [ -n "$frontend_pids" ]; then
    for pid in $frontend_pids; do
        if ps -p $pid > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Processus actif (PID: $pid)${NC}"
            
            # Mémoire RAM
            ram_kb=$(ps -p $pid -o rss= 2>/dev/null | awk '{print $1}')
            if [ -n "$ram_kb" ]; then
                ram_mb=$((ram_kb / 1024))
                ram_gb=$(awk "BEGIN {printf \"%.2f\", $ram_mb / 1024}")
                echo "   📦 Mémoire RAM: ${ram_mb} MB (${ram_gb} GB)"
            fi
            
            # GPU (Streamlit n'utilise pas GPU)
            echo "   🎮 GPU: Non utilisé (Frontend utilise CPU)"
            
            # Port
            port=$(netstat -tlnp 2>/dev/null | grep "$pid" | grep LISTEN | awk '{print $4}' | cut -d: -f2 | head -1)
            if [ -n "$port" ]; then
                echo "   🌐 Port: $port"
            fi
        fi
    done
else
    echo -e "${RED}❌ Aucun processus Frontend trouvé${NC}"
fi
echo ""

# 5. RÉSUMÉ GPU
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 RÉSUMÉ UTILISATION GPU"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if command -v nvidia-smi > /dev/null 2>&1; then
    echo ""
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | while IFS=',' read -r line; do
        # Parser la ligne CSV
        idx=$(echo "$line" | cut -d',' -f1 | tr -d ' ')
        name=$(echo "$line" | cut -d',' -f2 | tr -d ' ')
        mem_used=$(echo "$line" | cut -d',' -f3 | tr -d ' MiB' | tr -d ' ')
        mem_total=$(echo "$line" | cut -d',' -f4 | tr -d ' MiB' | tr -d ' ')
        util=$(echo "$line" | cut -d',' -f5 | tr -d ' %' | tr -d ' ')
        
        if [ -n "$mem_used" ] && [ -n "$mem_total" ] && [ "$mem_used" != "N/A" ]; then
            mem_percent=$(awk "BEGIN {printf \"%.1f\", ($mem_used * 100) / $mem_total}")
            mem_used_gb=$(awk "BEGIN {printf \"%.2f\", $mem_used / 1024}")
            mem_total_gb=$(awk "BEGIN {printf \"%.2f\", $mem_total / 1024}")
            
            echo "🎮 GPU $idx: $name"
            echo "   💾 VRAM: ${mem_used_gb} GB / ${mem_total_gb} GB (${mem_percent}%)"
            echo "   ⚡ Utilisation: ${util}%"
            
            # Identifier les processus sur ce GPU
            gpu_processes=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep -v "^$" || echo "")
            if [ -n "$gpu_processes" ]; then
                echo "   📋 Processus sur GPU $idx:"
                echo "$gpu_processes" | while IFS=',' read -r proc_line; do
                    pid=$(echo "$proc_line" | cut -d',' -f1 | tr -d ' ')
                    proc_name=$(echo "$proc_line" | cut -d',' -f2 | tr -d ' ')
                    mem=$(echo "$proc_line" | cut -d',' -f3 | tr -d ' MiB' | tr -d ' ')
                    if [ -n "$pid" ] && [ "$pid" != "pid" ]; then
                        proc_mem_gb=$(awk "BEGIN {printf \"%.2f\", $mem / 1024}")
                        ps_cmd=$(ps -p $pid -o cmd= 2>/dev/null | head -c 50 || echo "N/A")
                        echo "      - PID $pid ($proc_name): ${mem} MB (${proc_mem_gb} GB)"
                        echo "        Commande: ${ps_cmd}..."
                    fi
                done
            else
                echo "   📋 Aucun processus détecté"
            fi
            echo ""
        fi
    done
else
    echo -e "${YELLOW}⚠️  nvidia-smi non disponible${NC}"
fi

# 6. RÉSUMÉ MÉMOIRE RAM TOTALE
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 RÉSUMÉ MÉMOIRE RAM TOTALE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
total_ram=0
for pid in $vllm_pids $backend_pids $qdrant_pids $frontend_pids; do
    if ps -p $pid > /dev/null 2>&1; then
        ram_kb=$(ps -p $pid -o rss= 2>/dev/null | awk '{print $1}')
        if [ -n "$ram_kb" ]; then
            total_ram=$((total_ram + ram_kb))
        fi
    fi
done
total_ram_mb=$((total_ram / 1024))
total_ram_gb=$(awk "BEGIN {printf \"%.2f\", $total_ram_mb / 1024}")
echo "💾 RAM totale utilisée par les 4 processus: ${total_ram_mb} MB (${total_ram_gb} GB)"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "✅ Rapport terminé"
echo "═══════════════════════════════════════════════════════════════"
