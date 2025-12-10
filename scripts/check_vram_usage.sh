#!/bin/bash
# Script pour identifier précisément ce qui utilise la VRAM

echo "🔍 Analyse détaillée de l'utilisation VRAM"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# 1. Informations GPU
echo "📊 État des GPUs:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | while IFS=',' read -r idx name mem_used mem_total util; do
    idx=$(echo "$idx" | tr -d ' ')
    name=$(echo "$name" | tr -d ' ')
    mem_used=$(echo "$mem_used" | tr -d ' MiB' | tr -d ' ')
    mem_total=$(echo "$mem_total" | tr -d ' MiB' | tr -d ' ')
    util=$(echo "$util" | tr -d ' %' | tr -d ' ')
    
    if [ -n "$mem_used" ] && [ "$mem_used" != "N/A" ]; then
        mem_used_gb=$(awk "BEGIN {printf \"%.2f\", $mem_used / 1024}")
        mem_total_gb=$(awk "BEGIN {printf \"%.2f\", $mem_total / 1024}")
        mem_percent=$(awk "BEGIN {printf \"%.1f\", ($mem_used * 100) / $mem_total}")
        
        echo ""
        echo "🎮 GPU $idx: $name"
        echo "   💾 VRAM: ${mem_used_gb} GB / ${mem_total_gb} GB (${mem_percent}%)"
        echo "   ⚡ Utilisation: ${util}%"
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Processus utilisant la VRAM:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 2. Processus détectés par nvidia-smi
compute_apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)

if [ -n "$compute_apps" ] && [ "$compute_apps" != "" ]; then
    echo "$compute_apps" | while IFS=',' read -r line; do
        pid=$(echo "$line" | cut -d',' -f1 | tr -d ' ')
        proc_name=$(echo "$line" | cut -d',' -f2 | tr -d ' ')
        mem=$(echo "$line" | cut -d',' -f3 | tr -d ' MiB' | tr -d ' ')
        
        if [ -n "$pid" ] && [ "$pid" != "pid" ] && [ "$pid" != "" ]; then
            mem_gb=$(awk "BEGIN {printf \"%.2f\", $mem / 1024}")
            
            echo ""
            echo "🔹 PID: $pid"
            echo "   Nom: $proc_name"
            echo "   VRAM: ${mem} MB (${mem_gb} GB)"
            
            # Informations détaillées sur le processus
            if ps -p $pid > /dev/null 2>&1; then
                cmd=$(ps -p $pid -o cmd= 2>/dev/null | head -c 150)
                ram_kb=$(ps -p $pid -o rss= 2>/dev/null | awk '{print $1}')
                ram_mb=$((ram_kb / 1024))
                ram_gb=$(awk "BEGIN {printf \"%.2f\", $ram_mb / 1024}")
                
                echo "   RAM: ${ram_mb} MB (${ram_gb} GB)"
                echo "   Commande: ${cmd}..."
                
                # Vérifier si c'est vLLM
                if echo "$cmd" | grep -q "vllm"; then
                    echo "   ✅ Identifié: vLLM (LLM Service)"
                    echo "   📝 Modèle: $(echo "$cmd" | grep -oP '--model \K[^\s]+' || echo 'N/A')"
                    echo "   💡 vLLM alloue beaucoup de VRAM pour le cache KV (Key-Value cache)"
                fi
                
                # Vérifier si c'est Python avec sentence-transformers
                if echo "$cmd" | grep -q "uvicorn\|python.*app.main"; then
                    echo "   ✅ Identifié: Backend FastAPI (Embeddings)"
                fi
            else
                echo "   ⚠️  Processus terminé (VRAM peut être libérée)"
            fi
        fi
    done
else
    echo "❌ Aucun processus détecté par nvidia-smi compute-apps"
    echo ""
    echo "💡 Cela peut signifier:"
    echo "   - Les processus utilisent la VRAM via d'autres mécanismes"
    echo "   - La VRAM est allouée mais pas encore utilisée activement"
    echo "   - vLLM utilise la VRAM via son propre gestionnaire mémoire"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Recherche de processus Python/GPU:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 3. Chercher tous les processus Python qui pourraient utiliser GPU
ps aux | grep -E "python.*vllm|python.*uvicorn|python.*app.main" | grep -v grep | while read line; do
    pid=$(echo "$line" | awk '{print $2}')
    cmd=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
    
    # Vérifier si le processus utilise CUDA
    if [ -d "/proc/$pid" ]; then
        # Vérifier les fichiers ouverts liés à CUDA
        if ls -la /proc/$pid/fd/ 2>/dev/null | grep -q "nvidia"; then
            ram_kb=$(echo "$line" | awk '{print $6}')
            ram_mb=$((ram_kb / 1024))
            ram_gb=$(awk "BEGIN {printf \"%.2f\", $ram_mb / 1024}")
            
            echo ""
            echo "🔹 PID: $pid"
            echo "   RAM: ${ram_mb} MB (${ram_gb} GB)"
            echo "   Commande: $(echo "$cmd" | head -c 100)..."
            
            if echo "$cmd" | grep -q "vllm"; then
                echo "   ✅ vLLM détecté (utilise probablement la VRAM)"
            fi
        fi
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💡 Explication de l'utilisation VRAM:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Pour vLLM avec Llama-3-8B quantifié AWQ:"
echo "  - Modèle quantifié: ~4-5 GB"
echo "  - Cache KV (Key-Value): ~15-18 GB (pour max_model_len=4096)"
echo "  - Total: ~20-23 GB"
echo ""
echo "Le cache KV est alloué au démarrage pour optimiser les performances."
echo "C'est normal et attendu pour vLLM avec --gpu-memory-utilization 0.95"
echo ""
echo "═══════════════════════════════════════════════════════════════"
