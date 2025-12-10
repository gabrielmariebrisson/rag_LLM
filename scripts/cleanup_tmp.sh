#!/bin/bash
# Script de nettoyage des fichiers temporaires dans /tmp

set -e

TMP_DIR="/tmp"
DRY_RUN=${1:-""}  # Passer "dry" pour un test sans suppression

echo "🧹 Nettoyage des fichiers temporaires dans $TMP_DIR"
echo ""

# Fonction pour supprimer avec vérification
safe_remove() {
    local item=$1
    local description=$2
    
    if [ -e "$TMP_DIR/$item" ]; then
        if [ "$DRY_RUN" = "dry" ]; then
            echo "  [DRY-RUN] Supprimerait: $item ($description)"
        else
            echo "  🗑️  Suppression: $item"
            rm -rf "$TMP_DIR/$item" 2>/dev/null || echo "    ⚠️  Impossible de supprimer $item"
        fi
    fi
}

# 1. Cache fastembed (509MB) - Plus utilisé car on utilise sentence-transformers maintenant
if [ -d "$TMP_DIR/fastembed_cache" ]; then
    size=$(du -sh "$TMP_DIR/fastembed_cache" 2>/dev/null | cut -f1)
    echo "📦 Cache fastembed: $size (plus utilisé)"
    safe_remove "fastembed_cache" "Cache fastembed (remplacé par sentence-transformers)"
fi

# 2. Répertoires temporaires UUIDs (probablement orphelins)
echo ""
echo "🔍 Recherche de répertoires temporaires UUIDs..."
for item in "$TMP_DIR"/*; do
    if [ -d "$item" ]; then
        basename_item=$(basename "$item")
        # Vérifier si c'est un UUID (format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
        if [[ "$basename_item" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]]; then
            size=$(du -sh "$item" 2>/dev/null | cut -f1)
            # Vérifier si le répertoire est utilisé (processus actif)
            if ! lsof +D "$item" 2>/dev/null | grep -q .; then
                echo "  📁 Répertoire UUID orphelin: $basename_item ($size)"
                if [ "$DRY_RUN" != "dry" ]; then
                    rm -rf "$item" 2>/dev/null || echo "    ⚠️  Impossible de supprimer"
                fi
            else
                echo "  ⏸️  Répertoire UUID actif: $basename_item (conservé)"
            fi
        fi
    fi
done

# 3. Fichiers temporaires Python (tmp*.py, tmp*)
echo ""
echo "🔍 Recherche de fichiers temporaires Python..."
for item in "$TMP_DIR"/tmp*; do
    if [ -f "$item" ] || [ -d "$item" ]; then
        basename_item=$(basename "$item")
        # Vérifier si c'est un fichier temporaire standard
        if [[ "$basename_item" =~ ^tmp[a-zA-Z0-9]+$ ]]; then
            size=$(du -sh "$item" 2>/dev/null | cut -f1)
            if ! lsof "$item" 2>/dev/null | grep -q .; then
                echo "  📄 Fichier temporaire: $basename_item ($size)"
                if [ "$DRY_RUN" != "dry" ]; then
                    rm -rf "$item" 2>/dev/null || echo "    ⚠️  Impossible de supprimer"
                fi
            fi
        fi
    fi
done

# 4. Fichiers .lock vides ou orphelins
echo ""
echo "🔍 Recherche de fichiers .lock orphelins..."
for lockfile in "$TMP_DIR"/*.lock; do
    if [ -f "$lockfile" ]; then
        basename_lock=$(basename "$lockfile")
        # Vérifier si le fichier est vide ou si le processus associé n'existe plus
        if [ ! -s "$lockfile" ] || ! lsof "$lockfile" 2>/dev/null | grep -q .; then
            size=$(du -sh "$lockfile" 2>/dev/null | cut -f1)
            echo "  🔒 Lock orphelin: $basename_lock ($size)"
            if [ "$DRY_RUN" != "dry" ]; then
                rm -f "$lockfile" 2>/dev/null || echo "    ⚠️  Impossible de supprimer"
            fi
        fi
    fi
done

# 5. Cache PyTorch vide (torchinductor_root)
if [ -d "$TMP_DIR/torchinductor_root" ]; then
    size=$(du -sh "$TMP_DIR/torchinductor_root" 2>/dev/null | cut -f1)
    if [ "$size" = "0" ] || [ "$size" = "0K" ]; then
        echo ""
        echo "📦 Cache PyTorch vide: torchinductor_root"
        safe_remove "torchinductor_root" "Cache PyTorch vide"
    fi
fi

# 6. Cache TVM (petit mais peut être supprimé)
if [ -d "$TMP_DIR/tvm-ffi-torch-c-dlpack-by2k6gb9" ]; then
    size=$(du -sh "$TMP_DIR/tvm-ffi-torch-c-dlpack-by2k6gb9" 2>/dev/null | cut -f1)
    echo ""
    echo "📦 Cache TVM: tvm-ffi-torch-c-dlpack-by2k6gb9 ($size)"
    if ! lsof +D "$TMP_DIR/tvm-ffi-torch-c-dlpack-by2k6gb9" 2>/dev/null | grep -q .; then
        safe_remove "tvm-ffi-torch-c-dlpack-by2k6gb9" "Cache TVM"
    else
        echo "  ⏸️  Cache TVM actif (conservé)"
    fi
fi

# 7. Fichiers socket IPC orphelins (VSCode/Cursor)
echo ""
echo "🔍 Recherche de sockets IPC orphelins..."
for sock in "$TMP_DIR"/vscode-*.sock "$TMP_DIR"/cursor-*.sock; do
    if [ -S "$sock" ] || [ -f "$sock" ]; then
        basename_sock=$(basename "$sock")
        # Vérifier si le socket est utilisé
        if ! lsof "$sock" 2>/dev/null | grep -q .; then
            echo "  🔌 Socket IPC orphelin: $basename_sock"
            if [ "$DRY_RUN" != "dry" ]; then
                rm -f "$sock" 2>/dev/null || echo "    ⚠️  Impossible de supprimer"
            fi
        fi
    fi
done

# 8. Fichiers .pid et .token orphelins (Cursor)
echo ""
echo "🔍 Recherche de fichiers PID/Token orphelins..."
for pidfile in "$TMP_DIR"/cursor-*.pid "$TMP_DIR"/cursor-*.token; do
    if [ -f "$pidfile" ]; then
        basename_pid=$(basename "$pidfile")
        if [[ "$basename_pid" =~ \.pid$ ]]; then
            # Vérifier si le processus existe
            pid=$(cat "$pidfile" 2>/dev/null || echo "")
            if [ -n "$pid" ] && ! ps -p "$pid" > /dev/null 2>&1; then
                echo "  🆔 PID orphelin: $basename_pid (processus $pid n'existe plus)"
                if [ "$DRY_RUN" != "dry" ]; then
                    rm -f "$pidfile" 2>/dev/null || echo "    ⚠️  Impossible de supprimer"
                fi
            fi
        else
            # Pour les fichiers .token, vérifier si le PID associé existe
            # (les fichiers token ont souvent un PID correspondant)
            echo "  🔑 Token: $basename_pid (conservé par précaution)"
        fi
    fi
done

echo ""
if [ "$DRY_RUN" = "dry" ]; then
    echo "✅ Mode DRY-RUN terminé. Pour supprimer réellement, exécutez sans 'dry'"
else
    echo "✅ Nettoyage terminé!"
fi

echo ""
echo "💡 Espace libéré approximatif:"
echo "   - fastembed_cache: ~509MB (si supprimé)"
echo "   - Autres fichiers: variable"
