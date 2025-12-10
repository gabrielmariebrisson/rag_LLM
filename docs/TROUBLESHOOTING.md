# 🔧 Guide de Troubleshooting

Guide complet pour résoudre les problèmes courants du système RAG.

## 📋 Table des Matières

- [Problèmes d'Installation](#problèmes-dinstallation)
- [Problèmes de Configuration](#problèmes-de-configuration)
- [Problèmes Qdrant](#problèmes-qdrant)
- [Problèmes GPU/CUDA](#problèmes-gpucuda)
- [Problèmes LLM](#problèmes-llm)
- [Problèmes de Performance](#problèmes-de-performance)
- [Problèmes OpenTelemetry/Jaeger](#problèmes-opentelemetryjaeger)
- [Problèmes Frontend](#problèmes-frontend)
- [Problèmes de Déploiement](#problèmes-de-déploiement)

---

## Problèmes d'Installation

### Erreur : `pip install` échoue

**Symptômes** :
```
ERROR: Could not find a version that satisfies the requirement torch>=2.9.0
```

**Solutions** :

1. **Mettre à jour pip** :
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. **Installer PyTorch séparément** (si problème CUDA) :
   ```bash
   # Pour CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # Pour CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # CPU only
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Installer les dépendances par groupes** :
   ```bash
   pip install fastapi uvicorn pydantic
   pip install sentence-transformers transformers
   pip install qdrant-client
   # etc.
   ```

### Erreur : `Python 3.12` non trouvé

**Solutions** :

1. **Installer Python 3.12** :
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3.12 python3.12-venv python3.12-dev
   
   # macOS (Homebrew)
   brew install python@3.12
   
   # Windows
   # Télécharger depuis python.org
   ```

2. **Vérifier la version** :
   ```bash
   python3.12 --version
   ```

### Erreur : Permissions refusées lors de l'installation

**Symptômes** :
```
ERROR: Could not install packages due to an OSError: [Errno 13] Permission denied
```

**Solutions** :

1. **Utiliser un environnement virtuel** (recommandé) :
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   ```

2. **Ou utiliser `--user`** :
   ```bash
   pip install --user -r requirements.txt
   ```

---

## Problèmes de Configuration

### Erreur : `LLM_API_KEY or MISTRAL_API_KEY must be set`

**Symptômes** :
```
ValueError: LLM_API_KEY or MISTRAL_API_KEY must be set...
```

**Solutions** :

1. **Vérifier le fichier `.env`** :
   ```bash
   cat .env | grep LLM
   ```

2. **Pour vLLM local** :
   ```env
   LLM_BASE_URL=http://localhost:8001/v1
   LLM_API_KEY=dummy-key
   ```

3. **Pour OpenAI API** :
   ```env
   LLM_BASE_URL=https://api.openai.com/v1
   LLM_API_KEY=sk-your-key-here
   ```

4. **Pour Mistral API (legacy)** :
   ```env
   MISTRAL_API_KEY=your-mistral-key
   ```

### Erreur : Variables d'environnement non chargées

**Symptômes** : Configuration par défaut utilisée malgré `.env`

**Solutions** :

1. **Vérifier que `.env` est à la racine** :
   ```bash
   ls -la .env
   ```

2. **Vérifier le format** (pas d'espaces autour de `=`) :
   ```env
   # ✅ Correct
   LLM_API_KEY=sk-xxx
   
   # ❌ Incorrect
   LLM_API_KEY = sk-xxx
   ```

3. **Recharger l'environnement** :
   ```bash
   # Linux/Mac
   export $(cat .env | xargs)
   
   # Ou redémarrer le terminal
   ```

---

## Problèmes Qdrant

### Erreur : `Connection refused` à Qdrant

**Symptômes** :
```
ConnectionError: Connection refused to localhost:6333
```

**Solutions** :

1. **Vérifier que Qdrant est démarré** :
   ```bash
   curl http://localhost:6333/health
   ```

2. **Démarrer Qdrant** :
   ```bash
   ./bin/start_qdrant.sh
   # ou manuellement
   ./qdrant --config-path config/qdrant_config.yaml
   ```

3. **Vérifier les ports** :
   ```bash
   netstat -tuln | grep 6333
   # ou
   lsof -i :6333
   ```

4. **Vérifier la configuration** :
   ```env
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   ```

### Erreur : Collection introuvable

**Symptômes** :
```
CollectionNotFoundError: Collection 'squad_collection' not found
```

**Solutions** :

1. **Vérifier que la collection existe** :
   ```bash
   curl http://localhost:6333/collections/squad_collection
   ```

2. **Créer/Indexer la collection** :
   ```bash
   python scripts/migrate_to_qdrant.py
   ```

3. **Vérifier le nom de collection** :
   ```env
   QDRANT_COLLECTION_NAME=squad_collection
   ```

### Erreur : Collection vide

**Symptômes** : Aucun résultat de recherche retourné

**Solutions** :

1. **Vérifier le nombre de points** :
   ```bash
   curl http://localhost:6333/collections/squad_collection
   # Vérifier "points_count"
   ```

2. **Réindexer les données** :
   ```bash
   python scripts/migrate_to_qdrant.py
   ```

3. **Vérifier le CSV source** :
   ```bash
   head -5 data/raw/squad_2.0/train.csv
   ```

### Erreur : Timeout lors de l'indexation

**Symptômes** :
```
TimeoutError: Operation timed out
```

**Solutions** :

1. **Réduire la taille du batch** :
   ```python
   # Dans migrate_to_qdrant.py
   batch_size = 16  # Au lieu de 32
   ```

2. **Augmenter le timeout** :
   ```python
   # Dans qdrant_store.py
   timeout=300.0  # 5 minutes au lieu de 120
   ```

3. **Indexer par petits lots** :
   ```bash
   # Diviser le CSV et indexer progressivement
   split -l 1000 data/raw/squad_2.0/train.csv data/raw/squad_2.0/batch_
   ```

---

## Problèmes GPU/CUDA

### Erreur : `CUDA out of memory`

**Symptômes** :
```
RuntimeError: CUDA out of memory. Tried to allocate...
```

**Solutions** :

1. **Libérer la mémoire GPU** :
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

2. **Réduire la taille du batch** :
   ```python
   # Dans embeddings.py
   batch_size = 8  # Au lieu de 32
   ```

3. **Utiliser CPU temporairement** :
   ```python
   # Forcer CPU
   import os
   os.environ["CUDA_VISIBLE_DEVICES"] = ""
   ```

4. **Vérifier l'utilisation GPU** :
   ```bash
   nvidia-smi
   ```

### GPU non détecté

**Symptômes** :
```
CUDA available: False
```

**Solutions** :

1. **Vérifier l'installation CUDA** :
   ```bash
   nvcc --version
   nvidia-smi
   ```

2. **Réinstaller PyTorch avec CUDA** :
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Vérifier la compatibilité** :
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   print(torch.version.cuda)
   ```

4. **Note** : Les modèles fonctionnent aussi sur CPU, mais plus lentement.

### Erreur : `CUDA version mismatch`

**Symptômes** :
```
CUDA runtime version mismatch: expected 11.8, got 12.1
```

**Solutions** :

1. **Réinstaller PyTorch pour la bonne version CUDA** :
   ```bash
   # Pour CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Ou utiliser CPU** :
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

---

## Problèmes LLM

### Erreur : `Connection refused` à vLLM

**Symptômes** :
```
ConnectionError: Connection refused to localhost:8001
```

**Solutions** :

1. **Vérifier que vLLM est démarré** :
   ```bash
   curl http://localhost:8001/health
   ```

2. **Démarrer vLLM avec Docker** :
   ```bash
   docker-compose up -d vllm-service
   docker-compose logs -f vllm-service
   ```

3. **Vérifier la configuration** :
   ```env
   LLM_BASE_URL=http://localhost:8001/v1
   ```

### Erreur : OpenAI API key invalide

**Symptômes** :
```
401 Unauthorized: Invalid API key
```

**Solutions** :

1. **Vérifier la clé API** :
   ```bash
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $LLM_API_KEY"
   ```

2. **Vérifier les quotas** :
   - Aller sur https://platform.openai.com/usage
   - Vérifier les limites et crédits

3. **Regénérer la clé si nécessaire** :
   - Aller sur https://platform.openai.com/api-keys

### Erreur : Timeout LLM

**Symptômes** :
```
TimeoutError: Request timed out
```

**Solutions** :

1. **Augmenter le timeout** :
   ```python
   # Dans llm_client.py ou app/main.py
   timeout=120.0  # 2 minutes
   ```

2. **Réduire la taille du contexte** :
   ```python
   # Prendre moins de documents
   k = 3  # Au lieu de 5
   ```

3. **Vérifier la latence réseau** (API externe) :
   ```bash
   ping api.openai.com
   ```

### Erreur : Modèle introuvable

**Symptômes** :
```
404 Not Found: Model 'xxx' not found
```

**Solutions** :

1. **Vérifier le nom du modèle** :
   ```env
   # OpenAI
   LLM_MODEL_NAME=gpt-4o-mini  # Pas gpt-5-nano
   
   # vLLM
   LLM_MODEL_NAME=TheBloke/Mistral-7B-Instruct-v0.2-AWQ
   ```

2. **Vérifier l'accès au modèle** :
   ```bash
   # Pour HuggingFace
   huggingface-cli login
   ```

---

## Problèmes de Performance

### Latence élevée (> 5 secondes)

**Diagnostic** :

1. **Identifier le goulot d'étranglement** :
   ```python
   # Activer les logs détaillés
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Mesurer chaque étape** :
   ```python
   import time
   
   start = time.perf_counter()
   # Embedding
   embeddings = await embedding_service.embed_hybrid([query])
   print(f"Embedding: {time.perf_counter() - start:.2f}s")
   
   start = time.perf_counter()
   # Search
   results = await vectorstore.hybrid_search(query)
   print(f"Search: {time.perf_counter() - start:.2f}s")
   ```

**Solutions** :

1. **Utiliser GPU** :
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Réduire RERANKER_TOP_K** :
   ```env
   RERANKER_TOP_K=10  # Au lieu de 20
   ```

3. **Désactiver le reranker** :
   ```env
   USE_RERANKER=false
   ```

4. **Optimiser Qdrant** :
   ```python
   # HNSW parameters
   m=16, ef_construct=100
   ```

### Mémoire RAM insuffisante

**Symptômes** :
```
MemoryError: Unable to allocate array
```

**Solutions** :

1. **Réduire la taille du batch** :
   ```python
   batch_size = 8  # Au lieu de 32
   ```

2. **Limiter le nombre de documents** :
   ```python
   k = 3  # Moins de documents récupérés
   ```

3. **Utiliser un modèle plus petit** :
   ```env
   DENSE_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Au lieu de all-mpnet-base-v2
   ```

---

## Problèmes OpenTelemetry/Jaeger

### Erreur : `Connection refused` à Jaeger

**Symptômes** :
```
Failed to export spans: Connection refused to localhost:4317
```

**Solutions** :

1. **Désactiver l'export Jaeger** :
   ```env
   ENABLE_JAEGER_EXPORT=false
   ```

2. **Ou démarrer Jaeger** :
   ```bash
   docker-compose up -d jaeger
   ```

3. **Vérifier l'endpoint** :
   ```env
   OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
   ```

**Note** : Les traces OpenTelemetry continuent de fonctionner, elles sont juste exportées vers la console.

### Traces non visibles dans Jaeger

**Solutions** :

1. **Vérifier que Jaeger est accessible** :
   ```bash
   curl http://localhost:16686
   ```

2. **Vérifier le service name** :
   ```python
   # Dans app/main.py
   Resource.create({
       "service.name": "rag-system",
   })
   ```

3. **Attendre quelques secondes** : Les traces peuvent prendre du temps à apparaître.

---

## Problèmes Frontend

### Erreur : Backend non accessible

**Symptômes** :
```
Impossible de contacter le backend à http://localhost:8000
```

**Solutions** :

1. **Vérifier que le backend est démarré** :
   ```bash
   curl http://localhost:8000/health
   ```

2. **Vérifier la configuration** :
   ```bash
   # Dans .env ou variable d'environnement
   BACKEND_URL=http://localhost:8000
   ```

3. **Vérifier CORS** (si frontend sur autre port) :
   ```python
   # Dans app/main.py
   allow_origins=["http://localhost:8501"]
   ```

### Interface Streamlit vide/blanche

**Solutions** :

1. **Vérifier les logs** :
   ```bash
   streamlit run frontend/app.py --logger.level=debug
   ```

2. **Vérifier les dépendances** :
   ```bash
   pip install streamlit httpx pandas plotly
   ```

3. **Nettoyer le cache Streamlit** :
   ```bash
   rm -rf ~/.streamlit/cache
   ```

### Erreur : `ModuleNotFoundError` dans Streamlit

**Solutions** :

1. **Vérifier l'environnement virtuel** :
   ```bash
   which streamlit
   # Doit pointer vers venv/bin/streamlit
   ```

2. **Réinstaller les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

---

## Problèmes de Déploiement

### Erreur : Pod Kubernetes en `CrashLoopBackOff`

**Solutions** :

1. **Vérifier les logs** :
   ```bash
   kubectl logs -n rag-system deployment/rag-backend
   ```

2. **Vérifier les ressources** :
   ```bash
   kubectl describe pod -n rag-system <pod-name>
   ```

3. **Vérifier les secrets** :
   ```bash
   kubectl get secrets -n rag-system
   ```

### Erreur : Service Kubernetes non accessible

**Solutions** :

1. **Vérifier les services** :
   ```bash
   kubectl get svc -n rag-system
   ```

2. **Tester depuis un pod** :
   ```bash
   kubectl exec -it -n rag-system deployment/rag-backend -- \
     curl http://qdrant-service:6333/health
   ```

3. **Vérifier les network policies** :
   ```bash
   kubectl get networkpolicies -n rag-system
   ```

### Erreur : GPU non disponible dans Kubernetes

**Solutions** :

1. **Vérifier les nodes GPU** :
   ```bash
   kubectl get nodes -o json | jq '.items[] | {name: .metadata.name, gpu: .status.capacity."nvidia.com/gpu"}'
   ```

2. **Vérifier NVIDIA GPU Operator** :
   ```bash
   kubectl get pods -n gpu-operator
   ```

3. **Vérifier les ressources** :
   ```yaml
   resources:
     requests:
       nvidia.com/gpu: 1
   ```

---

## Debugging Général

### Activer les Logs Détaillés

#### Backend

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

#### Uvicorn

```bash
uvicorn app.main:app --log-level debug
```

#### Streamlit

```bash
streamlit run frontend/app.py --logger.level=debug
```

### Vérifier la Configuration

```python
# Script de diagnostic
from app.core.config import settings

print("=== Configuration ===")
print(f"LLM_BASE_URL: {settings.LLM_BASE_URL}")
print(f"LLM_MODEL_NAME: {settings.LLM_MODEL_NAME}")
print(f"QDRANT_HOST: {settings.QDRANT_HOST}")
print(f"QDRANT_PORT: {settings.QDRANT_PORT}")
print(f"USE_RERANKER: {settings.USE_RERANKER}")
```

### Test de Connectivité

```python
# Script test_connectivity.py
import asyncio
import httpx
from app.core.config import settings
from app.vectorstores.qdrant_store import QdrantVectorStore
from app.services.embeddings import EmbeddingService

async def test_all():
    # Test Qdrant
    try:
        vectorstore = QdrantVectorStore(settings, None)
        await vectorstore.connect()
        print("✅ Qdrant: Connecté")
    except Exception as e:
        print(f"❌ Qdrant: {e}")
    
    # Test LLM
    try:
        async with httpx.AsyncClient() as client:
            if settings.LLM_BASE_URL:
                resp = await client.get(f"{settings.LLM_BASE_URL.replace('/v1', '')}/health")
                print(f"✅ LLM: {resp.status_code}")
    except Exception as e:
        print(f"❌ LLM: {e}")

asyncio.run(test_all())
```


## Ressources Additionnelles

- [README.md](../README.md) : Guide de démarrage rapide
- [CONFIGURATION.md](CONFIGURATION.md) : Guide de configuration
- [DEPLOYMENT.md](DEPLOYMENT.md) : Guide de déploiement

---

**Version** : 2.0.0  
**Dernière mise à jour** : 2025-01-27

