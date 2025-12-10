# 📖 Guide de Configuration Complète

Ce guide détaille toutes les variables d'environnement et options de configuration du système RAG.

## 📋 Table des Matières

- [Vue d'Ensemble](#vue-densemble)
- [Configuration LLM](#configuration-llm)
- [Configuration Qdrant](#configuration-qdrant)
- [Configuration Embeddings](#configuration-embeddings)
- [Configuration Reranker](#configuration-reranker)
- [Configuration OpenTelemetry/Jaeger](#configuration-opentelemetryjaeger)
- [Configuration GPU/CUDA](#configuration-gpucuda)
- [Variables Legacy](#variables-legacy)
- [Troubleshooting](#troubleshooting)

---

## Vue d'Ensemble

Le système utilise des variables d'environnement pour la configuration. Créer un fichier `.env` à la racine du projet :

```bash
cp .env.example .env
# Éditer .env avec vos valeurs
```

Les variables sont chargées automatiquement via `pydantic-settings`.

---

## Configuration LLM

Le système supporte **3 modes de configuration LLM** :

### Mode 1 : vLLM Local (Recommandé pour Performance)

**Avantages** :
- ✅ Latence optimale (<80ms TTFT possible avec GPU)
- ✅ Pas de coût API externe
- ✅ Données restent locales (privacy)
- ✅ Contrôle total sur le modèle

**Configuration** :

```env
LLM_BASE_URL=http://localhost:8001/v1
LLM_MODEL_NAME=TheBloke/Mistral-7B-Instruct-v0.2-AWQ
LLM_API_KEY=dummy-key  # vLLM accepte n'importe quelle clé
```

**Prérequis** :
- Docker (ou vLLM installé localement)
- GPU NVIDIA avec CUDA 11.8+
- Au moins 16GB VRAM (pour Mistral-7B quantifié)

**Démarrer vLLM** :

```bash
docker-compose up -d vllm-service
```

Ou manuellement :

```bash
docker run --gpus all -p 8001:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model TheBloke/Mistral-7B-Instruct-v0.2-AWQ \
  --quantization awq \
  --dtype float16
```

**Modèles Recommandés** :
- `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` : 7B paramètres, quantifié AWQ
- `mistralai/Mistral-7B-Instruct-v0.2` : 7B paramètres, non quantifié (nécessite plus de VRAM)

### Mode 2 : OpenAI API

**Avantages** :
- ✅ Pas besoin de GPU local
- ✅ Modèles performants (GPT-4, GPT-4o)
- ✅ Gestion de l'infrastructure gérée par OpenAI

**Configuration** :

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-openai-api-key-here
LLM_MODEL_NAME=gpt-4o-mini
```

**Modèles Recommandés** :
- `gpt-4o-mini` : Bon compromis qualité/coût (~$0.15/1M tokens input)
- `gpt-4o` : Meilleure qualité, plus cher (~$2.50/1M tokens input)
- `gpt-4-turbo` : Alternative à gpt-4o

**Notes** :
- Latence réseau ajoutée (~500-2000ms selon région)
- Coût par requête selon usage
- Vérifier les quotas API dans votre compte OpenAI

### Mode 3 : Mistral API (Legacy)

**⚠️ Déprécié** : Utiliser OpenAI API ou vLLM local à la place.

**Configuration** :

```env
# Laisser LLM_BASE_URL vide ou non défini
MISTRAL_API_KEY=your-mistral-api-key-here
LLM_MODEL_NAME=mistral-tiny-2407
```

**Modèles Disponibles** :
- `mistral-tiny-2407` : Modèle léger
- `mistral-small-2407` : Modèle moyen
- `mistral-large-2407` : Modèle performant

---

## Configuration Qdrant

### Qdrant Local (Développement)

**Configuration** :

```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=squad_collection
```

**Démarrer Qdrant** :

```bash
./bin/start_qdrant.sh
```

Ou manuellement :

```bash
./qdrant --config-path config/qdrant_config.yaml
```

**Vérifier** : http://localhost:6333/dashboard

### Qdrant Cloud (Production)

**Configuration** :

```env
QDRANT_HOST=your-cluster.qdrant.io
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=squad_collection
QDRANT_API_KEY=your-qdrant-api-key  # Si requis
```

**Notes** :
- Consulter [Qdrant Cloud](https://cloud.qdrant.io/) pour créer un cluster
- L'API key est optionnelle selon votre plan

### Collection

La collection sera créée automatiquement au démarrage si elle n'existe pas.

**Configuration par défaut** :
- **Dense vectors** : 384 dimensions (all-MiniLM-L6-v2)
- **Sparse vectors** : Variable (dépend du modèle BERT)
- **Distance metric** : Cosine

Pour personnaliser, modifier `app/vectorstores/qdrant_store.py` :

```python
await vectorstore.initialize_collection(dense_dim=384)  # Modifier ici
```

---

## Configuration Embeddings

### Modèle Dense

**Variable** : `DENSE_MODEL`

**Valeur par défaut** : `sentence-transformers/all-MiniLM-L6-v2`

**Modèles Recommandés** :

| Modèle | Dimensions | Qualité | Vitesse | RAM Requise |
|--------|-----------|---------|---------|-------------|
| `all-MiniLM-L6-v2` | 384 | ⭐⭐⭐ | ⚡⚡⚡ | 1GB |
| `all-mpnet-base-v2` | 768 | ⭐⭐⭐⭐⭐ | ⚡⚡ | 2GB |
| `all-MiniLM-L12-v2` | 384 | ⭐⭐⭐⭐ | ⚡⚡⚡ | 1.5GB |

**Exemple** :

```env
DENSE_MODEL=sentence-transformers/all-mpnet-base-v2
```

**Notes** :
- Plus de dimensions = meilleure qualité mais plus lent
- Modifier `dense_dim` dans `initialize_collection()` si vous changez de modèle
- Le modèle est téléchargé automatiquement depuis HuggingFace au premier usage

### Modèle Sparse

**Variable** : `SPARSE_MODEL`

**Valeur par défaut** : `bert-base-uncased`

**Fonction** : Génère des embeddings sparse (SPLADE-like) pour la recherche hybride.

**Notes** :
- Modèle BERT standard, utilisé pour générer des indices de mots importants
- Pas de dimensions fixes (vecteur sparse)
- Généralement plus rapide que dense sur CPU

**Exemple** :

```env
SPARSE_MODEL=bert-base-uncased
```

---

## Configuration Reranker

### Activer/Désactiver

**Variable** : `USE_RERANKER`

**Valeur par défaut** : `true`

**Impact** :
- ✅ Améliore la précision (Recall@10 +3-5% observé)
- ⚠️ Ajoute ~20-50ms de latence
- ⚠️ Consomme CPU/GPU pour le modèle Cross-Encoder

**Configuration** :

```env
USE_RERANKER=true  # ou false
```

### Nombre de Documents Avant Reranking

**Variable** : `RERANKER_TOP_K`

**Valeur par défaut** : `20`

**Description** : Nombre de documents récupérés via hybrid search avant reranking.

**Impact** :
- Plus élevé = meilleure qualité finale mais plus lent
- Plus bas = plus rapide mais risque de rater des documents pertinents

**Recommandations** :
- **Développement** : 20 (bon compromis)
- **Production** : 30-50 (si latence acceptable)
- **Temps réel** : 10-15 (si latence critique)

**Exemple** :

```env
RERANKER_TOP_K=30
```

### Modèle Reranker

**Variable** : `RERANKER_MODEL`

**Valeur par défaut** : `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Modèles Disponibles** :

| Modèle | Qualité | Vitesse | RAM |
|--------|---------|---------|-----|
| `ms-marco-MiniLM-L-6-v2` | ⭐⭐⭐⭐ | ⚡⚡⚡ | 500MB |
| `ms-marco-MiniLM-L-12-v2` | ⭐⭐⭐⭐⭐ | ⚡⚡ | 1GB |

**Exemple** :

```env
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
```

---

## Configuration OpenTelemetry/Jaeger

### Activer/Désactiver Export Jaeger

**Variable** : `ENABLE_JAEGER_EXPORT`

**Valeur par défaut** : `true`

**Description** : Si `false`, les traces OpenTelemetry sont exportées vers la console uniquement.

**Configuration** :

```env
# Avec Jaeger
ENABLE_JAEGER_EXPORT=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Sans Jaeger (RunPod, etc.)
ENABLE_JAEGER_EXPORT=false
```

**Démarrer Jaeger** :

```bash
docker-compose up -d jaeger
```

**Accéder à Jaeger UI** : http://localhost:16686

### Endpoint OTLP

**Variable** : `OTEL_EXPORTER_OTLP_ENDPOINT`

**Valeur par défaut** : `http://localhost:4317`

**Description** : Endpoint gRPC pour exporter les traces vers Jaeger.

**Notes** :
- Port 4317 : gRPC (OTLP)
- Port 4318 : HTTP (OTLP)
- Jaeger UI : Port 16686

---

## Configuration GPU/CUDA

### Détection Automatique

Le système détecte automatiquement la disponibilité de CUDA et utilise le GPU si disponible :

```python
import torch
print(torch.cuda.is_available())  # True si GPU disponible
```

### Modèles Utilisant GPU

1. **Embeddings Dense** : `SentenceTransformer` → GPU automatique
2. **Embeddings Sparse** : `BERT` → GPU automatique
3. **Reranker** : `CrossEncoder` → GPU automatique (si disponible)
4. **vLLM** : Requiert GPU explicitement

### Configuration Manuelle (Optionnel)

Les modèles sont automatiquement déplacés sur GPU au chargement. Pas de configuration nécessaire dans `.env`.

### Vérification

```bash
# Vérifier CUDA
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Troubleshooting GPU

- **Erreur "CUDA out of memory"** : Réduire la taille du batch ou utiliser CPU
- **GPU non détecté** : Vérifier installation CUDA et PyTorch avec support CUDA
- **Modèles sur CPU** : Normal si GPU non disponible, les modèles fonctionnent sur CPU

---

## Variables Legacy

### FAISS Index (Migration)

**Variables** :
- `FAISS_INDEX_DIR` : Répertoire contenant l'index FAISS (défaut: `faiss_index`)

**Usage** : Utilisé lors de la migration depuis FAISS vers Qdrant.

### Embedding Model (Legacy)

**Variable** : `EMBEDDING_MODEL_NAME`

**Valeur par défaut** : `sentence-transformers/all-MiniLM-L6-v2`

**Usage** : Maintenu pour compatibilité. Utiliser `DENSE_MODEL` à la place.

### Mistral API (Legacy)

**Variables** :
- `MISTRAL_API_KEY` : Clé API Mistral (legacy)
- `MISTRAL_MODEL_NAME` : Nom du modèle Mistral (legacy)

**Usage** : Utiliser `LLM_API_KEY` et `LLM_MODEL_NAME` avec `LLM_BASE_URL` vide.

---

## Troubleshooting

### Problème : LLM ne répond pas

**Symptômes** : Erreur "No LLM configuration found" ou timeout

**Solutions** :

1. **Vérifier les variables** :
   ```bash
   # Vérifier .env
   cat .env | grep LLM
   ```

2. **vLLM local** :
   ```bash
   # Vérifier que vLLM est démarré
   curl http://localhost:8001/health
   
   # Vérifier les logs
   docker-compose logs vllm-service
   ```

3. **OpenAI API** :
   ```bash
   # Tester la clé API
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $LLM_API_KEY"
   ```

### Problème : Qdrant ne se connecte pas

**Symptômes** : Erreur "Connection refused" ou timeout

**Solutions** :

1. **Vérifier que Qdrant est démarré** :
   ```bash
   curl http://localhost:6333/health
   ```

2. **Vérifier les ports** :
   ```bash
   netstat -tuln | grep 6333
   ```

3. **Vérifier les variables** :
   ```env
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   ```

### Problème : Collection vide

**Symptômes** : Aucun résultat de recherche

**Solutions** :

1. **Réindexer les données** :
   ```bash
   python scripts/migrate_to_qdrant.py
   ```

2. **Vérifier la collection** :
   ```bash
   curl http://localhost:6333/collections/squad_collection
   ```

### Problème : GPU non utilisé

**Symptômes** : Modèles très lents, logs indiquant CPU

**Solutions** :

1. **Vérifier CUDA** :
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Vérifier PyTorch avec CUDA** :
   ```bash
   python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
   ```

3. **Réinstaller PyTorch avec CUDA** :
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Problème : Erreur Jaeger

**Symptômes** : Erreurs de connexion à Jaeger dans les logs

**Solutions** :

1. **Désactiver Jaeger** :
   ```env
   ENABLE_JAEGER_EXPORT=false
   ```

2. **Ou démarrer Jaeger** :
   ```bash
   docker-compose up -d jaeger
   ```

---

## Exemples de Configuration

### Configuration Minimale (OpenAI API)

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-key
LLM_MODEL_NAME=gpt-4o-mini
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### Configuration Production (vLLM + Qdrant Cloud)

```env
LLM_BASE_URL=http://localhost:8001/v1
LLM_MODEL_NAME=TheBloke/Mistral-7B-Instruct-v0.2-AWQ
QDRANT_HOST=your-cluster.qdrant.io
QDRANT_PORT=6333
QDRANT_API_KEY=your-key
USE_RERANKER=true
RERANKER_TOP_K=30
ENABLE_JAEGER_EXPORT=false
```

### Configuration Développement (Tous les services locaux)

```env
LLM_BASE_URL=http://localhost:8001/v1
LLM_MODEL_NAME=TheBloke/Mistral-7B-Instruct-v0.2-AWQ
QDRANT_HOST=localhost
QDRANT_PORT=6333
USE_RERANKER=true
RERANKER_TOP_K=20
ENABLE_JAEGER_EXPORT=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

---

## Ressources Additionnelles

- [README.md](../README.md) : Guide de démarrage rapide
- [API Documentation](API.md) : Documentation complète de l'API (à créer)
- [Architecture](ARCHITECTURE.md) : Documentation technique approfondie (à créer)

