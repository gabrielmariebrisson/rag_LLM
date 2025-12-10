# 📖 Guide de Configuration Complète

Ce guide détaille toutes les variables d'environnement et options de configuration du système RAG.

## 📋 Table des Matières

- [Vue d'Ensemble](#vue-densemble)
- [Configuration LLM](#configuration-llm)
- [Configuration HuggingFace Token](#configuration-huggingface-token)
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
LLM_MODEL_NAME=casperhansen/llama-3-8b-instruct-awq
LLM_API_KEY=dummy-key  # vLLM accepte n'importe quelle clé
HUGGING_FACE_HUB_TOKEN=your-huggingface-token-here  # Requis pour télécharger certains modèles
```

**Prérequis** :
- Docker (ou vLLM installé localement)
- GPU NVIDIA avec CUDA 11.8+
- Au moins 16GB VRAM (pour Llama-3-8B quantifié)
- **HuggingFace Token** : Requis pour télécharger certains modèles gated. Obtenez-le sur [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

**Démarrer vLLM** :

```bash
docker-compose up -d vllm-service
```

Ou manuellement :

```bash
docker run --gpus all -p 8001:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN=your-token-here \
  vllm/vllm-openai:latest \
  --model casperhansen/llama-3-8b-instruct-awq \
  --quantization awq \
  --dtype float16
```

**Modèles Recommandés** :
- `casperhansen/llama-3-8b-instruct-awq` : 8B paramètres, quantifié AWQ (recommandé)
- `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` : 7B paramètres, quantifié AWQ (alternative)
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
LLM_MODEL_NAME=casperhansen/llama-3-8b-instruct-awq
```

**Modèles Disponibles** :
- `mistral-tiny-2407` : Modèle léger
- `mistral-small-2407` : Modèle moyen
- `mistral-large-2407` : Modèle performant

---

## Configuration HuggingFace Token

### Variable : `HUGGING_FACE_HUB_TOKEN`

**Description** : Token d'accès HuggingFace Hub requis pour télécharger certains modèles gated (modèles privés ou nécessitant une authentification).

**Quand est-ce requis ?**
- Pour télécharger des modèles gated depuis HuggingFace Hub
- Pour vLLM qui télécharge les modèles au démarrage
- Pour les embeddings et reranker qui téléchargent depuis HuggingFace

**Comment obtenir un token ?**

1. Créer un compte sur [HuggingFace.co](https://huggingface.co/join)
2. Aller dans [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Créer un nouveau token avec les permissions "Read"
4. Copier le token (format : `hf_xxxxxxxxxxxxx`)

**Configuration** :

```env
HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxx
```

**Docker Compose** :

Le token est automatiquement passé au conteneur vLLM via `docker-compose.yml` :

```yaml
environment:
  - HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
```

**Kubernetes** :

Créer un Secret Kubernetes :

```bash
kubectl create secret generic huggingface-secret \
  --from-literal=token=hf_xxxxxxxxxxxxx
```

Le secret est référencé dans `k8s/vllm.yaml`.

**Notes** :
- Le token est optionnel si tous les modèles utilisés sont publics
- Pour les modèles gated (comme certains modèles Llama), le token est obligatoire
- Ne jamais commiter le token dans le code source (utiliser `.env` qui est dans `.gitignore`)

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
- **Dense vectors** : 1024 dimensions (BGE-M3)
- **Sparse vectors** : Variable (dépend du modèle SPLADE)
- **Distance metric** : Cosine

Pour personnaliser, modifier `app/vectorstores/qdrant_store.py` :

```python
await vectorstore.initialize_collection(dense_dim=settings.DENSE_DIM)  # Utilise DENSE_DIM depuis config (1024 par défaut)
```

---

## Configuration Embeddings

**Note importante** : Les embeddings utilisent maintenant `fastembed` au lieu de `sentence-transformers` pour de meilleures performances et une utilisation mémoire réduite.

### Modèle Dense

**Variable** : `DENSE_MODEL`

**Valeur par défaut** : `BAAI/bge-m3`

**Modèles Recommandés** :

| Modèle | Dimensions | Qualité | Vitesse | RAM Requise |
|--------|-----------|---------|---------|-------------|
| `BAAI/bge-m3` | 1024 | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ | 2GB |
| `BAAI/bge-large-en-v1.5` | 1024 | ⭐⭐⭐⭐⭐ | ⚡⚡ | 3GB |
| `BAAI/bge-base-en-v1.5` | 768 | ⭐⭐⭐⭐ | ⚡⚡⚡ | 1.5GB |

**Exemple** :

```env
DENSE_MODEL=BAAI/bge-m3
```

**Notes** :
- Plus de dimensions = meilleure qualité mais plus lent
- La dimension est automatiquement détectée depuis le modèle (1024 pour BGE-M3)
- Le modèle est téléchargé automatiquement depuis HuggingFace au premier usage
- `fastembed` gère automatiquement GPU/CPU

### Modèle Sparse

**Variable** : `SPARSE_MODEL`

**Valeur par défaut** : `prithivida/Splade_pp_en_v1`

**Fonction** : Génère des embeddings sparse (SPLADE) pour la recherche hybride via `fastembed.SparseTextEmbedding`.

**Notes** :
- Modèle SPLADE optimisé pour la recherche sparse
- Pas de dimensions fixes (vecteur sparse avec indices de vocabulaire)
- Généralement plus rapide que dense sur CPU
- Utilise `fastembed` pour un chargement et traitement optimisés

**Exemple** :

```env
SPARSE_MODEL=prithivida/Splade_pp_en_v1
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

**Valeur par défaut** : `100`

**Description** : Nombre de documents récupérés via hybrid search avant reranking.

**Impact** :
- Plus élevé = meilleure qualité finale mais plus lent
- Plus bas = plus rapide mais risque de rater des documents pertinents

**Recommandations** :
- **Développement** : 50-100 (bon compromis)
- **Production** : 100-150 (si latence acceptable)
- **Temps réel** : 20-30 (si latence critique)

**Exemple** :

```env
RERANKER_TOP_K=100
```

### Modèle Reranker

**Variable** : `RERANKER_MODEL`

**Valeur par défaut** : `BAAI/bge-reranker-v2-m3`

**Modèles Disponibles** :

| Modèle | Qualité | Vitesse | RAM |
|--------|---------|---------|-----|
| `BAAI/bge-reranker-v2-m3` | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ | 1GB |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ⭐⭐⭐⭐ | ⚡⚡⚡ | 500MB |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | ⭐⭐⭐⭐⭐ | ⚡⚡ | 1GB |

**Exemple** :

```env
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

**Notes** :
- BGE-reranker-v2-m3 est le modèle SOTA recommandé
- Utilise `CrossEncoder` de `sentence-transformers` (dépendance requise)

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

1. **Embeddings Dense** : `fastembed.TextEmbedding` → GPU automatique (si disponible)
2. **Embeddings Sparse** : `fastembed.SparseTextEmbedding` → GPU automatique (si disponible)
3. **Reranker** : `CrossEncoder` (sentence-transformers) → GPU automatique (si disponible)
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

**Valeur par défaut** : `BAAI/bge-m3`

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

### Problème : Erreur de téléchargement de modèle HuggingFace

**Symptômes** : Erreur "401 Unauthorized" ou "403 Forbidden" lors du téléchargement de modèles

**Solutions** :

1. **Vérifier que le token est défini** :
   ```bash
   echo $HUGGING_FACE_HUB_TOKEN
   ```

2. **Vérifier que le token est valide** :
   ```bash
   curl -H "Authorization: Bearer $HUGGING_FACE_HUB_TOKEN" \
     https://huggingface.co/api/whoami
   ```

3. **Pour Docker Compose** :
   ```bash
   # Vérifier que le token est dans .env
   cat .env | grep HUGGING_FACE_HUB_TOKEN
   
   # Redémarrer le service vLLM
   docker-compose restart vllm-service
   ```

4. **Pour Kubernetes** :
   ```bash
   # Vérifier que le secret existe
   kubectl get secret huggingface-secret
   
   # Créer le secret si nécessaire
   kubectl create secret generic huggingface-secret \
     --from-literal=token=hf_xxxxxxxxxxxxx
   ```

5. **Pour les modèles gated** :
   - S'assurer d'avoir accepté les conditions d'utilisation du modèle sur HuggingFace
   - Visiter la page du modèle et cliquer sur "Agree and access repository"

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
LLM_MODEL_NAME=casperhansen/llama-3-8b-instruct-awq
HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxx  # Requis pour télécharger les modèles
QDRANT_HOST=your-cluster.qdrant.io
QDRANT_PORT=6333
QDRANT_API_KEY=your-key
USE_RERANKER=true
RERANKER_TOP_K=100
ENABLE_JAEGER_EXPORT=false
```

### Configuration Développement (Tous les services locaux)

```env
LLM_BASE_URL=http://localhost:8001/v1
LLM_MODEL_NAME=casperhansen/llama-3-8b-instruct-awq
HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxx  # Requis pour télécharger les modèles
QDRANT_HOST=localhost
QDRANT_PORT=6333
USE_RERANKER=true
RERANKER_TOP_K=100
ENABLE_JAEGER_EXPORT=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

---

## Ressources Additionnelles

- [README.md](../README.md) : Guide de démarrage rapide
- [API Documentation](API.md) : Documentation complète de l'API (à créer)
- [Architecture](ARCHITECTURE.md) : Documentation technique approfondie (à créer)

