# 🚀 RAG System - Guide Complet

Système RAG (Retrieval-Augmented Generation) avec Qdrant, recherche hybride, reranking et vLLM.

## 📋 Table des matières

- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Démarrage des services](#démarrage-des-services)
- [Migration des données](#migration-des-données)
- [Lancement de l'application](#lancement-de-lapplication)
- [Tests et évaluation](#tests-et-évaluation)
- [Déploiement](#déploiement)
- [Troubleshooting](#troubleshooting)

---

## 🔧 Prérequis

### Logiciels requis

```bash
# Python 3.10+
python --version

# Docker & Docker Compose
docker --version
docker-compose --version

# Git
git --version
```

### GPU (optionnel, pour vLLM local)

- GPU NVIDIA avec CUDA 12+ (pour vLLM local)
- Ou utiliser Google Colab (voir `GPU_CLOUD_GUIDE.md`)
- Ou utiliser une API externe (OpenAI, Mistral)

---

## 📦 Installation

### 1. Cloner le repository

```bash
git clone <votre-repo-url>
cd rag_LLM
```

### 2. Créer un environnement virtuel

```bash
# Avec venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Avec conda
conda create -n rag_llm python=3.10
conda activate rag_llm
```

### 3. Installer les dépendances

```bash
# Mettre à jour pip
pip install --upgrade pip setuptools wheel

# Installer toutes les dépendances
pip install -r requirements.txt
```

### 4. Vérifier l'installation

```bash
# Vérifier que les packages critiques sont installés
python -c "import fastapi, qdrant_client, vllm, transformers; print('✅ Installation OK')"
```

---

## ⚙️ Configuration

### 1. Créer le fichier `.env`

```bash
# Copier le template (si disponible)
cp .env.example .env

# Ou créer manuellement
touch .env
```

### 2. Configurer les variables d'environnement

Éditez le fichier `.env` avec vos paramètres :

```bash
# ============================================
# Configuration LLM (choisir UNE option)
# ============================================

# Option 1 : vLLM local (nécessite GPU)
LLM_BASE_URL=http://localhost:8001/v1
LLM_API_KEY=dummy-key
LLM_MODEL_NAME=TheBloke/Mistral-7B-Instruct-v0.2-AWQ

# Option 2 : OpenAI API
# LLM_BASE_URL=
# LLM_API_KEY=sk-...
# LLM_MODEL_NAME=gpt-4o-mini

# Option 3 : Mistral API (legacy)
# MISTRAL_API_KEY=your-mistral-key
# MISTRAL_MODEL_NAME=mistral-tiny-2407

# ============================================
# Configuration Qdrant
# ============================================
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=squad_collection

# ============================================
# Configuration Reranker
# ============================================
USE_RERANKER=true
RERANKER_TOP_K=20

# ============================================
# Configuration Embeddings
# ============================================
DENSE_MODEL=sentence-transformers/all-MiniLM-L6-v2
SPARSE_MODEL=prunebert-base-uncased-6-minilayer
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# ============================================
# Configuration Backend
# ============================================
BACKEND_URL=http://localhost:8000

# ============================================
# Configuration OpenTelemetry (Jaeger)
# ============================================
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=rag-system
```

---

## 🚀 Démarrage des services

### Option A : Avec Docker Compose (Recommandé)

#### 1. Démarrer Qdrant + Jaeger

```bash
# Démarrer uniquement Qdrant et Jaeger (sans vLLM si pas de GPU)
docker-compose up -d jaeger

# Démarrer Qdrant séparément (si pas dans docker-compose)
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_data:/qdrant/storage \
  qdrant/qdrant:latest
```

#### 2. Démarrer vLLM (si GPU disponible)

```bash
# Vérifier que le GPU est disponible
nvidia-smi

# Démarrer vLLM
docker-compose up -d vllm-service

# Vérifier les logs
docker-compose logs -f vllm-service

# Vérifier que vLLM est prêt
curl http://localhost:8001/health
```

#### 3. Vérifier tous les services

```bash
# Voir l'état de tous les services
docker-compose ps

# Voir les logs de tous les services
docker-compose logs -f

# Arrêter tous les services
docker-compose down

# Arrêter et supprimer les volumes
docker-compose down -v
```

### Option B : Services locaux (sans Docker)

#### 1. Qdrant local

```bash
# Télécharger et lancer Qdrant
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_data:/qdrant/storage \
  qdrant/qdrant:latest
```

#### 2. vLLM local (si GPU disponible)

```bash
# Installer vLLM
pip install vllm

# Lancer le serveur vLLM
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Mistral-7B-Instruct-v0.2-AWQ \
  --quantization awq \
  --dtype float16 \
  --port 8000 \
  --host 0.0.0.0
```

#### 3. Jaeger local

```bash
# Lancer Jaeger
docker run -d \
  --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  -e COLLECTOR_OTLP_ENABLED=true \
  jaegertracing/all-in-one:latest
```

### Option C : Google Colab (sans GPU local)

Voir le notebook `colab_vllm_setup.ipynb` pour utiliser vLLM sur Colab.

---

## 📊 Migration des données

### 1. Préparer les données SQuAD

```bash
# Télécharger SQuAD 2.0 (si nécessaire)
# Les données doivent être dans squad_2.0/train.csv
# Format attendu : question,context,answer
```

### 2. Migrer vers Qdrant

```bash
# Lancer le script de migration
python scripts/migrate_to_qdrant.py

# Le script va :
# - Charger les données depuis squad_2.0/train.csv
# - Générer les embeddings hybrides (dense + sparse)
# - Indexer dans Qdrant
```

### 3. Vérifier la migration

```bash
# Vérifier via l'API Qdrant
curl http://localhost:6333/collections/squad_collection

# Ou via le dashboard Qdrant
open http://localhost:6333/dashboard
```

---

## 🎯 Lancement de l'application

### 1. Backend FastAPI

```bash
# Activer l'environnement virtuel
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Lancer le serveur de développement
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Ou en production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Endpoints disponibles** :
- API : http://localhost:8000
- Docs : http://localhost:8000/docs
- Health : http://localhost:8000/health

### 2. Frontend Streamlit

```bash
# Dans un nouveau terminal
source venv/bin/activate

# Lancer Streamlit
streamlit run frontend/app.py --server.port 8501

# Ou avec configuration personnalisée
streamlit run frontend/app.py \
  --server.port 8501 \
  --server.address 0.0.0.0
```

**Interface** : http://localhost:8501

### 3. Vérifier que tout fonctionne

```bash
# Test de l'endpoint /health
curl http://localhost:8000/health

# Test de l'endpoint /chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "k": 5,
    "language": "en"
  }'
```

---

## 🧪 Tests et évaluation

### 1. Évaluation du système de retrieval

```bash
# Lancer l'évaluation Recall@5
python scripts/evaluate_retrieval.py

# Le script va :
# - Charger 50 questions aléatoires de SQuAD
# - Tester avec et sans reranker
# - Calculer Recall@5 et latence
# - Afficher les résultats
```

### 2. Tests manuels

```bash
# Test avec curl
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who invented the telephone?",
    "k": 5,
    "language": "fr",
    "use_reranker": true
  }'

# Test avec Python
python -c "
import requests
response = requests.post('http://localhost:8000/chat', json={
    'query': 'What is photosynthesis?',
    'k': 5,
    'language': 'en'
})
print(response.json())
"
```

### 3. Visualiser les traces Jaeger

```bash
# Ouvrir Jaeger UI
open http://localhost:16686

# Rechercher les traces du service "rag-system"
# Filtrer par opération : rag.request, rag.retrieval, llm.generation
```

---

## 🚢 Déploiement

### Option A : Kubernetes

```bash
# Créer le namespace
kubectl create namespace rag-system

# Déployer vLLM (nécessite GPU)
kubectl apply -f k8s/vllm.yaml -n rag-system

# Déployer le backend
kubectl apply -f k8s/backend.yaml -n rag-system

# Déployer l'ingress
kubectl apply -f k8s/ingress.yaml -n rag-system

# Vérifier le déploiement
kubectl get pods -n rag-system
kubectl get services -n rag-system
```

### Option B : Docker Compose (Production)

```bash
# Lancer tous les services
docker-compose up -d

# Vérifier les logs
docker-compose logs -f

# Redémarrer un service
docker-compose restart backend

# Mettre à jour et redémarrer
docker-compose pull
docker-compose up -d
```

### Option C : Cloud (GCP, AWS, Azure)

Voir `GPU_CLOUD_GUIDE.md` pour les instructions détaillées.

---

## 🔍 Monitoring et Observabilité

### 1. Jaeger (Tracing)

```bash
# Accéder à l'UI Jaeger
open http://localhost:16686

# Rechercher les traces
# Service: rag-system
# Opérations: rag.request, rag.retrieval, rag.reranking, llm.generation
```

### 2. Logs

```bash
# Logs Docker Compose
docker-compose logs -f backend
docker-compose logs -f vllm-service

# Logs Kubernetes
kubectl logs -f deployment/rag-backend -n rag-system
```

### 3. Métriques

```bash
# Health check
curl http://localhost:8000/health

# Vérifier Qdrant
curl http://localhost:6333/collections/squad_collection
```

---

## 🛠️ Troubleshooting

### Problème : Qdrant ne démarre pas

```bash
# Vérifier que le port 6333 est libre
lsof -i :6333

# Vérifier les logs
docker logs qdrant

# Redémarrer Qdrant
docker restart qdrant
```

### Problème : vLLM ne démarre pas (GPU)

```bash
# Vérifier que le GPU est disponible
nvidia-smi

# Vérifier Docker GPU
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# Vérifier les logs vLLM
docker-compose logs vllm-service

# Utiliser une API externe à la place
# Modifier .env : LLM_BASE_URL= et LLM_API_KEY=sk-...
```

### Problème : Backend ne se connecte pas à Qdrant

```bash
# Vérifier que Qdrant est accessible
curl http://localhost:6333/collections

# Vérifier la configuration dans .env
cat .env | grep QDRANT

# Tester la connexion manuellement
python -c "
from qdrant_client import QdrantClient
client = QdrantClient(host='localhost', port=6333)
print(client.get_collections())
"
```

### Problème : Erreur "Services not loaded"

```bash
# Vérifier les logs du backend
docker-compose logs backend
# ou
tail -f logs/app.log

# Vérifier que Qdrant est démarré
docker ps | grep qdrant

# Redémarrer le backend
docker-compose restart backend
```

### Problème : Modèles trop lents à charger

```bash
# Les modèles sont téléchargés au premier lancement
# Vérifier l'espace disque
df -h

# Vérifier le cache HuggingFace
ls -lh ~/.cache/huggingface/

# Nettoyer le cache si nécessaire
rm -rf ~/.cache/huggingface/transformers/
```

### Problème : Port déjà utilisé

```bash
# Trouver le processus utilisant le port
lsof -i :8000  # Backend
lsof -i :6333  # Qdrant
lsof -i :8001  # vLLM
lsof -i :16686 # Jaeger

# Tuer le processus
kill -9 <PID>

# Ou changer le port dans .env ou docker-compose.yml
```

---

## 📚 Commandes utiles

### Docker

```bash
# Voir tous les conteneurs
docker ps -a

# Voir les images
docker images

# Nettoyer les conteneurs arrêtés
docker container prune

# Nettoyer les images non utilisées
docker image prune -a

# Voir l'utilisation des ressources
docker stats
```

### Python

```bash
# Vérifier les dépendances
pip list

# Mettre à jour une dépendance
pip install --upgrade <package>

# Vérifier les conflits
pip check

# Créer un requirements.txt à jour
pip freeze > requirements.txt
```

### Qdrant

```bash
# Lister les collections
curl http://localhost:6333/collections

# Voir les stats d'une collection
curl http://localhost:6333/collections/squad_collection

# Supprimer une collection
curl -X DELETE http://localhost:6333/collections/squad_collection
```

### vLLM

```bash
# Vérifier la santé
curl http://localhost:8001/health

# Lister les modèles disponibles
curl http://localhost:8001/v1/models

# Test de génération
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## 📖 Documentation supplémentaire

- **Architecture** : `README_ARCHITECTURE.md`
- **Qdrant** : `README_QDRANT.md`
- **vLLM** : `README_VLLM.md`
- **Ops** : `README_OPS.md`
- **GPU Cloud** : `GPU_CLOUD_GUIDE.md`

---

## 🎯 Workflow complet (Quick Start)

```bash
# 1. Installation
git clone <repo> && cd rag_LLM
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configuration
cp .env.example .env
# Éditer .env avec vos clés API

# 3. Démarrer les services
docker-compose up -d qdrant jaeger
# Si GPU : docker-compose up -d vllm-service

# 4. Migrer les données
python scripts/migrate_to_qdrant.py

# 5. Lancer le backend
uvicorn app.main:app --reload

# 6. Lancer le frontend (nouveau terminal)
streamlit run frontend/app.py

# 7. Tester
open http://localhost:8501
```

---

## 🆘 Support

Pour toute question ou problème :
1. Vérifier les logs : `docker-compose logs -f`
2. Vérifier la configuration : `cat .env`
3. Consulter la documentation dans les README_*.md
4. Vérifier les issues GitHub

---

## 📝 Notes importantes

- **GPU requis** : vLLM local nécessite un GPU NVIDIA. Utilisez une API externe ou Colab sinon.
- **Mémoire** : Les modèles nécessitent ~8GB RAM minimum.
- **Premier lancement** : Les modèles sont téléchargés automatiquement (peut prendre du temps).
- **Production** : Utiliser `--workers` avec uvicorn et configurer les timeouts.

---

**Bon développement ! 🚀**

