# 🚀 RAG System - Retrieval-Augmented Generation

Système RAG (Retrieval-Augmented Generation) avancé utilisant la recherche hybride (dense + sparse), le reranking avec Cross-Encoder, et supportant plusieurs backends LLM (vLLM, OpenAI, Mistral).

## 📋 Table des Matières

- [Description](#-description)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Démarrage](#-démarrage)
- [Utilisation](#-utilisation)
- [API Documentation](#-api-documentation)
- [Évaluation](#-évaluation)
- [Déploiement](#-déploiement)
- [Troubleshooting](#-troubleshooting)

## 🎯 Description

Ce projet implémente un système RAG complet basé sur le dataset SQuAD 2.0, avec :

- **Recherche Hybride** : Combinaison de recherche dense (vecteurs) et sparse (BM25-like) via Qdrant
- **Reranking** : Réordonnancement des résultats avec un modèle Cross-Encoder
- **Multi-LLM** : Support de vLLM local, OpenAI API, et Mistral API
- **Multi-langue** : Traduction automatique des réponses (10 langues supportées)
- **Observabilité** : Tracing distribué avec OpenTelemetry et Jaeger

## ✨ Fonctionnalités

- ✅ Recherche vectorielle hybride (dense + sparse embeddings)
- ✅ Reranking avec Cross-Encoder pour améliorer la précision
- ✅ Support multi-backend LLM (vLLM, OpenAI, Mistral)
- ✅ Interface web Streamlit moderne et intuitive
- ✅ API REST FastAPI avec documentation automatique (Swagger)
- ✅ Traduction automatique des réponses
- ✅ Observabilité avec OpenTelemetry/Jaeger
- ✅ Déploiement Docker et Kubernetes
- ✅ Scripts d'évaluation (Recall@K, TTFT)

## 📁 Structure du Projet

Le projet suit une structure modulaire et professionnelle :

```
rag_LLM/
├── app/                    # Code source backend (FastAPI)
├── frontend/               # Interface utilisateur (Streamlit)
├── scripts/                # Scripts utilitaires
├── tests/                  # Tests unitaires et d'intégration
├── config/                 # Fichiers de configuration
├── data/                   # Données (raw, processed, results)
├── docker/                 # Dockerfiles
├── docs/                   # Documentation
├── examples/               # Exemples d'utilisation
├── bin/                    # Scripts exécutables
├── logs/                   # Fichiers de logs
└── archive/                # Fichiers legacy
```


## 🏗️ Architecture

```
┌─────────────────┐
│   Frontend      │
│   Streamlit     │
│   (Port 8501)   │
└────────┬────────┘
         │ HTTP
         │
┌────────▼──────────────────────────────────────┐
│         Backend FastAPI (Port 8000)           │
│  ┌──────────────────────────────────────────┐ │
│  │  RAG Pipeline:                           │ │
│  │  1. Embedding Service (Dense + Sparse)  │ │
│  │  2. Hybrid Search (Qdrant)              │ │
│  │  3. Reranker (Cross-Encoder)            │ │
│  │  4. LLM Client (vLLM/OpenAI/Mistral)    │ │
│  │  5. Translation Service                 │ │
│  └──────────────────────────────────────────┘ │
└────────┬────────────────────┬─────────────────┘
         │                    │
    ┌────▼────┐         ┌─────▼─────┐
    │ Qdrant  │         │   vLLM    │
    │ :6333   │         │  :8001    │
    └─────────┘         └───────────┘
                              │
                         ┌────▼─────┐
                         │   GPU    │
                         │  (CUDA)  │
                         └──────────┘
```

### Composants Principaux

1. **Frontend (Streamlit)** : Interface utilisateur web
2. **Backend (FastAPI)** : API REST et orchestration du pipeline RAG
3. **Qdrant** : Base de données vectorielle pour la recherche hybride
4. **Embedding Service** : Génération d'embeddings dense (sentence-transformers) et sparse (BERT-based)
5. **Reranker** : Modèle Cross-Encoder pour réordonner les résultats
6. **LLM Client** : Client agnostique supportant vLLM, OpenAI, et Mistral
7. **vLLM Service** (optionnel) : Serveur d'inférence local pour latence optimale

## 📦 Prérequis

### Système

- **Python** : 3.12+ (recommandé 3.12.3)
- **CUDA** : 11.8+ ou 12.x (pour GPU, optionnel)
- **Docker** : Optionnel (pour vLLM et Jaeger)
- **RAM** : Minimum 8GB (16GB+ recommandé avec GPU)

### Services Externes (Optionnels)

- **Qdrant** : Local ou cloud (Qdrant Cloud)
- **OpenAI API** : Clé API valide
- **Mistral API** : Clé API valide

## 🔧 Installation

### 1. Cloner le Repository

```bash
git clone <repository-url>
cd rag_LLM
```

### 2. Créer un Environnement Virtuel

```bash
python3.12 -m venv venv
source venv/bin/activate  # Sur Linux/Mac
# ou
venv\Scripts\activate  # Sur Windows
```

### 3. Installer les Dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note** : L'installation peut prendre 10-15 minutes, notamment pour PyTorch et les bibliothèques CUDA.

### 4. Installer Qdrant (sans Docker)

Télécharger Qdrant depuis [qdrant.tech](https://qdrant.tech/documentation/guides/installation/):

```bash
# Linux (exemple)
wget https://github.com/qdrant/qdrant/releases/download/v1.7.0/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
chmod +x qdrant
mv qdrant ./
```

Ou utiliser le script fourni :
```bash
chmod +x bin/start_qdrant.sh
./bin/start_qdrant.sh
```

### 5. Préparer les Données

Le projet utilise le dataset SQuAD 2.0. Les données doivent être dans `data/raw/squad_2.0/train.csv`.

Si nécessaire, convertir depuis JSON :
```bash
python scripts/convert_squad_json_to_csv.py
```

## ⚙️ Configuration

### Variables d'Environnement

Créer un fichier `.env` à la racine du projet :

```bash
cp .env.example .env
# Éditer .env avec vos valeurs
```

Voir [docs/CONFIGURATION.md](docs/CONFIGURATION.md) pour un guide détaillé de toutes les variables.

### Configuration Minimale

#### Option 1 : OpenAI API

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-key-here
LLM_MODEL_NAME=gpt-4o-mini
```

#### Option 2 : vLLM Local (Recommandé pour performance)

```env
LLM_BASE_URL=http://localhost:8001/v1
LLM_MODEL_NAME=TheBloke/Mistral-7B-Instruct-v0.2-AWQ
```

#### Option 3 : Mistral API (Legacy)

```env
MISTRAL_API_KEY=your-mistral-key
LLM_MODEL_NAME=mistral-tiny-2407
```

### Configuration Qdrant

```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=squad_collection
```

### Configuration Reranker

```env
USE_RERANKER=true
RERANKER_TOP_K=20
```

## 🚀 Démarrage

### 1. Démarrer Qdrant

```bash
./bin/start_qdrant.sh
# ou manuellement :
./qdrant --config-path config/qdrant_config.yaml
```

Vérifier que Qdrant est accessible : http://localhost:6333/dashboard

### 2. Indexer les Données (Première Utilisation)

```bash
python scripts/migrate_to_qdrant.py
```

Cette étape peut prendre 10-30 minutes selon la taille du dataset.

### 3. Démarrer le Backend FastAPI

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

L'API sera accessible sur : http://localhost:8000

Documentation Swagger : http://localhost:8000/docs

### 4. Démarrer le Frontend Streamlit

Dans un nouveau terminal :

```bash
streamlit run frontend/app.py --server.port 8501
```

L'interface sera accessible sur : http://localhost:8501

### 5. (Optionnel) Démarrer vLLM avec Docker

Si vous utilisez vLLM local :

```bash
docker-compose up -d vllm-service
```

Vérifier les logs :
```bash
docker-compose logs -f vllm-service
```

### 6. (Optionnel) Démarrer Jaeger pour l'Observabilité

```bash
docker-compose up -d jaeger
```

Jaeger UI : http://localhost:16686

**Note** : Si Docker n'est pas disponible, définir `ENABLE_JAEGER_EXPORT=false` dans `.env`.

## 💻 Utilisation

### Interface Web (Streamlit)

1. Ouvrir http://localhost:8501
2. Saisir une question dans le champ de recherche
3. Sélectionner la langue de réponse
4. (Optionnel) Activer/désactiver le reranker
5. Cliquer sur "Rechercher"

### API REST

#### Endpoint : `/chat`

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "k": 5,
    "language": "en",
    "use_reranker": true
  }'
```

#### Endpoint : `/health`

```bash
curl http://localhost:8000/health
```

#### Endpoint : `/examples`

```bash
curl http://localhost:8000/examples
```

### Python Client

```python
import httpx

client = httpx.AsyncClient(base_url="http://localhost:8000")

response = await client.post("/chat", json={
    "query": "What is photosynthesis?",
    "k": 5,
    "language": "en"
})

result = response.json()
print(result["response"])
print(f"Documents récupérés: {len(result['retrieved_documents'])}")
```

## 📚 API Documentation

La documentation complète de l'API est disponible via Swagger UI :

- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc

Voir aussi [docs/API.md](docs/API.md) pour un guide détaillé (à créer).

## 📊 Évaluation

Le projet inclut un script d'évaluation pour mesurer les performances :

```bash
python scripts/evaluate_retrieval.py
```

Métriques calculées :
- **Recall@5** : Rappel à 5 documents
- **Recall@10** : Rappel à 10 documents
- **TTFT** : Time To First Token (latence LLM)

Les résultats sont sauvegardés dans `evaluation_results/`.

Voir [docs/EVALUATION.md](docs/EVALUATION.md) pour plus de détails (à créer).

## 🐳 Déploiement

### Docker Compose

Déployer tous les services :

```bash
docker-compose up -d
```

Services disponibles :
- Backend : http://localhost:8000
- Frontend : http://localhost:8501
- vLLM : http://localhost:8001
- Jaeger : http://localhost:16686

### Kubernetes

Les manifests Kubernetes sont dans `k8s/` :

```bash
kubectl apply -f k8s/
```

Voir [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) pour un guide complet (à créer).

## 🔍 Troubleshooting

### Problèmes Courants

#### Qdrant ne démarre pas

```bash
# Vérifier que le port 6333 est libre
netstat -tuln | grep 6333

# Vérifier les logs
tail -f qdrant.log
```

#### Erreur CUDA/GPU

```bash
# Vérifier que CUDA est installé
python -c "import torch; print(torch.cuda.is_available())"

# Si pas de GPU, les modèles utiliseront CPU automatiquement
```

#### Erreur de connexion à l'API LLM

- Vérifier `LLM_BASE_URL` et `LLM_API_KEY` dans `.env`
- Pour vLLM local, vérifier que le service est démarré : `curl http://localhost:8001/health`
- Pour OpenAI/Mistral, vérifier la validité de la clé API

#### Collection Qdrant vide

```bash
# Réindexer les données
python scripts/migrate_to_qdrant.py
```

### Logs et Debugging

#### Backend

```bash
# Mode debug
uvicorn app.main:app --reload --log-level debug
```

#### Frontend

```bash
# Mode debug
streamlit run frontend/app.py --logger.level=debug
```

#### OpenTelemetry

Si Jaeger n'est pas disponible, désactiver l'export :

```env
ENABLE_JAEGER_EXPORT=false
```

Voir [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) pour plus de solutions.

## 📖 Documentation Additionnelle

- [Configuration Détaillée](docs/CONFIGURATION.md) : Guide complet des variables d'environnement
- [Architecture](docs/ARCHITECTURE.md) : Documentation technique approfondie
- [API Reference](docs/API.md) : Documentation complète de l'API
- [Déploiement](docs/DEPLOYMENT.md) : Guide de déploiement production
- [Évaluation](docs/EVALUATION.md) : Guide d'évaluation des performances
- [Troubleshooting](docs/TROUBLESHOOTING.md) : Guide de résolution de problèmes
- [Contributing](CONTRIBUTING.md) : Guide de contribution au projet

## 🤝 Contribution

Les contributions sont les bienvenues ! Voir [CONTRIBUTING.md](CONTRIBUTING.md) pour les guidelines.

## 📚 Exemples

Des exemples d'utilisation sont disponibles dans le répertoire `examples/` :

- [simple_query.py](examples/simple_query.py) : Exemple basique
- [batch_queries.py](examples/batch_queries.py) : Traitement par batch
- [custom_integration.py](examples/custom_integration.py) : Intégration personnalisée

Voir [examples/README.md](examples/README.md) pour plus de détails.

## 📝 Licence

Ce projet est sous licence **MIT**.  

## 👥 Auteurs

**Gabriel Marie Brisson** – <gabriel@mariebrisson.fr>


## 🙏 Remerciements

- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
- [Qdrant](https://qdrant.tech/)
- [vLLM](https://github.com/vllm-project/vllm)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)

