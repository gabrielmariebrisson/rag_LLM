# Architecture Microservices RAG

## Structure du Projet

```
rag_LLM/
├── app/                          # Backend FastAPI
│   ├── __init__.py
│   ├── main.py                   # Application FastAPI principale
│   ├── schemas.py                # Modèles Pydantic (Request/Response)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py             # Configuration avec pydantic-settings
│   │   └── prompts.py            # Templates de prompts
│   ├── services/
│   │   ├── __init__.py
│   │   ├── translation.py        # Service de traduction
│   │   └── rag.py                # Logique métier RAG
│   └── utils/
│       ├── __init__.py
│       └── text_processing.py    # Nettoyage de texte (regex)
├── frontend/
│   └── app.py                    # Frontend Streamlit stateless
├── faiss_index/                  # Index FAISS (doit exister)
├── requirements.txt
└── .env                          # Variables d'environnement
```

## Démarrage

### 1. Configuration

Créer un fichier `.env` à la racine du projet :

```env
MISTRAL_API_KEY=your-api-key-here
MISTRAL_MODEL_NAME=mistral-tiny-2407
FAISS_INDEX_DIR=faiss_index
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
BACKEND_URL=http://localhost:8000
```

### 2. Installation des dépendances

```bash
pip install -r requirements.txt
```

### 3. Démarrer le Backend FastAPI

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Le backend sera accessible sur `http://localhost:8000`

- Documentation API : `http://localhost:8000/docs`
- Health check : `http://localhost:8000/health`

### 4. Démarrer le Frontend Streamlit

Dans un terminal séparé :

```bash
streamlit run frontend/app.py
```

Le frontend sera accessible sur `http://localhost:8501`

## Points Techniques

### Backend (FastAPI)

- **FAISS CPU-bound** : Tous les appels `similarity_search` sont exécutés dans un `ThreadPoolExecutor` via `asyncio.run_in_executor` pour ne pas bloquer l'event loop
- **Configuration centralisée** : Utilisation de `pydantic-settings` pour la gestion des variables d'environnement
- **Lifecycle management** : Le vectorstore FAISS est chargé une seule fois au démarrage via `@asynccontextmanager`
- **Traduction** : Gérée côté backend, le frontend envoie seulement la langue cible

### Frontend (Streamlit)

- **Stateless** : Aucune logique métier (RAG, traduction) dans le frontend
- **Communication** : Utilise `httpx` pour appeler le backend FastAPI
- **Historique** : Stocké uniquement dans `st.session_state` (frontend uniquement)

## Endpoints API

### POST /chat

Requête :
```json
{
  "query": "What is the capital of France?",
  "k": 5,
  "language": "fr"
}
```

Réponse :
```json
{
  "response": "La capitale de la France est Paris.",
  "retrieved_documents": [...],
  "language": "fr",
  "processing_time": 1.234
}
```

### GET /examples

Réponse :
```json
{
  "examples": ["What is the capital of France?", ...]
}
```

### GET /health

Réponse :
```json
{
  "status": "healthy",
  "vectorstore_loaded": true
}
```

