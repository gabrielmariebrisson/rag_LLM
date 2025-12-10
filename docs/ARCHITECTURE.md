# 🏗️ Architecture Technique

Documentation approfondie de l'architecture du système RAG.

## 📋 Table des Matières

- [Vue d'Ensemble](#vue-densemble)
- [Arborescence du Projet](#arborescence-du-projet)
- [Architecture Système](#architecture-système)
- [Pipeline RAG](#pipeline-rag)
- [Composants](#composants)
- [Décisions Techniques](#décisions-techniques)
- [Flux de Données](#flux-de-données)
- [Performance et Scalabilité](#performance-et-scalabilité)

---

## Vue d'Ensemble

Le système RAG est construit avec une architecture microservices modulaire :

- **Frontend** : Streamlit (interface utilisateur web)
- **Backend** : FastAPI (API REST et orchestration)
- **Vector Store** : Qdrant (base de données vectorielle)
- **LLM** : vLLM local, OpenAI API, ou Mistral API
- **Observabilité** : OpenTelemetry + Jaeger

---

## Arborescence du Projet

### Structure Complète

```
rag_LLM/
├── app/                          # Application FastAPI principale
│   ├── __init__.py
│   ├── main.py                   # Point d'entrée FastAPI (endpoints, lifespan)
│   ├── schemas.py                # Modèles Pydantic (ChatRequest, ChatResponse)
│   │
│   ├── core/                     # Configuration et prompts
│   │   ├── __init__.py
│   │   ├── config.py            # Settings (pydantic-settings)
│   │   ├── paths.py             # Chemins de fichiers
│   │   └── prompts.py           # Templates de prompts (system/user)
│   │
│   ├── services/                 # Services métier
│   │   ├── __init__.py
│   │   ├── embeddings.py        # EmbeddingService (dense + sparse)
│   │   ├── llm_client.py        # LLMClient (agnostique: vLLM/OpenAI/Mistral)
│   │   ├── rag.py               # Orchestration RAG (retrieve + generate)
│   │   ├── reranker.py          # RerankerService (Cross-Encoder)
│   │   └── translation.py       # TranslationService (GoogleTranslator)
│   │
│   ├── utils/                    # Utilitaires
│   │   ├── __init__.py
│   │   └── text_processing.py   # Nettoyage de texte (regex)
│   │
│   └── vectorstores/             # Gestion des bases vectorielles
│       ├── __init__.py
│       └── qdrant_store.py      # QdrantVectorStore (hybrid search)
│
├── frontend/                     # Interface utilisateur Streamlit
│   └── app.py                   # Application Streamlit (stateless)
│
├── scripts/                      # Scripts utilitaires
│   ├── convert_squad_json_to_csv.py  # Conversion SQuAD JSON → CSV
│   ├── evaluate_retrieval.py         # Évaluation Recall@5 (avec/sans reranker)
│   └── migrate_to_qdrant.py         # Migration données → Qdrant
│
├── tests/                        # Tests unitaires et d'intégration
│   ├── conftest.py              # Configuration pytest
│   └── test_search.py           # Tests de recherche
│
├── docs/                         # Documentation
│   ├── API.md                   # Documentation API REST
│   ├── ARCHITECTURE.md          # Architecture technique (ce fichier)
│   ├── CONFIGURATION.md          # Guide de configuration
│   ├── DEPLOYMENT.md             # Guide de déploiement
│   ├── EVALUATION.md             # Guide d'évaluation
│   └── TROUBLESHOOTING.md        # Dépannage
│
├── docker/                       # Dockerfiles
│   ├── Dockerfile.backend       # Image Docker backend FastAPI
│   └── Dockerfile.frontend      # Image Docker frontend Streamlit
│
├── k8s/                          # Manifests Kubernetes
│   ├── backend.yaml             # Deployment + Service backend
│   ├── ingress.yaml              # Ingress (routing)
│   └── vllm.yaml                 # Deployment + Service vLLM (GPU)
│
├── config/                       # Fichiers de configuration
│   └── qdrant_config.yaml       # Configuration Qdrant
│
├── data/                         # Données et résultats
│   └── results/
│       └── evaluation_results/  # Résultats d'évaluation
│           ├── evaluation_details.csv
│           ├── evaluation_results.json
│           └── evaluation_summary.csv
│
├── templates/                    # Templates et assets
│   └── assets/
│       └── architecture-diagram.png
│
├── archive/                      # Code legacy (POC Streamlit)
│   ├── azure_rag.py
│   ├── rag_LLM_web.py
│   └── README.md
│
├── bin/                          # Scripts shell
│   └── start_qdrant.sh          # Script de démarrage Qdrant
│
├── docker-compose.yml            # Orchestration Docker (vLLM, Jaeger, Qdrant)
├── requirements.txt              # Dépendances Python
├── pyproject.toml                # Configuration projet Python
├── pytest.ini                   # Configuration pytest
├── LICENSE                      # Licence
├── CONTRIBUTING.md              # Guide de contribution
└── README.md                    # Documentation principale
```

### Description des Répertoires Principaux

#### `app/`
**Rôle** : Code source de l'application FastAPI backend.

- **`main.py`** : Point d'entrée FastAPI, définit les endpoints (`/chat`, `/health`, `/examples`) et le lifecycle (`lifespan`).
- **`schemas.py`** : Modèles Pydantic pour validation des requêtes/réponses API.
- **`core/`** : Configuration centralisée et templates de prompts.
- **`services/`** : Services métier (embeddings, LLM, RAG, reranking, traduction).
- **`utils/`** : Fonctions utilitaires (nettoyage de texte).
- **`vectorstores/`** : Abstraction de la base vectorielle (Qdrant).

#### `frontend/`
**Rôle** : Interface utilisateur Streamlit (stateless).

- **`app.py`** : Application Streamlit qui communique avec le backend via HTTP.

#### `scripts/`
**Rôle** : Scripts utilitaires pour migration, évaluation, conversion.

- **`migrate_to_qdrant.py`** : Réindexe les données SQuAD dans Qdrant.
- **`evaluate_retrieval.py`** : Calcule Recall@5 avec/sans reranker.
- **`convert_squad_json_to_csv.py`** : Convertit SQuAD JSON en CSV.

#### `tests/`
**Rôle** : Tests unitaires et d'intégration.

- Utilise `pytest` pour l'exécution.
- **`conftest.py`** : Fixtures partagées.
- **`test_search.py`** : Tests de recherche.

#### `docs/`
**Rôle** : Documentation complète du projet.

- Guides d'architecture, configuration, déploiement, API, évaluation, dépannage.

#### `docker/` et `k8s/`
**Rôle** : Configuration de déploiement.

- **`docker/`** : Dockerfiles pour backend et frontend.
- **`k8s/`** : Manifests Kubernetes pour production.

#### `archive/`
**Rôle** : Code legacy du POC Streamlit original.

- Conservé pour référence historique.

---

## Architecture Système

### Diagramme d'Architecture Haut Niveau

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend Streamlit                       │
│                         (Port 8501)                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Interface Utilisateur :                                  │  │
│  │  - Champ de recherche                                     │  │
│  │  - Sélecteur de langue                                    │  │
│  │  - Affichage résultats                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP/REST
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    Backend FastAPI                               │
│                    (Port 8000)                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  API Endpoints:                                          │  │
│  │  - POST /chat      (Requêtes RAG)                        │  │
│  │  - GET  /health    (Santé)                               │  │
│  │  - GET  /examples  (Exemples questions)                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Services Internes:                                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │  │
│  │  │ Embedding    │  │ Reranker     │  │ LLM Client   │  │  │
│  │  │ Service      │  │ Service      │  │              │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │  │
│  │  │ Qdrant       │  │ Translation  │  │ OpenTelemetry│  │  │
│  │  │ VectorStore  │  │ Service      │  │ Tracing      │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────┬──────────────┬──────────────┬───────────────────────┘
            │              │              │
    ┌───────▼───────┐ ┌───▼─────┐ ┌─────▼──────┐
    │   Qdrant      │ │  vLLM   │ │   Jaeger   │
    │   :6333       │ │  :8001  │ │  :16686    │
    │               │ │         │ │            │
    │ Vector DB     │ │ LLM     │ │ Tracing    │
    │ Hybrid Search │ │ Server  │ │ UI         │
    └───────────────┘ └─────────┘ └────────────┘
                            │
                    ┌───────▼───────┐
                    │     GPU       │
                    │   (CUDA)      │
                    └───────────────┘
```

---

## Pipeline RAG

### Flux de Traitement Complet

```
Requête Utilisateur
        │
        ▼
┌───────────────────┐
│  1. Embedding     │  Génération embeddings hybrides (dense + sparse)
│     Query         │  - Dense: SentenceTransformer (BGE-large, 1024 dimensions, GPU optimisé)
└────────┬──────────┘  - Sparse: SparseEncoder SPLADE (prithivida/Splade_PP_en_v1, GPU optimisé)
         │
         ▼
┌───────────────────┐
│  2. Hybrid Search │  Recherche dans Qdrant
│     (Qdrant)      │  - Recherche dense (vecteurs)
└────────┬──────────┘  - Recherche sparse (mots-clés)
         │             - Fusion RRF (Reciprocal Rank Fusion)
         │
         ▼
┌───────────────────┐
│  3. Reranking     │  Réordonnancement (optionnel)
│  (Cross-Encoder)  │  - Score query-document avec BGE-reranker-v2-m3
└────────┬──────────┘  - Tri par score décroissant
         │             - Sélection top K
         │
         ▼
┌───────────────────┐
│  4. Prompt        │  Formatage du contexte
│     Formatting    │  - Concaténation des documents
└────────┬──────────┘  - Ajout de la question
         │             - System prompt + User prompt
         │
         ▼
┌───────────────────┐
│  5. LLM           │  Génération de réponse
│     Generation    │  - Appel API (vLLM/OpenAI/Mistral)
└────────┬──────────┘  - Streaming ou non-streaming
         │
         ▼
┌───────────────────┐
│  6. Text          │  Nettoyage du texte
│     Cleaning      │  - Retrait des artefacts
└────────┬──────────┘  - Formatage
         │
         ▼
┌───────────────────┐
│  7. Translation   │  Traduction (si nécessaire)
│     (Optional)    │  - deep-translator
└────────┬──────────┘  - 10 langues supportées
         │
         ▼
    Réponse Finale
```

### Détails des Étapes

#### 1. Embedding Query

**Composant** : `EmbeddingService`

**Traitement** :
- Génère un embedding dense via `SentenceTransformer` (BGE-large, 1024 dimensions, GPU optimisé)
- Génère un embedding sparse via `SparseEncoder` SPLADE (indices de mots importants, GPU optimisé)
- Les deux embeddings sont utilisés pour la recherche hybride
- Les modèles sont automatiquement chargés sur GPU (CUDA) si disponible (optimisé RTX 3090)

**Performance** :
- Dense : ~5-20ms sur GPU RTX 3090 (vs ~50-200ms sur CPU)
- Sparse : ~10-40ms sur GPU RTX 3090 (vs ~100-400ms sur CPU)

#### 2. Hybrid Search (Qdrant)

**Composant** : `QdrantVectorStore`

**Algorithme** : RRF (Reciprocal Rank Fusion)

**Formule RRF** :
```
score = 1/(k + rank_dense) + 1/(k + rank_sparse)
```

où `k = 60` (paramètre de fusion).

**Avantages** :
- Combine les forces de la recherche sémantique (dense) et lexicale (sparse)
- Meilleur rappel que la recherche dense seule
- Plus précise que la recherche sparse seule

#### 3. Reranking

**Composant** : `RerankerService`

**Modèle** : Cross-Encoder (BAAI/bge-reranker-v2-m3)

**Fonctionnement** :
- Prend la query et chaque document en entrée
- Génère un score de pertinence
- Tri par score décroissant

**Impact** :
- Améliore la précision (Recall@10 +3-5% observé)
- Ajoute ~20-50ms de latence
- Nécessite CPU/GPU pour le modèle Cross-Encoder

#### 4. Prompt Formatting

**Composant** : `app/core/prompts.py`

**Format** :
```
System: "You are a helpful assistant..."

User: "Context:
{document_1}

{document_2}

...

Question: {query}"
```

#### 5. LLM Generation

**Composant** : `LLMClient`

**Supporte** :
- vLLM local (OpenAI-compatible API)
- OpenAI API
- Mistral API (legacy)

**Configuration** : Via `LLM_BASE_URL` et `LLM_API_KEY`

#### 6. Text Cleaning

**Composant** : `app/utils/text_processing.py`

**Traitements** :
- Retrait des préfixes/suffixes indésirables
- Normalisation des espaces
- Formatage

#### 7. Translation

**Composant** : `app/services/translation.py`

**Bibliothèque** : `deep-translator`

**Langues supportées** : 10 langues (en, fr, es, de, it, pt, ja, zh-CN, ar, ru)

---

## Composants

### 1. EmbeddingService

**Responsabilité** : Génération d'embeddings hybrides (dense + sparse)

**Modèles** :
- **Dense** : SentenceTransformer (BAAI/bge-large-en-v1.5 par défaut, 1024 dimensions, GPU optimisé)
- **Sparse** : SparseEncoder (prithivida/Splade_PP_en_v1 par défaut, GPU optimisé)

**Méthodes** :
- `embed_hybrid(texts: List[str]) -> Tuple[List[np.ndarray], List[dict]]`
  - Génère embeddings dense et sparse pour une liste de textes

**GPU/CPU** : Utilise GPU automatiquement si disponible

**Cache** : Modèles chargés une fois en mémoire

### 2. QdrantVectorStore

**Responsabilité** : Gestion de la base de données vectorielle

**Fonctionnalités** :
- Recherche hybride (dense + sparse)
- Fusion RRF
- Gestion des collections
- Ajout de documents

**Méthodes principales** :
- `hybrid_search(query: str, top_k: int) -> List[Dict]`
- `add_documents(documents: List[str], metadatas: List[Dict])`
- `initialize_collection(dense_dim: int)`

**Configuration** :
- Distance : COSINE pour dense
- Vecteurs : "dense" (named vector) et "sparse" (named sparse vector)

### 3. RerankerService

**Responsabilité** : Réordonnancement des résultats avec Cross-Encoder

**Modèle** : BAAI/bge-reranker-v2-m3 (Cross-Encoder via sentence-transformers)

**Méthodes** :
- `rerank(query: str, documents: List[str], top_k: int) -> List[Tuple[str, float]]`

**Performance** :
- Latence : ~20-50ms pour 20 documents
- CPU/GPU : Utilise CPU par défaut, GPU si disponible

### 4. LLMClient

**Responsabilité** : Client agnostique pour différents backends LLM

**Architecture** :
- Détecte automatiquement le backend selon `LLM_BASE_URL`
- Supporte OpenAI API, vLLM local, Mistral API

**Méthodes** :
- `generate(messages: List[dict], model: str, stream: bool) -> str`

**Streaming** : Supporté pour vLLM et OpenAI API

### 5. Translation Service

**Responsabilité** : Traduction des réponses

**Bibliothèque** : deep-translator

**Usage** : Traduction post-génération si `language != "en"`

---

## Décisions Techniques

### Pourquoi Qdrant ?

**Raisons** :
1. ✅ **Recherche Hybride Native** : Support intégré dense + sparse
2. ✅ **Performance** : Optimisé pour la recherche vectorielle
3. ✅ **Scalabilité** : Support clustering et distribution
4. ✅ **API Async** : Compatible avec FastAPI async
5. ✅ **Open Source** : Pas de dépendance vendor lock-in

**Alternatives considérées** :
- FAISS : Pas de recherche sparse native, pas de distribution
- Pinecone : Service cloud payant, moins de contrôle
- Weaviate : Plus complexe, overhead pour ce use case

### Pourquoi Hybrid Search ?

**Recherche Dense (Vecteurs)** :
- ✅ Capte la sémantique
- ✅ Gère les synonymes
- ⚠️ Peut rater des mots-clés précis

**Recherche Sparse (Mots-clés)** :
- ✅ Précision sur les termes exacts
- ✅ Meilleur pour les entités nommées
- ⚠️ Ne capture pas la sémantique

**Hybrid (RRF)** :
- ✅ Combine les deux approches
- ✅ Meilleur rappel global
- ✅ Plus robuste aux variations de formulation

**Preuve** : Amélioration observée de +3-5% Recall@10 vs dense seul

### Pourquoi Reranking ?

**Justification** :
- La recherche hybride retourne de bons candidats
- Le reranking améliore l'ordre (précision)
- Cross-Encoder est spécialisé pour scoring query-document

**Trade-off** :
- +3-5% Recall@10
- +20-50ms latence
- Utilisation CPU/GPU

**Alternative** : Désactiver avec `USE_RERANKER=false` si latence critique

### Pourquoi FastAPI ?

**Raisons** :
1. ✅ **Async Native** : Performances élevées
2. ✅ **Documentation Auto** : OpenAPI/Swagger intégré
3. ✅ **Type Safety** : Validation Pydantic
4. ✅ **Modern** : Basé sur Python 3.6+ type hints
5. ✅ **Écosystème** : Compatible avec OpenTelemetry, uvicorn

### Pourquoi Streamlit ?

**Raisons** :
1. ✅ **Rapidité** : Interface en Python pur
2. ✅ **Intégration** : Communique facilement avec FastAPI
3. ✅ **UI Moderne** : Composants interactifs prêts à l'emploi
4. ✅ **Déploiement** : Facile à déployer

### Pourquoi OpenTelemetry ?

**Raisons** :
1. ✅ **Standard** : Standard ouvert (CNCF)
2. ✅ **Vendor Agnostic** : Pas de lock-in
3. ✅ **Tracing Distribué** : Suivre les requêtes end-to-end
4. ✅ **Debugging** : Identifier les goulots d'étranglement

**Backend** : Jaeger (optionnel, peut être désactivé)

### Pourquoi Multi-LLM Support ?

**Flexibilité** :
- **vLLM Local** : Latence optimale, pas de coût API
- **OpenAI API** : Modèles performants, pas de GPU requis
- **Mistral API** : Alternative (legacy)

**Agnosticisme** : Permet de changer de provider sans modifier le code

---

## Flux de Données

### Requête /chat Complète

```
1. Client → FastAPI
   POST /chat
   {
     "query": "What is photosynthesis?",
     "k": 5,
     "language": "en"
   }

2. FastAPI → EmbeddingService
   embed_hybrid(["What is photosynthesis?"])
   → (dense_vector, sparse_vector)

3. FastAPI → QdrantVectorStore
   hybrid_search(query, top_k=20)
   → Recherche dense + sparse
   → Fusion RRF
   → 20 documents

4. FastAPI → RerankerService
   rerank(query, documents, top_k=5)
   → 5 documents réordonnés

5. FastAPI → LLMClient
   generate(messages, model)
   → Réponse générée

6. FastAPI → TranslationService (si nécessaire)
   translate(response, target_lang)
   → Réponse traduite

7. FastAPI → Client
   {
     "response": "...",
     "retrieved_documents": [...],
     "language": "en",
     "processing_time": 1.234
   }
```

### Observabilité (OpenTelemetry)

```
Span racine: rag.request
  ├─ Span: rag.retrieval
  │    └─ hybrid_search()
  ├─ Span: rag.reranking
  │    └─ rerank()
  └─ Span: llm.generation
       └─ generate()
```

Tous les spans incluent :
- Attributs (query, k, language, etc.)
- Durée
- Status (OK, ERROR)
- Exceptions (si erreur)

---

## Performance et Scalabilité

### Latences Typiques

| Composant | Latence | Notes |
|-----------|---------|-------|
| Embedding (dense) | 10-50ms | GPU: ~10ms, CPU: ~50ms |
| Embedding (sparse) | 20-100ms | GPU: ~20ms, CPU: ~100ms |
| Hybrid Search | 20-100ms | Dépend de la taille de la collection |
| Reranking (20 docs) | 20-50ms | CPU: ~50ms, GPU: ~20ms |
| LLM Generation | 500-3000ms | vLLM local: ~500ms, API externe: ~3000ms |
| Translation | 200-500ms | Dépend de la langue |
| **Total (sans reranker)** | **550-3650ms** | |
| **Total (avec reranker)** | **570-3700ms** | |

### Optimisations

1. **Cache des Modèles** : Embeddings et reranker chargés une fois
2. **GPU** : Utilisation automatique si disponible
3. **Async** : FastAPI async pour gérer plusieurs requêtes simultanées
4. **Batch Processing** : Embeddings en batch si plusieurs documents

### Scalabilité

**Actuelle** :
- Monolithique : Backend + Services dans un seul processus
- Stateless : Chaque requête est indépendante

**Production Recommandée** :
- **Horizontal Scaling** : Plusieurs instances FastAPI (load balancer)
- **Qdrant Cluster** : Distribution pour grandes collections
- **vLLM Multi-GPU** : Scaling vertical pour LLM
- **Redis Cache** : Cache des embeddings fréquents

**Déploiement Kubernetes** :
- Backend : Deployment avec replicas
- vLLM : Deployment séparé avec GPU
- Qdrant : StatefulSet ou service externe
- Auto-scaling : HPA basé sur CPU/mémoire

---

## Sécurité

### Recommandations Production

1. **Authentification** : API keys ou OAuth2
2. **HTTPS** : TLS/SSL obligatoire
3. **Rate Limiting** : Protection contre les abus
4. **Validation Input** : Pydantic validation (déjà implémenté)
5. **CORS** : Restreindre les origines autorisées
6. **Secrets** : Variables d'environnement sécurisées

### Variables Sensibles

- `LLM_API_KEY` : Clé API OpenAI/Mistral
- `MISTRAL_API_KEY` : Clé API Mistral (legacy)
- `QDRANT_API_KEY` : Clé API Qdrant Cloud (si utilisé)

**Stockage** : Utiliser un gestionnaire de secrets (Vault, AWS Secrets Manager, etc.)

---

## Ressources Additionnelles

- [README.md](../README.md) : Guide de démarrage rapide
- [CONFIGURATION.md](CONFIGURATION.md) : Guide de configuration
- [API.md](API.md) : Documentation complète de l'API

---

**Version** : 2.0.0  
**Dernière mise à jour** : 2025-01-27

