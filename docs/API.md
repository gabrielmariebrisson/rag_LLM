# 📡 Documentation de l'API REST

Documentation complète de l'API FastAPI pour le système RAG.

## 📋 Table des Matières

- [Vue d'Ensemble](#vue-densemble)
- [Authentification](#authentification)
- [Endpoints](#endpoints)
- [Codes d'Erreur](#codes-derreur)
- [Exemples](#exemples)
- [Rate Limiting](#rate-limiting)

---

## Vue d'Ensemble

L'API REST est accessible sur `http://localhost:8000` (par défaut).

**Base URL** : `http://localhost:8000`  
**Version** : 2.0.0  
**Format** : JSON

### Documentation Interactive

- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc

### Content-Type

Toutes les requêtes et réponses utilisent `application/json`.

---

## Authentification

Actuellement, l'API ne nécessite pas d'authentification. En production, il est recommandé d'ajouter une authentification (API keys, OAuth2, etc.).

---

## Endpoints

### GET /health

Endpoint de santé de l'application.

**Description** : Vérifie l'état de tous les services critiques (Qdrant, vectorstore, reranker, LLM).

**Requête** : Aucun paramètre requis.

**Réponse** :

```json
{
  "status": "healthy",
  "vectorstore_loaded": true,
  "qdrant_host": "localhost",
  "qdrant_port": 6333,
  "collection_name": "squad_collection",
  "reranker_enabled": true,
  "llm_base_url": "http://localhost:8001/v1",
  "llm_model": "casperhansen/llama-3-8b-instruct-awq"
}
```

**Champs de réponse** :

| Champ | Type | Description |
|-------|------|-------------|
| `status` | string | "healthy" si tous les services sont opérationnels |
| `vectorstore_loaded` | boolean | True si le vectorstore est chargé |
| `qdrant_host` | string | Host de Qdrant |
| `qdrant_port` | integer | Port de Qdrant |
| `collection_name` | string | Nom de la collection Qdrant |
| `reranker_enabled` | boolean | État du reranker (activé/désactivé) |
| `llm_base_url` | string | URL de base du LLM ou "Mistral API (legacy)" |
| `llm_model` | string | Nom du modèle LLM configuré |

**Exemple avec curl** :

```bash
curl http://localhost:8000/health
```

**Exemple avec Python** :

```python
import httpx

client = httpx.Client(base_url="http://localhost:8000")
response = client.get("/health")
print(response.json())
```

**Codes de réponse** :

- `200 OK` : Service opérationnel
- `503 Service Unavailable` : Service non disponible (si services non chargés)

---

### POST /chat

Endpoint principal pour les requêtes RAG.

**Description** : Traite une question utilisateur et retourne une réponse générée avec les documents sources utilisés.

**Pipeline** :
1. Hybrid Search (dense + sparse) dans Qdrant
2. (Optionnel) Reranking avec Cross-Encoder
3. Formatage du prompt avec contexte
4. Génération de réponse via LLM
5. Nettoyage du texte généré
6. Traduction si nécessaire

**Requête** :

```json
{
  "query": "What is photosynthesis?",
  "k": 5,
  "language": "en",
  "use_reranker": true
}
```

**Paramètres** :

| Paramètre | Type | Requis | Défaut | Description |
|-----------|------|--------|--------|-------------|
| `query` | string | Oui | - | Question de l'utilisateur |
| `k` | integer | Non | 5 | Nombre de documents à récupérer (1-20) |
| `language` | string | Non | "en" | Langue cible de la réponse (code ISO : "en", "fr", "es", etc.) |
| `use_reranker` | boolean | Non | null | Override pour activer/désactiver reranker (null = utilise config global) |

**Langues supportées** :

- `en` : English
- `fr` : Français
- `es` : Español
- `de` : Deutsch
- `it` : Italiano
- `pt` : Português
- `ja` : 日本語
- `zh-CN` : 中文
- `ar` : العربية
- `ru` : Русский

**Réponse** :

```json
{
  "response": "Photosynthesis is the process by which plants convert light energy into chemical energy...",
  "retrieved_documents": [
    {
      "page_content": "Title: Biology\nContext: Photosynthesis is...\nAnswer: The process of converting light energy...",
      "metadata": {
        "title": "Biology",
        "context": "Photosynthesis is...",
        "answer": "The process of converting light energy..."
      }
    },
    {
      "page_content": "...",
      "metadata": {...}
    }
  ],
  "language": "en",
  "processing_time": 1.234
}
```

**Champs de réponse** :

| Champ | Type | Description |
|-------|------|-------------|
| `response` | string | Réponse générée et traduite |
| `retrieved_documents` | array | Liste des documents sources utilisés |
| `retrieved_documents[].page_content` | string | Contenu du document récupéré |
| `retrieved_documents[].metadata` | object | Métadonnées associées au document |
| `language` | string | Langue de la réponse générée |
| `processing_time` | float | Temps d'exécution en secondes |

**Exemple avec curl** :

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

**Exemple avec Python** :

```python
import httpx

client = httpx.AsyncClient(base_url="http://localhost:8000")

async def ask_question(query: str, language: str = "en"):
    response = await client.post("/chat", json={
        "query": query,
        "k": 5,
        "language": language,
        "use_reranker": True
    })
    result = response.json()
    print(f"Question: {query}")
    print(f"Réponse: {result['response']}")
    print(f"Temps: {result['processing_time']}s")
    print(f"Documents: {len(result['retrieved_documents'])}")
    return result

# Utilisation
import asyncio
result = asyncio.run(ask_question("What is photosynthesis?", "en"))
```

**Exemple avec JavaScript (fetch)** :

```javascript
async function askQuestion(query, language = "en") {
  const response = await fetch("http://localhost:8000/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query: query,
      k: 5,
      language: language,
      use_reranker: true
    })
  });
  
  const result = await response.json();
  console.log("Question:", query);
  console.log("Réponse:", result.response);
  console.log("Temps:", result.processing_time, "s");
  console.log("Documents:", result.retrieved_documents.length);
  return result;
}

// Utilisation
askQuestion("What is the capital of France?", "fr");
```

**Codes de réponse** :

- `200 OK` : Requête traitée avec succès
- `400 Bad Request` : Paramètres invalides (validation Pydantic échouée)
- `500 Internal Server Error` : Erreur lors du traitement
- `503 Service Unavailable` : Services non chargés (vectorstore, embedding_service, llm_client)

**Erreurs possibles** :

```json
{
  "detail": "Services not loaded. Please check server logs."
}
```

```json
{
  "detail": "Error processing request: [message d'erreur détaillé]"
}
```

---

### GET /examples

Retourne une liste d'exemples de questions.

**Description** : Fournit des exemples de questions pour l'interface utilisateur. Essaie de charger des exemples aléatoires depuis le dataset SQuAD, sinon retourne des exemples hardcodés.

**Requête** : Aucun paramètre requis.

**Réponse** :

```json
{
  "examples": [
    "What is the capital of France?",
    "Who invented the telephone?",
    "When did World War II end?",
    "What is photosynthesis?",
    "Who wrote Romeo and Juliet?",
    "..."
  ]
}
```

**Champs de réponse** :

| Champ | Type | Description |
|-------|------|-------------|
| `examples` | array[string] | Liste de questions d'exemple (généralement 10) |

**Exemple avec curl** :

```bash
curl http://localhost:8000/examples
```

**Exemple avec Python** :

```python
import httpx

client = httpx.Client(base_url="http://localhost:8000")
response = client.get("/examples")
examples = response.json()["examples"]
print(f"Found {len(examples)} example questions")
for example in examples:
    print(f"  - {example}")
```

**Codes de réponse** :

- `200 OK` : Exemples retournés avec succès

**Notes** :

- Si le fichier `data/raw/squad_2.0/train.csv` existe, retourne 10 questions aléatoires
- Sinon, retourne une liste d'exemples hardcodés en anglais
- Les exemples changent à chaque appel si le CSV est utilisé

---

## Codes d'Erreur

L'API utilise les codes de statut HTTP standards :

| Code | Signification | Description |
|------|---------------|-------------|
| `200` | OK | Requête réussie |
| `400` | Bad Request | Paramètres invalides ou validation échouée |
| `500` | Internal Server Error | Erreur serveur lors du traitement |
| `503` | Service Unavailable | Services non disponibles (Qdrant, LLM, etc.) |

### Format des Erreurs

Les erreurs sont retournées au format suivant :

```json
{
  "detail": "Message d'erreur détaillé"
}
```

**Exemples d'erreurs** :

```json
{
  "detail": "Services not loaded. Please check server logs."
}
```

```json
{
  "detail": "Error processing request: Connection refused to Qdrant"
}
```

```json
{
  "detail": [
    {
      "loc": ["body", "query"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## Exemples

### Exemple Complet : Workflow RAG

```python
import httpx
import asyncio

async def complete_rag_workflow():
    """Workflow complet : health check, exemples, puis question."""
    client = httpx.AsyncClient(base_url="http://localhost:8000")
    
    # 1. Vérifier la santé
    health = await client.get("/health")
    print("Health:", health.json()["status"])
    
    # 2. Récupérer des exemples
    examples = await client.get("/examples")
    example_questions = examples.json()["examples"]
    print(f"\nExemples disponibles: {len(example_questions)}")
    
    # 3. Poser une question
    question = example_questions[0]  # Première question d'exemple
    response = await client.post("/chat", json={
        "query": question,
        "k": 5,
        "language": "en",
        "use_reranker": True
    })
    
    result = response.json()
    print(f"\nQuestion: {question}")
    print(f"Réponse: {result['response']}")
    print(f"Temps de traitement: {result['processing_time']:.3f}s")
    print(f"Documents utilisés: {len(result['retrieved_documents'])}")
    
    # Afficher les sources
    for i, doc in enumerate(result['retrieved_documents'], 1):
        print(f"\nDocument {i}:")
        print(f"  Contenu: {doc['page_content'][:100]}...")
        print(f"  Métadonnées: {doc['metadata']}")
    
    await client.aclose()

# Exécuter
asyncio.run(complete_rag_workflow())
```

### Exemple : Traduction Multi-langue

```python
import httpx
import asyncio

async def multilingual_example():
    """Exemple avec traduction dans différentes langues."""
    client = httpx.AsyncClient(base_url="http://localhost:8000")
    
    question = "What is the capital of France?"
    languages = ["en", "fr", "es", "de"]
    
    for lang in languages:
        response = await client.post("/chat", json={
            "query": question,
            "k": 3,
            "language": lang
        })
        result = response.json()
        print(f"\n[{lang}] {result['response']}")
    
    await client.aclose()

asyncio.run(multilingual_example())
```

### Exemple : Comparaison avec/sans Reranker

```python
import httpx
import asyncio

async def compare_reranker():
    """Compare les résultats avec et sans reranker."""
    client = httpx.AsyncClient(base_url="http://localhost:8000")
    
    question = "What is photosynthesis?"
    
    # Sans reranker
    response_no_rerank = await client.post("/chat", json={
        "query": question,
        "k": 5,
        "use_reranker": False
    })
    
    # Avec reranker
    response_with_rerank = await client.post("/chat", json={
        "query": question,
        "k": 5,
        "use_reranker": True
    })
    
    print("Sans reranker:")
    print(f"  Temps: {response_no_rerank.json()['processing_time']:.3f}s")
    print(f"  Réponse: {response_no_rerank.json()['response'][:100]}...")
    
    print("\nAvec reranker:")
    print(f"  Temps: {response_with_rerank.json()['processing_time']:.3f}s")
    print(f"  Réponse: {response_with_rerank.json()['response'][:100]}...")
    
    await client.aclose()

asyncio.run(compare_reranker())
```

---

## Rate Limiting

Actuellement, l'API n'implémente pas de rate limiting. En production, il est recommandé d'ajouter :

- Limite par IP
- Limite par utilisateur/API key
- Throttling pour éviter la surcharge

**Exemple avec middleware FastAPI** :

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: Request, ...):
    ...
```

---

## Observabilité

L'API est instrumentée avec OpenTelemetry pour le tracing distribué.

### Spans OpenTelemetry

Les spans suivants sont créés automatiquement :

- `rag.request` : Requête complète RAG
- `rag.retrieval` : Recherche hybride dans Qdrant
- `rag.reranking` : Reranking des résultats
- `llm.generation` : Génération de réponse LLM

### Attributs des Spans

- `http.method` : Méthode HTTP
- `http.route` : Route de l'endpoint
- `query` : Question de l'utilisateur
- `k` : Nombre de documents
- `language` : Langue cible
- `use_reranker` : État du reranker
- `processing_time_ms` : Temps de traitement en millisecondes
- `response_length` : Longueur de la réponse
- `num_candidates` : Nombre de candidats récupérés
- `num_reranked` : Nombre de documents rerankés

### Accès aux Traces

Si Jaeger est configuré, accéder à : http://localhost:16686

---

## Ressources Additionnelles

- [README.md](../README.md) : Guide de démarrage rapide
- [CONFIGURATION.md](CONFIGURATION.md) : Guide de configuration
- [ARCHITECTURE.md](ARCHITECTURE.md) : Documentation technique approfondie

