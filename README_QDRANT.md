# Migration vers Qdrant avec Hybrid Search et Reranking

## Architecture

Le système utilise maintenant :
- **Qdrant** : Base de données vectorielle avec support recherche hybride
- **FastEmbed** : Embeddings denses (sentence-transformers/all-MiniLM-L6-v2)
- **SPLADE** : Embeddings sparse (prunebert-base-uncased-6-minilayer)
- **CrossEncoder** : Reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)

## Installation

### 1. Installer Qdrant

**Option A : Docker (Recommandé)**
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

**Option B : Binaire local**
Télécharger depuis https://qdrant.tech/documentation/guides/installation/

### 2. Installer les dépendances Python

```bash
pip install -r requirements.txt
```

Les nouvelles dépendances incluent :
- `qdrant-client[async]` : Client Qdrant asynchrone
- `fastembed` : Embeddings rapides
- `transformers` : Pour SPLADE
- `torch` : Backend pour modèles ML
- `sentence-transformers` : Pour CrossEncoder

## Configuration

Ajouter dans votre fichier `.env` :

```env
# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=squad_collection

# Reranker
USE_RERANKER=true
RERANKER_TOP_K=20

# Modèles
DENSE_MODEL=sentence-transformers/all-MiniLM-L6-v2
SPARSE_MODEL=prunebert-base-uncased-6-minilayer
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

## Migration des données

### Étape 1 : Migrer les données vers Qdrant

```bash
python scripts/migrate_to_qdrant.py
```

Ce script :
- Lit le CSV SQuAD (`squad_2.0/train.csv`)
- Génère les embeddings dense + sparse pour chaque document
- Indexe dans Qdrant avec métadonnées

**Note** : La première exécution peut prendre du temps car les modèles doivent être téléchargés.

### Étape 2 : Vérifier l'indexation

Vérifier que Qdrant contient les données :
```bash
curl http://localhost:6333/collections/squad_collection
```

## Démarrage

### Backend FastAPI

```bash
uvicorn app.main:app --reload --port 8000
```

Le backend :
- Se connecte à Qdrant au démarrage
- Vérifie/crée la collection si nécessaire
- Charge les modèles d'embeddings et reranker

### Frontend Streamlit

```bash
streamlit run frontend/app.py
```

## Évaluation des performances

### Script d'évaluation Recall@5

```bash
python scripts/evaluate_retrieval.py
```

Ce script :
- Charge 50 questions du dataset SQuAD
- Teste la recherche **sans reranker**
- Teste la recherche **avec reranker**
- Calcule le **Recall@5** pour chaque configuration
- Affiche les métriques :
  - Recall@5 sans reranker
  - Recall@5 avec reranker
  - Amélioration en %
  - Latence moyenne
- Exporte les résultats dans `evaluation_results/`

### Résultats attendus

Le reranker devrait améliorer le Recall@5 de **5-15%** selon le dataset, avec un overhead de latence de **50-100ms**.

## Utilisation de l'API

### Endpoint `/chat`

**Requête :**
```json
{
  "query": "What is the capital of France?",
  "k": 5,
  "language": "en",
  "use_reranker": true
}
```

**Réponse :**
```json
{
  "response": "The capital of France is Paris.",
  "retrieved_documents": [...],
  "language": "en",
  "processing_time": 1.234
}
```

### Paramètre `use_reranker`

- `null` ou non fourni : Utilise la configuration globale (`USE_RERANKER`)
- `true` : Force l'activation du reranker
- `false` : Force la désactivation du reranker

## Recherche Hybride

Le système combine :
1. **Recherche Dense** : Similarité sémantique via embeddings denses
2. **Recherche Sparse** : Matching lexical via SPLADE
3. **Fusion RRF** : Reciprocal Rank Fusion pour combiner les résultats
4. **Reranking** (optionnel) : CrossEncoder pour réordonner les top résultats

## Dépannage

### Qdrant ne démarre pas

Vérifier que le port 6333 est libre :
```bash
lsof -i :6333
```

### Erreur "Collection not found"

Exécuter le script de migration :
```bash
python scripts/migrate_to_qdrant.py
```

### Modèles trop lents

Les modèles sont chargés en mémoire au premier appel. Pour accélérer :
- Utiliser GPU si disponible (CUDA)
- Réduire `RERANKER_TOP_K` pour moins de documents à reranker
- Désactiver le reranker si la latence est critique

### Erreur SPLADE

Le modèle `prunebert-base-uncased-6-minilayer` doit être téléchargé depuis HuggingFace. Vérifier votre connexion internet et les permissions d'écriture.

## Performance

### Latence typique (sans GPU)

- Recherche hybride seule : ~50-100ms
- Avec reranker : ~150-200ms
- Génération LLM : ~500-1000ms
- **Total** : ~700-1300ms par requête

### Amélioration Recall@5

Avec reranker activé, on observe généralement :
- **+5-15%** de Recall@5
- Meilleure pertinence des documents récupérés
- Réponses LLM plus précises

## Notes techniques

1. **SPLADE** : Génère des vecteurs sparse de taille vocabulaire (~30k dimensions), mais seulement ~100-500 valeurs non-nulles par document
2. **RRF** : Utilise k=60 pour la fusion (paramètre standard)
3. **Reranker** : Traite les top 20 résultats de la recherche hybride, puis retourne les top k finaux
4. **Cache** : Les modèles sont chargés une seule fois et réutilisés pour toutes les requêtes

