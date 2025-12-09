# 📁 Fichiers utilisés lors de l'exécution de `scripts/migrate_to_qdrant.py`

## Fichiers Python (code source)

### Script principal
- `scripts/migrate_to_qdrant.py` - Script de migration

### Modules importés
- `app/core/config.py` - Configuration (Settings)
- `app/services/embeddings.py` - Service d'embeddings (dense + sparse)
- `app/vectorstores/qdrant_store.py` - Wrapper Qdrant

### Dépendances indirectes (via les imports)
- `app/services/__init__.py`
- `app/core/__init__.py`
- `app/vectorstores/__init__.py`

## Fichiers de configuration

### Fichiers lus directement
- `.env` - Variables d'environnement (chargé via `load_dotenv()`)
  - Contient : `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION_NAME`
  - Contient : `DENSE_MODEL`, `SPARSE_MODEL`, `RERANKER_MODEL`
  - Contient : `USE_RERANKER`, `RERANKER_TOP_K`

### Fichiers de configuration Qdrant
- `qdrant_config.yaml` - Configuration Qdrant (lu par le serveur Qdrant, pas directement par le script)

## Fichiers de données

### Fichier CSV principal
- `squad_2.0/train.csv` - Données SQuAD 2.0 (130,319 lignes)
  - Colonnes attendues : `id`, `title`, `question`, `context`, `answers`

## Fichiers de stockage Qdrant

### Données Qdrant (créées/écrites)
- `qdrant_data/` - Répertoire de stockage Qdrant
  - Collections créées : `squad_collection`
  - Points vectoriels indexés

## Modèles HuggingFace (téléchargés automatiquement)

### Modèles téléchargés au premier lancement
- **Modèle dense** : `sentence-transformers/all-MiniLM-L6-v2`
  - Cache : `~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/`
  - Taille : ~90 MB

- **Modèle sparse** : `bert-base-uncased`
  - Cache : `~/.cache/huggingface/hub/models--bert-base-uncased/`
  - Taille : ~440 MB

- **Tokenizer** : `bert-base-uncased` (tokenizer)
  - Cache : `~/.cache/huggingface/hub/models--bert-base-uncased/`

## Fichiers de cache Python

### Cache des packages
- `venv/` - Environnement virtuel Python (si utilisé)
- `__pycache__/` - Cache Python (créé automatiquement)

## Résumé

### Fichiers lus directement
1. ✅ `scripts/migrate_to_qdrant.py` (script principal)
2. ✅ `.env` (configuration)
3. ✅ `squad_2.0/train.csv` (données)

### Fichiers Python importés
4. ✅ `app/core/config.py`
5. ✅ `app/services/embeddings.py`
6. ✅ `app/vectorstores/qdrant_store.py`

### Fichiers créés/écrits
7. ✅ `qdrant_data/` (données Qdrant)
8. ✅ `__pycache__/` (cache Python)

### Modèles téléchargés (première exécution)
9. ✅ `~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/`
10. ✅ `~/.cache/huggingface/hub/models--bert-base-uncased/`

## Commandes pour vérifier les fichiers

```bash
# Vérifier que le CSV existe
ls -lh squad_2.0/train.csv

# Vérifier le fichier .env
cat .env

# Vérifier les modèles téléchargés
du -sh ~/.cache/huggingface/hub/models--*

# Vérifier les données Qdrant
du -sh qdrant_data/
```

