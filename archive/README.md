# 📦 Archive - Fichiers Legacy

Ce répertoire contient les fichiers legacy du projet qui ne sont plus utilisés dans la version actuelle mais conservés à des fins de référence historique ou de migration.

## 📋 Contenu

### `rag_LLM_web.py`
- **Description** : Ancienne version du frontend Streamlit utilisant FAISS
- **Remplacement** : `frontend/app.py` (nouvelle version avec Qdrant)
- **Date** : Version antérieure à 2.0.0
- **État** : Legacy - Non maintenu

### `azure_rag.py`
- **Description** : Script d'exploration pour Azure Cognitive Search
- **Usage** : POC initial, non intégré dans le projet actuel
- **État** : Legacy - Non maintenu

### `rag_LLM.ipynb`
- **Description** : Notebook Jupyter de développement et d'exploration
- **Usage** : Prototypes et tests initiaux
- **État** : Archive - Référence seulement

### `faiss_index/`
- **Description** : Index FAISS legacy (ancien système de vector store)
- **Remplacement** : Migration vers Qdrant effectuée
- **État** : Legacy - Index migré vers Qdrant dans `data/processed/qdrant_data/`
- **Fichiers** :
  - `index.faiss` : Index FAISS
  - `index.pkl` : Métadonnées

## ⚠️ Note Importante

Ces fichiers sont conservés uniquement à des fins :
- **Référence historique** : Comprendre l'évolution du projet
- **Migration** : Aide à la migration depuis l'ancien système
- **Restauration** : En cas de besoin de restaurer une version antérieure

**Ils ne doivent PAS être utilisés pour le développement actuel.**

Pour utiliser le système actuel, référez-vous à :
- `frontend/app.py` : Frontend actuel
- `app/` : Backend FastAPI actuel
- `data/processed/qdrant_data/` : Vector store actuel (Qdrant)

---

**Date d'archivage** : 2025-01-27  
**Version projet** : 2.0.0

