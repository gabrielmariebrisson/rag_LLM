# 🚀 Installation sans Docker sur RunPod

Ce guide explique comment utiliser le projet RAG sans Docker.

## ✅ Qdrant installé

Qdrant a été installé en binaire local (sans Docker) et est déjà démarré.

### Commandes utiles

**Démarrer Qdrant :**
```bash
cd /workspace/rag_LLM
./start_qdrant.sh
# ou en arrière-plan :
nohup ./qdrant --config-path ./qdrant_config.yaml > qdrant.log 2>&1 &
```

**Vérifier que Qdrant fonctionne :**
```bash
curl http://localhost:6333/collections
```

**Arrêter Qdrant :**
```bash
pkill qdrant
```

**Voir les logs :**
```bash
tail -f qdrant.log
```

## 📦 Prochaines étapes

1. **Installer les dépendances Python :**
```bash
cd /workspace/rag_LLM
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Configurer le fichier .env :**
```bash
# Créer ou modifier .env
cp .env.example .env  # si le fichier existe
# Puis éditer avec vos clés API
```

3. **Migrer les données vers Qdrant :**
```bash
python scripts/migrate_to_qdrant.py
```

4. **Lancer le backend :**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

5. **Lancer le frontend (dans un autre terminal) :**
```bash
streamlit run frontend/app.py --server.port 8501
```

## 🔧 Configuration Qdrant

Le fichier `qdrant_config.yaml` contient la configuration de Qdrant :
- Port HTTP : 6333
- Port gRPC : 6334
- Stockage : `./qdrant_data`

## ⚠️ Note sur Jaeger

Jaeger (tracing) nécessite Docker. Si vous n'en avez pas besoin, vous pouvez :
- Désactiver le tracing dans votre configuration
- Utiliser Qdrant Cloud (service cloud) à la place
- Installer Docker si nécessaire

## 🌐 Alternative : Qdrant Cloud

Si vous préférez utiliser Qdrant Cloud (service managé) :
1. Créer un compte sur https://cloud.qdrant.io
2. Créer un cluster
3. Mettre à jour `.env` avec l'URL et la clé API de votre cluster

