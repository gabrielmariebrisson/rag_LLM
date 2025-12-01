# Déploiement vLLM Local pour Latence Minimale

## Vue d'ensemble

Ce guide explique comment déployer vLLM en local pour obtenir une latence minimale (<40ms TTFT) et rendre le backend agnostique du fournisseur LLM (OpenAI, Mistral API, ou vLLM local).

## Architecture

Le système supporte maintenant trois modes de déploiement LLM :

1. **vLLM Local** : Déploiement local avec GPU pour latence minimale
2. **OpenAI API** : Utilisation de l'API OpenAI standard
3. **Mistral API** : Utilisation de l'API Mistral (legacy, backward compatible)

## Prérequis

- **Docker** et **Docker Compose** installés
- **GPU NVIDIA** avec drivers et **nvidia-container-toolkit** installé
- Au moins **8GB VRAM** pour le modèle Mistral-7B-Instruct-AWQ

### Installation nvidia-container-toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Démarrage vLLM

### 1. Démarrer le service vLLM

```bash
docker-compose up -d vllm-service
```

Le service va :
- Télécharger le modèle `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` (première fois)
- Charger le modèle en mémoire GPU
- Exposer l'API sur `http://localhost:8001`

### 2. Vérifier que vLLM est prêt

```bash
# Vérifier la santé
curl http://localhost:8001/health

# Lister les modèles disponibles
curl http://localhost:8001/v1/models
```

Vous devriez voir une réponse avec le modèle chargé.

### 3. Tester une requête

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Configuration Backend

### Mode vLLM Local (Recommandé pour latence)

Dans votre fichier `.env` :

```env
# vLLM Local
LLM_BASE_URL=http://localhost:8001/v1
LLM_API_KEY=dummy-key
LLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
```

**Note** : vLLM n'a pas besoin de vraie clé API, `dummy-key` fonctionne.

### Mode OpenAI API

```env
# OpenAI API
LLM_BASE_URL=
LLM_API_KEY=sk-...
LLM_MODEL_NAME=gpt-4o-mini
```

### Mode Mistral API (Legacy)

```env
# Mistral API (backward compatible)
MISTRAL_API_KEY=your-mistral-key
MISTRAL_MODEL_NAME=mistral-tiny-2407
```

## Démarrage du Backend

```bash
uvicorn app.main:app --reload --port 8000
```

Le backend détecte automatiquement le mode LLM selon la configuration :
- Si `LLM_BASE_URL` est défini → vLLM local
- Si `LLM_API_KEY` est défini (sans `LLM_BASE_URL`) → OpenAI API
- Si `MISTRAL_API_KEY` est défini → Mistral API (legacy)

## Performance

### Latence attendue avec vLLM

- **TTFT (Time To First Token)** : <40ms avec GPU
- **Tokens/s** : ~50-100 tokens/s selon GPU
- **Latence totale** (requête complète) : ~200-500ms pour réponse courte

### Comparaison des modes

| Mode | TTFT | Latence Totale | Coût |
|------|------|----------------|------|
| vLLM Local | <40ms | ~200-500ms | Gratuit (GPU requis) |
| OpenAI API | ~200-500ms | ~1-3s | Payant |
| Mistral API | ~300-800ms | ~1-4s | Payant |

## Dépannage

### vLLM ne démarre pas

**Erreur GPU** :
```bash
# Vérifier que Docker voit le GPU
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

**Erreur port déjà utilisé** :
```bash
# Changer le port dans docker-compose.yml
ports:
  - "8002:8000"  # Au lieu de 8001:8000
```

### Modèle trop lent

- Vérifier que le GPU est bien utilisé : `nvidia-smi` pendant l'exécution
- Réduire la taille du modèle ou utiliser une quantification plus agressive
- Vérifier la mémoire GPU disponible : `nvidia-smi`

### Erreur "Out of Memory"

Le modèle AWQ nécessite ~8GB VRAM. Options :
- Utiliser un modèle plus petit
- Réduire `--max-model-len` dans la commande vLLM
- Utiliser CPU (beaucoup plus lent) : retirer la section `deploy` dans docker-compose.yml

### Backend ne se connecte pas à vLLM

1. Vérifier que vLLM est démarré : `docker ps`
2. Vérifier l'URL dans `.env` : `LLM_BASE_URL=http://localhost:8001/v1`
3. Tester manuellement : `curl http://localhost:8001/health`

## Architecture Technique

### Client LLM Agnostique

Le service `LLMClient` (`app/services/llm_client.py`) :
- Détecte automatiquement le mode selon la configuration
- Utilise `openai.OpenAI()` pour vLLM et OpenAI API (compatible)
- Utilise `mistralai.Mistral()` pour Mistral API (legacy)
- Interface unifiée : `generate(messages, model) -> str`

### Compatibilité vLLM

vLLM expose une API **100% compatible OpenAI**, donc :
- Le client `openai.OpenAI(base_url="http://localhost:8001/v1")` fonctionne directement
- Aucune modification du code de génération nécessaire
- Support natif des messages au format OpenAI

## Monitoring

### Logs vLLM

```bash
docker-compose logs -f vllm-service
```

### Métriques de performance

Le backend expose des métriques dans la réponse `/chat` :
- `processing_time` : Temps total de traitement
- Pour plus de détails, ajouter du logging dans `generate_response()`

## Production

Pour la production avec vLLM :

1. **Scaling** : Utiliser plusieurs instances vLLM avec load balancer
2. **GPU** : Utiliser des GPU plus puissants (A100, H100) pour meilleure performance
3. **Monitoring** : Intégrer Prometheus/Grafana pour métriques
4. **Health Checks** : Le docker-compose inclut déjà un healthcheck

## Notes

- Le modèle AWQ est quantifié, donc plus léger mais légèrement moins précis
- vLLM supporte le batching automatique pour améliorer le throughput
- Pour des modèles plus grands, ajuster la mémoire GPU et les paramètres vLLM

