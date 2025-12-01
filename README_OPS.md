# Guide Opérationnel - Stack RAG avec Observabilité

## Vue d'ensemble

Ce guide explique comment déployer et monitorer le stack RAG complet avec observabilité de niveau production via Jaeger.

## Architecture du Stack

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Frontend   │────▶│   Backend    │────▶│    vLLM     │
│  Streamlit  │     │   FastAPI    │     │   (Local)   │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ├─────────────┐
                            │             │
                            ▼             ▼
                    ┌──────────────┐  ┌─────────────┐
                    │    Qdrant    │  │   Jaeger    │
                    │  (Vector DB) │  │  (Traces)   │
                    └──────────────┘  └─────────────┘
```

## Démarrage avec Docker Compose

### 1. Prérequis

- Docker et Docker Compose installés
- GPU NVIDIA avec `nvidia-container-toolkit` (pour vLLM)
- Au moins 8GB VRAM disponible

### 2. Configuration

Créer un fichier `.env` à la racine :

```env
# LLM Configuration (vLLM local)
LLM_BASE_URL=http://localhost:8001/v1
LLM_API_KEY=dummy-key
LLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=squad_collection

# OpenTelemetry (Jaeger)
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=rag-system
```

### 3. Démarrer le stack complet

```bash
# Démarrer tous les services
docker-compose up -d

# Vérifier les services
docker-compose ps

# Voir les logs
docker-compose logs -f
```

Services démarrés :
- **Jaeger** : `http://localhost:16686` (UI)
- **vLLM** : `http://localhost:8001` (API)
- **Backend FastAPI** : `http://localhost:8000` (API)
- **Frontend Streamlit** : `http://localhost:8501` (UI)

### 4. Vérifier que tout fonctionne

```bash
# Vérifier Jaeger
curl http://localhost:16686

# Vérifier vLLM
curl http://localhost:8001/health

# Vérifier Backend
curl http://localhost:8000/health
```

## Accès au Dashboard Jaeger

### URL

**Jaeger UI** : http://localhost:16686

### Visualisation des Traces

1. **Ouvrir Jaeger UI** dans votre navigateur
2. **Sélectionner le service** : `rag-system`
3. **Cliquer sur "Find Traces"**

### Waterfall des Requêtes

Chaque requête RAG génère une trace avec les spans suivants :

```
POST /chat
├── rag.retrieval          # Recherche hybride Qdrant
│   ├── hybrid_search      # Recherche dense + sparse
│   └── (optionnel) rag.reranking  # Reranking CrossEncoder
└── llm.generation         # Génération réponse LLM
```

**Interprétation** :
- **rag.retrieval** : Temps de recherche dans Qdrant (typiquement 50-100ms)
- **rag.reranking** : Temps de reranking (typiquement 50-100ms, si activé)
- **llm.generation** : Temps de génération LLM (typiquement 200-500ms avec vLLM local)

### Analyse des Performances

Dans Jaeger, vous pouvez :
- **Voir la latence** de chaque étape
- **Identifier les bottlenecks** (quelle étape prend le plus de temps)
- **Comparer les requêtes** avec/sans reranker
- **Détecter les erreurs** (spans rouges)

### Attributs des Spans

Chaque span contient des attributs utiles :
- `query` : La question de l'utilisateur
- `k` : Nombre de documents récupérés
- `model` : Modèle LLM utilisé
- `num_candidates` : Nombre de candidats avant reranking
- `num_final_results` : Nombre de résultats finaux
- `response_length` : Longueur de la réponse générée

## Déploiement Kubernetes

### Prérequis

- Cluster Kubernetes avec GPU support (nvidia-device-plugin)
- `kubectl` configuré
- Ingress controller installé (ex: nginx-ingress)

### 1. Créer le namespace

```bash
kubectl create namespace rag-system
```

### 2. Déployer vLLM

```bash
kubectl apply -f k8s/vllm.yaml -n rag-system
```

**Vérifier** :
```bash
kubectl get pods -n rag-system -l app=vllm-service
kubectl logs -n rag-system -l app=vllm-service
```

### 3. Déployer Qdrant

```bash
# Créer un deployment Qdrant (exemple basique)
kubectl create deployment qdrant-service --image=qdrant/qdrant:latest -n rag-system
kubectl expose deployment qdrant-service --port=6333 -n rag-system
```

### 4. Déployer Jaeger

```bash
# Utiliser le chart Helm officiel ou créer un deployment
kubectl create deployment jaeger-service --image=jaegertracing/all-in-one:latest -n rag-system
kubectl expose deployment jaeger-service --port=16686 --target-port=16686 -n rag-system
kubectl expose deployment jaeger-service --port=4317 --target-port=4317 --name=jaeger-otlp -n rag-system
```

### 5. Déployer le Backend

```bash
# Construire et pousser l'image Docker
docker build -t rag-backend:latest .
docker tag rag-backend:latest your-registry/rag-backend:latest
docker push your-registry/rag-backend:latest

# Mettre à jour k8s/backend.yaml avec votre registry
# Puis déployer
kubectl apply -f k8s/backend.yaml -n rag-system
```

### 6. Déployer le Frontend

```bash
# Créer un deployment Streamlit (exemple)
kubectl create deployment rag-frontend --image=streamlit/streamlit:latest -n rag-system
kubectl expose deployment rag-frontend --port=8501 -n rag-system
```

### 7. Configurer l'Ingress

```bash
# Modifier k8s/ingress.yaml avec votre domaine
# Puis appliquer
kubectl apply -f k8s/ingress.yaml -n rag-system
```

### 8. Vérifier le déploiement

```bash
# Voir tous les pods
kubectl get pods -n rag-system

# Voir les services
kubectl get svc -n rag-system

# Voir l'ingress
kubectl get ingress -n rag-system
```

## Monitoring et Troubleshooting

### Logs

**Docker Compose** :
```bash
# Logs de tous les services
docker-compose logs -f

# Logs d'un service spécifique
docker-compose logs -f vllm-service
docker-compose logs -f jaeger
```

**Kubernetes** :
```bash
# Logs du backend
kubectl logs -n rag-system -l app=rag-backend -f

# Logs de vLLM
kubectl logs -n rag-system -l app=vllm-service -f
```

### Métriques Jaeger

Dans Jaeger UI, vous pouvez voir :
- **Nombre de traces** par seconde
- **Latence P50, P95, P99**
- **Taux d'erreur**
- **Services les plus lents**

### Problèmes Courants

**1. Jaeger ne reçoit pas de traces**
- Vérifier que `OTEL_EXPORTER_OTLP_ENDPOINT` est correct
- Vérifier que Jaeger est accessible depuis le backend
- Vérifier les logs du backend pour erreurs OpenTelemetry

**2. vLLM ne démarre pas**
- Vérifier que le GPU est disponible : `nvidia-smi`
- Vérifier les logs : `docker-compose logs vllm-service`
- Vérifier la mémoire GPU disponible (besoin de ~8GB)

**3. Backend ne se connecte pas à vLLM**
- Vérifier que `LLM_BASE_URL` est correct
- Vérifier que le service vLLM est accessible
- Tester manuellement : `curl http://localhost:8001/health`

**4. Traces incomplètes**
- Vérifier que les spans sont bien créés dans le code
- Vérifier que le tracer est correctement initialisé
- Vérifier les logs OpenTelemetry

## Performance Attendue

Avec vLLM local (GPU) :
- **TTFT (Time To First Token)** : <40ms
- **Latence totale requête** : 200-500ms
- **Throughput** : 10-20 requêtes/seconde

Sans vLLM (API externe) :
- **Latence totale** : 1-3 secondes
- **Throughput** : 5-10 requêtes/seconde

## Commandes Utiles

### Docker Compose

```bash
# Démarrer
docker-compose up -d

# Arrêter
docker-compose down

# Redémarrer un service
docker-compose restart vllm-service

# Voir les logs en temps réel
docker-compose logs -f backend
```

### Kubernetes

```bash
# Port-forward pour accès local
kubectl port-forward -n rag-system svc/jaeger-service 16686:16686
kubectl port-forward -n rag-system svc/rag-backend 8000:8000

# Scale le backend
kubectl scale deployment rag-backend --replicas=3 -n rag-system

# Redémarrer un pod
kubectl rollout restart deployment rag-backend -n rag-system
```

## Sécurité Production

⚠️ **Important pour la production** :

1. **Secrets** : Ne pas commiter les clés API dans les manifests
2. **TLS** : Activer HTTPS via l'ingress
3. **Network Policies** : Restreindre l'accès réseau entre services
4. **Resource Limits** : Définir des limites CPU/Memory appropriées
5. **Jaeger** : Ne pas exposer Jaeger UI publiquement en production

## Support

Pour plus d'informations :
- **Jaeger** : https://www.jaegertracing.io/docs/
- **OpenTelemetry** : https://opentelemetry.io/docs/
- **vLLM** : https://docs.vllm.ai/
- **Kubernetes** : https://kubernetes.io/docs/

