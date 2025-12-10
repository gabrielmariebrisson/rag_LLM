# 🚀 Guide de Déploiement

Guide complet pour déployer le système RAG en production.

## 📋 Table des Matières

- [Vue d'Ensemble](#vue-densemble)
- [Déploiement Docker Compose](#déploiement-docker-compose)
- [Déploiement Kubernetes](#déploiement-kubernetes)
- [Configuration de Production](#configuration-de-production)
- [Monitoring et Observabilité](#monitoring-et-observabilité)
- [Scaling](#scaling)
- [Sécurité Production](#sécurité-production)

---

## Vue d'Ensemble

Le système peut être déployé de plusieurs façons :

1. **Développement Local** : Services individuels avec Docker Compose
2. **Production Kubernetes** : Déploiement complet avec auto-scaling
3. **Hybrid** : Qdrant Cloud + Services auto-hébergés

---

## Déploiement Docker Compose

### Prérequis

- Docker Engine 20.10+
- Docker Compose 2.0+
- NVIDIA Docker (si GPU disponible)

### Architecture Docker Compose

```
docker-compose.yml
├── jaeger (Tracing)
├── vllm-service (LLM Server)
└── (Backend et Frontend à démarrer manuellement ou ajouter au compose)
```

### Démarrage Rapide

#### 1. Démarrer les Services

```bash
# Démarrer tous les services
docker-compose up -d

# Vérifier les services
docker-compose ps

# Voir les logs
docker-compose logs -f vllm-service
```

#### 2. Services Disponibles

- **Jaeger UI** : http://localhost:16686
- **vLLM Service** : http://localhost:8001
- **Qdrant** : http://localhost:6333 (à démarrer manuellement ou ajouter au compose)

#### 3. Démarrer le Backend

Dans un terminal séparé :

```bash
# Configuration environnement
export LLM_BASE_URL=http://localhost:8001/v1
export LLM_MODEL_NAME=TheBloke/Mistral-7B-Instruct-v0.2-AWQ

# Démarrer le backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### 4. Démarrer le Frontend

```bash
streamlit run frontend/app.py --server.port 8501
```

### Configuration Docker Compose Avancée

#### Ajouter Qdrant au Compose

Créer `docker-compose.yml` étendu :

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./data/processed/qdrant_data:/qdrant/storage
    restart: unless-stopped
    networks:
      - rag-network

  backend:
    build:
      context: .
      dockerfile: docker/Dockerfile.backend
    container_name: rag-backend
    ports:
      - "8000:8000"
    environment:
      - LLM_BASE_URL=http://vllm-service:8000/v1
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    depends_on:
      - qdrant
      - vllm-service
    networks:
      - rag-network

  frontend:
    build:
      context: .
      dockerfile: docker/Dockerfile.frontend
    container_name: rag-frontend
    ports:
      - "8501:8501"
    environment:
      - BACKEND_URL=http://backend:8000
    depends_on:
      - backend
    networks:
      - rag-network

  # ... autres services (jaeger, vllm-service)

networks:
  rag-network:
    driver: bridge
```

### Dockerfiles

#### docker/Dockerfile.backend

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Port
EXPOSE 8000

# Commande
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### docker/Dockerfile.frontend

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Installer Streamlit
RUN pip install streamlit httpx pandas plotly

# Copier le frontend
COPY frontend/ ./frontend/
COPY templates/ ./templates/

# Port
EXPOSE 8501

# Commande
CMD ["streamlit", "run", "frontend/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

---

## Déploiement Kubernetes

### Prérequis

- Kubernetes cluster 1.24+
- kubectl configuré
- NVIDIA GPU Operator (si GPU requis)
- PersistentVolume pour Qdrant (optionnel)

### Structure des Manifests

```
k8s/
├── backend.yaml      # Deployment Backend FastAPI
├── vllm.yaml         # Deployment vLLM
└── ingress.yaml      # Ingress pour exposition externe
```

### Déploiement Étape par Étape

#### 1. Créer les Namespaces

```bash
kubectl create namespace rag-system
```

#### 2. Déployer Qdrant

**Option A : Qdrant Cloud** (Recommandé pour production)

```yaml
# k8s/qdrant-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: qdrant-api-key
  namespace: rag-system
type: Opaque
stringData:
  api-key: "your-qdrant-cloud-api-key"
```

**Option B : Qdrant dans Kubernetes**

```yaml
# k8s/qdrant.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
  namespace: rag-system
spec:
  serviceName: qdrant
  replicas: 1
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:latest
        ports:
        - containerPort: 6333
        - containerPort: 6334
        volumeMounts:
        - name: qdrant-data
          mountPath: /qdrant/storage
  volumeClaimTemplates:
  - metadata:
      name: qdrant-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 50Gi
---
apiVersion: v1
kind: Service
metadata:
  name: qdrant-service
  namespace: rag-system
spec:
  selector:
    app: qdrant
  ports:
  - port: 6333
    targetPort: 6333
```

```bash
kubectl apply -f k8s/qdrant.yaml
```

#### 3. Déployer vLLM

```bash
kubectl apply -f k8s/vllm.yaml
```

**Vérifier** :

```bash
kubectl get pods -n rag-system -l app=vllm-service
kubectl logs -n rag-system -l app=vllm-service
```

#### 4. Créer les Secrets

```bash
# Créer secret pour Mistral API (optionnel)
kubectl create secret generic mistral-secret \
  --from-literal=api-key=your-api-key \
  -n rag-system
```

#### 5. Déployer le Backend

```bash
# Modifier backend.yaml avec vos valeurs
kubectl apply -f k8s/backend.yaml
```

**Vérifier** :

```bash
kubectl get pods -n rag-system -l app=rag-backend
kubectl logs -n rag-system -l app=rag-backend
```

#### 6. Déployer l'Ingress

```bash
# Adapter ingress.yaml avec votre domaine
kubectl apply -f k8s/ingress.yaml
```

#### 7. Vérifier le Déploiement

```bash
# Tous les pods
kubectl get pods -n rag-system

# Services
kubectl get svc -n rag-system

# Ingress
kubectl get ingress -n rag-system
```

### Configuration Production Kubernetes

#### Backend avec Auto-Scaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-backend-hpa
  namespace: rag-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### Backend avec Resources Limits

```yaml
resources:
  requests:
    cpu: "1000m"
    memory: "4Gi"
  limits:
    cpu: "4000m"
    memory: "8Gi"
```

#### Pod Disruption Budget

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: rag-backend-pdb
  namespace: rag-system
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: rag-backend
```

---

## Configuration de Production

### Variables d'Environnement Production

#### Backend

```env
# LLM
LLM_BASE_URL=http://vllm-service:8000/v1
LLM_MODEL_NAME=TheBloke/Mistral-7B-Instruct-v0.2-AWQ

# Qdrant
QDRANT_HOST=qdrant-service
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=squad_collection

# Reranker
USE_RERANKER=true
RERANKER_TOP_K=30

# OpenTelemetry
ENABLE_JAEGER_EXPORT=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger-service:4317

# Performance
PYTHONUNBUFFERED=1
```

#### Frontend

```env
BACKEND_URL=http://rag-backend:8000
```

### Optimisations Performance

#### Backend

1. **Workers Uvicorn** :

```bash
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker
```

2. **Gunicorn avec Uvicorn Workers** :

```bash
gunicorn app.main:app \
  --bind 0.0.0.0:8000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --timeout 120
```

3. **Variables d'Environnement** :

```env
# Augmenter les limites Python
PYTHONOPTIMIZE=2
PYTHONDONTWRITEBYTECODE=1
```

#### Qdrant

1. **Optimisations Collection** :

```python
# Créer collection optimisée
await client.create_collection(
    collection_name="squad_collection",
    vectors_config={
        "dense": VectorParams(
            size=384,
            distance=Distance.COSINE,
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100
            )
        )
    }
)
```

2. **Indexation** :

```bash
# Indexer avec optimisations
python scripts/migrate_to_qdrant.py --batch-size 1000
```

---

## Monitoring et Observabilité

### OpenTelemetry + Jaeger

#### Configuration Jaeger dans Kubernetes

```yaml
# k8s/jaeger.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jaeger
  namespace: rag-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: jaeger
  template:
    metadata:
      labels:
        app: jaeger
    spec:
      containers:
      - name: jaeger
        image: jaegertracing/all-in-one:latest
        ports:
        - containerPort: 16686
        - containerPort: 4317
        - containerPort: 4318
        env:
        - name: COLLECTOR_OTLP_ENABLED
          value: "true"
---
apiVersion: v1
kind: Service
metadata:
  name: jaeger-service
  namespace: rag-system
spec:
  selector:
    app: jaeger
  ports:
  - port: 16686
    targetPort: 16686
    name: ui
  - port: 4317
    targetPort: 4317
    name: otlp-grpc
  - port: 4318
    targetPort: 4318
    name: otlp-http
```

#### Accéder à Jaeger

- **Local** : http://localhost:16686
- **Kubernetes** : http://jaeger-service.rag-system.svc.cluster.local:16686
- **Ingress** : http://your-domain.com/jaeger

### Métriques Prometheus (Optionnel)

#### Exporter Métriques FastAPI

Installer `prometheus-fastapi-instrumentator` :

```python
# app/main.py
from prometheus_fastapi_instrumentator import Instrumentator

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)
```

#### Configuration Prometheus

```yaml
# prometheus-config.yaml
scrape_configs:
  - job_name: 'rag-backend'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - rag-system
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: rag-backend
```

### Logging Structuré

#### Configuration Logging

```python
# app/main.py
import logging
import json
from pythonjsonlogger import jsonlogger

logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)
```

### Health Checks

#### Kubernetes Liveness/Readiness Probes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 2
```

---

## Scaling

### Scaling Horizontal (Backend)

#### Auto-Scaling Kubernetes

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-backend
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 30
      selectPolicy: Max
```

### Scaling Vertical (vLLM)

#### Multi-GPU vLLM

```yaml
# k8s/vllm-multigpu.yaml
spec:
  containers:
  - name: vllm
    resources:
      limits:
        nvidia.com/gpu: 4  # 4 GPUs
      requests:
        nvidia.com/gpu: 4
    args:
    - --tensor-parallel-size=4  # Utiliser 4 GPUs
```

#### Configuration vLLM Performance

```yaml
args:
  - --max-num-seqs=256          # Plus de requêtes simultanées
  - --max-model-len=4096        # Contexte plus long
  - --gpu-memory-utilization=0.9 # Utilisation GPU
```

### Scaling Qdrant

#### Qdrant Cluster Mode

```yaml
# Qdrant avec réplication
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
spec:
  replicas: 3  # 3 replicas
  serviceName: qdrant
  template:
    spec:
      containers:
      - name: qdrant
        env:
        - name: QDRANT__CLUSTER__ENABLED
          value: "true"
        - name: QDRANT__CLUSTER__P2P__PORT
          value: "6335"
```

### Load Balancing

#### Kubernetes Service avec Session Affinity

```yaml
apiVersion: v1
kind: Service
spec:
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 3600
```

---

## Sécurité Production

### Secrets Management

#### Kubernetes Secrets

```bash
# Créer secrets
kubectl create secret generic rag-secrets \
  --from-literal=llm-api-key=xxx \
  --from-literal=qdrant-api-key=xxx \
  -n rag-system
```

#### Utilisation dans Deployment

```yaml
env:
- name: LLM_API_KEY
  valueFrom:
    secretKeyRef:
      name: rag-secrets
      key: llm-api-key
```

### Network Policies

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: rag-backend-policy
  namespace: rag-system
spec:
  podSelector:
    matchLabels:
      app: rag-backend
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: rag-system
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: qdrant
    ports:
    - protocol: TCP
      port: 6333
```

### TLS/HTTPS

#### Cert-Manager pour Certificats Automatiques

```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: rag-tls
  namespace: rag-system
spec:
  secretName: rag-tls-secret
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - rag.example.com
```

---

## Troubleshooting Déploiement

### Logs

```bash
# Logs backend
kubectl logs -n rag-system -l app=rag-backend --tail=100

# Logs vLLM
kubectl logs -n rag-system -l app=vllm-service --tail=100

# Logs tous les pods
kubectl logs -n rag-system --all-containers=true --tail=50
```

### Debug Pod

```bash
# Exec dans pod
kubectl exec -it -n rag-system deployment/rag-backend -- /bin/bash

# Tester connexion Qdrant
kubectl exec -it -n rag-system deployment/rag-backend -- \
  python -c "import httpx; print(httpx.get('http://qdrant-service:6333/health').json())"
```

### Ressources

```bash
# Utilisation ressources
kubectl top pods -n rag-system

# Détails pod
kubectl describe pod -n rag-system <pod-name>
```

---

## Ressources Additionnelles

- [README.md](../README.md) : Guide de démarrage rapide
- [CONFIGURATION.md](CONFIGURATION.md) : Guide de configuration
- [ARCHITECTURE.md](ARCHITECTURE.md) : Documentation technique

---

**Version** : 2.0.0  
**Dernière mise à jour** : 2025-01-27

