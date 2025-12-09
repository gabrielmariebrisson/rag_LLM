# Guide : Utiliser GPU NVIDIA sans machine physique

## 🎯 Options disponibles

### 1. ✅ Google Colab (GRATUIT - Recommandé pour tester)

**Avantages** :
- ✅ **Gratuit** avec GPU NVIDIA (T4, parfois A100)
- ✅ Pas besoin de configuration complexe
- ✅ Accès direct via navigateur
- ✅ Parfait pour POC et développement

**Limitations** :
- ⚠️ Sessions limitées (12h max, peut être interrompue)
- ⚠️ GPU partagé (pas toujours disponible)
- ⚠️ Pas de Docker (mais on peut installer vLLM directement)
- ⚠️ Pas de persistance (sauf Google Drive)

**Comment utiliser** :

#### Option A : vLLM directement dans Colab

```python
# Cellule 1 : Installer les dépendances
!pip install vllm transformers torch

# Cellule 2 : Lancer vLLM
from vllm import LLM, SamplingParams

# Charger le modèle
llm = LLM(
    model="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
    quantization="awq",
    dtype="float16",
    gpu_memory_utilization=0.9
)

# Cellule 3 : Tester
sampling_params = SamplingParams(temperature=0.7, top_p=0.95)
prompts = ["Hello, my name is", "The capital of France is"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")
```

#### Option B : vLLM avec API OpenAI-compatible dans Colab

```python
# Cellule 1 : Installer
!pip install vllm fastapi uvicorn

# Cellule 2 : Lancer le serveur vLLM
import subprocess
import threading
import time

def start_vllm_server():
    subprocess.run([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        "--quantization", "awq",
        "--dtype", "float16",
        "--port", "8000",
        "--host", "0.0.0.0"
    ])

# Lancer en arrière-plan
thread = threading.Thread(target=start_vllm_server, daemon=True)
thread.start()

# Attendre que le serveur démarre
time.sleep(60)  # vLLM prend du temps à charger

# Cellule 3 : Tester l'API
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 0.7
    }
)
print(response.json())
```

**⚠️ Important pour Colab** :
- Activez le GPU : `Runtime > Change runtime type > GPU (T4 ou A100)`
- Le port 8000 est accessible via `ngrok` si vous voulez exposer l'API

---

### 2. ☁️ Cloud Providers (Payant mais professionnel)

#### A. **Google Cloud Platform (GCP) - Compute Engine**

**Avantages** :
- ✅ GPU NVIDIA (T4, V100, A100)
- ✅ Contrôle total (Docker, Kubernetes)
- ✅ Persistance des données
- ✅ Crédits gratuits ($300 pour nouveaux comptes)

**Coûts** :
- T4 : ~$0.35/heure (~$250/mois si 24/7)
- A100 : ~$3.50/heure

**Setup rapide** :
```bash
# 1. Créer une VM avec GPU
gcloud compute instances create vllm-gpu \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=50GB

# 2. Installer Docker + NVIDIA Container Toolkit
# (voir README_VLLM.md pour les détails)

# 3. Lancer votre docker-compose.yml
docker-compose up -d
```

#### B. **AWS EC2 - GPU Instances**

**Types d'instances** :
- `g4dn.xlarge` : T4, ~$0.50/heure
- `p3.2xlarge` : V100, ~$3.00/heure
- `p4d.24xlarge` : A100, ~$32/heure

**Setup** :
```bash
# 1. Lancer une instance avec Deep Learning AMI (pré-configurée)
# 2. SSH dans l'instance
# 3. docker-compose up -d
```

#### C. **Azure - NC Series**

**Instances** :
- `NC6s_v3` : V100, ~$3.00/heure
- `ND96asr_v4` : A100, ~$30/heure

**Avantages** :
- ✅ Intégration Azure ML
- ✅ Crédits étudiants disponibles

---

### 3. 🆓 Alternatives Gratuites (Limitées)

#### A. **Kaggle Notebooks**
- GPU gratuit (30h/semaine)
- Similaire à Colab
- Bon pour expérimenter

#### B. **Lambda Labs**
- GPU à la demande
- ~$0.50/heure pour A100
- Pay-as-you-go

#### C. **RunPod / Vast.ai**
- GPU partagés très bon marché
- ~$0.20-0.50/heure pour T4
- Parfait pour développement

---

## 🚀 Recommandation pour votre projet

### Pour tester rapidement (POC) :
**→ Google Colab** (gratuit, rapide à setup)

### Pour développement sérieux :
**→ GCP Compute Engine avec T4** (~$250/mois si 24/7, ou arrêter la VM quand pas utilisé)

### Pour production :
**→ GCP avec A100** ou **AWS p4d** (selon budget)

---

## 📝 Script Colab complet pour votre projet

Créez un notebook Colab avec ce contenu :

```python
# ============================================
# CELLULE 1 : Setup GPU et dépendances
# ============================================
# Runtime > Change runtime type > GPU (T4)

!pip install -q vllm fastapi uvicorn qdrant-client[async] fastembed transformers torch sentence-transformers

# ============================================
# CELLULE 2 : Lancer vLLM en arrière-plan
# ============================================
import subprocess
import threading
import time
import requests

def start_vllm():
    subprocess.run([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        "--quantization", "awq",
        "--dtype", "float16",
        "--port", "8000",
        "--host", "0.0.0.0"
    ])

thread = threading.Thread(target=start_vllm, daemon=True)
thread.start()
print("⏳ Démarrage de vLLM (peut prendre 2-3 minutes)...")
time.sleep(120)

# Vérifier que vLLM est prêt
try:
    resp = requests.get("http://localhost:8000/health", timeout=5)
    print("✅ vLLM est prêt!")
except:
    print("⚠️ vLLM n'est pas encore prêt, attendez un peu...")

# ============================================
# CELLULE 3 : Exposer via ngrok (optionnel)
# ============================================
!pip install -q pyngrok

from pyngrok import ngrok

# Exposer le port 8000
public_url = ngrok.connect(8000)
print(f"🌐 API accessible via : {public_url}")

# ============================================
# CELLULE 4 : Tester l'API
# ============================================
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "temperature": 0.7
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

---

## 🔧 Configuration pour votre backend

Si vous utilisez Colab avec ngrok, modifiez votre `.env` :

```bash
# Pour Colab avec ngrok
LLM_BASE_URL=https://xxxx-xxxx-xxxx.ngrok-free.app/v1
LLM_API_KEY=dummy-key
LLM_MODEL_NAME=TheBloke/Mistral-7B-Instruct-v0.2-AWQ
```

**⚠️ Note** : Colab + ngrok = URL publique, ne pas utiliser en production!

---

## 💡 Astuce : Colab Pro

Si vous utilisez Colab régulièrement :
- **Colab Pro** : $10/mois → GPU plus rapide, sessions plus longues
- **Colab Pro+** : $50/mois → A100, meilleure priorité

---

## 📊 Comparaison rapide

| Solution | Coût | Setup | Persistance | Production-ready |
|----------|------|-------|-------------|------------------|
| **Colab** | Gratuit | ⭐⭐⭐⭐⭐ | ❌ | ❌ |
| **GCP T4** | ~$250/mois | ⭐⭐⭐ | ✅ | ✅ |
| **AWS g4dn** | ~$360/mois | ⭐⭐⭐ | ✅ | ✅ |
| **RunPod** | ~$0.30/h | ⭐⭐⭐⭐ | ✅ | ⚠️ |

---

## 🎯 Conclusion

**Pour commencer** : Utilisez **Google Colab** (gratuit, rapide)  
**Pour production** : Migrez vers **GCP** ou **AWS** avec Docker

Voulez-vous que je crée un notebook Colab prêt à l'emploi pour votre projet ?

