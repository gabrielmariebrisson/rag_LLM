# 📊 Guide d'Évaluation

Guide complet pour évaluer les performances du système RAG.

## 📋 Table des Matières

- [Vue d'Ensemble](#vue-densemble)
- [Scripts d'Évaluation](#scripts-dévaluation)
- [Métriques](#métriques)
- [Interprétation des Résultats](#interprétation-des-résultats)
- [Exemples d'Utilisation](#exemples-dutilisation)
- [Optimisation](#optimisation)

---

## Vue d'Ensemble

Le système inclut des scripts d'évaluation pour mesurer :
- **Recall@K** : Précision de la récupération de documents
- **TTFT** : Time To First Token (latence LLM)
- **Latence** : Temps de traitement complet

Ces métriques permettent de :
- Comparer différentes configurations
- Évaluer l'impact du reranking
- Identifier les goulots d'étranglement
- Optimiser les performances

---

## Scripts d'Évaluation

### Script Principal : `evaluate_retrieval.py`

#### Description

Évalue les performances du système RAG en comparant :
- Recherche hybride **sans reranker**
- Recherche hybride **avec reranker**

#### Utilisation

```bash
python scripts/evaluate_retrieval.py
```

#### Paramètres (dans le script)

```python
results = asyncio.run(evaluate_retrieval(
    csv_path="squad_2.0/train.csv",
    sample_size=50,              # Nombre de questions à évaluer
    random_state=42,             # Seed pour reproductibilité
    k=5,                         # Nombre de documents à récupérer
    enable_ttft_measurement=True # Mesurer le TTFT
))
```

#### Paramètres Disponibles

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `csv_path` | str | `"data/raw/squad_2.0/train.csv"` | Chemin vers le CSV d'évaluation |
| `sample_size` | int | `50` | Nombre de questions à évaluer |
| `random_state` | int | `42` | Seed pour reproductibilité |
| `k` | int | `5` | Nombre de documents à récupérer initialement |
| `enable_ttft_measurement` | bool | `True` | Activer la mesure du TTFT |

#### Prérequis

1. **Dataset** : Fichier `data/raw/squad_2.0/train.csv` doit exister
2. **Qdrant** : Collection indexée et accessible
3. **Services** : Backend configuré (pour TTFT si activé)

#### Exécution

```bash
# Depuis la racine du projet
cd /workspace/rag_LLM

# Vérifier que Qdrant est démarré
curl http://localhost:6333/health

# Lancer l'évaluation
python scripts/evaluate_retrieval.py
```

#### Sortie

Le script génère :

1. **Console** : Résultats formatés en temps réel
2. **Fichiers** :
   - `data/results/evaluation_results/evaluation_results.json` : Résultats complets (JSON)
   - `data/results/evaluation_results/evaluation_details.csv` : Détails par question (CSV)
   - `data/results/evaluation_results/evaluation_summary.csv` : Résumé des métriques (CSV)

---

## Métriques

### Recall@K

#### Définition

**Recall@K** mesure si le document ground truth est présent dans les K premiers résultats récupérés.

**Formule** :
```
Recall@K = (Nombre de questions où le ground truth est dans top K) / (Total de questions)
```

#### Calcul

Le script utilise `calculate_recall_at_k()` qui :
1. Extrait le contexte depuis les métadonnées ou `page_content`
2. Compare avec le `ground_truth_context` du CSV
3. Utilise une comparaison exacte puis partielle (80% overlap)

#### Valeurs Possibles

- **0.0 - 1.0** : Rappel entre 0% et 100%
- **0.0** : Aucun document ground truth trouvé
- **1.0** : Tous les documents ground truth trouvés

#### Interprétation

- **> 0.8** : Excellent (80%+ de rappel)
- **0.6 - 0.8** : Bon (60-80% de rappel)
- **0.4 - 0.6** : Moyen (40-60% de rappel)
- **< 0.4** : À améliorer

### TTFT (Time To First Token)

#### Définition

**TTFT** mesure le temps entre l'envoi de la requête et la réception du premier token généré par le LLM.

#### Mesure

Le script utilise `measure_ttft()` qui :
1. **Si streaming disponible** : Mesure réelle du temps jusqu'au premier token
2. **Sinon** : Estimation basée sur la latence totale (30% de la latence totale)

#### Unités

- **Millisecondes (ms)**
- **Secondes (s)** : Pour valeurs élevées

#### Métriques TTFT

- **Moyenne** : TTFT moyen sur l'échantillon
- **P50** : Médiane (50e percentile)
- **P95** : 95e percentile
- **Min/Max** : Valeurs extrêmes

#### Interprétation

| TTFT | Performance | Recommandation |
|------|-------------|----------------|
| < 100ms | Excellent | Idéal pour production |
| 100-500ms | Bon | Acceptable pour la plupart des cas |
| 500-2000ms | Moyen | Considérer optimisation |
| > 2000ms | À améliorer | Vérifier configuration LLM |

### Latence

#### Définition

**Latence** mesure le temps total de traitement d'une requête :
- Recherche hybride
- (Optionnel) Reranking
- Génération LLM

#### Composants

1. **Latence sans reranker** :
   - Embedding query (~10-50ms)
   - Hybrid search (~20-100ms)
   - **Total** : ~30-150ms

2. **Latence avec reranker** :
   - Embedding query (~10-50ms)
   - Hybrid search (~20-100ms)
   - Reranking (~20-50ms)
   - **Total** : ~50-200ms

3. **Overhead reranker** :
   - Différence entre avec/sans reranker
   - Généralement ~20-50ms

#### Interprétation

- **< 100ms** : Excellent
- **100-500ms** : Bon
- **500-2000ms** : Moyen
- **> 2000ms** : À optimiser

---

## Interprétation des Résultats

### Format de Sortie

#### Console

```
============================================================
Résultats d'Évaluation RAG
============================================================

📊 Échantillon: 50 questions

🎯 Recall@5:
   Sans Reranker: 0.6200 (62.00%)
   Avec Reranker: 0.6400 (64.00%)
   Amélioration: +3.23%

🎯 Recall@10:
   Sans Reranker: 0.7800 (78.00%)
   Avec Reranker: 0.8100 (81.00%)
   Amélioration: +3.85%

⚡ Latence:
   Sans Reranker: 58.32 ms (moyenne)
   Avec Reranker: 79.65 ms (moyenne)
   Overhead: +21.32 ms

⚡ TTFT:
   Moyenne: 3276.50 ms
   P50: 2969.00 ms
   P95: 4512.30 ms
   Min: 1127.00 ms
   Max: 5234.00 ms
   Échantillon: 10 mesures
```

#### JSON (`evaluation_results.json`)

```json
{
  "sample_size": 50,
  "recall_at_5": {
    "without_reranker": 0.6200,
    "with_reranker": 0.6400,
    "improvement_percent": 3.23
  },
  "recall_at_10": {
    "without_reranker": 0.7800,
    "with_reranker": 0.8100,
    "improvement_percent": 3.85
  },
  "latency_ms": {
    "without_reranker_avg": 58.32,
    "with_reranker_avg": 79.65,
    "reranker_overhead_ms": 21.32
  },
  "ttft_stats": {
    "avg_ttft_ms": 3276.50,
    "p50_ttft_ms": 2969.00,
    "p95_ttft_ms": 4512.30,
    "min_ttft_ms": 1127.00,
    "max_ttft_ms": 5234.00,
    "sample_size": 10,
    "estimated_count": 10
  },
  "details": [...]
}
```

#### CSV (`evaluation_details.csv`)

Colonnes :
- `question` : Question évaluée
- `ground_truth_context` : Contexte attendu
- `recall_5_without_reranker` : Recall@5 sans reranker (0 ou 1)
- `recall_5_with_reranker` : Recall@5 avec reranker (0 ou 1)
- `recall_10_without_reranker` : Recall@10 sans reranker (0 ou 1)
- `recall_10_with_reranker` : Recall@10 avec reranker (0 ou 1)
- `time_without_reranker_ms` : Temps sans reranker (ms)
- `time_with_reranker_ms` : Temps avec reranker (ms)
- `ttft_ms` : TTFT mesuré (si disponible)
- `top_3_results_without` : Top 3 résultats sans reranker
- `top_3_results_with` : Top 3 résultats avec reranker

### Analyse des Résultats

#### Cas 1 : Recall@K Élevé (> 0.8)

**Interprétation** :
- ✅ Système performant
- ✅ Reranking apporte une amélioration modeste mais réelle

**Recommandations** :
- Conserver la configuration actuelle
- Optimiser la latence si nécessaire

#### Cas 2 : Recall@K Faible (< 0.5)

**Causes possibles** :
- Collection Qdrant mal indexée
- Embeddings de mauvaise qualité
- Dataset ground truth incomplet

**Solutions** :
1. Réindexer la collection : `python scripts/migrate_to_qdrant.py`
2. Vérifier les embeddings : Tester avec `scripts/test_search.py`
3. Augmenter `RERANKER_TOP_K` pour plus de candidats

#### Cas 3 : Amélioration Reranker Négligeable (< 1%)

**Interprétation** :
- Le reranking n'apporte pas de valeur ajoutée significative
- Overhead latence non justifié

**Recommandations** :
- Désactiver le reranker pour réduire la latence
- Ou augmenter `RERANKER_TOP_K` pour plus de candidats

#### Cas 4 : TTFT Élevé (> 3000ms)

**Causes possibles** :
- API externe (latence réseau)
- Modèle LLM trop lent
- Pas de GPU disponible

**Solutions** :
1. Utiliser vLLM local si GPU disponible
2. Réduire la taille du contexte
3. Utiliser un modèle plus petit/quantifié

---

## Exemples d'Utilisation

### Évaluation Standard

```bash
# Évaluation complète (50 questions, TTFT activé)
python scripts/evaluate_retrieval.py
```

### Évaluation Rapide (10 questions)

Modifier `sample_size=10` dans le script :

```python
results = asyncio.run(evaluate_retrieval(
    csv_path="squad_2.0/train.csv",
    sample_size=10,  # Réduire pour test rapide
    random_state=42,
    k=5,
    enable_ttft_measurement=False  # Désactiver TTFT pour aller plus vite
))
```

### Évaluation Complète (Tout le dataset)

```python
# Charger tout le dataset
df = pd.read_csv("data/raw/squad_2.0/train.csv")
sample_size = len(df)  # Évaluer toutes les questions
```

### Comparaison de Configurations

1. **Test sans reranker** :
   ```python
   # Dans .env
   USE_RERANKER=false
   ```
   Lancer évaluation → Sauvegarder résultats

2. **Test avec reranker** :
   ```python
   # Dans .env
   USE_RERANKER=true
   RERANKER_TOP_K=30
   ```
   Lancer évaluation → Comparer avec résultats précédents

### Visualisation des Résultats

#### Python/Pandas

```python
import pandas as pd
import matplotlib.pyplot as plt

# Charger les résultats
df = pd.read_csv("data/results/evaluation_results/evaluation_details.csv")

# Graphique Recall@K
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Recall@5
recall_5_without = df['recall_5_without_reranker'].mean()
recall_5_with = df['recall_5_with_reranker'].mean()
ax[0].bar(['Sans Reranker', 'Avec Reranker'], 
          [recall_5_without, recall_5_with])
ax[0].set_title('Recall@5')
ax[0].set_ylim(0, 1)

# Recall@10
recall_10_without = df['recall_10_without_reranker'].mean()
recall_10_with = df['recall_10_with_reranker'].mean()
ax[1].bar(['Sans Reranker', 'Avec Reranker'], 
          [recall_10_without, recall_10_with])
ax[1].set_title('Recall@10')
ax[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('data/results/evaluation_results/recall_comparison.png')
```

---

## Optimisation

### Améliorer Recall@K

#### 1. Ajuster les Modèles d'Embedding

```env
# Modèle dense plus performant (SOTA)
DENSE_MODEL=BAAI/bge-large-en-v1.5

# Modèle reranker plus performant (SOTA)
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

#### 2. Augmenter RERANKER_TOP_K

```env
# Plus de candidats avant reranking
RERANKER_TOP_K=10  # Optimal: même recall que 50, 7x plus rapide
```

#### 3. Optimiser Qdrant

```python
# HNSW parameters plus agressifs
hnsw_config=HnswConfigDiff(
    m=32,          # Plus de connexions (défaut: 16)
    ef_construct=200  # Plus de candidats à construire (défaut: 100)
)
```

### Réduire la Latence

#### 1. Désactiver Reranker

```env
USE_RERANKER=false
```

#### 2. Réduire RERANKER_TOP_K

```env
RERANKER_TOP_K=10  # Moins de candidats
```

#### 3. Utiliser GPU

- Vérifier que GPU est utilisé : `python -c "import torch; print(torch.cuda.is_available())"`
- Modèles embarquent automatiquement sur GPU si disponible

#### 4. Optimiser LLM

- Utiliser vLLM local au lieu d'API externe
- Quantifier le modèle (AWQ, GPTQ)
- Réduire la taille du contexte

### Améliorer TTFT

#### 1. Utiliser vLLM Local

```env
LLM_BASE_URL=http://localhost:8001/v1
LLM_MODEL_NAME=TheBloke/Mistral-7B-Instruct-v0.2-AWQ
```

#### 2. Streaming

Le script utilise le streaming si disponible pour mesurer le TTFT réel.

#### 3. Modèle Plus Petit

```env
# Modèle quantifié plus léger
LLM_MODEL_NAME=TheBloke/Mistral-7B-Instruct-v0.2-AWQ
```

---

## Troubleshooting

### Erreur : CSV introuvable

```
❌ Fichier CSV introuvable: data/raw/squad_2.0/train.csv
```

**Solution** :
```bash
# Convertir depuis JSON
python scripts/convert_squad_json_to_csv.py
```

### Erreur : Collection Qdrant vide

```
❌ Erreur lors de la recherche: Collection vide
```

**Solution** :
```bash
# Réindexer
python scripts/migrate_to_qdrant.py
```

### TTFT toujours None

**Cause** : Streaming non disponible ou LLM non configuré.

**Solution** :
- Vérifier configuration LLM dans `.env`
- Utiliser vLLM local ou OpenAI API avec streaming

### Résultats non reproductibles

**Cause** : `random_state` différent ou collection Qdrant modifiée.

**Solution** :
- Utiliser le même `random_state=42`
- Ne pas modifier la collection entre évaluations

---

## Ressources Additionnelles

- [README.md](../README.md) : Guide de démarrage rapide
- [CONFIGURATION.md](CONFIGURATION.md) : Guide de configuration
- [ARCHITECTURE.md](ARCHITECTURE.md) : Architecture technique

---

**Version** : 2.0.0  
**Dernière mise à jour** : 2025-01-27

