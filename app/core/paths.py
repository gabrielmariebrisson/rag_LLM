"""Helper pour les chemins du projet."""
from pathlib import Path

# Racine du projet (2 niveaux au-dessus de app/core)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Chemins de données
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_RESULTS = DATA_DIR / "results"
DATA_SNAPSHOTS = DATA_DIR / "snapshots"

# Chemins de config
CONFIG_DIR = PROJECT_ROOT / "config"

# Chemins de logs
LOGS_DIR = PROJECT_ROOT / "logs"

# Chemins spécifiques
SQUAD_CSV = DATA_RAW / "squad_2.0" / "train.csv"
QDRANT_CONFIG = CONFIG_DIR / "qdrant_config.yaml"
QDRANT_DATA = DATA_PROCESSED / "qdrant_data"
EVALUATION_RESULTS = DATA_RESULTS / "evaluation_results"

