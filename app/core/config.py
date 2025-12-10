"""
Configuration centralisée de l'application avec pydantic-settings.

Ce module définit toutes les variables d'environnement et leur validation.
Les valeurs sont chargées depuis le fichier .env ou les variables d'environnement système.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """
    Configuration de l'application via variables d'environnement.
    
    Toutes les variables peuvent être définies dans un fichier .env ou comme
    variables d'environnement système. Les valeurs par défaut sont utilisées
    si aucune valeur n'est fournie.
    
    Attributes:
        LLM_BASE_URL (Optional[str]): URL de base pour le LLM. None = Mistral API,
            "http://localhost:8001/v1" = vLLM local, "https://api.openai.com/v1" = OpenAI API
        LLM_API_KEY (Optional[str]): Clé API pour OpenAI/Mistral (optionnel pour vLLM local)
        LLM_MODEL_NAME (str): Nom du modèle LLM à utiliser
        MISTRAL_API_KEY (Optional[str]): Clé API Mistral (legacy, backward compatibility)
        MISTRAL_MODEL_NAME (Optional[str]): Nom du modèle Mistral (legacy)
        FAISS_INDEX_DIR (str): Répertoire contenant l'index FAISS (legacy, migration)
        EMBEDDING_MODEL_NAME (str): Nom du modèle d'embedding (legacy)
        QDRANT_HOST (str): Host de Qdrant (localhost ou cluster cloud)
        QDRANT_PORT (int): Port de Qdrant (défaut: 6333)
        QDRANT_COLLECTION_NAME (str): Nom de la collection Qdrant
        USE_RERANKER (bool): Activer/désactiver le reranking
        RERANKER_TOP_K (int): Nombre de documents avant reranking
        DENSE_MODEL (str): Modèle dense pour embeddings (SentenceTransformer, GPU optimisé)
        SPARSE_MODEL (str): Modèle sparse pour embeddings (SparseEncoder SPLADE, GPU optimisé)
        RERANKER_MODEL (str): Modèle Cross-Encoder pour reranking
        DENSE_DIM (int): Dimension des embeddings denses
        BACKEND_URL (str): URL du backend FastAPI (pour le frontend)
        HUGGING_FACE_HUB_TOKEN (Optional[str]): Token HuggingFace pour télécharger les modèles (requis pour certains modèles gated)
        
    Notes:
        - La validation LLM est effectuée automatiquement au chargement
        - Voir validate_llm_config() pour la logique de validation
        - Le fichier .env est chargé automatiquement depuis la racine du projet
    """
    
    # LLM Configuration (agnostique : OpenAI, Mistral API, ou vLLM local)
    LLM_BASE_URL: Optional[str] = None  # None = Mistral API, "http://localhost:8001/v1" = vLLM local
    LLM_API_KEY: Optional[str] = None  # Optionnel, requis seulement pour APIs externes
    LLM_MODEL_NAME: str = "casperhansen/llama-3-8b-instruct-awq"  # Nom du modèle
    
    # Legacy Mistral API (pour backward compatibility)
    MISTRAL_API_KEY: Optional[str] = None
    MISTRAL_MODEL_NAME: Optional[str] = None
    
    # FAISS Index (legacy, pour migration)
    FAISS_INDEX_DIR: str = "faiss_index"
    
    # Embedding Model (legacy)
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"
    
    # Qdrant Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "squad_collection"
    
    # Reranker Configuration
    USE_RERANKER: bool = True
    RERANKER_TOP_K: int = 10  # Nombre de docs avant reranking (optimal: 10 = même recall que 50, 7x plus rapide)
    
    # Embedding Models
    DENSE_MODEL: str = "BAAI/bge-large-en-v1.5"
    SPARSE_MODEL: str = "prithivida/Splade_PP_en_v1"  # Modèle SPLADE pour embeddings sparse (nom correct avec "i")
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    
    # Embedding Dimensions
    DENSE_DIM: int = 1024  # Dimension des embeddings denses (BGE-M3)
    
    # Backend URL (pour le frontend)
    BACKEND_URL: str = "http://localhost:8000"
    
    # HuggingFace Hub Token (pour télécharger les modèles)
    HUGGING_FACE_HUB_TOKEN: Optional[str] = None
    
    # GPU Configuration
    CUDA_DEVICE_ID: Optional[int] = None  # None = auto (cuda:0), 0 = cuda:0, 1 = cuda:1, etc.
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    def validate_llm_config(self) -> None:
        """
        Valide et normalise la configuration LLM.
        
        Détermine automatiquement quel provider LLM utiliser (vLLM, OpenAI, Mistral)
        en fonction des variables définies et applique les valeurs par défaut nécessaires.
        
        Raises:
            ValueError: Si aucune configuration LLM valide n'est trouvée.
                Message d'erreur indique les options disponibles.
                
        Notes:
            - Si LLM_BASE_URL est défini, utilise vLLM local ou OpenAI API
            - Si LLM_BASE_URL est None, utilise Mistral API (legacy) si MISTRAL_API_KEY est défini
            - vLLM local accepte n'importe quelle clé API (dummy-key si non fournie)
            - Les chaînes vides sont normalisées en None
            - Backward compatibility: MISTRAL_API_KEY peut être utilisé à la place de LLM_API_KEY
        """
        # Normaliser LLM_BASE_URL (chaîne vide = None)
        if self.LLM_BASE_URL and not self.LLM_BASE_URL.strip():
            self.LLM_BASE_URL = None
        
        # Backward compatibility: si MISTRAL_API_KEY est défini mais pas LLM_API_KEY
        if self.MISTRAL_API_KEY and (not self.LLM_API_KEY or not self.LLM_API_KEY.strip()):
            self.LLM_API_KEY = self.MISTRAL_API_KEY
        if self.MISTRAL_MODEL_NAME and self.LLM_MODEL_NAME == "casperhansen/llama-3-8b-instruct-awq":
            # Utiliser MISTRAL_MODEL_NAME si LLM_MODEL_NAME est encore à la valeur par défaut
            self.LLM_MODEL_NAME = self.MISTRAL_MODEL_NAME
        
        # Si LLM_BASE_URL est défini, on utilise vLLM local ou OpenAI API
        if self.LLM_BASE_URL:
            # vLLM local n'a pas besoin de vraie clé API
            if not self.LLM_API_KEY or not self.LLM_API_KEY.strip():
                self.LLM_API_KEY = "dummy-key"  # vLLM accepte n'importe quelle clé
        # Sinon, on utilise Mistral API (legacy) ou OpenAI API
        elif not self.LLM_API_KEY or not self.LLM_API_KEY.strip():
            raise ValueError(
                "LLM_API_KEY or MISTRAL_API_KEY must be set. "
                "For vLLM local, set LLM_BASE_URL=http://localhost:8001/v1. "
                "For Mistral API, set MISTRAL_API_KEY. "
                "For OpenAI API, set LLM_API_KEY."
            )


# Instance globale de configuration
settings = Settings()

# Validation au chargement du module
settings.validate_llm_config()

