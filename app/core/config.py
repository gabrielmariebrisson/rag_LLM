"""Configuration centralisée avec pydantic-settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Configuration de l'application via variables d'environnement."""
    
    # LLM Configuration (agnostique : OpenAI, Mistral API, ou vLLM local)
    LLM_BASE_URL: Optional[str] = None  # None = Mistral API, "http://localhost:8001/v1" = vLLM local
    LLM_API_KEY: Optional[str] = None  # Optionnel, requis seulement pour APIs externes
    LLM_MODEL_NAME: str = "mistral-tiny-2407"  # Nom du modèle
    
    # Legacy Mistral API (pour backward compatibility)
    MISTRAL_API_KEY: Optional[str] = None
    MISTRAL_MODEL_NAME: Optional[str] = None
    
    # FAISS Index (legacy, pour migration)
    FAISS_INDEX_DIR: str = "faiss_index"
    
    # Embedding Model (legacy)
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Qdrant Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "squad_collection"
    
    # Reranker Configuration
    USE_RERANKER: bool = True
    RERANKER_TOP_K: int = 20  # Nombre de docs avant reranking
    
    # Embedding Models
    DENSE_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    SPARSE_MODEL: str = "bert-base-uncased"  # Modèle BERT standard pour embeddings sparse (SPLADE-like)
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Backend URL (pour le frontend)
    BACKEND_URL: str = "http://localhost:8000"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    def validate_llm_config(self) -> None:
        """Valide la configuration LLM."""
        # Normaliser LLM_BASE_URL (chaîne vide = None)
        if self.LLM_BASE_URL and not self.LLM_BASE_URL.strip():
            self.LLM_BASE_URL = None
        
        # Backward compatibility: si MISTRAL_API_KEY est défini mais pas LLM_API_KEY
        if self.MISTRAL_API_KEY and (not self.LLM_API_KEY or not self.LLM_API_KEY.strip()):
            self.LLM_API_KEY = self.MISTRAL_API_KEY
        if self.MISTRAL_MODEL_NAME and self.LLM_MODEL_NAME == "mistral-tiny-2407":
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

