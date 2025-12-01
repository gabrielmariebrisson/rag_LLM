"""Configuration centralisée avec pydantic-settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Configuration de l'application via variables d'environnement."""
    
    # Mistral API
    MISTRAL_API_KEY: str
    MISTRAL_MODEL_NAME: str = "mistral-tiny-2407"
    
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
    SPARSE_MODEL: str = "prunebert-base-uncased-6-minilayer"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Backend URL (pour le frontend)
    BACKEND_URL: str = "http://localhost:8000"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    def validate_mistral_key(self) -> None:
        """Valide que la clé Mistral est définie et non vide."""
        if not self.MISTRAL_API_KEY or not self.MISTRAL_API_KEY.strip():
            raise ValueError(
                "MISTRAL_API_KEY is not set or is empty. "
                "Please set it in your .env file."
            )


# Instance globale de configuration
settings = Settings()

# Validation au chargement du module
settings.validate_mistral_key()

