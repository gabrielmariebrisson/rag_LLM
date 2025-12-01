"""Configuration centralisée avec pydantic-settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Configuration de l'application via variables d'environnement."""
    
    # Mistral API
    MISTRAL_API_KEY: str
    MISTRAL_MODEL_NAME: str = "mistral-tiny-2407"
    
    # FAISS Index
    FAISS_INDEX_DIR: str = "faiss_index"
    
    # Embedding Model
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    
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

