"""Fixtures pytest partagées."""
import pytest
import asyncio
from pathlib import Path
from app.core.config import Settings


@pytest.fixture
def mock_settings():
    """Configuration de test."""
    return Settings(
        LLM_BASE_URL="http://localhost:8001/v1",
        QDRANT_HOST="localhost",
        QDRANT_PORT=6333,
        QDRANT_COLLECTION_NAME="test_collection",
        USE_RERANKER=False  # Désactiver reranker pour tests plus rapides
    )


@pytest.fixture
def project_root():
    """Racine du projet."""
    return Path(__file__).parent.parent.resolve()


@pytest.fixture
def event_loop():
    """Event loop pour tests async."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

