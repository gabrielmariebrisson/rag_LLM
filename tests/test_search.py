import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.config import Settings
from app.services.embeddings import EmbeddingService
from app.vectorstores.qdrant_store import QdrantVectorStore

async def test_search():
    config = Settings()
    service = EmbeddingService(config)
    store = QdrantVectorStore(config, service)
    
    query = "What causes rain?"  # Ou une question pertinente pour votre dataset SQuAD
    print(f"🔍 Test de recherche pour : '{query}'")
    
    results = await store.hybrid_search(query, top_k=3)
    
    for i, res in enumerate(results):
        print(f"\n📄 Résultat {i+1} (Score RRF: {res['score']:.4f})")
        print(f"   ID: {res['id']}")
        print(f"   Contenu: {res['page_content'][:150]}...")

if __name__ == "__main__":
    asyncio.run(test_search())