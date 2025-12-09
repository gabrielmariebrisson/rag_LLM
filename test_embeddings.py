"""Test rapide des embeddings."""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.core.config import Settings
from app.services.embeddings import EmbeddingService

async def test_embeddings():
    """Test des embeddings dense et sparse."""
    print("🔄 Test des embeddings...")
    
    # Désactiver hf_transfer pour éviter les problèmes
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    
    config = Settings()
    print(f"📝 Modèle dense: {config.DENSE_MODEL}")
    print(f"📝 Modèle sparse: {config.SPARSE_MODEL}")
    
    service = EmbeddingService(config)
    
    # Test avec un texte simple
    test_text = "What is the capital of France?"
    print(f"\n📄 Texte de test: {test_text}")
    
    try:
        dense_emb, sparse_emb = await service.embed_hybrid([test_text])
        print(f"✅ Embedding dense généré: shape {len(dense_emb[0])}")
        print(f"✅ Embedding sparse généré: {len(sparse_emb[0])} tokens non-nuls")
        print(f"   Exemple de tokens sparse: {list(sparse_emb[0].items())[:5]}")
        print("\n✅ Test réussi !")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(test_embeddings())

