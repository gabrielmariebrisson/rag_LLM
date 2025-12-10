#!/usr/bin/env python3
"""Script pour nettoyer la VRAM et Qdrant."""
import asyncio
import os
import sys
import subprocess
from dotenv import load_dotenv

# Ajout du path pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.config import Settings
from qdrant_client import AsyncQdrantClient


async def cleanup_qdrant():
    """Supprime la collection Qdrant."""
    load_dotenv()
    config = Settings()
    
    print("🔄 Connexion à Qdrant...")
    client = AsyncQdrantClient(
        host=config.QDRANT_HOST,
        port=config.QDRANT_PORT,
        timeout=60.0
    )
    
    try:
        collections = await client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if config.QDRANT_COLLECTION_NAME in collection_names:
            print(f"🗑️  Suppression de la collection '{config.QDRANT_COLLECTION_NAME}'...")
            await client.delete_collection(config.QDRANT_COLLECTION_NAME)
            print(f"✅ Collection '{config.QDRANT_COLLECTION_NAME}' supprimée")
        else:
            print(f"ℹ️  La collection '{config.QDRANT_COLLECTION_NAME}' n'existe pas")
        
        # Afficher les collections restantes
        collections = await client.get_collections()
        if collections.collections:
            print(f"📋 Collections restantes: {[col.name for col in collections.collections]}")
        else:
            print("📋 Aucune collection restante")
            
    except Exception as e:
        print(f"❌ Erreur lors du nettoyage Qdrant: {e}")
    finally:
        await client.close()


def cleanup_vram():
    """Libère la VRAM en arrêtant les processus et en vidant le cache."""
    print("🔄 Nettoyage de la VRAM...")
    
    # 1. Arrêter vLLM si en cours
    try:
        result = subprocess.run(
            ["pgrep", "-f", "vllm.entrypoints.openai.api_server"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    print(f"🛑 Arrêt du processus vLLM (PID: {pid})...")
                    subprocess.run(["kill", "-9", pid], check=False)
                    print(f"✅ Processus {pid} arrêté")
        else:
            print("ℹ️  Aucun processus vLLM trouvé")
    except Exception as e:
        print(f"⚠️  Erreur lors de l'arrêt de vLLM: {e}")
    
    # 2. Vider le cache PyTorch si disponible
    try:
        import torch
        if torch.cuda.is_available():
            print("🧹 Vidage du cache CUDA...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✅ Cache CUDA vidé")
            
            # Afficher l'utilisation mémoire
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"   GPU {i}: {allocated:.2f} GB alloué, {reserved:.2f} GB réservé")
        else:
            print("ℹ️  CUDA non disponible")
    except ImportError:
        print("ℹ️  PyTorch non installé, impossible de vider le cache")
    except Exception as e:
        print(f"⚠️  Erreur lors du vidage du cache: {e}")
    
    # 3. Afficher l'état final avec nvidia-smi
    print("\n📊 État de la VRAM après nettoyage:")
    try:
        subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"], check=False)
    except Exception as e:
        print(f"⚠️  Impossible d'exécuter nvidia-smi: {e}")


async def main():
    """Fonction principale."""
    print("🧹 Nettoyage de la VRAM et Qdrant\n")
    
    # Nettoyer Qdrant
    print("=" * 50)
    print("1. NETTOYAGE QDRANT")
    print("=" * 50)
    await cleanup_qdrant()
    
    print("\n" + "=" * 50)
    print("2. NETTOYAGE VRAM")
    print("=" * 50)
    cleanup_vram()
    
    print("\n✅ Nettoyage terminé!")


if __name__ == "__main__":
    asyncio.run(main())
