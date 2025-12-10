"""Script de migration/réindexation : CSV SQuAD -> Qdrant."""
import asyncio
import os
import sys
import traceback
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.config import Settings
from app.core.paths import SQUAD_CSV
from app.services.embeddings import EmbeddingService
from app.vectorstores.qdrant_store import QdrantVectorStore


# --- FIX : Batch size réduit à 32 (au lieu de 100) pour éviter le WriteTimeout ---
async def migrate_csv_to_qdrant(csv_path: str, batch_size: int = 32):
    """Migre les données du CSV SQuAD vers Qdrant."""
    load_dotenv()
    config = Settings()
    
    print("🔄 Initialisation des services...")
    embedding_service = EmbeddingService(config)
    vectorstore = QdrantVectorStore(config, embedding_service)
    
    try:
        await vectorstore.connect()
        print("🔄 Initialisation de la collection Qdrant...")
        await vectorstore.initialize_collection(dense_dim=384)
    except Exception as e:
        print(f"❌ Erreur critique à l'initialisation : {e}")
        traceback.print_exc()
        return

    print(f"📖 Chargement du CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ {len(df)} lignes chargées")
    
    # Nettoyage des NaN
    df = df.fillna("")
    
    documents = []
    metadatas = []
    ids = []
    
    print("🛠️ Préparation des données...")
    for idx, row in df.iterrows():
        title = str(row.get('title', ''))
        context = str(row.get('context', ''))
        answers = str(row.get('answers', ''))
        
        content = f"Title: {title}\nContext: {context}\nAnswer: {answers}"
        documents.append(content)
        
        metadata = {
            "id": str(row.get('id', '')),
            "title": title,
            "question": str(row.get('question', '')),
            "context": context,
            "answers": answers,
            "page_content": content
        }
        metadatas.append(metadata)
        ids.append(str(row.get('id', '')))
    
    print(f"🚀 Indexation de {len(documents)} documents dans Qdrant (Batch size: {batch_size})...")
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    success_count = 0
    error_count = 0

    for i in tqdm(range(0, len(documents), batch_size), desc="Indexation", total=total_batches):
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        try:
            await vectorstore.add_documents(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
            success_count += 1
        except Exception as e:
            error_count += 1
            tqdm.write(f"\n❌ ERREUR BATCH {i//batch_size + 1}: {str(e)}")
            # On n'affiche le traceback complet que si c'est une nouvelle erreur
            if "Timeout" not in str(e):
                tqdm.write(traceback.format_exc())
            
            if error_count >= 10:
                print("\n🛑 Trop d'erreurs, arrêt d'urgence.")
                break
            continue
    
    await vectorstore.disconnect()
    print(f"\n✅ Migration terminée : {success_count} batches réussis, {error_count} échoués.")


if __name__ == "__main__":
    csv_path = str(SQUAD_CSV)
    if not os.path.exists(csv_path):
        print(f"❌ Fichier CSV introuvable: {csv_path}")
        sys.exit(1)
    
    asyncio.run(migrate_csv_to_qdrant(csv_path))