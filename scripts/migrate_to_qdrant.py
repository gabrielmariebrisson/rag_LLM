"""Script de migration/réindexation OPTIMISÉ : CSV SQuAD -> Qdrant."""
import asyncio
import os
import sys
import traceback
import uuid  # <--- Important
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List, Dict

# Ajout du path pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.config import Settings
from app.core.paths import SQUAD_CSV
from app.services.embeddings import EmbeddingService
from app.vectorstores.qdrant_store import QdrantVectorStore

def create_sentence_windows(text: str, window_size: int = 3) -> List[str]:
    """Découpe un texte en fenêtres glissantes de phrases."""
    if not isinstance(text, str) or not text.strip():
        return []
    
    # Nettoyage et split basique
    text_clean = text.replace('?', '?|').replace('.', '.|').replace('!', '!|')
    sentences = [s.strip() for s in text_clean.split('|') if s.strip()]
    
    if not sentences:
        return []
        
    windows = []
    stride = 2
    for i in range(0, len(sentences), stride):
        window = sentences[i : i + window_size]
        if window:
            windows.append(" ".join(window))
    return windows

async def migrate_csv_to_qdrant(csv_path: str, batch_size: int = 32):
    load_dotenv()
    config = Settings()
    
    print("🔄 Initialisation des services...")
    embedding_service = EmbeddingService(config)
    vectorstore = QdrantVectorStore(config, embedding_service)
    
    try:
        await vectorstore.connect()
        # On recrée la collection pour être propre
        await vectorstore.initialize_collection(dense_dim=config.DENSE_DIM) 
    except Exception as e:
        print(f"❌ Erreur critique à l'initialisation : {e}")
        traceback.print_exc()
        return

    print(f"📖 Chargement du CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # DÉDUPLICATION
    print(f"📊 Lignes brutes: {len(df)}")
    unique_df = df.drop_duplicates(subset=['context'])
    unique_df = unique_df.fillna("")
    print(f"📉 Contextes Uniques: {len(unique_df)}")
    
    documents_to_embed = []
    metadatas = []
    ids = []
    
    print("🛠️ Préparation des fenêtres...")
    
    for idx, row in tqdm(unique_df.iterrows(), total=len(unique_df)):
        context = str(row.get('context', ''))
        title = str(row.get('title', ''))
        
        windows = create_sentence_windows(context, window_size=3)
        if not windows and context:
            windows = [context]
            
        for window in windows:
            documents_to_embed.append(window)
            
            # ID DÉTERMINISTE COMPATIBLE QDRANT (UUID v5)
            # uuid.NAMESPACE_DNS est un namespace standard.
            # window est le contenu. Le résultat est un UUID valide string.
            doc_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, window))
            ids.append(doc_uuid)
            
            # Metadatas complètes
            metadatas.append({
                "title": title,
                "context": context,        
                "full_text": context,
                "source": "squad_2.0",
                "type": "window_chunk",
                "content_hash": doc_uuid # On stocke l'ID aussi comme hash pour le debug
            })

    print(f"🚀 Indexation de {len(documents_to_embed)} chunks...")
    
    total_batches = (len(documents_to_embed) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(documents_to_embed), batch_size), total=total_batches):
        batch_docs = documents_to_embed[i : i + batch_size]
        batch_metas = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        
        try:
            await vectorstore.add_documents(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
        except Exception as e:
            if "already exists" not in str(e):
                tqdm.write(f"⚠️ Erreur Batch {i}: {str(e)}")
            continue
    
    await vectorstore.disconnect()
    print(f"\n✅ Migration terminée.")

if __name__ == "__main__":
    csv_path = str(SQUAD_CSV)
    if not os.path.exists(csv_path):
        csv_path = "data/raw/squad_2.0/train.csv"
    
    if os.path.exists(csv_path):
        asyncio.run(migrate_csv_to_qdrant(csv_path))
    else:
        print("❌ CSV introuvable.")