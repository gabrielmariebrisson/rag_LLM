"""Script de migration/réindexation : CSV SQuAD -> Qdrant."""
import asyncio
import os
import sys
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.config import Settings
from app.services.embeddings import EmbeddingService
from app.vectorstores.qdrant_store import QdrantVectorStore


async def migrate_csv_to_qdrant(csv_path: str, batch_size: int = 100):
    """
    Migre les données du CSV SQuAD vers Qdrant.
    
    Args:
        csv_path: Chemin vers le fichier CSV SQuAD
        batch_size: Taille des batches pour l'indexation
    """
    # Charger la configuration
    load_dotenv()
    config = Settings()
    
    # Initialiser les services
    print("🔄 Initialisation des services...")
    embedding_service = EmbeddingService(config)
    vectorstore = QdrantVectorStore(config, embedding_service)
    await vectorstore.connect()
    
    # Initialiser la collection
    print("🔄 Initialisation de la collection Qdrant...")
    await vectorstore.initialize_collection(dense_dim=384)
    
    # Charger le CSV
    print(f"📖 Chargement du CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ {len(df)} lignes chargées")
    
    # Préparer les documents
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in df.iterrows():
        # Créer le contenu du document
        content = f"Title: {row['title']}\nContext: {row['context']}\nAnswer: {row['answers']}"
        documents.append(content)
        
        # Métadonnées
        metadata = {
            "page_content": content,
            "id": row['id'],
            "title": row['title'],
            "question": row['question'],
            "context": row['context'],
            "answers": str(row['answers']) if pd.notna(row['answers']) else ""
        }
        metadatas.append(metadata)
        ids.append(str(row['id']))
    
    # Indexer par batches
    print(f"🚀 Indexation de {len(documents)} documents dans Qdrant...")
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
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
        except Exception as e:
            print(f"❌ Erreur lors de l'indexation du batch {i//batch_size + 1}: {e}")
            continue
    
    # Fermer la connexion
    await vectorstore.disconnect()
    print("✅ Migration terminée avec succès!")


if __name__ == "__main__":
    csv_path = "squad_2.0/train.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ Fichier CSV introuvable: {csv_path}")
        print("💡 Assurez-vous que le fichier existe dans le répertoire racine du projet")
        sys.exit(1)
    
    print("=" * 60)
    print("Migration CSV SQuAD -> Qdrant")
    print("=" * 60)
    
    asyncio.run(migrate_csv_to_qdrant(csv_path))

