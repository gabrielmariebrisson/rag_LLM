"""Wrapper Qdrant pour recherche vectorielle hybride."""
from typing import List, Optional, Dict, Any
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SparseVectorParams,
    SparseVector,
)
from qdrant_client.http import models

from app.core.config import Settings
from app.services.embeddings import EmbeddingService


class QdrantVectorStore:
    """Wrapper pour Qdrant avec support recherche hybride (dense + sparse)."""
    
    def __init__(self, config: Settings, embedding_service: EmbeddingService):
        self.config = config
        self.embedding_service = embedding_service
        self.client: Optional[AsyncQdrantClient] = None
        self.collection_name = config.QDRANT_COLLECTION_NAME
    
    async def connect(self):
        """Établit la connexion avec Qdrant."""
        if self.client is None:
            self.client = AsyncQdrantClient(
                host=self.config.QDRANT_HOST,
                port=self.config.QDRANT_PORT,
                check_compatibility=False,  # Désactiver le check de compatibilité pour Qdrant 1.11.0
            )
    
    async def disconnect(self):
        """Ferme la connexion avec Qdrant."""
        if self.client:
            await self.client.close()
            self.client = None
    
    async def initialize_collection(self, dense_dim: int = 384):
        """
        Initialise la collection Qdrant avec configuration hybride.
        
        Args:
            dense_dim: Dimension des embeddings denses (384 pour all-MiniLM-L6-v2)
        """
        await self.connect()
        
        # Vérifier si la collection existe
        collections = await self.client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if self.collection_name in collection_names:
            print(f"✅ Collection '{self.collection_name}' existe déjà")
            return
        
        # Créer la collection avec configuration hybride
        # Pour Qdrant 1.11.0, utiliser des paramètres séparés pour dense et sparse
        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=dense_dim,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=models.SparseIndexParams()
                )
            }
        )
        print(f"✅ Collection '{self.collection_name}' créée avec recherche hybride")
    
    async def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ):
        """
        Ajoute des documents à la collection Qdrant.
        
        Args:
            documents: Liste des textes de documents
            metadatas: Liste des métadonnées pour chaque document
            ids: IDs optionnels pour les documents (générés automatiquement si None)
        """
        await self.connect()
        
        # Générer embeddings hybrides
        dense_embeddings, sparse_embeddings = await self.embedding_service.embed_hybrid(documents)
        
        # Préparer les points pour Qdrant
        points = []
        for i, (doc, dense_emb, sparse_emb, metadata) in enumerate(
            zip(documents, dense_embeddings, sparse_embeddings, metadatas)
        ):
            # Qdrant n'accepte que des entiers non signés ou des UUIDs
            # Convertir l'ID en entier si c'est une chaîne
            if ids and ids[i]:
                id_str = str(ids[i])
                try:
                    # D'abord, vérifier si c'est une chaîne hexadécimale (commence souvent par des chiffres mais contient a-f)
                    if isinstance(ids[i], str) and len(id_str) > 10 and any(c in "abcdefABCDEF" for c in id_str):
                        # C'est probablement une chaîne hexadécimale
                        point_id = int(id_str, 16) % (2**63)
                    else:
                        # Essayer de convertir en entier directement
                        point_id = int(id_str)
                        # S'assurer que c'est dans la plage valide pour Qdrant (0 à 2^63-1)
                        if point_id < 0 or point_id >= 2**63:
                            point_id = point_id % (2**63)
                except (ValueError, TypeError):
                    # Si la conversion échoue, utiliser un hash de la chaîne
                    point_id = abs(hash(id_str)) % (2**63)  # Entier non signé 64 bits
            else:
                point_id = i
            
            # Convertir sparse dict en format Qdrant
            sparse_indices = list(sparse_emb.keys())
            sparse_values = list(sparse_emb.values())
            
            point = PointStruct(
                id=point_id,
                vector={
                    "dense": dense_emb,
                    "sparse": SparseVector(
                        indices=sparse_indices,
                        values=sparse_values
                    )
                },
                payload=metadata
            )
            points.append(point)
        
        # Insérer les points
        await self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 20,
        filter: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        """
        Effectue une recherche hybride (dense + sparse) dans Qdrant.
        
        Args:
            query: Requête de recherche
            top_k: Nombre de résultats à retourner
            filter: Filtre optionnel pour les métadonnées
            
        Returns:
            Liste de résultats avec score, payload et id
        """
        await self.connect()
        
        # Générer embeddings hybrides pour la requête
        dense_embeddings, sparse_embeddings = await self.embedding_service.embed_hybrid([query])
        query_dense = dense_embeddings[0]
        query_sparse = sparse_embeddings[0]
        
        # Convertir sparse dict en format Qdrant
        sparse_indices = list(query_sparse.keys())
        sparse_values = list(query_sparse.values())
        
        # Recherche hybride dans Qdrant
        # Effectuer deux recherches séparées (dense et sparse) puis fusionner
        from qdrant_client.models import SearchRequest
        
        # Recherche dense
        dense_results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=("dense", query_dense),
            limit=top_k * 2,  # Prendre plus de résultats pour meilleure fusion
            query_filter=filter,
        )
        
        # Recherche sparse
        sparse_results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=("sparse", SparseVector(
                indices=sparse_indices,
                values=sparse_values
            )),
            limit=top_k * 2,
            query_filter=filter,
        )
        
        # Fusion RRF (Reciprocal Rank Fusion)
        # Créer un dictionnaire id -> score RRF
        rrf_scores = {}
        k = 60  # Paramètre RRF
        
        for rank, result in enumerate(dense_results, 1):
            point_id = result.id
            rrf_score = 1.0 / (k + rank)
            rrf_scores[point_id] = rrf_scores.get(point_id, 0) + rrf_score
        
        for rank, result in enumerate(sparse_results, 1):
            point_id = result.id
            rrf_score = 1.0 / (k + rank)
            rrf_scores[point_id] = rrf_scores.get(point_id, 0) + rrf_score
        
        # Créer un mapping id -> result complet
        all_results = {}
        for result in dense_results + sparse_results:
            if result.id not in all_results:
                all_results[result.id] = result
        
        # Trier par score RRF et prendre top_k
        sorted_results = sorted(
            all_results.items(),
            key=lambda x: rrf_scores.get(x[0], 0),
            reverse=True
        )[:top_k]
        
        # Formater les résultats
        search_result_points = [result for _, result in sorted_results]
        
        # Formater les résultats
        results = []
        for point in search_result_points:
            results.append({
                "id": point.id,
                "score": rrf_scores.get(point.id, 0.0),
                "payload": point.payload,
                "page_content": point.payload.get("page_content", ""),
                "metadata": {k: v for k, v in point.payload.items() if k != "page_content"}
            })
        
        return results

