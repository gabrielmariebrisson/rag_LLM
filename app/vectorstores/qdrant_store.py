"""
Wrapper Qdrant pour recherche vectorielle hybride.
"""
from typing import List, Optional, Dict, Any
import numpy as np
import hashlib
import uuid
import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SparseVectorParams,
    SparseVector,
)
from qdrant_client.http import models

from app.core.config import Settings
from app.services.embeddings import EmbeddingService


class QdrantVectorStore:
    
    def __init__(self, config: Settings, embedding_service: EmbeddingService):
        self.config = config
        self.embedding_service = embedding_service
        self.client: Optional[AsyncQdrantClient] = None
        self.collection_name = config.QDRANT_COLLECTION_NAME
    
    async def connect(self):
        if self.client is None:
            self.client = AsyncQdrantClient(
                host=self.config.QDRANT_HOST,
                port=self.config.QDRANT_PORT,
                check_compatibility=False,
                timeout=120.0 
            )
    
    async def disconnect(self):
        if self.client:
            await self.client.close()
            self.client = None
    
    async def initialize_collection(self, dense_dim: Optional[int] = None):
        await self.connect()
        collections = await self.client.get_collections()
        if self.collection_name in [col.name for col in collections.collections]:
            return
        
        # Utiliser DENSE_DIM depuis config si dense_dim n'est pas fourni
        if dense_dim is None:
            dense_dim = self.config.DENSE_DIM
        
        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={"dense": VectorParams(size=dense_dim, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams(index=models.SparseIndexParams())}
        )
    
    def _validate_id(self, id_val: Any) -> Any:
        """
        Valide l'ID pour Qdrant.
        Qdrant accepte: int (unsigned 64bit) OU str (UUID format).
        """
        # 1. Si c'est déjà un int, c'est bon
        if isinstance(id_val, int):
            return id_val
            
        # 2. Si c'est une string, on vérifie si c'est un UUID valide
        if isinstance(id_val, str):
            try:
                uuid.UUID(id_val)
                return id_val # C'est un UUID valide, on garde la string
            except ValueError:
                pass # Ce n'est pas un UUID
                
            # 3. Si c'est une string numérique ("123"), on convertit en int
            if id_val.isdigit():
                return int(id_val)
        
        # 4. Fallback ultime : On hash en entier déterministe
        # (Pour gérer les vieux IDs ou les formats exotiques)
        return int(hashlib.md5(str(id_val).encode()).hexdigest(), 16) % (2**63)

    async def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ):
        await self.connect()
        
        dense_embeddings, sparse_embeddings = await self.embedding_service.embed_hybrid(documents)
        
        points = []
        for i, (doc, dense_emb, sparse_emb, metadata) in enumerate(
            zip(documents, dense_embeddings, sparse_embeddings, metadatas)
        ):
            raw_id = ids[i] if ids and i < len(ids) else str(i)
            
            # Utilisation de la validation robuste
            point_id = self._validate_id(raw_id)
            
            if isinstance(dense_emb, np.ndarray):
                dense_vec = dense_emb.tolist()
            else:
                dense_vec = dense_emb
            
            sparse_indices = [int(idx) for idx in sparse_emb.keys()]
            sparse_values = [float(val) for val in sparse_emb.values()]
            if not sparse_indices: sparse_indices, sparse_values = [0], [0.0]

            points.append(PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vec,
                    "sparse": SparseVector(indices=sparse_indices, values=sparse_values)
                },
                payload=metadata
            ))
        
        if points:
            await self.client.upsert(collection_name=self.collection_name, points=points)

    async def hybrid_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        await self.connect()
        
        # 1. Embeddings
        dense_embeddings, sparse_embeddings = await self.embedding_service.embed_hybrid([query])
        query_dense = dense_embeddings[0]
        if isinstance(query_dense, np.ndarray): query_dense = query_dense.tolist()
        
        query_sparse = sparse_embeddings[0]
        # Convertir les clés en int (Qdrant attend des indices entiers)
        sparse_indices = [int(idx) for idx in query_sparse.keys()]
        sparse_values = [float(val) for val in query_sparse.values()]
        if not sparse_indices: sparse_indices, sparse_values = [0], [0.0]
        
        # 2. Retrieval (x3 candidats pour déduplication)
        search_limit = top_k * 3
        
        # Utiliser l'API REST directement pour compatibilité avec Qdrant 1.7.0
        # Le client Python 1.16.1 n'est pas compatible avec query_points + using pour Qdrant 1.7.0
        base_url = f"http://{self.config.QDRANT_HOST}:{self.config.QDRANT_PORT}"
        
        async with httpx.AsyncClient(timeout=120.0) as http_client:
            # Recherche dense via API REST
            dense_payload = {
                "vector": {
                    "name": "dense",
                    "vector": query_dense
                },
                "limit": search_limit,
                "with_payload": True
            }
            dense_response = await http_client.post(
                f"{base_url}/collections/{self.collection_name}/points/search",
                json=dense_payload
            )
            dense_response.raise_for_status()
            dense_data = dense_response.json()
            
            # Créer un objet compatible avec QueryResponse
            class SimplePoint:
                def __init__(self, point_id, score, payload):
                    self.id = point_id
                    self.score = score
                    self.payload = payload
            
            class SimpleQueryResponse:
                def __init__(self, points):
                    self.points = points
            
            dense_points = [
                SimplePoint(
                    point_id=item.get("id"),
                    score=item.get("score", 0.0),
                    payload=item.get("payload", {})
                )
                for item in dense_data.get("result", [])
            ]
            dense_results = SimpleQueryResponse(points=dense_points)

            # Recherche sparse via API REST
            sparse_payload = {
                "vector": {
                    "name": "sparse",
                    "vector": {
                        "indices": sparse_indices,
                        "values": sparse_values
                    }
                },
                "limit": search_limit,
                "with_payload": True
            }
            sparse_response = await http_client.post(
                f"{base_url}/collections/{self.collection_name}/points/search",
                json=sparse_payload
            )
            sparse_response.raise_for_status()
            sparse_data = sparse_response.json()
            
            # Créer un objet compatible avec QueryResponse
            sparse_points = [
                SimplePoint(
                    point_id=item.get("id"),
                    score=item.get("score", 0.0),
                    payload=item.get("payload", {})
                )
                for item in sparse_data.get("result", [])
            ]
            sparse_results = SimpleQueryResponse(points=sparse_points)
        
        # 3. RRF Fusion
        rrf_scores = {}
        k_param = 60
        
        # On utilise .points car query_points retourne un objet QueryResponse
        for rank, result in enumerate(dense_results.points, 1):
            rrf_scores[result.id] = rrf_scores.get(result.id, 0) + (1.0 / (k_param + rank))
        
        for rank, result in enumerate(sparse_results.points, 1):
            rrf_scores[result.id] = rrf_scores.get(result.id, 0) + (1.0 / (k_param + rank))
        
        # 4. Déduplication
        all_results_map = {res.id: res for res in list(dense_results.points) + list(sparse_results.points)}
        sorted_candidates = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        
        final_results = []
        seen_hashes = set()
        
        for point_id, score in sorted_candidates:
            point = all_results_map.get(point_id)
            if not point or not point.payload: continue
                
            # Récupération du hash pour déduplication
            # On utilise 'content_hash' s'il existe (mis par le script de migration)
            # Sinon on le recalcule sur le champ text/context
            content = point.payload.get("context") or point.payload.get("text") or point.payload.get("page_content", "")
            doc_hash = point.payload.get("content_hash")
            
            if not doc_hash:
                doc_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            if doc_hash in seen_hashes:
                continue
            
            seen_hashes.add(doc_hash)
            
            final_results.append({
                "id": point.id,
                "score": score,
                "payload": point.payload,
                "page_content": content,
                "metadata": point.payload
            })
            
            if len(final_results) >= top_k:
                break
        
        return final_results