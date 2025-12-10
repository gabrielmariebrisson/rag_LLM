"""
Wrapper Qdrant pour recherche vectorielle hybride.

Ce module implémente un wrapper autour d'AsyncQdrantClient pour :
- Recherche hybride (dense + sparse vectors) avec fusion RRF
- Gestion des collections Qdrant
- Ajout et récupération de documents avec embeddings hybrides

La recherche hybride combine :
- Dense vectors : Embeddings sémantiques (sentence-transformers)
- Sparse vectors : Embeddings basés sur les mots-clés (BERT-based SPLADE-like)
"""
from typing import List, Optional, Dict, Any
import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    SparseVectorParams,
    SparseVector,
    NamedVector,        # Import nécessaire
    NamedSparseVector,  # Import nécessaire
)
from qdrant_client.http import models

from app.core.config import Settings
from app.services.embeddings import EmbeddingService


class QdrantVectorStore:
    """
    Wrapper pour Qdrant avec support recherche hybride (dense + sparse).
    
    Cette classe gère la connexion à Qdrant, la création de collections,
    l'ajout de documents avec embeddings hybrides, et la recherche hybride
    avec fusion RRF (Reciprocal Rank Fusion).
    
    Attributes:
        config (Settings): Configuration de l'application.
        embedding_service (EmbeddingService): Service pour générer les embeddings.
        client (Optional[AsyncQdrantClient]): Client Qdrant asynchrone.
        collection_name (str): Nom de la collection Qdrant.
        
    Notes:
        - La collection utilise deux types de vecteurs : "dense" et "sparse"
        - La recherche hybride combine les résultats dense et sparse avec RRF
        - Les dimensions du vecteur dense dépendent du modèle (défaut: 384 pour all-MiniLM-L6-v2)
    """
    
    def __init__(self, config: Settings, embedding_service: EmbeddingService):
        """
        Initialise le QdrantVectorStore.
        
        Args:
            config: Configuration de l'application contenant les paramètres Qdrant.
            embedding_service: Service pour générer les embeddings dense et sparse.
        """
        self.config = config
        self.embedding_service = embedding_service
        self.client: Optional[AsyncQdrantClient] = None
        self.collection_name = config.QDRANT_COLLECTION_NAME
    
    async def connect(self):
        """
        Établit la connexion avec Qdrant.
        
        Crée un client AsyncQdrantClient si aucune connexion n'existe.
        Ne fait rien si la connexion existe déjà.
        
        Notes:
            - Le timeout est fixé à 120 secondes
            - check_compatibility est désactivé pour éviter les warnings
        """
        if self.client is None:
            self.client = AsyncQdrantClient(
                host=self.config.QDRANT_HOST,
                port=self.config.QDRANT_PORT,
                check_compatibility=False,
                timeout=120.0 
            )
    
    async def disconnect(self):
        """
        Ferme la connexion avec Qdrant.
        
        Ferme le client et réinitialise self.client à None.
        Appelé lors du shutdown de l'application.
        """
        if self.client:
            await self.client.close()
            self.client = None
    
    async def initialize_collection(self, dense_dim: int = 384):
        """
        Initialise la collection Qdrant avec configuration hybride.
        
        Crée la collection si elle n'existe pas avec :
        - Vecteurs dense : dimensions spécifiées, distance COSINE
        - Vecteurs sparse : configuration par défaut Qdrant
        
        Args:
            dense_dim: Dimensions du vecteur dense (défaut: 384 pour all-MiniLM-L6-v2).
                Doit correspondre aux dimensions du modèle DENSE_MODEL configuré.
                
        Notes:
            - Si la collection existe déjà, ne fait rien
            - La distance utilisée est COSINE pour les vecteurs dense
            - Les vecteurs sparse utilisent la configuration par défaut de Qdrant
        """
        await self.connect()
        
        collections = await self.client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if self.collection_name in collection_names:
            print(f"✅ Collection '{self.collection_name}' existe déjà")
            return
        
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
        
        Génère les embeddings hybrides (dense + sparse) pour chaque document
        et les insère dans Qdrant avec leurs métadonnées.
        
        Args:
            documents: Liste des textes de documents à ajouter.
            metadatas: Liste des métadonnées correspondantes (même longueur que documents).
            ids: IDs optionnels pour les documents. Si None, génère des IDs automatiquement.
                Defaults to None.
                
        Notes:
            - Les embeddings sont générés via embedding_service.embed_hybrid()
            - Les IDs sont convertis en entiers si possible, sinon hashés
            - Utilise upsert() donc les documents existants sont mis à jour
            - Les métadonnées sont stockées dans le payload de chaque point
        """
        await self.connect()
        
        dense_embeddings, sparse_embeddings = await self.embedding_service.embed_hybrid(documents)
        
        points = []
        for i, (doc, dense_emb, sparse_emb, metadata) in enumerate(
            zip(documents, dense_embeddings, sparse_embeddings, metadatas)
        ):
            try:
                raw_id = ids[i] if ids and i < len(ids) else str(i)
                point_id = int(raw_id) if str(raw_id).isdigit() else abs(hash(str(raw_id))) % (2**63)
            except Exception:
                point_id = i
            
            if isinstance(dense_emb, np.ndarray):
                dense_vec = dense_emb.tolist()
            else:
                dense_vec = dense_emb
            
            sparse_indices = [int(idx) for idx in sparse_emb.keys()]
            sparse_values = [float(val) for val in sparse_emb.values()]
            if not sparse_indices: sparse_indices, sparse_values = [0], [0.0]

            point = PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vec,
                    "sparse": SparseVector(indices=sparse_indices, values=sparse_values)
                },
                payload=metadata
            )
            points.append(point)
        
        if points:
            await self.client.upsert(collection_name=self.collection_name, points=points)

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Effectue une recherche hybride (dense + sparse) avec fusion RRF.
        
        Pipeline de recherche :
        1. Génère les embeddings hybrides pour la requête
        2. Recherche dense dans Qdrant (top_k * 2 candidats)
        3. Recherche sparse dans Qdrant (top_k * 2 candidats)
        4. Fusionne les résultats avec RRF (Reciprocal Rank Fusion)
        5. Retourne les top_k résultats les plus pertinents
        
        Args:
            query: Requête textuelle à rechercher.
            top_k: Nombre de résultats à retourner. Defaults to 20.
            
        Returns:
            List[Dict[str, Any]]: Liste des résultats, chacun contenant :
                - id: ID du point dans Qdrant
                - score: Score RRF calculé
                - page_content: Contenu du document (depuis payload)
                - metadata: Métadonnées du document (payload sans page_content)
                - payload: Payload complet depuis Qdrant
                
        Notes:
            - RRF combine les rangs des résultats dense et sparse : score = 1/(k + rank)
            - k_param = 60 pour le calcul RRF (bon compromis selon la littérature)
            - Les recherches dense et sparse retournent top_k * 2 candidats chacune
            - Les résultats sont triés par score RRF décroissant
            
        Example:
            >>> results = await vectorstore.hybrid_search("What is photosynthesis?", top_k=5)
            >>> print(f"Found {len(results)} results")
            >>> print(results[0]["page_content"][:100])  # Premier résultat
        """
        await self.connect()
        
        # 1. Encodage
        dense_embeddings, sparse_embeddings = await self.embedding_service.embed_hybrid([query])
        query_dense = dense_embeddings[0]
        query_sparse = sparse_embeddings[0]
        
        # Préparation sparse
        sparse_indices = list(query_sparse.keys())
        sparse_values = list(query_sparse.values())
        if not sparse_indices: sparse_indices, sparse_values = [0], [0.0]
        
        # Conversion dense
        if isinstance(query_dense, np.ndarray):
            query_dense = query_dense.tolist()
        
        # 2. Recherche DENSE
        dense_results = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_dense,
            using="dense",
            limit=top_k * 2,
            with_payload=True
        )


        # 3. Recherche SPARSE
        sparse_results = await self.client.query_points(
            collection_name=self.collection_name,
            query=SparseVector(
                indices=sparse_indices,
                values=sparse_values
            ),
            using="sparse",  # <-- OBLIGATOIRE !
            limit=top_k * 2,
            with_payload=True
        )
        
        # 4. Fusion RRF
        rrf_scores = {}
        k_param = 60
        
        # CORRECTION : Accéder aux résultats via '.points'
        for rank, result in enumerate(dense_results.points, 1):
            rrf_scores[result.id] = rrf_scores.get(result.id, 0) + (1.0 / (k_param + rank))
        
        for rank, result in enumerate(sparse_results.points, 1):
            rrf_scores[result.id] = rrf_scores.get(result.id, 0) + (1.0 / (k_param + rank))
        
        # Consolidation
        all_results_map = {res.id: res for res in list(dense_results.points) + list(sparse_results.points)}
        
        # Tri
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )[:top_k]
        
        # Formatage
        final_results = []
        for point_id, score in sorted_results:
            point = all_results_map.get(point_id)
            if point:
                final_results.append({
                    "id": point.id,
                    "score": score,
                    "payload": point.payload,
                    "page_content": point.payload.get("page_content", "") if point.payload else "",
                    "metadata": {k: v for k, v in point.payload.items() if k != "page_content"} if point.payload else {}
                })
        
        return final_results