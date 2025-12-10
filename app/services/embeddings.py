"""Service d'embeddings : Dense (fastembed) et Sparse (fastembed SPLADE)."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple
from fastembed import TextEmbedding, SparseTextEmbedding

from app.core.config import Settings

# Executor pour les opérations de chargement de modèles (potentiellement bloquantes)
_executor = ThreadPoolExecutor(max_workers=2)


class EmbeddingService:
    """Service pour générer des embeddings hybrides (dense + sparse) avec fastembed."""
    
    def __init__(self, config: Settings):
        self.config = config
        self.dense_model: Optional[TextEmbedding] = None
        self.sparse_model: Optional[SparseTextEmbedding] = None
    
    async def _ensure_dense_model(self) -> TextEmbedding:
        """Charge le modèle dense si nécessaire."""
        if self.dense_model is None:
            print(f"🔄 Chargement modèle Dense depuis {self.config.DENSE_MODEL}...")
            loop = asyncio.get_event_loop()
            self.dense_model = await loop.run_in_executor(
                _executor,
                lambda: TextEmbedding(model_name=self.config.DENSE_MODEL)
            )
            print("✅ Modèle Dense chargé")
        return self.dense_model
    
    async def _ensure_sparse_model(self) -> SparseTextEmbedding:
        """Charge le modèle sparse si nécessaire."""
        if self.sparse_model is None:
            print(f"🔄 Chargement modèle Sparse depuis {self.config.SPARSE_MODEL}...")
            loop = asyncio.get_event_loop()
            self.sparse_model = await loop.run_in_executor(
                _executor,
                lambda: SparseTextEmbedding(model_name=self.config.SPARSE_MODEL)
            )
            print("✅ Modèle Sparse chargé")
        return self.sparse_model
    
    async def embed_dense(self, texts: List[str]) -> List[List[float]]:
        """
        Génère des embeddings denses pour une liste de textes.
        
        Args:
            texts: Liste de textes à embedder
            
        Returns:
            Liste de vecteurs denses (listes de floats)
        """
        model = await self._ensure_dense_model()
        
        # fastembed.embed() est CPU-bound, on l'exécute dans un executor
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            _executor,
            lambda: list(model.embed(texts))
        )
        
        # Convertir les numpy arrays en listes de floats
        return [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]
    
    async def embed_sparse(self, texts: List[str]) -> List[dict]:
        """
        Génère des embeddings sparse pour une liste de textes.
        
        Args:
            texts: Liste de textes à embedder
            
        Returns:
            Liste de dictionnaires {str(index): float(weight)}
        """
        model = await self._ensure_sparse_model()
        
        # fastembed.embed() est CPU-bound, on l'exécute dans un executor
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            _executor,
            lambda: list(model.embed(texts))
        )
        
        # S'assurer que les clés sont des strings (compatibilité Qdrant)
        result = []
        for emb in embeddings:
            if isinstance(emb, dict):
                # Convertir les clés en strings si nécessaire
                sparse_dict = {str(k): float(v) for k, v in emb.items()}
                result.append(sparse_dict)
            else:
                # Fallback si le format n'est pas un dict
                result.append({})
        
        return result
    
    async def embed_hybrid(self, texts: List[str]) -> Tuple[List[List[float]], List[dict]]:
        """
        Génère des embeddings hybrides (dense + sparse) pour une liste de textes.
        
        Args:
            texts: Liste de textes à embedder
            
        Returns:
            Tuple (embeddings_denses, embeddings_sparse)
        """
        dense_embeddings, sparse_embeddings = await asyncio.gather(
            self.embed_dense(texts),
            self.embed_sparse(texts)
        )
        return dense_embeddings, sparse_embeddings
