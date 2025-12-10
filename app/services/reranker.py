"""Service de reranking avec CrossEncoder."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional
from sentence_transformers import CrossEncoder

from app.core.config import Settings


# ThreadPoolExecutor pour les opérations CPU-bound
_executor = ThreadPoolExecutor(max_workers=2)

# Cache du modèle
_reranker_model: CrossEncoder | None = None


def _get_cuda_device(device_id: Optional[int] = None) -> str:
    """Détermine le device CUDA à utiliser."""
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    
    if device_id is not None:
        # Utiliser le GPU spécifié
        if device_id >= torch.cuda.device_count():
            print(f"⚠️ GPU {device_id} non disponible, utilisation du GPU 0")
            return "cuda:0"
        return f"cuda:{device_id}"
    
    # Par défaut, utiliser cuda:0
    return "cuda:0"


def _load_reranker_model(model_name: str, device_id: Optional[int] = None) -> CrossEncoder:
    """Charge le modèle CrossEncoder."""
    device = _get_cuda_device(device_id)
    return CrossEncoder(model_name, device=device)


class RerankerService:
    """Service pour reranker les résultats de recherche."""
    
    def __init__(self, config: Settings):
        self.config = config
        self._model = None
    
    async def _ensure_model(self):
        """Charge le modèle de reranking si nécessaire."""
        global _reranker_model
        if _reranker_model is None:
            loop = asyncio.get_event_loop()
            device_id = self.config.CUDA_DEVICE_ID
            _reranker_model = await loop.run_in_executor(
                _executor,
                _load_reranker_model,
                self.config.RERANKER_MODEL,
                device_id
            )
            # Afficher le device utilisé
            import torch
            if torch.cuda.is_available():
                device = _get_cuda_device(device_id)
                print(f"✅ Reranker chargé sur {device}")
        return _reranker_model
    
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int
    ) -> List[Tuple[str, float]]:
        """
        Rerank une liste de documents par rapport à une requête.
        
        Args:
            query: Requête de l'utilisateur
            documents: Liste des textes de documents à reranker
            top_k: Nombre de documents à retourner après reranking
            
        Returns:
            Liste de tuples (document, score) triés par score décroissant
        """
        if not documents:
            return []
        
        model = await self._ensure_model()
        loop = asyncio.get_event_loop()
        
        # Créer les paires (query, document) pour le CrossEncoder
        pairs = [[query, doc] for doc in documents]
        
        # Calculer les scores (CPU-bound, donc dans executor)
        def compute_scores():
            scores = model.predict(pairs)
            return scores
        
        scores = await loop.run_in_executor(_executor, compute_scores)
        
        # Trier par score décroissant
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Retourner top_k
        return scored_docs[:top_k]

