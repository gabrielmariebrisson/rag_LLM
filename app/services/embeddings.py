"""Service d'embeddings : Dense (sentence-transformers pour BGE-M3) et Sparse (sentence-transformers SPLADE)."""
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple
import numpy as np

# Import conditionnel : sentence-transformers pour BGE-M3 et SPLADE
try:
    from sentence_transformers import SentenceTransformer, SparseEncoder
except ImportError:
    SentenceTransformer = None
    SparseEncoder = None

# Note: On ne garde plus l'import fastembed pour les embeddings sparse

from app.core.config import Settings

# Executor pour les opérations de chargement de modèles (potentiellement bloquantes)
_executor = ThreadPoolExecutor(max_workers=2)


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


def _load_dense_model_sentence_transformers(model_name: str, hf_token: Optional[str] = None, device_id: Optional[int] = None):
    """Charge le modèle dense avec sentence-transformers (optimisé GPU pour RTX 3090)."""
    print(f"🔄 Chargement modèle Dense (sentence-transformers) depuis {model_name}...")
    
    # Configurer le token HuggingFace si fourni
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    
    # Déterminer le device avant le chargement
    device = _get_cuda_device(device_id)
    
    # Charger le modèle directement sur le device approprié
    model = SentenceTransformer(model_name, device=device)
    
    # Vérification finale
    import torch
    if torch.cuda.is_available():
        print(f"✅ Modèle Dense chargé sur {device} (device: {model.device})")
        if hasattr(model, '_modules') and len(model._modules) > 0:
            first_module = next(iter(model._modules.values()))
            if hasattr(first_module, 'device'):
                print(f"   Module principal sur: {first_module.device}")
    else:
        print("⚠️ Modèle Dense chargé sur CPU (GPU non disponible)")
    
    return model


class EmbeddingService:
    """Service pour générer des embeddings hybrides (dense + sparse)."""
    
    def __init__(self, config: Settings):
        self.config = config
        self.dense_model: Optional[SentenceTransformer] = None
        # CHANGEMENT : Type changé de SparseTextEmbedding à SparseEncoder
        self.sparse_model = None  # Sera une instance de SparseEncoder
    
    async def _ensure_dense_model(self):
        """Charge le modèle dense si nécessaire."""
        if self.dense_model is None:
            # BGE-M3 n'est pas supporté par fastembed, utiliser sentence-transformers
            # Vérifier si le modèle nécessite sentence-transformers
            use_sentence_transformers = (
                "bge-m3" in self.config.DENSE_MODEL.lower() or
                "bge-large" in self.config.DENSE_MODEL.lower() or
                "bge-base" in self.config.DENSE_MODEL.lower() or
                SentenceTransformer is not None  # Si fastembed n'est pas disponible, utiliser sentence-transformers
            )
            
            if use_sentence_transformers:
                if SentenceTransformer is None:
                    raise ImportError(
                        "sentence-transformers is required for BGE-M3. "
                        "Install it with: pip install sentence-transformers"
                    )
                print(f"🔄 Chargement modèle Dense depuis {self.config.DENSE_MODEL}...")
                loop = asyncio.get_event_loop()
                hf_token = self.config.HUGGING_FACE_HUB_TOKEN
                device_id = self.config.CUDA_DEVICE_ID
                self.dense_model = await loop.run_in_executor(
                    _executor,
                    _load_dense_model_sentence_transformers,
                    self.config.DENSE_MODEL,
                    hf_token,
                    device_id
                )
                print("✅ Modèle Dense chargé (sentence-transformers)")
            else:
                # Utiliser fastembed pour les modèles supportés
                try:
                    from fastembed import TextEmbedding
                    print(f"🔄 Chargement modèle Dense (fastembed) depuis {self.config.DENSE_MODEL}...")
                    loop = asyncio.get_event_loop()
                    self.dense_model = await loop.run_in_executor(
                        _executor,
                        lambda: TextEmbedding(model_name=self.config.DENSE_MODEL)
                    )
                    print("✅ Modèle Dense chargé (fastembed)")
                except Exception as e:
                    print(f"⚠️ Erreur avec fastembed: {e}")
                    print("   Fallback vers sentence-transformers...")
                    if SentenceTransformer is None:
                        raise ImportError("Neither fastembed nor sentence-transformers available")
                    loop = asyncio.get_event_loop()
                    hf_token = self.config.HUGGING_FACE_HUB_TOKEN
                    device_id = self.config.CUDA_DEVICE_ID
                    self.dense_model = await loop.run_in_executor(
                        _executor,
                        _load_dense_model_sentence_transformers,
                        self.config.DENSE_MODEL,
                        hf_token,
                        device_id
                    )
                    print("✅ Modèle Dense chargé (sentence-transformers fallback)")
        
        return self.dense_model
    
    async def _ensure_sparse_model(self):
        """Charge le modèle sparse si nécessaire."""
        if self.sparse_model is None:
            # CHANGEMENT : Utilisation de SparseEncoder au lieu de SparseTextEmbedding
            if SparseEncoder is None:
                raise ImportError(
                    "sentence_transformers.SparseEncoder is required for sparse embeddings. "
                    "Ensure you have sentence-transformers>=2.5.0 installed"
                )
            print(f"🔄 Chargement modèle Sparse depuis {self.config.SPARSE_MODEL}...")
            loop = asyncio.get_event_loop()
            
            # CHANGEMENT : Création d'un SparseEncoder avec support GPU optimisé
            device = _get_cuda_device(self.config.CUDA_DEVICE_ID)
            self.sparse_model = await loop.run_in_executor(
                _executor,
                lambda: SparseEncoder(
                    self.config.SPARSE_MODEL,
                    device=device
                )
            )
            # Vérifier le device effectif
            actual_device = getattr(self.sparse_model, 'device', device)
            print(f"✅ Modèle Sparse chargé (sentence-transformers) sur {actual_device}")
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
        
        loop = asyncio.get_event_loop()
        
        # Gérer les deux types de modèles
        if isinstance(model, SentenceTransformer):
            # sentence-transformers
            embeddings = await loop.run_in_executor(
                _executor,
                lambda: model.encode(texts, convert_to_numpy=True, batch_size=128, show_progress_bar=False)
            )
            if isinstance(embeddings, np.ndarray):
                return embeddings.tolist()
            return embeddings
        else:
            # fastembed (fallback)
            embeddings = await loop.run_in_executor(
                _executor,
                lambda: list(model.embed(texts))
            )
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
        
        # CHANGEMENT : Utilisation de SparseEncoder.encode() avec support GPU
        loop = asyncio.get_event_loop()
        
        # SparseEncoder.encode() ne supporte pas convert_to_numpy, on l'appelle sans paramètres additionnels
        sparse_embeddings = await loop.run_in_executor(
            _executor,
            lambda: model.encode(texts)
        )
        
        # Convertir au format attendu par Qdrant
        import torch
        result = []
        for emb in sparse_embeddings:
            if isinstance(emb, dict):
                # SparseEncoder retourne déjà un dict dans certains cas
                result.append({str(k): float(v) for k, v in emb.items()})
            else:
                # Convertir tensor/array en dict
                # Si c'est un tensor PyTorch sparse, le convertir en dense puis en numpy
                if isinstance(emb, torch.Tensor):
                    # Les tensors sparse doivent être convertis en dense avant numpy
                    if emb.is_sparse:
                        emb = emb.to_dense()
                    emb = emb.cpu().numpy()
                # On filtre les valeurs nulles pour économiser de l'espace
                result.append({str(i): float(v) for i, v in enumerate(emb) if v != 0})
        
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