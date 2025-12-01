"""Service d'embeddings : Dense (fastembed) et Sparse (SPLADE)."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
import numpy as np

from fastembed import TextEmbedding
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

from app.core.config import Settings


# ThreadPoolExecutor pour les opérations CPU-bound
_executor = ThreadPoolExecutor(max_workers=2)

# Cache des modèles
_dense_model: Optional[TextEmbedding] = None
_sparse_model: Optional[SentenceTransformer] = None
_sparse_tokenizer: Optional[AutoTokenizer] = None
_sparse_bert_model: Optional[AutoModelForMaskedLM] = None


def _load_dense_model(model_name: str) -> TextEmbedding:
    """Charge le modèle dense avec fastembed."""
    return TextEmbedding(model_name=model_name)


def _load_sparse_model(model_name: str):
    """Charge le modèle SPLADE pour embeddings sparse."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return tokenizer, model


def _generate_splade_embedding(text: str, tokenizer, model) -> dict:
    """
    Génère un embedding SPLADE (sparse) pour un texte.
    
    SPLADE génère un vecteur sparse où chaque dimension correspond à un token du vocabulaire.
    Pour prunebert, on utilise les logits du MLM head.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
        
        # SPLADE: Pour chaque position dans la séquence, prendre le max sur le vocabulaire
        # puis prendre le max sur la séquence (max pooling)
        # Appliquer ReLU pour avoir des valeurs positives uniquement
        relu_logits = torch.clamp(logits, min=0)
        
        # Max pooling sur la dimension vocabulaire pour chaque position
        max_per_position = torch.max(relu_logits, dim=2)[0]  # [batch_size, seq_len]
        
        # Max pooling sur la séquence
        sparse_vec = torch.max(max_per_position, dim=1)[0].squeeze()  # [vocab_size] ou [batch_size, vocab_size]
        
        # Si batch_size > 1, prendre le premier élément
        if len(sparse_vec.shape) > 1:
            sparse_vec = sparse_vec[0]
        
        # Convertir en dictionnaire sparse (indice token -> valeur)
        # On garde seulement les valeurs > 0 et on limite à un seuil pour éviter trop de tokens
        sparse_dict = {}
        threshold = 0.1  # Seuil minimal pour inclure un token
        for idx, val in enumerate(sparse_vec.cpu().numpy()):
            if val > threshold:
                sparse_dict[int(idx)] = float(val)
    
    return sparse_dict


class EmbeddingService:
    """Service pour générer des embeddings denses et sparse."""
    
    def __init__(self, config: Settings):
        self.config = config
        self._dense_model = None
        self._sparse_tokenizer = None
        self._sparse_model = None
    
    async def _ensure_dense_model(self):
        """Charge le modèle dense si nécessaire."""
        global _dense_model
        if _dense_model is None:
            loop = asyncio.get_event_loop()
            _dense_model = await loop.run_in_executor(
                _executor,
                _load_dense_model,
                self.config.DENSE_MODEL
            )
        return _dense_model
    
    async def _ensure_sparse_model(self):
        """Charge le modèle sparse si nécessaire."""
        global _sparse_tokenizer, _sparse_bert_model
        if _sparse_tokenizer is None or _sparse_bert_model is None:
            loop = asyncio.get_event_loop()
            _sparse_tokenizer, _sparse_bert_model = await loop.run_in_executor(
                _executor,
                _load_sparse_model,
                self.config.SPARSE_MODEL
            )
        return _sparse_tokenizer, _sparse_bert_model
    
    async def embed_dense(self, texts: List[str]) -> List[List[float]]:
        """
        Génère des embeddings denses pour une liste de textes.
        
        Args:
            texts: Liste de textes à encoder
            
        Returns:
            Liste d'embeddings (vecteurs denses)
        """
        model = await self._ensure_dense_model()
        loop = asyncio.get_event_loop()
        
        # fastembed supporte le batch processing
        embeddings = await loop.run_in_executor(
            _executor,
            lambda: list(model.embed(texts))
        )
        
        return [emb.tolist() for emb in embeddings]
    
    async def embed_sparse(self, texts: List[str]) -> List[dict]:
        """
        Génère des embeddings sparse (SPLADE) pour une liste de textes.
        
        Args:
            texts: Liste de textes à encoder
            
        Returns:
            Liste de dictionnaires sparse (indice -> valeur)
        """
        tokenizer, model = await self._ensure_sparse_model()
        loop = asyncio.get_event_loop()
        
        # Générer embeddings pour chaque texte
        sparse_embeddings = []
        for text in texts:
            sparse_dict = await loop.run_in_executor(
                _executor,
                _generate_splade_embedding,
                text,
                tokenizer,
                model
            )
            sparse_embeddings.append(sparse_dict)
        
        return sparse_embeddings
    
    async def embed_hybrid(self, texts: List[str]) -> tuple[List[List[float]], List[dict]]:
        """
        Génère des embeddings hybrides (dense + sparse) pour une liste de textes.
        
        Args:
            texts: Liste de textes à encoder
            
        Returns:
            Tuple (embeddings_denses, embeddings_sparse)
        """
        dense_embeddings, sparse_embeddings = await asyncio.gather(
            self.embed_dense(texts),
            self.embed_sparse(texts)
        )
        return dense_embeddings, sparse_embeddings

