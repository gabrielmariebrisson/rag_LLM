"""Service d'embeddings : Dense (fastembed) et Sparse (SPLADE)."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple
import numpy as np
import threading
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM

from app.core.config import Settings

_executor = ThreadPoolExecutor(max_workers=8)

_dense_model: Optional[SentenceTransformer] = None
_sparse_model: Optional[SentenceTransformer] = None
_sparse_tokenizer: Optional[AutoTokenizer] = None
_sparse_bert_model: Optional[AutoModelForMaskedLM] = None
_sparse_model_lock = threading.Lock()


def _load_dense_model(model_name: str):
    """Charge le modèle dense avec sentence-transformers."""
    if model_name.startswith("sentence-transformers/"):
        model_name = model_name.replace("sentence-transformers/", "")
    
    print(f"🔄 Chargement modèle Dense depuis {model_name}...")
    
    # 1. Charger sur CPU
    model = SentenceTransformer(model_name, device="cpu")
    
    # 2. Vérifier s'il y a des meta tensors
    has_meta_tensors = False
    for param in model.parameters():
        if param.is_meta:
            has_meta_tensors = True
            break
    
    # 3. Gérer le déplacement sur GPU
    if torch.cuda.is_available():
        if has_meta_tensors:
            print("⚠️ Modèle contient des meta tensors, utilisation de to_empty()...")
            # Utiliser to_empty pour allouer la mémoire sans copier
            model = model.to_empty(device='cuda')
            # Recharger les poids
            model.load_state_dict(model.state_dict(), assign=True)
        else:
            # Déplacement normal
            model = model.to("cuda")
        print("✅ Modèle Dense déplacé sur CUDA")
    else:
        print("⚠️ Modèle Dense reste sur CPU")
        
    return model


def _load_sparse_model(model_name: str):
    """
    Charge le modèle BERT pour embeddings sparse.
    """
    print(f"🔄 Chargement modèle Sparse (BERT) depuis {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # --- FIX FINAL : MÊME STRATÉGIE QUE LE DENSE ---
    # 1. On charge le modèle nu (par défaut sur CPU)
    # On évite device_map="cuda" qui déclenche accelerate et le bug Meta Tensor
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    # 2. Déplacement explicite sur GPU
    if torch.cuda.is_available():
        model = model.to_empty(device="cuda")
        print("✅ Modèle Sparse déplacé sur CUDA")
    else:
        print("⚠️ Modèle Sparse reste sur CPU")
        
    model.eval()
    return tokenizer, model


def _generate_splade_embeddings_batch(texts: List[str], tokenizer, model, sub_batch_size: int = 128) -> List[dict]:
    """Génère des embeddings SPLADE optimisés pour GPU."""
    device = model.device
    all_results = []
    
    for i in range(0, len(texts), sub_batch_size):
        sub_texts = texts[i:i+sub_batch_size]
        
        try:
            with torch.no_grad():
                inputs = tokenizer(
                    sub_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256, 
                    padding=True
                ).to(device)
                
                outputs = model(**inputs)
                logits = outputs.logits
                
                # SPLADE: log(1 + relu(logits)) -> Max Pooling
                values, _ = torch.max(torch.log(1 + torch.relu(logits)), dim=1)
                
                values_np = values.cpu().numpy()
                threshold = 0.1
                
                for row in values_np:
                    indices = np.nonzero(row > threshold)[0]
                    scores = row[indices]
                    sparse_dict = {int(idx): float(sc) for idx, sc in zip(indices, scores)}
                    all_results.append(sparse_dict)
                    
        except Exception as e:
            print(f"⚠️ Erreur SPLADE batch: {e}")
            for _ in sub_texts: all_results.append({})
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    return all_results


class EmbeddingService:
    def __init__(self, config: Settings):
        self.config = config
        
    async def _ensure_dense_model(self):
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
        global _sparse_tokenizer, _sparse_bert_model
        if _sparse_tokenizer and _sparse_bert_model:
            return _sparse_tokenizer, _sparse_bert_model
        
        with _sparse_model_lock:
            if _sparse_tokenizer and _sparse_bert_model:
                return _sparse_tokenizer, _sparse_bert_model
            # Chargement direct (sans executor) pour éviter problèmes de contexte
            tokenizer, model = _load_sparse_model(self.config.SPARSE_MODEL)
            _sparse_tokenizer = tokenizer
            _sparse_bert_model = model
        
        return _sparse_tokenizer, _sparse_bert_model
    
    async def embed_dense(self, texts: List[str]) -> List[List[float]]:
        model = await self._ensure_dense_model()
        loop = asyncio.get_event_loop()
        
        embeddings = await loop.run_in_executor(
            _executor, 
            lambda: model.encode(texts, convert_to_numpy=True, batch_size=128)
        )
        return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
    
    async def embed_sparse(self, texts: List[str]) -> List[dict]:
        tokenizer, model = await self._ensure_sparse_model()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, lambda: _generate_splade_embeddings_batch(texts, tokenizer, model, sub_batch_size=128)
        )
    
    async def embed_hybrid(self, texts: List[str]) -> Tuple[List[List[float]], List[dict]]:
        return await asyncio.gather(self.embed_dense(texts), self.embed_sparse(texts))