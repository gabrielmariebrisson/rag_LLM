"""Service d'embeddings : Dense (fastembed) et Sparse (SPLADE)."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
import numpy as np
import threading

from fastembed import TextEmbedding
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

from app.core.config import Settings


# ThreadPoolExecutor pour les opérations CPU-bound
_executor = ThreadPoolExecutor(max_workers=4)

# Cache des modèles
_dense_model: Optional[SentenceTransformer] = None
_sparse_model: Optional[SentenceTransformer] = None
_sparse_tokenizer: Optional[AutoTokenizer] = None
_sparse_bert_model: Optional[AutoModelForMaskedLM] = None

# Verrou pour éviter les chargements multiples simultanés
_sparse_model_lock = threading.Lock()


def _load_dense_model(model_name: str):
    """Charge le modèle dense avec sentence-transformers (plus fiable que fastembed)."""
    # Utiliser sentence-transformers directement pour éviter les problèmes de téléchargement fastembed
    if model_name.startswith("sentence-transformers/"):
        model_name = model_name.replace("sentence-transformers/", "")
    return SentenceTransformer(model_name)


def _load_sparse_model(model_name: str):
    """Charge le modèle BERT pour embeddings sparse (SPLADE-like)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Détecter le périphérique disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Charger le modèle sur CPU d'abord pour éviter le mode meta
    # Ne pas utiliser device_map ni low_cpu_mem_usage qui peuvent causer le mode meta
    # Ne pas utiliser torch_dtype qui est déprécié
    try:
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=False,  # Désactiver pour éviter le mode meta
            device_map=None,  # Ne pas utiliser device_map
        )
    except Exception as e:
        # Si erreur, essayer sans aucun paramètre
        import warnings
        warnings.warn(f"Erreur lors du chargement: {e}, rechargement simple...")
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    # Vérifier que le modèle a des données réelles (pas en mode meta)
    first_param = next(model.parameters())
    if not hasattr(first_param, 'data') or first_param.data.numel() == 0:
        # Le modèle est en mode meta, le recharger différemment
        import warnings
        warnings.warn("Modèle en mode meta détecté, rechargement...")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        # Recharger sans aucun paramètre qui pourrait causer le mode meta
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    # S'assurer que le modèle est sur CPU d'abord (évite les problèmes de meta tensor)
    if str(next(model.parameters()).device) != "cpu":
        model = model.cpu()
    
    # Maintenant déplacer vers le périphérique cible de manière sûre
    if device == "cuda" and torch.cuda.is_available():
        # Vérifier que le modèle n'est pas en mode meta avant de déplacer
        try:
            test_param = next(model.parameters())
            if hasattr(test_param, 'data') and test_param.data.numel() > 0:
                model = model.to(device)
            else:
                raise RuntimeError("Modèle en mode meta, ne peut pas être déplacé vers CUDA")
        except RuntimeError as e:
            import warnings
            warnings.warn(f"{e}, le modèle restera sur CPU")
            device = "cpu"
    
    print(f"Modèle sparse chargé sur le périphérique: {device}")
    
    return tokenizer, model


def _generate_splade_embeddings_batch(texts: List[str], tokenizer, model, sub_batch_size: int = 8) -> List[dict]:
    """
    Génère des embeddings SPLADE (sparse) pour un batch de textes (optimisé pour GPU).
    Traite par sous-batches pour éviter les problèmes de mémoire.
    
    Args:
        texts: Liste de textes à encoder
        tokenizer: Tokenizer BERT
        model: Modèle BERT pour MLM
        sub_batch_size: Taille des sous-batches pour éviter OOM
        
    Returns:
        Liste de dictionnaires sparse (indice -> valeur)
    """
    # Vérifier que le modèle n'est pas en mode meta
    try:
        first_param = next(model.parameters())
        if not hasattr(first_param, 'data') or first_param.data.numel() == 0:
            raise RuntimeError("Modèle en mode meta détecté dans _generate_splade_embeddings_batch")
    except RuntimeError:
        raise
    except Exception:
        pass  # Si erreur, continuer quand même
    
    device = next(model.parameters()).device
    model.eval()
    
    all_results = []
    
    # Traiter par sous-batches pour éviter les problèmes de mémoire GPU
    for i in range(0, len(texts), sub_batch_size):
        sub_texts = texts[i:i+sub_batch_size]
        
        with torch.no_grad():
            # Tokeniser le sous-batch
            inputs = tokenizer(
                sub_texts,
                return_tensors="pt",
                truncation=True,
                max_length=256,  # Réduit de 512 pour économiser la mémoire GPU
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass sur tout le batch
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
        
        # SPLADE: Max pooling sur vocabulaire pour chaque position, puis max pooling sur séquence
        relu_logits = torch.clamp(logits, min=0)  # [batch_size, seq_len, vocab_size]
        
        # Max sur vocabulaire pour chaque position: [batch_size, seq_len]
        max_per_position = torch.max(relu_logits, dim=2)[0]
        
        # Max pooling sur la séquence: [batch_size] - mais on veut garder les indices
        # On prend le max global sur toutes les positions pour chaque token du vocabulaire
        # Approche: max sur (seq_len, vocab_size) pour chaque batch
        max_over_all = torch.max(relu_logits.view(logits.shape[0], -1), dim=1)[0]  # [batch_size]
        
        # Pour chaque texte du batch, créer un dictionnaire sparse
        # On utilise une approche simplifiée: prendre les top tokens actifs
        threshold = 0.1
        results = []
        
        for batch_idx in range(logits.shape[0]):
            # Pour ce texte, prendre le max sur toutes les positions pour chaque token
            text_logits = relu_logits[batch_idx]  # [seq_len, vocab_size]
            # Max sur les positions: [vocab_size]
            max_per_token = torch.max(text_logits, dim=0)[0]
            
            # Garder seulement les tokens > threshold
            active_mask = max_per_token > threshold
            active_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
            active_values = max_per_token[active_indices]
            
            # Créer le dictionnaire (utiliser .detach() pour éviter l'erreur de grad)
            sparse_dict = {}
            for idx, val in zip(active_indices.cpu().detach().numpy(), active_values.cpu().detach().numpy()):
                sparse_dict[int(idx)] = float(val)
            
            results.append(sparse_dict)
        
        all_results.extend(results)
        
        # Libérer les tensors explicitement
        del inputs, outputs, logits, relu_logits, max_per_position, max_per_token
        del active_mask, active_indices, active_values, text_logits
        
        # Nettoyer la mémoire GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    return all_results


def _generate_splade_embedding(text: str, tokenizer, model) -> dict:
    """
    Génère un embedding SPLADE (sparse) pour un texte.
    
    SPLADE génère un vecteur sparse où chaque dimension correspond à un token du vocabulaire.
    Pour BERT, on utilise les logits du MLM head.
    """
    # Utiliser le device du modèle (déjà configuré lors du chargement)
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,  # Réduit de 512 pour économiser la mémoire GPU
            padding=True
        )
        # Déplacer les inputs vers le même device que le modèle
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
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
        
        # S'assurer que sparse_vec est un 1D array
        if sparse_vec.dim() == 0:
            # Scalar, convertir en 1D array
            sparse_vec = sparse_vec.unsqueeze(0)
        elif sparse_vec.dim() > 1:
            # Prendre le premier élément si batch_size > 1
            sparse_vec = sparse_vec[0]
        
        # Convertir en numpy array 1D
        sparse_array = sparse_vec.cpu().numpy()
        if sparse_array.ndim == 0:
            sparse_array = sparse_array.reshape(1)
        
        # Convertir en dictionnaire sparse (indice token -> valeur)
        # On garde seulement les valeurs > 0 et on limite à un seuil pour éviter trop de tokens
        sparse_dict = {}
        threshold = 0.1  # Seuil minimal pour inclure un token
        for idx, val in enumerate(sparse_array):
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
        
        # Vérifier d'abord sans lock (lecture rapide)
        if _sparse_tokenizer is not None and _sparse_bert_model is not None:
            return _sparse_tokenizer, _sparse_bert_model
        
        # Utiliser un lock pour éviter les chargements multiples simultanés
        with _sparse_model_lock:
            # Vérifier à nouveau après avoir acquis le lock (double-check pattern)
            if _sparse_tokenizer is not None and _sparse_bert_model is not None:
                return _sparse_tokenizer, _sparse_bert_model
            
            # Charger dans le thread principal (synchrone) pour éviter meta tensor
            # Ne pas utiliser run_in_executor car cela peut causer des problèmes de meta tensor
            tokenizer, model = _load_sparse_model(self.config.SPARSE_MODEL)
            
            # Vérification finale que le modèle est valide
            try:
                # Tester que le modèle fonctionne en faisant un forward pass minimal
                test_input = tokenizer("test", return_tensors="pt", truncation=True, max_length=10)
                device = next(model.parameters()).device
                test_input = {k: v.to(device) for k, v in test_input.items()}
                with torch.no_grad():
                    _ = model(**test_input)
            except Exception as e:
                # Si erreur (notamment meta tensor), recharger complètement
                import warnings
                warnings.warn(f"Erreur lors du test du modèle: {e}, rechargement...")
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                from transformers import AutoModelForMaskedLM
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = AutoModelForMaskedLM.from_pretrained(
                    self.config.SPARSE_MODEL,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False,
                    device_map=None
                )
                model = model.cpu()
                if device == "cuda":
                    model = model.to(device)
            
            _sparse_tokenizer = tokenizer
            _sparse_bert_model = model
        
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
        
        # sentence-transformers supporte le batch processing
        embeddings = await loop.run_in_executor(
            _executor,
            lambda: model.encode(texts, convert_to_numpy=True)
        )
        
        # Convertir en liste de listes
        if isinstance(embeddings, np.ndarray):
            if len(embeddings.shape) == 1:
                # Un seul embedding (1D array)
                return [embeddings.tolist()]
            else:
                # Plusieurs embeddings (2D array)
                return embeddings.tolist()
        else:
            # Déjà une liste
            return [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
    
    async def embed_sparse(self, texts: List[str]) -> List[dict]:
        """
        Génère des embeddings sparse (SPLADE) pour une liste de textes (optimisé par batch).
        
        Args:
            texts: Liste de textes à encoder
            
        Returns:
            Liste de dictionnaires sparse (indice -> valeur)
        """
        tokenizer, model = await self._ensure_sparse_model()
        
        # Pour les opérations GPU, ne pas utiliser run_in_executor car cela peut causer
        # des problèmes de meta tensor. Les opérations GPU sont déjà asynchrones.
        # Utiliser run_in_executor seulement pour les parties CPU (tokenization peut être lente)
        loop = asyncio.get_event_loop()
        
        # Traiter tout le batch en une fois (beaucoup plus rapide sur GPU)
        # Passer le modèle directement sans run_in_executor pour éviter les problèmes de meta tensor
        sparse_embeddings = await loop.run_in_executor(
            _executor,
            lambda: _generate_splade_embeddings_batch(texts, tokenizer, model, sub_batch_size=8)
        )
        
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