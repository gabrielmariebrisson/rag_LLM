"""Service d'embeddings : Dense (sentence-transformers pour BGE-M3) et Sparse (fastembed SPLADE)."""
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple
import numpy as np

# Import conditionnel : sentence-transformers pour BGE-M3, fastembed pour SPLADE
from sentence_transformers import SentenceTransformer

from fastembed import SparseTextEmbedding


from app.core.config import Settings

# Executor pour les opérations de chargement de modèles (potentiellement bloquantes)
_executor = ThreadPoolExecutor(max_workers=2)


def _load_dense_model_sentence_transformers(model_name: str, hf_token: Optional[str] = None, force_cpu: bool = False):
    """Charge le modèle dense avec sentence-transformers (pour BGE-M3 et autres modèles non supportés par fastembed)."""
    print(f"🔄 Chargement modèle Dense (sentence-transformers) depuis {model_name}...")
    
    # Configurer le token HuggingFace si fourni
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    
    # Charger le modèle
    model = SentenceTransformer(model_name)
    
    # Déplacer sur GPU si disponible et non forcé sur CPU
    import torch
    if not force_cpu and torch.cuda.is_available():
        try:
            # Vérifier la mémoire disponible avant de déplacer
            if torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0) < 2 * 1024**3:  # Moins de 2GB libres
                print("⚠️ Mémoire GPU insuffisante, utilisation du CPU")
                force_cpu = True
            else:
                model = model.to("cuda")
                print("✅ Modèle Dense déplacé sur CUDA")
        except Exception as e:
            print(f"⚠️ Erreur lors du déplacement sur CUDA: {e}, utilisation du CPU")
            force_cpu = True
    
    if force_cpu:
        print("⚠️ Modèle Dense reste sur CPU")
    
    return model


class EmbeddingService:
    """Service pour générer des embeddings hybrides (dense + sparse)."""
    
    def __init__(self, config: Settings):
        self.config = config
        self.dense_model: Optional[SentenceTransformer] = None
        self.sparse_model: Optional[SparseTextEmbedding] = None
        self.use_cpu_for_dense: bool = False  # Flag pour forcer CPU si OOM sur GPU
    
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
                self.dense_model = await loop.run_in_executor(
                    _executor,
                    _load_dense_model_sentence_transformers,
                    self.config.DENSE_MODEL,
                    hf_token,
                    self.use_cpu_for_dense
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
                    self.dense_model = await loop.run_in_executor(
                        _executor,
                        _load_dense_model_sentence_transformers,
                        self.config.DENSE_MODEL,
                        hf_token
                    )
                    print("✅ Modèle Dense chargé (sentence-transformers fallback)")
        
        return self.dense_model
    
    async def _ensure_sparse_model(self):
        """Charge le modèle sparse si nécessaire."""
        if self.sparse_model is None:
            if SparseTextEmbedding is None:
                raise ImportError(
                    "fastembed.SparseTextEmbedding is required for sparse embeddings. "
                    "Install it with: pip install fastembed"
                )
            
            # Configurer le token HuggingFace si fourni
            hf_token = self.config.HUGGING_FACE_HUB_TOKEN
            if hf_token:
                os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            
            # Liste de modèles à essayer (fallback) - basée sur les modèles réellement supportés
            models_to_try = [
                self.config.SPARSE_MODEL,
                "prithivida/Splade_PP_en_v1",  
            ]
            
            # Supprimer les doublons tout en préservant l'ordre
            seen = set()
            models_to_try = [m for m in models_to_try if m not in seen and not seen.add(m)]
            
            loop = asyncio.get_event_loop()
            last_error = None
            cache_corrupted = False
            
            for model_name in models_to_try:
                try:
                    print(f"🔄 Tentative de chargement modèle Sparse: {model_name}...")
                    self.sparse_model = await loop.run_in_executor(
                        _executor,
                        lambda mn=model_name: SparseTextEmbedding(model_name=mn)
                    )
                    print(f"✅ Modèle Sparse chargé: {model_name}")
                    return self.sparse_model
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    print(f"⚠️ Échec du chargement de {model_name}: {error_str}")
                    
                    if "NO_SUCHFILE" in error_str or "File doesn't exist" in error_str:
                        # Problème de cache corrompu détecté
                        cache_corrupted = True
                        print(f"   → Cache corrompu détecté pour {model_name}")
                        continue
                    elif "not supported" in error_str.lower():
                        # Modèle non supporté, on continue avec le suivant
                        continue
                    else:
                        # Autre erreur, on continue quand même
                        continue
            
            # Si le cache est corrompu, on essaie de le nettoyer et réessayer une fois
            if cache_corrupted and self.sparse_model is None:
                import shutil
                cache_path = "/tmp/fastembed_cache"
                if os.path.exists(cache_path):
                    print(f"🧹 Nettoyage du cache corrompu: {cache_path}")
                    try:
                        shutil.rmtree(cache_path)
                        print("✅ Cache nettoyé, nouvelle tentative...")
                        
                        # Réessayer avec le premier modèle de la liste
                        try:
                            model_name = models_to_try[0]
                            print(f"🔄 Nouvelle tentative avec {model_name} après nettoyage du cache...")
                            self.sparse_model = await loop.run_in_executor(
                                _executor,
                                lambda mn=model_name: SparseTextEmbedding(model_name=mn)
                            )
                            print(f"✅ Modèle Sparse chargé après nettoyage: {model_name}")
                            return self.sparse_model
                        except Exception as e2:
                            print(f"⚠️ Échec même après nettoyage: {e2}")
                            last_error = e2
                    except Exception as cleanup_error:
                        print(f"⚠️ Impossible de nettoyer le cache: {cleanup_error}")
            
            # Si tous les modèles ont échoué, on lève l'erreur
            if self.sparse_model is None:
                error_msg = f"❌ Impossible de charger un modèle sparse. Dernière erreur: {last_error}"
                print(error_msg)
                print("💡 Suggestions:")
                print("   1. Vérifiez votre connexion internet")
                print("   2. Vérifiez que le token HuggingFace est correct (si requis)")
                print("   3. Supprimez manuellement le cache: rm -rf /tmp/fastembed_cache")
                print("   4. Vérifiez que fastembed est à jour: pip install --upgrade fastembed")
                print("   5. Vérifiez les modèles supportés: python -c \"from fastembed import SparseTextEmbedding; print(SparseTextEmbedding.list_supported_models())\"")
                raise RuntimeError(f"Échec du chargement du modèle sparse: {last_error}")
        
        return self.sparse_model
    
    async def embed_dense(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Génère des embeddings denses pour une liste de textes.
        
        Args:
            texts: Liste de textes à embedder
            batch_size: Taille du batch (réduite automatiquement en cas d'OOM)
            
        Returns:
            Liste de vecteurs denses (listes de floats)
        """
        model = await self._ensure_dense_model()
        
        loop = asyncio.get_event_loop()
        
        # Gérer les deux types de modèles
        if isinstance(model, SentenceTransformer):
            # sentence-transformers avec gestion d'erreur CUDA OOM
            import torch
            
            current_batch_size = batch_size
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    # Vider le cache avant chaque tentative
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    def encode_batch():
                        return model.encode(
                            texts, 
                            convert_to_numpy=True, 
                            batch_size=current_batch_size, 
                            show_progress_bar=False,
                            device="cpu" if self.use_cpu_for_dense else None
                        )
                    
                    embeddings = await loop.run_in_executor(_executor, encode_batch)
                    
                    if isinstance(embeddings, np.ndarray):
                        return embeddings.tolist()
                    return embeddings
                    
                except RuntimeError as e:
                    error_str = str(e)
                    if "out of memory" in error_str.lower() or "cuda" in error_str.lower():
                        if attempt < max_retries - 1:
                            # Réduire la taille du batch
                            current_batch_size = max(1, current_batch_size // 2)
                            print(f"⚠️ CUDA OOM détecté, réduction du batch_size à {current_batch_size}")
                            
                            # Vider le cache
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            # Si c'est la dernière tentative, basculer sur CPU
                            if attempt == max_retries - 2:
                                print("🔄 Basculement du modèle sur CPU...")
                                self.use_cpu_for_dense = True
                                # Recharger le modèle sur CPU
                                self.dense_model = None
                                await self._ensure_dense_model()
                            continue
                        else:
                            # Dernière tentative : forcer CPU
                            if not self.use_cpu_for_dense:
                                print("🔄 Dernière tentative : basculement sur CPU...")
                                self.use_cpu_for_dense = True
                                self.dense_model = None
                                await self._ensure_dense_model()
                                # Réessayer avec CPU
                                embeddings = await loop.run_in_executor(
                                    _executor,
                                    lambda: model.encode(texts, convert_to_numpy=True, batch_size=current_batch_size, show_progress_bar=False)
                                )
                                if isinstance(embeddings, np.ndarray):
                                    return embeddings.tolist()
                                return embeddings
                    else:
                        # Autre erreur, on la propage
                        raise
                except Exception as e:
                    raise
            
            # Ne devrait jamais arriver ici
            raise RuntimeError("Échec après toutes les tentatives")
        else:
            # fastembed
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
        try:
            model = await self._ensure_sparse_model()
        except Exception as e:
            print(f"⚠️ Impossible de charger le modèle sparse: {e}")
            print("   → Génération d'embeddings sparse vides (recherche dense uniquement)")
            # Retourner des embeddings vides si le modèle ne peut pas être chargé
            return [{} for _ in texts]
        
        try:
            # fastembed.embed() est CPU-bound, on l'exécute dans un executor
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                _executor,
                lambda: list(model.embed(texts))
            )
        except Exception as e:
            print(f"⚠️ Erreur lors de la génération d'embeddings sparse: {e}")
            print("   → Génération d'embeddings sparse vides pour ce batch")
            # Retourner des embeddings vides en cas d'erreur
            return [{} for _ in texts]
        
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
    
    async def embed_hybrid(self, texts: List[str], batch_size: int = 32) -> Tuple[List[List[float]], List[dict]]:
        """
        Génère des embeddings hybrides (dense + sparse) pour une liste de textes.
        
        Args:
            texts: Liste de textes à embedder
            batch_size: Taille du batch pour embed_dense (optionnel)
            
        Returns:
            Tuple (embeddings_denses, embeddings_sparse)
        """
        # Exécuter dense et sparse en parallèle, mais gérer les erreurs séparément
        try:
            dense_embeddings, sparse_embeddings = await asyncio.gather(
                self.embed_dense(texts, batch_size=batch_size),
                self.embed_sparse(texts),
                return_exceptions=True
            )
            
            # Gérer les exceptions
            if isinstance(dense_embeddings, Exception):
                raise dense_embeddings
            if isinstance(sparse_embeddings, Exception):
                print(f"⚠️ Erreur dans embed_sparse: {sparse_embeddings}")
                print("   → Utilisation d'embeddings sparse vides")
                sparse_embeddings = [{} for _ in texts]
            
        except Exception as e:
            # Si dense échoue, on ne peut pas continuer
            print(f"❌ Erreur critique dans embed_dense: {e}")
            raise
        
        return dense_embeddings, sparse_embeddings