"""Service RAG : récupération de documents et génération de réponses."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from langchain_core.documents import Document
from mistralai import Mistral

from app.core.config import Settings
from app.core.prompts import SYSTEM_PROMPT, format_user_prompt
from app.utils.text_processing import clean_response
from app.vectorstores.qdrant_store import QdrantVectorStore
from app.services.reranker import RerankerService


# ThreadPoolExecutor pour les opérations CPU-bound (LLM)
_executor = ThreadPoolExecutor(max_workers=4)


async def retrieve_documents(
    vectorstore: QdrantVectorStore,
    query: str,
    k: int = 5,
    use_reranker: Optional[bool] = None,
    reranker_service: Optional[RerankerService] = None,
    config: Optional[Settings] = None
) -> List[Document]:
    """
    Récupère les documents pertinents depuis Qdrant avec option de reranking.
    
    Pipeline: Hybrid Search (top 20) -> (optionnel) Reranking -> Top k
    
    Args:
        vectorstore: Instance QdrantVectorStore
        query: Question de l'utilisateur
        k: Nombre de documents finaux à retourner
        use_reranker: Override pour activer/désactiver reranker (None = utilise config)
        reranker_service: Service de reranking (requis si use_reranker=True)
        config: Configuration (pour déterminer use_reranker si None)
        
    Returns:
        Liste des documents récupérés (format LangChain Document)
    """
    # Déterminer si on utilise le reranker
    if use_reranker is None:
        if config is None:
            use_reranker = False
        else:
            use_reranker = config.USE_RERANKER
    
    # Recherche hybride dans Qdrant (top 20 pour avoir assez de candidats)
    top_k_search = config.RERANKER_TOP_K if config and use_reranker else k
    search_results = await vectorstore.hybrid_search(query, top_k=top_k_search)
    
    # Si reranking activé, reranker les résultats
    if use_reranker and reranker_service and len(search_results) > 0:
        # Extraire les textes des documents
        document_texts = [result["page_content"] for result in search_results]
        
        # Reranker
        reranked = await reranker_service.rerank(query, document_texts, top_k=k)
        
        # Créer un mapping texte -> résultat original
        text_to_result = {result["page_content"]: result for result in search_results}
        
        # Reconstruire les résultats rerankés
        reranked_results = []
        for doc_text, score in reranked:
            if doc_text in text_to_result:
                original_result = text_to_result[doc_text]
                original_result["rerank_score"] = score
                reranked_results.append(original_result)
        
        search_results = reranked_results[:k]
    else:
        # Prendre les top k sans reranking
        search_results = search_results[:k]
    
    # Convertir en format LangChain Document
    documents = []
    for result in search_results:
        doc = Document(
            page_content=result["page_content"],
            metadata=result["metadata"]
        )
        documents.append(doc)
    
    return documents


async def generate_response(
    query: str,
    context: str,
    config: Settings
) -> str:
    """
    Génère une réponse via l'API Mistral.
    
    Args:
        query: Question de l'utilisateur
        context: Contexte récupéré depuis la base vectorielle
        config: Configuration de l'application
        
    Returns:
        Réponse générée et nettoyée
    """
    # Formater le prompt utilisateur
    user_prompt = format_user_prompt(context, query)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    # Appel Mistral API (synchrone mais rapide)
    # Si nécessaire, on peut aussi le mettre dans un executor
    loop = asyncio.get_event_loop()
    
    def call_mistral():
        with Mistral(api_key=config.MISTRAL_API_KEY) as mistral:
            response = mistral.chat.complete(
                model=config.MISTRAL_MODEL_NAME,
                messages=messages,
                stream=False
            )
            if not response.choices:
                return ""
            return response.choices[0].message.content
    
    raw_response = await loop.run_in_executor(_executor, call_mistral)
    
    # Nettoyer la réponse
    cleaned_response = clean_response(raw_response)
    
    return cleaned_response
