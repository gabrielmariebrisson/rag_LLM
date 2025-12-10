"""
Service RAG : récupération de documents et génération de réponses.

Ce module implémente le pipeline RAG principal :
- Recherche hybride (dense + sparse) dans Qdrant
- Reranking optionnel avec Cross-Encoder
- Génération de réponses via LLM
- Nettoyage du texte généré

Toutes les opérations sont instrumentées avec OpenTelemetry pour le tracing distribué.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from langchain_core.documents import Document
from opentelemetry import trace

from app.core.config import Settings
from app.core.prompts import SYSTEM_PROMPT, format_user_prompt
from app.utils.text_processing import clean_response
from app.vectorstores.qdrant_store import QdrantVectorStore
from app.services.reranker import RerankerService
from app.services.llm_client import LLMClient

# Tracer pour les spans manuels
tracer = trace.get_tracer(__name__)


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
    
    Pipeline complet :
    1. Hybrid Search (dense + sparse) dans Qdrant (top RERANKER_TOP_K ou k)
    2. (Optionnel) Reranking avec Cross-Encoder si activé
    3. Sélection des top k documents finaux
    
    Args:
        vectorstore: Instance QdrantVectorStore pour la recherche vectorielle.
        query: Question de l'utilisateur à rechercher.
        k: Nombre de documents finaux à retourner. Defaults to 5.
        use_reranker: Override pour activer/désactiver reranker.
            Si None, utilise la valeur de config.USE_RERANKER. Defaults to None.
        reranker_service: Service de reranking (requis si use_reranker=True).
            Defaults to None.
        config: Configuration de l'application pour déterminer use_reranker
            si None. Defaults to None.
            
    Returns:
        List[Document]: Liste des documents récupérés au format LangChain Document.
            Chaque document contient page_content et metadata.
            
    Notes:
        - Si use_reranker=True, récupère config.RERANKER_TOP_K documents avant reranking
        - Si use_reranker=False, récupère directement k documents
        - La recherche hybride utilise RRF (Reciprocal Rank Fusion) pour combiner dense + sparse
        - Les spans OpenTelemetry sont créés pour "rag.retrieval" et "rag.reranking"
        
    Example:
        >>> docs = await retrieve_documents(
        ...     vectorstore=vectorstore,
        ...     query="What is photosynthesis?",
        ...     k=5,
        ...     use_reranker=True,
        ...     reranker_service=reranker_service,
        ...     config=settings
        ... )
        >>> print(f"Retrieved {len(docs)} documents")
        >>> print(docs[0].page_content[:100])
    """
    # Déterminer si on utilise le reranker
    if use_reranker is None:
        if config is None:
            use_reranker = False
        else:
            use_reranker = config.USE_RERANKER
    
    # Span pour la recherche hybride
    with tracer.start_as_current_span("rag.retrieval") as span:
        span.set_attribute("query", query)
        span.set_attribute("k", k)
        span.set_attribute("use_reranker", use_reranker)
        
        # Recherche hybride dans Qdrant (top 20 pour avoir assez de candidats)
        top_k_search = config.RERANKER_TOP_K if config and use_reranker else k
        search_results = await vectorstore.hybrid_search(query, top_k=top_k_search)
        
        span.set_attribute("num_candidates", len(search_results))
        
        # Si reranking activé, reranker les résultats
        if use_reranker and reranker_service and len(search_results) > 0:
            # Span pour le reranking
            with tracer.start_as_current_span("rag.reranking") as rerank_span:
                rerank_span.set_attribute("num_documents", len(search_results))
                rerank_span.set_attribute("top_k", k)
                
                # Extraire les textes des documents
                document_texts = [result["page_content"] for result in search_results]
                
                # Reranker
                reranked = await reranker_service.rerank(query, document_texts, top_k=k)
                
                rerank_span.set_attribute("num_reranked", len(reranked))
                
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
        
        span.set_attribute("num_final_results", len(search_results))
    
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
    config: Settings,
    llm_client: Optional[LLMClient] = None
) -> str:
    """
    Génère une réponse via le LLM configuré (agnostique : OpenAI, Mistral API, ou vLLM local).
    
    Formatage du prompt :
    - System prompt : Instructions pour le LLM
    - User prompt : Contexte + Question formatée
    
    Le texte généré est ensuite nettoyé pour retirer les artefacts.
    
    Args:
        query: Question de l'utilisateur à répondre.
        context: Contexte récupéré depuis la base vectorielle (documents concaténés).
        config: Configuration de l'application contenant le nom du modèle.
        llm_client: Client LLM réutilisable. Si None, un nouveau client est créé.
            Defaults to None.
            
    Returns:
        str: Réponse générée et nettoyée par le LLM.
        
    Notes:
        - L'appel LLM est exécuté dans un ThreadPoolExecutor pour ne pas bloquer l'event loop
        - Le prompt est formaté avec format_user_prompt() qui combine contexte et question
        - La réponse est nettoyée avec clean_response() pour retirer les artefacts
        - Les spans OpenTelemetry incluent : model, query_length, context_length, response_length
        
    Example:
        >>> response = await generate_response(
        ...     query="What is photosynthesis?",
        ...     context="Photosynthesis is the process...",
        ...     config=settings,
        ...     llm_client=llm_client
        ... )
        >>> print(response)  # "Photosynthesis is the process by which..."
    """
    # Formater le prompt utilisateur
    user_prompt = format_user_prompt(context, query)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    # Créer le client LLM si nécessaire
    if llm_client is None:
        llm_client = LLMClient(config)
    
    # Span pour la génération LLM
    with tracer.start_as_current_span("llm.generation") as span:
        span.set_attribute("model", config.LLM_MODEL_NAME)
        span.set_attribute("query_length", len(query))
        span.set_attribute("context_length", len(context))
        span.set_attribute("llm_base_url", config.LLM_BASE_URL or "external_api")
        
        # Appel LLM (synchrone mais rapide, exécuté dans executor)
        loop = asyncio.get_event_loop()
        
        def call_llm():
            return llm_client.generate(
                messages=messages,
                model=config.LLM_MODEL_NAME,
                stream=False
            )
        
        raw_response = await loop.run_in_executor(_executor, call_llm)
        
        span.set_attribute("response_length", len(raw_response) if raw_response else 0)
        
        # Nettoyer la réponse
        cleaned_response = clean_response(raw_response)
        
        return cleaned_response
