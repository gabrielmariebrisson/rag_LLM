"""Service RAG : récupération de documents et génération de réponses."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS
from mistralai import Mistral

from app.core.config import Settings
from app.core.prompts import SYSTEM_PROMPT, format_user_prompt
from app.utils.text_processing import clean_response


# ThreadPoolExecutor pour exécuter les opérations CPU-bound
_executor = ThreadPoolExecutor(max_workers=4)


async def retrieve_documents(
    vectorstore: FAISS,
    query: str,
    k: int = 5
) -> List[Document]:
    """
    Récupère les documents pertinents depuis le vectorstore.
    
    Exécute similarity_search dans un executor pour ne pas bloquer l'event loop.
    
    Args:
        vectorstore: Instance FAISS chargée
        query: Question de l'utilisateur
        k: Nombre de documents à récupérer
        
    Returns:
        Liste des documents récupérés
    """
    loop = asyncio.get_event_loop()
    
    # Exécuter similarity_search dans un thread séparé (CPU-bound)
    results = await loop.run_in_executor(
        _executor,
        vectorstore.similarity_search,
        query,
        k
    )
    
    return results


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

