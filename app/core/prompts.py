"""Templates de prompts pour le système RAG."""

SYSTEM_PROMPT = (
    "You are a helpful assistant. Write responses in complete, well-developed sentences. "
    "Express ideas clearly and naturally, avoiding overly brief or list-style answers."
)


def format_user_prompt(context: str, query: str) -> str:
    """
    Formate le prompt utilisateur avec le contexte et la question.
    
    Args:
        context: Contexte récupéré depuis la base vectorielle
        query: Question de l'utilisateur
        
    Returns:
        Prompt formaté pour l'API Mistral
    """
    return f"Context:\n{context}\n\nQuestion: {query}"

