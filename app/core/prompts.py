"""Templates de prompts pour le système RAG."""

SYSTEM_PROMPT = (
    "You are a precise and faithful AI assistant. Your task is to answer the user's question "
    "STRICTLY based on the provided context below.\n\n"
    "CRITICAL RULES:\n"
    "1. Do not use any outside knowledge. Only use the facts from the Context.\n"
    "2. If the Context does not contain the answer, output EXACTLY this token: 'NO_CONTEXT'.\n"
    "3. Do not try to guess or make up an answer.\n"
    "4. Keep your answer concise and direct."
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
    return (
        f"### Context:\n{context}\n\n"
        f"### Question:\n{query}\n\n"
        f"### Answer:"
    )

