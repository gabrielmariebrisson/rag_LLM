"""Fonctions de nettoyage et traitement de texte."""
import re


def clean_response(response: str) -> str:
    """
    Nettoie la réponse du LLM en extrayant le texte utile.
    
    Extrait le texte depuis les artefacts de formatage comme:
    - array(['texte'],)
    - 'text': 'texte'
    Si le token 'NO_ANSWER_FOUND' est détecté, retourne un espace unique ' '.
    
    Args:
        response: Réponse brute du LLM
        
    Returns:
        Texte nettoyé
    """
    if not isinstance(response, str):
        return " "
    
    extracted_text = response
    
    # 1. Extraction via les patterns (Legacy artifacts: array, json...)
    motif_principal = r"array\(\['(.*?)'\],"
    match = re.search(motif_principal, response)
    
    if match:
        extracted_text = match.group(1)
    else:
        motif_secondaire = r"'text':\s*'([^']+)'"
        match = re.search(motif_secondaire, response)
        if match:
            extracted_text = match.group(1)
    
    final_text = extracted_text.strip()
    
    # 3. Gestion du Sentinel Token
    if "NO_ANSWER_FOUND" in final_text or "no_answer_found" in final_text.lower():
        return ""
        
    return final_text