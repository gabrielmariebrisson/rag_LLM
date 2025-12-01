"""Fonctions de nettoyage et traitement de texte."""
import re


def clean_response(response: str) -> str:
    """
    Nettoie la réponse du LLM en extrayant le texte utile.
    
    Extrait le texte depuis les artefacts de formatage comme:
    - array(['texte'],)
    - 'text': 'texte'
    
    Args:
        response: Réponse brute du LLM
        
    Returns:
        Texte nettoyé
    """
    if not isinstance(response, str):
        return ""
    
    # Pattern principal: array(['texte'],)
    motif_principal = r"array\(\['(.*?)'\],"
    resultat = re.search(motif_principal, response)
    
    if resultat:
        return resultat.group(1)
    
    # Pattern secondaire: 'text': 'texte'
    motif_secondaire = r"'text':\s*'([^']+)'"
    resultat = re.search(motif_secondaire, response)
    
    if resultat:
        return resultat.group(1)
    
    # Si aucun pattern ne correspond, retourner la réponse originale
    return response

