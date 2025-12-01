"""Service de traduction utilisant GoogleTranslator."""
from deep_translator import GoogleTranslator
from typing import Optional


# Cache simple en mémoire pour éviter les appels répétés
_translation_cache: dict[str, str] = {}


def translate_text(
    text: str, 
    target_lang: str, 
    source_lang: str = 'en'
) -> str:
    """
    Traduit un texte de la langue source vers la langue cible.
    
    Args:
        text: Texte à traduire
        target_lang: Langue cible (code ISO, ex: 'fr', 'en', 'es')
        source_lang: Langue source (défaut: 'en')
        
    Returns:
        Texte traduit, ou texte original en cas d'erreur
    """
    if not text or not isinstance(text, str):
        return text
    
    # Si la langue source et cible sont identiques, pas de traduction
    if source_lang == target_lang:
        return text
    
    # Vérifier le cache
    cache_key = f"{source_lang}_{target_lang}_{text}"
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]
    
    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated = translator.translate(text)
        _translation_cache[cache_key] = translated
        return translated
    except Exception:
        # En cas d'erreur, retourner le texte original
        return text


def clear_cache() -> None:
    """Vide le cache de traduction."""
    global _translation_cache
    _translation_cache.clear()

