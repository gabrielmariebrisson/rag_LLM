"""Client LLM agnostique : compatible OpenAI API, Mistral API, et vLLM local."""
from typing import List, Optional
from openai import OpenAI
from mistralai import Mistral

from app.core.config import Settings


class LLMClient:
    """Client LLM agnostique supportant OpenAI, Mistral API, et vLLM local."""
    
    def __init__(self, config: Settings):
        self.config = config
        self._openai_client: Optional[OpenAI] = None
        self._mistral_client: Optional[Mistral] = None
        self._use_mistral_direct = False
        
        # Déterminer quel client utiliser
        if config.LLM_BASE_URL:
            # vLLM local ou OpenAI API avec base_url custom
            self._openai_client = OpenAI(
                base_url=config.LLM_BASE_URL,
                api_key=config.LLM_API_KEY or "dummy-key"
            )
            self._use_mistral_direct = False
        elif config.LLM_API_KEY:
            # OpenAI API standard
            self._openai_client = OpenAI(api_key=config.LLM_API_KEY)
            self._use_mistral_direct = False
        elif config.MISTRAL_API_KEY:
            # Mistral API (legacy, via client Mistral direct)
            self._mistral_client = Mistral(api_key=config.MISTRAL_API_KEY)
            self._use_mistral_direct = True
        else:
            raise ValueError(
                "No LLM configuration found. Set LLM_BASE_URL for vLLM, "
                "LLM_API_KEY for OpenAI, or MISTRAL_API_KEY for Mistral."
            )
    
    def generate(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """
        Génère une réponse via le LLM configuré.
        
        Args:
            messages: Liste de messages au format OpenAI (role, content)
            model: Nom du modèle (utilise config.LLM_MODEL_NAME si None)
            stream: Si True, retourne un générateur (non implémenté pour l'instant)
            
        Returns:
            Réponse générée
        """
        if stream:
            raise NotImplementedError("Streaming not yet implemented")
        
        model_name = model or self.config.LLM_MODEL_NAME
        
        if self._use_mistral_direct:
            # Utiliser client Mistral direct (legacy)
            with self._mistral_client as mistral:
                response = mistral.chat.complete(
                    model=model_name,
                    messages=messages,
                    stream=False
                )
                if not response.choices:
                    return ""
                return response.choices[0].message.content
        else:
            # Utiliser client OpenAI (compatible vLLM et OpenAI API)
            response = self._openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=False
            )
            if not response.choices:
                return ""
            return response.choices[0].message.content

