"""Schémas Pydantic pour les requêtes et réponses API."""
from pydantic import BaseModel, Field
from typing import List, Optional


class ChatRequest(BaseModel):
    """Requête pour l'endpoint /chat."""
    query: str = Field(..., description="La question de l'utilisateur")
    k: int = Field(default=5, ge=1, le=20, description="Nombre de documents à récupérer")
    language: str = Field(default="en", description="Langue cible de la réponse")
    use_reranker: Optional[bool] = Field(default=None, description="Override pour activer/désactiver reranker (None = utilise config global)")


class DocumentResponse(BaseModel):
    """Représentation d'un document récupéré."""
    page_content: str = Field(..., description="Contenu du fragment récupéré")
    metadata: dict = Field(default_factory=dict, description="Métadonnées associées")


class ChatResponse(BaseModel):
    """Réponse de l'endpoint /chat."""
    response: str = Field(..., description="Réponse générée et traduite")
    retrieved_documents: List[DocumentResponse] = Field(..., description="Liste des documents sources utilisés")
    language: str = Field(..., description="Langue de la réponse générée")
    processing_time: Optional[float] = Field(None, description="Temps d'exécution en secondes")


class ExampleResponse(BaseModel):
    """Réponse de l'endpoint /examples."""
    examples: List[str] = Field(..., description="Liste des questions suggérées pour l'UI")

