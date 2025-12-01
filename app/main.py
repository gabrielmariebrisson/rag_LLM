"""Application FastAPI principale."""
import time
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from app.core.config import settings
from app.schemas import ChatRequest, ChatResponse, DocumentResponse, ExampleResponse
from app.services.rag import retrieve_documents, generate_response
from app.services.translation import translate_text
from app.services.embeddings import EmbeddingService
from app.services.reranker import RerankerService
from app.services.llm_client import LLMClient
from app.vectorstores.qdrant_store import QdrantVectorStore


# Variables globales pour stocker les services
vectorstore: Optional[QdrantVectorStore] = None
embedding_service: Optional[EmbeddingService] = None
reranker_service: Optional[RerankerService] = None
llm_client: Optional[LLMClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application."""
    global vectorstore, embedding_service, reranker_service, llm_client
    
    # Startup: Initialiser Qdrant et services
    try:
        # Initialiser les services
        embedding_service = EmbeddingService(settings)
        reranker_service = RerankerService(settings)
        llm_client = LLMClient(settings)
        
        # Initialiser Qdrant
        vectorstore = QdrantVectorStore(settings, embedding_service)
        await vectorstore.connect()
        
        # Vérifier/créer la collection
        await vectorstore.initialize_collection(dense_dim=384)  # 384 pour all-MiniLM-L6-v2
        
        print(f"✅ Qdrant connecté sur {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        print(f"✅ Collection '{settings.QDRANT_COLLECTION_NAME}' prête")
        print(f"✅ Reranker: {'Activé' if settings.USE_RERANKER else 'Désactivé'}")
        
        # Afficher la configuration LLM
        if settings.LLM_BASE_URL:
            print(f"✅ LLM: vLLM local sur {settings.LLM_BASE_URL}")
        elif settings.LLM_API_KEY:
            print(f"✅ LLM: OpenAI API (model: {settings.LLM_MODEL_NAME})")
        elif settings.MISTRAL_API_KEY:
            print(f"✅ LLM: Mistral API (model: {settings.LLM_MODEL_NAME})")
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        raise
    
    yield
    
    # Shutdown: Cleanup
    if vectorstore:
        await vectorstore.disconnect()
    vectorstore = None
    embedding_service = None
    reranker_service = None
    llm_client = None
    print("🔄 Services déchargés")


# Créer l'application FastAPI
app = FastAPI(
    title="RAG System API",
    description="API pour le système RAG avec SQuAD dataset (Qdrant + Hybrid Search + Reranking)",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware pour permettre les requêtes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les origines autorisées
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Endpoint de santé."""
    return {
        "status": "healthy",
        "vectorstore_loaded": vectorstore is not None,
        "qdrant_host": settings.QDRANT_HOST,
        "qdrant_port": settings.QDRANT_PORT,
        "collection_name": settings.QDRANT_COLLECTION_NAME,
        "reranker_enabled": settings.USE_RERANKER,
        "llm_base_url": settings.LLM_BASE_URL or "Mistral API (legacy)",
        "llm_model": settings.LLM_MODEL_NAME
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint principal pour les requêtes RAG.
    
    Orchestration: Hybrid Search -> (optionnel) Reranking -> Prompt Formatting -> LLM Call -> Text Cleaning -> Translation
    """
    if vectorstore is None or embedding_service is None or llm_client is None:
        raise HTTPException(
            status_code=503,
            detail="Services not loaded. Please check server logs."
        )
    
    start_time = time.perf_counter()
    
    try:
        # 1. Retrieval avec option de reranking
        retrieved_docs = await retrieve_documents(
            vectorstore,
            request.query,
            request.k,
            use_reranker=request.use_reranker,
            reranker_service=reranker_service,
            config=settings
        )
        
        # 2. Construire le contexte
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # 3. Génération de réponse (LLM + nettoyage)
        response_text = await generate_response(
            request.query,
            context,
            settings,
            llm_client=llm_client
        )
        
        # 4. Traduction si nécessaire
        if request.language != "en":
            response_text = translate_text(
                response_text,
                request.language,
                source_lang="en"
            )
        
        # 5. Formater les documents pour la réponse
        document_responses = [
            DocumentResponse(
                page_content=doc.page_content,
                metadata=doc.metadata
            )
            for doc in retrieved_docs
        ]
        
        processing_time = time.perf_counter() - start_time
        
        return ChatResponse(
            response=response_text,
            retrieved_documents=document_responses,
            language=request.language,
            processing_time=round(processing_time, 3)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@app.get("/examples", response_model=ExampleResponse)
async def get_examples():
    """
    Retourne une liste d'exemples de questions.
    
    Essaie de charger depuis le CSV, sinon retourne des exemples hardcodés.
    """
    csv_path = "squad_2.0/train.csv"
    
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            examples = df.sample(min(10, len(df)))["question"].tolist()
            return ExampleResponse(examples=examples)
    except Exception:
        pass
    
    # Exemples de fallback
    fallback_examples = [
        "What is the capital of France?",
        "Who invented the telephone?",
        "When did World War II end?",
        "What is photosynthesis?",
        "Who wrote Romeo and Juliet?",
        "What Robert Redford movie was shot here in 1002?",
        "What has the process of revolution in the UK did?",
        "How long have railroads been important since in Montana"
    ]
    
    return ExampleResponse(examples=fallback_examples)
