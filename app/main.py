"""Application FastAPI principale."""
import time
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import pandas as pd

from app.core.config import settings
from app.schemas import ChatRequest, ChatResponse, DocumentResponse, ExampleResponse
from app.services.rag import retrieve_documents, generate_response
from app.services.translation import translate_text


# Variable globale pour stocker le vectorstore
vectorstore: Optional[FAISS] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application."""
    global vectorstore
    
    # Startup: Charger le vectorstore FAISS
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME
        )
        vectorstore = FAISS.load_local(
            settings.FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✅ Vectorstore chargé depuis {settings.FAISS_INDEX_DIR}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du vectorstore: {e}")
        raise
    
    yield
    
    # Shutdown: Cleanup si nécessaire
    vectorstore = None
    print("🔄 Vectorstore déchargé")


# Créer l'application FastAPI
app = FastAPI(
    title="RAG System API",
    description="API pour le système RAG avec SQuAD dataset",
    version="1.0.0",
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
        "vectorstore_loaded": vectorstore is not None
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint principal pour les requêtes RAG.
    
    Orchestration: Retrieval -> Prompt Formatting -> LLM Call -> Text Cleaning -> Translation
    """
    if vectorstore is None:
        raise HTTPException(
            status_code=503,
            detail="Vectorstore not loaded. Please check server logs."
        )
    
    start_time = time.perf_counter()
    
    try:
        # 1. Retrieval (via executor pour ne pas bloquer l'event loop)
        retrieved_docs = await retrieve_documents(
            vectorstore,
            request.query,
            request.k
        )
        
        # 2. Construire le contexte
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # 3. Génération de réponse (LLM + nettoyage)
        response_text = await generate_response(
            request.query,
            context,
            settings
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

