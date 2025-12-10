"""
Application FastAPI principale pour le système RAG.

Ce module configure l'application FastAPI avec :
- Endpoints REST pour les requêtes RAG
- Configuration OpenTelemetry pour le tracing distribué
- Gestion du cycle de vie des services (Qdrant, Embeddings, Reranker, LLM)
- Middleware CORS pour le frontend Streamlit
"""
import time
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# OpenTelemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from app.core.config import settings

# Configuration OpenTelemetry
resource = Resource.create({
    "service.name": "rag-system",
    "service.version": "2.0.0"
})

# Créer le provider de traces
trace_provider = TracerProvider(resource=resource)

# Configuration de l'export OpenTelemetry
# Par défaut, on essaie d'exporter vers Jaeger, mais on peut le désactiver avec ENABLE_JAEGER_EXPORT=false
jaeger_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
jaeger_enabled = os.getenv("ENABLE_JAEGER_EXPORT", "true").lower() == "true"

if jaeger_enabled:
    try:
        # Créer l'exporter OTLP vers Jaeger
        # Note: L'exporter ne lève pas d'exception à la création, seulement lors de l'export
        # Si Jaeger n'est pas disponible, BatchSpanProcessor affichera des erreurs dans les logs
        # mais l'application continuera de fonctionner normalement
        otlp_exporter = OTLPSpanExporter(endpoint=jaeger_endpoint, insecure=True)
        span_processor = BatchSpanProcessor(otlp_exporter)
        trace_provider.add_span_processor(span_processor)
        print(f"✅ OpenTelemetry: Export vers Jaeger configuré ({jaeger_endpoint})")
        print("   ⚠️  Si Jaeger n'est pas disponible, des erreurs de connexion apparaîtront dans les logs")
        print("   💡 Pour désactiver l'export Jaeger, définissez ENABLE_JAEGER_EXPORT=false")
    except Exception as e:
        # Si la création de l'exporter échoue (peu probable), utiliser console
        print(f"⚠️  OpenTelemetry: Erreur lors de la configuration Jaeger: {e}")
        print("   → Utilisation de l'exporteur console")
        console_exporter = ConsoleSpanExporter()
        console_processor = BatchSpanProcessor(console_exporter)
        trace_provider.add_span_processor(console_processor)
else:
    # Jaeger désactivé explicitement, utiliser uniquement l'exporteur console
    print("ℹ️  OpenTelemetry: Export Jaeger désactivé (ENABLE_JAEGER_EXPORT=false)")
    print("   → Utilisation de l'exporteur console")
    console_exporter = ConsoleSpanExporter()
    console_processor = BatchSpanProcessor(console_exporter)
    trace_provider.add_span_processor(console_processor)

# Définir le provider global
trace.set_tracer_provider(trace_provider)

# Créer un tracer pour les spans manuels
tracer = trace.get_tracer(__name__)
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
    """
    Gère le cycle de vie de l'application FastAPI.
    
    Initialise tous les services au démarrage et les nettoie à l'arrêt.
    
    Args:
        app: Instance de l'application FastAPI.
        
    Yields:
        None: L'application reste active entre le yield.
        
    Raises:
        Exception: Si l'initialisation des services échoue (Qdrant, Embeddings, etc.).
        
    Notes:
        - Services initialisés : EmbeddingService, RerankerService, LLMClient, QdrantVectorStore
        - La collection Qdrant est créée automatiquement si elle n'existe pas
        - Les erreurs d'initialisation empêchent le démarrage de l'application
    """
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
        await vectorstore.initialize_collection(dense_dim=settings.DENSE_DIM)
        
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

# Instrumenter FastAPI avec OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

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
    """
    Endpoint de santé de l'application.
    
    Vérifie l'état de tous les services critiques :
    - Connexion à Qdrant
    - Chargement du vectorstore
    - Configuration du reranker
    - Configuration LLM
    
    Returns:
        dict: Dictionnaire contenant :
            - status (str): "healthy" si tous les services sont opérationnels
            - vectorstore_loaded (bool): True si le vectorstore est chargé
            - qdrant_host (str): Host de Qdrant
            - qdrant_port (int): Port de Qdrant
            - collection_name (str): Nom de la collection Qdrant
            - reranker_enabled (bool): État du reranker
            - llm_base_url (str): URL de base du LLM ou "Mistral API (legacy)"
            - llm_model (str): Nom du modèle LLM
            
    Example:
        >>> response = await health_check()
        >>> print(response["status"])  # "healthy"
        >>> print(response["vectorstore_loaded"])  # True
    """
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
    
    Orchestre le pipeline complet RAG :
    1. Hybrid Search (dense + sparse) dans Qdrant
    2. (Optionnel) Reranking avec Cross-Encoder
    3. Formatage du prompt avec contexte
    4. Génération de réponse via LLM
    5. Nettoyage du texte généré
    6. Traduction si nécessaire
    
    Args:
        request (ChatRequest): Requête contenant :
            - query (str): Question de l'utilisateur
            - k (int): Nombre de documents à récupérer (1-20)
            - language (str): Langue cible de la réponse (code ISO, ex: "fr", "en")
            - use_reranker (Optional[bool]): Override pour activer/désactiver reranker
            
    Returns:
        ChatResponse: Réponse contenant :
            - response (str): Réponse générée et traduite
            - retrieved_documents (List[DocumentResponse]): Documents sources utilisés
            - language (str): Langue de la réponse générée
            - processing_time (float): Temps d'exécution en secondes
            
    Raises:
        HTTPException 503: Si les services ne sont pas chargés (vectorestore, embedding_service, llm_client)
        HTTPException 500: Si une erreur survient lors du traitement
        
    Notes:
        - Le pipeline est instrumenté avec OpenTelemetry pour le tracing
        - La latence est mesurée et incluse dans la réponse
        - Les spans OpenTelemetry incluent les attributs : query, k, language, use_reranker
        
    Example:
        >>> request = ChatRequest(
        ...     query="What is the capital of France?",
        ...     k=5,
        ...     language="en",
        ...     use_reranker=True
        ... )
        >>> response = await chat(request)
        >>> print(response.response)  # "The capital of France is Paris."
        >>> print(f"Retrieved {len(response.retrieved_documents)} documents")
    """
    if vectorstore is None or embedding_service is None or llm_client is None:
        raise HTTPException(
            status_code=503,
            detail="Services not loaded. Please check server logs."
        )
    
    # Span racine pour la requête complète
    with tracer.start_as_current_span("rag.request") as root_span:
        root_span.set_attribute("http.method", "POST")
        root_span.set_attribute("http.route", "/chat")
        root_span.set_attribute("query", request.query)
        root_span.set_attribute("k", request.k)
        root_span.set_attribute("language", request.language)
        root_span.set_attribute("use_reranker", request.use_reranker or False)
        
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
            
            root_span.set_attribute("processing_time_ms", round(processing_time * 1000, 2))
            root_span.set_attribute("response_length", len(response_text))
            root_span.set_status(trace.Status(trace.StatusCode.OK))
            
            return ChatResponse(
                response=response_text,
                retrieved_documents=document_responses,
                language=request.language,
                processing_time=round(processing_time, 3)
            )
            
        except Exception as e:
            root_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            root_span.record_exception(e)
            raise HTTPException(
                status_code=500,
                detail=f"Error processing request: {str(e)}"
            )


@app.get("/examples", response_model=ExampleResponse)
async def get_examples():
    """
    Retourne une liste d'exemples de questions pour l'interface utilisateur.
    
    Essaie de charger des exemples aléatoires depuis le dataset SQuAD (train.csv),
    sinon retourne une liste d'exemples hardcodés.
    
    Returns:
        ExampleResponse: Réponse contenant :
            - examples (List[str]): Liste de 10 questions d'exemple (ou moins si le dataset est plus petit)
            
    Notes:
        - Les exemples sont tirés aléatoirement du fichier `data/raw/squad_2.0/train.csv`
        - Si le fichier n'existe pas ou une erreur survient, retourne des exemples de fallback
        - Les exemples de fallback sont des questions génériques en anglais
        
    Example:
        >>> response = await get_examples()
        >>> print(f"Found {len(response.examples)} example questions")
        >>> print(response.examples[0])  # "What is the capital of France?"
    """
    from app.core.paths import SQUAD_CSV
    
    csv_path = str(SQUAD_CSV)
    
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
