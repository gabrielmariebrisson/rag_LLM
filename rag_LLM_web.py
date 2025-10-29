import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from mistralai import Mistral
from dotenv import load_dotenv
from numpy import ndarray
import numpy as np
import re

from deep_translator import GoogleTranslator

LANGUAGES = {
    "fr": "🇫🇷 Français",
    "en": "🇬🇧 English",
    "es": "🇪🇸 Español",
    "de": "🇩🇪 Deutsch",
    "it": "🇮🇹 Italiano",
    "pt": "🇵🇹 Português",
    "ja": "🇯🇵 日本語",
    "zh-CN": "🇨🇳 中文",
    "ar": "🇸🇦 العربية",
    "ru": "🇷🇺 Русский"
}

# Initialisation de la langue
if 'language' not in st.session_state:
    st.session_state.language = 'fr'

# Sélecteur de langue
lang = st.sidebar.selectbox(
    "🌐 Language / Langue", 
    options=list(LANGUAGES.keys()),
    format_func=lambda x: LANGUAGES[x],
    index=list(LANGUAGES.keys()).index(st.session_state.language)
)

st.session_state.language = lang

# Cache pour les traductions (évite de retranduire à chaque fois)
if 'translations_cache' not in st.session_state:
    st.session_state.translations_cache = {}

def _(text, source_lang='fr'):
    """Fonction de traduction automatique avec cache"""
    
    # Si la langue cible est la même que la source, pas de traduction
    if lang == source_lang:
        return text
    
    # Vérifier le cache
    cache_key = f"{source_lang}_{lang}_{text}"
    if cache_key in st.session_state.translations_cache:
        return st.session_state.translations_cache[cache_key]
    
    # Traduire
    try:
        translated = GoogleTranslator(source=source_lang, target=lang).translate(text)
        st.session_state.translations_cache[cache_key] = translated
        return translated
    except:
        return text

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL_NAME = os.getenv("MISTRAL_MODEL_NAME", "mistral-tiny-2407")


@st.cache_resource
def load_vectorstore():
    """Charge le vectorstore FAISS sauvegardé"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vectorstore

try:
    vectorstore = load_vectorstore()
except Exception as e:
    st.error(_(f"❌ Erreur lors du chargement du vectorstore: {e}"))
    st.stop()

def rag_query(query, k=5):
    """Fonction RAG utilisant Mistral API"""
    if not MISTRAL_API_KEY or MISTRAL_API_KEY.strip() == "":
        raise ValueError(_("MISTRAL_API_KEY is not set or is empty. Please set it in your .env file."))
    
    results = vectorstore.similarity_search(query, k=k)
    
    context = "\n\n".join([doc.page_content for doc in results])
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Write responses in complete, well-developed sentences. Express ideas clearly and naturally, avoiding overly brief or list-style answers."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    
    with Mistral(api_key=MISTRAL_API_KEY) as mistral:
        response = mistral.chat.complete(
            model=MISTRAL_MODEL_NAME, 
            messages=messages, 
            stream=False
        )
    
    return {
        "response": response.choices[0].message.content,
        "retrieved_documents": results,
        "context": context
    }

# Bouton de redirection
st.markdown(
    f"""
    <a href="https://gabriel.mariebrisson.fr" target="_blank" style="text-decoration:none;">
    <div style="
    display: inline-block;
    background: linear-gradient(135deg, #6A11CB 0%, #2575FC 100%);
    color: white;
    padding: 12px 25px;
    border-radius: 30px;
    text-align: center;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    box-shadow: 0 4px 15px rgba(37, 117, 252, 0.3);
    transition: all 0.3s ease;
    text-transform: uppercase;
    letter-spacing: 1px;
    border: 2px solid transparent;
    position: relative;
    overflow: hidden;
    ">
    {_("Retour")}
    </div>
    </a>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title=_("RAG System with SQuAD"), layout="wide")

st.title(_("RAG System with SQuAD Dataset"))
st.markdown(_("""
Il s'agit d'un système de génération augmentée par récupération (RAG) construit avec :
- **Modèle d'embedding** : sentence-transformers/all-MiniLM-L6-v2  
- **Base de données vectorielle** : FAISS (LangChain)  
- **Modèle de langage (LLM)** : API Mistral  
- **Jeu de données** : SQuAD 2.0
"""))


if not MISTRAL_API_KEY:
    st.error(_("⚠️ MISTRAL_API_KEY n'est pas définie. Veuillez la configurer dans votre fichier .env"))
    st.stop()

if 'history' not in st.session_state:
    st.session_state.history = []

st.sidebar.header(_("⚙️ Paramètres"))
num_results = st.sidebar.slider(_("Nombre de documents à récupérer"), 1, 10, 2)
st.sidebar.markdown(_(f"**Modèle**: {MISTRAL_MODEL_NAME}"))

st.sidebar.markdown("---")
st.sidebar.header(_("📊 À propos du dataset"))
st.sidebar.markdown(_("""
**SQuAD 2.0** (Stanford Question Answering Dataset) contient :
- Plus de 100 000 questions
- Questions sur des articles Wikipedia
- Domaines variés : histoire, science, géographie, culture, etc.

**Exemples de sujets :**
- 🏛️ Histoire et politique
- 🔬 Sciences et technologie
- 🌍 Géographie
- 🎭 Arts et littérature
- ⚽ Sports
"""))

@st.cache_data
def load_example_questions():
    """Charge quelques exemples de questions depuis le vectorstore"""
    import pandas as pd
    try:
        df = pd.read_csv("squad_2.0/train.csv")

        examples = df.sample(min(10, len(df)))["question"].tolist()
        return examples
    except:
        return [
            "What is the capital of France?",
            "Who invented the telephone?",
            "When did World War II end?",
            "What is photosynthesis?",
            "Who wrote Romeo and Juliet?",
            'What Robert Redford movie was shot here in 1002?',
            'What has the process of revolution in the UK did?', 
            'How long have railroads been important since in Montana'
        ]


example_questions = load_example_questions()

st.header(_("❓ Posez votre question"))
with st.expander(_("💡 Exemples de questions du dataset SQuAD"), expanded=False):
    st.markdown(_("Voici quelques exemples de questions auxquelles le système peut répondre :"))
    cols = st.columns(2)

    for i, example in enumerate(example_questions[:8]):
        col_idx = i % 2
        with cols[col_idx]:
            if st.button(f"📌 {_(example[:60], 'en')}{'...' if len(example) > 60 else ''}", key=f"example_{i}"):
                st.session_state.query_input = example
                st.session_state.selected_question = example
                st.rerun()

    
default_query = st.session_state.get('selected_question', '')
query = st.text_input(
    _("Entrez votre question:"), 
    value=default_query,
    placeholder=_("Ex: What is the capital of France?"),
)

col1, col2 = st.columns([1, 5])
with col1:
    submit_button = st.button(_("🔍 Rechercher"), type="primary")
with col2:
    clear_button = st.button(_("🗑️ Effacer l'historique"))

if clear_button:
    st.session_state.history = []
    if 'selected_question' in st.session_state:
        del st.session_state.selected_question
    st.rerun()

if submit_button and query and query.strip():
    with st.spinner(_("🔄 Génération de la réponse...")):
        try:
            result = rag_query(query, k=num_results)

            motif_principal = r"array\(\['(.*?)'\],"
            motif_secondaire = r"'text':\s*'([^']+)'"

            resultat = re.search(motif_principal, result["response"])

            if not resultat:
                resultat = re.search(motif_secondaire, result["response"])

            # Si on n’a trouvé aucun match, on prend directement la réponse brute
            if resultat:
                texte_reponse = resultat.group(1)
            else:
                texte_reponse = result["response"]

            # Sauvegarde dans l’historique
            st.session_state.history.append({
                "query": query,
                "response": texte_reponse,
                "retrieved_documents": result["retrieved_documents"]
            })
            
            if 'selected_question' in st.session_state:
                del st.session_state.selected_question
            
            st.success(_("✅ Réponse générée avec succès !"))
            st.subheader(_("💬 Réponse"))

            st.markdown(f"**{_(texte_reponse, 'en')}**")


        
        except ValueError as e:
            st.error(_(f"❌ Erreur de configuration: {e}"))
            st.info(_("💡 Vérifiez que votre clé API Mistral est correctement configurée dans le fichier .env"))
        except Exception as e:
            st.error(_(f"❌ Une erreur s'est produite: {str(e)}"))
            with st.expander(_("🔍 Détails de l'erreur")):
                st.exception(e)

elif submit_button and (not query or not query.strip()):
    st.warning(_("⚠️ Veuillez entrer une question."))

if st.session_state.history:
    st.header(_("📚 Historique des requêtes"))
    for i, item in enumerate(reversed(st.session_state.history)):
        with st.expander(_(f"Query {len(st.session_state.history) - i}: {item['query']}"), expanded=False):
            st.write(_("**Réponse:**"))
            st.info(_(item["response"], 'en'))

            
            st.write(_("**Documents récupérés:**"))
            for j, doc in enumerate(item["retrieved_documents"]):
                # Extraire les informations du document
                motif_doc = r"Title:\s*(.*?)\nContext:\s*(.*?)\nAnswer:\s*(.*?)$"
                resultat = re.search(motif_doc, doc.page_content, re.DOTALL)
                
                # Récupérer la question depuis les métadonnées
                question = doc.metadata.get("question", "Question non disponible")
                
                if resultat:
                    titre = resultat.group(1)
                    contexte = resultat.group(2)
                    reponse_brute = resultat.group(3)
                    
                    # Extraire le texte de la réponse depuis le dictionnaire
                    motif_reponse = r"'text'\s*:\s*array\(\['(.*?)'\]"
                    match_reponse = re.search(motif_reponse, reponse_brute)
                    reponse = match_reponse.group(1)
                    
                    with st.expander(f"📄 Document {j+1} - {_(titre, 'en')}", expanded=False):
                        st.markdown(f"**{_('Question')}:** {_(question, 'en')}")
                        st.markdown(f"**{_('Contexte')}:**")
                        st.write(_(contexte, 'en'))
                        st.markdown(f"**{_('Réponse')}:** {_(reponse, 'en')}")
                        st.divider()
                else:
                    # Fallback si le regex ne match pas
                    with st.expander(f"📄 Document {j+1}", expanded=False):
                        st.markdown(f"**{_('Question')}:** {_(question, 'en')}")
                        st.write(doc.page_content)
                        st.divider()


# Titre principal
st.title(_("Portfolio de Projet : Implémentation d'un Système RAG"))
st.subheader(_("Retrieval Augmented Generation avec Azure OpenAI et Stockage Blob"))

# 1. Présentation du Projet
with st.container():
    st.header(_("1. Présentation du Projet"))
    st.markdown(_("""
    Ce projet vise à concevoir un pipeline RAG (Retrieval Augmented Generation) sur la plateforme Azure. 
    L'objectif est de permettre à un chatbot basé sur Azure OpenAI de répondre de manière pertinente à des 
    questions à partir de données externes stockées dans Azure Blob Storage et indexées dans Azure Cognitive Search.
    """))

# 2. Architecture Générale
with st.container():
    st.header(_("2. Architecture Générale"))
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(_("""
        **Composants principaux :**
        
        - **Azure Blob Storage** : héberge les fichiers CSV utilisés comme source de connaissance 
          (ex. : base de données des vins)
        
        - **Azure Cognitive Search** : gère l'indexation et la recherche vectorielle grâce aux embeddings
        
        - **Azure OpenAI Service** : fournit les modèles de génération et d'embedding via les déploiements 
          "demo-alfredo" (LLM) et "demo-embedding"
        
        - **LangChain** : gère l'orchestration entre recherche, vectorisation et génération de texte
        """))
    
    with col2:
        # Vous pouvez ajouter l'image ici si vous l'avez
        st.info(_("💡 **Architecture du pipeline RAG**\n\nDonnées → Blob Storage → Cognitive Search → OpenAI → Réponse"))

# 3. Étapes de Développement
with st.container():
    st.header(_("3. Étapes de Développement"))
    
    tabs = st.tabs([
        _("📥 Chargement"),
        _("✂️ Préparation"),
        _("🔢 Embedding"),
        _("🔍 Recherche")
    ])
    
    with tabs[0]:
        st.subheader(_("a. Chargement des Données"))
        st.markdown(_("""
        Les fichiers de données (ex : `wine-ratings.csv`) sont importés depuis le Blob Storage 
        et chargés avec la classe `CSVLoader` de LangChain.
        """))
        st.code("""
from langchain.document_loaders import CSVLoader

loader = CSVLoader("wine-ratings.csv")
documents = loader.load()
        """, language="python")
    
    with tabs[1]:
        st.subheader(_("b. Préparation et Découpage"))
        st.markdown(_("""
        Les documents sont découpés en fragments de 1000 caractères pour une indexation efficace 
        via le module `CharacterTextSplitter`.
        """))
        st.code("""
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
        """, language="python")
    
    with tabs[2]:
        st.subheader(_("c. Embedding et Indexation"))
        st.markdown(_("""
        Chaque fragment est converti en vecteur d'embedding à l'aide du modèle `demo-embedding`. 
        Les vecteurs sont ensuite stockés dans Azure Cognitive Search.
        """))
        st.code("""
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import AzureCognitiveSearch

embeddings = AzureOpenAIEmbeddings(
    deployment="demo-embedding"
)
vectorstore = AzureCognitiveSearch.from_documents(
    chunks, embeddings
)
        """, language="python")
    
    with tabs[3]:
        st.subheader(_("d. Recherche et Génération"))
        st.markdown(_("""
        Lorsqu'une requête utilisateur est reçue, le système exécute une recherche par similarité 
        dans l'index et envoie les résultats au modèle GPT Azure (`demo-alfredo`) pour générer 
        une réponse contextuelle.
        """))
        st.code("""
results = vectorstore.similarity_search_with_relevance_scores(
    query, k=5
)
response = openai.ChatCompletion.create(
    deployment="demo-alfredo",
    messages=[
        {"role": "system", "content": "Assistant RAG"},
        {"role": "user", "content": f"Context: {results}\\n\\nQuestion: {query}"}
    ]
)
        """, language="python")

# 4. Fonctionnalités du Code
with st.container():
    st.header(_("4. Fonctionnalités du Code"))
    
    features = [
        ("🔐", _("Chargement automatique des clés API et endpoints à partir d'un fichier `.env`")),
        ("🔒", _("Connexion sécurisée à Azure Search via les variables d'environnement")),
        ("📊", _("Indexation automatique des documents à partir de CSV")),
        ("🎯", _("Recherche vectorielle avec score de pertinence")),
        ("💬", _("Génération de réponses par `openai.ChatCompletion.create()`"))
    ]
    
    for icon, feature in features:
        st.markdown(f"{icon} {feature}")

# 5. Exemple de Résultat
with st.container():
    st.header(_("5. Exemple de Résultat"))
    
    st.markdown(_("**Requête :**"))
    st.info(_("What is the best Cabernet Sauvignon wine in Napa Valley above 94 points?"))
    
    st.markdown(_("**Résultat :**"))
    st.success(_("""
    → Le système renvoie les 5 documents les plus pertinents, extrait le contenu du plus pertinent 
    et l'utilise pour enrichir la réponse générée par le modèle GPT.
    """))

# 6. Outils et Technologies
with st.container():
    st.header(_("6. Outils et Technologies"))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(_("""
        **Langage :**
        - 🐍 Python
        
        **Bibliothèques :**
        - 🦜 LangChain
        - 🤖 OpenAI
        - ⚙️ dotenv
        """))
    
    with col2:
        st.markdown(_("""
        **Services Azure :**
        - ☁️ Azure OpenAI
        - 🔍 Azure Cognitive Search
        - 📦 Azure Blob Storage
        
        **Format de données :**
        - 📊 CSV
        """))

# 7. Difficultés et Optimisations
with st.container():
    st.header(_("7. Difficultés et Optimisations"))
    
    challenges = [
        _("Gestion des connexions sécurisées à Azure via des variables d'environnement"),
        _("Encodage des textes et taille des chunks pour maximiser la pertinence des embeddings"),
        _("Optimisation du scoring vectoriel dans Azure Search pour accélérer la recherche")
    ]
    
    for challenge in challenges:
        st.markdown(f"- ⚡ {challenge}")

# 8. Résultats et Performances
with st.container():
    st.header(_("8. Résultats et Performances"))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label=_("Temps de réponse moyen"),
            value="< 2s",
            delta=_("Excellent")
        )
    
    with col2:
        st.metric(
            label=_("Pertinence"),
            value=_("Élevée"),
            delta=_("Cohérent")
        )
    
    with col3:
        st.metric(
            label=_("Type de données"),
            value=_("Non structurées"),
            delta=_("Complexes")
        )
    
    st.markdown(_("""
    Le système permet de récupérer et de synthétiser des informations complexes à partir de bases 
    de données non structurées. Les résultats de similarité sont pertinents et cohérents avec les 
    données sources.
    """))

# 9. Perspectives d'Amélioration
with st.container():
    st.header(_("9. Perspectives d'Amélioration"))
    
    improvements = [
        ("🔄", _("Connexion à un stockage Blob dynamique pour actualiser l'index en temps réel")),
        ("🌐", _("Ajout d'une interface web interactive pour tester le RAG directement depuis le navigateur")),
        ("⚡", _("Intégration d'un cache Redis pour réduire le coût des requêtes récurrentes"))
    ]
    
    for icon, improvement in improvements:
        st.markdown(f"{icon} {improvement}")

        
st.markdown("---")
st.markdown(_(
    """
    Développé par [Gabriel Marie-Brisson](https://gabriel.mariebrisson.fr)
    """
))