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

st.markdown("---")
st.markdown(_(
    """
    Développé par [Gabriel Marie-Brisson](https://gabriel.mariebrisson.fr)
    """
))