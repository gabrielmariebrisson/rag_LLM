import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from mistralai import Mistral
from dotenv import load_dotenv
import numpy as np
import re
import pandas as pd
import plotly.graph_objects as go
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

# Configuration de la page - DOIT être en premier
st.set_page_config(page_title="RAG System with SQuAD", layout="wide")

# Initialisation de la langue
if 'language' not in st.session_state:
    st.session_state.language = 'fr'

# Sélecteur de langue
lang = st.sidebar.selectbox(
    "🌍 Language / Langue", 
    options=list(LANGUAGES.keys()),
    format_func=lambda x: LANGUAGES[x],
    index=list(LANGUAGES.keys()).index(st.session_state.language)
)

st.session_state.language = lang

# Cache pour les traductions
if 'translations_cache' not in st.session_state:
    st.session_state.translations_cache = {}

def _(text, source_lang='fr'):
    """Fonction de traduction automatique avec cache"""
    if lang == source_lang:
        return text
    
    cache_key = f"{source_lang}_{lang}_{text}"
    if cache_key in st.session_state.translations_cache:
        return st.session_state.translations_cache[cache_key]
    
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

            if resultat:
                texte_reponse = resultat.group(1)
            else:
                texte_reponse = result["response"]

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
                motif_doc = r"Title:\s*(.*?)\nContext:\s*(.*?)\nAnswer:\s*(.*?)$"
                resultat = re.search(motif_doc, doc.page_content, re.DOTALL)
                
                question = doc.metadata.get("question", "Question non disponible")
                
                if resultat:
                    titre = resultat.group(1)
                    contexte = resultat.group(2)
                    reponse_brute = resultat.group(3)
                    
                    motif_reponse = r"'text'\s*:\s*array\(\['(.*?)'\]"
                    match_reponse = re.search(motif_reponse, reponse_brute)
                    reponse = match_reponse.group(1) if match_reponse else reponse_brute
                    
                    with st.expander(f"📄 Document {j+1} - {_(titre, 'en')}", expanded=False):
                        st.markdown(f"**{_('Question')}:** {_(question, 'en')}")
                        st.markdown(f"**{_('Contexte')}:**")
                        st.write(_(contexte, 'en'))
                        st.markdown(f"**{_('Réponse')}:** {_(reponse, 'en')}")
                        st.divider()
                else:
                    with st.expander(f"📄 Document {j+1}", expanded=False):
                        st.markdown(f"**{_('Question')}:** {_(question, 'en')}")
                        st.write(doc.page_content)
                        st.divider()

st.markdown("---")

# Section Architecture et Projet
st.title(_("Retrieval Augmented Generation avec Azure OpenAI et Stockage Blob"))

with st.container():
    st.header(_("1. Présentation du Projet"))
    st.markdown(_("""
    Bien que la partie du haut ne soit pas fait avec azure, le projet initile est basé sur azure.
    Le côté interactif au dessus sert a créer une interface web pour tester le projet, afin de mieux comprendre le fonctionnement.
    Ce projet vise à concevoir un pipeline RAG (Retrieval Augmented Generation) sur la plateforme Azure. 
    L'objectif est de permettre à un chatbot basé sur Azure OpenAI de répondre de manière pertinente à des 
    questions à partir de données externes stockées dans Azure Blob Storage et indexées dans Azure Cognitive Search.
    Les outils gratuit utilisés ont un fonctionnement similaire à ceux d'Azure, permettant ainsi de simuler le pipeline RAG.
                  
    L'interet d'un rag est qu'il permet d'avoir une réponse plus pertinente car le LLM va se baser sur des données externes. Ainsi on minimise les hallucinations du modèle de langage.
    Dans certains de mes projets j'ai eu l'occasion d'ajouter du contexte externe à un modèle de langage, de facon automatique, et le résultat est toujours plus pertinent.
                  
    Le RAG va trouver des documents pertinents dans une base de données externe, avant que le modèle de langage recoive la question. Le modèle de langage va ensuite générer une réponse en se basant sur les documents récupérés.
    """))

with st.container():
    st.header(_("2. Architecture Générale"))
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(_("""
        **Composants principaux :**
        
        - **Azure Blob Storage** : héberge les fichiers CSV utilisés comme source de connaissance 
          (ex. : base de données des vins)
        
        - **Azure Cognitive Search (aka Azure IA Search)** : gère l'indexation et la recherche vectorielle grâce aux embeddings
        
        - **Azure OpenAI Service** : fournit les modèles de génération et d'embedding via les déploiements Chatgpt.
        
        - **LangChain** : gère l'orchestration entre recherche, vectorisation et génération de texte, c'est a dire l'orchestration du pipeline RAG.
        
        """))
    
    with col2:
        st.image("templates/assets/architecture-diagram.png", caption=_("Architecture du pipeline RAG"), use_container_width=True)

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
        dans l'index et envoie les résultats au modèle GPT Azure (`demo`) pour générer 
        une réponse contextuelle.
        """))
        st.code("""
results = vectorstore.similarity_search_with_relevance_scores(
    query, k=5
)
response = openai.ChatCompletion.create(
    deployment="demo",
    messages=[
        {"role": "system", "content": "Assistant RAG"},
        {"role": "user", "content": f"Context: {results}\\n\\nQuestion: {query}"}
    ]
)
        """, language="python")

with st.container():
    st.header(_("4. Fonctionnalités du Code"))
    
    features = [
        ("🔐", _("Chargement automatique des clés API et endpoints à partir d'un fichier `.env`")),
        ("🔑", _("Connexion sécurisée à Azure Search via les variables d'environnement")),
        ("📊", _("Indexation automatique des documents à partir de CSV")),
        ("🎯", _("Recherche vectorielle avec score de pertinence")),
        ("💬", _("Génération de réponses par `openai.ChatCompletion.create()`"))
    ]
    
    for icon, feature in features:
        st.markdown(f"{icon} {feature}")

with st.container():
    st.header(_("5. Résultats et Performances"))
    
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

# Section Coûts
st.header(_("📋 Hypothèses de Calcul"))

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(_("Taille du Dataset"), "5 Go", _("Données brutes"))
with col2:
    st.metric(_("Requêtes/Mois"), "100 000", _("Scénario modéré à élevé"))
with col3:
    st.metric(_("Tokens/Requête"), "1 000", "500 entrée + 500 sortie")

st.markdown(_("""
Les hypothèses suivantes servent de base à cette estimation :

- **Modèle d'Embedding :** `all-MiniLM-L6-v2` 
- **Ratio de Stockage :** Le volume de l'index vectoriel est estimé à **1,5 fois** la taille des données brutes (soit 7,5 Go)
- **Fréquence des Requêtes LLM :** 100 000 requêtes par mois
- **Taille Moyenne Requête/Réponse :** 1 000 jetons par requête (500 jetons d'entrée + 500 jetons de sortie)
"""))

st.header(_("🗃️ Coûts Fixes (Infrastructure Azure)"))

infrastructure_data = {
    _("Composant Azure"): [
        "Azure AI Search",
        "Azure Blob Storage",
        "Azure App Service",
        "**TOTAL**"
    ],
    _("Rôle"): [
        _("Base de données vectorielle et recherche"),
        _("Stockage des données brutes (5 Go)"),
        _("Hébergement de l'application Streamlit"),
        ""
    ],
    _("Niveau de Service"): [
        "Basic Tier (1 Search Unit)",
        "Standard LRS - Hot Tier",
        "Basic Tier (B1)",
        ""
    ],
    _("Coût Mensuel (USD)"): [
        73.73,
        0.09,
        54.75,
        128.57
    ]
}

df_infrastructure = pd.DataFrame(infrastructure_data)

st.dataframe(
    df_infrastructure,
    use_container_width=True,
    hide_index=True,
    column_config={
        _("Coût Mensuel (USD)"): st.column_config.NumberColumn(format="$%.2f")
    }
)

st.markdown(_("""
**Justification :**
- **Azure AI Search (Basic Tier)** : 73,73 $/mois - Offre 5 Go de quota vectoriel et 15 Go de stockage total, suffisant pour 5 Go de données brutes
- **Azure Blob Storage** : 0,09 $/mois - Coût très faible pour 5 Go (0,018 $/Go/mois en Hot Tier)
- **Azure App Service (B1)** : 54,75 $/mois - 1 CPU, 1.75 GB RAM pour une application de production légère avec SLA
"""))

st.header(_("🤖 Coûts Variables (Utilisation du LLM)"))

st.markdown(_("""
**Scénario : 100 000 Requêtes/Mois**
- **Total Tokens d'Entrée :** 100,000 × 500 = 50,000,000 jetons (50 Millions)
- **Total Tokens de Sortie :** 100,000 × 500 = 50,000,000 jetons (50 Millions)
"""))

llm_data = {
    _("Modèle LLM"): [
        "Mistral Large (Azure)",
        "GPT-4o (OpenAI/Azure)",
        "Mistral Tiny (API Directe)"
    ],
    _("Prix Input ($/1M)"): [4.00, 2.50, 0.14],
    _("Prix Output ($/1M)"): [12.00, 10.00, 0.14],
    _("Coût Input (50M)"): [200.00, 125.00, 7.50],
    _("Coût Output (50M)"): [600.00, 500.00, 7.50],
    _("Coût Total (USD)"): [800.00, 625.00, 15.00]
}

df_llm = pd.DataFrame(llm_data)

st.dataframe(
    df_llm,
    use_container_width=True,
    hide_index=True,
    column_config={
        _("Prix Input ($/1M)"): st.column_config.NumberColumn(format="$%.2f"),
        _("Prix Output ($/1M)"): st.column_config.NumberColumn(format="$%.2f"),
        _("Coût Input (50M)"): st.column_config.NumberColumn(format="$%.2f"),
        _("Coût Output (50M)"): st.column_config.NumberColumn(format="$%.2f"),
        _("Coût Total (USD)"): st.column_config.NumberColumn(format="$%.2f")
    }
)

st.markdown(_("""
**Note :** Le coût de **Mistral Tiny** est inclus à titre de référence, car c'est le modèle que vous utilisez actuellement. 
Il est environ **40 fois moins cher** que GPT-4o, mais sa performance est significativement inférieure aux modèles de pointe.
"""))

st.header(_("📊 Synthèse des Coûts Mensuels Totaux"))

summary_data = {
    _("Scénario"): [
        "Azure RAG + Mistral Large",
        "Azure RAG + GPT-4o",
        "RAG Local + Mistral Tiny"
    ],
    _("Coût Fixe (Infrastructure)"): [128.57, 128.57, 0.00],
    _("Coût Variable (LLM)"): [800.00, 625.00, 15.00],
    _("Coût Total Mensuel (USD)"): [928.57, 753.57, 15.00]
}

df_summary = pd.DataFrame(summary_data)

st.dataframe(
    df_summary,
    use_container_width=True,
    hide_index=True,
    column_config={
        _("Coût Fixe (Infrastructure)"): st.column_config.NumberColumn(format="$%.2f"),
        _("Coût Variable (LLM)"): st.column_config.NumberColumn(format="$%.2f"),
        _("Coût Total Mensuel (USD)"): st.column_config.NumberColumn(format="$%.2f")
    }
)

col1, col2 = st.columns(2)

with col1:
    st.subheader(_("💵 Répartition des Coûts (Mistral Large)"))
    
    costs_mistral = [128.57, 800.00]
    labels_mistral = [_("Infrastructure"), _("LLM (Mistral Large)")]
    colors = ["#3498db", "#e74c3c"]
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=labels_mistral,
        values=costs_mistral,
        marker=dict(colors=colors),
        textposition="inside",
        textinfo="label+percent+value"
    )])
    
    fig_pie.update_layout(
        title=_("Coût Total : 928,57 $/mois"),
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader(_("📈 Comparaison des Scénarios"))
    
    scenarios = ["Mistral Large\n(Azure)", "GPT-4o\n(Azure)", "Mistral Tiny\n(Local)"]
    costs = [928.57, 753.57, 15.00]
    colors_bar = ["#e74c3c", "#f39c12", "#27ae60"]
    
    fig_bar = go.Figure(data=[go.Bar(
        x=scenarios,
        y=costs,
        marker=dict(color=colors_bar),
        text=[f"${c:.2f}" for c in costs],
        textposition="outside"
    )])
    
    fig_bar.update_layout(
        title=_("Coût Total Mensuel par Scénario"),
        yaxis_title=_("Coût (USD)"),
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)

st.header(_("🎯 Analyse et Conclusion"))

st.markdown(_("""
### Points Clés

**1. Coût d'Infrastructure Fixe :** Environ **130 $ par mois**
   - Le niveau **Basic** d'Azure AI Search est suffisant pour gérer les 5 Go de données
   - Les coûts de stockage et d'hébergement sont négligeables en comparaison

**2. Facteur de Coût Dominant : Le LLM**
   - Le LLM représente plus de **80 %** du coût total dans un scénario d'utilisation modérée
   - Le choix du modèle a un impact considérable sur le budget mensuel

**3. Comparaison des Modèles**
   - **Mistral Large (Azure)** : 928,57 $/mois - Modèle de pointe avec bonne performance
   - **GPT-4o (Azure)** : 753,57 $/mois - Légèrement moins cher malgré un prix input plus faible
   - **Mistral Tiny (Local)** : 15,00 $/mois - Extrêmement économique, mais performance limitée

**4. Observation Intéressante**
   - Sur Azure, **GPT-4o** est plus économique que **Mistral Large**, bien que l'API directe de Mistral Large soit généralement moins chère
   - Cela souligne l'importance des **frais de plateforme Azure** pour les modèles tiers

### Limitations de cette Estimation

- Les coûts sont basés sur un scénario de **100 000 requêtes/mois**. Votre utilisation réelle peut varier
- Les prix Azure peuvent fluctuer ; consultez le [Calculateur de Prix Azure](https://azure.microsoft.com/en-us/pricing/calculator/) pour les estimations les plus récentes
- Les coûts de bande passante sortante (egress) ne sont pas inclus dans cette estimation
- Les coûts de développement, de maintenance et de support ne sont pas pris en compte
"""))

st.markdown("---")
st.markdown(_(
    """
    Développé par [Gabriel Marie-Brisson](https://gabriel.mariebrisson.fr)
    """
))