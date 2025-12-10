# 🛑⚠️ ATTENTION : BRANCHE DE DÉPLOIEMENT UNIQUEMENT ⚠️🛑

> **🚨 VEUILLEZ BASCULER SUR LA BRANCHE [`entreprise-rag-scale`](../../tree/entreprise-rag-scale) POUR ACCÉDER À LA VERSION OPTIMISÉE ET PROFESSIONNELLE DU PROJET.**
>
> **Cette branche actuelle est une version allégée, spécifiquement configurée pour héberger la démonstration publique sur Streamlit Cloud. Elle ne reflète pas la performance réelle du code de production.**

---

# 🧠 RAG System Demo with SQuAD 2.0

Une application de démonstration interactive de **Génération Augmentée par Récupération (RAG)**, permettant d'interroger le dataset SQuAD 2.0 (Wikipedia) via une interface web fluide.

## 🚀 Fonctionnalités de la Démo

* **Moteur de Réponse RAG** : Utilise **Mistral AI** pour la génération et **FAISS** pour la recherche vectorielle.
* **Dataset SQuAD 2.0** : Plus de 100 000 questions/réponses sur des sujets variés (Histoire, Science, Culture).
* **Interface Multilingue** : Traduction automatique de l'interface et des réponses (FR, EN, ES, DE, etc.).
* **Étude de Cas Azure** : Une section détaillée présentant l'architecture cible sur Azure, incluant une estimation financière des coûts (Azure OpenAI vs Mistral).

## 🛠️ Installation (Local)

1.  **Prérequis** : Python 3.8+
2.  **Installation des dépendances** :
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configuration** :
    Créez un fichier `.env` à la racine du projet et ajoutez votre clé API Mistral :
    ```env
    MISTRAL_API_KEY=votre_cle_api_ici
    ```
4.  **Lancement** :
    ```bash
    streamlit run rag_LLM_web.py
    ```

## 📚 Stack Technique (Démo)
- **Frontend** : Streamlit
- **Orchestration** : LangChain
- **Vector Store** : FAISS (Facebook AI Similarity Search)
- **LLM** : Mistral AI API

---
**Développé par [Gabriel Marie-Brisson](https://gabriel.mariebrisson.fr)**
