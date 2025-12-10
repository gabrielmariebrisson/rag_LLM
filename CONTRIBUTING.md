# 🤝 Guide de Contribution

Merci de votre intérêt pour contribuer au projet RAG System ! Ce document fournit des guidelines pour contribuer au projet.

## 📋 Table des Matières

- [Code de Conduite](#code-de-conduite)
- [Comment Contribuer](#comment-contribuer)
- [Processus de Contribution](#processus-de-contribution)
- [Standards de Code](#standards-de-code)
- [Tests](#tests)
- [Documentation](#documentation)
- [Questions et Support](#questions-et-support)

---

## Code de Conduite

En participant à ce projet, vous acceptez de respecter notre code de conduite :

- **Respect** : Soyez respectueux envers tous les contributeurs
- **Collaboration** : Travaillons ensemble pour améliorer le projet
- **Bienveillance** : Soyez ouvert aux critiques constructives

---

## Comment Contribuer

### Types de Contributions

Nous acceptons différents types de contributions :

1. **Rapports de Bugs** : Signaler des problèmes
2. **Suggestions de Fonctionnalités** : Proposer de nouvelles idées
3. **Corrections** : Corriger des bugs
4. **Améliorations** : Améliorer le code existant
5. **Documentation** : Améliorer la documentation
6. **Tests** : Ajouter ou améliorer les tests

### Avant de Contribuer

1. **Vérifier les Issues existantes** : Votre problème/feature existe-t-il déjà ?
2. **Discuter les gros changements** : Ouvrir une issue pour discuter avant de coder
3. **S'assurer de pouvoir contribuer** : Vérifier la licence et les droits

---

## Processus de Contribution

### 1. Fork le Repository

```bash
# Fork sur GitHub, puis cloner votre fork
git clone https://github.com/votre-username/rag_LLM.git
cd rag_LLM
```

### 2. Configurer le Remote

```bash
# Ajouter le repository original comme upstream
git remote add upstream https://github.com/original-owner/rag_LLM.git

# Vérifier les remotes
git remote -v
```

### 3. Créer une Branche

```bash
# Mettre à jour votre fork
git checkout main
git pull upstream main

# Créer une branche feature/fix
git checkout -b feature/ma-fonctionnalite
# ou
git checkout -b fix/mon-bug
```

### 4. Développer

- Écrire du code propre et bien documenté
- Suivre les standards de code du projet
- Ajouter des tests si applicable
- Mettre à jour la documentation si nécessaire

### 5. Commit

```bash
# Commits atomiques avec messages clairs
git add .
git commit -m "feat(rag): ajout support nouveau modèle embedding"
```

**Format des messages** : `<type>(<scope>): <description>`

Types :
- `feat` : Nouvelle fonctionnalité
- `fix` : Correction de bug
- `docs` : Documentation
- `style` : Formatage
- `refactor` : Refactoring
- `test` : Tests
- `chore` : Maintenance

### 6. Push et Pull Request

```bash
# Pousser vers votre fork
git push origin feature/ma-fonctionnalite

# Créer une Pull Request sur GitHub
```

### 7. Review Process

- Attendre les commentaires des maintainers
- Répondre aux commentaires et faire les modifications nécessaires
- Une fois approuvée, votre PR sera mergée

---

## Standards de Code

### Formatage

Nous utilisons **Black** pour le formatage automatique :

```bash
# Formater le code
black app/ frontend/ scripts/

# Vérifier sans formater
black --check app/
```

**Configuration** : Line length 100, Python 3.12+

### Linting

Nous utilisons **Flake8** pour le linting :

```bash
flake8 app/ frontend/ scripts/ --max-line-length=100
```

### Type Hints

Tous les nouveaux code doivent utiliser des type hints :

```python
def retrieve_documents(
    vectorstore: QdrantVectorStore,
    query: str,
    k: int = 5
) -> List[Document]:
    ...
```

### Docstrings

Format **Google Style** :

```python
def ma_fonction(param1: str, param2: int) -> bool:
    """
    Description courte de la fonction.
    
    Args:
        param1: Description du paramètre 1.
        param2: Description du paramètre 2.
        
    Returns:
        Description de la valeur de retour.
        
    Raises:
        ValueError: Si param1 est vide.
        
    Example:
        >>> result = ma_fonction("test", 42)
        >>> print(result)
        True
    """
    ...
```


---

## Tests

### Exigences

- **Nouveaux code** : Doivent inclure des tests
- **Couverture** : Maintenir >80% de couverture
- **Tests doivent passer** : Tous les tests doivent passer avant merge

### Écrire des Tests

```python
# tests/test_rag.py
import pytest
from app.services.rag import retrieve_documents

@pytest.mark.asyncio
async def test_retrieve_documents(mock_vectorstore):
    """Test de récupération de documents."""
    query = "What is AI?"
    docs = await retrieve_documents(
        vectorstore=mock_vectorstore,
        query=query,
        k=5
    )
    assert len(docs) == 5
    assert all(doc.page_content for doc in docs)
```

### Exécuter les Tests

```bash
# Tous les tests
pytest

# Tests spécifiques
pytest tests/test_rag.py

# Avec coverage
pytest --cov=app --cov-report=html
```

---

## Documentation

### Mettre à Jour la Documentation

Si votre contribution modifie :
- **Configuration** : Mettre à jour `docs/CONFIGURATION.md`
- **API** : Mettre à jour `docs/API.md`
- **Architecture** : Mettre à jour `docs/ARCHITECTURE.md`
- **Utilisation** : Mettre à jour `README.md`

### Ajouter des Exemples

- Ajouter des exemples dans `examples/`
- Mettre à jour la documentation avec les exemples

---

## Checklist Pull Request

Avant de soumettre une PR, vérifier :

- [ ] Code formaté avec Black
- [ ] Linting passé (Flake8)
- [ ] Tests passent (pytest)
- [ ] Tests ajoutés pour nouvelles fonctionnalités
- [ ] Coverage >80%
- [ ] Docstrings ajoutées/mises à jour
- [ ] Documentation mise à jour si nécessaire
- [ ] Pas de régression (tests existants passent)
- [ ] Type hints ajoutés
- [ ] Commit messages suivent le format conventionnel

---

## Questions et Support

### Où Poser des Questions ?

- **Issues GitHub** : Pour questions techniques et bugs
- **Discussions GitHub** : Pour questions générales et discussions
- **Email** : gabriel@mariebrisson.fr (pour questions privées)

### Rapport de Bug

Lors du rapport d'un bug, inclure :

1. **Description** : Description claire du bug
2. **Reproduction** : Steps pour reproduire
3. **Comportement attendu** : Ce qui devrait se passer
4. **Comportement actuel** : Ce qui se passe réellement
5. **Environnement** :
   - OS
   - Python version
   - Versions des dépendances
6. **Logs** : Logs d'erreur si disponibles
7. **Screenshots** : Si applicable

### Suggestion de Fonctionnalité

Pour suggérer une fonctionnalité :

1. **Ouvrir une Issue** : Utiliser le template "Feature Request"
2. **Décrire le use case** : Pourquoi cette fonctionnalité est utile
3. **Proposer une implémentation** : Comment l'implémenter (optionnel)
4. **Discuter** : Attendre les retours avant d'implémenter

---

## Workflow Git Recommandé

### Branches

- `main` : Branche principale (production)
- `develop` : Branche de développement
- `feature/*` : Nouvelles fonctionnalités
- `fix/*` : Corrections de bugs
- `docs/*` : Documentation
- `test/*` : Tests

### Rebase vs Merge

Nous préférons les **rebase** pour garder l'historique propre :

```bash
# Avant de créer une PR, rebase sur upstream/main
git fetch upstream
git rebase upstream/main

# Résoudre les conflits si nécessaire
# Puis force push (sur votre fork seulement)
git push origin feature/ma-fonctionnalite --force-with-lease
```

### Petits Commits

Faire des commits atomiques et fréquents :

```bash
# ✅ Bon : commits séparés
git commit -m "feat(embeddings): ajout support nouveau modèle"
git commit -m "test(embeddings): ajout tests nouveau modèle"
git commit -m "docs(embeddings): mise à jour documentation"

# ❌ Éviter : un gros commit avec tout
git commit -m "ajout nouveau modèle"
```

---

## License

En contribuant, vous acceptez que vos contributions soient sous la même licence que le projet (MIT).

---

## Reconnaissance

Les contributeurs seront listés dans :
- `README.md` (section Contributors)
- Releases GitHub

Merci de contribuer ! 🎉

---

## Ressources

- [README.md](README.md) : Guide de démarrage rapide
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) : Guide de troubleshooting

---

**Version** : 2.0.0  
**Dernière mise à jour** : 2025-01-27

