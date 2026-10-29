# Compte-Rendu : JOUR 5 — Augmentation de Contexte & Génération LLM Locale

## 1. Objectifs du Jour 5

Le Jour 5 correspond aux étapes 3 et 4 du pipeline RAG :
1. **Augmentation de contexte (`src/generation/context_builder.py`)** :
   - Extraction des tranches exactes de texte `content[first_character_index:last_character_index]` depuis les fichiers sources indexés.
   - Assemblage et formatage structuré des blocs de contexte avec en-têtes explicites.
   - Gestion stricte du budget de tokens (`max_context_tokens`, défaut : 2048) avec troncature élégante.
   - Mise en cache en mémoire des lectures de fichiers (`_file_cache`) pour des performances I/O optimales.
2. **Génération LLM locale (`src/generation/llm_generator.py`)** :
   - Chargement paresseux (*lazy loading*) du modèle `Qwen/Qwen3-0.6B` et de son tokenizer via `transformers` et `torch`.
   - Exécution purement CPU en précision `float32` sans dépendance externe (daemon Ollama interdit par le sujet).
   - Prompt système de grounding strict anti-hallucination.
   - Désactivation du mode *thinking* (`enable_thinking=False` dans le chat template et nettoyage automatique des balises `<think>...</think>`).
   - Génération déterministe (`do_sample=False`, `max_new_tokens=256`).
3. **Intégration Pipeline & CLI (`src/pipeline.py`, `src/cli.py`)** :
   - Commande `answer "<query>" [--k 5]` : recherche, construction du contexte, inférence et affichage des sources avec la réponse.
   - Commande `answer_dataset --student_search_results_path <path> --save_directory <dir>` : génération par lots avec barre de progression `tqdm` et sérialisation conforme au modèle `StudentSearchResultsAndAnswer`.
4. **Validation & Tests** :
   - Tests unitaires et d'intégration dans `tests/test_context_builder.py` et `tests/test_generator.py`.
   - Couverture complète des cas limites et compatibilité stricte `flake8` et `mypy --strict`.

---

## 2. Architecture & Composants Implémentés

### 2.1. `src/generation/context_builder.py`

Le `ContextBuilder` est responsable de la transformation des métadonnées de récupération (`MinimalSource`) en un texte structuré assimilable par le modèle de langage :
- **Extraction sans corruption** : Ouvre les fichiers en `utf-8` avec `errors="ignore"`. Clampe automatiquement les indices négatifs ou dépassant la taille du fichier. Renvoie `None` sans lever d'exception si le fichier est manquant ou si la plage est invalide (`start >= end`).
- **Formatage des blocs** :
  ```text
  --- Source: data/raw/vllm-0.10.1/... [chars 1242:1995] ---
  <snippet_content>
  ```
- **Contrôle du budget de tokens** :
  - Utilise le tokenizer du modèle pour compter les tokens avec exactitude.
  - Si aucun tokenizer n'est fourni (ex. tests unitaires isolés), utilise une heuristique de repli (~4 caractères / token).
  - Empile les snippets dans l'ordre de pertinence (`top-1`, `top-2`, ...) tant que `accumulated_tokens + block_tokens <= max_context_tokens`.
  - Si même le premier extrait dépasse le budget, il est tronqué au niveau des caractères pour respecter la limite maximale autorisée.

### 2.2. `src/generation/llm_generator.py`

Le `LLMGenerator` encapsule l'inférence locale :
- **Chargement Paresseux (*Lazy Loading*)** : Les modules `torch` et `transformers`, ainsi que les poids du modèle (~1,2 Go), ne sont jamais importés ni initialisés lors de l'import du module ou lors de l'exécution des commandes d'indexation, de recherche ou d'évaluation (`index`, `search`, `search_dataset`, `evaluate`). Le démarrage de ces commandes reste donc instantané.
- **Support Hors-Ligne & Robustesse** : Tente en priorité le chargement depuis le cache local (`local_files_only=True`), évitant tout blocage réseau en environnement sandboxé ou déconnecté.
- **Prompt Système Anti-Hallucination** :
  ```text
  You are a precise technical assistant for the vLLM codebase.
  Answer the user's question concisely, accurately, and strictly grounded in the provided context.
  Do not extrapolate or hallucinate. If the context does not contain sufficient information to answer the question, state:
  "The provided context does not contain sufficient information to answer this question."
  ```
- **Suppression du Reasoning Verbose** :
  - `enable_thinking=False` pré-remplit `<think>\n\n</think>` dans le template Qwen3 pour que l'inférence commence directement sur la réponse finale.
  - Une passe regex `re.sub(r"<think>.*?</think>", "", ...)` élimine tout bloc résiduel.

### 2.3. Pipeline & Commandes CLI

- **`answer`** :
  ```bash
  uv run python -m src answer "How does PagedAttention manage KV cache?" --k 3
  ```
  Affiche les sources retrouvées (chemins relatifs et indices de caractères exacts), puis la réponse concise générée, et renvoie un objet `MinimalAnswer`.
- **`answer_dataset`** :
  ```bash
  uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/AnsweredQuestions/dataset_code_public.json \
    --save_directory data/output/search_results_and_answer/
  ```
  Lit le fichier de résultats de recherche produit au Jour 3, génère les réponses pour chaque question avec barre de progression `tqdm`, et sauvegarde un fichier JSON conforme au schéma `StudentSearchResultsAndAnswer`.

---

## 3. Résultats des Tests & Vérification

### 3.1. Tests Unitaires (`pytest`)
53 tests unitaires exécutés avec succès :
```text
tests/test_chunkers.py ..........                                        [ 18%]
tests/test_cli_edge_cases.py ......                                      [ 30%]
tests/test_context_builder.py .....                                      [ 39%]
tests/test_evaluator.py ........                                         [ 54%]
tests/test_generator.py ...                                              [ 60%]
tests/test_indexer.py ..                                                 [ 64%]
tests/test_models.py ......                                              [ 75%]
tests/test_retriever.py .....                                            [ 84%]
tests/test_tokenizer.py ........                                         [100%]

============================== 53 passed in 1.11s ==============================
```

### 3.2. Qualité de Code & Typage Stricte
- **`flake8 src tests`** : 0 avertissement, 0 erreur (respect strict de PEP 8 et limite de 79 caractères par ligne).
- **`mypy src tests --strict`** : 0 erreur sur l'ensemble des 29 fichiers sources du projet.

