**Gemini**

Ta sélection d'outils (`bm25s`, `chonkie`, `fire`, `tqdm`, `pydantic`) est pertinente pour boucler le projet en une semaine: tu gardes le contrôle total sur ton pipeline sans la boîte noire d'un framework comme LangChain, tout en profitant d'implémentations C/Numba ultra-rapides.

Avant d'attaquer le planning jour par jour, voici **3 recadrages techniques indispensables** sur les librairies que tu as mentionnées pour t'éviter de perdre deux jours en milieu de semaine:

1. **Attention au piège `ollama` vs `Hugging Face` (`Qwen/Qwen3-0.6B`)**: Le sujet impose le modèle `Qwen/Qwen3-0.6B` par défaut et précise que lors de la correction, l'évaluateur et les scripts de test lancent uniquement `uv sync` puis tes commandes `uv run python -m src ...`. Or, `ollama` nécessite un démon en arrière-plan (`ollama serve`) qui n'est pas installé par défaut sur les postes de 42 et que `uv sync` ne peut pas installer. Utilise directement **`transformers` + `torch`** (sur CPU) pour charger `Qwen/Qwen3-0.6B` comme tu l'as déjà fait sur ton projet précédent: tout s'installe proprement via `uv sync` et fonctionne sans serveur externe.
2. **Le piège de `chonkie` (Tokens vs Caractères)**: Le sujet impose que `--max_chunk_size` (2000 par défaut) soit mesuré **strictement en nombre de caractères** (`last_character_index - first_character_index <= 2000`). Beaucoup de _chunkers_ dans `chonkie` raisonnent par défaut en _tokens_. Si tu utilises `chonkie`, tu dois impérativement le configurer en mode caractères (`tokenizer="character"` ou équivalent selon la classe) et ajouter un garde-fou final qui vérifie et redécoupe tout chunk dont `end_index - start_index > max_chunk_size`.
   MD+ 1
3. **Le tokenizer par défaut de `bm25s` sur du code Python**: `bm25s` est une fusée (il indexe 15 000 chunks en moins d'une seconde), mais son tokenizer par défaut est conçu pour de la prose anglaise. Sur du code, si une question demande _"How does the async LLM engine handle requests?"_ et que le code contient `class AsyncLLMEngine` ou `async_llm_engine`, un tokenizer naïf ne fera jamais le lien. Tu devras lui injecter une fonction de tokenisation personnalisée.

## Roadmap sur 7 Jours (Mandatory + Bonus à fort ROI)

| Jour       | Objectif principal                   | Livrables clés & jalons de validation                                                                |
| ---------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| **Jour 1** | **Fondations, Modèles & Évaluateur** | Architecture `src/`, `Makefile`, modèles `pydantic`, CLI `Fire` et commande `evaluate`.              |
| **Jour 2** | **Exploration & Double Chunking**    | Filtrage du corpus `vllm-0.10.1` et implémentation des 2 stratégies (Code vs Docs).                  |
| **Jour 3** | **Indexation `bm25s` & Recherche**   | Commandes `index`, `search`, `search_dataset` et passage des seuils à la `moulinette`. MD            |
| **Jour 4** | **Optimisation du Recall@5**         | Tokenisation CamelCase/snake*case, injection des chemins de fichiers, gestion des \_edge cases*.     |
| **Jour 5** | **Augmentation & Génération LLM**    | Intégration de `Qwen/Qwen3-0.6B`, gestion du budget de tokens, `answer` et `answer_dataset`.         |
| **Jour 6** | **Sprint Bonus (3 à 5 points)**      | _Caching_, _Incremental Indexing_, _Local HTTP API_ (+ optionnel: _Embeddings + Hybrid_).            |
| **Jour 7** | **Blindage, Linting & README**       | Passage `flake8` / `mypy --strict`, tests aux limites (`k=0`, query vide), rédaction du `README.md`. |

### Jour 1: Architecture, Modèles Pydantic et ton propre Évaluateur

L'objectif du premier jour est d'avoir une colonne vertébrale propre qui passe déjà `flake8` et `mypy`, et surtout d'avoir ton propre outil de mesure avant même d'indexer le moindre fichier.

- **Initialisation de l'environnement**: Configure ton `pyproject.toml`, `uv.lock`, et ton `Makefile` (`install`, `run`, `debug`, `clean`, `lint`, `lint-strict`). Vérifie que ton `.gitignore` exclut bien `data/`, les poids de modèles et les caches.
- **Modèles de données (`src/models.py`)**: Implémente les 8 classes `pydantic` imposées par le sujet (`MinimalSource`, `UnansweredQuestion`, `AnsweredQuestion`, `RagDataset`, `MinimalSearchResults`, `MinimalAnswer`, `StudentSearchResults`, `StudentSearchResultsAndAnswer`).
- **Squelette CLI (`src/__main__.py`)**: Monte ta classe ou ton dictionnaire de commandes avec `Python Fire` pour que `uv run python -m src <command>` réponde sur les 6 commandes requises.
- **Commande `evaluate` (`src/evaluator.py`)**: Code immédiatement le calcul de l' **IoU (Intersection over Union)** sur les plages de caractères `[first_character_index, last_character_index]` et du **Recall@k** (avec le seuil IoU $\ge 0.05$ sur le même `file_path`). Cela te permettra de comprendre intimement comment la `moulinette` te note.
  MD

### Jour 2: Exploration du Corpus et Double Stratégie de Chunking

Le dossier `data/raw/vllm-0.10.1` contient des milliers de fichiers, mais tous ne sont pas utiles (fichiers binaires, images, caches, tests unitaires géants ou fichiers de configuration obscurs).

- **Audit des datasets publics**: Inspecte `dataset_docs_public.json` et `dataset_code_public.json` dans `AnsweredQuestions`. Regarde quelles extensions de fichiers apparaissent dans les `sources` de référence (`.py`, `.md`, `.rst`, etc.) et quelle est la longueur moyenne des extraits attendus.
  MD+ 1
- **Stratégie 1 — Markdown / Texte**: Utilise un découpage récursif structurel (via `chonkie.RecursiveChunker` ou basé sur les titres `#`, `##`, paragraphes `\n\n`, puis lignes `\n`).
- **Stratégie 2 — Code Python**: Utilise un découpage orienté syntaxe Python (via `chonkie.CodeChunker` ou le module natif `ast` de Python qui te donne les lignes exactes de chaque classe et fonction). Si une fonction ou une classe dépasse `--max_chunk_size` (2000 caractères), redécoupe-la proprement avec un léger _overlap_ (chevauchement de 100 à 200 caractères) pour ne pas couper une logique en deux.
- **Validation vitale du Jour 2**: Écris un test unitaire `pytest` qui vérifie sur tout le corpus qu' **aucun** chunk généré n'a `last_character_index - first_character_index > max_chunk_size` et que `last_character_index > first_character_index >= 0`.
  MD

### Jour 3: Indexation Lexical (bm25s) et Pipeline de Recherche

C'est le jour où tu connectes les tuyaux et où tu lances la `moulinette` pour la première fois.

- **Commande `index`**: Parcourt `data/raw/` avec une barre de progression `tqdm`, applique le bon _chunker_ selon l'extension du fichier, tokenise les chunks, construit la matrice `bm25s` et sauvegarde à la fois l'index `bm25s` et les métadonnées des chunks (`file_path`, `first_character_index`, `last_character_index`) dans `data/processed/`.
- **Commandes `search` et `search_dataset`**: Charge l'index depuis `data/processed/`, récupère les `top-k` chunks pour chaque question et génère le fichier JSON `StudentSearchResults` dans le dossier `--save_directory`.
  MD
- **Premier crash-test avec la `moulinette`**: Lance le binaire `moulinette evaluate_student_search_results` sur tes sorties Docs et Code. Tu auras ton score de référence (_baseline_).
  MD+ 1

### Jour 4: Chasse au Recall@5 (Objectif > 85 % Docs et > 60 % Code)

Si ton premier jet du Jour 3 est un peu juste sur le code (le seuil est à 50 % sur Code et 80 % sur Docs), voici les **3 leviers magiques** à implémenter dans ton pré-processeur de texte avant de donner les mots à `bm25s`:

1. **L'éclatement des identifiants (Identifier Splitting)**: Dans ta fonction de tokenisation, si tu rencontres `AsyncLLMEngine` ou `max_model_len`, génère à la fois le token original (`asyncllmengine`, `max_model_len`) **et** les sous-mots éclatés (`async`, `llm`, `engine`, `max`, `model`, `len`). Ainsi, que la question cite le nom de la variable exacte ou paraphrase en anglais, `bm25s` matche!
2. **L'injection du `file_path` dans les tokens du chunk**: Un chunk de 1500 caractères situé au milieu de `data/raw/vllm-0.10.1/docs/features/lora.md` ne contient parfois même pas le mot `"lora"` (il dit juste _"This adapter allows..."_). Si tu préfixes virtuellement les tokens de chaque chunk par les mots de son chemin de fichier (`docs`, `features`, `lora`), ton Recall@5 va bondir de 10 à 15 % instantanément.
3. **La déduplication / diversification spatiale (NMS)**: Parfois, deux chunks qui se chevauchent dans le même fichier trustent les places #1 et #2 du `top-5`, te faisant perdre une place précieuse sur les 5 cartouches disponibles. Si deux chunks du même fichier se chevauchent déjà fortement, fusionne-les (tant que l'union reste $\le 2000$ caractères) ou écarte le doublon!
   MD

### Jour 5: Augmentation et Génération avec Qwen/Qwen3-0.6B

Une fois que ton _Retrieval_ détruit les scores de la `moulinette`, tu passes aux étapes 3 et 4 du RAG.

- **Reconstruction du contexte (Augmenting)**: Tes `MinimalSource` ne stockent que `file_path`, `first_character_index` et `last_character_index`. Au moment de générer la réponse, tu vas relire dans `file_path` exactement la tranche `content[first:last]` pour les meilleurs résultats récupérés.
- **Gestion du budget de tokens**: Concatène les extraits les plus pertinents (par exemple le top-3 ou top-5) en vérifiant avec le tokenizer de Qwen que ton prompt complet (System Prompt + Extraits + Question) reste bien dans une fenêtre raisonnable (ex. 2048 ou 4096 tokens max) pour que l'inférence sur CPU reste rapide.
- **Prompting anti-hallucination (Grounding)**: Rédige un prompt système clair demandant à `Qwen/Qwen3-0.6B` de répondre **uniquement** à partir des extraits de code/documentation fournis, de manière concise et directe (pense à désactiver le mode _thinking_ de Qwen3 avec `enable_thinking=False` dans le `apply_chat_template` si tu veux éviter qu'il génère 500 tokens de réflexion `<think>...</think>` sur CPU!).
- **Commandes `answer` et `answer_dataset`**: Sérialise le tout proprement avec `StudentSearchResultsAndAnswer`.

### Jour 6: Le Sprint Bonus (3 à 5 Bonus accessibles en une journée)

Ne touche aux bonus que lorsque la partie obligatoire est validée à 100 %. Sur ce sujet, les bonus s'intègrent très naturellement:

1. **Bonus 4 — Caching ( env. 45 min | +1 pt)**:
   - Mets en cache l'index en mémoire lorsqu'il est chargé, et ajoute un cache persistant sur disque (dans `data/processed/cache.json` ou via `sqlite3` / `diskcache`) qui associe `(hash_de_la_question, k)` aux résultats `MinimalSearchResults` et `MinimalAnswer`. Lors de la soutenance, relancer deux fois la même requête passera de plusieurs secondes à `0.001s`.
2. **Bonus 3 — Incremental Indexing (env. 1h15 | +1 pt)**:
   - Lors de la commande `index`, sauvegarde dans `data/processed/manifest.json` le hash SHA-256 (ou le `mtime` + taille) de chaque fichier indexé. Si on relance `index`, compare les fichiers du disque avec le manifeste: ne re-découpe que les fichiers modifiés ou nouveaux, supprime les chunks des fichiers effacés, et reconstruis la matrice `bm25s`.
3. **Bonus 5 — Local HTTP API (env. 1h | +1 pt)**:
   - Avec `FastAPI` et `uvicorn`, crée un petit fichier `src/api.py` exposant `GET /health`, `POST /search` et `POST /answer` qui utilisent directement tes modèles `pydantic` et ton pipeline en mémoire. Ajoute une commande CLI `serve --port 8000` dans ton interface `Fire` pour le démontrer facilement en soutenance.
4. **Bonus 1 & 2 — Semantic Embeddings + Hybrid Retrieval (env. 2h30 | +2 pts)**:
   - Si tu as encore de l'énergie l'après-midi du Jour 6: utilise `fastembed` (basé sur ONNX Runtime, beaucoup plus rapide sur un CPU de campus que PyTorch) ou `sentence-transformers` avec `all-MiniLM-L6-v2` pour calculer les vecteurs de tes chunks, et combine le classement `bm25s` avec le classement cosinus via la formule **RRF (Reciprocal Rank Fusion)** qui tient en 10 lignes de Python:
     $$
     \text{Score}_{\text{RRF}}(d) = \frac{1}{60 + \text{rang}_{\text{BM25}}(d)} + \frac{1}{60 + \text{rang}_{\text{Vecteur}}(d)}
     $$

### Jour 7: Blindage "Soutenance", Linting et README

En soutenance, le correcteur va pousser ton outil dans ses retranchements avant même de regarder tes bonus:

- **Chasse aux crashs (Edge cases)**: Teste systématiquement ton CLI avec une query vide `""`, une query contenant uniquement des caractères spéciaux `"!@#$%^&*()"`, `--k 0`, `--k -5`, `--max_chunk_size 0`, un `--dataset_path` qui n'existe pas, ou un fichier JSON corrompu. Ton programme doit afficher un message d'erreur propre et ne **jamais** lever de _traceback_ Python brut.
- **Qualité du code**: Passe `make lint` et `make lint-strict` (`flake8` + `mypy --strict`) sur tout `src/` et `tests/`.
- **Rédaction du `README.md` (en anglais)**: Rédige un README complet respectant la première ligne obligatoire (_This project has been created as part of the 42 curriculum by marberge_) et détaillant ton architecture, tes deux stratégies de chunking, ton algorithme de recherche, l'analyse de tes scores `recall@k` selon la taille des chunks, tes choix de design et les défis rencontrés.
