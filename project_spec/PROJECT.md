**RAG against the machine** consiste à concevoir de A à Z un moteur de _Retrieval-Augmented Generation_ capable d'ingérer les quelque 300 000 lignes du dépôt `vllm-0.10.1`, d'en extraire les extraits exacts répondant à une question technique, puis de faire rédiger une réponse sourcée par un modèle local (`Qwen/Qwen3-0.6B`).

## 1\. Le Pipeline en 4 Étapes

Le sujet découpe l'architecture en quatre briques séquentielles où chaque étape nourrit la suivante:

1. **Indexing (Indexation)**: Lecture des fichiers pertinents dans `data/raw/vllm-0.10.1`, découpage (_chunking_) selon **deux stratégies distinctes obligatoires** (une pour le code Python, une pour le Markdown/texte), puis création et sauvegarde sur disque (`data/processed/`) d'un index lexical basé au choix sur **TF-IDF** ou **BM25**.
2. **Retrieving (Recherche)**: Pour une question donnée (seule ou en batch depuis les fichiers JSON), interrogation de l'index en quelques millisecondes pour retourner les `top-k` extraits les plus pertinents, identifiés par leur chemin (`file_path`) et leurs positions de début et de fin en caractères (`first_character_index`, `last_character_index`).
   MD
3. **Augmenting (Augmentation)**: Filtrage et assemblage des extraits récupérés pour construire un prompt adapté à la fenêtre de contexte du modèle sans dépasser son budget de tokens.
4. **Generating (Génération)**: Appel au modèle local **Qwen/Qwen3-0.6B** pour produire une réponse cohérente, fidèle aux sources (_grounded_) et sans hallucination, sérialisée en JSON.
   MD

## 2\. Les Règles du Jeu et Critères de Validation

La validation repose sur des seuils de performance stricts mesurés par le binaire `moulinette` ainsi que sur la rigueur de ton environnement Python:

| Critère                   | Exigence / Seuil          | Détail crucial                                         |
| ------------------------- | ------------------------- | ------------------------------------------------------ |
| **Recall@5 — Docs**       | **\>= 80 %** MD           | Évalué sur `dataset_docs_public.json`. MD              |
| **Recall@5 — Code**       | **\>= 50 %** MD           | Évalué sur `dataset_code_public.json`. MD              |
| **Temps d'indexation**    | **<= 5 minutes**          | Pour l'intégralité du corpus `vllm-0.10.1`.            |
| **Vitesse de recherche**  | **<= 90 secondes**        | Pour traiter un lot de 200 questions.                  |
| **Taille max d'un chunk** | **<= 2000 caractères** MD | Paramétrable via `--max_chunk_size` (défaut: 2000). MD |

**Les 3 pièges éliminatoires à anticiper dès le jour 1:**

- **La limite stricte des 2000 caractères**: La moulinette vérifie que chaque source retournée respecte `max_context_length = 2000`. Si **un seul** extrait dans ton JSON dépasse 2000 caractères (`last_character_index - first_character_index > 2000`), toute ta sortie est rejetée comme invalide.
  MD+ 1
- **La correspondance exacte du `file_path`**: La moulinette compare les chemins caractère par caractère avec le dataset de référence (`data/raw/vllm-0.10.1/...`). Un `./` en trop ou un chemin absolu fera tomber ton score à 0 %.
  MD
- **La robustesse aux cas limites (Edge cases)**: En soutenance, ton programme sera testé avec des requêtes vides, absurdes, `k=0`, des fichiers manquants ou des JSON malformés. Aucun crash ni _traceback_ non géré n'est toléré.

## 3\. Comprendre la Métrique: Recall@k et IoU (0.05)

La moulinette évalue uniquement la partie recherche via la commande `evaluate_student_search_results` (la commande `evaluate_student_answers` n'est pas implémentée dans le binaire fourni).

Pour chaque question du dataset `AnsweredQuestions`, il existe une ou plusieurs sources de référence (_ground truth_) définies par un fichier et une plage `[start, end]`. Une source de référence est considérée comme **trouvée** dans ton `top-k` si l'un de tes chunks:

1. Pointe exactement sur le même `file_path`.
   MD
2. Chevauche la plage de caractères de référence avec un **IoU (_Intersection over Union_) d'au moins 0.05 (5 %)**:
   $$
   \text{IoU} = \frac{\text{Longueur de l'intersection}}{\text{Longueur de l'union}} \ge 0.05
   $$

> **L'astuce à retenir:** Un IoU de 0.05 est très permissif! Si la réponse officielle fait 200 caractères dans un fichier et que tu retournes un chunk de 1800 caractères qui englobe ne serait-ce que la moitié de cette zone, ton IoU sera supérieur à 0.05 et le point est validé. En revanche, le vrai défi sur le code source (50 % requis) vient du vocabulaire: une question en anglais naturel utilise rarement les mêmes mots qu'un identifiant Python en `snake_case` ou `camelCase`. La façon dont tu vas **tokeniser** les noms de fonctions, les classes et les docstrings lors de l'indexation fera toute la différence.

## 4\. Architecture, Outils Obligatoires et CLI

Ton dépôt Git (qui ne doit contenir **aucune** donnée lourde, ni corpus, ni poids de modèles, ni fichiers générés) doit respecter ton standard habituel à 42 enrichi de quelques outils imposés:

- **Environnement & Qualité**: Python 3.10+, gestionnaire `uv` obligatoire (`pyproject.toml` et `uv.lock` à la racine), `Makefile` classique (`install`, `run`, `debug`, `clean`, `lint`, `lint-strict`), respect de `flake8`, typage statique validé par `mypy`, et docstrings PEP 257.
  MD
- **Modèles de données (`pydantic`)**: Tous les échanges entre les étapes et les sorties JSON doivent passer par les modèles Pydantic imposés (`MinimalSource`, `UnansweredQuestion`, `AnsweredQuestion`, `RagDataset`, `MinimalSearchResults`, `MinimalAnswer`, `StudentSearchResults`, `StudentSearchResultsAndAnswer`).
  MD
- **Interface en ligne de commande (`Python Fire` + `tqdm`)**: Ton module `src/` doit être appelable via `uv run python -m src <command>` et exposer 6 commandes sans aucun chemin codé en dur:
  1.  `index --max_chunk_size <int>`: Découpe `data/raw/` et sauvegarde l'index dans `data/processed/`. 2. `search <query> --k <int>`: Recherche interactive pour une seule question. 3. `search_dataset --dataset_path <path> --k <int> --save_directory <dir>`: Recherche en batch produisant le JSON `StudentSearchResults` (en séparant bien les sous-dossiers `UnansweredQuestions` et `AnsweredQuestions` pour ne pas écraser tes fichiers).
      MD 4. `answer <query> --k <int>`: Recherche + génération de réponse pour une question unique. 5. `answer_dataset --student_search_results_path <path> --save_directory <dir>`: Génération des réponses en batch à partir d'un fichier de résultats de recherche. 6. `evaluate --student_search_results_path <path> --dataset_path <path>`: Ton propre calculateur de `recall@k` pour tester tes idées rapidement (interdiction d'appeler ou d'importer la `moulinette` dans ton code).

## 5\. Les 5 Bonus (Une fois le Mandatory à 100 %)

Chaque bonus vaut 1 point et doit tourner sur une machine campus **CPU uniquement**:

1. **Semantic embeddings**: Ajout d'un index vectoriel léger sur CPU (ex. `all-MiniLM-L6-v2`) en parallèle de l'index lexical.
2. **Hybrid retrieval**: Fusion des classements lexical et sémantique (par exemple via _Reciprocal Rank Fusion_).
3. **Incremental indexing**: Ré-indexation uniquement des fichiers modifiés plutôt que de tout le corpus.
4. **Caching**: Mise en cache de l'index et des résultats de requêtes pour accélérer les démarrages à froid et les requêtes répétées.
5. **Local HTTP API**: Exposition de la recherche et de la génération via une petite API HTTP locale.

---

Le premier détail critique à corriger dans ton template concerne `src/main.py`: le sujet impose que toutes les commandes soient lancées via `uv run python -m src <command> [options]`. En Python, lorsqu'on exécute un dossier comme un module avec le drapeau `-m src`, l'interpréteur cherche exclusivement un fichier nommé **`src/__main__.py`**. Si tu gardes uniquement `src/main.py`, la commande du correcteur plantera immédiatement.

Voici une architecture modulaire pensée pour séparer chaque étape du pipeline RAG (Indexation, Recherche, Génération, Évaluation) tout en préparant le terrain pour les bonus sans polluer le code obligatoire:

```
├── Makefile
├── pyproject.toml
├── uv.lock
├── README.md
├── data/                          # Ignoré par Git (.gitignore)
│   ├── raw/vllm-0.10.1/           # Le corpus source (lecture seule)
│   ├── datasets/                  # AnsweredQuestions/ et UnansweredQuestions/
│   ├── processed/                 # Ton index sauvegardé, métadonnées et cache
│   └── output/                    # Les JSON générés pour la moulinette
├── src/
│   ├── __init__.py
│   ├── __main__.py                # Point d'entrée obligatoire pour "python -m src"
│   ├── cli.py                     # Interface Python Fire (les 6 commandes CLI)
│   ├── config.py                  # Chemins par défaut, constantes (max_chunk_size=2000, etc.)
│   ├── models.py                  # Les 8 modèles Pydantic du sujet + modèles internes
│   ├── pipeline.py                # Orchestrateur central (appelé par cli.py et api.py)
│   │
│   ├── indexing/                  # ÉTAPE 1 : Ingestion, Chunking et Indexation
│   │   ├── __init__.py
│   │   ├── corpus_loader.py       # Parcours de data/raw/, filtrage des fichiers utiles
│   │   ├── chunkers.py            # Stratégie 1 (Code Python) & Stratégie 2 (Markdown/Docs)
│   │   ├── tokenizer.py           # Pré-processing : split CamelCase/snake_case + file_path
│   │   └── indexer.py             # Construction/sauvegarde de l'index bm25s (+ incrémental)
│   │
│   ├── retrieval/                 # ÉTAPE 2 : Recherche des meilleurs extraits
│   │   ├── __init__.py
│   │   ├── lexical_retriever.py   # Recherche rapide via bm25s
│   │   └── hybrid_retriever.py    # (Bonus 1 & 2) Embeddings + fusion RRF
│   │
│   ├── generation/                # ÉTAPES 3 & 4 : Augmentation et Génération LLM
│   │   ├── __init__.py
│   │   ├── context_builder.py     # Lecture des tranches [first:last] et budget de tokens
│   │   └── llm_generator.py       # Chargement de Qwen/Qwen3-0.6B et génération
│   │
│   ├── evaluation/                # ÉTAPE 5 : Auto-évaluation
│   │   ├── __init__.py
│   │   └── evaluator.py           # Calcul de l'IoU (>= 0.05) et des scores Recall@k
│   │
│   └── utils/                     # Briques transversales
│       ├── __init__.py
│       ├── io_helpers.py          # Lecture/écriture JSON sécurisée (try/except)
│       ├── cache.py               # (Bonus 4) Cache mémoire et disque des requêtes
│       └── api.py                 # (Bonus 5) Serveur HTTP local (FastAPI)
│
└── tests/
    ├── test_chunkers.py           # Vérifie qu'aucun chunk ne dépasse 2000 caractères
    ├── test_evaluator.py          # Teste le calcul d'IoU et de Recall@k
    ├── test_tokenizer.py          # Teste le découpage des noms de fonctions/variables
    └── test_edge_cases.py         # Teste k=0, query vide, JSON malformé, fichiers absents
```

## Pourquoi ce découpage va te faire gagner un temps précieux

### 1\. La racine de src/: Le chef d'orchestre et les contrats

- **`models.py`**: C'est la fondation du projet. Tu y places les 8 classes Pydantic imposées par le sujet (`MinimalSource`, `UnansweredQuestion`, `AnsweredQuestion`, `RagDataset`, `MinimalSearchResults`, `MinimalAnswer`, `StudentSearchResults`, `StudentSearchResultsAndAnswer`). Tu peux aussi y ajouter un modèle interne `Chunk` (qui étend `MinimalSource` ou l'utilise) pour transporter le texte pendant l'indexation.
- **`cli.py` vs `pipeline.py` (L'astuce d'architecture)**: Ne mets jamais la logique métier directement dans les fonctions exposées à `Python Fire`. `cli.py` ne doit faire que trois choses: recevoir les arguments de la ligne de commande, valider les entrées tordues (comme `k <= 0` ou une question vide) avec un `try/except` propre pour éviter tout crash en soutenance, et appeler `pipeline.py`.
  - _Pourquoi c'est génial?_ Quand tu voudras faire le **Bonus 5 (Local HTTP API)** dans `utils/api.py`, tes routes HTTP appelleront exactement les mêmes méthodes de `pipeline.py` en 15 lignes de code.

### 2\. Le dossier src/indexing/: Séparer la découpe du moteur mathématique

Le sujet exige deux stratégies de découpage distinctes (code Python et Markdown/texte) et un index lexical sauvegardé dans `data/processed/`.

- **`corpus_loader.py`**: S'occupe uniquement de parcourir `data/raw/vllm-0.10.1`, d'ignorer le bruit (fichiers binaires, images, `__pycache__`, fichiers `.git`) et de conserver le chemin relatif exact (`data/raw/vllm-0.10.1/...`) exigé par la moulinette. C'est aussi ici que tu pourras glisser le calcul de hash de fichier pour le **Bonus 3 (Incremental indexing)**.
  MD
- **`chunkers.py`**: Reçoit le texte brut d'un fichier et son chemin, choisit la stratégie selon l'extension (`.py` vs `.md` /`.rst` /`.txt`), et garantit mathématiquement que chaque morceau respecte `last_character_index - first_character_index <= max_chunk_size`.
  MD
- **`tokenizer.py`**: C'est ton arme secrète pour le score de **Recall@5 sur le code**. En isolant la tokenisation dans son propre fichier, tu pourras facilement tester et affiner le découpage du `snake_case`, du `CamelCase` et l'injection des mots du `file_path` sans toucher à `bm25s`.
  MD
- **`indexer.py`**: Prend la liste des chunks, appelle `tokenizer.py`, construit l'objet `bm25s` et sauvegarde le tout dans `data/processed/` avec les barres de progression `tqdm`.

### 3\. Le dossier src/retrieval/: Prêt pour l'hybride

- **`lexical_retriever.py`**: Charge l'index depuis `data/processed/` (en le gardant en mémoire pour éviter de le recharger à chaque question d'un batch de 200 questions!) et renvoie une liste de `MinimalSource`.
- **`hybrid_retriever.py`**: Reste vide (ou n'est créé) qu'au Jour 6 si tu fais les **Bonus 1 & 2**. Ton code obligatoire reste 100 % propre et indépendant.

### 4\. Le dossier src/generation/: Économiser la RAM et le CPU

- **`context_builder.py`**: Comme ton index ne stocke que les coordonnées `(file_path, first_character_index, last_character_index)`, ce module ouvre uniquement les `k` fichiers gagnants, extrait la sous-chaîne exacte `texte[first:last]`, et assemble le prompt sans dépasser la fenêtre de tokens.
  MD
- **`llm_generator.py`**: Isoler `Qwen/Qwen3-0.6B` dans ce fichier a un avantage énorme: **le chargement paresseux (_lazy loading_)**. Tu n'importeras et ne chargeras le modèle en mémoire que lorsque les commandes `answer` ou `answer_dataset` sont appelées. Ainsi, tes commandes `index`, `search`, `search_dataset` et `evaluate` démarreront en un éclair sans charger 1,2 Go de poids PyTorch en RAM!

### 5\. Le dossier src/evaluation/: Ton miroir de la moulinette

- **`evaluator.py`**: Contient deux fonctions pures très simples à tester unitairement: `compute_iou(span_a, span_b)` et `compute_recall_at_k(student_results, ground_truth, k)`.
