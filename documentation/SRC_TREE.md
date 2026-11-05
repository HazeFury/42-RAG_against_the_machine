# Arborescence et Description du Code Source (`src/`)

Ce document détaille l'organisation complète du code contenu dans le dossier `src/`, le rôle précis de chaque fichier, et la correspondance entre les fichiers et les 6 commandes du projet.

---

## 1. Arborescence Visuelle de `src/`

```text
src/
├── __init__.py                  
├── __main__.py                  # Point d'entrée de la commande 'python -m src' (Fire)
├── cli.py                       # Contrôleur CLI : validation des arguments et affichage
├── config.py                    # Constantes par défaut (tailles, seuils, modèles, chemins)
├── models.py                    # Modèles de données Pydantic imposés par le sujet
├── pipeline.py                  # Orchestrateur central connectant toutes les étapes
│
├── indexing/                    # ÉTAPE 1 : Ingestion, Découpage et Indexation
│   ├── __init__.py              
│   ├── corpus_loader.py         # Parcours récursif, filtrage des fichiers et chemins relatifs
│   ├── chunkers.py              # Double stratégie de découpage (Python AST vs Markdown) <= 2000 car.
│   ├── tokenizer.py             # Tokenizer adapté au code (camelCase, snake_case, injection de chemin)
│   └── indexer.py               # Création, calcul BM25s et persistance sur disque
│
├── retrieval/                   # ÉTAPE 2 : Recherche et Récupération Lexicale
│   ├── __init__.py             
│   └── lexical_retriever.py     # Chargement de l'index BM25 en RAM et déduplication spatiale
│
├── generation/                  # ÉTAPES 3 & 4 : Augmentation de Contexte et Génération LLM
│   ├── __init__.py              
│   ├── context_builder.py       # Extraction des tranches exactes [first:last] et budget de tokens
│   └── llm_generator.py         # Inférence CPU locale (Qwen3-0.6B) en Lazy Loading avec grounding
│
├── evaluation/                  # ÉTAPE 5 : Auto-Évaluation de la Qualité
│   ├── __init__.py              
│   └── evaluator.py             # Calcul mathématique de l'IoU 1D (seuil 0.05) et Recall@k
│
└── utils/                       # Utilitaires Généraux
    ├── __init__.py              
    └── io_helpers.py            # Lecture/Écriture JSON sécurisée, normalisation de chemins et fsync
```

---

## 2. Description Détaillée de Chaque Fichier

### Racine de `src/`

- **`src/__main__.py`** :
  - **Rôle** : Point d'entrée exécuté quand on lance `uv run python -m src <commande>`.
  - **Fonctionnement** : Importe la bibliothèque Python `fire` et la classe `RAGCLI` de `src/cli.py`, puis appelle `fire.Fire(RAGCLI)`. Gère également l'erreur si une dépendance manque.

- **`src/cli.py`** :
  - **Rôle** : Interface en ligne de commande.
  - **Fonctionnement** : Reçoit les arguments du terminal, valide rigoureusement les entrées pour éviter tout crash (vérifie que la requête n'est pas vide, que $k > 0$, que `400 <= max_chunk_size <= 2000`, que les fichiers existent), délègue le travail à `RagPipeline`, et gère l'affichage sur la sortie standard (`stdout`) ou d'erreur (`stderr`).

- **`src/config.py`** :
  - **Rôle** : Fichier centralisant toutes les constantes et configurations du projet.
  - **Fonctionnement** : Définit les valeurs par défaut et les bornes : `MIN_MAX_CHUNK_SIZE = 400`, `MAX_MAX_CHUNK_SIZE = 2000`, `DEFAULT_MAX_CHUNK_SIZE = 2000`, `DEFAULT_TOP_K = 5`, `DEFAULT_DATASET_TOP_K = 10`, `DEFAULT_IOU_THRESHOLD = 0.05`, le nom du modèle `DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"`, et les chemins vers `data/raw`, `data/processed`, etc.

- **`src/models.py`** :
  - **Rôle** : Contrat de données strict imposé par la Section VI.4 du sujet 42.
  - **Fonctionnement** : Implémente avec **Pydantic** les 8 classes de données :
    - `MinimalSource` : Coordonnées d'un extrait (`file_path`, `first_character_index`, `last_character_index`).
    - `Chunk` : Modèle interne associant `MinimalSource` et son texte brut `text`.
    - `UnansweredQuestion` : Question avec son identifiant UUID unique.
    - `AnsweredQuestion` : Question avec ses sources de référence et sa réponse idéale.
    - `RagDataset` : Conteneur pour parser les datasets JSON officiels.
    - `MinimalSearchResults` : Résultat d'une question avec les sources trouvées.
    - `MinimalAnswer` : Résultat d'une question avec les sources et la réponse rédigée.
    - `StudentSearchResults` : Format évalué par la moulinette pour `search_dataset`.
    - `StudentSearchResultsAndAnswer` : Format final pour `answer_dataset`.

- **`src/pipeline.py`** :
  - **Rôle** : L'orchestrateur central (*Chef d'orchestre*).
  - **Fonctionnement** : La classe `RagPipeline` instancie et connecte les modules entre eux. Elle conserve une instance unique en cache du retriever lexical (`_retriever`) et du modèle LLM (`_generator`) pour éviter de recharger les données à chaque requête. Lors des recherches par lots (`search_dataset`), elle détecte automatiquement le domaine (`docs` ou `code` selon le nom du dataset) pour appliquer le multiplicateur de score ciblé.

---

### Sous-dossier `src/indexing/` (Indexation)

- **`src/indexing/corpus_loader.py`** :
  - **Rôle** : Découverte et filtrage des fichiers sources à indexer.
  - **Fonctionnement** : Parcourt récursivement le dossier `data/raw/` avec `os.walk`. Élimine les binaires, images, archives, dossiers cachés (`.git`, `__pycache__`). Ne conserve que le code source (`.py`, `.c`, `.cu`, `.h`, etc.) et la documentation (`.md`, `.rst`, `.txt`). Convertit impérativement chaque chemin en chemin relatif strict (`data/raw/vllm-0.10.1/...`).

- **`src/indexing/chunkers.py`** :
  - **Rôle** : Découpage intelligent du texte sous la limite stricte des 2000 caractères.
  - **Fonctionnement** : Implémente deux stratégies principales et des mécanismes de robustesse :
    - *Code Python* (`chunk_python_code`) : Parse l'arbre syntaxique abstrait (AST) de Python pour découper selon les frontières réelles des fonctions et des classes sans casser la logique.
    - *Markdown / Documentation* (`chunk_markdown_text`) : Découpe hiérarchiquement selon les titres (`#`, `##`), puis les sauts de paragraphe (`\n\n`). Fusionne automatiquement les titres isolés sans corps textuel (*bare headings*) avec la section suivante pour éviter les chunks orphelins et maximiser le Recall.
    - *Découpage par lignes adaptatif* (`slice_range_by_lines`) : Si une fonction ou un bloc dépasse `max_chunk_size`, il est découpé par lignes avec un chevauchement adaptatif (`min(overlap, max_chunk_size // 4)`) garantissant une progression positive stricte sans boucle infinie.

- **`src/indexing/tokenizer.py`** :
  - **Rôle** : Préparation du texte pour le moteur de recherche lexical.
  - **Fonctionnement** :
    - Éclate les identifiants de programmation `snake_case` et `camelCase` tout en conservant le terme complet (ex: `gpu_memory` donne `['gpu_memory', 'gpu', 'memory']`).
    - Supprime les mots vides de la langue anglaise (*stopwords*) tout en conservant les mots-clés techniques.
    - *Path Token Injection* : Extrait les mots du chemin du fichier (`docs`, `features`, `lora`) et les injecte avec un poids renforcé dans les tokens du chunk pour doper le Recall.

- **`src/indexing/indexer.py`** :
  - **Rôle** : Moteur de construction et de sauvegarde de l'index BM25.
  - **Fonctionnement** : Utilise `bm25s` pour calculer la matrice lexicale sur tous les chunks tokenisés, puis sauvegarde les fichiers binaires dans `data/processed/bm25_index/` et le fichier de correspondance dans `data/processed/metadata.json`.

---

### Sous-dossier `src/retrieval/` (Recherche)

- **`src/retrieval/lexical_retriever.py`** :
  - **Rôle** : Moteur de recherche rapide en mémoire.
  - **Fonctionnement** : Charge l'index BM25 et les métadonnées une seule fois en RAM. Tokenise la requête de l'utilisateur, interroge l'index pour extraire un pool élargi de candidats (jusqu'à 50 candidats via `fetch_k = min(max(k * 10, 50), len(metadata))`), applique un multiplicateur de score de domaine (`_compute_multiplier`, boost 2.0x pour docs ou code selon le focus), puis applique une **déduplication spatiale** : si deux extraits proviennent du même fichier et se touchent ou se chevauchent, ils sont fusionnés en un seul (tant que l'union reste $\le 2000$ caractères) pour maximiser la diversité du top-$k$.

---

### Sous-dossier `src/generation/` (Génération LLM)

- **`src/generation/context_builder.py`** :
  - **Rôle** : Reconstruction du texte pour le prompt du LLM.
  - **Fonctionnement** : Prend la liste des `MinimalSource`, ouvre les fichiers sur le disque (avec cache en mémoire), extrait la tranche exacte `texte[first:last]`, et assemble les blocs avec des en-têtes clairs. Utilise le tokenizer du modèle pour garantir que la taille totale ne dépasse pas le budget alloué (2048 tokens). En cas d'aléa de tokenisation, un repli précis intercepte les erreurs pour estimer les tokens sans interruption.

- **`src/generation/llm_generator.py`** :
  - **Rôle** : Inférence du modèle de langage local optimisée matériellement.
  - **Fonctionnement** :
    - *Lazy Loading* : N'importe `torch` et `transformers` que lorsqu'une réponse est demandée.
    - *Optimisation matérielle & bfloat16* : Détection GPU/MPS avec `device_map="auto"`, `torch_dtype=bfloat16` et repli propre sur CPU en bfloat16 avec faible empreinte mémoire (`low_cpu_mem_usage=True`).
    - *Mode d'inférence rapide* : Génération en `torch.inference_mode()`, `use_cache=True` et budget limité à 128 nouveaux tokens pour minimiser la latence.
    - *Gestion flexible des tenseurs* : Transfère proprement les entrées vers `model.device`, compatible à la fois avec `BatchEncoding` Hugging Face et les mocks dictionnaires de tests.
    - *Nettoyage du raisonnement* : Désactive le thinking dans le template (`enable_thinking=False`) et supprime systématiquement toute balise `<think>...</think>` résiduelle via regex multi-lignes.
    - *Grounding strict* : Applique un prompt système de grounding strict demandant de répondre factuellement et de manière concise à partir des extraits fournis.

---

### Sous-dossier `src/evaluation/` (Évaluation)

- **`src/evaluation/evaluator.py`** :
  - **Rôle** : Calcul des métriques officielles d'évaluation.
  - **Fonctionnement** : Compare les sources trouvées avec les sources de vérité terrain. Calcule l'**IoU 1D** ($\frac{\text{Intersection}}{\text{Union}}$) entre les plages de caractères. Si $\text{IoU} \ge 0.05$ (5 %) dans le même fichier, l'extrait est considéré trouvé. Calcule et formate le Recall@1, Recall@3, Recall@5 et Recall@10.

---

### Sous-dossier `src/utils/` (Utilitaires)

- **`src/utils/io_helpers.py`** :
  - **Rôle** : Entrées/Sorties sécurisées sur le système de fichiers.
  - **Fonctionnement** :
    - `normalize_corpus_path()` : Remplace les antislashes `\` par des slashes `/` et retire les `./` initiaux.
    - `safe_read_json()` : Lit et parse un JSON en interceptant les erreurs de décodage.
    - `safe_write_json()` : Crée automatiquement les dossiers parents, écrit le JSON et appelle `os.fsync()` pour forcer l'écriture immédiate sur le disque physique.

---

## 3. Matrice de Correspondance : Commandes vs Fichiers

Voici le tableau récapitulatif montrant exactement **quels fichiers et dossiers sont sollicités lors de chaque commande** :

| Commande CLI | Point d'entrée | Orchestrateur | Modules métiers appelés | Données produites / modifiées |
| :--- | :--- | :--- | :--- | :--- |
| **`index`** | `cli.py:index` | `pipeline.py:index` | `src/indexing/corpus_loader.py`<br>`src/indexing/chunkers.py`<br>`src/indexing/tokenizer.py`<br>`src/indexing/indexer.py` | `data/processed/bm25_index/`<br>`data/processed/metadata.json`<br>`data/processed/index_info.json` |
| **`search`** | `cli.py:search` | `pipeline.py:search` | `src/retrieval/lexical_retriever.py`<br>`src/indexing/tokenizer.py` | Affichage terminal (`stdout`) |
| **`search_dataset`** | `cli.py:search_dataset` | `pipeline.py:search_dataset` | `src/retrieval/lexical_retriever.py`<br>`src/utils/io_helpers.py`<br>`src/models.py` | Fichier JSON dans `data/output/search_results/<Scope>/...` (`StudentSearchResults`) |
| **`answer`** | `cli.py:answer` | `pipeline.py:answer` | `src/retrieval/lexical_retriever.py`<br>`src/generation/context_builder.py`<br>`src/generation/llm_generator.py` | Affichage terminal (`stdout`) |
| **`answer_dataset`** | `cli.py:answer_dataset` | `pipeline.py:answer_dataset` | `src/generation/context_builder.py`<br>`src/generation/llm_generator.py`<br>`src/utils/io_helpers.py`<br>`src/models.py` | Fichier JSON dans `data/output/search_results_and_answer/<Scope>/...` (`StudentSearchResultsAndAnswer`) |
| **`evaluate`** | `cli.py:evaluate` | `pipeline.py:evaluate` | `src/evaluation/evaluator.py`<br>`src/models.py`<br>`src/utils/io_helpers.py` | Affichage terminal du rapport Recall@k |

