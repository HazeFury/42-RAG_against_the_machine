# Journal de Bord — Jour 1 : Fondations, Modèles Pydantic et Évaluateur

## 1. Objectifs de la Journée

L'objectif du premier jour était d'établir l'ossature technique complète du projet conformément aux spécifications du sujet 42 (« RAG against the machine » v2.0) :
- Configurer un environnement Python 3.10+ propre, automatisé par `uv` et validé par un `Makefile` strict.
- Mettre en place les 8 modèles de données Pydantic obligatoires pour assurer l'intégrité et la validation des échanges de données.
- Construire le moteur d'évaluation indépendant (`evaluator.py`) répliquant fidèlement la notation de la moulinette officielle (calcul d'IoU $\ge 0.05$ et Recall@k).
- Mettre en place l'architecture modulaire du package `src/` avec son point d'entrée exécutable `__main__.py` et l'interface en ligne de commande `Python Fire` (`src/cli.py`).
- Mettre en place la suite de tests unitaires et garantir une conformité totale à `flake8` et `mypy --strict`.

---

## 2. Nettoyage et Configuration de l'Environnement

### 2.1 Suppression des artefacts du template initial
- Suppression des fichiers modèles non pertinents issus du template standard (`src/main.py`, `src/core/`, `src/utils/math_helpers.py`, `tests/test_math_helpers.py`).

### 2.2 Fichier `.gitignore`
- Exclusion de tous les fichiers lourds et temporaires :
  - `data/` (le corpus brut `data/raw/`, les datasets et les fichiers générés ne doivent jamais être commités).
  - Caches Python et linters (`__pycache__/`, `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`).
  - Environnements virtuels (`.venv/`).
  - Poids de modèles (`*.bin`, `*.model`, etc.).

### 2.3 Fichier `pyproject.toml`
- Configuration pour Python `>=3.10`.
- Déclaration des dépendances de production :
  - `pydantic>=2.0.0` : Validation des schémas de données.
  - `bm25s>=0.2.0` : Indexation et recherche lexicale ultra-rapide.
  - `fire>=0.6.0` : Interface CLI.
  - `tqdm>=4.66.0` : Barres de progression pour les opérations longues.
  - `transformers>=4.40.0` & `torch>=2.0.0` : Génération LLM avec `Qwen/Qwen3-0.6B` sur CPU.
  - `fastapi>=0.110.0` & `uvicorn>=0.28.0` : Support de l'API HTTP locale (Bonus 5).
- Déclaration des outils de test et de typage dans `[dependency-groups].dev` (`flake8`, `mypy`, `pytest`).
- Configuration stricte de mypy : `strict = true`, `disallow_untyped_defs = true`, etc.

### 2.4 Fichier `Makefile`
Conformité stricte avec le chapitre V.2 du sujet :
- `install` : Synchronise les dépendances via `uv sync`.
- `run` : Exécute le point d'entrée CLI via `uv run python -m src --help`.
- `debug` : Lance le débogueur interactif standard `uv run python -m pdb -m src`.
- `clean` / `fclean` / `re` : Nettoyage des caches et du venv.
- `lint` : Lance `flake8 .` et `mypy .` avec les drapeaux requis.
- `lint-strict` : Lance `flake8 .` et `mypy . --strict`.
- `test` : Lance la suite de tests unitaires via `pytest`.

### 2.5 Fichier `.flake8`
- Paramétrage de la longueur maximale des lignes à 79 caractères (standard PEP 8 / 42).
- Exclusion des répertoires temporaires (`.venv`, `__pycache__`, `data`, etc.).

---

## 3. Architecture Modulaire (`src/`)

L'arborescence cible a été créée afin de séparer clairement chaque responsabilité du pipeline :

```text
├── Makefile
├── pyproject.toml
├── .flake8
├── .gitignore
├── documentation/
│   └── DAY1.md
├── src/
│   ├── __init__.py
│   ├── __main__.py          # Point d'entrée pour uv run python -m src
│   ├── cli.py               # Interface Python Fire et validation des arguments
│   ├── config.py            # Constantes globales et chemins par défaut
│   ├── models.py            # Modèles Pydantic imposés par le sujet
│   ├── pipeline.py          # Orchestrateur central (liant CLI et logique métier)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py     # Calcul d'IoU (>= 0.05) et Recall@k
│   ├── indexing/
│   │   └── __init__.py      # Réservé pour le corpus loader, chunkers et indexer (Jour 2/3)
│   ├── retrieval/
│   │   └── __init__.py      # Réservé pour le BM25 retriever (Jour 3/4)
│   ├── generation/
│   │   └── __init__.py      # Réservé pour le contexte et Qwen3-0.6B (Jour 5)
│   └── utils/
│       ├── __init__.py
│       └── io_helpers.py    # Gestion sécurisée des I/O JSON et normalisation des chemins
└── tests/
    ├── test_cli_edge_cases.py
    ├── test_evaluator.py
    └── test_models.py
```

---

## 4. Détail des Modules Développés

### 4.1 Modèles de Données Pydantic (`src/models.py`)
Conformément à la section VI.4 du sujet, les 8 modèles suivants ont été implémentés :
1. `MinimalSource` : représente un extrait avec `file_path`, `first_character_index`, `last_character_index` (plus une propriété `span_length`).
2. `UnansweredQuestion` : question non répondue avec génération automatique d'UUID.
3. `AnsweredQuestion` : hérite de `UnansweredQuestion` et intègre `sources: list[MinimalSource]` et `answer: str`.
4. `RagDataset` : conteneur de dataset typé `list[Union[AnsweredQuestion, UnansweredQuestion]]`.
5. `MinimalSearchResults` : regroupe `question_id`, `question` et `retrieved_sources`.
6. `MinimalAnswer` : hérite de `MinimalSearchResults` et intègre la réponse textuelle générée.
7. `StudentSearchResults` : structure sérialisée pour la moulinette contenant la liste des `MinimalSearchResults` et `k`.
8. `StudentSearchResultsAndAnswer` : structure finale contenant la liste des `MinimalAnswer` et `k`.

### 4.2 Module d'Évaluation (`src/evaluation/evaluator.py`)
Ce module permet d'évaluer le système de recherche en toute autonomie (sans dépendre du binaire externe) :
- **`compute_iou(start_a, end_a, start_b, end_b)`** : Calcule l'Intersection over Union 1D entre deux intervalles de caractères :
  $$\text{intersection} = \max(0, \min(A_{\text{end}}, B_{\text{end}}) - \max(A_{\text{start}}, B_{\text{start}}))$$
  $$\text{union} = (A_{\text{end}} - A_{\text{start}}) + (B_{\text{end}} - B_{\text{start}}) - \text{intersection}$$
  $$\text{IoU} = \frac{\text{intersection}}{\text{union}}$$
- **`is_matching_source(candidate, ground_truth, min_iou=0.05)`** : Vérifie la stricte identité du chemin (`file_path` normalisé sans `./`) et l'IoU $\ge 0.05$.
- **`calculate_question_recall(retrieved_sources, ground_truth_sources, k, min_iou=0.05)`** : Calcule le taux de sources attendues retrouvées dans le top-$k$.
- **`evaluate_dataset(...)`** : Calcule la moyenne du `Recall@k` pour $k \in \{1, 3, 5, 10\}$ sur toutes les questions du dataset.
- **`format_evaluation_report(...)`** : Produit une sortie formatée identique à celle de la moulinette.

### 4.3 Interface Ligne de Commande & Point d'Entrée (`src/cli.py` & `src/__main__.py`)
- L'exécution se fait via `uv run python -m src <command> [options]`.
- Les 6 commandes imposées par la section VI.6 sont déclarées :
  1. `index` (`--max_chunk_size`, `--corpus_dir`, `--output_dir`)
  2. `search` (`<query>`, `--k`, `--index_dir`)
  3. `search_dataset` (`--dataset_path`, `--k`, `--save_directory`, `--index_dir`)
  4. `answer` (`<query>`, `--k`, `--index_dir`, `--model_name`)
  5. `answer_dataset` (`--student_search_results_path`, `--save_directory`, `--model_name`)
  6. `evaluate` (`--student_search_results_path`, `--dataset_path`)
- **Blindage anti-crash (Edge cases)** : Toutes les entrées dégénérées (requête vide, espaces seuls, $k \le 0$, `max_chunk_size \le 0`, fichiers ou dossiers inexistants, JSON malformé) sont interceptées et gérées avec un message d'erreur explicite sans jamais produire de traceback Python non géré.

### 4.4 Utilitaires transversaux (`src/utils/io_helpers.py` & `src/config.py`)
- Gestionnaire de lecture et d'écriture JSON sécurisée avec gestion de contexte (`with open(...)`) et création automatique des répertoires parents.
- Fonction `normalize_corpus_path` pour éliminer les préfixes `./` ou les séparateurs Windows `\` et garantir la correspondance exacte exigée par la moulinette (`data/raw/vllm-0.10.1/...`).
- Constantes par défaut regroupées dans `config.py` (`DEFAULT_MAX_CHUNK_SIZE = 2000`, `DEFAULT_IOU_THRESHOLD = 0.05`, etc.).

---

## 5. Suite de Tests Unitaires (`tests/`)

Trois modules de test ont été rédigés :
1. **`tests/test_evaluator.py`** :
   - Validation mathématique de l'IoU (intervalles disjoints, identiques, chevauchements partiels, inclusions).
   - Test des limites du seuil de 5% d'IoU.
   - Validation du calcul de `Recall@k` individuel et moyenné au niveau d'un dataset.
2. **`tests/test_models.py`** :
   - Vérification de la validation Pydantic (champs obligatoires, types).
   - Test de la sérialisation / désérialisation JSON sans perte d'information sur les types Union.
   - Propriété `span_length` de `MinimalSource`.
3. **`tests/test_cli_edge_cases.py`** :
   - Test des cas limites sur toutes les commandes CLI (paramètres négatifs, chemins manquants, JSON corrompu).

---

## 6. Validation Qualité et Rigueur

- **Linters** :
  - `flake8 .` : **0 erreur, 0 avertissement** (lignes $\le 79$ caractères, respect de la PEP 8).
  - `mypy . --strict` : **100% de succès** sur l'ensemble des 13 fichiers sources et 3 fichiers de test (aucun type implicite `Any`, signatures complètes).
- **Langue & Conventions** :
  - 100% du code, des docstrings (PEP 257 / Google style), des commentaires et des messages CLI sont rédigés en **anglais**.

---

## 7. Préparation du Jour 2

Le socle est en place et validé. Le **Jour 2** sera consacré à :
1. L'exploration et le filtrage du corpus `data/raw/vllm-0.10.1` (`src/indexing/corpus_loader.py`).
2. L'implémentation des deux stratégies distinctes de chunking (`src/indexing/chunkers.py`) :
   - **Stratégie 1 (Code Python)** : Découpage orienté syntaxe (`ast` ou blocs de fonctions/classes) avec respect strict de la taille maximale ($\le 2000$ caractères).
   - **Stratégie 2 (Markdown / Texte)** : Découpage récursif basé sur la structure des documents (titres `#`, doubles sauts de ligne, paragraphes).
3. L'écriture des tests de validation garantissant qu'aucun chunk ne dépasse 2000 caractères.

