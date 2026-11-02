# Journal de Bord — Jour 3 : Indexation Lexicale BM25, Moteur de Recherche et Validation des Seuils

## 1. Objectifs de la Journée

L'objectif du troisième jour était de construire et connecter le moteur complet d'indexation et de recherche lexicale :
- **Moteur d'indexation BM25** (`src/indexing/indexer.py`) : ingestion du corpus, chunking, tokenisation avec barres de progression `tqdm`, création et persistance sur disque de la matrice `bm25s` et des métadonnées des chunks (`data/processed/metadata.json`).
- **Moteur de recherche lexicale** (`src/retrieval/lexical_retriever.py`) : chargement en mémoire une seule fois (*lazy singleton*), tokenisation des requêtes, recherche vectorielle rapide et déduplication spatiale / fusion des extraits adjacents.
- **Raccordement de la CLI et du Pipeline** (`src/pipeline.py`, `src/cli.py`, `src/__main__.py`) : exposition fonctionnelle des commandes `index`, `search` et `search_dataset`.
- **Suite de tests unitaires et d'intégration** (`tests/test_indexer.py`, `tests/test_retriever.py`).
- **Évaluation réelle de bout en bout** sur les deux jeux de données de référence (`dataset_docs_public.json` et `dataset_code_public.json`).

---

## 2. Moteur d'Indexation (`src/indexing/indexer.py`)

### 2.1 Fonctionnement & Architecture
La classe `IndexBuilder` orchestre le pipeline d'indexation complet :
1. **Parcours du corpus** via `load_corpus` (découverte des 2 105 fichiers utiles dans `data/raw/vllm-0.10.1`).
2. **Découpage (*Chunking*)** via `chunk_file` (stratégie AST pour le code Python, stratégie hiérarchique par titres pour Markdown/texte).
3. **Tokenisation** via `tokenize` avec éclatement `camelCase` / `snake_case` et injection pondérée des mots-clés du chemin (`path_weight=2`).
4. **Modèle BM25 & Persistance** :
   - Indexation via `bm25s.BM25()`.
   - Sauvegarde de la matrice BM25 dans `data/processed/bm25_index/` (`params.index.json`, `vocab.index.json`, `*.csc.index.npy`).
   - Sauvegarde des coordonnées des chunks dans `data/processed/metadata.json` (`file_path`, `first_character_index`, `last_character_index`).
   - Sauvegarde du résumé dans `data/processed/index_info.json`.

### 2.2 Performances d'Indexation
- **Nombre de fichiers traités** : 2 105 fichiers.
- **Nombre total de chunks indexés** : 47 221 chunks.
- **Durée totale d'ingestion & indexation** : **8.27 secondes** (seuil éliminatoire du sujet : $\le$ 5 minutes).

---

## 3. Moteur de Recherche Lexicale (`src/retrieval/lexical_retriever.py`)

### 3.1 Chargement en Mémoire Unique (*In-Memory Cache*)
Pour respecter la contrainte de débit ($200\text{ questions en } \le 90\text{ secondes}$), la classe `LexicalRetriever` charge la matrice BM25 et les métadonnées une seule fois à l'initialisation. Aucun accès disque redondant n'est effectué pendant les recherches par lot.

### 3.2 Déduplication Spatiale et Fusion de Chunks (*Spatial Merging*)
Pour éviter de gaspiller les $k$ places disponibles du classement sur des extraits redondants du même fichier :
- Le retriever extrait d'abord un lot élargi de candidats ($k \times 3$).
- Si deux chunks candidats proviennent du **même fichier** et se chevauchent :
  - Si leur union $\le 2000$ caractères, ils sont **fusionnés** en un seul extrait continu `[min(start), max(end)]`.
  - Si leur union dépasse 2000 caractères, le doublon est écarté afin de garantir la diversité des $k$ meilleurs résultats.
- Les résultats finaux sont tronqués strictement à $k$.
- **Garantie absolue** : chaque `MinimalSource` retournée respecte rigoureusement `last_character_index - first_character_index <= 2000`.

### 3.3 Traitement par Lot (*Batch Retrieval*)
La méthode `retrieve_batch` traite l'ensemble des questions d'un dataset en s'appuyant sur l'implémentation vectorisée native de `bm25s`, offrant un débit supérieur à **400 questions par seconde**.

---

## 4. Intégration CLI & Pipeline (`src/cli.py` & `src/pipeline.py`)

Les commandes CLI sont désormais pleinement opérationnelles :
- `uv run python -m src index [--max_chunk_size 2000] [--corpus_dir data/raw] [--output_dir data/processed]` : Ingestion avec barres `tqdm`.
- `uv run python -m src search "<query>" [--k 5]` : Recherche interactive affichant les chemins et coordonnées :
  ```text
  data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py [4624:4742]
  data/raw/vllm-0.10.1/docs/design/arch_overview.md [1474:2157]
  data/raw/vllm-0.10.1/tests/utils.py [2089:2197]
  ```
- `uv run python -m src search_dataset --dataset_path <path> --k 10 --save_directory <dir>` : Recherche en batch générant le JSON `StudentSearchResults` en préservant le nom du fichier d'origine.
- `uv run python -m src evaluate --student_search_results_path <path> --dataset_path <path>` : Calculateur de `Recall@k` avec IoU $\ge 0.05$.

---

## 5. Résultats et Validation des Seuils du Sujet

L'évaluation a été exécutée sur l'intégralité des datasets de référence `AnsweredQuestions` :

### 5.1 Dataset Documentation (`dataset_docs_public.json` — 100 questions)
- **Recall@1** : 0.630 (63.0%)
- **Recall@3** : 0.780 (78.0%)
- **Recall@5** : **0.850 (85.0%)** $\rightarrow$ **VALIDÉ** (Seuil requis : $\ge 80\%$)
- **Recall@10** : 0.900 (90.0%)
- **Temps de recherche pour 100 questions** : **0.23 seconde** (426 questions/s).

### 5.2 Dataset Code Source (`dataset_code_public.json` — 99 questions)
- **Recall@1** : 0.475 (47.5%)
- **Recall@3** : 0.687 (68.7%)
- **Recall@5** : **0.768 (76.8%)** $\rightarrow$ **VALIDÉ** (Seuil requis : $\ge 50\%$, dépassé de **+26.8%**)
- **Recall@10** : 0.859 (85.9%)
- **Temps de recherche pour 99 questions** : **0.21 seconde** (464 questions/s).

### 5.3 Tableau Récapitulatif des Exigences 42

| Critère officiel | Exigence sujet | Résultat obtenu | Statut |
| :--- | :--- | :--- | :--- |
| **Recall@5 — Docs** | $\ge 80\%$ | **85.0%** | **Validé** |
| **Recall@5 — Code** | $\ge 50\%$ | **76.8%** | **Validé (+26.8%)** |
| **Temps d'indexation** | $\le 5$ minutes | **8.27 secondes** | **Validé** |
| **Débit de recherche** | $200\text{ q} \le 90\text{ s}$ | **< 0.5 seconde** | **Validé (180x plus rapide)** |
| **Taille max des chunks** | $\le 2000$ car. | **100% $\le 2000$** | **Validé** |

---

## 6. Suite de Tests Unitaires & Qualité de Code

- **Tests unitaires** : 25 tests passés avec succès (`tests/test_indexer.py`, `tests/test_retriever.py`, `tests/test_chunkers.py`, `tests/test_tokenizer.py`).
- **`flake8 .`** : **0 erreur, 0 avertissement** (PEP 8 stricte, lignes $\le 79$ caractères).
- **`mypy src tests --strict`** : **100% de succès** sur l'intégralité des 25 fichiers sources.
- **Robustesse** : Gestion gracieuse de toutes les entrées dégénérées (requêtes vides, $k \le 0$, index absent, etc.) sans aucun crash.

---

## 7. Prochaine Étape : JOUR 4 & JOUR 5

- **Jour 4 : Optimisation fine** : Ajustements éventuels, déduplication et préparation des bonus.
- **Jour 5 : Augmentation et Génération LLM (`Qwen/Qwen3-0.6B`)** : Construction du contexte dans le budget de tokens, génération sans hallucination et commandes `answer` / `answer_dataset`.

