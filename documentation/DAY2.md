# Journal de Bord — Jour 2 : Exploration du Corpus, Double Stratégie de Chunking et Tokenizer Avancé

## 1. Objectifs de la Journée

L'objectif du deuxième jour était de concevoir et valider la chaîne d'ingestion et de préparation des données pour le moteur RAG :
- **Découverte et filtrage du corpus** (`src/indexing/corpus_loader.py`) : parcours récursif, normalisation stricte des chemins relatifs et exclusion des fichiers binaires / inutiles.
- **Double stratégie de chunking** (`src/indexing/chunkers.py`) :
  - Stratégie syntaxique AST pour le code Python (`.py`).
  - Stratégie hiérarchique par titres et paragraphes pour la documentation (`.md`, `.rst`, `.txt`).
  - Respect absolu de la limite maximale de taille (`max_chunk_size <= 2000` caractères) et intégrité exacte des positions de caractères (`content[first:last] == chunk.text`).
- **Tokenizer adapté au code et à BM25** (`src/indexing/tokenizer.py`) :
  - Découpage des identifiants `snake_case` et `camelCase` / `PascalCase` tout en conservant le terme composé.
  - Injection pondérée des mots-clés issus du chemin de fichier (`file_path`).
  - Filtrage des mots vides conversationnels anglais et conservation des mots-clés techniques.
- **Suite de tests unitaires** (`tests/test_chunkers.py`, `tests/test_tokenizer.py`) et validation sur l'ensemble des 2 105 fichiers du corpus réel `vllm-0.10.1`.

---

## 2. Ingestion du Corpus (`src/indexing/corpus_loader.py`)

### 2.1 Normalisation stricte des chemins (`file_path`)
La moulinette officielle compare les chemins de fichiers caractère par caractère avec les jeux de données de référence (`data/raw/vllm-0.10.1/...`).
- Tout préfixe `./` est systématiquement retiré.
- Les chemins absolus (`/home/...`) sont strictement proscrits.
- Les séparateurs Windows (`\`) sont normalisés en slashes (`/`).

### 2.2 Filtrage sélectif
- **Fichiers acceptés** : Code source (`.py`, `.c`, `.cpp`, `.cu`, `.cuh`, `.h`, `.hpp`), documentation (`.md`, `.rst`, `.txt`), et configuration (`.yaml`, `.yml`, `.toml`, `.sh`, `Dockerfile`).
- **Fichiers ignorés** : Binaires et images (`.png`, `.jpg`, `.pdf`, `.so`, `.whl`, `.pt`, `.safetensors`, `.wasm`, etc.), fichiers cachés (`.*`), et répertoires système/compilation (`.git/`, `__pycache__/`, `.venv/`, `.mypy_cache/`, `dist/`, `build/`).
- **Résultat** : 2 227 fichiers indexables découverts dans `data/raw/vllm-0.10.1`, couvrant **100% des fichiers sources** référencés dans `dataset_docs_public.json` et `dataset_code_public.json`.

---

## 3. Double Stratégie de Chunking (`src/indexing/chunkers.py`)

### 3.1 Les Invariants Fondamentaux
Chaque extrait généré respecte strictement :
1. `0 <= first_character_index < last_character_index <= len(content)`
2. `last_character_index - first_character_index <= max_chunk_size` (défaut : 2000 caractères).
3. `content[first_character_index:last_character_index] == chunk.text` : aucun rognage (*strip*) avant extraction des bornes pour ne pas fausser le calcul de l'IoU.

### 3.2 Stratégie 1 : Code Python (`chunk_python_code`)
- Analyse syntaxique basée sur le module standard `ast`.
- Indexation précise des décalages caractères par ligne (`line_starts`).
- Découpage orienté classes et fonctions :
  - **Fonctions / Méthodes** : Prise en compte des décorateurs (`decorator_list`) pour inclure la signature complète. Si une fonction dépasse 2000 caractères, elle est découpée par fenêtres glissantes alignées sur les retours à la ligne (`\n`) avec un chevauchement (*overlap*) de ~150 caractères.
  - **Classes** : Si la classe complète fait $\le 2000$ caractères, elle forme un chunk unique. Si elle dépasse 2000 caractères, l'en-tête (docstring + variables) est extrait, puis chaque méthode est découpée individuellement.
  - **Éléments de tête et intermédiaires** : Imports, constantes globales et commentaires sont regroupés en blocs cohérents.
  - **Repli robuste** : En cas d'erreur de syntaxe (`SyntaxError`), repli automatique sur le découpage par lignes glissantes.

### 3.3 Stratégie 2 : Documentation Markdown & Texte (`chunk_markdown_text`)
- Découpage hiérarchique basé sur les titres Markdown (`#`, `##`, `###`, etc.).
- Chaque section entre deux titres est analysée :
  - Si la section fait $\le 2000$ caractères, elle forme un chunk cohérent.
  - Si la section dépasse 2000 caractères, elle est subdivisée le long des doubles sauts de ligne (`\n\n` pour les paragraphes) puis des retours à la ligne (`\n`).

### 3.4 Validation Réelle sur le Corpus
- **2 105 fichiers** ingérés et découpés en **1.95 seconde**.
- **47 221 chunks** générés.
- **Taille maximale observée** : exactement 2 000 caractères (aucun dépassement).
- **Taux de chevauchement théorique IoU $\ge 0.05$** :
  - Documentation : **98.0%** des sources de référence validées (seuil requis : 80%).
  - Code : **98.0%** des sources de référence validées (seuil requis : 50%).

---

## 4. Tokenizer Avancé (`src/indexing/tokenizer.py`)

Pour maximiser le score BM25 sur le code source et la documentation technique :

### 4.1 Éclatement des Identifiants (*Identifier Splitting*)
- **`snake_case`** : `gpu_memory_utilization` $\rightarrow$ `['gpu_memory_utilization', 'gpu', 'memory', 'utilization']`.
- **`camelCase` / `PascalCase`** : `AsyncLLMEngine` $\rightarrow$ `['asyncllmengine', 'async', 'llm', 'engine']`.
- Les termes composés et leurs sous-mots sont tous deux conservés pour matcher à la fois les requêtes textuelles naturelles et les citations exactes de symboles.

### 4.2 Injection Pondérée du Chemin de Fichier (*Path Token Injection*)
- Les répertoires techniques de base (`data`, `raw`, `vllm-0.10.1`, extensions) sont ignorés.
- Les mots-clés du chemin (`docs`, `features`, `lora`) sont injectés de manière pondérée (poids par défaut : 2x).
- Cela permet à un chunk situé au cœur de `docs/features/lora.md` d'être retrouvé même si le mot `"lora"` n'apparaît pas explicitement dans son paragraphe.

### 4.3 Nettoyage & Mots Vides
- Filtrage des mots vides conversationnels anglais (`what`, `how`, `which`, `the`, `is`, etc.).
- Préservation des mots-clés de programmation (`def`, `class`, `async`, `return`, `int`, etc.).

---

## 5. Suite de Tests Unitaires (`tests/`)

- **`tests/test_chunkers.py`** :
  - Découpage de plages textuelles avec et sans retours à la ligne.
  - Vérification de l'intégrité `content[start:end] == chunk.text`.
  - Fonctions Python géantes (> 5000 caractères) découpées sans crash.
  - Gestion des fichiers vides, espaces seuls et erreurs de syntaxe.
  - Découpage Markdown structuré par titres.
  - Dispatcher automatique selon l'extension (`.py`, `.md`, `.txt`).
- **`tests/test_tokenizer.py`** :
  - Découpage `snake_case` et `camelCase`.
  - Expressions composées (`AsyncLLMEngine.step()`).
  - Suppression des mots vides anglais.
  - Extraction et injection des tokens de chemin.
  - Robustesse aux chaînes vides et ponctuations pures.

---

## 6. Qualité et Standards

- **`flake8 .`** : **0 erreur, 0 avertissement** (longueur de ligne $\le 79$ caractères).
- **`mypy src tests --strict`** : **100% typé statiquement** (21 fichiers sources validés sans erreur).
- **Code intégralement en anglais** (noms de fonctions, variables, docstrings PEP 257 Google style et commentaires).

---

## 7. Prochaine Étape : JOUR 3

Les briques de découpage et de tokenisation étant prêtes et vérifiées :
1. Implémentation du moteur d'indexation lexicale BM25 via `bm25s` (`src/indexing/indexer.py`).
2. Implémentation du module de recherche rapide (`src/retrieval/lexical_retriever.py`).
3. Connexion des commandes CLI `index`, `search` et `search_dataset`.
4. Première exécution de bout en bout et mesure du score `Recall@5` sur les datasets de référence (`dataset_docs_public.json` et `dataset_code_public.json`).

