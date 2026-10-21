Voici un prompt complet, structuré comme un cahier des charges d'ingénierie, conçu spécifiquement pour un agent autonome (comme Antigravity).

Je l'ai rédigé **directement en anglais**: c'est le meilleur moyen de garantir que l'agent écrira 100 % du code, des variables, des commentaires, des messages d'erreur CLI et des docstrings PEP 257 en anglais sans jamais glisser de français par erreur.

> **Petit rappel de binôme avant que tu ne lances l'agent:** même si tu utilises l'agent pour générer la base rapidement, demande-lui d'avancer **brique par brique** et relis bien chaque fichier (surtout le _chunker_, le _tokenizer_ et le calcul d' _IoU_). En soutenance à 42, tu auras une étape de _Recode_ où le correcteur te demandera de modifier en direct une partie du pipeline ou de la structure de données en quelques minutes, et tu dois être capable de défendre chaque choix technique sans hésiter.

### Le message à copier-coller à ton agent Antigravity

````
# Project Specification: "RAG against the machine" (42 School Project)

You are an expert Python software engineer. Your task is to design, implement, and test a complete, modular Retrieval-Augmented Generation (RAG) system from scratch over the \`vllm-0.10.1\` codebase (~300,000 lines of code and documentation).

Read this entire specification carefully before writing any code. Every architectural rule, data model, CLI signature, and performance threshold is strictly enforced by an automated grading binary (\`moulinette\`).

---

## 1. Core Engineering & Quality Rules (Strict Enforcement)

1. **Language**: Everything in the codebase MUST be written in **English** (variable names, function/class names, PEP 257 docstrings, comments, CLI messages, and tests).
2. **Python & Package Manager**:
   - Python **3.10+**.
   - Managed exclusively with **\`uv\`** (\`pyproject.toml\` and \`uv.lock\` at the repository root). The evaluation environment only runs \`uv sync\`.
3. **Coding Standards & Static Typing**:
   - Must pass \`flake8 .\` with zero warnings/errors.
   - Complete type hints everywhere. Must pass both:
     - \`mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs\`
     - \`mypy . --strict\`
   - Include PEP 257 compliant docstrings (Google style) for all modules, classes, and functions.
4. **Exception Handling & Robustness**:
   - **NEVER crash with an unhandled traceback.**
   - Use \`try-except\` blocks and context managers (\`with open(...)\`) for all file I/O, JSON parsing, indexing, and model inference.
   - Gracefully handle degenerate inputs: empty query (\`""\`), whitespace-only or punctuation-only query (\`"!@#$%^&*()"\`), \`k <= 0\`, \`max_chunk_size <= 0\`, missing files/directories, unbuilt index, or malformed JSON files. Print a clear error message and exit cleanly or return an empty valid result as appropriate.
5. **Git Ignore**:
   - Never track \`data/\`, model weights, caches, or generated outputs in Git. Ensure \`.gitignore\` excludes \`data/\`, \`__pycache__/\`, \`.mypy_cache/\`, \`.pytest_cache/\`, \`.venv/\`, and model cache directories.

---

## 2. Target Repository & Module Architecture

The project must be runnable as \`uv run python -m src <command> [options]\`. Use the following clean, modular architecture:

\`\`\`text
├── Makefile
├── pyproject.toml
├── uv.lock
├── README.md
├── data/                          # Ignored by Git
│   ├── raw/vllm-0.10.1/           # Input corpus (read-only)
│   ├── datasets/                  # AnsweredQuestions/ and UnansweredQuestions/
│   ├── processed/                 # Persisted lexical/hybrid indices & chunk metadata
│   └── output/                    # Generated JSON outputs for evaluation
├── src/
│   ├── __init__.py
│   ├── __main__.py                # Entry point for \`uv run python -m src\`
│   ├── cli.py                     # Python Fire CLI controller & input validation
│   ├── config.py                  # Default paths, constants, model names
│   ├── models.py                  # Mandatory Pydantic models + internal structures
│   ├── pipeline.py                # Main RAG orchestrator connecting all stages
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── corpus_loader.py       # File discovery, filtering, and hash tracking
│   │   ├── chunkers.py            # Distinct Python code chunker & Markdown/text chunker
│   │   ├── tokenizer.py           # Code-aware tokenizer (CamelCase/snake_case + file_path)
│   │   └── indexer.py             # BM25 (bm25s) index creation, persistence & incremental update
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── lexical_retriever.py   # Fast BM25 retrieval + spatial deduplication
│   │   └── hybrid_retriever.py    # (Optional Bonus) Vector embeddings + RRF fusion
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── context_builder.py     # Reads exact [first:last] slices & manages token budget
│   │   └── llm_generator.py       # Lazy-loaded Qwen/Qwen3-0.6B inference via transformers
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py           # IoU (>= 0.05) and Recall@k computation
│   └── utils/
│       ├── __init__.py
│       ├── io_helpers.py          # Safe JSON reading/writing & path resolution
│       ├── cache.py               # Memory & disk caching for fast repeated queries
│       └── api.py                 # Local HTTP API (FastAPI) for Bonus
└── tests/
    ├── test_chunkers.py
    ├── test_evaluator.py
    ├── test_tokenizer.py
    └── test_edge_cases.py

## 3\. Mandatory Pydantic Data Models (src/models.py)

Implement these exact Pydantic models to validate data exchanged between stages and serialize JSON outputs:

```python

import uuid
from typing import List, Union
from pydantic import BaseModel, Field

class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int

class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str

class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str

class RagDataset(BaseModel):
    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]

class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]

class MinimalAnswer(MinimalSearchResults):
    answer: str

class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int

class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int
````

## 4\. Detailed Stage Requirements & Technical Strategy

### Stage 1: Indexing (src/indexing/)

- **Corpus Discovery (`corpus_loader.py`)**:
  - Walk `data/raw/vllm-0.10.1` (configurable path).
    - **CRITICAL RULE ON `file_path`**: Every `MinimalSource.file_path` MUST match the relative path from the project root verbatim (e.g., `"data/raw/vllm-0.10.1/docs/features/lora.md"` or `"data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py"`). Normalize paths with forward slashes (`/`), no leading `./`.
    - Filter out binary files, images (`.png`, `.jpg`, `.svg`, `.ico`, `.gif`, `.pdf`, `.so`, `.whl`), `.git/`, `__pycache__/`, and unreadable non-UTF8 files. Index relevant text/code files (`.py`, `.md`, `.rst`, `.txt`, `.yaml`, `.yml`, `.json`, `.toml`, `.sh`, `.cmake`, `.c`, `.cu`, `.cpp`, `.h`, `.hpp`, `Dockerfile`).
- **Two Distinct Chunking Strategies (`chunkers.py`)**:
  - **Hard Constraint**: Every single chunk MUST satisfy `0 <= first_character_index < last_character_index <= len(file_content)` and **`last_character_index - first_character_index <= max_chunk_size`** (default `2000` characters). If a single retrieved chunk exceeds 2000 characters, the grader invalidates the entire submission!
    - **Exact Character Offsets**: `first_character_index` and `last_character_index` must be the exact character indices in the raw file string `content[first:last]`. Never mutate or strip the file string before computing indices.
    - **Strategy A — Python Code Chunking (`.py` files)**:
    - Split Python code along syntactic boundaries (e.g., top-level and class-level `def`, `async def`, `class` blocks using `ast` line offsets or regex/structural boundaries). - Group small adjacent statements/imports together up to `max_chunk_size`. - If a single function/class exceeds `max_chunk_size`, split it along line breaks (`\n`) or character windows with a small overlap (~150–200 characters) while strictly enforcing `last_character_index - first_character_index <= max_chunk_size`.
    - **Strategy B — Markdown / Text Chunking (`.md`, `.rst`, `.txt`, etc.)**:
    - Split structurally by Markdown headings (`\n# `, `\n## `, `\n### `), then double newlines (`\n\n`), then single newlines (`\n`), and finally hard character slices if a single line exceeds `max_chunk_size`, with ~150 characters overlap between chunks.
- **Code & Path-Aware Tokenization (`tokenizer.py`)**:
  - Standard English tokenizers fail on source code. Implement a custom tokenizer function for `bm25s`:
    1. Extract alphanumeric and underscore tokens. 2. Split `snake_case` (e.g., `async_llm_engine` -> `async`, `llm`, `engine`) and `camelCase` / `PascalCase` (e.g., `AsyncLLMEngine` -> `async`, `llm`, `engine`), while **also keeping** the original compound identifier (`asyncllmengine`) so both verbatim identifier queries and natural language queries match. 3. **Path Injection**: Tokenize the chunk's `file_path` (excluding `"data"`, `"raw"`, `"vllm-0.10.1"`) and prepend/include those path tokens (weighted 2x or 3x) into the chunk's token list. This drastically improves Recall@5 when a chunk inside `docs/features/lora.md` doesn't repeat the word `"lora"` in every paragraph. 4. Remove common English stopwords while keeping meaningful technical keywords.
- **Lexical Index Persistence (`indexer.py`)**:
  - Use `bm25s` to build the BM25 index over all tokenized chunks.
    - Display progress bars using `tqdm` for chunking and indexing.
    - Save the `bm25s` index and chunk metadata (`file_path`, `first_character_index`, `last_character_index`) under `data/processed/`.
    - Total indexing time must be well under **5 minutes** (with `bm25s`, target < 30 seconds).

### Stage 2: Retrieval (src/retrieval/)

- **Performance & Quality Targets**:
  - Must process 200 questions in under **90 seconds**.
    - Must achieve **Recall@5 >= 80%** on `dataset_docs_public.json` and **Recall@5 >= 50%** on `dataset_code_public.json`.
- **Smart Retrieval & Spatial Deduplication (`lexical_retriever.py`)**:
  - Keep the loaded `bm25s` index cached in memory across batch queries so it is only loaded once per process.
    - When retrieving top- `k` results for a query, fetch the top `k * 3` candidates from `bm25s` and apply spatial deduplication/merging:
    - If two candidate chunks come from the same `file_path` and overlap (or are immediately adjacent) and their union span `max(last1, last2) - min(first1, first2) <= max_chunk_size` (2000), merge them into a single `MinimalSource` covering `[min(first1, first2), max(last1, last2)]` or skip near-duplicate spans so you don't waste top- `k` slots on redundant overlapping chunks.
    - Guarantee that every returned `MinimalSource` strictly satisfies `last_character_index - first_character_index <= 2000`.

### Stage 3 & 4: Augmentation & Answer Generation (src/generation/)

- **Model Requirement**:
  - Default model: **`Qwen/Qwen3-0.6B`** loaded via Hugging Face `transformers` and `torch` (CPU-compatible). Do NOT use Ollama, as the evaluation environment only runs `uv sync` and has no external daemon.
    - **Lazy Loading**: Only import and initialize the LLM and tokenizer when `answer` or `answer_dataset` is called, so `index`, `search`, `search_dataset`, and `evaluate` remain instantaneous.
- **Context Building (`context_builder.py`)**:
  - Given the retrieved `List[MinimalSource]`, open each referenced `file_path` and slice `content[source.first_character_index:source.last_character_index]`.
    - Format each snippet clearly with its `file_path` header and fit as many top snippets as possible within the model's context budget (e.g., max 2048 input tokens).
- **Grounded Generation (`llm_generator.py`)**:
  - Construct a chat prompt instructing `Qwen/Qwen3-0.6B` to answer the user's question concisely, accurately, and **strictly grounded** in the provided context snippets without hallucinating.
    - If using `tokenizer.apply_chat_template`, pass `enable_thinking=False` (supported by Qwen3) so the model generates the direct answer rapidly on CPU without wasting tokens on `<think>...</think>` blocks. Set `max_new_tokens=256`, `do_sample=False` for fast, deterministic CPU generation.

### Stage 5: Self-Evaluation (src/evaluation/evaluator.py)

- Never import or call the external `moulinette` binary in your code.
- Implement the exact official evaluation metric for the `evaluate` CLI command:
  - **IoU (Intersection over Union)** between a retrieved source $R = [r_{\text{start}}, r_{\text{end}}]$ and a ground-truth source $G = [g_{\text{start}}, g_{\text{end}}]$:
    - If $R.\text{file\_path} \neq G.\text{file\_path}$, $\text{IoU} = 0.0$. - Otherwise:
      $$
      \text{intersection} = \max(0, \min(r_{\text{end}}, g_{\text{end}}) - \max(r_{\text{start}}, g_{\text{start}}))
      $$
      $$
      \text{union} = (r_{\text{end}} - r_{\text{start}}) + (g_{\text{end}} - g_{\text{start}}) - \text{intersection}
      $$
      $$
      \text{IoU} = \frac{\text{intersection}}{\text{union}} \quad (\text{if union} > 0 \text{ else } 0.0)
      $$
    - A ground-truth source $G$ is considered **found** in the top- $k$ retrieved sources if at least one retrieved source in `retrieved_sources[:k]` has $\text{IoU}(R, G) \ge 0.05$.
    - For a single question with $M$ ground-truth sources, the question recall at $k$ is:
      $$
      \text{Recall@}k(q) = \frac{\text{Number of ground-truth sources for } q \text{ found in top-}k}{M}
      $$
    - Dataset `Recall@k` is the mean of $\text{Recall@}k(q)$ across all questions in the ground-truth dataset. Compute and print `Recall@1`, `Recall@3`, `Recall@5`, and `Recall@10`.

## 5\. Mandatory CLI Commands (src/\_\_main\_\_.py & src/cli.py)

Expose a CLI using **Python Fire** invoked via `uv run python -m src <command> [options]`. All input/output paths must be configurable arguments (never hard-coded):

1. `uv run python -m src index --max_chunk_size 2000 [--corpus_dir data/raw] [--output_dir data/processed]`
   - Ingests `data/raw/`, chunks all files respecting `max_chunk_size`, and saves the index under `data/processed/`.
2. `uv run python -m src search "<query>" --k 5 [--index_dir data/processed]`
   - Searches the index for a single query, prints the ranked source locations (`file_path [first:last]`), and returns/displays the result.
3. `uv run python -m src search_dataset --dataset_path <path> --k 10 --save_directory <dir> [--index_dir data/processed]`
   - Reads a `RagDataset` JSON file from `--dataset_path`, runs retrieval with a `tqdm` progress bar, and writes a `StudentSearchResults` JSON file into `--save_directory` using the exact same filename as the input dataset (e.g., `<save_directory>/dataset_docs_public.json`). Automatically creates `--save_directory` if it does not exist.
4. `uv run python -m src answer "<query>" --k 5 [--index_dir data/processed] [--model_name Qwen/Qwen3-0.6B]`
   - Retrieves the top- `k` sources and generates an answer for a single query using `Qwen/Qwen3-0.6B`, printing both sources and answer.
5. `uv run python -m src answer_dataset --student_search_results_path <path> --save_directory <dir> [--model_name Qwen/Qwen3-0.6B]`
   - Reads a `StudentSearchResults` JSON file from `--student_search_results_path`, generates answers for each question with a `tqdm` progress bar, and writes a `StudentSearchResultsAndAnswer` JSON file into `--save_directory` preserving the input filename.
6. `uv run python -m src evaluate --student_search_results_path <path> --dataset_path <path>`
   - Evaluates a `StudentSearchResults` JSON file against a ground-truth `AnsweredQuestions` `RagDataset` JSON file and prints `Recall@1`, `Recall@3`, `Recall@5`, and `Recall@10`.

## 6\. Makefile Requirements

Provide a `Makefile` at the root of the repository with these exact targets:

- `install`: `uv sync`
- `run`: `uv run python -m src --help` (or default demonstration command)
- `debug`: `uv run python -m pdb -m src`
- `clean`: Remove `__pycache__`, `.mypy_cache`, `.pytest_cache`, `.ruff_cache`, temporary build artifacts.
- `lint`: Run `uv run flake8 .` and `uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs`
- `lint-strict`: Run `uv run flake8 .` and `uv run mypy . --strict`

## 7\. High-ROI Bonuses to Include Cleanly

Once the mandatory pipeline is rock-solid, integrate these bonuses so they work out of the box on a CPU-only machine:

1. **Incremental Indexing (`src/indexing/indexer.py`)**: Store a `manifest.json` in `data/processed/` mapping each `file_path` to its SHA-256 hash (or mtime + size) and `max_chunk_size`. When `index` runs again with the same `max_chunk_size`, only re-chunk new/modified files, reuse cached chunks for unchanged files, remove deleted files, and rebuild the `bm25s` matrices.
2. **Query & Index Caching (`src/utils/cache.py`)**: Cache the loaded index in memory and persist query results (`search` and `answer`) on disk in `data/processed/query_cache.json` keyed by `(query_hash, k, index_version)` for instant repeated queries.
3. **Local HTTP API (`src/utils/api.py`)**: Add a `serve` command to the CLI (`uv run python -m src serve --host 127.0.0.1 --port 8000`) using `FastAPI` + `uvicorn` exposing `GET /health`, `POST /search`, and `POST /answer` validated with the Pydantic models.

## 8\. Step-by-Step Execution Plan

Execute the implementation in this order:

1. Inspect the existing project files, datasets (`data/datasets/AnsweredQuestions/`), and corpus location (`data/raw/vllm-0.10.1`).
2. Configure `pyproject.toml` with dependencies (`pydantic`, `bm25s`, `fire`, `tqdm`, `transformers`, `torch`, `fastapi`, `uvicorn`, plus dev dependencies `flake8`, `mypy`, `pytest`) and run `uv sync`.
3. Implement `src/config.py`, `src/models.py`, `src/utils/io_helpers.py`, and `src/evaluation/evaluator.py`.
4. Implement `src/indexing/corpus_loader.py`, `src/indexing/chunkers.py`, `src/indexing/tokenizer.py`, and `src/indexing/indexer.py`.
5. Implement `src/retrieval/lexical_retriever.py` and `src/utils/cache.py`.
6. Implement `src/generation/context_builder.py` and `src/generation/llm_generator.py`.
7. Implement `src/pipeline.py`, `src/utils/api.py`, `src/cli.py`, and `src/__main__.py`.
8. Write unit tests in `tests/` (testing chunk size <= 2000, IoU math, tokenizer splitting, and CLI edge cases) and run `uv run pytest`.
9. Run the end-to-end indexing and retrieval pipeline on both `dataset_docs_public.json` and `dataset_code_public.json`, evaluate with both `uv run python -m src evaluate` and `./moulinette evaluate_student_search_results`, and tune tokenization/chunking until **Recall@5 >= 80% on Docs** and **Recall@5 >= 50% on Code**.
10. Run `make lint` and `make lint-strict` and fix every single `flake8` or `mypy` issue.

```
---

### Ce que je te conseille de vérifier pendant que l'agent travaille

1. **L'emplacement de tes dossiers \`data/\`** : Dans ton premier message, tu mentionnais que les questions étaient dans \`datasets_public/public/...\`. Or, les scripts d'évaluation automatiques du sujet s'attendent à la convention \`data/raw/vllm-0.10.1\` et \`data/datasets/AnsweredQuestions/...\` (et \`UnansweredQuestions/...\`)[cite: 1]. Vérifie ou crée un lien symbolique (ou déplace les dossiers dans \`data/\`) pour que ton arborescence locale colle exactement à celle de la soutenance[cite: 1].
2. **Le test immédiat avec \`./moulinette\`** : Dès que l'agent a terminé l'étape 9, regarde les scores de \`Recall@5\` sur \`dataset_code_public.json\` et \`dataset_docs_public.json\`[cite: 1]. Si le score sur le code est autour de 45-48 %, dis-le-moi et on ajustera ensemble les poids du tokenizer ou le découpage AST des fichiers Python pour franchir largement la barre des 50 %[cite: 1].

Veux-tu qu'on vérifie ensemble comment sont rangés tes dossiers \`vllm-0.10.1\` et \`datasets_public\` actuellement avant que tu lances l'agent ?
```
