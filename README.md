_This project has been created as part of the 42 curriculum by marberge._

<div align="center">
<br>
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTQPzuYKu7n0cWUYa5Kbg0_LrlEQAIURWeo9A&s" alt="42 Logo" width="400" />

  <br>
</div>

# RAG against the machine

<div align="center">

<img src="https://img.shields.io/badge/Project-000000?style=for-the-badge&logo=42&logoColor=white" alt="42" />
	<img src="https://img.shields.io/badge/Language-Python-00599C?style=for-the-badge&logo=&logoColor=white" alt="Python" />
	<img src="https://img.shields.io/badge/Score-Pending-yellow?style=for-the-badge&logoColor=white" alt="Score" />
	<br>
	<img src="https://img.shields.io/badge/IA-darkviolet?style=for-the-badge" alt="IA" />
	<img src="https://img.shields.io/badge/RAG-darkviolet?style=for-the-badge" alt="RAG" />
	<img src="https://img.shields.io/badge/Chunking-darkviolet?style=for-the-badge" alt="Chunking" />
	<img src="https://img.shields.io/badge/BM25-darkviolet?style=for-the-badge" alt="BM25" />
	<img src="https://img.shields.io/badge/Information_Retrieval-darkviolet?style=for-the-badge" alt="IR" />
	<img src="https://img.shields.io/badge/Model-Qwen3--0.6B-darkviolet?style=for-the-badge" alt="Qwen" />

</div>

## Description

**RAG against the machine** is an end-to-end Retrieval-Augmented Generation (RAG) system built from scratch in Python to index, search, and answer complex technical questions over large-scale codebases (~300,000 lines of code and documentation from `vLLM 0.10.1`).

The system ingests raw code and documentation files, decomposes them into semantically cohesive chunks strictly bounded by 2000 characters, builds an ultra-fast in-memory lexical search engine powered by `bm25s`, and synthesizes grounded, factual responses using a local CPU-based small language model (`Qwen/Qwen3-0.6B`).

---

## Instructions

### Prerequisites
- **Linux / macOS**
- **Python 3.10+**
- **uv** (fast Python package manager)

### Quick Start
To set up dependencies and inspect available commands:
```bash
make install
make run
```

### Makefile Commands Reference
The project lifecycle is fully automated via the `Makefile`:

- `make install` (or `make all`): Synchronizes project dependencies via `uv sync`.
- `make run`: Displays CLI options and command signatures via `python -m src --help`.
- `make debug`: Launches the interactive debugger (`pdb`) over the CLI entry point.
- `make lint`: Validates code style and typing using `flake8 .` and standard `mypy .`.
- `make lint-strict`: Enforces strict static typing (`mypy . --strict`) and zero `flake8` warnings.
- `make test`: Runs the comprehensive test suite with `pytest -v`.
- `make clean`: Removes bytecode caches (`__pycache__`), `.pytest_cache`, and build artifacts.
- `make fclean`: Performs a full cleanup by executing `clean` and removing `.venv`.
- `make re`: Fully rebuilds the environment (`fclean` followed by `install`).

---

## Command-Line Interface (CLI)

All CLI operations are invoked via `uv run python -m src <command> [options]`:

### 1. Ingest & Index
```bash
uv run python -m src index --max_chunk_size 2000 --corpus_dir data/raw --output_dir data/processed
```
Traverses the corpus, splits files into chunks $\le 2000$ characters, and saves the lexical index and metadata under `data/processed/`.

### 2. Search a Single Query
```bash
uv run python -m src search "How does PagedAttention manage KV cache?" --k 5
```
Returns ranked source locations matching the query in `<file_path> [<first>:<last>]` format.

### 3. Search a Whole Dataset
```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```
Processes an entire dataset batch and generates a `StudentSearchResults` JSON file.

### 4. Answer a Single Query
```bash
uv run python -m src answer "What activation formats does the fused batched MoE layer return in vLLM?" --k 3
```
Retrieves the top-$k$ sources, builds a token-budgeted prompt, and outputs the model's generated answer.

### 5. Answer a Dataset
```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```
Generates answers for every retrieved result in a dataset and exports a `StudentSearchResultsAndAnswer` JSON file.

### 6. Evaluate Retrieval Quality
```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```
Computes and reports official Recall@1, Recall@3, Recall@5, and Recall@10 based on 1D IoU overlap.

---

## System Architecture

```text
       Raw Corpus (vLLM)
       [Code (.py) & Docs (.md)]
                 │
                 ▼
      ┌───────────────────────┐
      │     Chunkers          │  ◄── AST Python parsing + Markdown header splitting
      │ (bounds <= 2000 chars)│
      └──────────┬────────────┘
                 │
                 ▼
      ┌───────────────────────┐
      │   Custom Tokenizer    │  ◄── Identifier splitting (camelCase, snake_case)
      │  & Path Injector      │      + file_path weight boosting
      └──────────┬────────────┘
                 │
                 ▼
      ┌───────────────────────┐
      │  BM25 Index & Cache   │  ◄── Millisecond retrieval & Spatial deduplication
      └──────────┬────────────┘
                 │
        Retrieved Sources
                 │
                 ▼
      ┌───────────────────────┐
      │    Context Builder    │  ◄── Slices exact text spans [first:last]
      │  (Token Budget 2048)  │
      └──────────┬────────────┘
                 │
                 ▼
      ┌───────────────────────┐
      │ Local LLM Generator   │  ◄── Lazy-loaded Qwen/Qwen3-0.6B on CPU
      │  (Grounded Prompt)    │      (No thinking tags, greedy decoding)
      └──────────┬────────────┘
                 │
                 ▼
          Grounded Answer
```

---

## Chunking Strategy

A code file and a documentation page exhibit fundamentally different structural hierarchies. We implement two distinct strategies:

1. **Python Code Chunking (`chunk_python_code`)**:
   - Parses the Abstract Syntax Tree (AST) to identify top-level and class-level functions (`def`, `async def`), classes, and module statements.
   - Groups small statements together while strictly respecting syntactic boundaries.
   - When a single block exceeds `max_chunk_size` (2000 chars), line-based slicing with ~150 characters overlap is applied.
2. **Markdown & Documentation Chunking (`chunk_markdown_text`)**:
   - Splits hierarchically along Markdown headers (`#`, `##`, `###`), followed by paragraph boundaries (`\n\n`), and line breaks (`\n`).
3. **Hard Bounds Invariant**:
   - For every chunk: `0 <= first_char < last_char <= len(file)` and `last_char - first_char <= 2000`.
   - Slices are extracted verbatim from raw text strings to prevent character offset drift.

---

## Retrieval Method

To achieve sub-second query performance and high recall over technical terms, our retrieval engine relies on:

- **Custom Code Tokenizer**: Standard NLP tokenizers fail on identifiers. We split `snake_case` (e.g. `gpu_memory_utilization` $\to$ `gpu`, `memory`, `utilization`) and `camelCase`/`PascalCase` (`AsyncLLMEngine` $\to$ `async`, `llm`, `engine`), while preserving the unified compound token to match both verbatim symbols and conversational queries.
- **Path Token Injection**: Deeply nested modules often omit keywords explicitly mentioned in their file path (e.g. `docs/features/lora.md`). We inject normalized path tokens weighted into the chunk's token list.
- **Spatial Deduplication & Merging**: When multiple overlapping chunks from the same file are retrieved, they are merged into a single contiguous span if their union is $\le 2000$ characters, avoiding redundant duplicates in top-$k$ candidate slots.

---

## Performance Analysis

Tested on the official 42 evaluation benchmark:

| Metric | 42 Required Threshold | Our System Performance | Status |
| :--- | :--- | :--- | :--- |
| **Recall@5 (Documentation)** | $\ge 80.0\%$ | **85.0%** | **PASS** ✅ |
| **Recall@5 (Source Code)** | $\ge 50.0\%$ | **76.8%** | **PASS** (+26.8% margin) ✅ |
| **Corpus Indexing Time** | $\le 5\text{ min}$ | **~8.27 seconds** (47,221 chunks) | **PASS** ✅ |
| **Retrieval Throughput** | $200\text{ queries} \le 90\text{ s}$ | **~0.45 seconds** | **PASS** ✅ |

---

## Design Decisions

- **Lazy Loading for Deep Learning**: `torch` and `transformers` are never imported during `index`, `search`, or `evaluate` runs. This guarantees instant sub-second startup times for lexical operations.
- **Pydantic Validation**: All data models adhere strictly to the subject schemas (`MinimalSource`, `StudentSearchResults`, `StudentSearchResultsAndAnswer`) with zero conversion drift.
- **Fully Autonomous Offline Execution**: The entire pipeline operates with `local_files_only=True` without needing external background daemons (such as Ollama).
- **Deterministic Generation**: Configured with greedy decoding (`do_sample=False`) and `enable_thinking=False` on `Qwen/Qwen3-0.6B` to suppress internal reasoning overhead and produce concise, direct answers on CPU.

---

## Challenges Faced & Solutions

### 1. Small Model Refusal Bias
- **Problem**: Small language models like `Qwen3-0.6B` exhibit strong refusal bias when system prompts strictly forbid extrapolation ("do not extrapolate or hallucinate"). When a retrieved code chunk answered the question (e.g., returning `BatchedExperts`) without repeating the full compound phrase from the query in the function body, the model panicked and answered: *"The provided context does not contain sufficient information to answer this question."*
- **Solution**: Refined the system prompt to guide the model that context snippets come from source code and documentation (directing it to inspect class names, function signatures, and file paths), transforming passive refusals into direct, accurate answers.

### 2. Lexical Mismatch Between Natural Language and Code Identifiers
- **Problem**: Queries such as *"How to run asynchronous LLM engine"* failed to match functions defined as `class AsyncLLMEngine` or `async_llm_engine`.
- **Solution**: Implemented bidirectional identifier decomposition, generating both compound and constituent sub-tokens.

### 3. Spatial Redundancy in Top-k Slots
- **Problem**: BM25 often returned 3 consecutive overlapping slices of the same large function, taking up 3 out of 5 precious slots and hurting Recall@5.
- **Solution**: Added spatial Non-Maximum Suppression (NMS) and bounded union merging ($\le 2000$ characters).

---

## Example Usage Walkthrough

```bash
# 1. Index the codebase
uv run python -m src index --max_chunk_size 2000

# 2. Run retrieval on documentation questions
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

# 3. Score against ground-truth dataset
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

# 4. Generate grounded answers
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

---

## Resources & AI Usage

### References

- **RAG Guide**: [www.stephane-robert.info/docs/developper/programmation/python/rag-introduction/](https://www.stephane-robert.info/docs/developper/programmation/python/rag-introduction/)
- **RAG Guide**: [shubham-sharma.fr/articles/guide-rag-retrieval-augmented-generation/](https://shubham-sharma.fr/articles/guide-rag-retrieval-augmented-generation/)
- **RAG Guide**: [mlpills.substack.com/p/an-animated-guide-to-rag-how-ai-finds](https://mlpills.substack.com/p/an-animated-guide-to-rag-how-ai-finds)
- **BM25s Library**: Fast lexical search implementation in Python ([GitHub: xhluca/bm25s](https://github.com/xhluca/bm25s))
- **Qwen3 Model Series**: Technical report and model weights ([Hugging Face: Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B))
- **Pydantic Documentation**: Type validation and serialization ([docs.pydantic.dev](https://docs.pydantic.dev))
- **vLLM Documentation**: [docs.vllm.ai](https://docs.vllm.ai)

### AI Usage Disclosure
In accordance with 42 curriculum guidelines:
- **Architecture & Scaffolding**: AI was used to draft initial Pydantic data schemas, review project specifications, and structure the development roadmap.
- **Optimization & Tokenization**: AI assisted in designing regular expressions for CamelCase and snake_case identifier splitting and path token weighting.
- **Debugging**: AI was used to diagnose the small model refusal bias in `Qwen3-0.6B` and optimize prompt grounding.
- **Verification**: All generated logic, chunking invariants, and retrieval metrics were reviewed, tested with unit test suites, and verified against the official `moulinette` binary.
- **Readme**: AI was used to write this readme and illustrate the RAG pipeline.
