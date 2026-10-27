"""Unit tests for the BM25 code and path-aware tokenizer."""

from src.indexing.tokenizer import (
    extract_path_tokens,
    split_identifier,
    tokenize,
)


def test_split_identifier_snake_case() -> None:
    """snake_case identifiers must produce the full term and subwords."""
    tokens = split_identifier("gpu_memory_utilization")
    assert "gpu_memory_utilization" in tokens
    assert "gpu" in tokens
    assert "memory" in tokens
    assert "utilization" in tokens


def test_split_identifier_camel_case() -> None:
    """camelCase / PascalCase must produce compound lowercased and subwords."""
    tokens = split_identifier("AsyncLLMEngine")
    assert "asyncllmengine" in tokens
    assert "async" in tokens
    assert "llm" in tokens
    assert "engine" in tokens

    tokens_paged = split_identifier("PagedAttention")
    assert "pagedattention" in tokens_paged
    assert "paged" in tokens_paged
    assert "attention" in tokens_paged


def test_split_identifier_with_punctuation() -> None:
    """Surrounding punctuation and method calls are handled."""
    tokens = split_identifier("step()")
    assert tokens == ["step"]


def test_tokenize_compound_expression() -> None:
    """Tokenizing an expression with methods and identifiers."""
    text = "AsyncLLMEngine.step() handles requests"
    tokens = tokenize(text)

    # Check decomposed and compound tokens
    assert "asyncllmengine" in tokens
    assert "async" in tokens
    assert "llm" in tokens
    assert "engine" in tokens
    assert "step" in tokens
    assert "handles" in tokens
    assert "requests" in tokens


def test_tokenize_stopwords_removal() -> None:
    """Common English conversational stopwords are removed."""
    text = "What is the best way to run inference in vLLM?"
    tokens = tokenize(text)

    # Conversational stopwords removed
    assert "what" not in tokens
    assert "is" not in tokens
    assert "the" not in tokens
    assert "to" not in tokens
    assert "in" not in tokens

    # Meaningful words kept
    assert "best" in tokens
    assert "way" in tokens
    assert "run" in tokens
    assert "inference" in tokens


def test_extract_path_tokens() -> None:
    """Extract meaningful tokens from file paths."""
    path = "data/raw/vllm-0.10.1/docs/features/lora.md"
    path_tokens = extract_path_tokens(path)

    assert "docs" in path_tokens
    assert "features" in path_tokens
    assert "lora" in path_tokens

    # Standard prefixes stripped
    assert "data" not in path_tokens
    assert "raw" not in path_tokens
    assert "vllm" not in path_tokens
    assert "md" not in path_tokens


def test_tokenize_with_path_injection() -> None:
    """Path tokens are prepended and weighted in the output token list."""
    path = "data/raw/vllm-0.10.1/docs/features/lora.md"
    tokens = tokenize(
        text="Loading an adapter dynamically",
        file_path=path,
        path_weight=2,
    )

    # Verify that lora appears at least twice from path injection
    assert tokens.count("lora") >= 2
    assert "adapter" in tokens
    assert "dynamically" in tokens


def test_tokenize_edge_cases() -> None:
    """Handle empty strings, whitespace, and pure punctuation."""
    assert tokenize("") == []
    assert tokenize("   \n\t  ") == []
    assert tokenize("!@#$%^&*()_+-=[]{}|;':,.<>?/") == []

