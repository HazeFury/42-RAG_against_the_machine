"""Unit tests for the LexicalRetriever and query processing."""

from pathlib import Path
import pytest

from src.indexing.indexer import build_index
from src.retrieval.lexical_retriever import LexicalRetriever


@pytest.fixture
def mock_index(tmp_path: Path) -> Path:
    """Fixture that builds a small index for retrieval testing."""
    corpus_dir = tmp_path / "raw"
    corpus_dir.mkdir()

    # File 1: Python module
    (corpus_dir / "scheduler.py").write_text(
        "class AsyncScheduler:\n"
        "    def schedule_next_request(self) -> None:\n"
        "        pass\n",
        encoding="utf-8",
    )

    # File 2: Markdown documentation
    (corpus_dir / "lora.md").write_text(
        "# LoRA Adapters\n\n"
        "Load adapters dynamically using HTTP endpoint.\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "processed"
    build_index(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        max_chunk_size=1000,
        show_progress=False,
    )
    return output_dir


def test_retriever_query_match(mock_index: Path) -> None:
    """Test retrieving sources for relevant keywords."""
    retriever = LexicalRetriever(index_dir=mock_index)
    results = retriever.retrieve("AsyncScheduler", k=5)

    assert len(results) >= 1
    top = results[0]
    assert "scheduler.py" in top.file_path
    assert top.first_character_index >= 0
    assert top.last_character_index > top.first_character_index
    assert top.last_character_index - top.first_character_index <= 1000


def test_retriever_path_boost(mock_index: Path) -> None:
    """Querying for path terms like lora returns lora.md."""
    retriever = LexicalRetriever(index_dir=mock_index)
    results = retriever.retrieve("lora adapters", k=5)

    assert len(results) >= 1
    assert "lora.md" in results[0].file_path


def test_retriever_edge_cases(mock_index: Path) -> None:
    """Test robustness with empty, whitespace, punctuation, and k<=0."""
    retriever = LexicalRetriever(index_dir=mock_index)

    # Empty query
    assert retriever.retrieve("", k=5) == []
    assert retriever.retrieve("   ", k=5) == []

    # Pure punctuation
    assert retriever.retrieve("!@#$%^&*()", k=5) == []

    # Invalid k
    assert retriever.retrieve("scheduler", k=0) == []
    assert retriever.retrieve("scheduler", k=-5) == []


def test_retriever_batch(mock_index: Path) -> None:
    """Test batch retrieval over a list of queries."""
    retriever = LexicalRetriever(index_dir=mock_index)
    queries = [
        "AsyncScheduler",
        "LoRA Adapters",
        "",
        "unknown_nonexistent_term_xyz",
    ]
    batch_res = retriever.retrieve_batch(queries, k=3, show_progress=False)

    assert len(batch_res) == 4
    # Query 1 matches scheduler
    assert len(batch_res[0]) >= 1
    assert "scheduler.py" in batch_res[0][0].file_path

    # Query 2 matches lora
    assert len(batch_res[1]) >= 1
    assert "lora.md" in batch_res[1][0].file_path

    # Query 3 was empty string
    assert batch_res[2] == []


def test_retriever_missing_index(tmp_path: Path) -> None:
    """Initializing retriever with non-existent directory raises error."""
    missing_dir = tmp_path / "non_existent"
    with pytest.raises(FileNotFoundError):
        LexicalRetriever(index_dir=missing_dir)
