"""Unit tests for the BM25 index builder and persistence."""

from pathlib import Path

from src.indexing.indexer import IndexBuilder, build_index
from src.utils.io_helpers import safe_read_json


def test_index_builder_empty_corpus(tmp_path: Path) -> None:
    """Indexing an empty directory should succeed with 0 chunks."""
    corpus_dir = tmp_path / "empty_corpus"
    corpus_dir.mkdir()
    output_dir = tmp_path / "processed"

    builder = IndexBuilder(max_chunk_size=1000)
    total_chunks, duration = builder.build_and_save(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        show_progress=False,
    )

    assert total_chunks == 0
    assert duration >= 0.0
    assert (output_dir / "metadata.json").is_file()
    meta = safe_read_json(output_dir / "metadata.json")
    assert meta == []


def test_index_builder_mock_corpus(tmp_path: Path) -> None:
    """Indexing a mock corpus creates bm25_index and metadata.json."""
    corpus_dir = tmp_path / "raw"
    corpus_dir.mkdir()

    # Create mock Python and Markdown files
    py_file = corpus_dir / "sample.py"
    py_file.write_text(
        "def compute_features(x: int) -> int:\n"
        "    return x * 42\n",
        encoding="utf-8",
    )

    md_file = corpus_dir / "guide.md"
    md_file.write_text(
        "# Setup Guide\n\n"
        "This is an installation guide for the system.\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "processed"

    total_chunks, duration = build_index(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        max_chunk_size=1000,
        show_progress=False,
    )

    assert total_chunks >= 2
    assert duration > 0.0

    # Verify persisted artifacts
    bm25_dir = output_dir / "bm25_index"
    assert bm25_dir.is_dir()
    assert (bm25_dir / "params.index.json").is_file()
    assert (bm25_dir / "vocab.index.json").is_file()

    # Verify metadata.json
    meta = safe_read_json(output_dir / "metadata.json")
    assert isinstance(meta, list)
    assert len(meta) == total_chunks
    for entry in meta:
        assert "file_path" in entry
        assert "first_character_index" in entry
        assert "last_character_index" in entry
        assert entry["last_character_index"] > entry["first_character_index"]

    # Verify index_info.json
    info = safe_read_json(output_dir / "index_info.json")
    assert isinstance(info, dict)
    assert info["total_chunks"] == total_chunks
    assert info["max_chunk_size"] == 1000
