"""Unit tests for CLI robustness and edge cases handling."""

from pathlib import Path
from src.cli import RAGCLI


def test_cli_index_edge_cases(capsys: object) -> None:
    """Index should reject negative chunk sizes or missing directories."""
    cli = RAGCLI()

    # Negative chunk size
    cli.index(max_chunk_size=-10)

    # Missing corpus directory
    cli.index(corpus_dir="non_existent_directory_xyz")


def test_cli_search_edge_cases() -> None:
    """Search should handle empty queries, k <= 0, or missing index."""
    cli = RAGCLI()

    # Empty query
    cli.search(query="")
    cli.search(query="   ")

    # k <= 0
    cli.search(query="vllm config", k=0)
    cli.search(query="vllm config", k=-5)

    # Missing index directory
    cli.search(query="valid query", k=5, index_dir="missing_index_dir_xyz")


def test_cli_search_dataset_edge_cases() -> None:
    """Search dataset should handle missing dataset files and invalid k."""
    cli = RAGCLI()

    # Missing dataset
    cli.search_dataset(dataset_path="missing_file.json", k=10)

    # Invalid k
    cli.search_dataset(dataset_path="pyproject.toml", k=0)


def test_cli_answer_edge_cases() -> None:
    """Answer should handle empty queries and invalid k gracefully."""
    cli = RAGCLI()

    # Empty query
    cli.answer(query="")

    # k <= 0
    cli.answer(query="How to run?", k=0)


def test_cli_answer_dataset_edge_cases() -> None:
    """Answer dataset should handle non-existent results file."""
    cli = RAGCLI()

    cli.answer_dataset(student_search_results_path="missing_results.json")


def test_cli_evaluate_edge_cases(tmp_path: Path) -> None:
    """Evaluate should handle missing or corrupted files gracefully."""
    cli = RAGCLI()

    # Missing files
    cli.evaluate(
        student_search_results_path="missing_1.json",
        dataset_path="missing_2.json",
    )

    # Corrupted / invalid JSON file
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{invalid json", encoding="utf-8")

    cli.evaluate(
        student_search_results_path=str(bad_json),
        dataset_path=str(bad_json),
    )
