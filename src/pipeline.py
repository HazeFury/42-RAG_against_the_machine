"""RAG Pipeline orchestrator connecting indexing, retrieval, and generation."""

from pathlib import Path
from typing import Optional

from src.evaluation.evaluator import (
    evaluate_files,
    format_evaluation_report,
)
from src.models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
)
from src.utils.io_helpers import safe_read_json


class RagPipeline:
    """Central orchestrator for the RAG system stages."""

    def __init__(self) -> None:
        """Initialize the pipeline."""
        pass

    def index(
        self,
        max_chunk_size: int = 2000,
        corpus_dir: str | Path = "data/raw",
        output_dir: str | Path = "data/processed",
    ) -> bool:
        """Build and persist the lexical index over the raw corpus.

        Args:
            max_chunk_size: Maximum allowed characters per chunk.
            corpus_dir: Directory containing raw codebase to ingest.
            output_dir: Directory where processed indices will be saved.

        Returns:
            True on success, False otherwise.
        """
        # Full indexing implementation will be plugged in on Day 2 & 3
        print(f"Indexing corpus from {corpus_dir} into {output_dir}...")
        print(f"Max chunk size: {max_chunk_size}")
        return True

    def search(
        self,
        query: str,
        k: int = 5,
        index_dir: str | Path = "data/processed",
    ) -> list[MinimalSource]:
        """Search the index for the top-k most relevant sources.

        Args:
            query: User search query.
            k: Number of sources to retrieve.
            index_dir: Directory containing persisted index.

        Returns:
            List of retrieved MinimalSource objects.
        """
        # Full retrieval implementation will be plugged in on Day 3
        return []

    def search_dataset(
        self,
        dataset_path: str | Path,
        k: int = 10,
        save_directory: str | Path = "data/output/search_results",
        index_dir: str | Path = "data/processed",
    ) -> Optional[str]:
        """Search index in batch for an entire question dataset.

        Args:
            dataset_path: Path to dataset JSON file.
            k: Number of sources to retrieve per question.
            save_directory: Directory where result JSON should be written.
            index_dir: Directory containing persisted index.

        Returns:
            Output file path on success, or None on failure.
        """
        # Full batch search implementation will be plugged in on Day 3
        return None

    def answer(
        self,
        query: str,
        k: int = 5,
        index_dir: str | Path = "data/processed",
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> Optional[MinimalAnswer]:
        """Retrieve sources and generate an answer for a single query.

        Args:
            query: User question.
            k: Number of sources to retrieve.
            index_dir: Directory containing persisted index.
            model_name: Hugging Face model identifier.

        Returns:
            MinimalAnswer object or None on failure.
        """
        # Full generation implementation will be plugged in on Day 5
        return None

    def answer_dataset(
        self,
        student_search_results_path: str | Path,
        save_directory: str | Path = "data/output/search_results_and_answer",
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> Optional[str]:
        """Generate answers in batch from precomputed search results.

        Args:
            student_search_results_path: Path to StudentSearchResults JSON.
            save_directory: Output directory for final JSON.
            model_name: Hugging Face model identifier.

        Returns:
            Output file path on success, or None on failure.
        """
        # Full batch answer implementation will be plugged in on Day 5
        return None

    def evaluate(
        self,
        student_search_results_path: str | Path,
        dataset_path: str | Path,
    ) -> Optional[str]:
        """Evaluate student search results against ground-truth dataset.

        Args:
            student_search_results_path: Path to StudentSearchResults JSON.
            dataset_path: Path to ground-truth RagDataset JSON.

        Returns:
            Formatted evaluation report string, or None on error.
        """
        res_data = safe_read_json(student_search_results_path)
        if res_data is None:
            return None

        try:
            student_res = StudentSearchResults.model_validate(res_data)
        except Exception:
            return None

        scores = evaluate_files(student_search_results_path, dataset_path)
        if scores is None:
            return None

        return format_evaluation_report(
            scores=scores,
            total_questions=len(student_res.search_results),
        )
