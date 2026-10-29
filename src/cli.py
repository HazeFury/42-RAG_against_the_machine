"""Command-Line Interface for the RAG system using Python Fire."""

from pathlib import Path
import sys

from src.pipeline import RagPipeline


class RAGCLI:
    """Command-line interface exposing the 6 mandatory RAG commands."""

    def __init__(self) -> None:
        """Initialize CLI with underlying pipeline."""
        self.pipeline = RagPipeline()

    def index(
        self,
        max_chunk_size: int = 2000,
        corpus_dir: str = "data/raw",
        output_dir: str = "data/processed",
    ) -> None:
        """Ingest codebase and build searchable index.

        Args:
            max_chunk_size: Maximum characters per chunk (<= 2000).
            corpus_dir: Path to directory containing raw source files.
            output_dir: Path to directory where index will be stored.
        """
        try:
            if max_chunk_size <= 0:
                print(
                    "Error: max_chunk_size must be a positive integer.",
                    file=sys.stderr,
                )
                return

            if not Path(corpus_dir).exists():
                print(
                    f"Error: Corpus directory '{corpus_dir}' does not exist.",
                    file=sys.stderr,
                )
                return

            self.pipeline.index(
                max_chunk_size=max_chunk_size,
                corpus_dir=corpus_dir,
                output_dir=output_dir,
            )
        except Exception as e:
            print(f"Error during indexing: {e}", file=sys.stderr)

    def search(
        self,
        query: str = "",
        k: int = 5,
        index_dir: str = "data/processed",
    ) -> None:
        """Search the index and display top-k source locations.

        Args:
            query: Question or keywords to search for.
            k: Number of sources to retrieve.
            index_dir: Path to persisted index directory.
        """
        try:
            if not query or not query.strip():
                print("No search query provided. Returning 0 results.")
                return

            if k <= 0:
                print("Requested k <= 0. Returning 0 results.")
                return

            if not Path(index_dir).exists():
                print(
                    f"Error: Index directory '{index_dir}' not found. "
                    "Run 'index' command first.",
                    file=sys.stderr,
                )
                return

            sources = self.pipeline.search(
                query=query,
                k=k,
                index_dir=index_dir,
            )
            if not sources:
                print(f"No results found for: {query}")
                return

            for src in sources:
                print(
                    f"{src.file_path} "
                    f"[{src.first_character_index}:{src.last_character_index}]"
                )
        except Exception as e:
            print(f"Error during search: {e}", file=sys.stderr)

    def search_dataset(
        self,
        dataset_path: str = "",
        k: int = 10,
        save_directory: str = "data/output/search_results",
        index_dir: str = "data/processed",
    ) -> None:
        """Execute search in batch for an entire dataset.

        Args:
            dataset_path: Path to dataset JSON file.
            k: Number of sources to retrieve per question.
            save_directory: Output directory for the results file.
            index_dir: Path to persisted index directory.
        """
        try:
            if not dataset_path or not Path(dataset_path).is_file():
                print(
                    f"Error: Dataset file '{dataset_path}' not found.",
                    file=sys.stderr,
                )
                return

            if k <= 0:
                print("Error: k must be greater than 0.", file=sys.stderr)
                return

            out_path = self.pipeline.search_dataset(
                dataset_path=dataset_path,
                k=k,
                save_directory=save_directory,
                index_dir=index_dir,
            )
            if out_path:
                print(f"Saved student_search_results to {out_path}")
        except Exception as e:
            print(f"Error during dataset search: {e}", file=sys.stderr)

    def answer(
        self,
        query: str = "",
        k: int = 5,
        index_dir: str = "data/processed",
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """Answer a single query using retrieved context and local LLM.

        Args:
            query: User question.
            k: Number of sources to retrieve.
            index_dir: Path to persisted index directory.
            model_name: Model identifier on Hugging Face.
        """
        try:
            if not query or not query.strip():
                print("No query provided.")
                return

            if k <= 0:
                print("Requested k <= 0. Cannot retrieve context.")
                return

            ans = self.pipeline.answer(
                query=query,
                k=k,
                index_dir=index_dir,
                model_name=model_name,
            )
            if ans is None:
                print("Error: Failed to generate answer.", file=sys.stderr)
                return

            print(f"Question: {ans.question}\n")
            print("Retrieved sources:")
            for i, src in enumerate(ans.retrieved_sources, 1):
                print(
                    f"  {i}. {src.file_path} "
                    f"[{src.first_character_index}:{src.last_character_index}]"
                )
            print(f"\nAnswer:\n{ans.answer}")
        except Exception as e:
            print(f"Error during answering: {e}", file=sys.stderr)

    def answer_dataset(
        self,
        student_search_results_path: str = "",
        save_directory: str = "data/output/search_results_and_answer",
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """Generate answers in batch from student search results JSON file.

        Args:
            student_search_results_path: Path to StudentSearchResults JSON.
            save_directory: Output directory for generated answers JSON.
            model_name: Model identifier on Hugging Face.
        """
        try:
            if (
                not student_search_results_path
                or not Path(student_search_results_path).is_file()
            ):
                print(
                    "Error: Student search results file "
                    f"'{student_search_results_path}' not found.",
                    file=sys.stderr,
                )
                return

            out_path = self.pipeline.answer_dataset(
                student_search_results_path=student_search_results_path,
                save_directory=save_directory,
                model_name=model_name,
            )
            if out_path:
                print(f"Saved student_search_results_and_answer to {out_path}")
        except Exception as e:
            print(
                f"Error during dataset answer generation: {e}",
                file=sys.stderr,
            )

    def evaluate(
        self,
        student_search_results_path: str = "",
        dataset_path: str = "",
    ) -> None:
        """Compute and display Recall@k against a ground-truth dataset.

        Args:
            student_search_results_path: Path to StudentSearchResults JSON.
            dataset_path: Path to ground-truth RagDataset JSON.
        """
        try:
            if (
                not student_search_results_path
                or not Path(student_search_results_path).is_file()
            ):
                print(
                    "Error: Search results file "
                    f"'{student_search_results_path}' not found.",
                    file=sys.stderr,
                )
                return

            if not dataset_path or not Path(dataset_path).is_file():
                print(
                    f"Error: Ground-truth dataset '{dataset_path}' not found.",
                    file=sys.stderr,
                )
                return

            report = self.pipeline.evaluate(
                student_search_results_path=student_search_results_path,
                dataset_path=dataset_path,
            )
            if report is None:
                print(
                    "Error: Failed to evaluate files. Ensure JSON formats "
                    "match StudentSearchResults and RagDataset schemas.",
                    file=sys.stderr,
                )
                return

            print(report)
        except Exception as e:
            print(f"Error during evaluation: {e}", file=sys.stderr)
