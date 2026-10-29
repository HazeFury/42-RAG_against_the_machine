"""RAG Pipeline orchestrator connecting indexing, retrieval, and generation."""

from pathlib import Path
from typing import Optional
import uuid

from tqdm import tqdm

from src.evaluation.evaluator import (
    evaluate_files,
    format_evaluation_report,
)
from src.generation.context_builder import ContextBuilder
from src.generation.llm_generator import LLMGenerator
from src.indexing.indexer import build_index
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.retrieval.lexical_retriever import LexicalRetriever
from src.utils.io_helpers import safe_read_json, safe_write_json


class RagPipeline:
    """Central orchestrator for the RAG system stages."""

    def __init__(self) -> None:
        """Initialize the pipeline with lazy loaded components."""
        self._retriever: Optional[LexicalRetriever] = None
        self._retriever_index_dir: Optional[Path] = None
        self._generator: Optional[LLMGenerator] = None
        self._context_builder: Optional[ContextBuilder] = None

    def _get_generator(
        self, model_name: str = "Qwen/Qwen3-0.6B"
    ) -> LLMGenerator:
        """Get or initialize cached LLM generator.

        Args:
            model_name: Hugging Face model identifier.

        Returns:
            LLMGenerator instance.
        """
        if self._generator is None or self._generator.model_name != model_name:
            self._generator = LLMGenerator(model_name=model_name)
        return self._generator

    def _get_context_builder(
        self, model_name: str = "Qwen/Qwen3-0.6B"
    ) -> ContextBuilder:
        """Get or initialize ContextBuilder with model tokenizer.

        Args:
            model_name: Hugging Face model identifier.

        Returns:
            ContextBuilder instance.
        """
        if self._context_builder is None:
            generator = self._get_generator(model_name=model_name)
            tokenizer = generator.get_tokenizer()
            self._context_builder = ContextBuilder(tokenizer=tokenizer)
        return self._context_builder

    def _get_retriever(self, index_dir: str | Path) -> LexicalRetriever:
        """Get or initialize cached in-memory retriever for index directory.

        Args:
            index_dir: Directory containing index and metadata.

        Returns:
            LexicalRetriever instance.
        """
        path = Path(index_dir)
        if self._retriever is None or self._retriever_index_dir != path:
            self._retriever = LexicalRetriever(index_dir=path)
            self._retriever_index_dir = path
        return self._retriever

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
        total_chunks, elapsed = build_index(
            corpus_dir=corpus_dir,
            output_dir=output_dir,
            max_chunk_size=max_chunk_size,
            show_progress=True,
        )
        # Reset cached retriever so next search reloads new index
        self._retriever = None
        self._retriever_index_dir = None

        print(
            f"Ingestion complete! Indexed {total_chunks} chunks "
            f"under {output_dir}/ in {elapsed:.2f}s"
        )
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
        retriever = self._get_retriever(index_dir=index_dir)
        return retriever.retrieve(query=query, k=k)

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
        raw_data = safe_read_json(dataset_path)
        if raw_data is None:
            return None

        try:
            dataset = RagDataset.model_validate(raw_data)
        except Exception:
            return None

        retriever = self._get_retriever(index_dir=index_dir)

        questions: list[str] = []
        question_ids: list[str] = []
        for q in dataset.rag_questions:
            if isinstance(q, dict):
                questions.append(str(q.get("question", "")))
                question_ids.append(str(q.get("question_id", "")))
            else:
                questions.append(str(getattr(q, "question", "")))
                question_ids.append(str(getattr(q, "question_id", "")))

        batch_sources = retriever.retrieve_batch(
            queries=questions,
            k=k,
            show_progress=True,
        )

        search_results: list[MinimalSearchResults] = []
        for q_id, q_text, sources in zip(
            question_ids, questions, batch_sources
        ):
            search_results.append(
                MinimalSearchResults(
                    question_id=q_id,
                    question=q_text,
                    retrieved_sources=sources,
                )
            )

        student_res = StudentSearchResults(
            search_results=search_results,
            k=k,
        )

        dataset_filename = Path(dataset_path).name
        out_file = Path(save_directory) / dataset_filename
        success = safe_write_json(out_file, student_res.model_dump())
        if not success:
            return None

        return str(out_file)

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
        if not query or not query.strip() or k <= 0:
            return None

        sources = self.search(query=query, k=k, index_dir=index_dir)
        generator = self._get_generator(model_name=model_name)
        context_builder = self._get_context_builder(model_name=model_name)

        context = context_builder.build_context(sources)
        generated_answer = generator.generate_answer(
            question=query, context=context
        )

        return MinimalAnswer(
            question_id=str(uuid.uuid4()),
            question=query,
            retrieved_sources=sources,
            answer=generated_answer,
        )

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
        raw_data = safe_read_json(student_search_results_path)
        if raw_data is None:
            return None

        try:
            student_results = StudentSearchResults.model_validate(raw_data)
        except Exception:
            return None

        generator = self._get_generator(model_name=model_name)
        context_builder = self._get_context_builder(model_name=model_name)

        answers: list[MinimalAnswer] = []
        for item in tqdm(
            student_results.search_results,
            desc="Generating answers",
            unit="q",
        ):
            context = context_builder.build_context(item.retrieved_sources)
            gen_text = generator.generate_answer(
                question=item.question, context=context
            )
            answers.append(
                MinimalAnswer(
                    question_id=item.question_id,
                    question=item.question,
                    retrieved_sources=item.retrieved_sources,
                    answer=gen_text,
                )
            )

        output_obj = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=student_results.k,
        )

        dataset_filename = Path(student_search_results_path).name
        out_file = Path(save_directory) / dataset_filename
        success = safe_write_json(out_file, output_obj.model_dump())
        if not success:
            return None

        return str(out_file)

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
