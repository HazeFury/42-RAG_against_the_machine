"""BM25 indexing engine over code and documentation chunks."""

import json
from pathlib import Path
import time
from typing import Any

import bm25s
from tqdm import tqdm

from src.config import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_PROCESSED_DIR,
)
from src.indexing.chunkers import chunk_file
from src.indexing.corpus_loader import load_corpus
from src.indexing.tokenizer import tokenize
from src.models import Chunk


class IndexBuilder:
    """Orchestrates chunking, tokenization, and BM25 index persistence."""

    def __init__(
        self,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    ) -> None:
        """Initialize index builder with configuration.

        Args:
            max_chunk_size: Maximum allowed character length per chunk.
        """
        self.max_chunk_size = max_chunk_size

    def build_and_save(
        self,
        corpus_dir: str | Path = DEFAULT_CORPUS_DIR,
        output_dir: str | Path = DEFAULT_PROCESSED_DIR,
        show_progress: bool = True,
    ) -> tuple[int, float]:
        """Ingest corpus, tokenize chunks, and persist BM25 index and metadata.

        Args:
            corpus_dir: Directory containing raw codebase to ingest.
            output_dir: Directory where processed indices will be saved.
            show_progress: Whether to display tqdm progress bars.

        Returns:
            Tuple of (total_chunks_indexed, elapsed_seconds).
        """
        start_time = time.time()
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        chunks: list[Chunk] = []
        corpus_files = list(load_corpus(corpus_dir=corpus_dir))

        # 1. Chunk all discovered files
        chunk_iter = tqdm(
            corpus_files,
            desc="Chunking files",
            disable=not show_progress,
        )
        for file_path, content in chunk_iter:
            file_chunks = chunk_file(
                file_path=file_path,
                content=content,
                max_chunk_size=self.max_chunk_size,
            )
            chunks.extend(file_chunks)

        if not chunks:
            # Create empty placeholder if corpus is empty
            self._save_metadata(out_path, [])
            return 0, time.time() - start_time

        # 2. Tokenize chunks
        corpus_tokens: list[list[str]] = []
        token_iter = tqdm(
            chunks,
            desc="Tokenizing chunks",
            disable=not show_progress,
        )
        for c in token_iter:
            c_tokens = tokenize(
                text=c.text,
                file_path=c.file_path,
                path_weight=2,
            )
            corpus_tokens.append(c_tokens)

        # 3. Build BM25 index
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=show_progress)

        # 4. Save BM25 index and chunk metadata
        bm25_save_dir = out_path / "bm25_index"
        bm25_save_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(str(bm25_save_dir))

        metadata = [
            {
                "file_path": c.file_path,
                "first_character_index": c.first_character_index,
                "last_character_index": c.last_character_index,
            }
            for c in chunks
        ]
        self._save_metadata(out_path, metadata)

        # 5. Save index info
        elapsed = time.time() - start_time
        info: dict[str, Any] = {
            "max_chunk_size": self.max_chunk_size,
            "total_files": len(corpus_files),
            "total_chunks": len(chunks),
            "indexing_duration_seconds": round(elapsed, 2),
        }
        with open(out_path / "index_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        return len(chunks), elapsed

    @staticmethod
    def _save_metadata(
        output_dir: Path,
        metadata: list[dict[str, Any]],
    ) -> None:
        """Persist chunk coordinates metadata to JSON.

        Args:
            output_dir: Directory to store metadata.json.
            metadata: List of chunk coordinate dictionaries.
        """
        meta_file = output_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def build_index(
    corpus_dir: str | Path = DEFAULT_CORPUS_DIR,
    output_dir: str | Path = DEFAULT_PROCESSED_DIR,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    show_progress: bool = True,
) -> tuple[int, float]:
    """Convenience function to build and save the BM25 index.

    Args:
        corpus_dir: Directory containing raw codebase to ingest.
        output_dir: Directory where processed indices will be saved.
        max_chunk_size: Maximum allowed characters per chunk.
        show_progress: Whether to display tqdm progress bars.

    Returns:
        Tuple of (total_chunks_indexed, elapsed_seconds).
    """
    builder = IndexBuilder(max_chunk_size=max_chunk_size)
    return builder.build_and_save(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        show_progress=show_progress,
    )
