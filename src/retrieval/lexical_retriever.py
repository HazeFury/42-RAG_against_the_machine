"""Lexical retrieval engine based on BM25 and spatial deduplication."""

import json
from pathlib import Path
from typing import Any, Optional

import bm25s
from tqdm import tqdm

from src.config import DEFAULT_MAX_CHUNK_SIZE, DEFAULT_PROCESSED_DIR
from src.indexing.tokenizer import tokenize
from src.models import MinimalSource


class LexicalRetriever:
    """In-memory retriever querying persisted BM25 index and chunk metadata."""

    def __init__(
        self,
        index_dir: str | Path = DEFAULT_PROCESSED_DIR,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    ) -> None:
        """Initialize retriever and load index and metadata into memory.

        Args:
            index_dir: Directory where bm25_index and metadata.json reside.
            max_chunk_size: Maximum allowed character span per returned source.
        """
        self.index_dir = Path(index_dir)
        self.max_chunk_size = max_chunk_size
        self._retriever: Optional[bm25s.BM25] = None
        self._metadata: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load BM25 index and chunk coordinates metadata from disk."""
        bm25_path = self.index_dir / "bm25_index"
        meta_path = self.index_dir / "metadata.json"

        if not bm25_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Index files missing under '{self.index_dir}'. "
                "Please run 'index' command first."
            )

        # Load chunk coordinates
        with open(meta_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        if not self._metadata:
            self._retriever = None
            return

        # Load BM25 index
        self._retriever = bm25s.BM25.load(str(bm25_path), mmap=False)

    @property
    def total_chunks(self) -> int:
        """Return total number of indexed chunks."""
        return len(self._metadata)

    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> list[MinimalSource]:
        """Retrieve top-k most relevant sources for a single question.

        Args:
            query: Question or keywords string.
            k: Number of sources to retrieve.

        Returns:
            List of MinimalSource objects ranked by relevance.
        """
        if k <= 0 or not self._metadata or self._retriever is None:
            return []

        clean_query = query.strip()
        if not clean_query:
            return []

        q_tokens = tokenize(clean_query)
        if not q_tokens:
            return []

        # Retrieve extra candidates for spatial deduplication and merging
        fetch_k = min(max(k * 3, 10), len(self._metadata))
        doc_indices, _ = self._retriever.retrieve(
            [q_tokens],
            k=fetch_k,
            show_progress=False,
        )

        candidate_indices: list[int] = [int(idx) for idx in doc_indices[0]]
        candidates = [self._metadata[idx] for idx in candidate_indices]

        return self._deduplicate_and_merge(candidates=candidates, k=k)

    def retrieve_batch(
        self,
        queries: list[str],
        k: int = 10,
        show_progress: bool = True,
    ) -> list[list[MinimalSource]]:
        """Retrieve top-k sources for a batch of questions.

        Args:
            queries: List of question strings.
            k: Number of sources per question.
            show_progress: Whether to show tqdm progress bar.

        Returns:
            List of lists of MinimalSource objects.
        """
        if not queries:
            return []
        if k <= 0 or not self._metadata or self._retriever is None:
            return [[] for _ in queries]

        # Tokenize all queries
        tokenized_batch: list[list[str]] = []
        valid_query_mask: list[bool] = []

        for q in queries:
            if not q.strip():
                tokenized_batch.append([])
                valid_query_mask.append(False)
            else:
                toks = tokenize(q.strip())
                tokenized_batch.append(toks)
                valid_query_mask.append(len(toks) > 0)

        # Retrieve valid queries in batch
        fetch_k = min(max(k * 3, 10), len(self._metadata))
        batch_results: list[list[MinimalSource]] = []

        q_iter = tqdm(
            range(len(queries)),
            desc="Searching dataset",
            disable=not show_progress,
        )

        for i in q_iter:
            if not valid_query_mask[i]:
                batch_results.append([])
                continue

            q_tokens = tokenized_batch[i]
            doc_indices, _ = self._retriever.retrieve(
                [q_tokens],
                k=fetch_k,
                show_progress=False,
            )
            candidate_indices = [int(idx) for idx in doc_indices[0]]
            candidates = [self._metadata[idx] for idx in candidate_indices]
            sources = self._deduplicate_and_merge(candidates=candidates, k=k)
            batch_results.append(sources)

        return batch_results

    def _deduplicate_and_merge(
        self,
        candidates: list[dict[str, Any]],
        k: int,
    ) -> list[MinimalSource]:
        """Merge overlapping chunks and deduplicate redundant candidates.

        Args:
            candidates: List of raw candidate metadata dicts.
            k: Desired maximum number of sources.

        Returns:
            List of MinimalSource models <= k.
        """
        accepted: list[dict[str, Any]] = []

        for cand in candidates:
            cand_fp = cand["file_path"]
            cand_start = cand["first_character_index"]
            cand_end = cand["last_character_index"]

            matched = False
            for existing in accepted:
                if existing["file_path"] == cand_fp:
                    ex_start = existing["first_character_index"]
                    ex_end = existing["last_character_index"]

                    # Check for overlap between existing and candidate
                    overlap = max(
                        0,
                        min(ex_end, cand_end) - max(ex_start, cand_start),
                    )
                    if overlap > 0:
                        union_start = min(ex_start, cand_start)
                        union_end = max(ex_end, cand_end)

                        # Merge if combined span respects 2000 character limit
                        if union_end - union_start <= self.max_chunk_size:
                            existing["first_character_index"] = union_start
                            existing["last_character_index"] = union_end
                        # Either merged or skipped as redundant overlap
                        matched = True
                        break

            if not matched:
                accepted.append(
                    {
                        "file_path": cand_fp,
                        "first_character_index": cand_start,
                        "last_character_index": cand_end,
                    }
                )

            if len(accepted) >= k:
                break

        # Convert to MinimalSource models ensuring hard bounds
        results: list[MinimalSource] = []
        for item in accepted[:k]:
            start = item["first_character_index"]
            end = item["last_character_index"]
            if end - start > self.max_chunk_size:
                end = start + self.max_chunk_size

            results.append(
                MinimalSource(
                    file_path=item["file_path"],
                    first_character_index=start,
                    last_character_index=end,
                )
            )

        return results
