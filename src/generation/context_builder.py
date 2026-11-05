"""Context builder for extracting and assembling grounded LLM context.

Reads exact character slices from corpus files based on MinimalSource
objects, formats structured context blocks, and enforces token limits.
"""

from pathlib import Path
from typing import Any

from src.models import MinimalSource


class ContextBuilder:
    """Extracts raw text slices and builds context within a token budget."""

    def __init__(
        self,
        max_context_tokens: int = 1024,
        max_sources: int = 5,
        tokenizer: Any | None = None,
        base_dir: str | Path | None = None,
    ) -> None:
        """Initialize the ContextBuilder.

        Args:
            max_context_tokens: Maximum token budget for context blocks.
            max_sources: Maximum number of top sources to include in prompt.
            tokenizer: Optional HF tokenizer used for token counting.
            base_dir: Optional base directory to resolve relative paths.
        """
        self.max_context_tokens = max_context_tokens
        self.max_sources = max_sources
        self.tokenizer = tokenizer
        self.base_dir = Path(base_dir) if base_dir else None
        self._file_cache: dict[str, str] = {}

    def count_tokens(self, text: str) -> int:
        """Estimate or count the number of tokens in a text string.

        Args:
            text: Text to count.

        Returns:
            Integer count of tokens.
        """
        if not text:
            return 0
        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except (TypeError, ValueError, AttributeError, RuntimeError):
                return max(1, len(text) // 4)
        return max(1, len(text) // 4)

    def _read_file_content(self, file_path_str: str) -> str | None:
        """Read and cache file content with encoding error tolerance.

        Args:
            file_path_str: Path string to the file.

        Returns:
            Decoded file content or None if file cannot be read.
        """
        if file_path_str in self._file_cache:
            return self._file_cache[file_path_str]

        path = Path(file_path_str)
        if not path.is_file() and self.base_dir:
            path = self.base_dir / file_path_str

        if not path.is_file():
            return None

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            self._file_cache[file_path_str] = content
            return content
        except OSError:
            return None

    def extract_snippet(self, source: MinimalSource) -> str | None:
        """Extract the exact character range text for a MinimalSource.

        Args:
            source: Source specification containing file path and char bounds.

        Returns:
            The raw text snippet, or None if reading fails or bounds invalid.
        """
        content = self._read_file_content(source.file_path)
        if content is None:
            return None

        content_len = len(content)
        start = max(0, source.first_character_index)
        end = min(content_len, source.last_character_index)

        if start >= end or start >= content_len:
            return None

        return content[start:end]

    def build_context(self, sources: list[MinimalSource]) -> str:
        """Construct formatted context string from top sources within budget.

        Args:
            sources: List of MinimalSource objects in ranking priority order.

        Returns:
            Formatted context string, or empty string if no valid sources.
        """
        if not sources:
            return ""

        context_blocks: list[str] = []
        accumulated_tokens = 0

        for source in sources[: self.max_sources]:
            snippet = self.extract_snippet(source)
            if not snippet or not snippet.strip():
                continue

            header = (
                f"--- Source: {source.file_path} "
                f"[chars {source.first_character_index}:"
                f"{source.last_character_index}] ---"
            )
            block = f"{header}\n{snippet}"
            block_tokens = self.count_tokens(block)

            if accumulated_tokens + block_tokens <= self.max_context_tokens:
                context_blocks.append(block)
                accumulated_tokens += block_tokens
            else:
                if not context_blocks:
                    ratio = (
                        len(block) / block_tokens if block_tokens > 0 else 4.0
                    )
                    char_limit = int(self.max_context_tokens * ratio)
                    truncated = snippet[: max(0, char_limit)].strip()
                    if truncated:
                        context_blocks.append(f"{header}\n{truncated}")
                break

        return "\n\n".join(context_blocks)
