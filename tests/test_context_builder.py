"""Unit tests for ContextBuilder in src/generation/context_builder.py."""

from pathlib import Path

from src.generation.context_builder import ContextBuilder
from src.models import MinimalSource


class DummyTokenizer:
    """Mock tokenizer for deterministic token budgeting tests."""

    def encode(
        self, text: str, add_special_tokens: bool = False
    ) -> list[int]:
        """Split text by words to simulate tokenization.

        Args:
            text: Text to split.
            add_special_tokens: Unused flag.

        Returns:
            List of dummy token IDs.
        """
        return [hash(word) for word in text.split()]


def test_context_builder_extract_snippet(tmp_path: Path) -> None:
    """Test extracting exact character slices from files."""
    test_file = tmp_path / "sample.py"
    content = "0123456789abcdefghijklmnopqrstuvwxyz"
    test_file.write_text(content, encoding="utf-8")

    builder = ContextBuilder()
    source = MinimalSource(
        file_path=str(test_file),
        first_character_index=10,
        last_character_index=16,
    )
    snippet = builder.extract_snippet(source)
    assert snippet == "abcdef"
    assert snippet == content[10:16]


def test_context_builder_bounds_handling(tmp_path: Path) -> None:
    """Test out-of-bounds, negative, or inverted character ranges."""
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Hello World", encoding="utf-8")

    builder = ContextBuilder()

    # Negative first_character_index clamped to 0
    src1 = MinimalSource(
        file_path=str(test_file),
        first_character_index=-5,
        last_character_index=5,
    )
    assert builder.extract_snippet(src1) == "Hello"

    # Inverted bounds returns None
    src2 = MinimalSource(
        file_path=str(test_file),
        first_character_index=8,
        last_character_index=4,
    )
    assert builder.extract_snippet(src2) is None

    # Missing file returns None
    src3 = MinimalSource(
        file_path=str(tmp_path / "missing.py"),
        first_character_index=0,
        last_character_index=10,
    )
    assert builder.extract_snippet(src3) is None


def test_context_builder_empty_sources() -> None:
    """Test build_context with empty list returns empty string."""
    builder = ContextBuilder()
    assert builder.build_context([]) == ""


def test_context_builder_token_budgeting(tmp_path: Path) -> None:
    """Verify context stops adding snippets when budget is exceeded."""
    f1 = tmp_path / "doc1.md"
    f2 = tmp_path / "doc2.md"
    f3 = tmp_path / "doc3.md"

    f1.write_text("word " * 10, encoding="utf-8")
    f2.write_text("word " * 10, encoding="utf-8")
    f3.write_text("word " * 10, encoding="utf-8")

    tokenizer = DummyTokenizer()
    # A source block has header (~5 words) + 10 words ~ 15 tokens
    # Budget of 25 allows block 1 (~15 tokens) but not block 1 + block 2 (~30)
    builder = ContextBuilder(max_context_tokens=25, tokenizer=tokenizer)

    s1 = MinimalSource(
        file_path=str(f1),
        first_character_index=0,
        last_character_index=len(f1.read_text()),
    )
    s2 = MinimalSource(
        file_path=str(f2),
        first_character_index=0,
        last_character_index=len(f2.read_text()),
    )
    s3 = MinimalSource(
        file_path=str(f3),
        first_character_index=0,
        last_character_index=len(f3.read_text()),
    )

    context = builder.build_context([s1, s2, s3])
    assert f"Source: {f1}" in context
    assert f"Source: {f2}" not in context
    assert f"Source: {f3}" not in context


def test_context_builder_heuristic_count() -> None:
    """Test fallback token counting without tokenizer."""
    builder = ContextBuilder(tokenizer=None)
    assert builder.count_tokens("") == 0
    assert builder.count_tokens("12345678") == 2
