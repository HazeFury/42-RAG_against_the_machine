"""Unit tests for the syntax-aware and document-aware chunkers."""

from src.indexing.chunkers import (
    chunk_file,
    chunk_markdown_text,
    chunk_plain_text,
    chunk_python_code,
    slice_range_by_lines,
)


def test_slice_range_by_lines_small() -> None:
    """A range smaller than max_chunk_size returns a single slice."""
    content = "Hello world!\nLine 2.\n"
    spans = slice_range_by_lines(content, 0, len(content), max_chunk_size=100)
    assert spans == [(0, len(content))]


def test_slice_range_by_lines_multiline() -> None:
    """A range larger than max_chunk_size splits cleanly at newlines."""
    lines = [f"Line {i} with some extra padding text.\n" for i in range(100)]
    content = "".join(lines)
    max_size = 200
    spans = slice_range_by_lines(
        content, 0, len(content), max_chunk_size=max_size, overlap=40
    )

    assert len(spans) > 1
    for start, end in spans:
        assert 0 <= start < end <= len(content)
        assert end - start <= max_size
        # Verify slice ends at a newline if not the final slice
        if end < len(content):
            assert content[end - 1] == "\n"


def test_slice_range_by_lines_no_newlines() -> None:
    """A large block without newlines is hard-sliced within bounds."""
    content = "A" * 5000
    max_size = 500
    spans = slice_range_by_lines(
        content, 0, len(content), max_chunk_size=max_size, overlap=50
    )

    assert len(spans) > 1
    for start, end in spans:
        assert 0 <= start < end <= len(content)
        assert end - start <= max_size


def test_chunk_python_code_basic() -> None:
    """Test AST-based chunking on standard Python code."""
    code = (
        '"""Module docstring."""\n\n'
        "import os\n"
        "import sys\n\n"
        "GLOBAL_CONST = 42\n\n"
        "@decorator\n"
        "def sample_func(x: int) -> int:\n"
        '    """Function docstring."""\n'
        "    return x * 2\n\n"
        "class MyClass:\n"
        "    def method_one(self) -> None:\n"
        "        pass\n\n"
        "    def method_two(self) -> str:\n"
        '        return "done"\n'
    )
    chunks = chunk_python_code("test.py", code, max_chunk_size=500)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.file_path == "test.py"
        assert 0 <= chunk.first_character_index < chunk.last_character_index
        assert chunk.last_character_index <= len(code)
        assert chunk.span_length <= 500
        assert (
            code[chunk.first_character_index : chunk.last_character_index]
            == chunk.text
        )


def test_chunk_python_code_huge_function() -> None:
    """A function exceeding max_chunk_size is sliced without crashing."""
    lines = ["def massive_function() -> None:\n"]
    for i in range(200):
        lines.append(f"    var_{i} = calculate_something({i})\n")
    code = "".join(lines)
    max_size = 1000

    chunks = chunk_python_code("massive.py", code, max_chunk_size=max_size)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.span_length <= max_size
        assert (
            code[chunk.first_character_index : chunk.last_character_index]
            == chunk.text
        )


def test_chunk_python_code_empty_and_whitespace() -> None:
    """Empty or whitespace-only code returns no chunks."""
    assert chunk_python_code("empty.py", "") == []
    assert chunk_python_code("empty.py", "   \n\n  \t  ") == []


def test_chunk_python_code_syntax_error_fallback() -> None:
    """Invalid Python syntax falls back to line-based slicing cleanly."""
    invalid_code = "def invalid syntax (():\n" * 50
    chunks = chunk_python_code("invalid.py", invalid_code, max_chunk_size=200)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.span_length <= 200
        assert (
            invalid_code[
                chunk.first_character_index : chunk.last_character_index
            ]
            == chunk.text
        )


def test_chunk_markdown_text_structure() -> None:
    """Test Markdown chunking along headings."""
    doc = (
        "# Main Title\n\n"
        "Introduction paragraph with explanations.\n\n"
        "## Section 1: Setup\n\n"
        "Here are instructions to set up the project:\n"
        "- Step 1: install\n"
        "- Step 2: run\n\n"
        "### Subsection 1.1\n\n"
        "Details about subsection.\n\n"
        "## Section 2: Usage\n\n"
        "Examples of commands.\n"
    )
    chunks = chunk_markdown_text("README.md", doc, max_chunk_size=300)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.file_path == "README.md"
        assert chunk.span_length <= 300
        assert (
            doc[chunk.first_character_index : chunk.last_character_index]
            == chunk.text
        )


def test_chunk_markdown_text_no_headings() -> None:
    """Markdown without headings falls back to paragraph slicing."""
    text = "Paragraph one.\n\nParagraph two with more details.\n\n" * 20
    chunks = chunk_markdown_text("plain.md", text, max_chunk_size=400)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.span_length <= 400
        assert (
            text[chunk.first_character_index : chunk.last_character_index]
            == chunk.text
        )


def test_chunk_file_dispatcher() -> None:
    """Test chunk_file routing according to file extension."""
    py_code = "def hello(): pass\n"
    md_text = "# Title\nContent\n"
    txt_text = "Plain text\n"

    c_py = chunk_file("module.py", py_code, max_chunk_size=1000)
    c_md = chunk_file("doc.md", md_text, max_chunk_size=1000)
    c_txt = chunk_file("info.txt", txt_text, max_chunk_size=1000)
    c_plain = chunk_plain_text(
        "data.yaml", "key: value\n", max_chunk_size=1000
    )

    assert len(c_py) == 1
    assert len(c_md) == 1
    assert len(c_txt) == 1
    assert len(c_plain) == 1
    assert c_py[0].text == py_code
