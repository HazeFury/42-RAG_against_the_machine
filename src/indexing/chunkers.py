"""Syntax and document-aware chunking strategies for code and documentation.

Implements two distinct chunking strategies:
1. Python Code Chunking: AST-based syntactic boundary chunking.
2. Markdown / Text Chunking: Hierarchical heading and paragraph chunking.
"""

import ast
from pathlib import Path
import re

from src.config import DEFAULT_MAX_CHUNK_SIZE
from src.models import Chunk


def slice_range_by_lines(
    content: str,
    start: int,
    end: int,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    overlap: int = 150,
) -> list[tuple[int, int]]:
    """Slice a character range into overlapping windows <= max_chunk_size.

    Prefers splitting at newline boundaries to avoid breaking lines of code
    or sentences in half.

    Args:
        content: The full raw document string.
        start: Starting character index of the range.
        end: Ending character index of the range.
        max_chunk_size: Maximum allowed length for each sub-span.
        overlap: Target character overlap between consecutive slices.

    Returns:
        List of (first_character_index, last_character_index) tuples.
    """
    if start >= end:
        return []
    if end - start <= max_chunk_size:
        return [(start, end)]

    spans: list[tuple[int, int]] = []
    curr = start

    while curr < end:
        chunk_end = min(curr + max_chunk_size, end)

        if chunk_end < end:
            # Try to break at a newline in the upper half of the window
            search_start = curr + max(1, max_chunk_size // 2)
            last_nl = content.rfind("\n", search_start, chunk_end)
            if last_nl != -1 and last_nl > curr:
                chunk_end = last_nl + 1

        # Strict hard cap enforcement
        if chunk_end - curr > max_chunk_size:
            chunk_end = curr + max_chunk_size

        spans.append((curr, chunk_end))

        if chunk_end >= end:
            break

        # Advance with overlap, ensuring strictly positive progress
        next_curr = max(curr + 1, chunk_end - overlap)
        next_nl = content.rfind("\n", curr, next_curr)
        if next_nl != -1 and next_nl >= curr:
            next_curr = next_nl + 1

        if next_curr <= curr:
            next_curr = curr + 1

        curr = next_curr

    return spans


def chunk_python_code(
    file_path: str,
    content: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[Chunk]:
    """Chunk Python source code along syntactic AST boundaries.

    Functions and classes are identified via AST. If a function or class
    exceeds max_chunk_size, it is sliced with line-based overlapping windows.
    If AST parsing fails, falls back to line-based sliding windows.

    Args:
        file_path: Relative corpus file path.
        content: Raw Python code string.
        max_chunk_size: Maximum chunk character length.

    Returns:
        List of Chunk objects satisfying all invariants.
    """
    if not content or not content.strip():
        return []

    lines = content.splitlines(keepends=True)
    line_starts: list[int] = [0]
    for line in lines:
        line_starts.append(line_starts[-1] + len(line))

    try:
        tree = ast.parse(content)
    except SyntaxError:
        spans = slice_range_by_lines(
            content=content,
            start=0,
            end=len(content),
            max_chunk_size=max_chunk_size,
        )
        return [
            Chunk(
                file_path=file_path,
                first_character_index=s,
                last_character_index=e,
                text=content[s:e],
            )
            for s, e in spans
        ]

    spans_list: list[tuple[int, int]] = []
    curr_pos = 0

    for node in tree.body:
        s_line = node.lineno
        decorators = getattr(node, "decorator_list", None)
        if decorators:
            s_line = min(s_line, *(d.lineno for d in decorators))
        e_line = getattr(node, "end_lineno", s_line)

        s_char = line_starts[s_line - 1]
        e_char = line_starts[min(e_line, len(lines))]

        # Handle any gap (comments, docstrings, spaces) before this node
        if s_char > curr_pos:
            gap = content[curr_pos:s_char]
            if gap.strip():
                spans_list.extend(
                    slice_range_by_lines(
                        content=content,
                        start=curr_pos,
                        end=s_char,
                        max_chunk_size=max_chunk_size,
                    )
                )

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans_list.extend(
                slice_range_by_lines(
                    content=content,
                    start=s_char,
                    end=e_char,
                    max_chunk_size=max_chunk_size,
                )
            )
        elif isinstance(node, ast.ClassDef):
            if e_char - s_char <= max_chunk_size:
                spans_list.append((s_char, e_char))
            else:
                methods = [
                    m
                    for m in node.body
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                if methods:
                    fm = methods[0]
                    fm_sline = fm.lineno
                    if getattr(fm, "decorator_list", None):
                        fm_sline = min(
                            fm_sline, *(d.lineno for d in fm.decorator_list)
                        )
                    pre_end = line_starts[fm_sline - 1]
                    if pre_end > s_char and content[s_char:pre_end].strip():
                        spans_list.extend(
                            slice_range_by_lines(
                                content=content,
                                start=s_char,
                                end=pre_end,
                                max_chunk_size=max_chunk_size,
                            )
                        )
                    for m in methods:
                        ms_line = m.lineno
                        if getattr(m, "decorator_list", None):
                            ms_line = min(
                                ms_line, *(d.lineno for d in m.decorator_list)
                            )
                        me_line = getattr(m, "end_lineno", ms_line)
                        ms_char = line_starts[ms_line - 1]
                        me_char = line_starts[min(me_line, len(lines))]
                        spans_list.extend(
                            slice_range_by_lines(
                                content=content,
                                start=ms_char,
                                end=me_char,
                                max_chunk_size=max_chunk_size,
                            )
                        )
                else:
                    spans_list.extend(
                        slice_range_by_lines(
                            content=content,
                            start=s_char,
                            end=e_char,
                            max_chunk_size=max_chunk_size,
                        )
                    )
        else:
            spans_list.extend(
                slice_range_by_lines(
                    content=content,
                    start=s_char,
                    end=e_char,
                    max_chunk_size=max_chunk_size,
                )
            )

        curr_pos = e_char

    # Handle any trailing content at end of file
    if curr_pos < len(content):
        trailing = content[curr_pos:]
        if trailing.strip():
            spans_list.extend(
                slice_range_by_lines(
                    content=content,
                    start=curr_pos,
                    end=len(content),
                    max_chunk_size=max_chunk_size,
                )
            )

    chunks: list[Chunk] = []
    for s, e in spans_list:
        if s < e and content[s:e].strip():
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=s,
                    last_character_index=e,
                    text=content[s:e],
                )
            )
    return chunks


def chunk_markdown_text(
    file_path: str,
    content: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[Chunk]:
    """Chunk Markdown and documentation by headings and paragraphs.

    Args:
        file_path: Relative corpus file path.
        content: Raw markdown text string.
        max_chunk_size: Maximum chunk character length.

    Returns:
        List of Chunk objects.
    """
    if not content or not content.strip():
        return []

    heading_pattern = re.compile(r"(?m)^(#{1,6}\s+.*)$")
    matches = list(heading_pattern.finditer(content))

    if not matches:
        spans = slice_range_by_lines(
            content=content,
            start=0,
            end=len(content),
            max_chunk_size=max_chunk_size,
        )
        return [
            Chunk(
                file_path=file_path,
                first_character_index=s,
                last_character_index=e,
                text=content[s:e],
            )
            for s, e in spans
            if s < e and content[s:e].strip()
        ]

    breakpoints = sorted(
        list(set([0] + [m.start() for m in matches] + [len(content)]))
    )
    spans_list: list[tuple[int, int]] = []

    for i in range(len(breakpoints) - 1):
        sec_start = breakpoints[i]
        sec_end = breakpoints[i + 1]
        if content[sec_start:sec_end].strip():
            spans_list.extend(
                slice_range_by_lines(
                    content=content,
                    start=sec_start,
                    end=sec_end,
                    max_chunk_size=max_chunk_size,
                )
            )

    return [
        Chunk(
            file_path=file_path,
            first_character_index=s,
            last_character_index=e,
            text=content[s:e],
        )
        for s, e in spans_list
        if s < e and content[s:e].strip()
    ]


def chunk_plain_text(
    file_path: str,
    content: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[Chunk]:
    """Fallback chunker for plain text, configuration and other files.

    Args:
        file_path: Relative corpus file path.
        content: Raw text string.
        max_chunk_size: Maximum chunk character length.

    Returns:
        List of Chunk objects.
    """
    if not content or not content.strip():
        return []

    spans = slice_range_by_lines(
        content=content,
        start=0,
        end=len(content),
        max_chunk_size=max_chunk_size,
    )
    return [
        Chunk(
            file_path=file_path,
            first_character_index=s,
            last_character_index=e,
            text=content[s:e],
        )
        for s, e in spans
        if s < e and content[s:e].strip()
    ]


def chunk_file(
    file_path: str | Path,
    content: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[Chunk]:
    """Dispatch chunking strategy according to the file extension.

    Args:
        file_path: Relative corpus file path.
        content: Raw text string of the file.
        max_chunk_size: Maximum allowed characters per chunk.

    Returns:
        List of Chunk objects.
    """
    norm_path = str(file_path).replace("\\", "/")
    if norm_path.endswith(".py"):
        return chunk_python_code(
            file_path=norm_path,
            content=content,
            max_chunk_size=max_chunk_size,
        )
    if norm_path.endswith((".md", ".rst", ".txt")):
        return chunk_markdown_text(
            file_path=norm_path,
            content=content,
            max_chunk_size=max_chunk_size,
        )
    return chunk_plain_text(
        file_path=norm_path,
        content=content,
        max_chunk_size=max_chunk_size,
    )
