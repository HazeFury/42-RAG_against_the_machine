"""Corpus loader module for discovering and reading codebase files."""

import hashlib
import os
from pathlib import Path
from typing import Iterator, Optional

from src.utils.io_helpers import normalize_corpus_path

# Standard extensions considered useful text / code files
SUPPORTED_EXTENSIONS: set[str] = {
    ".py",
    ".md",
    ".rst",
    ".txt",
    ".c",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".yaml",
    ".yml",
    ".toml",
    ".sh",
}

# Supported standalone file names without extensions
SUPPORTED_FILENAMES: set[str] = {
    "Dockerfile",
}

# Directories to always skip during corpus discovery
IGNORED_DIRECTORIES: set[str] = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    "egg-info",
}


def compute_file_hash(content: str) -> str:
    """Compute SHA-256 hash of a file's string content.

    Args:
        content: Raw text content of the file.

    Returns:
        Hexadecimal SHA-256 string.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def is_indexable_file(file_path: Path) -> bool:
    """Check whether a given file path should be ingested into the corpus.

    Args:
        file_path: Path object pointing to a file.

    Returns:
        True if the file matches supported extensions/names and is not ignored.
    """
    filename = file_path.name

    # Skip hidden files
    if filename.startswith("."):
        return False

    # Check standalone filenames like Dockerfile
    if filename in SUPPORTED_FILENAMES:
        return True

    # Check extension
    suffix = file_path.suffix.lower()
    return suffix in SUPPORTED_EXTENSIONS


def read_file_safely(file_path: Path | str) -> Optional[str]:
    """Read a text file strictly as UTF-8.

    Args:
        file_path: Path to the target file.

    Returns:
        String content of the file, or None if reading or decoding failed.
    """
    path = Path(file_path)
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (UnicodeDecodeError, OSError):
        return None


def discover_corpus_files(
    corpus_dir: str | Path = "data/raw",
) -> list[str]:
    """Recursively discover all indexable file paths within corpus directory.

    Paths are strictly normalized relative to the project root, using forward
    slashes and without leading './'.

    Args:
        corpus_dir: Directory containing raw codebase to ingest.

    Returns:
        Sorted list of normalized relative file paths.
    """
    root_path = Path(corpus_dir)
    if not root_path.exists():
        return []

    discovered: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        # Prune ignored directories in-place
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".") and d not in IGNORED_DIRECTORIES
        ]

        for fname in filenames:
            fpath = Path(dirpath) / fname
            if is_indexable_file(fpath):
                norm_path = normalize_corpus_path(fpath)
                discovered.append(norm_path)

    discovered.sort()
    return discovered


def load_corpus(
    corpus_dir: str | Path = "data/raw",
) -> Iterator[tuple[str, str]]:
    """Yield pairs of (normalized_file_path, raw_content) from corpus.

    Args:
        corpus_dir: Root directory of raw files.

    Yields:
        Tuples of (relative_path, content_string).
    """
    file_paths = discover_corpus_files(corpus_dir=corpus_dir)
    for path_str in file_paths:
        content = read_file_safely(path_str)
        if content is not None and len(content.strip()) > 0:
            yield path_str, content
