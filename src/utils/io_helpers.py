"""I/O helper utilities for safe reading and writing of project files."""

import json
from pathlib import Path
from typing import Any, Optional


def normalize_corpus_path(path: str | Path) -> str:
    """Normalize relative corpus path to use forward slashes and no dot.

    Args:
        path: Path to normalize.

    Returns:
        Normalized relative path string (e.g. data/raw/vllm-0.10.1/...).
    """
    clean_path = str(path).replace("\\", "/")
    if clean_path.startswith("./"):
        clean_path = clean_path[2:]
    return clean_path


def safe_read_json(file_path: str | Path) -> Optional[Any]:
    """Safely load and parse a JSON file with error handling.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Parsed JSON data, or None if reading or parsing failed.
    """
    path = Path(file_path)
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def safe_write_json(file_path: str | Path, data: Any, indent: int = 2) -> bool:
    """Safely serialize and write data to a JSON file.

    Ensures parent directories exist and uses a context manager.

    Args:
        file_path: Destination path for the JSON file.
        data: Data serializable to JSON.
        indent: Indentation level for pretty-printing.

    Returns:
        True if written successfully, False otherwise.
    """
    path = Path(file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except (OSError, TypeError):
        return False
