"""Code-aware tokenizer for BM25 indexing and retrieval.

Implements identifier splitting for snake_case and camelCase,
path token injection, and technical keyword preservation.
"""

from pathlib import Path
import re
from typing import Optional

# Standard English conversational stopwords to filter out
STOPWORDS: set[str] = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "her",
    "here",
    "here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "let's",
    "me",
    "more",
    "most",
    "mustn't",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "weren't",
    "what",
    "what's",
    "when",
    "when's",
    "where",
    "where's",
    "which",
    "while",
    "who",
    "who's",
    "whom",
    "why",
    "why's",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

# Standard directory names and file extensions to ignore in path injection
IGNORED_PATH_PARTS: set[str] = {
    "data",
    "raw",
    "vllm",
    "vllm-0.10.1",
    "src",
    "tests",
    "py",
    "md",
    "rst",
    "txt",
    "json",
    "yaml",
    "yml",
}

# Regex to detect camelCase / PascalCase word boundaries
CAMEL_PATTERN = re.compile(
    r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+"
)


def split_identifier(token: str) -> list[str]:
    """Split an identifier into compound and sub-word tokens.

    Supports snake_case and camelCase/PascalCase, returning both the full
    identifier and the decomposed subwords.

    Args:
        token: Raw word or identifier string.

    Returns:
        List of lowercased tokens.
    """
    clean_token = token.strip("()[]{}<>,;:'\"`~!?*+-=/|\\")
    if not clean_token:
        return []

    tokens: list[str] = []

    # Handle snake_case
    if "_" in clean_token:
        clean_snake = re.sub(r"[^a-zA-Z0-9_]", "", clean_token).lower()
        if not re.search(r"[a-z0-9]", clean_snake):
            return []
        if clean_snake and clean_snake not in STOPWORDS:
            tokens.append(clean_snake)
        for part in clean_snake.split("_"):
            if (
                part
                and part not in STOPWORDS
                and part not in tokens
                and re.search(r"[a-z0-9]", part)
            ):
                tokens.append(part)
        return tokens

    # Handle camelCase / PascalCase
    camel_parts = CAMEL_PATTERN.findall(clean_token)
    if len(camel_parts) > 1:
        compound = "".join(camel_parts).lower()
        if compound and compound not in STOPWORDS:
            tokens.append(compound)
        for cp in camel_parts:
            cp_lower = cp.lower()
            if (
                cp_lower
                and cp_lower not in STOPWORDS
                and cp_lower not in tokens
            ):
                tokens.append(cp_lower)
        return tokens

    # Regular single word
    word_lower = clean_token.lower()
    if word_lower and word_lower not in STOPWORDS:
        tokens.append(word_lower)
    return tokens


def extract_path_tokens(file_path: str | Path) -> list[str]:
    """Extract meaningful keyword tokens from a relative file path.

    Args:
        file_path: Relative file path to extract keywords from.

    Returns:
        List of lowercased keyword tokens from the path.
    """
    clean_path = str(file_path).replace("\\", "/")
    parts = re.split(r"[/._\-]+", clean_path)
    tokens: list[str] = []

    for part in parts:
        p_lower = part.lower()
        if (
            p_lower
            and p_lower not in IGNORED_PATH_PARTS
            and p_lower not in STOPWORDS
            and not p_lower.isdigit()
            and len(p_lower) > 1
        ):
            subwords = split_identifier(part)
            for sw in subwords:
                if sw not in tokens and sw not in IGNORED_PATH_PARTS:
                    tokens.append(sw)

    return tokens


def tokenize(
    text: str,
    file_path: Optional[str | Path] = None,
    path_weight: int = 2,
) -> list[str]:
    """Tokenize text into BM25-ready tokens with optional path injection.

    Args:
        text: Input string (chunk content or query).
        file_path: Optional source file path for path token injection.
        path_weight: Repetition count for injected path tokens.

    Returns:
        List of processed tokens.
    """
    tokens: list[str] = []

    # Inject weighted path tokens if provided
    if file_path is not None and path_weight > 0:
        p_tokens = extract_path_tokens(file_path)
        for _ in range(path_weight):
            tokens.extend(p_tokens)

    # Split text into raw word units
    raw_words = re.findall(r"[a-zA-Z0-9_]+|[^\s\w]", text)
    for raw in raw_words:
        # Ignore standalone punctuation marks and standalone underscores
        if not re.search(r"[a-zA-Z0-9]", raw):
            continue
        sub_tokens = split_identifier(raw)
        tokens.extend(sub_tokens)

    return tokens
