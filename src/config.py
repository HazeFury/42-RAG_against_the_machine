"""Configuration and constants for the RAG system."""

from pathlib import Path

# Character limits
DEFAULT_MAX_CHUNK_SIZE: int = 2000

# Retrieval defaults
DEFAULT_TOP_K: int = 5
DEFAULT_DATASET_TOP_K: int = 10
DEFAULT_IOU_THRESHOLD: float = 0.05
EVALUATION_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)

# Paths defaults
DEFAULT_CORPUS_DIR: Path = Path("data/raw")
DEFAULT_PROCESSED_DIR: Path = Path("data/processed")
DEFAULT_OUTPUT_DIR: Path = Path("data/output")
DEFAULT_DATASET_DIR: Path = Path("data/datasets")

# Target model
DEFAULT_MODEL_NAME: str = "Qwen/Qwen3-0.6B"
