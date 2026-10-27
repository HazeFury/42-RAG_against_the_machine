"""Data models for RAG system validation and serialization.

These models strictly implement the schema defined in Section VI.4
of the 42 subject specification.
"""

from typing import Union
import uuid

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Represents a single source location in the indexed corpus."""

    file_path: str
    first_character_index: int
    last_character_index: int

    @property
    def span_length(self) -> int:
        """Return the length of the character range."""
        return self.last_character_index - self.first_character_index


class UnansweredQuestion(BaseModel):
    """Represents a question without ground-truth answers or sources."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represents a question with ground-truth sources and answer."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represents a dataset of questions (answered or unanswered)."""

    rag_questions: list[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """Represents the search results for a single question."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represents the search results and generated answer for a question."""

    answer: str


class StudentSearchResults(BaseModel):
    """Container for search results evaluated by moulinette."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Container for search results and answers evaluated by moulinette."""

    search_results: list[MinimalAnswer]
    k: int
