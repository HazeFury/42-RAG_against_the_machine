"""Unit tests for Pydantic data models."""

import pytest
from pydantic import ValidationError

from src.models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)


def test_minimal_source() -> None:
    """Test MinimalSource model fields and properties."""
    src = MinimalSource(
        file_path="data/raw/vllm-0.10.1/test.py",
        first_character_index=10,
        last_character_index=50,
    )
    assert src.file_path == "data/raw/vllm-0.10.1/test.py"
    assert src.first_character_index == 10
    assert src.last_character_index == 50
    assert src.span_length == 40

    with pytest.raises(ValidationError):
        MinimalSource.model_validate({"file_path": "a.py"})


def test_unanswered_question() -> None:
    """Test default UUID generation for UnansweredQuestion."""
    q1 = UnansweredQuestion(question="How to configure vLLM?")
    assert q1.question == "How to configure vLLM?"
    assert len(q1.question_id) > 0

    q2 = UnansweredQuestion(
        question_id="custom_id",
        question="Another question?",
    )
    assert q2.question_id == "custom_id"


def test_answered_question() -> None:
    """Test AnsweredQuestion includes sources and answer."""
    src = MinimalSource(
        file_path="data/raw/vllm-0.10.1/docs/test.md",
        first_character_index=0,
        last_character_index=100,
    )
    aq = AnsweredQuestion(
        question_id="q1",
        question="What is this?",
        sources=[src],
        answer="It is documentation.",
    )
    assert aq.answer == "It is documentation."
    assert len(aq.sources) == 1
    assert aq.sources[0].file_path == "data/raw/vllm-0.10.1/docs/test.md"


def test_rag_dataset_mixed() -> None:
    """Test RagDataset parses both answered and unanswered questions."""
    raw_data = {
        "rag_questions": [
            {
                "question_id": "q1",
                "question": "Unanswered question?",
            },
            {
                "question_id": "q2",
                "question": "Answered question?",
                "sources": [
                    {
                        "file_path": "a.py",
                        "first_character_index": 0,
                        "last_character_index": 10,
                    }
                ],
                "answer": "This is answered.",
            },
        ]
    }
    dataset = RagDataset.model_validate(raw_data)
    assert len(dataset.rag_questions) == 2
    assert isinstance(dataset.rag_questions[0], UnansweredQuestion)
    assert isinstance(dataset.rag_questions[1], AnsweredQuestion)


def test_student_search_results() -> None:
    """Test StudentSearchResults serialization and parsing."""
    results = StudentSearchResults(
        k=5,
        search_results=[
            MinimalSearchResults(
                question_id="q1",
                question="Search test",
                retrieved_sources=[
                    MinimalSource(
                        file_path="file.py",
                        first_character_index=0,
                        last_character_index=100,
                    )
                ],
            )
        ],
    )
    dumped = results.model_dump()
    loaded = StudentSearchResults.model_validate(dumped)
    assert loaded.k == 5
    assert len(loaded.search_results) == 1
    assert loaded.search_results[0].question_id == "q1"


def test_student_search_results_and_answer() -> None:
    """Test StudentSearchResultsAndAnswer model."""
    ans = StudentSearchResultsAndAnswer(
        k=10,
        search_results=[
            MinimalAnswer(
                question_id="q1",
                question="Question 1",
                retrieved_sources=[],
                answer="Generated answer.",
            )
        ],
    )
    assert ans.k == 10
    assert ans.search_results[0].answer == "Generated answer."
