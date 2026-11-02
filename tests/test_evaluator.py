"""Unit tests for the evaluator module (IoU and Recall@k)."""

import pytest

from src.evaluation.evaluator import (
    calculate_question_recall,
    compute_iou,
    evaluate_dataset,
    is_matching_source,
)
from src.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


def test_compute_iou_disjoint() -> None:
    """Disjoint spans should yield an IoU of 0.0."""
    assert compute_iou(0, 100, 100, 200) == 0.0
    assert compute_iou(0, 50, 60, 100) == 0.0


def test_compute_iou_identical() -> None:
    """Identical spans should yield an IoU of 1.0."""
    assert compute_iou(100, 500, 100, 500) == 1.0


def test_compute_iou_partial_overlap() -> None:
    """Partial overlap should match the intersection / union ratio."""
    # Span A: [0, 100] (len 100)
    # Span B: [50, 150] (len 100)
    # Intersection: [50, 100] (len 50)
    # Union: 100 + 100 - 50 = 150
    # IoU: 50 / 150 = 1/3
    assert pytest.approx(compute_iou(0, 100, 50, 150), 1e-5) == 1.0 / 3.0


def test_compute_iou_contained_span() -> None:
    """Small span inside a larger span."""
    # Span A: [0, 1000] (len 1000)
    # Span B: [200, 300] (len 100)
    # Intersection: 100, Union: 1000, IoU = 0.1
    assert pytest.approx(compute_iou(0, 1000, 200, 300), 1e-5) == 0.1


def test_compute_iou_invalid_spans() -> None:
    """Inverted or zero-length spans should return 0.0 without error."""
    assert compute_iou(100, 50, 0, 100) == 0.0
    assert compute_iou(50, 50, 50, 100) == 0.0


def test_is_matching_source() -> None:
    """Test matching logic with file paths and IoU thresholds."""
    gt = MinimalSource(
        file_path="data/raw/vllm-0.10.1/docs/test.md",
        first_character_index=100,
        last_character_index=300,
    )
    # Same file, overlapping with IoU >= 0.05
    cand_match = MinimalSource(
        file_path="data/raw/vllm-0.10.1/docs/test.md",
        first_character_index=150,
        last_character_index=350,
    )
    assert is_matching_source(cand_match, gt, min_iou=0.05) is True

    # Same file, but IoU < 0.05
    # GT len 200, Candidate len 2000, overlap 5 -> IoU ~ 0.002
    cand_low_iou = MinimalSource(
        file_path="data/raw/vllm-0.10.1/docs/test.md",
        first_character_index=295,
        last_character_index=2295,
    )
    assert is_matching_source(cand_low_iou, gt, min_iou=0.05) is False

    # Different file path
    cand_diff_file = MinimalSource(
        file_path="data/raw/vllm-0.10.1/docs/other.md",
        first_character_index=100,
        last_character_index=300,
    )
    assert is_matching_source(cand_diff_file, gt, min_iou=0.05) is False


def test_calculate_question_recall() -> None:
    """Test Recall@k calculation for a single question."""
    gt1 = MinimalSource(
        file_path="a.py",
        first_character_index=0,
        last_character_index=100,
    )
    gt2 = MinimalSource(
        file_path="b.py",
        first_character_index=0,
        last_character_index=100,
    )
    ground_truth = [gt1, gt2]

    # Retrieved matches gt1 at rank 1, gt2 at rank 3
    retrieved = [
        MinimalSource(
            file_path="a.py",
            first_character_index=10,
            last_character_index=90,
        ),
        MinimalSource(
            file_path="c.py",
            first_character_index=0,
            last_character_index=50,
        ),
        MinimalSource(
            file_path="b.py",
            first_character_index=0,
            last_character_index=100,
        ),
    ]

    # Recall@1: only gt1 found -> 1 / 2 = 0.5
    assert calculate_question_recall(retrieved, ground_truth, k=1) == 0.5
    # Recall@2: gt1 found -> 0.5
    assert calculate_question_recall(retrieved, ground_truth, k=2) == 0.5
    # Recall@3: gt1 and gt2 found -> 1.0
    assert calculate_question_recall(retrieved, ground_truth, k=3) == 1.0


def test_evaluate_dataset() -> None:
    """Test dataset-level Recall@k averaging."""
    q1 = AnsweredQuestion(
        question_id="q1",
        question="Question 1",
        sources=[
            MinimalSource(
                file_path="a.py",
                first_character_index=0,
                last_character_index=100,
            )
        ],
        answer="Answer 1",
    )
    q2 = AnsweredQuestion(
        question_id="q2",
        question="Question 2",
        sources=[
            MinimalSource(
                file_path="b.py",
                first_character_index=0,
                last_character_index=100,
            )
        ],
        answer="Answer 2",
    )
    gt_dataset = RagDataset(rag_questions=[q1, q2])

    # Student found q1 in top 1, but did not find q2
    student_res = StudentSearchResults(
        k=5,
        search_results=[
            MinimalSearchResults(
                question_id="q1",
                question="Question 1",
                retrieved_sources=[
                    MinimalSource(
                        file_path="a.py",
                        first_character_index=0,
                        last_character_index=100,
                    )
                ],
            ),
            MinimalSearchResults(
                question_id="q2",
                question="Question 2",
                retrieved_sources=[
                    MinimalSource(
                        file_path="wrong.py",
                        first_character_index=0,
                        last_character_index=100,
                    )
                ],
            ),
        ],
    )

    scores = evaluate_dataset(student_res, gt_dataset, k_values=(1, 5))
    assert scores[1] == 0.5
    assert scores[5] == 0.5
