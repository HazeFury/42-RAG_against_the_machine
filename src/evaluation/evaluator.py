"""Evaluation module for calculating IoU and Recall@k metrics.

This module replicates the official evaluation logic used by the 42 moulinette
without importing or calling the moulinette binary.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from src.config import DEFAULT_IOU_THRESHOLD, EVALUATION_K_VALUES
from src.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
from src.utils.io_helpers import normalize_corpus_path, safe_read_json


def compute_iou(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> float:
    """Compute 1D Intersection over Union between two character spans.

    Args:
        start_a: Start character index of first span.
        end_a: End character index of first span.
        start_b: Start character index of second span.
        end_b: End character index of second span.

    Returns:
        Intersection over union value between 0.0 and 1.0.
    """
    if start_a >= end_a or start_b >= end_b:
        return 0.0

    intersection = max(0, min(end_a, end_b) - max(start_a, start_b))
    len_a = end_a - start_a
    len_b = end_b - start_b
    union = len_a + len_b - intersection

    if union <= 0:
        return 0.0
    return float(intersection / union)


def is_matching_source(
    candidate: MinimalSource,
    ground_truth: MinimalSource,
    min_iou: float = DEFAULT_IOU_THRESHOLD,
) -> bool:
    """Determine if a candidate source matches a ground-truth source.

    Both sources must point to the exact same file path and have an IoU
    greater than or equal to min_iou.

    Args:
        candidate: The retrieved source.
        ground_truth: The ground-truth reference source.
        min_iou: Minimum IoU threshold (default 0.05).

    Returns:
        True if matching, False otherwise.
    """
    cand_path = normalize_corpus_path(candidate.file_path)
    gt_path = normalize_corpus_path(ground_truth.file_path)

    if cand_path != gt_path:
        return False

    iou = compute_iou(
        candidate.first_character_index,
        candidate.last_character_index,
        ground_truth.first_character_index,
        ground_truth.last_character_index,
    )
    return iou >= min_iou


def calculate_question_recall(
    retrieved_sources: Sequence[MinimalSource],
    ground_truth_sources: Sequence[MinimalSource],
    k: int,
    min_iou: float = DEFAULT_IOU_THRESHOLD,
) -> float:
    """Calculate Recall@k for a single question.

    Args:
        retrieved_sources: Ranked list of retrieved sources.
        ground_truth_sources: List of expected ground-truth sources.
        k: Cut-off rank.
        min_iou: Minimum IoU required for a match.

    Returns:
        Recall@k score between 0.0 and 1.0.
    """
    if not ground_truth_sources:
        return 1.0
    if k <= 0 or not retrieved_sources:
        return 0.0

    top_k = retrieved_sources[:k]
    found_count = 0

    for gt in ground_truth_sources:
        if any(is_matching_source(r, gt, min_iou) for r in top_k):
            found_count += 1

    return float(found_count / len(ground_truth_sources))


def evaluate_dataset(
    student_results: StudentSearchResults,
    ground_truth: RagDataset,
    k_values: Sequence[int] = EVALUATION_K_VALUES,
    min_iou: float = DEFAULT_IOU_THRESHOLD,
) -> dict[int, float]:
    """Calculate mean Recall@k across all answered questions in the dataset.

    Args:
        student_results: Search results produced by the system.
        ground_truth: Dataset containing ground-truth questions.
        k_values: Sequence of k ranks to compute (default 1, 3, 5, 10).
        min_iou: Minimum IoU threshold.

    Returns:
        Dictionary mapping each k to its average recall score.
    """
    # Index student search results by question_id
    student_map: dict[str, MinimalSearchResults] = {
        res.question_id: res for res in student_results.search_results
    }

    # Extract answered questions with non-empty ground-truth sources
    answered_questions: list[AnsweredQuestion] = [
        q
        for q in ground_truth.rag_questions
        if isinstance(q, AnsweredQuestion)
    ]

    if not answered_questions:
        return {k: 0.0 for k in k_values}

    recalls_by_k: dict[int, list[float]] = {k: [] for k in k_values}

    for question in answered_questions:
        student_res = student_map.get(question.question_id)
        retrieved = student_res.retrieved_sources if student_res else []

        for k in k_values:
            score = calculate_question_recall(
                retrieved_sources=retrieved,
                ground_truth_sources=question.sources,
                k=k,
                min_iou=min_iou,
            )
            recalls_by_k[k].append(score)

    return {
        k: sum(scores) / len(scores) if scores else 0.0
        for k, scores in recalls_by_k.items()
    }


def format_evaluation_report(
    scores: dict[int, float],
    total_questions: int,
) -> str:
    """Format evaluation scores into the standard report string.

    Args:
        scores: Dictionary mapping k values to average recall scores.
        total_questions: Number of evaluated questions.

    Returns:
        Formatted multi-line report string.
    """
    lines: list[str] = [
        "Student data is valid: True",
        "Evaluation Results",
        "=" * 40,
        f"Questions evaluated: {total_questions}",
    ]
    for k in sorted(scores.keys()):
        score = scores[k]
        percentage = score * 100
        line = f"Recall@{k}: {score:.3f} ({percentage:.1f}%)"
        if k == 5:
            line += "  (target: docs >= 80%, code >= 50%)"
        lines.append(line)
    return "\n".join(lines)


def evaluate_files(
    student_search_results_path: str | Path,
    dataset_path: str | Path,
    k_values: Sequence[int] = EVALUATION_K_VALUES,
    min_iou: float = DEFAULT_IOU_THRESHOLD,
) -> Optional[dict[int, float]]:
    """Load JSON files and evaluate Recall@k metrics.

    Args:
        student_search_results_path: Path to StudentSearchResults JSON.
        dataset_path: Path to ground-truth RagDataset JSON.
        k_values: Sequence of k ranks to evaluate.
        min_iou: Minimum IoU threshold.

    Returns:
        Dictionary of Recall@k scores or None if input data is invalid.
    """
    res_data = safe_read_json(student_search_results_path)
    if res_data is None:
        return None

    gt_data = safe_read_json(dataset_path)
    if gt_data is None:
        return None

    try:
        student_results = StudentSearchResults.model_validate(res_data)
        ground_truth = RagDataset.model_validate(gt_data)
    except Exception:
        return None

    return evaluate_dataset(
        student_results=student_results,
        ground_truth=ground_truth,
        k_values=k_values,
        min_iou=min_iou,
    )
