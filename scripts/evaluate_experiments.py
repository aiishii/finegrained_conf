#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate all experiments and generate comprehensive metrics summary

This script evaluates all experimental results in the experiments/ directory,
calculating various metrics (ECE, Brier Score, AUC, ROC-AUC, etc.) for each
method and generating a comprehensive summary CSV file.

Usage:
    # Evaluate all experiments
    python scripts/evaluate_experiments.py \
        --experiments_dir experiments/ \
        --output_file evaluation_summary.csv

    # Evaluate specific run_ids
    python scripts/evaluate_experiments.py \
        --experiments_dir experiments/ \
        --run_ids jemhop_qa-train-gpt-4.1-mini-2025-04-14-20250910164009 \
        --output_file evaluation_summary.csv

    # Evaluate using only common question sets (fair comparison mode)
    # This mode groups methods by dataset and sample count, then evaluates
    # only on questions where all methods have valid confidence and correctness data
    python scripts/evaluate_experiments.py \
        --experiments_dir experiments/ \
        --output_file evaluation_summary_common.csv \
        --common-questions-mode
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np

# Add src directory to path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from finegrained_conf.evaluation.metrics import (
    compute_ece,
    compute_brier_score,
    compute_roc_pr_auc,
    compute_spurious_roc_pr_auc_for_questions,
    to_float_or_nan,
    align_conf_corr
)


def load_results_file(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL results file"""
    if not filepath.exists():
        return []

    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def extract_dataset_and_count_from_run_id(run_id: str) -> Tuple[str, Optional[int]]:
    """
    Extract dataset name and sample count from run_id

    The run_id format is expected to be like:
    - {dataset}-{model}-{method}-{count}-{timestamp}
    - Example: 2wiki_qa-dev-gpt-4.1-mini-2025-04-14-triple_logprob_test03-300-20251216073407

    Args:
        run_id: Run ID string

    Returns:
        Tuple of (dataset_name, sample_count or None)
    """
    parts = run_id.split('-')

    # Dataset name is typically the first part (may contain underscores)
    dataset = parts[0] if parts else 'unknown'

    # Try to find sample count (second-to-last element if it's a number)
    sample_count = None
    if len(parts) >= 2:
        try:
            # Check if second-to-last element is a number
            potential_count = parts[-2]
            if potential_count.isdigit():
                sample_count = int(potential_count)
        except (ValueError, IndexError):
            pass

    return dataset, sample_count


def compute_spurious_metrics_for_method(answer_records, evidence_records):
    """Compute ROC-AUC / PR-AUC for spurious answer detection for one method.

    This function now uses the same validation logic as extract_valid_question_ids_*
    to ensure consistency in common-questions-mode.

    Args:
        answer_records: Records from results_answer.jsonl for this method
        evidence_records: Records from results_evidence.jsonl for this method
    """
    ans_conf_by_qid = {}
    ans_corr_by_qid = {}
    for rec in answer_records:
        qid = rec.get("question_id")
        if qid is None:
            continue
        conf = to_float_or_nan(rec.get("confidence"))
        corr = to_float_or_nan(rec.get("correct"))
        # Check both confidence and correctness are valid (matching extract_valid_question_ids_answer)
        if np.isnan(conf) or np.isnan(corr):
            continue
        ans_conf_by_qid[qid] = conf
        ans_corr_by_qid[qid] = corr

    # question_id -> list of triple confidences / correctness
    # Changed: Now require ALL triples to be valid with matching indices
    # (matching extract_valid_question_ids_evidence)
    triple_conf_by_qid = defaultdict(list)
    triple_corr_by_qid = defaultdict(list)

    # First, group evidence records by question_id
    evidence_by_qid = defaultdict(list)
    for rec in evidence_records:
        qid = rec.get("question_id")
        if qid is not None:
            evidence_by_qid[qid].append(rec)

    # Then, validate each question's triples (matching extract_valid_question_ids_evidence)
    for qid, question_records in evidence_by_qid.items():
        # Collect indices of triples with valid confidence and correctness
        valid_conf_indices = []
        valid_corr_indices = []
        valid_confs = []
        valid_corrs = []

        for idx, rec in enumerate(question_records):
            conf = rec.get('confidence')
            corr = rec.get('correct')

            # Skip padding (both None)
            if conf is None and corr is None:
                continue

            # Check if confidence is valid (not None and not NaN)
            conf_float = to_float_or_nan(conf) if conf is not None else np.nan
            if not np.isnan(conf_float):
                valid_conf_indices.append(idx)
                valid_confs.append(conf_float)

            # Check if correctness is valid (not None and not NaN)
            corr_float = to_float_or_nan(corr) if corr is not None else np.nan
            if not np.isnan(corr_float):
                valid_corr_indices.append(idx)
                valid_corrs.append(corr_float)

        # Only include this question if:
        # 1. The same triples have valid confidence and correctness (indices match)
        # 2. There is at least one valid triple
        if valid_conf_indices == valid_corr_indices and len(valid_conf_indices) > 0:
            triple_conf_by_qid[qid] = valid_confs
            triple_corr_by_qid[qid] = valid_corrs

    answer_correctness = []
    evidence_correctness = []
    triple_confidences_per_question = []

    # DEBUG: Print validation statistics
    print(f"      [compute_spurious] ans_corr_by_qid: {len(ans_corr_by_qid)} questions")
    print(f"      [compute_spurious] triple_conf_by_qid: {len(triple_conf_by_qid)} questions")

    # answer と evidence の両方がそろっている question_id だけ使う
    for qid, ans_corr in ans_corr_by_qid.items():
        if qid not in triple_conf_by_qid:
            continue
        confs = triple_conf_by_qid[qid]
        if not confs:
            continue
        corrs = triple_corr_by_qid[qid]

        # If all triples are correct (1), evidence is correct (1), otherwise spurious (0)
        evid_corr = 1.0 if all(c == 1.0 for c in corrs) else 0.0

        answer_correctness.append(ans_corr)
        evidence_correctness.append(evid_corr)
        triple_confidences_per_question.append(confs)

    # DEBUG: Print final processed count
    print(f"      [compute_spurious] Final processed: {len(answer_correctness)} questions")

    if not answer_correctness:
        return float("nan"), float("nan"), 0

    return compute_spurious_roc_pr_auc_for_questions(
        answer_correctness,
        evidence_correctness,
        triple_confidences_per_question,
        debug=True,
    )


def extract_valid_question_ids_answer(records: List[Dict[str, Any]]) -> set:
    """
    Extract question IDs with valid confidence and correctness data for answer-level

    Args:
        records: List of answer-level record dictionaries

    Returns:
        Set of valid question IDs
    """
    valid_qids = set()
    for rec in records:
        qid = rec.get('question_id')
        conf = rec.get('confidence')
        corr = rec.get('correct')

        if qid is not None and conf is not None and corr is not None:
            if not np.isnan(to_float_or_nan(conf)) and not np.isnan(to_float_or_nan(corr)):
                valid_qids.add(qid)

    return valid_qids


def extract_valid_question_ids_evidence(records: List[Dict[str, Any]]) -> set:
    """
    Extract question IDs with valid confidence and correctness data for all triples

    This function ensures that for each question:
    1. Each triple has both valid confidence and correctness (or is padding)
    2. The number of valid confidence values equals the number of valid correctness values
    3. Valid values are at the same triple indices

    Args:
        records: List of evidence-level record dictionaries

    Returns:
        Set of valid question IDs
    """
    # Group by question_id
    question_groups = defaultdict(list)
    for rec in records:
        qid = rec.get('question_id')
        if qid is not None:
            question_groups[qid].append(rec)

    valid_qids = set()
    for qid, question_records in question_groups.items():
        # Collect indices of triples with valid confidence and correctness
        valid_conf_indices = []
        valid_corr_indices = []

        for idx, rec in enumerate(question_records):
            conf = rec.get('confidence')
            corr = rec.get('correct')

            # Skip padding (both None)
            if conf is None and corr is None:
                continue

            # Check if confidence is valid (not None and not NaN)
            if conf is not None and not np.isnan(to_float_or_nan(conf)):
                valid_conf_indices.append(idx)

            # Check if correctness is valid (not None and not NaN)
            if corr is not None and not np.isnan(to_float_or_nan(corr)):
                valid_corr_indices.append(idx)

        # The question is valid only if:
        # 1. The same triples have valid confidence and correctness (indices match)
        # 2. There is at least one valid triple
        if valid_conf_indices == valid_corr_indices and len(valid_conf_indices) > 0:
            valid_qids.add(qid)

    return valid_qids


def filter_records_by_questions(records: List[Dict[str, Any]], valid_qids: set) -> List[Dict[str, Any]]:
    """
    Filter records to only include those with question IDs in valid_qids

    Args:
        records: List of record dictionaries
        valid_qids: Set of valid question IDs

    Returns:
        Filtered list of records
    """
    return [rec for rec in records if rec.get('question_id') in valid_qids]


def calculate_answer_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate metrics for answer-level results

    Args:
        records: List of record dictionaries with keys: prediction, confidence, correct

    Returns:
        Dictionary of metric names and values
    """
    if not records:
        return {}

    # Extract valid data
    confidences = []
    correctness = []

    for rec in records:
        conf = rec.get('confidence')
        corr = rec.get('correct')

        if conf is not None and corr is not None and not np.isnan(to_float_or_nan(conf)) and not np.isnan(to_float_or_nan(corr)):
            confidences.append(to_float_or_nan(conf))
            correctness.append(to_float_or_nan(corr))

    confidences = np.array(confidences)
    correctness = np.array(correctness)

    if len(confidences) == 0:
        return {
            'accuracy': np.nan,
            'ece': np.nan,
            'brier_score': np.nan,
            'roc_auc': np.nan,
            'pr_auc': np.nan,
            'selective_auc': np.nan,
            'num_samples': 0
        }

    # Calculate metrics
    metrics = {}

    # Accuracy
    metrics['accuracy'] = float(np.mean(correctness))

    # ECE
    try:
        metrics['ece'] = compute_ece(confidences, correctness)
    except Exception as e:
        warnings.warn(f"Failed to compute ECE: {e}")
        metrics['ece'] = np.nan

    # Brier Score
    try:
        metrics['brier_score'] = compute_brier_score(confidences, correctness)
    except Exception as e:
        warnings.warn(f"Failed to compute Brier Score: {e}")
        metrics['brier_score'] = np.nan

    # ROC-AUC and PR-AUC (for error detection)
    # Higher confidence should indicate correct answer, so use 1-confidence for error detection
    try:
        y_true = (1 - correctness).astype(int)  # 1 = incorrect (error)
        y_score = 1 - confidences  # Higher score = more likely to be incorrect

        if len(np.unique(y_true)) >= 2:
            roc_auc, pr_auc = compute_roc_pr_auc(y_true, y_score)
            metrics['roc_auc'] = roc_auc
            metrics['pr_auc'] = pr_auc
        else:
            metrics['roc_auc'] = np.nan
            metrics['pr_auc'] = np.nan
    except Exception as e:
        warnings.warn(f"Failed to compute ROC-AUC/PR-AUC: {e}")
        metrics['roc_auc'] = np.nan
        metrics['pr_auc'] = np.nan

    # Selective AUC (coverage vs accuracy)
    try:
        # Sort by confidence (descending)
        sorted_indices = np.argsort(-confidences)
        sorted_correctness = correctness[sorted_indices]

        # Calculate cumulative accuracy at different coverage levels
        cumulative_correct = np.cumsum(sorted_correctness)
        coverage = np.arange(1, len(sorted_correctness) + 1)
        cumulative_accuracy = cumulative_correct / coverage

        # AUC of coverage vs accuracy curve (normalized by coverage)
        coverage_norm = coverage / len(coverage)
        # Use trapezoid (numpy 2.0+) or fallback to trapz for older versions
        try:
            selective_auc = float(np.trapezoid(cumulative_accuracy, coverage_norm))
        except AttributeError:
            selective_auc = float(np.trapz(cumulative_accuracy, coverage_norm))
        metrics['selective_auc'] = selective_auc
    except Exception as e:
        warnings.warn(f"Failed to compute Selective AUC: {e}")
        metrics['selective_auc'] = np.nan

    metrics['num_samples'] = int(len(confidences))

    return metrics


def calculate_evidence_metrics(records: List[Dict[str, Any]], strict: bool = False) -> Dict[str, float]:
    """
    Calculate metrics for evidence/triple-level results

    Uses align_conf_corr to handle alignment consistently with run_evidence_experiment.py.

    Args:
        records: List of record dictionaries with keys: question_id, triple_index, subject, confidence, correct
        strict: If True, discard entire questions when array lengths don't match (default: False)

    Returns:
        Dictionary of metric names and values
    """
    if not records:
        return {}

    # Group records by question_id to build 2D arrays (one row per question)
    question_groups = defaultdict(list)
    for rec in records:
        qid = rec.get('question_id')
        if qid is not None:
            question_groups[qid].append(rec)

    # Build 2D arrays for align_conf_corr
    conf_nested = []
    corr_nested = []

    # Debug counters (only in strict mode)
    debug_total_questions = 0
    debug_skipped_questions = 0

    for qid, question_records in question_groups.items():
        # Sort by triple_index to maintain order
        question_records.sort(key=lambda x: x.get('triple_index', 0))

        # Build arrays for this question
        question_conf = []
        question_corr = []

        for rec in question_records:
            conf = rec.get('confidence')
            corr = rec.get('correct')

            # Include all values (None will be handled by align_conf_corr)
            question_conf.append(conf)
            question_corr.append(corr)

        # In strict mode, check for array length mismatch by counting non-None values
        # JSONL format: All questions have same record count (padded with None)
        # So we must count non-None values to detect the original array length mismatch
        if strict:
            debug_total_questions += 1
            conf_non_none = sum(1 for c in question_conf if c is not None)
            corr_non_none = sum(1 for c in question_corr if c is not None)

            # Skip this question if array lengths don't match (strict filtering)
            if conf_non_none != corr_non_none:
                debug_skipped_questions += 1
                continue

        conf_nested.append(question_conf)
        corr_nested.append(question_corr)

    # Print debug info in strict mode
    if strict and debug_total_questions > 0:
        import warnings
        # Count total valid samples before align_conf_corr
        total_conf_nones = sum(sum(1 for c in row if c is None) for row in conf_nested)
        total_corr_nones = sum(sum(1 for c in row if c is None) for row in corr_nested)
        total_valid_before = sum(
            sum(1 for c, r in zip(conf_row, corr_row)
                if c is not None and r is not None)
            for conf_row, corr_row in zip(conf_nested, corr_nested)
        )
        warnings.warn(
            f"Strict filtering: {debug_skipped_questions}/{debug_total_questions} questions skipped "
            f"({100*debug_skipped_questions/debug_total_questions:.1f}%). "
            f"Conf Nones: {total_conf_nones}, Corr Nones: {total_corr_nones}, "
            f"Valid samples: {total_valid_before}"
        )

    # Use align_conf_corr to get flattened, filtered arrays
    # Note: In strict mode, mismatched questions are already filtered out above
    # so we pass strict=False to align_conf_corr to avoid double-filtering
    confidences, correctness = align_conf_corr(conf_nested, corr_nested, strict=False)

    # Additional debug output in strict mode
    if strict and debug_total_questions > 0:
        warnings.warn(
            f"After align_conf_corr: {len(confidences)} samples, "
            f"accuracy={correctness.mean():.6f}"
        )

    if len(confidences) == 0:
        return {
            'accuracy': np.nan,
            'ece': np.nan,
            'brier_score': np.nan,
            'roc_auc': np.nan,
            'pr_auc': np.nan,
            'selective_auc': np.nan,
            'num_samples': 0
        }

    # Calculate metrics (same as answer level)
    metrics = {}

    # Accuracy
    metrics['accuracy'] = float(np.mean(correctness))

    # ECE
    try:
        metrics['ece'] = compute_ece(confidences, correctness)
    except Exception as e:
        warnings.warn(f"Failed to compute ECE: {e}")
        metrics['ece'] = np.nan

    # Brier Score
    try:
        metrics['brier_score'] = compute_brier_score(confidences, correctness)
    except Exception as e:
        warnings.warn(f"Failed to compute Brier Score: {e}")
        metrics['brier_score'] = np.nan

    # ROC-AUC and PR-AUC (for error detection)
    try:
        y_true = (1 - correctness).astype(int)
        y_score = 1 - confidences

        if len(np.unique(y_true)) >= 2:
            roc_auc, pr_auc = compute_roc_pr_auc(y_true, y_score)
            metrics['roc_auc'] = roc_auc
            metrics['pr_auc'] = pr_auc
        else:
            metrics['roc_auc'] = np.nan
            metrics['pr_auc'] = np.nan
    except Exception as e:
        warnings.warn(f"Failed to compute ROC-AUC/PR-AUC: {e}")
        metrics['roc_auc'] = np.nan
        metrics['pr_auc'] = np.nan

    # Selective AUC
    try:
        sorted_indices = np.argsort(-confidences)
        sorted_correctness = correctness[sorted_indices]
        cumulative_correct = np.cumsum(sorted_correctness)
        coverage = np.arange(1, len(sorted_correctness) + 1)
        cumulative_accuracy = cumulative_correct / coverage
        coverage_norm = coverage / len(coverage)
        # Use trapezoid (numpy 2.0+) or fallback to trapz for older versions
        try:
            selective_auc = float(np.trapezoid(cumulative_accuracy, coverage_norm))
        except AttributeError:
            selective_auc = float(np.trapz(cumulative_accuracy, coverage_norm))
        metrics['selective_auc'] = selective_auc
    except Exception as e:
        warnings.warn(f"Failed to compute Selective AUC: {e}")
        metrics['selective_auc'] = np.nan

    metrics['num_samples'] = int(len(confidences))

    return metrics


def evaluate_run_with_common_questions(
    run_dir: Path,
    common_qids: Optional[set] = None,
    strict: bool = False
) -> List[Dict[str, Any]]:
    """
    Evaluate a single experimental run using only common question sets

    Args:
        run_dir: Directory containing results_answer.jsonl and/or results_evidence.jsonl
        common_qids: Set of common question IDs where all methods have both valid answer and evidence data
        strict: If True, use strict filtering mode for evidence metrics (default: False)

    Returns:
        List of metric dictionaries (one per method and level)
    """
    run_id = run_dir.name
    results = []

    # Load both answer and evidence files at the beginning
    answer_file = run_dir / "results_answer.jsonl"
    evidence_file = run_dir / "results_evidence.jsonl"

    records_answer = load_results_file(answer_file) if answer_file.exists() else []
    records_evidence = load_results_file(evidence_file) if evidence_file.exists() else []

    # Process answer-level results
    if records_answer and common_qids:
        # Group answer records by method
        answer_methods_dict = defaultdict(list)
        for rec in records_answer:
            answer_methods_dict[rec.get('method', 'unknown')].append(rec)

        # Group evidence records by method (for spurious metrics)
        evidence_methods_dict = defaultdict(list)
        for rec in records_evidence:
            evidence_methods_dict[rec.get('method', 'unknown')].append(rec)

        for method, method_records in answer_methods_dict.items():
            # Filter by common question IDs
            filtered_records = filter_records_by_questions(method_records, common_qids)

            if not filtered_records:
                continue

            metrics = calculate_answer_metrics(filtered_records)

            if metrics:
                first_rec = filtered_records[0]
                result = {
                    'run_id': run_id,
                    'method': method,
                    'level': 'answer',
                    'dataset': first_rec.get('dataset', 'unknown'),
                    'split': first_rec.get('split', 'unknown'),
                    'model': first_rec.get('model', 'unknown'),
                    **metrics
                }

                # Compute spurious metrics if evidence data is available for this method
                if method in evidence_methods_dict:
                    # Use the same common_qids for spurious metrics
                    # These questions have both valid answer and evidence data for all methods
                    filtered_evidence_for_spurious = filter_records_by_questions(
                        evidence_methods_dict[method],
                        common_qids
                    )

                    # DEBUG: Check what questions we're filtering
                    filtered_answer_qids = set(rec.get('question_id') for rec in filtered_records if rec.get('question_id'))
                    filtered_evidence_qids = set(rec.get('question_id') for rec in filtered_evidence_for_spurious if rec.get('question_id'))
                    print(f"    DEBUG [{run_id}][{method}]: common_qids={len(common_qids)}, "
                          f"filtered_answer_qids={len(filtered_answer_qids)}, "
                          f"filtered_evidence_qids={len(filtered_evidence_qids)}")

                    spurious_roc_auc, spurious_pr_auc, num_processed = compute_spurious_metrics_for_method(
                        filtered_records,
                        filtered_evidence_for_spurious
                    )

                    print(f"    DEBUG [{run_id}][{method}]: num_processed={num_processed}")

                    result['spurious_roc_auc'] = spurious_roc_auc
                    result['spurious_pr_auc'] = spurious_pr_auc
                    result['spurious_num_processed'] = num_processed
                else:
                    result['spurious_roc_auc'] = np.nan
                    result['spurious_pr_auc'] = np.nan
                    result['spurious_num_processed'] = 0

                results.append(result)

    # Process evidence-level results
    if records_evidence and common_qids:
        # Group by method
        methods_dict = defaultdict(list)
        for rec in records_evidence:
            methods_dict[rec.get('method', 'unknown')].append(rec)

        for method, method_records in methods_dict.items():
            # Filter by common question IDs
            filtered_records = filter_records_by_questions(method_records, common_qids)

            if not filtered_records:
                continue

            metrics = calculate_evidence_metrics(filtered_records, strict=strict)

            if metrics:
                first_rec = filtered_records[0]
                result = {
                    'run_id': run_id,
                    'method': method,
                    'level': 'evidence',
                    'dataset': first_rec.get('dataset', 'unknown'),
                    'split': first_rec.get('split', 'unknown'),
                    'model': first_rec.get('model', 'unknown'),
                    **metrics
                }
                results.append(result)

    return results


def evaluate_run(run_dir: Path, strict: bool = False) -> List[Dict[str, Any]]:
    """
    Evaluate a single experimental run

    Args:
        run_dir: Directory containing results_answer.jsonl and/or results_evidence.jsonl
        strict: If True, use strict filtering mode for evidence metrics (default: False)

    Returns:
        List of metric dictionaries (one per method and level)
    """
    run_id = run_dir.name
    results = []

    # Load both answer and evidence files at the beginning
    answer_file = run_dir / "results_answer.jsonl"
    evidence_file = run_dir / "results_evidence.jsonl"

    records_answer = load_results_file(answer_file) if answer_file.exists() else []
    records_evidence = load_results_file(evidence_file) if evidence_file.exists() else []

    # Process answer-level results
    if records_answer:
        # Group answer records by method
        answer_methods_dict = defaultdict(list)
        for rec in records_answer:
            answer_methods_dict[rec.get('method', 'unknown')].append(rec)

        # Group evidence records by method (for spurious metrics)
        evidence_methods_dict = defaultdict(list)
        for rec in records_evidence:
            evidence_methods_dict[rec.get('method', 'unknown')].append(rec)

        for method, method_records in answer_methods_dict.items():
            metrics = calculate_answer_metrics(method_records)

            if metrics:
                first_rec = method_records[0]
                result = {
                    'run_id': run_id,
                    'method': method,
                    'level': 'answer',
                    'dataset': first_rec.get('dataset', 'unknown'),
                    'split': first_rec.get('split', 'unknown'),
                    'model': first_rec.get('model', 'unknown'),
                    **metrics
                }

                # Compute spurious metrics if evidence data is available for this method
                if method in evidence_methods_dict:
                    spurious_roc_auc, spurious_pr_auc, num_processed = compute_spurious_metrics_for_method(
                        method_records,
                        evidence_methods_dict[method]
                    )
                    result['spurious_roc_auc'] = spurious_roc_auc
                    result['spurious_pr_auc'] = spurious_pr_auc
                    result['spurious_num_processed'] = num_processed
                else:
                    result['spurious_roc_auc'] = np.nan
                    result['spurious_pr_auc'] = np.nan
                    result['spurious_num_processed'] = 0

                results.append(result)

    # Process evidence-level results
    if records_evidence:
        # Group by method
        methods_dict = defaultdict(list)
        for rec in records_evidence:
            methods_dict[rec.get('method', 'unknown')].append(rec)

        for method, method_records in methods_dict.items():
            metrics = calculate_evidence_metrics(method_records, strict=strict)

            if metrics:
                first_rec = method_records[0]
                result = {
                    'run_id': run_id,
                    'method': method,
                    'level': 'evidence',
                    'dataset': first_rec.get('dataset', 'unknown'),
                    'split': first_rec.get('split', 'unknown'),
                    'model': first_rec.get('model', 'unknown'),
                    **metrics
                }
                results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate experimental results and generate metrics summary"
    )
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="experiments/",
        help="Directory containing experimental results (default: experiments/)"
    )
    parser.add_argument(
        "--run_ids",
        type=str,
        nargs='+',
        default=None,
        help="Specific run IDs to evaluate (default: all runs)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_summary.csv",
        help="Output CSV file path (default: evaluation_summary.csv)"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=['csv', 'json', 'both'],
        default='csv',
        help="Output format (default: csv)"
    )
    parser.add_argument(
        "--strict-filtering",
        action='store_true',
        help="Use strict filtering mode: discard entire questions when array lengths don't match (default: flexible filtering)"
    )
    parser.add_argument(
        "--common-questions-mode",
        action='store_true',
        help="Evaluate using only common question sets across methods with the same dataset and sample count"
    )

    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)

    if not experiments_dir.exists():
        print(f"Error: Experiments directory not found: {experiments_dir}")
        return

    print(f"\n{'='*80}")
    print(f"Evaluating Experimental Results")
    print(f"{'='*80}")
    print(f"Experiments directory: {experiments_dir}")
    print(f"Filtering mode: {'strict' if args.strict_filtering else 'flexible'}")
    print(f"Evaluation mode: {'common questions' if args.common_questions_mode else 'all questions'}")

    # Find all run directories
    if args.run_ids:
        run_dirs = [experiments_dir / run_id for run_id in args.run_ids]
        run_dirs = [d for d in run_dirs if d.exists() and d.is_dir()]
    else:
        run_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]

    if not run_dirs:
        print("No experimental runs found")
        return

    print(f"Found {len(run_dirs)} experimental run(s)")
    print(f"{'='*80}\n")

    # Evaluate each run
    all_results = []

    if args.common_questions_mode:
        # Common questions mode: First pass to extract valid question IDs
        print("Phase 1: Extracting valid question IDs (both answer AND evidence) from all methods...")

        # Structure: {(dataset, num_samples): {(run_id, method): set()}}
        # Each set contains question IDs where BOTH answer AND evidence are valid
        # Use (run_id, method) tuple as key to avoid overwriting when same method exists in different runs
        valid_qids_by_group = defaultdict(lambda: defaultdict(set))

        for run_dir in sorted(run_dirs):
            # Extract dataset and sample count from run_id
            run_id = run_dir.name
            dataset_from_run_id, num_samples_from_run_id = extract_dataset_and_count_from_run_id(run_id)

            answer_file = run_dir / "results_answer.jsonl"
            evidence_file = run_dir / "results_evidence.jsonl"

            records_answer = load_results_file(answer_file) if answer_file.exists() else []
            records_evidence = load_results_file(evidence_file) if evidence_file.exists() else []

            # Group by method
            answer_methods = defaultdict(list)
            evidence_methods = defaultdict(list)

            for rec in records_answer:
                answer_methods[rec.get('method', 'unknown')].append(rec)

            for rec in records_evidence:
                evidence_methods[rec.get('method', 'unknown')].append(rec)

            # Determine dataset and num_samples for grouping
            # Priority: run_id > first record
            if records_answer:
                first_rec = records_answer[0]
                dataset = dataset_from_run_id if dataset_from_run_id != 'unknown' else first_rec.get('dataset', 'unknown')
            elif records_evidence:
                first_rec = records_evidence[0]
                dataset = dataset_from_run_id if dataset_from_run_id != 'unknown' else first_rec.get('dataset', 'unknown')
            else:
                dataset = dataset_from_run_id

            # Use sample count from run_id if available, otherwise fallback to actual record count
            if num_samples_from_run_id is not None:
                num_samples = num_samples_from_run_id
            elif records_answer:
                # Fallback: use number of unique question_ids from answer records
                unique_qids = set(rec.get('question_id') for rec in records_answer if rec.get('question_id') is not None)
                num_samples = len(unique_qids)
            elif records_evidence:
                # Fallback: use number of unique question_ids from evidence records
                unique_qids = set(rec.get('question_id') for rec in records_evidence if rec.get('question_id') is not None)
                num_samples = len(unique_qids)
            else:
                num_samples = 0

            group_key = (dataset, num_samples)

            # Extract valid question IDs for each method where BOTH answer AND evidence are valid
            # Use (run_id, method) as key to avoid overwriting
            all_methods = set(answer_methods.keys()) | set(evidence_methods.keys())
            for method in all_methods:
                answer_records = answer_methods.get(method, [])
                evidence_records = evidence_methods.get(method, [])

                # Extract valid answer question IDs
                valid_answer_qids = extract_valid_question_ids_answer(answer_records) if answer_records else set()

                # Extract valid evidence question IDs
                valid_evidence_qids = extract_valid_question_ids_evidence(evidence_records) if evidence_records else set()

                # Store only questions where BOTH answer AND evidence are valid
                both_valid_qids = valid_answer_qids & valid_evidence_qids
                valid_qids_by_group[group_key][(run_id, method)] = both_valid_qids

        # Compute common question IDs for each group
        print("\nPhase 2: Computing common question sets...")
        # Structure: {(dataset, num_samples): set()}
        common_qids_by_group = {}

        for group_key, methods_data in valid_qids_by_group.items():
            dataset, num_samples = group_key
            print(f"\nGroup: dataset={dataset}, num_samples={num_samples}")

            # Compute intersection of valid question IDs across ALL (run_id, method) combinations
            # This ensures all methods in the group use the same common question set
            # where every method has BOTH valid answer AND evidence data

            all_qids_sets = [qids for qids in methods_data.values() if qids]
            if all_qids_sets:
                common_qids = set.intersection(*all_qids_sets)
                print(f"  {len(all_qids_sets)} (run_id, method) combination(s), "
                      f"{len(common_qids)} common questions (both answer and evidence valid)")
            else:
                common_qids = set()

            # Store common question IDs for this group
            common_qids_by_group[group_key] = common_qids

        # Phase 3: Evaluate with common question sets
        print(f"\nPhase 3: Evaluating with common question sets...")

        for run_dir in sorted(run_dirs):
            print(f"Evaluating: {run_dir.name}")

            # Extract dataset and sample count from run_id (same logic as Phase 1)
            run_id = run_dir.name
            dataset_from_run_id, num_samples_from_run_id = extract_dataset_and_count_from_run_id(run_id)

            # Determine which group this run belongs to
            answer_file = run_dir / "results_answer.jsonl"
            evidence_file = run_dir / "results_evidence.jsonl"

            records_answer = load_results_file(answer_file) if answer_file.exists() else []
            records_evidence = load_results_file(evidence_file) if evidence_file.exists() else []

            # Determine dataset and num_samples for grouping (same logic as Phase 1)
            if records_answer:
                first_rec = records_answer[0]
                dataset = dataset_from_run_id if dataset_from_run_id != 'unknown' else first_rec.get('dataset', 'unknown')
            elif records_evidence:
                first_rec = records_evidence[0]
                dataset = dataset_from_run_id if dataset_from_run_id != 'unknown' else first_rec.get('dataset', 'unknown')
            else:
                dataset = dataset_from_run_id

            # Use sample count from run_id if available, otherwise fallback to actual record count
            if num_samples_from_run_id is not None:
                num_samples = num_samples_from_run_id
            elif records_answer:
                unique_qids = set(rec.get('question_id') for rec in records_answer if rec.get('question_id') is not None)
                num_samples = len(unique_qids)
            elif records_evidence:
                unique_qids = set(rec.get('question_id') for rec in records_evidence if rec.get('question_id') is not None)
                num_samples = len(unique_qids)
            else:
                num_samples = 0

            group_key = (dataset, num_samples)

            if group_key in common_qids_by_group:
                common_qids = common_qids_by_group[group_key]

                run_results = evaluate_run_with_common_questions(
                    run_dir,
                    common_qids=common_qids,
                    strict=args.strict_filtering
                )
                all_results.extend(run_results)
                print(f"  ✓ Found {len(run_results)} method(s)")
            else:
                print(f"  ⚠ No common question group found for this run (dataset={dataset}, num_samples={num_samples})")

    else:
        # Standard mode: Evaluate all questions
        for run_dir in sorted(run_dirs):
            print(f"Evaluating: {run_dir.name}")

            run_results = evaluate_run(run_dir, strict=args.strict_filtering)
            all_results.extend(run_results)

            print(f"  ✓ Found {len(run_results)} method(s)")

    if not all_results:
        print("\nNo results to evaluate")
        return

    # Sort results
    all_results.sort(key=lambda x: (x['run_id'], x['level'], x['method']))

    # Define column order
    column_order = [
        'run_id', 'method', 'level', 'dataset', 'split', 'model',
        'accuracy', 'ece', 'brier_score',
        'roc_auc', 'pr_auc', 'selective_auc',
        'spurious_roc_auc', 'spurious_pr_auc', 'spurious_num_processed',
        'num_samples'
    ]

    # Save results
    print(f"\n{'='*80}")
    print(f"Saving Results")
    print(f"{'='*80}")

    if args.output_format in ['csv', 'both']:
        csv_file = Path(args.output_file)
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=column_order)
            writer.writeheader()
            for result in all_results:
                # Write only existing fields
                row = {k: result.get(k, '') for k in column_order}
                writer.writerow(row)
        print(f"✓ Saved CSV to: {csv_file}")

    if args.output_format in ['json', 'both']:
        json_file = Path(args.output_file).with_suffix('.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved JSON to: {json_file}")

    print(f"\n{'='*80}")
    print(f"Summary Statistics")
    print(f"{'='*80}")

    # Count unique values
    unique_runs = set(r['run_id'] for r in all_results)
    unique_methods = set(r['method'] for r in all_results)

    print(f"Total runs: {len(unique_runs)}")
    print(f"Total methods: {len(unique_methods)}")
    print(f"Total evaluations: {len(all_results)}")
    print(f"\nMethods evaluated:")

    method_counts = defaultdict(int)
    for r in all_results:
        method_counts[r['method']] += 1

    for method in sorted(method_counts.keys()):
        print(f"  - {method}: {method_counts[method]} evaluation(s)")

    print(f"\n{'='*80}")
    print(f"Top 5 Methods by Answer ECE")
    print(f"{'='*80}")

    answer_results = [r for r in all_results if r['level'] == 'answer']
    if answer_results:
        answer_results_sorted = sorted(answer_results, key=lambda x: x.get('ece', 0), reverse=False)[:5]
        print(f"{'Method':<40} {'Accuracy':>10} {'ECE':>10} {'Brier Score':>12}")
        print("-" * 80)
        for r in answer_results_sorted:
            print(f"{r['method']:<40} {r.get('accuracy', np.nan):>10.6f} {r.get('ece', np.nan):>10.6f} {r.get('brier_score', np.nan):>12.6f}")

    evidence_results = [r for r in all_results if r['level'] == 'evidence']
    if evidence_results:
        print(f"\n{'='*80}")
        print(f"Top 5 Methods by Evidence ECE")
        print(f"{'='*80}")
        evidence_results_sorted = sorted(evidence_results, key=lambda x: x.get('ece', 0), reverse=False)[:5]
        print(f"{'Method':<40} {'Accuracy':>10} {'ECE':>10} {'Brier Score':>12}")
        print("-" * 80)
        for r in evidence_results_sorted:
            print(f"{r['method']:<40} {r.get('accuracy', np.nan):>10.6f} {r.get('ece', np.nan):>10.6f} {r.get('brier_score', np.nan):>12.6f}")

    print(f"\n{'='*80}")
    print(f"Evaluation Complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()