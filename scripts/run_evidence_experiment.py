#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import json
import time
from pathlib import Path

from typing import Any, Mapping, Sequence
import yaml
from finegrained_conf.experiments.evidence_level import (
    run_experiment_triple,
    run_experiment_triple_input,
    plot_calibration, 
    reevaluate_correctness
)
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from finegrained_conf.evaluation.metrics import *
from finegrained_conf.datasets.data_utils import load_dataset
from finegrained_conf.io.run_metadata import ExperimentRecorder, build_run_id

AVAILABLE_MODELS = ["gpt-4.1-mini-2025-04-14", "gpt-4o-mini-2024-07-18", "gpt-4o-2024-11-20", "gpt-4.1-2025-04-14", "Llama-4-Maverick-17B-128E-Instruct-FP8", "Llama-3.3-70B-Instruct", "Phi-4", "gpt-4.1-nano-2025-04-14"]
AVAILABLE_DATASETS = ["trivia_qa", "sci_q", "truthful_qa", "jemhop_qa", "hotpot_qa", "2wiki_qa"]

AVAILABLE_METHODS = [
    "triple_label_prob",
    "triple_logprob",
    "triple_logprob_cot",
    "triple_is_true_logprob",
    "triple_is_true_cot1s_logprob",
    "triple_is_true_cot2s_logprob",
    "triple_verb_1s_cot_is_true",
    "triple_is_true_prob",
    "triple_verb_1s_top_1",
    "triple_verb_1s_top_1_ansconf",
    "triple_verb_1s_top_1_noconf",
    "triple_verb_1s_top_1_a",
    "triple_verb_1s_top_2",
    "triple_verb_1s_top_4",
    "triple_verb_2s_top_1",
    "triple_verb_2s_top_2",
    "triple_verb_2s_top_4",
    "triple_verb_1s_cot",
    "triple_verb_1s_cot_a",
    "triple_verb_2s_cot",
    "triple_verb_1s_cot_ansconf",
    "triple_verb_1s_cot_noconf",
    "triple_verb_2s_cot_a",
    "triple_ling_1s_human",
    "triple_ling_1s_human_a",
    "triple_verb_3s",
    "triple_verb_meta",
    "triple_verb_2s_top_1_a",
    "triple_is_true_prob_a",
    "triple_is_true_logprob_a",
    "triple_ling_2s_human",
    "triple_ling_2s_human_a",
    "triple_ling_1s_human7",
    "triple_ling_2s_human7",
    "triple_cot_level_baseline",
]


def _parse_triple_prediction(pred: Any) -> Mapping[str, Any]:
    if isinstance(pred, Mapping):
        return {
            "subject": pred.get("subject") or pred.get("subj") or pred.get("head") or pred.get("h"),
            "relation": pred.get("relation") or pred.get("rel") or pred.get("predicate") or pred.get("r"),
            "object": pred.get("object") or pred.get("obj") or pred.get("tail") or pred.get("t"),
        }

    if isinstance(pred, (list, tuple)) and len(pred) >= 3:
        subj, rel, obj = pred[0], pred[1], pred[2]
    elif isinstance(pred, str):
        cleaned = pred.strip("() ")
        parts = [p.strip() for p in cleaned.split(",")]
        subj, rel, obj = (parts + [None, None, None])[:3]
    else:
        subj = rel = obj = None

    return {"subject": subj, "relation": rel, "object": obj}


def _normalize_confidences(conf_list: Sequence[Any] | None) -> list[Any]:
    if conf_list is None:
        return []

    prioritized_keys = [
        "confidence",
        "prob",
        "prod_prob",
        "mean_prob",
        "sum_prob",
        "normalized_prod_prob",
        "linear_scaled_prob",
        "mean_logprob",
        "sum_logprob",
    ]

    def _scalar_from_mapping(value: Any) -> Any:
        if isinstance(value, Mapping):
            for key in prioritized_keys:
                if key in value:
                    return value[key]
            # fallback to first numeric value
            for v in value.values():
                if isinstance(v, (int, float)):
                    return v
        return value

    return [_scalar_from_mapping(v) for v in conf_list]

def parse_args():
    """Parse command line arguments"""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=str,
        help="Load YAML file from configs/ (e.g., paper_experiments/2wiki_gpt41.yaml)",
    )

    config_args, remaining = config_parser.parse_known_args()
    config_data = None
    config_path = None

    if config_args.config:
        base_dir = Path(__file__).resolve().parent.parent / "configs"
        potential_path = Path(config_args.config)
        if not potential_path.is_absolute():
            potential_path = base_dir / config_args.config
        config_path = potential_path
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

    config_dataset = (config_data or {}).get("dataset", {})
    config_model = (config_data or {}).get("model", {})
    config_methods = (config_data or {}).get("methods", {})
    config_eval = (config_data or {}).get("evaluation", {})

    def _extract_method_names(method_entries):
        names = []
        for entry in method_entries or []:
            if isinstance(entry, dict):
                name = entry.get("name")
                if name:
                    names.append(name)
            else:
                names.append(entry)
        return names

    default_dataset = config_dataset.get("name", "sci_q")
    default_split = config_dataset.get("split", "validation")
    default_num_samples = config_dataset.get("num_samples", 100)
    default_model = config_model.get("name", "gpt-4o-mini-2024-07-18")
    default_model_provider = config_model.get("provider")
    default_methods = _extract_method_names(config_methods.get("evidence_level")) or AVAILABLE_METHODS

    parser = argparse.ArgumentParser(
        description="Language model confidence calibration experiment",
        parents=[config_parser],
    )

    parser.add_argument(
        "--model",
        choices=AVAILABLE_MODELS,
        default=default_model,
        help="Model to use",
    )

    parser.add_argument(
        "--model_provider",
        type=str,
        default=default_model_provider,
        help="Model provider (configurable via config)",
    )

    parser.add_argument(
        "--dataset",
        choices=AVAILABLE_DATASETS,
        default=default_dataset,
        help="Dataset to use",
    )

    parser.add_argument(
        "--dataset_split",
        default=default_split,
        help="Dataset split to use",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=AVAILABLE_METHODS,
        default=default_methods,
        help="Methods to use (multiple allowed)",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=default_num_samples,
        help="Number of samples to use in experiment",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results",
    )

    parser.add_argument(
        "--result_file_path",
        type=str,
        default=None,
        help="For metrics calculation only",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="OpenAI API key (optional if already configured)",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display calibration plots",
    )

    parser.add_argument(
        "--language",
        choices=["en", "ja"],
        default="en",
        help="Language to use (en: English, ja: Japanese)",
    )

    parser.add_argument(
        "--test_suffix",
        type=str,
        default="",
        help="Test case suffix (starting with _)",
    )

    parser.add_argument(
        "--read_tmp_result",
        action="store_true",
        help="Load intermediate result file",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="Re-run evaluation even if intermediate result file exists",
    )

    parser.add_argument(
        "--input_pred",
        action="store_true",
        help="Input prediction results into prompt",
    )

    parser.add_argument(
        "--input_pattern",
        default="pattern1",
    )
    parser.add_argument(
        "--evaluation_model",
        type=str,
        default=config_eval.get("model"),
        help="Evaluation model (configurable via config)",
    )
    parser.add_argument(
        "--evaluation_method",
        type=str,
        default=config_eval.get("method"),
        help="Evaluation method (configurable via config)",
    )
    parser.add_argument(
        "--evaluation_temperature",
        type=float,
        default=config_eval.get("temperature"),
        help="Evaluation temperature (configurable via config)",
    )
    parser.add_argument(
        "--evaluation_max_tokens",
        type=int,
        default=config_eval.get("max_tokens"),
        help="Evaluation max tokens (configurable via config)",
    )
    parser.add_argument(
        "--evaluation_fuzzy_match_threshold",
        type=float,
        default=config_eval.get("fuzzy_match_threshold"),
        help="Evaluation fuzzy match threshold (configurable via config)",
    )

    args = parser.parse_args(remaining)

    evaluation_settings = {} if config_eval is None else dict(config_eval)
    for key, arg_name in [
        ("model", "evaluation_model"),
        ("method", "evaluation_method"),
        ("temperature", "evaluation_temperature"),
        ("max_tokens", "evaluation_max_tokens"),
        ("fuzzy_match_threshold", "evaluation_fuzzy_match_threshold"),
    ]:
        value = getattr(args, arg_name, None)
        if value is not None:
            evaluation_settings[key] = value
    args.evaluation_settings = evaluation_settings if evaluation_settings else None
    args.config_path = str(config_path) if config_path else None

    return args

def main():
    """Main execution function"""
    args = parse_args()

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    os.makedirs(args.output_dir, exist_ok=True)
    if args.read_tmp_result:
        overwrite = False
    else:
        overwrite = True

    print(f"Experiment configuration:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset} {args.dataset_split}")
    print(f"  Methods: {args.methods}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Test case: {args.test_suffix}")
    print(f"  Load intermediate file: {overwrite}")
    print(f"  Result file: {args.result_file_path}")
    if args.eval:print(f"  Re-run eval: {args.eval}")
    if args.input_pred:
        print(f"  Input predictions to prompt: {args.input_pred}")
        print(f"  Input prediction pattern: {args.input_pattern}")

    timestamp = time.strftime("%Y%m%d%H%M%S")
    run_id = build_run_id(
        dataset=args.dataset,
        split=args.dataset_split,
        model=args.model,
        methods=args.methods,
        test_id=args.test_suffix,
        num_samples=args.num_samples,
        timestamp=timestamp,
    )
    recorder = ExperimentRecorder(
        run_id=run_id,
        model=args.model,
        dataset=args.dataset,
        split=args.dataset_split,
    )
    dataset = load_dataset(args.dataset, split=args.dataset_split, max_samples=args.num_samples)
    
    if args.result_file_path and not args.eval:
        with open(args.result_file_path, 'r', encoding='utf-8') as f:
            print(f"load result_file : {args.result_file_path}")
            results = json.load(f)
        for method, v in results.items():
            v["answer_metrics"] = compute_metrics(v["answer_confidences"], v["answer_correctness"], model_name=method)
            conf_aligned, corr_aligned = align_conf_corr(v["org_triple_confidences"], v["org_triple_correctness"])
            v["triple_metrics"] = compute_metrics(conf_aligned, corr_aligned, model_name=method)
            results[method] = v

    elif args.input_pred:
        results = run_experiment_triple_input(
            model_name=args.model,
            dataset_name=args.dataset,
            split=args.dataset_split,
            methods=args.methods,
            num_samples=args.num_samples,
            language=args.language,
            test_suffix=args.test_suffix,
            input_pattern=args.input_pattern,
            overwrite=overwrite,
            debug=True
        )
    elif args.eval:

        results = reevaluate_correctness(
            args.model, args.dataset, args.dataset_split, args.methods,
            num_samples=args.num_samples, language=args.language,
            test_suffix=args.test_suffix, debug=True, result_file_path=args.result_file_path
        )
    else:
        results = run_experiment_triple(
            model_name=args.model,
            dataset_name=args.dataset,
            split=args.dataset_split,
            methods=args.methods,
            num_samples=args.num_samples,
            language=args.language,
            test_suffix=args.test_suffix,
            overwrite=overwrite,
            debug=True,
            recorder=recorder,
        )

    summary = {}
    for method, result in results.items():
        triple_correctness = [
                    (elem if elem is not None else 0.0)
                    for elem in result["triple_correctness"]
                ]

        triple_conf_per_question = result.get("org_triple_confidences", [])
        triple_corr_per_question = result.get("org_triple_correctness", [])

        evidence_correctness = []
        for corr_seq in triple_corr_per_question:
            try:
                corr_list = [to_float_or_nan(x) for x in corr_seq]
            except TypeError:
                evidence_correctness.append(np.nan)
                continue
            corr_list = [c for c in corr_list if not np.isnan(c)]
            if not corr_list:
                evidence_correctness.append(np.nan)
            else:
                evidence_correctness.append(1 if set(corr_list) == {1.0} else 0)

        spurious_roc_auc, spurious_pr_auc = compute_spurious_roc_pr_auc_for_questions(
            result.get("answer_correctness", []),
            evidence_correctness,
            triple_conf_per_question,
        )
        summary[method] = {
            "answer_accuracy": np.nanmean(result["answer_correctness"]),
            "answer_ece": result["answer_metrics"]["ece"],
            "answer_ece_t": result["answer_metrics"]["ece_t"],
            "answer_auc": result["answer_metrics"]["auc"],
            "answer_roc_auc": result["answer_metrics"].get("roc_auc"),
            "answer_pr_auc": result["answer_metrics"].get("pr_auc"),
            "answer_ece_temperatures": result["answer_metrics"]["ece_temperatures"],
            "answer_mean_temp_ece": result["answer_metrics"]["mean_temp_ece"],
            "answer_std_temp_ece": result["answer_metrics"]["std_temp_ece"],
            "answer_ece_fold_values": result["answer_metrics"]["ece_fold_values"],
            "answer_bs": result["answer_metrics"]["raw_bs"],
            "answer_bs_t": result["answer_metrics"]["bs_t"],
            "answer_bs_temperatures": result["answer_metrics"]["bs_temperatures"],
            "answer_mean_temp_bs": result["answer_metrics"]["mean_temp_bs"],
            "answer_std_temp_bs": result["answer_metrics"]["std_temp_bs"],
            "answer_bs_fold_values": result["answer_metrics"]["bs_fold_values"],
            "triple_accuracy": np.nanmean([to_float_or_nan(x) for x in triple_correctness]),
            "triple_ece": result["triple_metrics"]["ece"],
            "triple_ece_t": result["triple_metrics"]["ece_t"],
            "triple_auc": result["triple_metrics"]["auc"],
            "triple_roc_auc": result["triple_metrics"].get("roc_auc"),
            "triple_pr_auc": result["triple_metrics"].get("pr_auc"),
            "triple_spurious_roc_auc": spurious_roc_auc,
            "triple_spurious_pr_auc": spurious_pr_auc,
            "triple_ece_temperatures": result["triple_metrics"]["ece_temperatures"],
            "triple_mean_temp_ece": result["triple_metrics"]["mean_temp_ece"],
            "triple_std_temp_ece": result["triple_metrics"]["std_temp_ece"],
            "triple_ece_fold_values": result["triple_metrics"]["ece_fold_values"],
            "triple_bs": result["triple_metrics"]["raw_bs"],
            "triple_bs_t": result["triple_metrics"]["bs_t"],
            "triple_bs_temperatures": result["triple_metrics"]["bs_temperatures"],
            "triple_mean_temp_bs": result["triple_metrics"]["mean_temp_bs"],
            "triple_std_temp_bs": result["triple_metrics"]["std_temp_bs"],
            "triple_bs_fold_values": result["triple_metrics"]["bs_fold_values"],
        }

    print("\nExperiment results summary:")
    print(json.dumps(summary, indent=2))

    print("\n| Method | Accuracy | ECE ↓ | ECE-t ↓ | BS ↓ | BS-t ↓ | AUC ↑ | ROC ↑ |")
    print("|----------|------|--------|---------|-------|---------|-------|-------|")
    for method, metrics in summary.items():
        print(f"| {method}-A | {metrics['answer_accuracy']:.3f} | {metrics['answer_ece']:.3f} | {metrics['answer_ece_t']:.3f} | {metrics['answer_bs']:.3f} | {metrics['answer_bs_t']:.3f} | {metrics['answer_auc']:.3f} | |")
    for method, metrics in summary.items():
        print(f"| {method}-T | {metrics['triple_accuracy']:.3f} | {metrics['triple_ece']:.3f} | {metrics['triple_ece_t']:.3f} | {metrics['triple_bs']:.3f} | {metrics['triple_bs_t']:.3f} | {metrics['triple_auc']:.3f} | {metrics['triple_spurious_roc_auc']:.3f} |")
    
    samples = [dataset.get_sample(i) for i in range(len(dataset))]
    for method, result in results.items():
        answers = result.get("answers", [])
        confidences = result.get("answer_confidences", [])
        correctness = result.get("answer_correctness", [])
        recorder.record_answers(method, samples, answers, confidences, correctness)

        triple_preds = result.get("triples", [])
        triple_confidences = result.get("org_triple_confidences", [])
        triple_correctness = result.get("org_triple_correctness", [])

        structured_triples = []
        structured_confidences = []
        structured_correctness = []

        for preds in triple_preds:
            structured_triples.append([] if preds is None else [
                _parse_triple_prediction(p)
                for p in preds
            ])
        for confs in triple_confidences:
            normalized = _normalize_confidences(confs)
            structured_confidences.append(normalized)
        for corrs in triple_correctness:
            structured_correctness.append(corrs if corrs is not None else [])

        recorder.record_evidence(
            method=method,
            samples=samples,
            evidence_predictions=structured_triples,
            evid_confidences=structured_confidences,
            evid_correctness=structured_correctness,
        )

    recorder.write_summary(summary, timestamp)
    recorder.touch_empty_evidence()

    if args.plot:
        for method, result in results.items():
            plot_calibration(
                result["confidences"],
                result["correctness"],
                title=f"{args.model} - {args.dataset} - {method}"
            )

if __name__ == "__main__":
    main()