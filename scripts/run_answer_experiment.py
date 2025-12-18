#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Language model confidence calibration experiment script

Runs reproduction experiments for the "Just Ask for Calibration" paper.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from finegrained_conf.experiments.answer_level import (
    run_experiment,
)
from finegrained_conf.datasets.data_utils import load_dataset
from finegrained_conf.io.run_metadata import (
    ExperimentRecorder,
    TestInstance,
    build_run_id,
)

AVAILABLE_MODELS = ["gpt-4.1-mini-2025-04-14", "gpt-4o-mini-2024-07-18", "gpt-4o-2024-11-20", "gpt-4.1-2025-04-14", "Llama-4-Maverick-17B-128E-Instruct-FP8", "Llama-3.3-70B-Instruct", "Phi-4", "gpt-4.1-nano-2025-04-14"]
AVAILABLE_DATASETS = ["trivia_qa", "sci_q", "truthful_qa", "jemhop_qa", "hotpot_qa", "2wiki_qa"]

AVAILABLE_METHODS = [
    "label_prob",
    "label_prob_cot",
    "logprob",
    "logprob_cot",
    "is_true_prob",
    "is_true_logprob",
    "is_true_2s_logprob",
    "verb_1s_top_1",
    "verb_1s_top_2",
    "verb_1s_top_4",
    "verb_2s_top_1",
    "verb_2s_top_2",
    "verb_2s_top_4",
    "verb_1s_cot",
    "verb_2s_cot",
    "ling_1s_human",
    "ling_2s_human",
    "ling_1s_human7",
    "ling_2s_human7"
]

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
    default_methods = _extract_method_names(config_methods.get("answer_level")) or AVAILABLE_METHODS

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

    timestamp = time.strftime("%Y%m%d%H%M%S")
    run_id = build_run_id(
        dataset=args.dataset,
        split=args.dataset_split,
        model=args.model,
        methods=args.methods,
        timestamp=timestamp,
    )

    print(f"Experiment configuration:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Methods: {args.methods}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Test case: {args.test_suffix}")
    print(f"  Load intermediate file: {overwrite}")
    if args.input_pred:
        print(f"  Input predictions to prompt: {args.input_pred}")
    print(f"  run_id: {run_id}")
    recorder = ExperimentRecorder(
        run_id=run_id,
        model=args.model,
        dataset=args.dataset,
        split=args.dataset_split,
    )
    dataset_name = args.dataset
    split_name = args.dataset_split
    dataset = load_dataset(dataset_name, split=split_name, max_samples=args.num_samples)

    samples = []
    test_instances = []
    for idx in range(len(dataset)):
        sample = dataset.get_sample(idx)

        if "question_index" not in sample or sample["question_index"] is None:
            sample["question_index"] = idx

        if "question_id" not in sample or sample["question_id"] in (None, ""):
            if sample.get("id") not in (None, ""):
                sample["question_id"] = sample["id"]
            else:
                sample["question_id"] = sample["question_index"]

        if "dataset_native_id" not in sample or sample["dataset_native_id"] in (None, ""):
            if sample.get("id") not in (None, ""):
                sample["dataset_native_id"] = sample["id"]
            else:
                sample["dataset_native_id"] = sample["question_id"]

        samples.append(sample)

        test_instances.append(
            TestInstance(
                dataset=dataset_name,
                split=split_name,
                question_index=sample["question_index"],
                question_id=str(sample["question_id"]),
                dataset_native_id=str(sample["dataset_native_id"]),
            )
        )

    if args.input_pred:
        results = run_experiment_input(
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
            test_instances=test_instances,
        )
    else:
        results = run_experiment(
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
            test_instances=test_instances,
        )

    # Organize and display results
    summary = {}
    for method, result in results.items():
        summary[method] = {
            "accuracy": np.mean(result["answer_correctness"]),
            "ece": result["answer_metrics"]["ece"],
            "ece_t": result["answer_metrics"]["ece_t"],
            "auc": result["answer_metrics"]["auc"],
            "answer_roc_auc": result["answer_metrics"].get("roc_auc"),
            "answer_pr_auc": result["answer_metrics"].get("pr_auc"),
            "ece_temperatures": result["answer_metrics"]["ece_temperatures"],
            "mean_temp_ece": result["answer_metrics"]["mean_temp_ece"],
            "std_temp_ece": result["answer_metrics"]["std_temp_ece"],
            "ece_fold_values": result["answer_metrics"]["ece_fold_values"],
            "bs": result["answer_metrics"]["raw_bs"],
            "bs_t": result["answer_metrics"]["bs_t"],
            "bs_temperatures": result["answer_metrics"]["bs_temperatures"],
            "mean_temp_bs": result["answer_metrics"]["mean_temp_bs"],
            "std_temp_bs": result["answer_metrics"]["std_temp_bs"],
            "bs_fold_values": result["answer_metrics"]["bs_fold_values"],
        }

        answers = result.get("answers", [])
        confidences = result.get("answer_confidences", [])
        correctness = result.get("answer_correctness", result.get("correctness", []))
        recorder.record_answers(method, samples, answers, confidences, correctness)

    print("\nExperiment results summary:")
    print(json.dumps(summary, indent=2))

    print("\n| Method | Accuracy | ECE ↓ | ECE-t ↓ | BS ↓ | BS-t ↓ | AUC ↑ |")
    print("|---------|-------|--------|---------|-------|---------|-------|")
    for method, metrics in summary.items():
        print(f"| {method} | {metrics['accuracy']:.3f} | {metrics['ece']:.3f} | {metrics['ece_t']:.3f} | {metrics['bs']:.3f} | {metrics['bs_t']:.3f} | {metrics['auc']:.3f} |")
    recorder.write_summary(summary, timestamp)
    recorder.touch_empty_evidence()
    recorder.touch_empty_responses()

    if args.plot:
        for method, result in results.items():
            plot_calibration(
                result["confidences"],
                result["correctness"],
                title=f"{args.model} - {args.dataset} - {method}"
            )
    

if __name__ == "__main__":
    main()