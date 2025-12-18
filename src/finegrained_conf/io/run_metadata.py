from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Mapping, MutableMapping, Sequence


@dataclass
class TestInstance:
    dataset: str
    split: str
    question_index: int
    question_id: str
    dataset_native_id: str
    triple_index: int | None = None

    @property
    def answer_test_id(self) -> str:
        return build_test_id(self.dataset, self.split, self.question_id)

    @property
    def evidence_test_id(self) -> str:
        if self.triple_index is None:
            raise ValueError("triple_index is required for evidence test ids")
        return build_test_id(self.dataset, self.split, self.question_id, self.triple_index)

def build_run_id(dataset: str, split: str, model: str, methods: Sequence[str], test_id: str, num_samples: int, timestamp: str | None = None) -> str:
    ts = timestamp or datetime.utcnow().strftime("%Y%m%d%H%M%S")
    if methods:
        primary_method = methods[0]
        if len(methods) > 1:
            primary_method = f"{primary_method}+{len(methods)-1}"
    else:
        primary_method = "unknown"
    safe_model = model.replace("/", "-")
    return f"{dataset}-{split}-{safe_model}-{primary_method}{test_id}-{num_samples}-{ts}"


def build_test_id(dataset: str, split: str, question_id: str, triple_index: int | None = None) -> str:
    base = f"{dataset}-{split}-q{question_id}"
    if triple_index is not None:
        return f"{base}-t{triple_index}"
    return base


def _append_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


@dataclass
class ExperimentRecorder:
    run_id: str
    model: str
    dataset: str
    split: str
    run_dir: Path = field(default_factory=lambda: Path("experiments"))

    def __post_init__(self) -> None:
        self.run_dir = self.run_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.responses_path = self.run_dir / "responses.jsonl"
        self.answer_results_path = self.run_dir / "results_answer.jsonl"
        self.evidence_results_path = self.run_dir / "results_evidence.jsonl"
        self.summary_path = self.run_dir / "summary_metrics.json"

    def log_api_call(
        self,
        method: str,
        test_instance: TestInstance,
        call_role: str,
        sample_index: int,
        attempt: int,
        status: str,
        payload: Mapping[str, Any],
    ) -> None:
        timestamp = datetime.utcnow().isoformat() + "Z"
        test_id = (
            test_instance.answer_test_id
            if test_instance.triple_index is None
            else test_instance.evidence_test_id
        )

        record: MutableMapping[str, Any] = {
            "run_id": self.run_id,
            "test_id": test_id,
            "dataset": self.dataset,
            "split": self.split,
            "model": self.model,
            "method": method,
            "call_role": call_role,
            "sample_index": sample_index,
            "attempt": attempt,
            "status": status,
            "question_id": test_instance.question_id,
            "dataset_native_id": test_instance.dataset_native_id,
            "question_index": test_instance.question_index,
            "triple_index": test_instance.triple_index,
            "timestamp": timestamp,
        }
        record.update(payload)

        _append_jsonl(self.responses_path, [record])

    def record_answers(
        self,
        method: str,
        samples: Sequence[Mapping[str, Any]],
        answers: Sequence[Any],
        confidences: Sequence[Any],
        correctness: Sequence[Any],
    ) -> None:
        records: List[MutableMapping[str, Any]] = []

        for idx, sample in enumerate(samples):
            if idx >= len(answers):
                break
            question_id = str(sample.get("qid", sample.get("question_id", idx)))
            dataset_native_id = str(sample.get("qid", question_id))
            test_instance = TestInstance(
                dataset=self.dataset,
                split=self.split,
                question_index=idx,
                question_id=question_id,
                dataset_native_id=dataset_native_id,
            )
            records.append(
                {
                    "run_id": self.run_id,
                    "test_id": test_instance.answer_test_id,
                    "method": method,
                    "model": self.model,
                    "dataset": self.dataset,
                    "split": self.split,
                    "question_id": question_id,
                    "dataset_native_id": dataset_native_id,
                    "question_index": idx,
                    "prediction": answers[idx],
                    "confidence": confidences[idx] if idx < len(confidences) else None,
                    "correct": correctness[idx] if idx < len(correctness) else None,
                }
            )

        _append_jsonl(self.answer_results_path, records)

    def record_evidence(
        self,
        method: str,
        samples: Sequence[Mapping[str, Any]],
        evidence_predictions: Sequence[Sequence[Mapping[str, Any]]],
        evid_confidences: Sequence[Sequence[Any]],
        evid_correctness: Sequence[Sequence[Any]],
    ) -> None:
        records: List[MutableMapping[str, Any]] = []

        for idx, sample in enumerate(samples):
            question_id = str(sample.get("qid", sample.get("question_id", idx)))
            dataset_native_id = str(sample.get("qid", question_id))
            triple_preds = evidence_predictions[idx] if idx < len(evidence_predictions) else []
            triple_confidences = evid_confidences[idx] if idx < len(evid_confidences) else []
            triple_correctness = evid_correctness[idx] if idx < len(evid_correctness) else []

            for triple_idx, triple_pred in enumerate(triple_preds, start=1):
                test_instance = TestInstance(
                    dataset=self.dataset,
                    split=self.split,
                    question_index=idx,
                    question_id=question_id,
                    dataset_native_id=dataset_native_id,
                    triple_index=triple_idx,
                )
                confidence = triple_confidences[triple_idx - 1] if triple_idx - 1 < len(triple_confidences) else None
                correct = triple_correctness[triple_idx - 1] if triple_idx - 1 < len(triple_correctness) else None
                records.append(
                    {
                        "run_id": self.run_id,
                        "test_id": test_instance.evidence_test_id,
                        "method": method,
                        "model": self.model,
                        "dataset": self.dataset,
                        "split": self.split,
                        "question_id": question_id,
                        "dataset_native_id": dataset_native_id,
                        "question_index": idx,
                        "triple_index": triple_idx,
                        "answer_test_id": test_instance.answer_test_id,
                        "subject": triple_pred.get("subject"),
                        "relation": triple_pred.get("relation"),
                        "object": triple_pred.get("object"),
                        "confidence": confidence,
                        "correct": correct,
                    }
                )

        _append_jsonl(self.evidence_results_path, records)

    def write_summary(self, summary: Mapping[str, Any], timestamp: str) -> None:
        payload = {
            "run_id": self.run_id,
            "model": self.model,
            "dataset": self.dataset,
            "split": self.split,
            "timestamp": timestamp,
            "methods": summary,
        }
        self.summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def touch_empty_evidence(self) -> None:
        if not self.evidence_results_path.exists():
            self.evidence_results_path.write_text("", encoding="utf-8")

    def touch_empty_responses(self) -> None:
        if not self.responses_path.exists():
            self.responses_path.write_text("", encoding="utf-8")
