# Fine-grained Confidence Estimation for Spurious Correctness Detection

This repository contains the official code for reproducing the experiments in the IJCNLP-AACL 2025 paper:

**"Fine-grained Confidence Estimation for Spurious Correctness Detection in Large Language Models"**
Ai Ishii, Naoya Inoue, Hisami Suzuki, Satoshi Sekine

## Overview

This work proposes a **fine-grained confidence estimation framework** that computes confidence scores for **individual evidence triplets** within reasoning chains, enabling precise localization of errors in LLM outputs.

### Key Features

- **Evidence-level confidence estimation**: Assigns confidence scores to (Subject, Relation, Object) triplets
- **Spurious correctness detection**: Identifies cases where answers are correct but reasoning contains errors
- **Multiple confidence methods**: Token probability, Label probability, Verbalized confidence
- **Cross-lingual evaluation**: Japanese (JEMHopQA) and English (2WikiMultiHopQA) datasets
- **Multiple models**: GPT-4.1, Llama-4-Maverick, Phi-4

### Main Results

- **Evidence confidence calibration**: Label prob. achieves ECE of 0.096-0.190 across models
- **Spurious correctness detection**: ROC-AUC up to 0.84 (Phi-4)
- **Answer calibration improvement**: Joint generation improves ECE by 26-43%

## Repository Structure

```
public_release/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
│
├── src/                     # Core package
│   └── finegrained_conf/
│       ├── config/          # Configuration management
│       ├── datasets/        # Dataset loading utilities
│       ├── evaluation/      # Metrics and evaluation
│       ├── experiments/     # Experiment runners
│       ├── io/              # Input/output utilities
│       ├── llm/             # LLM API wrappers
│       ├── prompts/         # Prompt templates
│       └── utils/           # General utilities
│
├── scripts/                 # Experiment scripts
│   ├── run_answer_experiment.py     # Answer-level experiments
│   ├── run_evidence_experiment.py   # Evidence-level experiments
│   └── evaluate_experiments.py      # Evaluation and metrics
│
├── configs/                 # Configuration files
│   ├── default.yaml         # Default configuration
│   ├── llm_config.example.yaml  # LLM API config template
│   └── paper_experiments/   # Paper-specific configs
│
├── data/                    # Datasets (not included)
│   └── README.md            # Data download instructions
│
└── examples/                # Usage examples (optional)
```

## Setup

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/aiishii/finegrained_conf.git
cd finegrained_conf/public_release

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Configuration

Create `configs/llm_config.yaml` from the template:

```bash
cp configs/llm_config.example.yaml configs/llm_config.yaml
```

Edit `configs/llm_config.yaml` and add your API keys:

```yaml
# For OpenAI models (GPT-4.1, GPT-4.1-mini, GPT-4.1-nano)
openai:
  api_key: "your-openai-api-key"
  organization: "your-org-id"  # Optional

# For Azure AI Foundry models (Llama-4-Maverick, Phi-4)
azure:
  api_key: "your-azure-api-key"
  endpoint: "your-endpoint-url"
```

**Important**: Never commit `llm_config.yaml` to version control.

### 3. Dataset Preparation

Download the required datasets following instructions in [`data/README.md`](data/README.md):

- **JEMHopQA** (Japanese): https://github.com/aiishii/JEMHopQA
- **2WikiMultiHopQA** (English): https://github.com/Alab-NII/2wikimultihop

Place datasets in:
```
data/
├── jemhop_qa/
│   ├── train_ver1.2.json
│   └── dev_ver1.2.json
└── 2wiki_qa/
    ├── train.json
    └── dev.json
```

## Running Experiments

### Evidence-Level Experiments (Main Results)

```bash
# Run evidence-level experiment with Label prob. method
python scripts/run_evidence_experiment.py \
    --dataset jemhop_qa \
    --split train \
    --model gpt-4.1-mini-2025-04-14 \
    --method triple_label_prob \
    --num_samples 1000 \
    --output_dir experiments/

# Run with verbalized confidence
python scripts/run_evidence_experiment.py \
    --dataset jemhop_qa \
    --split train \
    --model gpt-4.1-mini-2025-04-14 \
    --method triple_verb_1s_top_1 \
    --num_samples 1000 \
    --output_dir experiments/
```

### Answer-Level Experiments (Baseline)

```bash
# Run answer-only experiment
python scripts/run_answer_experiment.py \
    --dataset jemhop_qa \
    --split train \
    --model gpt-4.1-mini-2025-04-14 \
    --method label_prob \
    --num_samples 1000 \
    --output_dir experiments/
```

### Evaluation

```bash
# Evaluate all experiments and generate metrics
python scripts/evaluate_experiments.py \
    --experiments_dir experiments/ \
    --output_file evaluation_summary.csv

# Use common questions mode for fair comparison
python scripts/evaluate_experiments.py \
    --experiments_dir experiments/ \
    --output_file evaluation_summary_common.csv \
    --common-questions-mode
```

## Key Methods

The paper evaluates five confidence extraction methods:

### Model-based Methods
- **Token prob.**: Geometric mean of token-level probabilities (requires logprobs)
- **Label prob.**: Frequency-based confidence from N=10 samples (temperature=0.7)

### Verbalized Methods
- **Verb. 1S**: Single-pass verbalized confidence (numerical 0.00-1.00)
- **Verb. 1S CoT**: Verbalized with Chain-of-Thought reasoning first
- **Ling. 1S**: Linguistic expressions (e.g., "almost certain", "likely")

### Recommended Method

Based on the paper results, **Label prob.** is recommended for:
- Consistent performance across all models (ECE 0.096-0.190)
- Superior spurious correctness detection (ROC-AUC 0.69-0.84)
- Robustness across model architectures

## Expected Outputs

Experiments generate JSONL files with structure:
```json
{
  "question_id": "q001",
  "question": "Which director won...",
  "gold_answer": "Akira Kurosawa",
  "prediction": "Akira Kurosawa",
  "answer_confidence": 0.85,
  "evidence": [
    {
      "triple": ["Akira Kurosawa", "first Academy Award", "1951"],
      "confidence": 0.9,
      "correct": true
    },
    {
      "triple": ["Quentin Tarantino", "first Academy Award", "1994"],
      "confidence": 0.3,
      "correct": false
    }
  ],
  "answer_correct": true,
  "spurious_correct": true
}
```

## Metrics Reported

- **Calibration**: ECE (Expected Calibration Error), Brier Score
- **Discrimination**: AUC (Selective accuracy-coverage), ROC-AUC, PR-AUC
- **Temperature scaling**: ECE-t, BS-t (with optimal temperature)

## Key Files Included in This Release

### Essential for Reproduction:
- `scripts/run_evidence_experiment.py` - Main experiment runner
- `scripts/evaluate_experiments.py` - Metrics calculation
- `src/finegrained_conf/` - Complete package (23 files)
- `configs/paper_experiments/` - Paper-specific configurations

**Rationale**: This release focuses on **paper reproduction** only. Migration and analysis tools are for internal experimental workflows and are not needed to replicate the paper results.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{ishii2025finegrained,
  title={Fine-grained Confidence Estimation for Spurious Correctness Detection in Large Language Models},
  author={Ishii, Ai and Inoue, Naoya and Suzuki, Hisami and Sekine, Satoshi},
  booktitle={Proceedings of the 6th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 13th International Joint Conference on Natural Language Processing (IJCNLP-AACL 2025)},
  year={2025}
}
```

Related dataset:
```bibtex
@inproceedings{ishii2024jemhopqa,
  title={JEMHopQA: Dataset for Japanese Explainable Multi-Hop Question Answering},
  author={Ishii, Ai and Inoue, Naoya and Suzuki, Hisami and Sekine, Satoshi},
  booktitle={Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  pages={9515--9525},
  year={2024}
}
```

## License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

## Contact

- Ai Ishii: ai.ishii@jaist.ac.jp
- Project repository: https://github.com/aiishii/finegrained_conf

## Acknowledgments

This work was supported by BIPROGY Inc., which provided the computing environment and research funding.
