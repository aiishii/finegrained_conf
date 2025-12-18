# File Manifest

This document lists all files included in the public release and their purpose.

## Documentation (4 files)

- `README.md` - Main documentation with setup and usage instructions
- `REPRODUCIBILITY.md` - Detailed guide for reproducing paper results
- `MANIFEST.md` - This file (list of all included files)
- `data/README.md` - Dataset download and preparation instructions

## Configuration (5 files)

- `requirements.txt` - Python package dependencies
- `.gitignore` - Git ignore patterns for outputs and secrets
- `configs/default.yaml` - Default configuration settings
- `configs/llm_config.example.yaml` - Template for API configuration
- `configs/paper_experiments/*.yaml` - Paper-specific experiment configs (2 files)

## Core Scripts (3 files)

Essential scripts for running experiments:

- `scripts/run_answer_experiment.py` - Run answer-level experiments
- `scripts/run_evidence_experiment.py` - Run evidence-level experiments (main)
- `scripts/evaluate_experiments.py` - Calculate metrics and evaluate results

## Source Package (23 Python files)

The `src/finegrained_conf/` package contains:

### Configuration (2 files)
- `config/__init__.py`
- `config/llm_config.py` - LLM API configuration management

### Datasets (2 files)
- `datasets/__init__.py`
- `datasets/data_utils.py` - Dataset loading and preprocessing

### Evaluation (3 files)
- `evaluation/__init__.py`
- `evaluation/llm_evaluator.py` - LLM-based automated evaluation
- `evaluation/metrics.py` - Calibration and discrimination metrics

### Experiments (4 files)
- `experiments/__init__.py`
- `experiments/answer_level.py` - Answer-level experiment runner
- `experiments/evidence_level.py` - Evidence-level experiment runner
- `experiments/utils.py` - Shared experiment utilities

### I/O (2 files)
- `io/__init__.py`
- `io/run_metadata.py` - Experiment metadata and result recording

### LLM Interface (4 files)
- `llm/__init__.py`
- `llm/api_wrapper.py` - Generic LLM API wrapper
- `llm/openai_client.py` - OpenAI/Azure specific client
- `llm/token_utils.py` - Token probability utilities

### Prompts (3 files)
- `prompts/__init__.py`
- `prompts/answer_prompts.py` - Answer-level prompt templates
- `prompts/evidence_prompts.py` - Evidence-level prompt templates (main)

### Utilities (2 files)
- `utils/__init__.py`
- `utils/parser.py` - Parsing utilities for triplets and responses

### Package Root (1 file)
- `__init__.py` - Package initialization

## Total Count

- **Documentation**: 4 files
- **Configuration**: 5 files
- **Scripts**: 3 files
- **Source code**: 23 Python files
- **Total**: 35 files

## Excluded from This Release

The following file types are **excluded** for public release:

- Migration scripts (`migrate_*.py`, `convert_*.py`, `batch_*.sh`)
- Analysis and comparison scripts (`analyze_*.py`, `compare_*.py`)
- Jupyter notebooks (`*.ipynb`)
- Legacy code (`src_old/`)
- Experimental outputs (`experiments/`, `selected_experiments/`)
- Result files (`*.tsv`, `*.csv`, `*.png`, `*.pdf`, `*.log`)
- Temporary files (`tmp_results/`, `output/`)

**Rationale**: This release focuses on **paper reproduction only**. Internal tools for migration, batch processing, and result analysis are specific to our experimental workflow and are not needed to replicate the paper results.

## File Size Estimate

- Source code: ~150 KB
- Documentation: ~40 KB
- Configuration: ~10 KB
- **Total**: ~200 KB (excluding datasets)

Datasets (not included, must be downloaded separately):
- JEMHopQA: ~5 MB
- 2WikiMultiHopQA: ~100 MB
