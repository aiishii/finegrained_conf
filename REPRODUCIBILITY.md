# Reproducibility Guide

This document provides detailed instructions for reproducing the main results from the IJCNLP-AACL 2025 paper.

## Paper Table 1: Evidence Confidence Extraction Performance

To reproduce Table 1 (triplet-level evidence confidence), run all methods on JEMHopQA:

```bash
# Create output directory
mkdir -p experiments/

# GPT-4.1-mini experiments
for method in triple_label_prob triple_verb_1s_top_1 triple_verb_1s_cot; do
    python scripts/run_evidence_experiment.py \
        --dataset jemhop_qa \
        --split train \
        --model gpt-4.1-mini-2025-04-14 \
        --method $method \
        --num_samples 1000 \
        --output_dir experiments/
done

# Llama-4-Maverick experiments
for method in triple_label_prob triple_verb_1s_top_1; do
    python scripts/run_evidence_experiment.py \
        --dataset jemhop_qa \
        --split train \
        --model Llama-4-Maverick-17B-128E-Instruct-FP8 \
        --method $method \
        --num_samples 1000 \
        --output_dir experiments/
done

# Phi-4 experiments
for method in triple_label_prob triple_verb_1s_top_1; do
    python scripts/run_evidence_experiment.py \
        --dataset jemhop_qa \
        --split train \
        --model Phi-4 \
        --method $method \
        --num_samples 1000 \
        --output_dir experiments/
done

# Evaluate all
python scripts/evaluate_experiments.py \
    --experiments_dir experiments/ \
    --output_file paper_table1_results.csv \
    --common-questions-mode
```

Expected metrics (from paper Table 1):
- **GPT-4.1-mini + Label prob.**: ECE=0.172, AUC=0.781
- **Llama-4-Maverick + Label prob.**: ECE=0.190, AUC=0.733
- **Phi-4 + Label prob.**: ECE=0.107, AUC=0.695

## Paper Table 2: Spurious Correctness Detection

To reproduce Table 2 (ROC-AUC for spurious correctness):

```bash
# Run evidence-level experiments (if not already done)
# Then evaluate with spurious correctness metrics
python scripts/evaluate_experiments.py \
    --experiments_dir experiments/ \
    --output_file paper_table2_results.csv \
    --common-questions-mode \
    --spurious-detection
```

Expected ROC-AUC (from paper Table 2):
- **GPT-4.1-mini**: 0.74 (evidence) vs 0.59 (answer)
- **Llama-4-Maverick**: 0.69 (evidence) vs 0.53 (answer)
- **Phi-4**: 0.84 (evidence) vs 0.65 (answer)

## Paper Table 3: Answer Confidence Improvement

To reproduce Table 3 (joint generation benefits):

```bash
# Answer-only baseline
python scripts/run_answer_experiment.py \
    --dataset jemhop_qa \
    --split train \
    --model gpt-4.1-mini-2025-04-14 \
    --method label_prob \
    --num_samples 1000 \
    --output_dir experiments/answer_only/

# Joint (answer + evidence) - use evidence experiment results
# Compare metrics between answer-only and joint

python scripts/evaluate_experiments.py \
    --experiments_dir experiments/ \
    --output_file paper_table3_comparison.csv \
    --compare-joint-vs-only
```

Expected improvements (from paper Table 3):
- **GPT-4.1-mini**: ECE 26% reduction, AUC +18%
- **Llama-4-Maverick**: ECE 43% reduction, AUC +32%

## Cross-lingual Validation (2WikiMultiHopQA)

To reproduce English dataset results:

```bash
# Run on 2WikiMultiHopQA
python scripts/run_evidence_experiment.py \
    --dataset 2wiki_qa \
    --split dev \
    --model gpt-4.1-2025-04-14 \
    --method triple_label_prob \
    --num_samples 300 \
    --output_dir experiments/2wiki/

python scripts/evaluate_experiments.py \
    --experiments_dir experiments/2wiki/ \
    --output_file 2wiki_results.csv
```

Expected (from paper Appendix B.1.1):
- **GPT-4.1 + Label prob.**: ECE=0.108, ROC-AUC=0.824

## Important Notes

### Sampling and Randomness

1. **Label prob. method**: Uses temperature=0.7, top-p=0.95, n=10 samples
   - Results may vary slightly across runs due to API randomness
   - The paper reports single-run results

2. **Other methods**: Use temperature=0.0 (deterministic)
   - Results should be exactly reproducible

### API Differences

- **Model versions**: Ensure you use the exact model versions specified
  - GPT-4.1-mini: `gpt-4.1-mini-2025-04-14`
  - Llama-4-Maverick: Azure AI Foundry, version Oct 2024
  - Phi-4: Azure AI Foundry, version Oct 2024

- **Token probability access**:
  - Token prob. method requires API access to token-level probabilities
  - Not available for all models on Azure AI Foundry Serverless

### Dataset Sampling

The paper uses:
- **JEMHopQA**: 1,000 samples from train split, 3 few-shot examples
- **2WikiMultiHopQA**: 300 samples from dev split

Few-shot exemplar IDs are hard-coded in the scripts to ensure reproducibility.

### Evaluation Details

1. **Automated evaluation**: Uses GPT-4.1 for judging evidence correctness
   - Agreement with human judgment: 93-100% (see paper Appendix A.2)
   - Results validated on 300 manual samples

2. **Temperature scaling**:
   - Applied using 5-fold cross-validation
   - Metrics with "-t" suffix use optimal temperature

### Computational Resources

Estimated API costs (at time of publication):
- Full JEMHopQA run (all methods, 1,000 samples): ~$50-100 USD
- Single method (1,000 samples): ~$10-20 USD
- 2WikiMultiHopQA (300 samples): ~$5-10 USD

Runtime:
- Label prob. (10 samples): ~30-60 min per 1,000 questions
- Verbalized methods: ~15-30 min per 1,000 questions

## Verification Checklist

After running experiments, verify:

- [ ] All experiment JSONL files generated in `experiments/`
- [ ] Evaluation CSV contains ECE, BS, AUC, ROC-AUC columns
- [ ] ECE values within ±0.05 of paper results (accounting for sampling variance)
- [ ] ROC-AUC values within ±0.03 of paper results
- [ ] Evidence confidence shows monotonic improvement over answer-level

## Troubleshooting

### API Errors

**Rate limits**:
```bash
# Add --rate_limit flag to slow down requests
python scripts/run_evidence_experiment.py ... --rate_limit 10  # 10 req/min
```

**Authentication issues**:
- Verify `configs/llm_config.yaml` contains correct API keys
- Check API key has sufficient quota

### Missing Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# If specific package missing
pip install <package-name>
```

### Path Issues

Ensure scripts are run from `public_release/` directory:
```bash
cd public_release/
python scripts/run_evidence_experiment.py ...
```

## Questions

If you encounter issues reproducing the results:

1. Check this guide and the main README
2. Verify dataset format matches expected structure
3. Ensure model versions match exactly
4. Open an issue: https://github.com/aiishii/finegrained_conf/issues
