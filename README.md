# PEFT Scaling Laws: Sample Efficiency Scaling Laws for Parameter-Efficient Fine-Tuning

This repository contains the experimental data and source code for the paper:

> **How Many Examples Do You Need? Sample Efficiency Scaling Laws for Parameter-Efficient Fine-Tuning**
> Yaowen Sun, Gang Li, Hai Fu — Navy Submarine Academy, PLA, Qingdao, China

## Overview

This paper investigates a practical question: given N labeled examples, should one use LoRA or full fine-tuning? We provide a systematic empirical analysis across 2,263 training runs (2,127 unique configurations after deduplication) covering two architectural families, four adaptation methods, and seven sample sizes.

## Repository Structure

```
├── src/                          # Source code
│   ├── training.py               # Training loop (encoder-only models)
│   ├── training_causal.py        # Training loop (decoder-only models)
│   ├── constants.py              # Hyperparameter constants
│   ├── aggregate_results.py      # Result aggregation utilities
│   ├── statistical_analysis.py   # Statistical testing
│   ├── fit_scaling_law.py        # Power-law curve fitting E(N) = aN^{-b} + c
│   └── plot_scaling.py           # Figure generation scripts
├── data/                         # Experimental results
│   ├── results.csv               # 2,263 rows (2,127 unique configs; 136 duplicate runs included)
│   ├── results.json              # Same data in JSON format
│   ├── scaling_law_params.json   # Fitted power-law parameters per method
│   └── statistical_analysis.json # Pre-computed statistical tests
└── figures/                      # All paper figures (matplotlib output)
```

## Experimental Setup

### Encoder-only (2,208 runs)
- **Tasks**: CoLA, MRPC, QNLI, RTE, SST-2
- **Models**: DistilBERT, BERT-base, RoBERTa-base
- **Methods**: Full FT, BitFit, LoRA, TopHeavy-LoRA
- **Sample sizes**: 50, 100, 200, 500, 1000, 2000, 5000
- **Seeds**: 42, 123, 456, 789, 1024

### Decoder-only (55 runs)
- **Tasks**: CoLA, MRPC, SST-2
- **Model**: Qwen2.5-0.5B
- **Methods**: Full FT, LoRA
- **Sample sizes**: 100, 500, 2000
- **Seeds**: 42, 123, 456

## Hardware & Environment

All experiments were conducted on a single workstation:

| Component | Specification |
|---|---|
| CPU | Intel Core i9-12900K (16C/24T) |
| RAM | 128 GB DDR5 |
| GPU | NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM) |
| OS | Ubuntu 22.04 (WSL2) |

### Software Versions

| Package | Version |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| Transformers | 5.4.0 |
| PEFT | 0.18.1 |
| Datasets | 4.8.4 |
| scikit-learn | 1.8.0 |

## Key Results

- **Encoder-only**: PEFT methods match or beat full FT below ~700 examples. BitFit crosses over near N ≈ 705. TopHeavy-LoRA never crosses over in the observed range.
- **Decoder-only** (Qwen): Full FT wins at every sample size. No crossover observed.
- Power-law fits achieve R² up to 0.99 on encoder-only data.

## Requirements

```
torch>=2.11.0
transformers>=5.4.0
peft>=0.18.1
datasets>=4.8.0
scikit-learn>=1.8.0
scipy
matplotlib
numpy
```

## License

This repository is provided for academic reproducibility purposes. Please cite the paper if you use this code or data.

## Citation

```bibtex
@article{sun2026peftscaling,
  title={How Many Examples Do You Need? Sample Efficiency Scaling Laws for Parameter-Efficient Fine-Tuning},
  author={Sun, Yaowen and Li, Gang and Fu, Hai},
  year={2026}
}
```
