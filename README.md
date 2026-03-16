# SENSE: Efficient EEG-to-Text via Privacy-Preserving Semantic Retrieval

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**SENSE (SEmantic Neural Sparse Extraction)** is a lightweight, privacy-preserving framework that translates non-invasive electroencephalography (EEG) signals into fluent natural language without the need for memory-intensive Large Language Model (LLM) fine-tuning.


## Key Features

* **Modular Architecture**: Decouples neural decoding into local semantic retrieval and prompt-based language generation.
* **Privacy-Preserving**: Raw EEG signals remain on-device; only abstract semantic keywords (Bag-of-Words) are shared with LLM APIs.
* **Lightweight**: The EEG-to-keyword module contains only ~6M parameters and completes training in minutes.
* **Zero-Shot Generation**: Leverages off-the-shelf LLMs (Gemini, GPT-4, LLaMA) via structured prompting to synthesize text.
* **Cross-Subject Generalization**: Demonstrates consistent performance across diverse neural patterns without per-subject calibration.

---

## Codebase Structure

```text
SENSE/
├── run_pipeline.py          # Main orchestrator (Encoding -> Alignment -> LLM -> Eval)
├── env.yaml                 # Conda environment configuration
├── src/
│   ├── models.py            # MLP Similarity Refiner & Loss functions (Focal, Contrastive, BCE)
│   ├── trainer.py           # Training logic & N-Hot Vector Encoding
│   ├── aligner.py           # Semantic alignment and Cosine Similarity engine
│   ├── encoders.py          # Frozen ChannelNet EEG-to-CLIP latent extraction
│   ├── llm_client.py        # Multi-provider LLM manager (OpenAI, Gemini, Together AI)
│   └── metrics.py           # Evaluation suite (BLEU, ROUGE, METEOR, BERTScore)
├── scripts/
│   └── prepare_dataset.py   # Pre-processing script for ImageNet-EEG data
├── data/                    # Storage for .pth datasets and word corpus
├── checkpoints/             # Trained MLP weights (.pth)
└── results/                 # Generated captions and metrics
```
---

## Installation
### 1. Create Conda Environment
```
conda env create -f env.yaml
conda activate sense_env
```

### 2. Configure API Keys

SENSE supports multiple LLM providers.

Create a `.env` file in the repository root:

```
OPENAI_API_KEY=your_key_here
TOGETHER_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```
---
## Dataset Preparation

The ImageNet-EEG dataset may be distributed across multiple folders.

Use the provided preprocessing script to consolidate everything into a single `.pth` dataset.

```python scripts/prepare_dataset.py```

---

## Running the pipeline
All experiments are controlled via `run_pipeline.py`. You can toggle between three main modes: 

### 1. Naive Baseline (No Training)

Direct cosine similarity between EEG latents and CLIP vocabulary embeddings.

``` 
python run_pipeline.py
--dataset data/eeg_test.pth
--mode naive
--top_k 15
```
### 2. Train Similarity Refiner
Trains the MLP projection layer that aligns EEG embeddings with the CLIP semantic space.

```
python run_pipeline.py
--dataset data/imagenet_eeg_train.pth
--mode train
--loss focal
--epochs 50
```
Available loss functions:

- `bce`
- `contrastive`
- `focal`

The trained model is saved in `checkpoints/`

### 3. Inference + Caption Generation

Runs the full pipeline:
```
EEG → semantic retrieval → LLM caption generation → evaluation
```
```python run_pipeline.py
--dataset data/eeg_test.pth
--mode inference
--checkpoint checkpoints/mlp_focal_100eps.pth
--loss focal
```

---
## Privacy Design

SENSE is designed with **neural data privacy in mind**.

Raw EEG signals never leave the local device, only **abstract semantic tokens** are sent to LLMs.

This allows:

- local deployment
- privacy-preserving BCI systems
- compatibility with external APIs

---

## Experimental Dataset

Experiments use the **ImageNet EEG dataset**:

- 128-channel EEG
- 6 subjects
- ~10k EEG samples
- images paired with captions

