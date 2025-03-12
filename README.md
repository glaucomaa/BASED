---

# Based Transformer Proof-of-Concept

This project provides a proof-of-concept implementation of the "Based" architecture from the paper [Simple linear attention language models balance the recall‐throughput tradeoff](https://arxiv.org/abs/2402.18668). The implementation uses a combination of global linear attention via a second-order Taylor series approximation and local sliding-window softmax attention.

## Features

- **Pure PyTorch Implementation**
- **Hybrid Attention Mechanism:** Combines global linear attention for long-range interactions and local sliding-window attention for precise local recall.
- **Configurable Model Capacity** 

## File Structure

- **main.py:** Contains training, evaluation, and saving/loading routines.
- **model.py:** Contains the full model definition, data processing.
- **utils.py:** Contains several utility functions to support the model training process.
- **run.sh:** A Bash script to run the training script.
- **README.md:** This file.

## Requirements

- Python 3.7+
- PyTorch
- Hugging Face Datasets
- tqdm

## Usage

To train the model on the wikitext-2-raw-v1 dataset, simply run:
```bash
bash run.sh
```

## Neural Network Training Report

### Overview
This report summarizes the training progress of a neural language model across 8 epochs. The model shows steady improvement in training loss, while validation metrics indicate potential overfitting in later epochs.

### Training Metrics

| Epoch | Training Loss | Validation Loss | Perplexity |
|-------|---------------|-----------------|------------|
| 1     | 6.9349        | 6.0969          | 444.46     |
| 2     | 5.9711        | 5.8228          | 337.90     |
| 3     | 5.5517        | 5.7313          | 308.36     |
| 4     | 5.2387        | 5.7088          | 301.50     |
| 5     | 5.0027        | 5.7102          | 301.92     |
| 6     | 4.8359        | 5.7195          | 304.74     |
| 7     | 4.7317        | 5.7254          | 306.57     |
| 8     | 4.6836        | 5.7279          | 307.31     |

### Loss Progression

#### Training Loss
- **Initial loss:** 10.8660 (Epoch 1, Step 0)
- **Final loss:** 4.8001 (Epoch 8, Step 847)
- **Overall reduction:** 55.8%

#### Learning Rate Schedule
- **Starting LR:** 0.000010
- **Peak LR:** 0.000999 (Epoch 1, Step 200)
- **Final LR:** 0.000000 (Epoch 8)

### Performance Analysis

**Key Observations:**
1. **Training Loss:** Consistently decreased from 6.9349 to 4.6836 across epochs.
2. **Validation Loss:** Initially improved from 6.0969 to 5.7088 (Epoch 4), then gradually increased.
3. **Perplexity:** Significantly improved from 444.46 to 301.50 (Epoch 4) before slightly increasing.

**Convergence:**  
The model improves steadily until Epoch 5, then plateaus. Meanwhile, validation loss rises after Epoch 4, suggesting the model is overfitting to training data rather than learning general patterns.

### Conclusion
The model demonstrates promising results with a significant reduction in perplexity from 444.46 to 301.50. However, the divergence between training and validation metrics indicates there is room for improvement in generalization.

---
