<div align="center">

# Fashion-MNIST Image Classification
### Convolutional Neural Network vs. Fully-Connected ANN — PyTorch

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)](#)

A complete, from-scratch deep learning pipeline that classifies clothing images from the
Fashion-MNIST dataset, comparing a Convolutional Neural Network against a fully-connected
baseline across accuracy, efficiency, and error patterns.

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [Results at a Glance](#results-at-a-glance)
3. [CNN Theory & Concepts](#cnn-theory--concepts)
4. [Project Structure](#project-structure)
5. [Dataset](#dataset)
6. [Model Architectures](#model-architectures)
7. [Training Configuration](#training-configuration)
8. [Setup & Usage](#setup--usage)
9. [Evaluation & Visualizations](#evaluation--visualizations)
10. [Error Analysis](#error-analysis)
11. [Key Takeaways](#key-takeaways)
12. [Future Work](#future-work)
13. [License](#license)

---

## Overview

This project builds and compares two image classifiers on the **Fashion-MNIST** dataset
(60,000 labeled 28×28 grayscale clothing images across 10 categories):

- **ANN** — a fully-connected baseline network
- **CNN** — a convolutional network with batch normalization and pooling

Both models are trained under identical conditions (same data splits, optimizer, learning
rate, batch size, and epoch count) so that the only variable is architecture itself. The
project includes the full pipeline — data loading, preprocessing, training, evaluation,
visualization, and a saved deployable model — plus a presentation-ready results deck.

---

## Results at a Glance

| Metric | ANN | CNN | Winner |
|---|:---:|:---:|:---:|
| **Accuracy** | 86.69% | **91.51%** | CNN |
| **Precision (macro)** | 87.37% | **91.51%** | CNN |
| **Recall (macro)** | 86.69% | **91.51%** | CNN |
| **F1 Score (macro)** | 86.41% | **91.49%** | CNN |
| Training Time | **58.7 s** | 336.0 s | ANN |
| Parameters | 567,434 | **421,834** | CNN |

> The CNN outperforms the ANN on every classification metric **while using ~26% fewer
> parameters** — clear evidence that convolution + weight sharing extracts far more useful
> signal per parameter than a fully-connected stack on image data.

---

## CNN Theory & Concepts

The project notebook (`Fashion_MNIST_CNN.ipynb`) opens with a theory primer before the
practical assignment. The diagrams below are taken directly from that section.

### 1. Typical CNN Architecture Pipeline

A CNN processes an image through alternating convolution and pooling stages before a final
fully-connected classifier head.

![CNN Pipeline Architecture](assets/diagrams/cnn_pipeline_architecture.png)

### 2. The Convolution Operation

A small filter (kernel) slides across the input image, computing a dot product at each
position to produce a feature map that highlights local patterns such as edges and textures.

![Convolution Operation](assets/diagrams/convolution_operation.png)

### 3. Max Pooling

Pooling downsamples feature maps by keeping only the strongest activation in each window,
reducing spatial size while retaining the most salient features.

![Max Pooling Operation](assets/diagrams/max_pooling_operation.png)

### 4. Hierarchical Feature Learning

Successive convolutional layers learn increasingly abstract representations — from raw
edges, to corners and curves, to shapes, to full objects.

![Hierarchical Feature Learning](assets/diagrams/hierarchical_feature_learning.png)

### 5. Evolution of Popular CNN Architectures

Context for where this project's simple two-block CNN sits relative to landmark
architectures in deep learning history.

![CNN Architecture Timeline](assets/diagrams/cnn_architecture_timeline.png)

---

## Project Structure

```
fashion_mnist_cnn_project/
├── Fashion_MNIST_CNN.ipynb          # Main notebook — theory + full practical pipeline
├── Fashion_MNIST_CNN_Presentation.pptx  # 10-slide results presentation
├── README.md                        # This file
├── requirements.txt
├── data/
│   └── fashion_mnist.csv            # label + 784 pixel columns (28x28 grayscale)
├── src/
│   ├── dataset.py                   # CSV loading, splitting, Dataset/DataLoader
│   ├── models.py                    # ANN and CNN architectures
│   ├── train.py                     # Training / validation loop
│   ├── evaluate.py                  # Accuracy / precision / recall / F1 / confusion matrix
│   ├── visualize.py                 # Plotting: curves, confusion matrix, predictions
│   └── main.py                      # CLI script — runs the whole pipeline end-to-end
├── models/
│   └── cnn_fashion_mnist.pth        # Saved trained CNN weights
├── outputs/
│   ├── ann_vs_cnn_comparison.csv    # Performance comparison table
│   ├── ann_curves.png               # ANN training/validation curves
│   ├── cnn_curves.png               # CNN training/validation curves
│   ├── ann_confusion_matrix.png
│   ├── cnn_confusion_matrix.png
│   ├── ann_sample_predictions.png
│   ├── cnn_sample_predictions.png
│   └── cnn_misclassified.png
└── assets/
    └── diagrams/                    # Theory diagrams (extracted from the notebook)
```

---

## Dataset

**Fashion-MNIST** — a drop-in replacement for MNIST consisting of Zalando article images.

| Property | Value |
|---|---|
| Total images | 60,000 labeled, grayscale |
| Image size | 28 × 28 pixels |
| Classes | 10 clothing categories |
| Split | 75% train / 10% validation / 15% test (stratified) |
| Normalization | Pixel values scaled to `[0, 1]` |
| Augmentation | Random horizontal flip + small random shift (training set only) |

**Class labels:** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag,
Ankle boot.

---

## Model Architectures

### ANN (Baseline)

```
Flatten (784)
  → FC(512) → ReLU → Dropout
  → FC(256) → ReLU → Dropout
  → FC(128) → ReLU
  → FC(10)  [output]
```

### CNN

```
Conv2d(1→32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)   # 28×28 → 14×14
Conv2d(32→64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)  # 14×14 → 7×7
Flatten (64×7×7)
  → FC(128) → ReLU → Dropout
  → FC(10)  [output]
```

---

## Training Configuration

| Setting | Value |
|---|---|
| Loss function | Cross-Entropy Loss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 128 |
| Epochs | 10 |
| Device | CPU / GPU (auto-detected via `torch.cuda.is_available()`) |

Both models are trained with **identical** hyperparameters and data splits so that
architecture is the sole variable being compared.

---

## Setup & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

**Option A — Notebook (recommended for the full report/write-up):**

```bash
jupyter notebook Fashion_MNIST_CNN.ipynb
```

Run all cells top to bottom. Part A covers CNN theory; Part B trains, evaluates, and
visualizes both models end-to-end.

**Option B — Script:**

```bash
cd src
python main.py --csv ../data/fashion_mnist.csv --epochs 10 --batch_size 128 --lr 0.001 --optimizer adam
```

This trains both models, prints the ANN vs. CNN comparison table, saves all plots to
`outputs/`, saves the comparison table to `outputs/ann_vs_cnn_comparison.csv`, and saves
the CNN weights to `models/cnn_fashion_mnist.pth`.

### 3. Load the saved model for inference

```python
import torch
from src.models import CNN

model = CNN(num_classes=10)
model.load_state_dict(torch.load("models/cnn_fashion_mnist.pth", map_location="cpu"))
model.eval()
```

---

## Evaluation & Visualizations

### Training vs. Validation Curves

| ANN | CNN |
|---|---|
| ![ANN Curves](assets/results/ann_curves.png) | ![CNN Curves](assets/results/cnn_curves.png) |

The CNN converges faster, reaches a lower validation loss, and keeps a narrower gap between
training and validation accuracy — a sign of better generalization and less overfitting.

### Confusion Matrices

| ANN | CNN |
|---|---|
| ![ANN Confusion Matrix](assets/results/ann_confusion_matrix.png) | ![CNN Confusion Matrix](assets/results/cnn_confusion_matrix.png) |

### Sample Predictions (CNN)

![CNN Sample Predictions](assets/results/cnn_sample_predictions.png)

*Green titles indicate correct predictions; red indicates misclassifications.*

---

## Error Analysis

![CNN Misclassified Samples](assets/results/cnn_misclassified.png)

**Where the CNN still struggles:**

- **Shirt is the hardest class** — 129 shirts were misclassified as T-shirt/top and 56 as
  Coat. These garments share silhouette and texture at 28×28 resolution.
- **Upper-body garments overlap** — Pullover, Coat, and Dress are confused with each other
  more than with footwear or bags. The cleanest diagonals belong to Trouser, Sandal,
  Sneaker, and Bag — classes with distinctive shapes.

---

## Key Takeaways

- The CNN beat the ANN baseline on **every** classification metric — accuracy, precision,
  recall, and F1 — while using **26% fewer parameters**.
- Convolution + pooling captures local spatial patterns (edges, textures, shapes) that a
  flattened fully-connected network cannot exploit directly.
- The extra training time (336s vs. 59s) is a reasonable trade-off for a meaningfully more
  accurate and more parameter-efficient model.
- Remaining confusion is concentrated in visually similar upper-body garments — a natural
  next target for improvement.

---

## Future Work

- Deeper CNN / residual blocks for richer feature extraction
- Learning-rate scheduling for smoother convergence
- Test-time augmentation to boost robustness
- Class-weighted loss to address the Shirt/T-shirt/Coat confusion
- Lightweight architectures (e.g., MobileNet-style blocks) for deployment on edge devices

---

## Deliverables Checklist

- [x] Jupyter Notebook (`Fashion_MNIST_CNN.ipynb`) — theory + full practical pipeline
- [x] Source code (`src/`)
- [x] Saved model (`models/cnn_fashion_mnist.pth`)
- [x] Performance comparison table (`outputs/ann_vs_cnn_comparison.csv`)
- [x] Training/validation graphs (`outputs/*_curves.png`)
- [x] Presentation — `Fashion_MNIST_CNN_Presentation.pptx` (10 slides)
- [x] README.md (this file)

---

## License

This project is provided for educational purposes. Fashion-MNIST is released by Zalando
Research under the MIT License.

<div align="center">

---

*Built with PyTorch · Fashion-MNIST · CNN vs. ANN Comparison Study*

</div>
