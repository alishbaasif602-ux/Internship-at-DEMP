# Sentiment Analysis using RNN, LSTM & GRU

**A comparative study of three recurrent neural network architectures for binary sentiment classification on the IMDb Movie Reviews Dataset.**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-supported-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)


## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models](#models)
- [How to Run](#how-to-run)
- [Results](#results)
- [Deliverables](#deliverables)
- [Key Findings](#key-findings)


## Overview

This project builds and compares three recurrent architectures — **Simple RNN**, **LSTM**, and **GRU** — for classifying IMDb movie reviews as **positive** or **negative**. All three models share an identical preprocessing pipeline, embedding layer, and training configuration, so that differences in performance can be attributed to the recurrent layer alone.

The project includes the underlying theory and mathematics for each architecture, a full implementation, a rigorous evaluation (Accuracy, Precision, Recall, F1-score, Confusion Matrix), and an analysis of *why* the results turn out the way they do.

## Project Structure

```
4th week/
├── IMDB_MOVIE_dataset.csv                  # Dataset (review, sentiment columns)
├── Sentiment_Analysis_RNN_LSTM_GRU.ipynb   # Main notebook — theory, math, diagrams, full implementation (TensorFlow/Keras)
├── train_pytorch_and_save.py               # PyTorch implementation — trains RNN/LSTM/GRU, saves best_model.pth
├── Performance_Comparison_Table.md         # Accuracy / Precision / Recall / F1 / training time per model
├── Sentiment_Analysis_Presentation.pptx    # 10-slide project summary with diagrams
├── best_model_<NAME>.keras / .h5           # Saved best model — Keras format (from the notebook)
├── best_model_<NAME>.pth                   # Saved best model — PyTorch format (from train_pytorch_and_save.py)
├── performance_comparison.csv              # Metrics exported by the PyTorch script
└── README.md
```

## Dataset

| | |
|---|---|
| **Source** | IMDb Movie Reviews |
| **Size** | 50,000 reviews |
| **Class balance** | 25,000 positive / 25,000 negative |
| **Columns** | `review` (raw text), `sentiment` (`positive` / `negative`) |

## Methodology

```
Raw Text → Cleaning → Tokenization → Vocabulary → Sequences → Padding → Model
```

1. **Cleaning** — lowercase, strip HTML tags (e.g. `<br />`), remove punctuation/digits, collapse whitespace.
2. **Tokenization** — split cleaned text into word tokens.
3. **Vocabulary** — top 10,000 most frequent words; rare words map to `<OOV>`.
4. **Sequences** — words converted to integer IDs.
5. **Padding** — every review fixed to length 200 (pad or truncate).

## Models

All three models use the same architecture skeleton — only the recurrent layer differs:

```
Input → Embedding(128) → [SimpleRNN | LSTM | GRU](64) → Dropout(0.4) → Dense(1, sigmoid)
```

| Model | Gates | Memory | Notes |
|---|---|---|---|
| Simple RNN | None | Single hidden state | Fastest, but loses long-range context (vanishing gradients) |
| LSTM | Forget, Input, Output | Hidden state + cell state | Best long-term memory control |
| GRU | Update, Reset | Single hidden state | ~25% fewer parameters than LSTM, faster to train |

## How to Run

### Option A — Notebook (TensorFlow/Keras)
```bash
pip install tensorflow scikit-learn pandas matplotlib seaborn
```
Open `Sentiment_Analysis_RNN_LSTM_GRU.ipynb` and run all cells. The best model is saved automatically as `best_model_<NAME>.keras` / `.h5`.

### Option B — PyTorch script (produces `.pth`)
```bash
pip install torch scikit-learn pandas numpy
python train_pytorch_and_save.py
```
Trains all three models, prints metrics, and saves `performance_comparison.csv` and `best_model_<NAME>.pth`.

> Both options require `IMDB_MOVIE_dataset.csv` in the same folder. After running either one, update `Performance_Comparison_Table.md` and the presentation's results slide with the real metrics produced.

## Results

Final numbers are produced by running the notebook or script locally (see `comparison_df` / `performance_comparison.csv`) and should be copied into [`Performance_Comparison_Table.md`](Performance_Comparison_Table.md):

| Model | Accuracy | Precision | Recall | F1-score | Training Time (s) |
|---|---|---|---|---|---|
| Simple RNN | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ |
| LSTM | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ |
| GRU | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ |

## Deliverables

- ✅ Jupyter Notebook — theory, mathematics, diagrams, full implementation (TensorFlow/Keras)
- ✅ PyTorch training script — saves best model as `.pth`
- ✅ Performance comparison table
- ✅ Presentation deck (10 slides, diagram-led)
- ✅ README (this file)

## Key Findings

- **Gated architectures (LSTM, GRU) outperform Simple RNN** on long text such as movie reviews, because their gating mechanisms avoid the vanishing-gradient problem that limits Simple RNN's memory to short spans.
- **GRU trains faster than LSTM** (fewer gates → fewer computations per time step) while typically achieving comparable accuracy — a good default when compute or time is limited.
- **LSTM gives the most explicit control over long-term memory** via its dedicated cell state, making it the strongest choice when handling very long or complex sequences.

*(Confirm these against your own run's `comparison_df` before final submission.)*
