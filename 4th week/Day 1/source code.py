# %% [markdown]
# # Sentiment Analysis using RNN, LSTM & GRU
# ### IMDb Movie Reviews Dataset — Theory, Mathematics, Implementation & Comparison
# 
# **Objective:** Build, train and compare three recurrent architectures — **Simple RNN**, **LSTM**, and **GRU** — for binary sentiment classification (positive / negative) on the IMDb 50,000 movie review dataset, and analyse *why* one architecture outperforms the others.

# . Setup & Imports

# %%
!pip install seaborn

# %%
!pip install seaborn matplotlib pandas numpy scikit-learn nltk tensorflow

# %%
!pip install seaborn

# %%
import sys
!{sys.executable} -m pip install seaborn matplotlib pandas numpy scikit-learn nltk tensorflow

# %%
# Core libraries
import re
import time
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Deep learning
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

sns.set_style("whitegrid")
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# %% [markdown]
# <a id="7"></a>
# ## 7. Data Loading — IMDb Movie Reviews Dataset (local CSV)
# 
# We use the standard **IMDb dataset** (50,000 reviews, perfectly balanced: 25,000 positive / 25,000 negative), supplied as a local CSV with two columns: `review` (raw text) and `sentiment` (`positive` / `negative`).
# 
# Place the CSV in the **same folder as this notebook** (as in the project's `4th week/` folder) before running. The loader below checks a few common filenames automatically.
# 

# %%
import os

# Try the common filenames this dataset is usually saved under
candidate_paths = [
    "IMDB_MOVIE_dataset.csv",
    "IMDB MOVIE dataset.csv",
    "IMDB Dataset.csv",
    "imdb_dataset.csv",
]

csv_path = next((p for p in candidate_paths if os.path.exists(p)), None)
if csv_path is None:
    raise FileNotFoundError(
        "Could not find the IMDB CSV in the current folder. "
        "Update `csv_path` below to point at your file, e.g. csv_path = 'IMDB_MOVIE_dataset.csv'"
    )

df = pd.read_csv(csv_path)
print("Loaded:", csv_path)

# Map text labels -> 0/1
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

print("Total reviews:", len(df))
print(df["sentiment"].value_counts())
df.head()

# %% [markdown]
# <a id="8"></a>
# ## 8. Data Preprocessing
# 
# ### 8.1 Text Cleaning
# We remove HTML artifacts (e.g. `<br />` tags common in IMDb reviews), punctuation, digits, and extra whitespace, and lowercase everything.
# 
# ### 8.2 Tokenization & Vocabulary Building
# We use Keras' `Tokenizer` to split cleaned text into words and build a **vocabulary** — a mapping from the most frequent `VOCAB_SIZE` words to unique integer indices. Rare/out-of-vocabulary words are mapped to a single `<OOV>` token.
# 
# ### 8.3 Sequence Conversion
# Each review (a list of words) is converted into a list of integer indices using the vocabulary.
# 
# ### 8.4 Padding
# Reviews have different lengths, but neural networks need fixed-size input. We **pad** short reviews with zeros and **truncate** long reviews to a fixed `MAX_LEN`.
# 

# %%
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)                 # remove HTML tags e.g. <br />
    text = re.sub(r"[^a-z\s]", " ", text)               # keep only letters
    text = re.sub(r"\s+", " ", text).strip()            # collapse whitespace
    return text

df["clean_review"] = df["review"].apply(clean_text)
df[["review", "clean_review"]].head(3)

# %%
VOCAB_SIZE = 10000
MAX_LEN = 200

# Split BEFORE fitting the tokenizer (avoid test-set leakage into vocabulary)
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["clean_review"].values, df["sentiment"].values,
    test_size=0.2, random_state=SEED, stratify=df["sentiment"].values
)

# --- Tokenization & Vocabulary Building ---
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

print("Vocabulary size (fitted):", len(tokenizer.word_index))
print("Sample word -> index mapping:", dict(list(tokenizer.word_index.items())[:10]))

# --- Convert text to integer sequences ---
X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

# --- Padding ---
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

print("Train shape:", X_train_pad.shape, " Test shape:", X_test_pad.shape)

# %% [markdown]
# <a id='9'></a>
# ## 9. Exploratory Visualizations

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Class balance
sns.countplot(x=df["sentiment"].map({0: "Negative", 1: "Positive"}), ax=axes[0], palette=["#c53030", "#2c7a7b"])
axes[0].set_title("Class Distribution")
axes[0].set_xlabel("")

# Review length distribution (in words, before padding)
review_lengths = [len(t.split()) for t in df["clean_review"]]
sns.histplot(review_lengths, bins=50, ax=axes[1], color="#2b6cb0")
axes[1].axvline(MAX_LEN, color="#c05621", linestyle="--", label=f"MAX_LEN = {MAX_LEN}")
axes[1].set_title("Review Length Distribution (words)")
axes[1].legend()

plt.tight_layout()
plt.show()

print("Median review length:", int(np.median(review_lengths)), "words")
print(f"% of reviews longer than MAX_LEN ({MAX_LEN}):",
      round(100 * np.mean(np.array(review_lengths) > MAX_LEN), 2), "%")

# %% [markdown]
# <a id='10'></a>
# ## 10. Model Building

# %%
# Rendered diagram: Overall model architecture (pre-generated with matplotlib)
# Re-run make_diagrams.py or the cell above to regenerate if needed.
from IPython.display import Image, display
display(Image(filename='diagrams/model_architecture.png'))

# %%
EMBEDDING_DIM = 128
RNN_UNITS = 64
DROPOUT_RATE = 0.4

def build_model(rnn_type="RNN"):
    model = Sequential(name=f"{rnn_type}_SentimentModel")
    model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN))

    if rnn_type == "RNN":
        model.add(SimpleRNN(RNN_UNITS))
    elif rnn_type == "LSTM":
        model.add(LSTM(RNN_UNITS))
    elif rnn_type == "GRU":
        model.add(GRU(RNN_UNITS))
    else:
        raise ValueError("rnn_type must be 'RNN', 'LSTM', or 'GRU'")

    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
rnn_model = build_model("RNN")
lstm_model = build_model("LSTM")
gru_model = build_model("GRU")

# Build the models
rnn_model.build(input_shape=(None, MAX_LEN))
lstm_model.build(input_shape=(None, MAX_LEN))
gru_model.build(input_shape=(None, MAX_LEN))

# Show model summaries
print("========== RNN ==========")
rnn_model.summary()

print("\n========== LSTM ==========")
lstm_model.summary()

print("\n========== GRU ==========")
gru_model.summary()

# %%
lstm_model.summary()

# %%
gru_model.summary()

# %% [markdown]
# <a id="11"></a>
# ## 11. Training
# 
# Each model is trained under **identical conditions** (same data split, same epochs, same batch size, same early-stopping rule) so that the comparison in Section 13 is fair. We record **wall-clock training time** for each model to answer "which model trained fastest?" in the analysis.
# 

# %%
EPOCHS = 8
BATCH_SIZE = 128

early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

def train_model(model, name):
    print(f"\n{'='*20} Training {name} {'='*20}")
    start = time.time()
    history = model.fit(
        X_train_pad, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )
    elapsed = time.time() - start
    print(f"{name} training time: {elapsed:.2f} seconds")
    return history, elapsed

histories = {}
train_times = {}

histories["RNN"], train_times["RNN"] = train_model(rnn_model, "Simple RNN")
histories["LSTM"], train_times["LSTM"] = train_model(lstm_model, "LSTM")
histories["GRU"], train_times["GRU"] = train_model(gru_model, "GRU")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
colors = {"RNN": "#c53030", "LSTM": "#2b6cb0", "GRU": "#2c7a7b"}

for name, hist in histories.items():
    axes[0].plot(hist.history["accuracy"], label=f"{name} (train)", color=colors[name])
    axes[0].plot(hist.history["val_accuracy"], "--", label=f"{name} (val)", color=colors[name])
    axes[1].plot(hist.history["loss"], label=f"{name} (train)", color=colors[name])
    axes[1].plot(hist.history["val_loss"], "--", label=f"{name} (val)", color=colors[name])

axes[0].set_title("Accuracy over Epochs"); axes[0].set_xlabel("Epoch"); axes[0].legend(fontsize=8)
axes[1].set_title("Loss over Epochs"); axes[1].set_xlabel("Epoch"); axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# <a id="12"></a>
# ## 12. Evaluation — Accuracy, Precision, Recall, F1-score, Confusion Matrix
# 
# - **Accuracy** = $\frac{TP+TN}{TP+TN+FP+FN}$ — overall correctness.
# - **Precision** = $\frac{TP}{TP+FP}$ — of predicted positives, how many were actually positive.
# - **Recall** = $\frac{TP}{TP+FN}$ — of actual positives, how many did we correctly catch.
# - **F1-score** = $2 \cdot \frac{Precision \cdot Recall}{Precision+Recall}$ — harmonic mean, balances precision & recall.
# 

# %%
models = {"RNN": rnn_model, "LSTM": lstm_model, "GRU": gru_model}
results = {}

for name, model in models.items():
    y_prob = model.predict(X_test_pad, verbose=0)
    y_pred = (y_prob > 0.5).astype(int).ravel()

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Training Time (s)": train_times[name],
        "y_pred": y_pred
    }
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    ax.set_title(f"{name} — Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.show()

# %% [markdown]
# <a id='13'></a>
# ## 13. Comparison Table & Charts

# %%
comparison_df = pd.DataFrame({
    name: {
        "Accuracy": round(res["Accuracy"], 4),
        "Precision": round(res["Precision"], 4),
        "Recall": round(res["Recall"], 4),
        "F1": round(res["F1"], 4),
        "Training Time (s)": round(res["Training Time (s)"], 1)
    } for name, res in results.items()
}).T

comparison_df.index.name = "Model"
comparison_df

# %%
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1"]
fig, ax = plt.subplots(figsize=(9, 5))
comparison_df[metrics_to_plot].plot(kind="bar", ax=ax,
                                     color=["#4c51bf", "#2b6cb0", "#2c7a7b", "#c05621"])
ax.set_title("Model Comparison — Accuracy / Precision / Recall / F1")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.legend(loc="lower right")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
comparison_df["Training Time (s)"].plot(kind="bar", ax=ax, color=["#c53030", "#2b6cb0", "#2c7a7b"])
ax.set_title("Training Time Comparison")
ax.set_ylabel("Seconds")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# 


