"""
Sentiment Analysis (IMDb) — PyTorch version
Trains Simple RNN, LSTM, and GRU models and saves the best one as a .pth checkpoint.

Run this locally (needs `pip install torch scikit-learn pandas numpy`):
    python train_pytorch_and_save.py

Expects IMDB_MOVIE_dataset.csv (columns: review, sentiment) in the same folder.
Produces:
    - best_model_<NAME>.pth        (state_dict checkpoint of the best model)
    - performance_comparison.csv   (Accuracy/Precision/Recall/F1/Training time per model)
"""

import re
import time
import json
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
VOCAB_SIZE = 10000
MAX_LEN = 200
EMBEDDING_DIM = 128
RNN_UNITS = 64
DROPOUT_RATE = 0.4
BATCH_SIZE = 128
EPOCHS = 8
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

CSV_CANDIDATES = [
    "IMDB_MOVIE_dataset.csv",
    "IMDB MOVIE dataset.csv",
    "IMDB Dataset.csv",
    "imdb_dataset.csv",
]

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
import os
csv_path = next((p for p in CSV_CANDIDATES if os.path.exists(p)), None)
if csv_path is None:
    raise FileNotFoundError(
        "Could not find the IMDB CSV in the current folder. "
        "Set csv_path manually, e.g. csv_path = 'IMDB_MOVIE_dataset.csv'"
    )
df = pd.read_csv(csv_path)
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
print(f"Loaded {csv_path} — {len(df)} reviews")

# ---------------------------------------------------------------------------
# 2. Clean text
# ---------------------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_review"] = df["review"].apply(clean_text)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["clean_review"].values, df["sentiment"].values,
    test_size=0.2, random_state=SEED, stratify=df["sentiment"].values
)

# ---------------------------------------------------------------------------
# 3. Build vocabulary (word -> index), convert to sequences, pad
# ---------------------------------------------------------------------------
PAD_IDX, OOV_IDX = 0, 1

counter = Counter()
for text in X_train_text:
    counter.update(text.split())

most_common = counter.most_common(VOCAB_SIZE - 2)  # reserve 0=PAD, 1=OOV
word2idx = {word: i + 2 for i, (word, _) in enumerate(most_common)}
print("Vocabulary size:", len(word2idx) + 2)

def text_to_seq(text):
    return [word2idx.get(w, OOV_IDX) for w in text.split()]

def pad_seq(seq, max_len=MAX_LEN):
    seq = seq[:max_len]
    return seq + [PAD_IDX] * (max_len - len(seq))

X_train_pad = np.array([pad_seq(text_to_seq(t)) for t in X_train_text], dtype=np.int64)
X_test_pad = np.array([pad_seq(text_to_seq(t)) for t in X_test_text], dtype=np.int64)

# ---------------------------------------------------------------------------
# 4. Dataset / DataLoader
# ---------------------------------------------------------------------------
class ReviewsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(ReviewsDataset(X_train_pad, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(ReviewsDataset(X_test_pad, y_test), batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------------------------
# 5. Models — Simple RNN, LSTM, GRU (same architecture, different recurrent layer)
# ---------------------------------------------------------------------------
class SentimentModel(nn.Module):
    def __init__(self, rnn_type, vocab_size, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(embed_dim, hidden_dim, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        # hidden: (1, batch, hidden_dim) for RNN/GRU; for LSTM it's a tuple (h_n, c_n)
        last_hidden = hidden[0] if isinstance(hidden, tuple) else hidden
        last_hidden = last_hidden.squeeze(0)
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out.squeeze(1)  # logits

# ---------------------------------------------------------------------------
# 6. Train / evaluate helpers
# ---------------------------------------------------------------------------
def train_model(model, name):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\n{'='*20} Training {name} {'='*20}")
    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"  Epoch {epoch+1}/{EPOCHS} — loss: {avg_loss:.4f}")
    elapsed = time.time() - start
    print(f"{name} training time: {elapsed:.2f}s")
    return elapsed

def evaluate_model(model):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds.tolist())
            all_labels.extend(yb.numpy().astype(int).tolist())
    return {
        "Accuracy": accuracy_score(all_labels, all_preds),
        "Precision": precision_score(all_labels, all_preds),
        "Recall": recall_score(all_labels, all_preds),
        "F1": f1_score(all_labels, all_preds),
    }

# ---------------------------------------------------------------------------
# 7. Train all three models
# ---------------------------------------------------------------------------
results = {}
models = {}

for rnn_type in ["RNN", "LSTM", "GRU"]:
    model = SentimentModel(rnn_type, VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, DROPOUT_RATE)
    elapsed = train_model(model, rnn_type)
    metrics = evaluate_model(model)
    metrics["Training Time (s)"] = round(elapsed, 1)
    results[rnn_type] = metrics
    models[rnn_type] = model
    print(f"{rnn_type} — {metrics}")

# ---------------------------------------------------------------------------
# 8. Comparison table
# ---------------------------------------------------------------------------
comparison_df = pd.DataFrame(results).T
comparison_df.index.name = "Model"
comparison_df.to_csv("performance_comparison.csv")
print("\nComparison table:\n", comparison_df)
print("Saved: performance_comparison.csv")

# ---------------------------------------------------------------------------
# 9. Save the best model as .pth
# ---------------------------------------------------------------------------
best_name = comparison_df["F1"].astype(float).idxmax()
best_model = models[best_name]

checkpoint = {
    "model_state_dict": best_model.state_dict(),
    "rnn_type": best_name,
    "vocab_size": VOCAB_SIZE,
    "embedding_dim": EMBEDDING_DIM,
    "hidden_dim": RNN_UNITS,
    "dropout": DROPOUT_RATE,
    "max_len": MAX_LEN,
    "metrics": results[best_name],
}
torch.save(checkpoint, f"best_model_{best_name}.pth")

# Also persist the vocabulary so the saved model can be reused for inference later
with open("vocab.json", "w") as f:
    json.dump(word2idx, f)

print(f"\nBest model: {best_name}  (F1 = {results[best_name]['F1']:.4f})")
print(f"Saved: best_model_{best_name}.pth, vocab.json")

# --- To reload later ---
# ckpt = torch.load("best_model_<NAME>.pth", map_location="cpu")
# model = SentimentModel(ckpt["rnn_type"], ckpt["vocab_size"], ckpt["embedding_dim"],
#                         ckpt["hidden_dim"], ckpt["dropout"])
# model.load_state_dict(ckpt["model_state_dict"])
# model.eval()
