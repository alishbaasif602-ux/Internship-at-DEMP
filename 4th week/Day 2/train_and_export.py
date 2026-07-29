import sys, os, time, json
sys.path.append("src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from preprocessing import prepare_dataset
from models import MODEL_REGISTRY
from train_eval import train_model, evaluate_model

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("saved_models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

WINDOW_SIZE = 30
TRAIN_RATIO = 0.8
EPOCHS = 60
BATCH_SIZE = 32
LR = 1e-3

data = prepare_dataset("data/stock_prices.csv", window_size=WINDOW_SIZE, train_ratio=TRAIN_RATIO)
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
scaler = data["scaler"]

results, histories, trained_models = {}, {}, {}

for name, ModelClass in MODEL_REGISTRY.items():
    print(f"--- Training {name} ---")
    torch.manual_seed(SEED)
    model = ModelClass(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
    trained_model, history, train_time = train_model(
        model, X_train, y_train, X_test, y_test, device,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
    )
    metrics = evaluate_model(trained_model, X_test, y_test, scaler, device)
    metrics["training_time_sec"] = train_time

    results[name] = metrics
    histories[name] = history
    trained_models[name] = trained_model

    # ---- Save model checkpoint (.pth) ----
    save_path = f"saved_models/{name.lower()}_model.pth"
    torch.save({
        "model_state_dict": trained_model.state_dict(),
        "model_class": name,
        "window_size": WINDOW_SIZE,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "metrics": {k: v for k, v in metrics.items() if k in ["MAE", "MSE", "RMSE", "R2", "training_time_sec"]},
    }, save_path)
    print(f"Saved {save_path}")
    print(f"{name} -> MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f} R2={metrics['R2']:.4f} time={train_time:.2f}s")

# ---- Performance comparison table (CSV + styled PNG) ----
remarks = {
    "RNN": "Fastest; weaker long-range memory",
    "LSTM": "Best long-range memory; most parameters",
    "GRU": "Near-LSTM accuracy; fewer params, faster",
}
rows = []
for name, m in results.items():
    rows.append({
        "Model": name,
        "MAE": round(m["MAE"], 3),
        "MSE": round(m["MSE"], 3),
        "RMSE": round(m["RMSE"], 3),
        "R2 Score": round(m["R2"], 3),
        "Training Time (s)": round(m["training_time_sec"], 2),
        "Remarks": remarks[name],
    })
summary_df = pd.DataFrame(rows).set_index("Model")
summary_df.to_csv("outputs/performance_comparison.csv")
print(summary_df)

best_model = summary_df["RMSE"].astype(float).idxmin()
with open("outputs/best_model.json", "w") as f:
    json.dump({"best_model": best_model}, f)
print("Best model (lowest RMSE):", best_model)

# Styled table image
fig, ax = plt.subplots(figsize=(11, 2.2))
ax.axis("off")
tbl = ax.table(
    cellText=summary_df.reset_index().values,
    colLabels=summary_df.reset_index().columns,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 2.0)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#1E2761")
        cell.set_text_props(color="white", weight="bold")
    else:
        cell.set_facecolor("#F5F7FF" if row % 2 == 0 else "white")
        if col == 0:
            cell.set_text_props(weight="bold")
plt.tight_layout()
plt.savefig("outputs/performance_comparison_table.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved outputs/performance_comparison_table.png")

# ---- Forecast graphs ----
# 1. Actual vs predicted per model
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
for ax, (name, m) in zip(axes, results.items()):
    ax.plot(m["y_true"], label="Actual", color="black", linewidth=1.2)
    ax.plot(m["y_pred"], label="Predicted", color="#2563eb", linewidth=1.2, alpha=0.85)
    ax.set_title(f"{name}: Actual vs. Predicted")
    ax.legend(); ax.grid(alpha=0.3)
axes[-1].set_xlabel("Test set time step")
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png", dpi=200)
plt.close()

# 2. Loss curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (name, h) in zip(axes, histories.items()):
    ax.plot(h["train_loss"], label="Train loss")
    ax.plot(h["val_loss"], label="Val loss")
    ax.set_title(name); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
axes[0].set_ylabel("MSE Loss (scaled space)")
plt.tight_layout()
plt.savefig("outputs/loss_curves.png", dpi=200)
plt.close()

# 3. Overlaid forecast comparison
plt.figure(figsize=(11, 5))
plt.plot(results["RNN"]["y_true"], label="Actual", color="black", linewidth=1.5)
plt.plot(results["RNN"]["y_pred"], label="RNN", linestyle="--", alpha=0.85)
plt.plot(results["LSTM"]["y_pred"], label="LSTM", linestyle="--", alpha=0.85)
plt.plot(results["GRU"]["y_pred"], label="GRU", linestyle="--", alpha=0.85)
plt.title("Forecast Comparison: RNN vs. LSTM vs. GRU")
plt.xlabel("Test set time step"); plt.ylabel("Price")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/forecast_comparison_all_models.png", dpi=200)
plt.close()

# 4. Bar chart comparison of metrics (useful for slides)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
metrics_to_plot = ["MAE", "RMSE", "R2 Score"]
colors = ["#2563eb", "#16a34a", "#dc2626"]
for ax, metname in zip(axes, metrics_to_plot):
    vals = summary_df[metname].astype(float)
    ax.bar(vals.index, vals.values, color=colors)
    ax.set_title(metname)
    ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("outputs/metric_bar_comparison.png", dpi=200)
plt.close()

print("All forecast graphs + tables saved to outputs/")
