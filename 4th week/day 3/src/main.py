"""
main.py
-------
End-to-end script: loads Fashion-MNIST CSV, trains an ANN and a CNN,
evaluates both, prints the comparison table, saves the CNN model (.pth),
and saves all plots to /outputs.

Usage:
    python main.py --csv ../data/fashion_mnist.csv --epochs 10
"""

import argparse
import os
import torch
import pandas as pd

from dataset import get_dataloaders
from models import ANN, CNN, count_parameters
from train import train_model
from evaluate import get_predictions, compute_metrics, find_misclassified
from visualize import (
    plot_training_curves, plot_confusion_matrix,
    plot_sample_predictions, plot_misclassified
)


def run(csv_path, epochs=10, batch_size=64, lr=1e-3, optimizer_name="adam",
        outputs_dir="../outputs", models_dir="../models"):
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(csv_path, batch_size=batch_size)

    results = {}

    for name, model in [("ANN", ANN()), ("CNN", CNN())]:
        print(f"\n===== Training {name} =====")
        history, training_time = train_model(
            model, train_loader, val_loader, device,
            epochs=epochs, lr=lr, optimizer_name=optimizer_name
        )
        plot_training_curves(history, title_prefix=name,
                              save_path=f"{outputs_dir}/{name.lower()}_curves.png")

        images, y_true, y_pred = get_predictions(model, test_loader, device)
        metrics = compute_metrics(y_true, y_pred)

        plot_confusion_matrix(metrics["confusion_matrix"], title=f"{name} Confusion Matrix",
                               save_path=f"{outputs_dir}/{name.lower()}_confusion_matrix.png")
        plot_sample_predictions(images, y_true, y_pred,
                                 save_path=f"{outputs_dir}/{name.lower()}_sample_predictions.png")

        if name == "CNN":
            wrong_idx = find_misclassified(images, y_true, y_pred, n=10)
            plot_misclassified(images, y_true, y_pred, wrong_idx,
                                save_path=f"{outputs_dir}/{name.lower()}_misclassified.png")
            torch.save(model.state_dict(), f"{models_dir}/cnn_fashion_mnist.pth")
            print(f"Saved CNN weights to {models_dir}/cnn_fashion_mnist.pth")

        results[name] = {
            "Accuracy": round(metrics["accuracy"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "F1 Score": round(metrics["f1"], 4),
            "Training Time (s)": round(training_time, 1),
            "Number of Parameters": count_parameters(model),
        }

    comparison_df = pd.DataFrame(results).T
    print("\n===== ANN vs CNN Comparison =====")
    print(comparison_df)
    comparison_df.to_csv(f"{outputs_dir}/ann_vs_cnn_comparison.csv")

    return comparison_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="../data/fashion_mnist.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    args = parser.parse_args()

    run(args.csv, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, optimizer_name=args.optimizer)
