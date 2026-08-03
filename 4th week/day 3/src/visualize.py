"""
visualize.py
------------
Plotting helpers: loss/accuracy curves, confusion matrix heatmap,
sample predictions, and misclassified images.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import CLASS_NAMES


def plot_training_curves(history, title_prefix="", save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{title_prefix} Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc")
    axes[1].set_title(f"{title_prefix} Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(cm, title="Confusion Matrix", save_path=None):
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_sample_predictions(images, y_true, y_pred, n=10, save_path=None):
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    idxs = np.random.choice(len(images), size=n, replace=False)
    for ax, idx in zip(axes.flatten(), idxs):
        img = images[idx].squeeze()
        ax.imshow(img, cmap="gray")
        correct = y_true[idx] == y_pred[idx]
        color = "green" if correct else "red"
        ax.set_title(f"T:{CLASS_NAMES[y_true[idx]]}\nP:{CLASS_NAMES[y_pred[idx]]}",
                      color=color, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_misclassified(images, y_true, y_pred, indices, save_path=None):
    n = len(indices)
    if n == 0:
        print("No misclassified samples found.")
        return
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, idx in zip(axes, indices):
        img = images[idx].squeeze()
        ax.imshow(img, cmap="gray")
        ax.set_title(f"T:{CLASS_NAMES[y_true[idx]]}\nP:{CLASS_NAMES[y_pred[idx]]}",
                      color="red", fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
