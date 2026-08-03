"""
evaluate.py
-----------
Model evaluation utilities: accuracy/precision/recall/F1, confusion matrix,
and helpers to collect predictions + misclassified samples.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)


@torch.no_grad()
def get_predictions(model, loader, device):
    """Run the model over a DataLoader and return (images, y_true, y_pred)."""
    model.eval()
    all_images, all_labels, all_preds = [], [], []

    for images, labels in loader:
        images_dev = images.to(device)
        outputs = model(images_dev)
        preds = outputs.argmax(dim=1).cpu()

        all_images.append(images.numpy())
        all_labels.append(labels.numpy())
        all_preds.append(preds.numpy())

    images_arr = np.concatenate(all_images, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    preds_arr = np.concatenate(all_preds, axis=0)
    return images_arr, labels_arr, preds_arr


def compute_metrics(y_true, y_pred):
    """Return a dict of accuracy, macro precision/recall/F1, and the confusion matrix."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def find_misclassified(images, y_true, y_pred, n=10):
    """Return indices of up to n misclassified samples."""
    wrong_idx = np.where(y_true != y_pred)[0]
    return wrong_idx[:n]
