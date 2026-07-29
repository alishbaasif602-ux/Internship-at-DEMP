"""
train_eval.py
-------------
Training loop, evaluation metrics (MAE, MSE, RMSE, R2), and helper to
time model training for the performance-comparison table.
"""

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def to_tensor(X, y, device):
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)  # [N, window, 1]
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(device)  # [N, 1]
    return X_t, y_t


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    epochs=50,
    batch_size=32,
    lr=1e-3,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_train_t, y_train_t = to_tensor(X_train, y_train, device)
    X_val_t, y_val_t = to_tensor(X_val, y_val, device)

    n_samples = X_train_t.shape[0]
    history = {"train_loss": [], "val_loss": []}

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        for i in range(0, n_samples, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        train_loss = epoch_loss / n_samples

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

    training_time = time.time() - start_time
    return model, history, training_time


def evaluate_model(model, X_test, y_test, scaler, device):
    """Runs inference, inverse-transforms predictions back to price scale,
    and computes MAE, MSE, RMSE, R2 on the ORIGINAL price scale (more
    interpretable than scaled [0,1] errors)."""
    model.eval()
    X_test_t, _ = to_tensor(X_test, y_test, device)
    with torch.no_grad():
        preds_scaled = model(X_test_t).cpu().numpy().flatten()

    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "y_true": y_true,
        "y_pred": y_pred,
    }
