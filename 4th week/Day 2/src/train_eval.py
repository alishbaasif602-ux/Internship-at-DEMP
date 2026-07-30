"""
Training loop and evaluation utilities shared by the RNN, LSTM, and GRU
forecasters.
"""
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


def _to_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(model, X_train, y_train, X_test, y_test, device,
                 epochs=60, batch_size=32, lr=1e-3):
    """Train with Adam + MSE loss, tracking train/val loss per epoch."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_loader = _to_loader(X_train, y_train, batch_size, shuffle=True)
    X_test_t = torch.from_numpy(X_test).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)

    history = {"train_loss": [], "val_loss": []}
    start = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_preds = model(X_test_t)
            val_loss = criterion(val_preds, y_test_t).item()

        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_loss"].append(val_loss)

    train_time = time.time() - start
    return model, history, train_time


def evaluate_model(model, X_test, y_test, scaler, device):
    """Evaluate on the held-out test set, inverse-transformed to price scale."""
    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).to(device)
        preds_scaled = model(X_test_t).cpu().numpy()

    y_true = scaler.inverse_transform(y_test).flatten()
    y_pred = scaler.inverse_transform(preds_scaled).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2,
        "y_true": y_true, "y_pred": y_pred,
    }
