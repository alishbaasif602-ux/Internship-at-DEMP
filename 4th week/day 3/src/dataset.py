"""
dataset.py
----------
Data loading, preprocessing, and PyTorch Dataset/DataLoader utilities
for the Fashion-MNIST CSV dataset (label, pixel1 ... pixel784).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def load_csv(path: str):
    """Load the Fashion-MNIST CSV and split into images (N,28,28) and labels (N,)."""
    df = pd.read_csv(path)
    labels = df["label"].values.astype(np.int64)
    images = df.drop(columns=["label"]).values.astype(np.float32)
    images = images.reshape(-1, 28, 28)
    return images, labels


def train_val_test_split(images, labels, val_size=0.1, test_size=0.15, seed=42):
    """Split into train / validation / test sets (stratified by label)."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=(val_size + test_size),
        stratify=labels, random_state=seed
    )
    relative_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test,
        stratify=y_temp, random_state=seed
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


class FashionMNISTDataset(Dataset):
    """PyTorch Dataset wrapping normalized Fashion-MNIST images."""

    def __init__(self, images, labels, augment=False):
        # Normalize pixel values to [0, 1]
        self.images = images.astype(np.float32) / 255.0
        self.labels = labels.astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def _augment(self, img):
        # Simple, cheap augmentations: random horizontal flip + small random shift
        if np.random.rand() < 0.5:
            img = np.fliplr(img).copy()
        if np.random.rand() < 0.5:
            shift_x, shift_y = np.random.randint(-2, 3, size=2)
            img = np.roll(img, shift_x, axis=1)
            img = np.roll(img, shift_y, axis=0)
        return img

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.augment:
            img = self._augment(img)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1,28,28)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return img_tensor, label_tensor


def get_dataloaders(csv_path, batch_size=64, val_size=0.1, test_size=0.15,
                     augment_train=True, seed=42, num_workers=2):
    """Convenience function: load CSV -> split -> build DataLoaders."""
    images, labels = load_csv(csv_path)
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        images, labels, val_size=val_size, test_size=test_size, seed=seed
    )

    train_ds = FashionMNISTDataset(X_train, y_train, augment=augment_train)
    val_ds = FashionMNISTDataset(X_val, y_val, augment=False)
    test_ds = FashionMNISTDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
