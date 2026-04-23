import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Return CUDA device if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    """
    Count the number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy_from_logits(logits, labels):
    """
    Compute binary accuracy from raw logits.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    correct = (preds == labels).float().sum()
    return correct.item() / labels.size(0)


class Timer:
    """
    Simple timer context manager.
    """
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.elapsed = self.end - self.start


def save_json(data, path):
    """
    Save dictionary to JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def plot_training_curves(history_by_model, save_path):
    """
    Plot validation accuracy by epoch for all benchmark models.

    history_by_model format:
    {
        "A": [{"epoch": 1, "val_accuracy": ...}, ...],
        "B": [{"epoch": 1, "val_accuracy": ...}, ...],
        ...
    }
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for model_name, history in history_by_model.items():
        epochs = [entry["epoch"] for entry in history]
        val_accuracies = [entry["val_accuracy"] for entry in history]
        plt.plot(epochs, val_accuracies, marker="o", label=f"Model {model_name}")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy by Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    print("Device:", get_device())
    print("Utils loaded successfully.")