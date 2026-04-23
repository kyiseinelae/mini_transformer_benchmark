import argparse
import os
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import SequenceDataset
from model import MiniTransformerClassifier
from utils import (
    set_seed,
    get_device,
    count_parameters,
    plot_training_curves,
    save_json,
)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        tokens = batch["tokens"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits, _ = model(tokens, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        preds = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        tokens = batch["tokens"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits, _ = model(tokens, attention_mask)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        preds = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def run_experiment(config, train_loader, val_loader, test_loader, device):
    model = MiniTransformerClassifier(
        vocab_size=5,
        max_len=20,
        d_model=config["d_model"],
        ff_dim=config["ff_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        use_positional_encoding=config["use_positional_encoding"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    history = []
    start_time = time.time()

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

        print(
            f"[{config['alias']}] Epoch {epoch:02d}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    training_time = time.time() - start_time
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    result = {
        "Model": config["alias"],
        "Positional Encoding": "Yes" if config["use_positional_encoding"] else "No",
        "Heads": config["num_heads"],
        "Layers": config["num_layers"],
        "Val Acc": history[-1]["val_accuracy"],
        "Test Acc": test_acc,
        "Train Time (s)": training_time,
        "Parameter Count": count_parameters(model),
    }

    return result, history


def main():
    parser = argparse.ArgumentParser(description="Run benchmark for mini Transformer variants")

    parser.add_argument("--train_csv", type=str, default="train.csv")
    parser.add_argument("--val_csv", type=str, default="validation.csv")
    parser.add_argument("--test_csv", type=str, default="test.csv")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--ff_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset = SequenceDataset(args.train_csv)
    val_dataset = SequenceDataset(args.val_csv)
    test_dataset = SequenceDataset(args.test_csv)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    base_config = {
        "d_model": args.d_model,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
    }

    experiment_configs = [
        {
            **base_config,
            "alias": "A",
            "use_positional_encoding": True,
            "num_heads": 1,
            "num_layers": 1,
        },
        {
            **base_config,
            "alias": "B",
            "use_positional_encoding": True,
            "num_heads": 4,
            "num_layers": 1,
        },
        {
            **base_config,
            "alias": "C",
            "use_positional_encoding": False,
            "num_heads": 4,
            "num_layers": 1,
        },
        {
            **base_config,
            "alias": "D",
            "use_positional_encoding": True,
            "num_heads": 4,
            "num_layers": 2,
        },
    ]

    benchmark_rows = []
    history_by_model = {}

    print("Running benchmark...")
    print(f"Device: {device}")

    for config in experiment_configs:
        print("\n" + "=" * 60)
        print(f"Running Model {config['alias']}")
        print("=" * 60)

        result, history = run_experiment(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
        )

        benchmark_rows.append(result)
        history_by_model[config["alias"]] = history

    benchmark_df = pd.DataFrame(benchmark_rows)

    csv_path = os.path.join(args.output_dir, "benchmark_results.csv")
    json_path = os.path.join(args.output_dir, "benchmark_results.json")
    plot_path = os.path.join(args.output_dir, "training_curve.png")

    benchmark_df.to_csv(csv_path, index=False)
    save_json(
        {
            "benchmark_results": benchmark_rows,
            "training_history": history_by_model,
        },
        json_path,
    )

    plot_training_curves(history_by_model, plot_path)

    print("\nBenchmark completed.")
    print("\nBenchmark Table:")
    print(benchmark_df.to_string(index=False))

    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved JSON to: {json_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()