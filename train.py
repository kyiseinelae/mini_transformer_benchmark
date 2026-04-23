import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import SequenceDataset
from model import MiniTransformerClassifier
from utils import (
    set_seed,
    get_device,
    count_parameters,
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


def main():
    parser = argparse.ArgumentParser(description="Train a mini Transformer classifier from scratch")

    parser.add_argument("--train_csv", type=str, default="train.csv")
    parser.add_argument("--val_csv", type=str, default="validation.csv")
    parser.add_argument("--test_csv", type=str, default="test.csv")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--ff_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_positional_encoding", type=int, default=1)

    parser.add_argument("--save_dir", type=str, default="outputs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    os.makedirs(args.save_dir, exist_ok=True)

    train_dataset = SequenceDataset(args.train_csv)
    val_dataset = SequenceDataset(args.val_csv)
    test_dataset = SequenceDataset(args.test_csv)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = MiniTransformerClassifier(
        vocab_size=5,
        max_len=20,
        d_model=args.d_model,
        ff_dim=args.ff_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_positional_encoding=bool(args.use_positional_encoding),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = []

    print("Training started...")
    print(f"Device: {device}")
    print(f"Trainable parameters: {count_parameters(model)}")

    for epoch in range(1, args.epochs + 1):
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
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print("\nFinal Test Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    results = {
        "config": vars(args),
        "device": str(device),
        "parameter_count": count_parameters(model),
        "final_test_loss": test_loss,
        "final_test_accuracy": test_acc,
        "history": history,
    }

    save_json(results, os.path.join(args.save_dir, "train_results.json"))
    torch.save(model.state_dict(), os.path.join(args.save_dir, "model.pt"))

    print(f"\nSaved results to: {os.path.join(args.save_dir, 'train_results.json')}")
    print(f"Saved model to: {os.path.join(args.save_dir, 'model.pt')}")


if __name__ == "__main__":
    main()