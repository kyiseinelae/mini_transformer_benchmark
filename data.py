import pandas as pd
import torch
from torch.utils.data import Dataset

MAX_LEN = 20
VOCAB = {"PAD": 0, "A": 1, "B": 2, "C": 3, "D": 4}


def compute_label(seq):
    """
    seq: list of valid tokens WITHOUT padding
    label = 1 if first token appears in second half, else 0
    """
    length = len(seq)
    mid = length // 2
    first_token = seq[0]
    second_half = seq[mid:]
    return 1 if first_token in second_half else 0


class SequenceDataset(Dataset):
    """
    Loads dataset from CSV files provided in the assignment.
    Expected columns:
    - token_01 ... token_20
    - mask_01 ... mask_20
    - label
    """

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        self.tokens = torch.tensor(
            self.df[[f"token_{i:02d}" for i in range(1, MAX_LEN + 1)]].values,
            dtype=torch.long,
        )

        self.attention_mask = torch.tensor(
            self.df[[f"mask_{i:02d}" for i in range(1, MAX_LEN + 1)]].values,
            dtype=torch.float32,
        )

        self.labels = torch.tensor(
            self.df["label"].values,
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "tokens": self.tokens[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def inspect_example(csv_path: str, index: int = 0):
    df = pd.read_csv(csv_path)

    token_cols = [f"token_{i:02d}" for i in range(1, MAX_LEN + 1)]
    mask_cols = [f"mask_{i:02d}" for i in range(1, MAX_LEN + 1)]

    tokens = df.loc[index, token_cols].tolist()
    mask = df.loc[index, mask_cols].tolist()
    label = df.loc[index, "label"]

    valid_tokens = [t for t, m in zip(tokens, mask) if m == 1]
    recomputed_label = compute_label(valid_tokens)

    print("Raw tokens:       ", tokens)
    print("Attention mask:   ", mask)
    print("Valid tokens:     ", valid_tokens)
    print("CSV label:        ", label)
    print("Recomputed label: ", recomputed_label)


if __name__ == "__main__":
    dataset = SequenceDataset("train.csv")
    print("Dataset size:", len(dataset))

    sample = dataset[0]
    print("Sample tokens shape:", sample["tokens"].shape)
    print("Sample mask shape:", sample["attention_mask"].shape)
    print("Sample label:", sample["labels"].item())

    inspect_example("train.csv", index=0)