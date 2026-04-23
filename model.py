import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    Adds position information to token embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 20):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention.
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):
        """
        Q, K, V shape: (batch_size, num_heads, seq_len, head_dim)
        attention_mask shape: (batch_size, seq_len)
        """
        d_k = Q.size(-1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # scores shape: (batch_size, num_heads, seq_len, seq_len)

        if attention_mask is not None:
            # Expand mask for broadcasting
            # (batch_size, 1, 1, seq_len)
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, V)
        return output, attention_weights


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention implemented from scratch.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len, d_model)
        output shape: (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, num_heads, seq_len, head_dim)
        output shape: (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention_output, attention_weights = self.attention(
            Q, K, V, attention_mask
        )

        attention_output = self.combine_heads(attention_output)
        output = self.W_o(attention_output)

        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    """
    Position-wise feed-forward network.
    """

    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, ff_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    One Transformer encoder block:
    self-attention -> add & norm -> feed-forward -> add & norm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        attention_output, attention_weights = self.self_attention(x, attention_mask)
        x = self.norm1(x + self.dropout1(attention_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x, attention_weights


class MiniTransformerClassifier(nn.Module):
    """
    Mini Transformer for sequence classification.
    Pipeline:
    Input -> Embedding -> Positional Encoding -> Encoder Blocks
          -> Mean Pooling over valid tokens -> Classification Head
    """

    def __init__(
        self,
        vocab_size: int = 5,
        max_len: int = 20,
        d_model: int = 64,
        ff_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=0,
        )

        self.use_positional_encoding = use_positional_encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def mean_pool(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean pooling over valid (non-padding) tokens only.
        x shape: (batch_size, seq_len, d_model)
        attention_mask shape: (batch_size, seq_len)
        """
        mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        x = x * mask
        summed = x.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts
        return pooled

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor):
        """
        tokens shape: (batch_size, seq_len)
        attention_mask shape: (batch_size, seq_len)
        """
        x = self.embedding(tokens)

        if self.use_positional_encoding:
            x = self.positional_encoding(x)

        attention_maps = []

        for layer in self.encoder_layers:
            x, attention_weights = layer(x, attention_mask)
            attention_maps.append(attention_weights)

        pooled = self.mean_pool(x, attention_mask)
        logits = self.classifier(pooled).squeeze(-1)

        return logits, attention_maps


if __name__ == "__main__":
    batch_size = 2
    seq_len = 20
    vocab_size = 5

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = (tokens != 0).float()

    model = MiniTransformerClassifier(
        vocab_size=vocab_size,
        max_len=20,
        d_model=64,
        ff_dim=128,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
        use_positional_encoding=True,
    )

    logits, attention_maps = model(tokens, attention_mask)

    print("Tokens shape:", tokens.shape)
    print("Attention mask shape:", attention_mask.shape)
    print("Logits shape:", logits.shape)
    print("Number of attention maps:", len(attention_maps))