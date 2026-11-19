import torch
from torch import nn


class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_dim, block_size, head_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.scale = head_size ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal_mask", torch.ones(block_size, block_size, dtype=torch.bool).tril())

    def forward(self, x):
        length = x.size(1)
        scores = (self.query(x) @ self.key(x).transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~self.causal_mask[:length, :length], float("-inf"))
        return self.dropout(scores.softmax(dim=-1)) @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, block_size, num_heads, dropout=0.0):
        super().__init__()
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.heads = nn.ModuleList(
            SelfAttentionHead(embedding_dim, block_size, embedding_dim // num_heads, dropout)
            for _ in range(num_heads)
        )
        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        return self.proj(torch.cat([head(x) for head in self.heads], dim=-1))


class Block(nn.Module):
    def __init__(self, embedding_dim, block_size, n_heads, dropout=0.0, modern=False):
        super().__init__()
        self.sa = MultiHeadAttention(embedding_dim, block_size, n_heads, dropout)
        self.ffwd = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU() if modern else nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))
