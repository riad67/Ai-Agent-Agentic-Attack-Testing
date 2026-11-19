from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .transformer_blocks import Block


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 16
    embedding_dim: int = 32
    n_heads: int = 2
    n_layers: int = 2
    dropout: float = 0.0
    modern: bool = False


class TinyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.block_size, config.embedding_dim)
        self.blocks = nn.Sequential(*[
            Block(config.embedding_dim, config.block_size, config.n_heads, config.dropout, config.modern)
            for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.embedding_dim)
        self.head = nn.Linear(config.embedding_dim, config.vocab_size, bias=not config.modern)
        if config.modern:
            self.head.weight = self.token_embedding.weight
            nn.init.normal_(self.token_embedding.weight, std=0.02)

    def forward(self, ids, targets=None):
        positions = torch.arange(ids.size(1), device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        logits = self.head(self.ln_f(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.inference_mode()
    def generate(self, ids, new_tokens, temperature, excluded_ids):
        for _ in range(new_tokens):
            logits, _ = self(ids[:, -self.config.block_size:])
            scores = logits[:, -1, :].clone()
            scores[:, excluded_ids] = float("-inf")
            if temperature == 0:
                next_id = scores.argmax(dim=-1, keepdim=True)
            else:
                next_id = torch.multinomial((scores / temperature).softmax(dim=-1), 1)
            ids = torch.cat((ids, next_id), dim=1)
        return ids
