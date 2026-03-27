from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        try:
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception:
            # Colab normally downloads pretrained weights, but we still want an offline-safe fallback.
            backbone = resnet18(weights=None)
        in_features = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*layers)
        self.proj = nn.Linear(in_features, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x).flatten(1)
        return self.proj(features)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_id: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(embed_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(token_ids)
        outputs, _ = self.rnn(embedded)
        mask = token_ids.ne(self.embedding.padding_idx).unsqueeze(-1)
        pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.proj(outputs), self.proj(pooled)


class KeywordGraphEncoder(nn.Module):
    def __init__(self, num_keywords: int, embed_dim: int, steps: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_keywords, embed_dim)
        self.steps = steps
        self.update = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, adjacency: torch.Tensor) -> torch.Tensor:
        nodes = self.embedding.weight
        for _ in range(self.steps):
            neighbors = adjacency @ nodes
            nodes = self.update(torch.cat([nodes, neighbors], dim=-1)) + nodes
        return nodes
