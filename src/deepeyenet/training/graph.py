from __future__ import annotations

import torch

from deepeyenet.data.dataset import SampleRecord


def build_keyword_adjacency(records: list[SampleRecord], keyword_to_idx: dict[str, int]) -> torch.Tensor:
    n = len(keyword_to_idx)
    adjacency = torch.eye(n, dtype=torch.float32)
    for record in records:
        indices = [keyword_to_idx[key] for key in record.keywords if key in keyword_to_idx]
        for i in indices:
            for j in indices:
                adjacency[i, j] += 1.0
    degree = adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adjacency / degree
