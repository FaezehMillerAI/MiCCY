from __future__ import annotations

import torch
from torch import nn

from deepeyenet.models.decoder import AttentionDecoder
from deepeyenet.models.encoders import ImageEncoder, KeywordGraphEncoder, TextEncoder


class DeepEyeReasoner(nn.Module):
    def __init__(
        self,
        clinical_vocab_size: int,
        report_vocab_size: int,
        num_keywords: int,
        embed_dim: int,
        hidden_dim: int,
        clinical_pad_id: int,
        report_pad_id: int,
        graph_steps: int,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(clinical_vocab_size, embed_dim, hidden_dim, clinical_pad_id)
        self.graph_encoder = KeywordGraphEncoder(num_keywords, embed_dim, graph_steps)
        self.keyword_head = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_keywords),
        )
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.decoder = AttentionDecoder(report_vocab_size, embed_dim, hidden_dim, report_pad_id)

    def build_graph_context(self, keyword_indices: list[torch.Tensor], keyword_nodes: torch.Tensor) -> torch.Tensor:
        contexts = []
        global_context = keyword_nodes.mean(dim=0)
        for indices in keyword_indices:
            if indices.numel() == 0:
                contexts.append(global_context)
            else:
                contexts.append(keyword_nodes[indices.to(keyword_nodes.device)].mean(dim=0))
        return torch.stack(contexts, dim=0)

    def forward(
        self,
        images: torch.Tensor,
        clinical_ids: torch.Tensor,
        keyword_indices: list[torch.Tensor],
        adjacency: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        image_features = self.image_encoder(images)
        token_states, text_features = self.text_encoder(clinical_ids)
        keyword_nodes = self.graph_encoder(adjacency)
        graph_features = self.build_graph_context(keyword_indices, keyword_nodes)
        joint = torch.cat([image_features, text_features, graph_features], dim=-1)
        keyword_logits = self.keyword_head(joint)
        fused = self.fusion(joint)
        encoder_states = torch.stack([image_features, text_features, graph_features, fused], dim=1)
        decoder_logits, attention_maps = self.decoder(decoder_input_ids, encoder_states, fused)
        return {
            "decoder_logits": decoder_logits,
            "keyword_logits": keyword_logits,
            "attention_maps": attention_maps,
        }

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        clinical_ids: torch.Tensor,
        keyword_indices: list[torch.Tensor],
        adjacency: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_steps: int,
    ) -> dict[str, torch.Tensor]:
        image_features = self.image_encoder(images)
        _, text_features = self.text_encoder(clinical_ids)
        keyword_nodes = self.graph_encoder(adjacency)
        graph_features = self.build_graph_context(keyword_indices, keyword_nodes)
        joint = torch.cat([image_features, text_features, graph_features], dim=-1)
        keyword_logits = self.keyword_head(joint)
        fused = self.fusion(joint)
        encoder_states = torch.stack([image_features, text_features, graph_features, fused], dim=1)
        tokens, attention_maps = self.decoder.greedy_decode(encoder_states, fused, bos_id, eos_id, max_steps)
        return {
            "tokens": tokens,
            "keyword_logits": keyword_logits,
            "attention_maps": attention_maps,
        }
