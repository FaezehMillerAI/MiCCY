from __future__ import annotations

import torch
from torch import nn


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_id: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(embed_dim + embed_dim, hidden_dim, batch_first=True)
        self.attn = nn.Linear(hidden_dim + embed_dim, 1)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.init_proj = nn.Linear(embed_dim, hidden_dim)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_states: torch.Tensor,
        context_seed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.init_proj(context_seed).unsqueeze(0)
        embedded = self.embedding(decoder_input_ids)
        logits = []
        attn_maps = []
        for step in range(embedded.size(1)):
            token = embedded[:, step : step + 1, :]
            repeated_hidden = hidden[-1].unsqueeze(1).expand(-1, encoder_states.size(1), -1)
            scores = self.attn(torch.cat([repeated_hidden, encoder_states], dim=-1)).squeeze(-1)
            weights = scores.softmax(dim=-1)
            context = torch.bmm(weights.unsqueeze(1), encoder_states)
            output, hidden = self.rnn(torch.cat([token, context], dim=-1), hidden)
            logits.append(self.out(output))
            attn_maps.append(weights)
        return torch.cat(logits, dim=1), torch.stack(attn_maps, dim=1)

    @torch.no_grad()
    def greedy_decode(
        self,
        encoder_states: torch.Tensor,
        context_seed: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = encoder_states.size(0)
        hidden = self.init_proj(context_seed).unsqueeze(0)
        tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=encoder_states.device)
        outputs = []
        attn_maps = []
        for _ in range(max_steps):
            embedded = self.embedding(tokens[:, -1:])
            repeated_hidden = hidden[-1].unsqueeze(1).expand(-1, encoder_states.size(1), -1)
            scores = self.attn(torch.cat([repeated_hidden, encoder_states], dim=-1)).squeeze(-1)
            weights = scores.softmax(dim=-1)
            context = torch.bmm(weights.unsqueeze(1), encoder_states)
            output, hidden = self.rnn(torch.cat([embedded, context], dim=-1), hidden)
            logits = self.out(output[:, -1])
            next_token = logits.argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            outputs.append(next_token)
            attn_maps.append(weights)
            if bool((next_token == eos_id).all()):
                break
        if outputs:
            generated = torch.cat(outputs, dim=1)
            attn = torch.stack(attn_maps, dim=1)
        else:
            generated = torch.empty((batch_size, 0), dtype=torch.long, device=encoder_states.device)
            attn = torch.empty((batch_size, 0, encoder_states.size(1)), device=encoder_states.device)
        return generated, attn
