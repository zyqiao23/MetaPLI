"""Attentive Neural Process components used by MetaPLI."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def build_mlp(dim_in: int, dim_hid: int, dim_out: int, depth: int) -> nn.Sequential:
    if depth < 2:
        raise ValueError("depth must be at least 2")
    layers: list[nn.Module] = [nn.Linear(dim_in, dim_hid), nn.ReLU(inplace=False)]
    for _ in range(depth - 2):
        layers.extend([nn.Linear(dim_hid, dim_hid), nn.ReLU(inplace=False)])
    layers.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*layers)


class LatentEncoder(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_hid: int,
        dim_latent: int,
        depth: int,
        heads: int,
        dim_label: int,
    ) -> None:
        super().__init__()
        self.input_projection = build_mlp(dim_in, dim_hid, dim_hid, depth)
        self.latent_encoder_mlp = nn.Linear(dim_hid + dim_label, dim_hid)
        self.self_attn = nn.MultiheadAttention(dim_hid, heads, batch_first=True)
        penultimate = int(0.5 * (dim_hid + dim_latent))
        self.penultimate = nn.Linear(dim_hid, penultimate)
        self.mu = nn.Linear(penultimate, dim_latent)
        self.log_sigma = nn.Linear(penultimate, dim_latent)

    def forward(
        self,
        raw_context_x: torch.Tensor,
        context_y: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.distributions.Normal, torch.Tensor, torch.Tensor]:
        valid_counts = None
        if attention_mask is not None:
            if (
                attention_mask.dtype != torch.bool
                or attention_mask.shape != raw_context_x.shape[:2]
            ):
                raise ValueError(
                    "attention_mask must be a boolean mask matching the context batch"
                )
            valid_counts = (~attention_mask).sum(dim=1, keepdim=True)
            if (valid_counts == 0).any():
                raise ValueError("Each task must contain at least one unmasked context")
        context_x = self.input_projection(raw_context_x)
        encoded = self.latent_encoder_mlp(torch.cat([context_x, context_y], dim=-1))
        encoded, _ = self.self_attn(
            encoded,
            encoded,
            encoded,
            key_padding_mask=attention_mask,
        )
        if attention_mask is None:
            pooled = encoded.mean(dim=1)
        else:
            pooled = (
                encoded.masked_fill(attention_mask.unsqueeze(-1), 0).sum(dim=1)
                / valid_counts
            )
        hidden = torch.relu(self.penultimate(pooled))
        mu = self.mu(hidden)
        sigma = 0.1 + 0.9 * torch.sigmoid(self.log_sigma(hidden))
        return torch.distributions.Normal(mu, sigma), mu, sigma


class DeterministicEncoder(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_hid: int,
        dim_latent: int,
        depth: int,
        heads: int,
        dim_label: int,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim_hid, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            dim_hid, heads, vdim=dim_latent, batch_first=True
        )
        self.deter_encoder_mlp = nn.Linear(dim_hid + dim_hid, dim_hid)
        self.input_projection = build_mlp(dim_in, dim_hid, dim_hid, depth)
        self.activity_projection = nn.Linear(dim_label, dim_hid)
        self.penultimate = nn.Linear(dim_hid, dim_hid)

    def forward(
        self,
        raw_context_x: torch.Tensor,
        context_y: torch.Tensor,
        raw_target_x: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context_mask  # retained in the signature for checkpoint/API compatibility
        context_x = self.input_projection(raw_context_x)
        target_x = self.input_projection(raw_target_x)
        activity = self.activity_projection(context_y)
        encoded = self.deter_encoder_mlp(torch.cat((context_x, activity), dim=-1))
        encoded, _ = self.self_attn(encoded, encoded, encoded)
        hidden_r = torch.relu(self.penultimate(encoded))
        hidden, _ = self.cross_attn(target_x, context_x, hidden_r)
        return hidden


class Decoder(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_hid: int,
        dim_latent: int,
        depth: int,
        dim_label: int,
    ) -> None:
        super().__init__()
        self.input_projection = build_mlp(dim_in, dim_hid, dim_hid, depth)
        self.decoder_mlp = nn.Linear(2 * dim_latent + dim_hid, 2 * dim_label)

    def forward(
        self, r: torch.Tensor, z: torch.Tensor, raw_target_x: torch.Tensor
    ) -> tuple[torch.distributions.Normal, torch.Tensor, torch.Tensor]:
        target_x = self.input_projection(raw_target_x)
        hidden = torch.cat([r, z, target_x], dim=-1)
        mu, log_sigma = self.decoder_mlp(hidden).contiguous().chunk(2, dim=-1)
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)
        return torch.distributions.Normal(mu, sigma), mu, sigma
