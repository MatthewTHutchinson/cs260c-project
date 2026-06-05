"""Compact GRU policy for feature-based racing commands."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PolicyConfig:
    input_dim: int
    hidden_dim: int = 128
    num_layers: int = 1
    output_dim: int = 4
    dropout: float = 0.0


class FeaturePolicyGRU(nn.Module):
    """Map feature history to roll/pitch/yaw-rate and thrust commands."""

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [batch, time, feature], got shape={tuple(x.shape)}")
        out, _ = self.gru(x)
        return self.head(out[:, -1])

    @torch.no_grad()
    def predict_clipped(self, x: torch.Tensor) -> torch.Tensor:
        y = self.forward(x)
        rates = torch.tanh(y[..., :3])
        thrust = torch.sigmoid(y[..., 3:4])
        return torch.cat([rates, thrust], dim=-1)


def save_checkpoint(
    path: str,
    *,
    model: FeaturePolicyGRU,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    feature_names: tuple[str, ...],
    metadata: dict,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": model.config.__dict__,
            "feature_mean": feature_mean.cpu(),
            "feature_std": feature_std.cpu(),
            "feature_names": feature_names,
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> tuple[FeaturePolicyGRU, dict]:
    payload = torch.load(path, map_location=map_location)
    model = FeaturePolicyGRU(PolicyConfig(**payload["config"]))
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload

