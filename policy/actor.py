"""Policy networks for the drone gate-racing agent.

DronePolicy  — deterministic actor used in BC and DAgger phases.
ActorCritic  — shared-trunk actor + value head used in the PPO phase.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

_LOG_STD_MIN = -5.0
_LOG_STD_MAX = 0.0
_ACTION_EPS = 1e-6


def _mlp(in_dim: int, hidden: list[int], out_dim: int, activate_out: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    if activate_out:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class DronePolicy(nn.Module):
    """Deterministic MLP actor: obs → action ∈ [-1, 1]^4."""

    def __init__(self, obs_dim: int = 12, action_dim: int = 4, hidden: list[int] = None):
        super().__init__()
        hidden = hidden or [256, 256]
        self.net = _mlp(obs_dim, hidden, action_dim, activate_out=True)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self.forward(obs)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str = "cpu") -> "DronePolicy":
        self.load_state_dict(torch.load(path, map_location=device))
        return self


class ActorCritic(nn.Module):
    """Stochastic actor + value critic sharing a trunk, used for PPO.

    Actor outputs a Gaussian whose mean passes through tanh so the
    deterministic action stays in [-1, 1].  Log-std is a learned
    parameter (not input-dependent).
    """

    def __init__(
        self,
        obs_dim: int = 12,
        action_dim: int = 4,
        hidden: list[int] = None,
        init_log_std: float = -2.0,
    ):
        super().__init__()
        hidden = hidden or [256, 256]

        # Shared trunk
        trunk_layers: list[nn.Module] = []
        d = obs_dim
        for h in hidden:
            trunk_layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        self.trunk = nn.Sequential(*trunk_layers)
        self._trunk_out_dim = d

        self.actor_mean = nn.Linear(d, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))
        self.critic = nn.Linear(d, 1)

        self._action_dim = action_dim

    def forward(self, obs: torch.Tensor):
        """Returns (mean, std, value)."""
        feats = self.trunk(obs)
        mean = self.actor_mean(feats)
        log_std = self.actor_log_std.clamp(_LOG_STD_MIN, _LOG_STD_MAX)
        std = log_std.exp().clamp(1e-4, 1.0)
        value = self.critic(feats).squeeze(-1)
        return mean, std, value

    def _squash(self, raw_action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(raw_action)

    def _squashed_log_prob(self, dist: Normal, raw_action: torch.Tensor) -> torch.Tensor:
        action = self._squash(raw_action)
        correction = torch.log(1.0 - action.pow(2) + _ACTION_EPS)
        return dist.log_prob(raw_action).sum(-1) - correction.sum(-1)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample action and return (action, log_prob, value)."""
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std.expand_as(mean))
        if deterministic:
            action = self._squash(mean)
            return action, torch.zeros(action.shape[0], device=obs.device), value
        raw = dist.rsample()
        action = self._squash(raw)
        log_prob = self._squashed_log_prob(dist, raw)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Returns (log_prob, entropy, value) for given obs-action pairs."""
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std.expand_as(mean))
        clipped_actions = actions.clamp(-1.0 + _ACTION_EPS, 1.0 - _ACTION_EPS)
        raw_actions = torch.atanh(clipped_actions)
        log_prob = self._squashed_log_prob(dist, raw_actions)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        action, _, _ = self.get_action(obs, deterministic=deterministic)
        return action

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str = "cpu") -> "ActorCritic":
        self.load_state_dict(torch.load(path, map_location=device))
        return self
