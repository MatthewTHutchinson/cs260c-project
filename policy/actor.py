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


def policy_uses_images(policy_cfg: dict) -> bool:
    policy_type = str(policy_cfg.get("type", "mlp")).strip().lower()
    return policy_type in {"multimodal", "state_vision", "state+vision", "vision_fusion"}


def _normalize_image_shape(image_shape) -> tuple[int, int, int]:
    if image_shape is None:
        raise ValueError("image_shape is required for multimodal policies.")
    if len(image_shape) != 3:
        raise ValueError("image_shape must be [C, H, W].")
    c, h, w = (int(v) for v in image_shape)
    return c, h, w


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


class _ImageEncoder(nn.Module):
    """Compact CNN for onboard RGB frames."""

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        out_dim: int = 128,
        channels: list[int] | None = None,
    ):
        super().__init__()
        in_channels, _, _ = _normalize_image_shape(image_shape)
        channels = channels or [32, 64, 64]
        layers: list[nn.Module] = []
        c = in_channels
        for ch in channels:
            layers += [
                nn.Conv2d(c, ch, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
            ]
            c = ch
        layers += [
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(c * 4 * 4, out_dim),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dtype != torch.float32:
            images = images.float()
        if images.max().item() > 1.0:
            images = images / 255.0
        return self.net(images)


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


class MultimodalDronePolicy(nn.Module):
    """Deterministic actor over state vectors and onboard RGB images."""

    expects_image = True

    def __init__(
        self,
        obs_dim: int,
        image_shape: tuple[int, int, int],
        action_dim: int = 4,
        hidden: list[int] | None = None,
        image_feature_dim: int = 128,
        image_channels: list[int] | None = None,
    ):
        super().__init__()
        hidden = hidden or [256, 256]
        self.image_encoder = _ImageEncoder(
            image_shape=image_shape,
            out_dim=image_feature_dim,
            channels=image_channels,
        )
        self.net = _mlp(obs_dim + image_feature_dim, hidden, action_dim, activate_out=True)

    def forward(self, obs: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        image_feats = self.image_encoder(images)
        return self.net(torch.cat([obs, image_feats], dim=-1))

    @torch.no_grad()
    def act(self, obs: torch.Tensor, images: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self.forward(obs, images)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str = "cpu") -> "MultimodalDronePolicy":
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


class MultimodalActorCritic(nn.Module):
    """PPO actor-critic conditioned on state vectors and onboard RGB images."""

    expects_image = True

    def __init__(
        self,
        obs_dim: int,
        image_shape: tuple[int, int, int],
        action_dim: int = 4,
        hidden: list[int] | None = None,
        init_log_std: float = -2.0,
        image_feature_dim: int = 128,
        image_channels: list[int] | None = None,
    ):
        super().__init__()
        hidden = hidden or [256, 256]
        self.image_encoder = _ImageEncoder(
            image_shape=image_shape,
            out_dim=image_feature_dim,
            channels=image_channels,
        )

        trunk_layers: list[nn.Module] = []
        d = obs_dim + image_feature_dim
        for h in hidden:
            trunk_layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        self.trunk = nn.Sequential(*trunk_layers)
        self._trunk_out_dim = d

        self.actor_mean = nn.Linear(d, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))
        self.critic = nn.Linear(d, 1)
        self._action_dim = action_dim

    def _features(self, obs: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        image_feats = self.image_encoder(images)
        fused = torch.cat([obs, image_feats], dim=-1)
        return self.trunk(fused)

    def forward(self, obs: torch.Tensor, images: torch.Tensor):
        feats = self._features(obs, images)
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

    def get_action(self, obs: torch.Tensor, images: torch.Tensor, deterministic: bool = False):
        mean, std, value = self.forward(obs, images)
        dist = Normal(mean, std.expand_as(mean))
        if deterministic:
            action = self._squash(mean)
            return action, torch.zeros(action.shape[0], device=obs.device), value
        raw = dist.rsample()
        action = self._squash(raw)
        log_prob = self._squashed_log_prob(dist, raw)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, images: torch.Tensor):
        mean, std, value = self.forward(obs, images)
        dist = Normal(mean, std.expand_as(mean))
        clipped_actions = actions.clamp(-1.0 + _ACTION_EPS, 1.0 - _ACTION_EPS)
        raw_actions = torch.atanh(clipped_actions)
        log_prob = self._squashed_log_prob(dist, raw_actions)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, images: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        action, _, _ = self.get_action(obs, images, deterministic=deterministic)
        return action

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str = "cpu") -> "MultimodalActorCritic":
        self.load_state_dict(torch.load(path, map_location=device))
        return self


def build_deterministic_policy(policy_cfg: dict, obs_dim: int):
    if policy_uses_images(policy_cfg):
        image_shape = _normalize_image_shape(policy_cfg.get("image_shape"))
        return MultimodalDronePolicy(
            obs_dim=obs_dim,
            image_shape=image_shape,
            action_dim=int(policy_cfg["action_dim"]),
            hidden=policy_cfg.get("hidden"),
            image_feature_dim=int(policy_cfg.get("image_feature_dim", 128)),
            image_channels=policy_cfg.get("image_channels"),
        )
    return DronePolicy(
        obs_dim=obs_dim,
        action_dim=int(policy_cfg["action_dim"]),
        hidden=policy_cfg.get("hidden"),
    )


def build_actor_critic(policy_cfg: dict):
    if policy_uses_images(policy_cfg):
        image_shape = _normalize_image_shape(policy_cfg.get("image_shape"))
        return MultimodalActorCritic(
            obs_dim=int(policy_cfg["obs_dim"]),
            image_shape=image_shape,
            action_dim=int(policy_cfg["action_dim"]),
            hidden=policy_cfg.get("hidden"),
            init_log_std=float(policy_cfg.get("init_log_std", -2.0)),
            image_feature_dim=int(policy_cfg.get("image_feature_dim", 128)),
            image_channels=policy_cfg.get("image_channels"),
        )
    return ActorCritic(
        obs_dim=int(policy_cfg["obs_dim"]),
        action_dim=int(policy_cfg["action_dim"]),
        hidden=policy_cfg.get("hidden"),
        init_log_std=float(policy_cfg.get("init_log_std", -2.0)),
    )
