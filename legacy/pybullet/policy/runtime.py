"""Utilities for routing env observations into state-only or multimodal policies."""

from __future__ import annotations

import numpy as np
import torch


def policy_expects_images(policy) -> bool:
    return bool(getattr(policy, "expects_image", False))


def env_supports_images(env) -> bool:
    return hasattr(env, "get_last_camera_frame")


def get_env_image(env) -> np.ndarray:
    if not env_supports_images(env):
        raise RuntimeError("Environment does not expose get_last_camera_frame().")
    image = env.get_last_camera_frame()
    if image is None:
        image = env.render_onboard_camera() if hasattr(env, "render_onboard_camera") else None
    if image is None:
        raise RuntimeError("Policy expects images, but the environment did not provide a camera frame.")
    return np.asarray(image, dtype=np.uint8)


def image_batch_to_tensor(images, device: str) -> torch.Tensor:
    arr = np.asarray(images, dtype=np.uint8)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError("Expected images with shape [N, H, W, 3] or [H, W, 3].")
    arr = np.ascontiguousarray(arr.transpose(0, 3, 1, 2))
    return torch.from_numpy(arr).to(device)


def obs_batch_to_tensor(obs, device: str) -> torch.Tensor:
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, ...]
    return torch.from_numpy(arr).to(device)


def single_step_tensors(obs, env, device: str, policy=None) -> tuple[torch.Tensor, torch.Tensor | None]:
    obs_t = obs_batch_to_tensor(obs, device)
    if policy is None or not policy_expects_images(policy):
        return obs_t, None
    image = get_env_image(env)
    image_t = image_batch_to_tensor(image, device)
    return obs_t, image_t


def policy_forward(policy, obs_t: torch.Tensor, image_t: torch.Tensor | None = None) -> torch.Tensor:
    if policy_expects_images(policy):
        if image_t is None:
            raise RuntimeError("Multimodal policy forward() requires image tensors.")
        return policy(obs_t, image_t)
    return policy(obs_t)


def policy_act(policy, obs_t: torch.Tensor, image_t: torch.Tensor | None = None, deterministic: bool = True) -> torch.Tensor:
    if policy_expects_images(policy):
        if image_t is None:
            raise RuntimeError("Multimodal policy act() requires image tensors.")
        return policy.act(obs_t, image_t, deterministic=deterministic)
    return policy.act(obs_t, deterministic=deterministic)


def actor_critic_get_action(policy, obs_t: torch.Tensor, image_t: torch.Tensor | None = None, deterministic: bool = False):
    if policy_expects_images(policy):
        if image_t is None:
            raise RuntimeError("Multimodal PPO policy get_action() requires image tensors.")
        return policy.get_action(obs_t, image_t, deterministic=deterministic)
    return policy.get_action(obs_t, deterministic=deterministic)


def actor_critic_evaluate_actions(
    policy,
    obs_t: torch.Tensor,
    actions_t: torch.Tensor,
    image_t: torch.Tensor | None = None,
):
    if policy_expects_images(policy):
        if image_t is None:
            raise RuntimeError("Multimodal PPO policy evaluate_actions() requires image tensors.")
        return policy.evaluate_actions(obs_t, actions_t, image_t)
    return policy.evaluate_actions(obs_t, actions_t)
