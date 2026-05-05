"""Phase B — Behaviour Cloning.

Collects an expert demonstration dataset by rolling out ExpertPolicy in
GateRaceAviary, then trains a DronePolicy with MSE loss.

Usage
-----
    python -m training.bc --config configs/default.yaml --out logs/bc
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from env.gate_race_aviary import make_env
from expert.expert_policy import ExpertPolicy
from policy.actor import (
    ActorCritic,
    DronePolicy,
    build_deterministic_policy,
    policy_uses_images,
    warmstart_multimodal_policy_from_state,
)
from policy.runtime import get_env_image, policy_forward


# ------------------------------------------------------------------
# Dataset collection
# ------------------------------------------------------------------

def collect_expert_dataset(
    env,
    expert: ExpertPolicy,
    n_episodes: int,
    seed: int = 0,
    collect_images: bool = False,
):
    """Roll out expert and return list of (obs, action) tuples."""
    dataset = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = expert.act(env)
            if collect_images:
                image = np.transpose(get_env_image(env), (2, 0, 1)).copy()
                dataset.append((obs.copy(), action.copy(), image))
            else:
                dataset.append((obs.copy(), action.copy()))
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    return dataset


def dataset_to_tensors(dataset):
    obs_arr = np.array([d[0] for d in dataset], dtype=np.float32)
    act_arr = np.array([d[1] for d in dataset], dtype=np.float32)
    img_t = None
    if dataset and len(dataset[0]) >= 3:
        img_arr = np.array([d[2] for d in dataset], dtype=np.uint8)
        if img_arr.ndim == 4 and img_arr.shape[-1] == 3:
            img_arr = np.transpose(img_arr, (0, 3, 1, 2))
        if img_arr.ndim != 4 or img_arr.shape[1] != 3:
            raise ValueError("Expected image dataset with shape [N, 3, H, W] or [N, H, W, 3].")
        img_t = torch.from_numpy(img_arr)
    return torch.from_numpy(obs_arr), torch.from_numpy(act_arr), img_t


def build_state_teacher(
    obs_dim: int,
    policy_cfg: dict,
    ppo_cfg: dict,
    ckpt: str,
    teacher_type: str,
    device: str,
):
    teacher_type = str(teacher_type).strip().lower()
    if teacher_type == "bc":
        teacher = DronePolicy(
            obs_dim=obs_dim,
            action_dim=int(policy_cfg["action_dim"]),
            hidden=policy_cfg.get("hidden"),
        )
    elif teacher_type == "ppo":
        teacher = ActorCritic(
            obs_dim=obs_dim,
            action_dim=int(policy_cfg["action_dim"]),
            hidden=policy_cfg.get("hidden"),
            init_log_std=float(ppo_cfg.get("init_log_std", -2.0)),
        )
    else:
        raise ValueError("teacher_type must be 'bc' or 'ppo'.")
    teacher.load(ckpt, device=device).to(device).eval()
    return teacher


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_bc(
    policy,
    obs_t: torch.Tensor,
    act_t: torch.Tensor,
    img_t: torch.Tensor | None = None,
    teacher=None,
    distill_coef: float = 0.0,
    n_epochs: int = 200,
    lr: float = 3e-4,
    batch_size: int = 256,
    device: str = "cpu",
) -> list[float]:
    policy = policy.to(device)
    obs_t, act_t = obs_t.to(device), act_t.to(device)
    if img_t is not None:
        img_t = img_t.to(device)
    if teacher is not None:
        teacher = teacher.to(device).eval()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(obs_t)
    epoch_losses = []

    for epoch in range(n_epochs):
        idx = torch.randperm(n, device=device)
        batch_losses = []
        for start in range(0, n, batch_size):
            bi = idx[start : start + batch_size]
            batch_images = img_t[bi] if img_t is not None else None
            pred = policy_forward(policy, obs_t[bi], batch_images)
            loss = F.mse_loss(pred, act_t[bi])
            if teacher is not None and distill_coef > 0.0:
                with torch.no_grad():
                    teacher_pred = teacher.act(obs_t[bi], deterministic=True)
                loss = loss + float(distill_coef) * F.mse_loss(pred, teacher_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_loss = float(np.mean(batch_losses))
        epoch_losses.append(epoch_loss)
        if (epoch + 1) % 20 == 0:
            print(f"  BC epoch {epoch + 1:4d}/{n_epochs}  loss={epoch_loss:.5f}")

    return epoch_losses


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="logs/bc")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BC] device={device}")

    env = make_env(cfg["env"])
    expert = ExpertPolicy()
    policy_cfg = cfg["policy"]
    obs_dim = int(env.observation_space.shape[0])
    cfg_obs_dim = int(policy_cfg.get("obs_dim", obs_dim))
    if cfg_obs_dim != obs_dim:
        raise ValueError(f"policy.obs_dim={cfg_obs_dim} does not match env observation dim {obs_dim}")
    collect_images = policy_uses_images(policy_cfg)
    if collect_images:
        cfg_image_shape = tuple(int(v) for v in policy_cfg.get("image_shape", []))
        env_image_shape = tuple(int(v) for v in getattr(env, "policy_image_shape", ()))
        if cfg_image_shape != env_image_shape:
            raise ValueError(
                f"policy.image_shape={cfg_image_shape} does not match env image shape {env_image_shape}"
            )
    policy = build_deterministic_policy(policy_cfg, obs_dim=obs_dim)

    warmstart_teacher = None
    warmstart_ckpt = policy_cfg.get("warmstart_state_ckpt")
    if collect_images and warmstart_ckpt:
        warmstart_teacher = build_state_teacher(
            obs_dim=obs_dim,
            policy_cfg=policy_cfg,
            ppo_cfg=cfg.get("ppo", {}),
            ckpt=str(warmstart_ckpt),
            teacher_type=str(policy_cfg.get("warmstart_state_type", "ppo")),
            device=device,
        )
        transferred = warmstart_multimodal_policy_from_state(policy, warmstart_teacher)
        print(f"[BC] Warm-started multimodal actor from state teacher ({transferred} tensors copied)")

    distill_teacher = None
    distill_coef = float(policy_cfg.get("distill_coef", 0.0))
    distill_ckpt = policy_cfg.get("distill_teacher_ckpt")
    if distill_coef > 0.0 and distill_ckpt:
        if (
            warmstart_teacher is not None
            and str(distill_ckpt) == str(warmstart_ckpt)
            and str(policy_cfg.get("distill_teacher_type", policy_cfg.get("warmstart_state_type", "ppo"))).strip().lower()
                == str(policy_cfg.get("warmstart_state_type", "ppo")).strip().lower()
        ):
            distill_teacher = warmstart_teacher
        else:
            distill_teacher = build_state_teacher(
                obs_dim=obs_dim,
                policy_cfg=policy_cfg,
                ppo_cfg=cfg.get("ppo", {}),
                ckpt=str(distill_ckpt),
                teacher_type=str(policy_cfg.get("distill_teacher_type", "ppo")),
                device=device,
            )
        print(f"[BC] Distilling from state teacher with coef={distill_coef:.3f}")

    bc_cfg = cfg["bc"]
    print(f"[BC] Collecting {bc_cfg['n_expert_episodes']} expert episodes …")
    dataset = collect_expert_dataset(
        env,
        expert,
        bc_cfg["n_expert_episodes"],
        seed=args.seed,
        collect_images=collect_images,
    )
    obs_t, act_t, img_t = dataset_to_tensors(dataset)
    print(f"[BC] Dataset size: {len(dataset)} transitions")

    print("[BC] Training …")
    losses = train_bc(
        policy, obs_t, act_t,
        img_t=img_t,
        teacher=distill_teacher,
        distill_coef=distill_coef,
        n_epochs=bc_cfg["n_epochs"],
        lr=bc_cfg["lr"],
        batch_size=bc_cfg["batch_size"],
        device=device,
    )

    ckpt = os.path.join(args.out, "policy_bc.pt")
    policy.save(ckpt)
    np.save(os.path.join(args.out, "bc_losses.npy"), np.array(losses))

    # Also save the raw dataset for DAgger / PPO phases
    np.save(os.path.join(args.out, "dataset_obs.npy"), obs_t.numpy())
    np.save(os.path.join(args.out, "dataset_act.npy"), act_t.numpy())
    if img_t is not None:
        np.save(os.path.join(args.out, "dataset_img.npy"), img_t.numpy())

    env.close()
    print(f"[BC] Done. Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
