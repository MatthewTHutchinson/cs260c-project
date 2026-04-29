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
from policy.actor import DronePolicy


# ------------------------------------------------------------------
# Dataset collection
# ------------------------------------------------------------------

def collect_expert_dataset(env, expert: ExpertPolicy, n_episodes: int, seed: int = 0):
    """Roll out expert and return list of (obs, action) tuples."""
    dataset = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = expert.act(env)
            dataset.append((obs.copy(), action.copy()))
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    return dataset


def dataset_to_tensors(dataset):
    obs_arr = np.array([d[0] for d in dataset], dtype=np.float32)
    act_arr = np.array([d[1] for d in dataset], dtype=np.float32)
    return torch.from_numpy(obs_arr), torch.from_numpy(act_arr)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_bc(
    policy: DronePolicy,
    obs_t: torch.Tensor,
    act_t: torch.Tensor,
    n_epochs: int = 200,
    lr: float = 3e-4,
    batch_size: int = 256,
    device: str = "cpu",
) -> list[float]:
    policy = policy.to(device)
    obs_t, act_t = obs_t.to(device), act_t.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(obs_t)
    epoch_losses = []

    for epoch in range(n_epochs):
        idx = torch.randperm(n, device=device)
        batch_losses = []
        for start in range(0, n, batch_size):
            bi = idx[start : start + batch_size]
            pred = policy(obs_t[bi])
            loss = F.mse_loss(pred, act_t[bi])
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
    policy = DronePolicy(
        obs_dim=policy_cfg["obs_dim"],
        action_dim=policy_cfg["action_dim"],
        hidden=policy_cfg["hidden"],
    )

    bc_cfg = cfg["bc"]
    print(f"[BC] Collecting {bc_cfg['n_expert_episodes']} expert episodes …")
    dataset = collect_expert_dataset(env, expert, bc_cfg["n_expert_episodes"], seed=args.seed)
    obs_t, act_t = dataset_to_tensors(dataset)
    print(f"[BC] Dataset size: {len(dataset)} transitions")

    print("[BC] Training …")
    losses = train_bc(
        policy, obs_t, act_t,
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

    env.close()
    print(f"[BC] Done. Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
