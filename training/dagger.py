"""Phase C — DAgger (Dataset Aggregation).

Iteratively rolls out the current policy, queries the expert on visited
states, aggregates the dataset, and retrains with BC loss.

Usage
-----
    python -m training.dagger --config configs/default.yaml \
        --bc-ckpt logs/bc/policy_bc.pt \
        --bc-data logs/bc \
        --out logs/dagger
"""

import argparse
import os
import numpy as np
import torch
import yaml

from env.gate_race_aviary import make_env
from expert.expert_policy import ExpertPolicy
from policy.actor import DronePolicy
from training.bc import train_bc, dataset_to_tensors


def rollout_policy(env, policy: DronePolicy, expert: ExpertPolicy,
                   n_episodes: int, device: str, seed: int = 0):
    """Roll out policy, label each state with the expert action."""
    new_obs, new_acts = [], []
    policy.eval()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                policy(obs_t)  # policy acts (we don't use its action for expert queries)
            expert_action = expert.act(env)
            new_obs.append(obs.copy())
            new_acts.append(expert_action.copy())
            # Step with policy action so distribution shift is applied
            policy_action = policy.act(obs_t).squeeze(0).cpu().numpy()
            obs, _, terminated, truncated, _ = env.step(policy_action)
            done = terminated or truncated
    return (np.array(new_obs, dtype=np.float32),
            np.array(new_acts, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--bc-ckpt", default="logs/bc/policy_bc.pt")
    parser.add_argument("--bc-data", default="logs/bc")
    parser.add_argument("--out", default="logs/dagger")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DAgger] device={device}")

    env = make_env(cfg["env"])
    expert = ExpertPolicy()
    policy_cfg = cfg["policy"]
    policy = DronePolicy(
        obs_dim=policy_cfg["obs_dim"],
        action_dim=policy_cfg["action_dim"],
        hidden=policy_cfg["hidden"],
    )
    policy.load(args.bc_ckpt, device=device)

    dagger_cfg = cfg["dagger"]

    # Load BC dataset as the initial aggregate dataset
    obs_agg = np.load(os.path.join(args.bc_data, "dataset_obs.npy"))
    act_agg = np.load(os.path.join(args.bc_data, "dataset_act.npy"))
    print(f"[DAgger] Initial dataset size: {len(obs_agg)}")

    all_losses = []
    for rnd in range(dagger_cfg["n_rounds"]):
        print(f"\n[DAgger] Round {rnd + 1}/{dagger_cfg['n_rounds']}")

        new_obs, new_acts = rollout_policy(
            env, policy, expert,
            n_episodes=dagger_cfg["n_rollout_episodes"],
            device=device,
            seed=args.seed + rnd * 1000,
        )
        obs_agg = np.concatenate([obs_agg, new_obs], axis=0)
        act_agg = np.concatenate([act_agg, new_acts], axis=0)
        print(f"[DAgger]   Aggregated dataset size: {len(obs_agg)}")

        obs_t, act_t = dataset_to_tensors(list(zip(obs_agg, act_agg)))
        losses = train_bc(
            policy, obs_t, act_t,
            n_epochs=dagger_cfg["n_epochs"],
            lr=dagger_cfg["lr"],
            batch_size=dagger_cfg["batch_size"],
            device=device,
        )
        all_losses.extend(losses)

        ckpt = os.path.join(args.out, f"policy_dagger_r{rnd + 1:02d}.pt")
        policy.save(ckpt)

    final_ckpt = os.path.join(args.out, "policy_dagger.pt")
    policy.save(final_ckpt)
    np.save(os.path.join(args.out, "dagger_losses.npy"), np.array(all_losses))

    # Save final aggregated dataset for PPO phase
    np.save(os.path.join(args.out, "dataset_obs.npy"), obs_agg)
    np.save(os.path.join(args.out, "dataset_act.npy"), act_agg)

    env.close()
    print(f"[DAgger] Done. Checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
