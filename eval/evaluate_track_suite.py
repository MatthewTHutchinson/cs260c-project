"""Evaluate a policy across a list of named tracks.

Usage
-----
    python3 -m eval.evaluate_track_suite --config configs/multitrack_ppo.yaml --type ppo --ckpt logs/ppo_multitrack_v1/policy_ppo_best.pt
"""

from __future__ import annotations

import argparse
from copy import deepcopy

import numpy as np
import torch
import yaml

from env.gate_race_aviary import make_env
from eval.evaluate import evaluate
from expert.expert_policy import ExpertPolicy
from policy.actor import ActorCritic, DronePolicy


def _build_actor(args, cfg, device: str):
    policy_cfg = cfg["policy"]
    if args.type == "expert":
        return ExpertPolicy()
    if args.type == "bc":
        actor = DronePolicy(
            obs_dim=policy_cfg["obs_dim"],
            action_dim=policy_cfg["action_dim"],
            hidden=policy_cfg["hidden"],
        )
        actor.load(args.ckpt, device=device).to(device).eval()
        return actor
    actor = ActorCritic(
        obs_dim=policy_cfg["obs_dim"],
        action_dim=policy_cfg["action_dim"],
        hidden=policy_cfg["hidden"],
        init_log_std=float(cfg["ppo"].get("init_log_std", -2.0)),
    )
    actor.load(args.ckpt, device=device).to(device).eval()
    return actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multitrack_ppo.yaml")
    parser.add_argument("--type", default="ppo", choices=["expert", "bc", "ppo"])
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--track-names",
        nargs="*",
        default=None,
        help="Override the suite track names. Defaults to validation_env.track_names from the config.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    base_env_cfg = deepcopy(cfg.get("validation_env", cfg["env"]))
    track_names = args.track_names or list(base_env_cfg.get("track_names", []))
    if not track_names:
        raise ValueError("No track names provided and validation_env.track_names is empty.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor = _build_actor(args, cfg, device)

    rows = []
    for idx, track_name in enumerate(track_names):
        env_cfg = deepcopy(base_env_cfg)
        env_cfg["track_name"] = track_name
        env_cfg.pop("track_names", None)
        env_cfg["sample_track_on_reset"] = False
        env = make_env(env_cfg)
        stats = evaluate(
            env,
            actor,
            n_episodes=args.episodes,
            device=device,
            seed=args.seed + idx * 100,
            print_stats=False,
        )
        env.close()
        rows.append((track_name, stats))

    print("\nTrack suite results:")
    for track_name, stats in rows:
        finish = "n/a" if stats["mean_finish_steps"] is None else f"{stats['mean_finish_steps']:.1f}"
        print(
            f"{track_name}"
            f" | completion={stats['completion_rate']:.2f}"
            f" | crash={stats['crash_rate']:.2f}"
            f" | gates={stats['mean_gates']:.2f}"
            f" | return={stats['mean_return']:.2f}"
            f" | finish={finish}"
        )

    avg_completion = float(np.mean([s["completion_rate"] for _, s in rows]))
    avg_crash = float(np.mean([s["crash_rate"] for _, s in rows]))
    avg_gates = float(np.mean([s["mean_gates"] for _, s in rows]))
    avg_return = float(np.mean([s["mean_return"] for _, s in rows]))
    print(
        f"\nAggregate | completion={avg_completion:.2f}"
        f" | crash={avg_crash:.2f}"
        f" | gates={avg_gates:.2f}"
        f" | return={avg_return:.2f}"
    )


if __name__ == "__main__":
    main()
