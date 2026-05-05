"""Rank PPO checkpoints by evaluation performance.

Usage
-----
    python3 -m eval.rank_checkpoints --dir logs/ppo_stable_v1
    python3 -m eval.rank_checkpoints --dir logs/ppo_stable_v1 --episodes 5 --top-k 8
"""

import argparse
from pathlib import Path

import torch
import yaml

from env.gate_race_aviary import make_env
from eval.evaluate import evaluate
from policy.actor import build_actor_critic, policy_uses_images


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem == "policy_ppo":
        return (10**12, stem)
    try:
        return (int(stem.split("_")[-1]), stem)
    except ValueError:
        return (10**12 - 1, stem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory containing policy_ppo*.pt checkpoints")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    ckpt_dir = Path(args.dir)
    checkpoints = sorted(ckpt_dir.glob("policy_ppo*.pt"), key=_checkpoint_sort_key)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env(cfg["env"])
    policy_cfg = cfg["policy"]
    if policy_uses_images(policy_cfg):
        cfg_image_shape = tuple(int(v) for v in policy_cfg.get("image_shape", []))
        env_image_shape = tuple(int(v) for v in getattr(env, "policy_image_shape", ()))
        if cfg_image_shape != env_image_shape:
            raise ValueError(
                f"policy.image_shape={cfg_image_shape} does not match env image shape {env_image_shape}"
            )

    rows = []
    for ckpt in checkpoints:
        build_cfg = dict(policy_cfg)
        build_cfg["init_log_std"] = float(cfg["ppo"].get("init_log_std", -2.0))
        actor = build_actor_critic(build_cfg)
        actor.load(str(ckpt), device=device).to(device).eval()
        stats = evaluate(
            env,
            actor,
            n_episodes=args.episodes,
            device=device,
            seed=args.seed,
            print_stats=False,
        )
        rows.append({
            "checkpoint": str(ckpt),
            **stats,
        })

    env.close()

    rows.sort(
        key=lambda r: (
            r["completion_rate"],
            r["mean_gates"],
            r["mean_return"],
            -r["crash_rate"],
        ),
        reverse=True,
    )

    print("\nTop checkpoints:")
    for row in rows[: args.top_k]:
        print(
            f"{row['checkpoint']}"
            f" | completion={row['completion_rate']:.2f}"
            f" | mean_gates={row['mean_gates']:.2f}"
            f" | mean_return={row['mean_return']:.2f}"
            f" | crash={row['crash_rate']:.2f}"
        )


if __name__ == "__main__":
    main()
