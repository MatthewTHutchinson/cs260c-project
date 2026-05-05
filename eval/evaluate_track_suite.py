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

from env.gate_race_aviary import make_env, compute_obs_dim_from_config
from eval.evaluate import evaluate
from expert.expert_policy import ExpertPolicy
from policy.actor import build_actor_critic, build_deterministic_policy, policy_uses_images


def _build_actor(args, cfg, device: str, obs_dim: int):
    policy_cfg = cfg["policy"]
    if args.type == "expert":
        return ExpertPolicy()
    if args.type == "bc":
        actor = build_deterministic_policy(policy_cfg, obs_dim=obs_dim)
        actor.load(args.ckpt, device=device).to(device).eval()
        return actor
    build_cfg = dict(policy_cfg)
    build_cfg["init_log_std"] = float(cfg["ppo"].get("init_log_std", -2.0))
    actor = build_actor_critic(build_cfg)
    actor.load(args.ckpt, device=device).to(device).eval()
    return actor


def _resolve_suite_specs(cfg: dict, track_override: list[str] | None) -> list[tuple[str, dict, list[str]]]:
    """Return [(suite_name, env_cfg, track_names), ...] for evaluation."""
    if track_override:
        env_cfg = deepcopy(cfg.get("validation_env", cfg["env"]))
        return [("override", env_cfg, list(track_override))]

    validation_suites = cfg.get("validation_suites")
    if validation_suites:
        specs = []
        for idx, suite in enumerate(validation_suites):
            suite_name = str(suite.get("name", f"suite_{idx + 1}"))
            env_cfg = deepcopy(suite.get("env", cfg.get("validation_env", cfg["env"])))
            track_names = list(env_cfg.get("track_names", []))
            if not track_names:
                raise ValueError(f"validation suite '{suite_name}' has no track_names")
            specs.append((suite_name, env_cfg, track_names))
        return specs

    env_cfg = deepcopy(cfg.get("validation_env", cfg["env"]))
    track_names = list(env_cfg.get("track_names", []))
    if not track_names:
        raise ValueError("No track names provided and validation_env.track_names is empty.")
    return [("default", env_cfg, track_names)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multitrack_ppo.yaml")
    parser.add_argument("--type", default="ppo", choices=["expert", "bc", "ppo"])
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-noise-std", type=float, default=0.0)
    parser.add_argument("--action-noise-std", type=float, default=0.0)
    parser.add_argument(
        "--track-names",
        nargs="*",
        default=None,
        help="Override the suite track names. Defaults to validation_env.track_names from the config.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    suite_specs = _resolve_suite_specs(cfg, args.track_names)
    obs_dims = [compute_obs_dim_from_config(env_cfg) for _, env_cfg, _ in suite_specs]
    if len(set(obs_dims)) != 1:
        raise ValueError(f"validation suites use mismatched observation dims: {obs_dims}")
    obs_dim = obs_dims[0]
    cfg_obs_dim = int(cfg["policy"].get("obs_dim", obs_dim))
    if cfg_obs_dim != obs_dim:
        raise ValueError(f"policy.obs_dim={cfg_obs_dim} does not match validation observation dim {obs_dim}")
    if policy_uses_images(cfg["policy"]):
        ref_env = make_env(suite_specs[0][1])
        env_image_shape = tuple(int(v) for v in getattr(ref_env, "policy_image_shape", ()))
        ref_env.close()
        cfg_image_shape = tuple(int(v) for v in cfg["policy"].get("image_shape", []))
        if cfg_image_shape != env_image_shape:
            raise ValueError(
                f"policy.image_shape={cfg_image_shape} does not match validation env image shape {env_image_shape}"
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor = _build_actor(args, cfg, device, obs_dim)

    rows = []
    suite_count = len(suite_specs)
    row_idx = 0
    for suite_name, base_env_cfg, track_names in suite_specs:
        for track_name in track_names:
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
                seed=args.seed + row_idx * 100,
                obs_noise_std=args.obs_noise_std,
                action_noise_std=args.action_noise_std,
                print_stats=False,
            )
            env.close()
            label = f"{suite_name}/{track_name}" if suite_count > 1 else track_name
            rows.append((label, stats))
            row_idx += 1

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
