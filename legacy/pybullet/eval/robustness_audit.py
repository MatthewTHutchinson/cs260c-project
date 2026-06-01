"""Run a configurable robustness audit over held-out tracks and stress scenarios.

Usage
-----
    python3 -m eval.robustness_audit \
      --config configs/robustness_obs_v1.yaml \
      --type ppo \
      --ckpt logs/ppo_generalization_obs_v1/policy_ppo_best.pt \
      --out logs/robustness_obs_v1
"""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

from env.gate_race_aviary import compute_obs_dim_from_config, make_env
from eval.evaluate import evaluate
from expert.expert_policy import ExpertPolicy
from policy.actor import build_actor_critic, build_deterministic_policy, policy_uses_images


def _build_actor(actor_type: str, ckpt: str | None, cfg: dict, device: str, obs_dim: int):
    policy_cfg = cfg["policy"]
    if actor_type == "expert":
        return ExpertPolicy()
    if actor_type == "bc":
        actor = build_deterministic_policy(policy_cfg, obs_dim=obs_dim)
        actor.load(ckpt, device=device).to(device).eval()
        return actor
    build_cfg = dict(policy_cfg)
    build_cfg["init_log_std"] = float(cfg.get("ppo", {}).get("init_log_std", -2.0))
    actor = build_actor_critic(build_cfg)
    actor.load(ckpt, device=device).to(device).eval()
    return actor


def _find_validation_suite(cfg: dict, suite_name: str) -> dict:
    for suite in cfg.get("validation_suites", []):
        if str(suite.get("name")) == suite_name:
            return suite
    raise KeyError(f"Unknown validation suite '{suite_name}' in config.")


def _merge_env(base_env: dict, overrides: dict | None) -> dict:
    env_cfg = deepcopy(base_env)
    for key, value in (overrides or {}).items():
        env_cfg[key] = value
    return env_cfg


def _resolve_scenario_tracks(cfg: dict, scenario: dict) -> list[tuple[str, dict, str]]:
    """Return [(label, env_cfg, track_name), ...] for one scenario."""
    if "suite_name" in scenario:
        suite = _find_validation_suite(cfg, str(scenario["suite_name"]))
        base_env = deepcopy(suite.get("env", cfg.get("validation_env", cfg["env"])))
        track_names = list(scenario.get("track_names", base_env.get("track_names", [])))
        if not track_names:
            raise ValueError(f"Scenario '{scenario['name']}' has no tracks.")
        env_cfg = _merge_env(base_env, scenario.get("env_overrides"))
        return [(f"{scenario['name']}/{track_name}", env_cfg, track_name) for track_name in track_names]

    track_names = list(scenario.get("track_names", []))
    if not track_names:
        raise ValueError(f"Scenario '{scenario['name']}' must define suite_name or track_names.")
    base_env = _merge_env(cfg["env"], scenario.get("env_overrides"))
    return [(f"{scenario['name']}/{track_name}", base_env, track_name) for track_name in track_names]


def _aggregate(rows: list[dict]) -> dict:
    finish_vals = [row["mean_finish_steps"] for row in rows if row["mean_finish_steps"] is not None]
    return {
        "completion_rate": float(np.mean([row["completion_rate"] for row in rows])),
        "crash_rate": float(np.mean([row["crash_rate"] for row in rows])),
        "oob_rate": float(np.mean([row["oob_rate"] for row in rows])),
        "mean_return": float(np.mean([row["mean_return"] for row in rows])),
        "mean_gates": float(np.mean([row["mean_gates"] for row in rows])),
        "mean_finish_steps": float(np.mean(finish_vals)) if finish_vals else None,
        "mean_max_speed": float(np.mean([row["mean_max_speed"] for row in rows])),
        "min_completion_rate": float(np.min([row["completion_rate"] for row in rows])),
        "max_crash_rate": float(np.max([row["crash_rate"] for row in rows])),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, cfg_path: str, ckpt: str | None, scenarios: list[dict]) -> None:
    lines = [
        "# Robustness Audit",
        "",
        f"- Config: `{cfg_path}`",
        f"- Checkpoint: `{ckpt}`" if ckpt else "- Checkpoint: expert",
        "",
    ]
    for scenario in scenarios:
        agg = scenario["aggregate"]
        lines.extend([
            f"## {scenario['name']}",
            "",
            f"- Episodes per track: `{scenario['episodes']}`",
            f"- Completion: `{agg['completion_rate']*100:.1f}%`",
            f"- Crash: `{agg['crash_rate']*100:.1f}%`",
            f"- Return: `{agg['mean_return']:.2f}`",
            f"- Gates: `{agg['mean_gates']:.2f}`",
            f"- Finish steps: `{agg['mean_finish_steps']:.1f}`" if agg["mean_finish_steps"] is not None else "- Finish steps: `n/a`",
            f"- Mean max speed: `{agg['mean_max_speed']:.2f}` m/s",
            "",
            "| Track | Completion | Crash | Return | Gates | Finish |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in scenario["tracks"]:
            finish = "n/a" if row["mean_finish_steps"] is None else f"{row['mean_finish_steps']:.1f}"
            lines.append(
                f"| {row['track_name']} | {row['completion_rate']*100:.1f}% | "
                f"{row['crash_rate']*100:.1f}% | {row['mean_return']:.2f} | "
                f"{row['mean_gates']:.2f} | {finish} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/robustness_obs_v1.yaml")
    parser.add_argument("--type", default="ppo", choices=["expert", "bc", "ppo"])
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override episodes per track for every scenario.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="logs/robustness_audit")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Optional subset of scenario names to run.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    audit_cfg = cfg.get("robustness_audit")
    if not audit_cfg or not audit_cfg.get("scenarios"):
        raise ValueError("Config must define robustness_audit.scenarios.")

    scenarios = list(audit_cfg["scenarios"])
    if args.scenarios:
        wanted = set(args.scenarios)
        scenarios = [scenario for scenario in scenarios if scenario["name"] in wanted]
        if not scenarios:
            raise ValueError(f"No matching scenarios found for {args.scenarios}.")

    obs_dim = compute_obs_dim_from_config(cfg["env"])
    cfg_obs_dim = int(cfg["policy"].get("obs_dim", obs_dim))
    if cfg_obs_dim != obs_dim:
        raise ValueError(f"policy.obs_dim={cfg_obs_dim} does not match env observation dim {obs_dim}")
    if policy_uses_images(cfg["policy"]):
        ref_env = make_env(cfg["env"])
        env_image_shape = tuple(int(v) for v in getattr(ref_env, "policy_image_shape", ()))
        ref_env.close()
        cfg_image_shape = tuple(int(v) for v in cfg["policy"].get("image_shape", []))
        if cfg_image_shape != env_image_shape:
            raise ValueError(
                f"policy.image_shape={cfg_image_shape} does not match env image shape {env_image_shape}"
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor = _build_actor(args.type, args.ckpt, cfg, device, obs_dim)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_summaries: list[dict] = []
    csv_rows: list[dict] = []

    scenario_offset = 0
    for scenario in scenarios:
        episodes = int(args.episodes or scenario.get("episodes", audit_cfg.get("episodes", 20)))
        obs_noise_std = float(scenario.get("obs_noise_std", 0.0))
        action_noise_std = float(scenario.get("action_noise_std", 0.0))
        track_specs = _resolve_scenario_tracks(cfg, scenario)

        track_rows = []
        print(f"\n[Audit] Scenario: {scenario['name']} ({episodes} episodes/track)")
        for track_idx, (label, base_env_cfg, track_name) in enumerate(track_specs):
            env_cfg = deepcopy(base_env_cfg)
            env_cfg["track_name"] = track_name
            env_cfg.pop("track_names", None)
            env_cfg["sample_track_on_reset"] = False

            env = make_env(env_cfg)
            stats = evaluate(
                env,
                actor,
                n_episodes=episodes,
                device=device,
                seed=args.seed + scenario_offset * 1000 + track_idx * 100,
                obs_noise_std=obs_noise_std,
                action_noise_std=action_noise_std,
                print_stats=False,
            )
            env.close()

            row = {
                "scenario": scenario["name"],
                "track_name": track_name,
                "episodes": episodes,
                "obs_noise_std": obs_noise_std,
                "action_noise_std": action_noise_std,
                "disturbance_force_std": float(env_cfg.get("disturbance_force_std", 0.0)),
                "disturbance_torque_std": float(env_cfg.get("disturbance_torque_std", 0.0)),
                "start_longitudinal_jitter": float(env_cfg.get("start_longitudinal_jitter", 0.0)),
                "start_lateral_jitter": float(env_cfg.get("start_lateral_jitter", 0.0)),
                "start_vertical_jitter": float(env_cfg.get("start_vertical_jitter", 0.0)),
                "start_yaw_jitter": float(env_cfg.get("start_yaw_jitter", 0.0)),
                **stats,
            }
            track_rows.append(row)
            csv_rows.append(row)
            finish = "n/a" if stats["mean_finish_steps"] is None else f"{stats['mean_finish_steps']:.1f}"
            print(
                f"  {label}"
                f" | completion={stats['completion_rate']:.2f}"
                f" | crash={stats['crash_rate']:.2f}"
                f" | return={stats['mean_return']:.2f}"
                f" | finish={finish}"
            )

        aggregate = _aggregate(track_rows)
        scenario_summaries.append(
            {
                "name": scenario["name"],
                "episodes": episodes,
                "obs_noise_std": obs_noise_std,
                "action_noise_std": action_noise_std,
                "env_overrides": deepcopy(scenario.get("env_overrides", {})),
                "aggregate": aggregate,
                "tracks": track_rows,
            }
        )
        finish = "n/a" if aggregate["mean_finish_steps"] is None else f"{aggregate['mean_finish_steps']:.1f}"
        print(
            f"  Aggregate"
            f" | completion={aggregate['completion_rate']:.2f}"
            f" | crash={aggregate['crash_rate']:.2f}"
            f" | return={aggregate['mean_return']:.2f}"
            f" | finish={finish}"
        )
        scenario_offset += 1

    _write_csv(out_dir / "per_track.csv", csv_rows)
    (out_dir / "summary.json").write_text(json.dumps(scenario_summaries, indent=2))
    _write_markdown(out_dir / "report.md", args.config, args.ckpt, scenario_summaries)

    print(f"\n[Audit] Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
