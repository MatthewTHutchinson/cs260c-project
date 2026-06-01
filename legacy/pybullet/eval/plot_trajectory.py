"""Plot top-down drone trajectories for presentation/report assets.

Examples
--------
python -m eval.plot_trajectory --type ppo --ckpt logs/ppo_generalization_obs_v1/policy_ppo_best.pt --output assets/presentation/state_trajectory.png
python -m eval.plot_trajectory --type expert --output assets/presentation/expert_trajectory.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from env.gate_race_aviary import make_env
from expert.expert_policy import ExpertPolicy
from policy.actor import build_actor_critic, build_deterministic_policy, policy_uses_images
from policy.runtime import get_env_image, image_batch_to_tensor, obs_batch_to_tensor, policy_act


def _load_actor(actor_type: str, ckpt: str | None, cfg: dict, env, device: str):
    policy_cfg = cfg["policy"]
    obs_dim = int(env.observation_space.shape[0])
    cfg_obs_dim = int(policy_cfg.get("obs_dim", obs_dim))
    if cfg_obs_dim != obs_dim:
        raise ValueError(f"policy.obs_dim={cfg_obs_dim} does not match env observation dim {obs_dim}")
    if policy_uses_images(policy_cfg):
        cfg_image_shape = tuple(int(v) for v in policy_cfg.get("image_shape", []))
        env_image_shape = tuple(int(v) for v in getattr(env, "policy_image_shape", ()))
        if cfg_image_shape != env_image_shape:
            raise ValueError(
                f"policy.image_shape={cfg_image_shape} does not match env image shape {env_image_shape}"
            )

    if actor_type == "expert":
        return ExpertPolicy()
    if actor_type == "bc":
        actor = build_deterministic_policy(policy_cfg, obs_dim=obs_dim)
        actor.load(ckpt, device=device).to(device).eval()
        return actor
    if actor_type == "ppo":
        actor = build_actor_critic(policy_cfg)
        actor.load(ckpt, device=device).to(device).eval()
        return actor
    raise ValueError(f"Unknown actor type: {actor_type}")


def _step_actor(actor, obs: np.ndarray, env, device: str) -> np.ndarray:
    if isinstance(actor, ExpertPolicy):
        return actor.act(env)
    obs_t = obs_batch_to_tensor(obs, device)
    img_t = None
    if getattr(actor, "expects_image", False):
        img_t = image_batch_to_tensor(get_env_image(env), device)
    with torch.no_grad():
        return policy_act(actor, obs_t, img_t, deterministic=True).squeeze(0).cpu().numpy()


def _gate_axes(gate: dict) -> tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(gate["normal"], dtype=np.float64)
    normal = normal / max(np.linalg.norm(normal), 1e-9)
    lateral = np.array([-normal[1], normal[0], 0.0], dtype=np.float64)
    lateral = lateral / max(np.linalg.norm(lateral), 1e-9)
    return normal, lateral


def _draw_gate(ax, gate: dict, idx: int) -> None:
    center = np.asarray(gate["center"], dtype=np.float64)
    normal, lateral = _gate_axes(gate)
    half_width = float(gate.get("radius", 0.75))
    p0 = center - lateral * half_width
    p1 = center + lateral * half_width
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#d28422", linewidth=4, solid_capstyle="round")
    ax.arrow(
        center[0],
        center[1],
        normal[0] * 0.35,
        normal[1] * 0.35,
        head_width=0.08,
        head_length=0.12,
        color="#2b728a",
        length_includes_head=True,
    )
    ax.text(center[0], center[1] + 0.18, f"G{idx}", ha="center", va="bottom", fontsize=9, color="#2f2f2f")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generalization_obs_v1.yaml")
    parser.add_argument("--type", default="ppo", choices=["expert", "bc", "ppo"])
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--track-name", default="heldout_diamond")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=650)
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg = dict(cfg["env"])
    env_cfg["track_name"] = args.track_name
    env_cfg.pop("track_names", None)
    env_cfg["sample_track_on_reset"] = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env(env_cfg, gui=False, camera_follow=False)
    actor = _load_actor(args.type, args.ckpt, cfg, env, device)

    obs, info = env.reset(seed=args.seed)
    positions = []
    gates_passed = 0
    done = False
    step = 0
    while not done and step < args.max_steps:
        state = env.get_full_state()
        positions.append(np.asarray(state["pos"], dtype=np.float64).copy())
        action = _step_actor(actor, obs, env, device)
        obs, _, terminated, truncated, info = env.step(action)
        gates_passed = int(info.get("gates_passed", gates_passed))
        done = terminated or truncated
        step += 1
    env.close()

    pts = np.asarray(positions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=180)
    fig.patch.set_facecolor("#f8f7f2")
    ax.set_facecolor("#f8f7f2")
    for idx, gate in enumerate(env.gates):
        _draw_gate(ax, gate, idx)
    if len(pts):
        ax.plot(pts[:, 0], pts[:, 1], color="#1e5574", linewidth=2.5, label=args.type.upper())
        ax.scatter(pts[0, 0], pts[0, 1], color="#3a7f4f", s=55, zorder=5, label="start")
        ax.scatter(pts[-1, 0], pts[-1, 1], color="#b74242", s=55, zorder=5, label="end")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#dad6c7", linewidth=0.8)
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    title = args.title or f"{args.type.upper()} trajectory on {args.track_name}"
    ax.set_title(f"{title}\nsteps={step}, gates_passed={gates_passed}", pad=12)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f"[plot_trajectory] Wrote {output}")


if __name__ == "__main__":
    main()
