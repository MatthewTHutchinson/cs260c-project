"""Evaluation script — per-checkpoint metrics.

Reports: completion rate, crash/OOB rate, mean gates passed, mean return,
time-to-finish (for completed runs), and max speed.

Usage
-----
    python3 -m eval.evaluate --type expert --gui --realtime
    python3 -m eval.evaluate --type bc  --ckpt logs/bc/policy_bc.pt --gui --realtime
    python3 -m eval.evaluate --type ppo --ckpt logs/ppo/policy_ppo.pt --gui --realtime
"""

import argparse
import time
import numpy as np
import torch
import yaml

from env.gate_race_aviary import make_env
from expert.expert_policy import ExpertPolicy
from policy.actor import build_actor_critic, build_deterministic_policy, policy_uses_images
from policy.runtime import get_env_image, image_batch_to_tensor, obs_batch_to_tensor, policy_act


def evaluate(
    env,
    actor,
    n_episodes: int = 20,
    device: str = "cpu",
    seed: int = 0,
    realtime: bool = False,
    episode_delay: float = 0.0,
    print_stats: bool = True,
    obs_noise_std: float = 0.0,
    action_noise_std: float = 0.0,
) -> dict:
    """Roll out actor for n_episodes and return an aggregated stats dict."""
    n_gates_total = getattr(env, "n_gates_total", env.n_gates)
    control_dt = 1.0 / float(getattr(env, "CTRL_FREQ", 20))
    rng = np.random.default_rng(seed + 1009)

    returns, lengths, gates_list = [], [], []
    completions, crashes, oobs   = 0, 0, 0
    finish_times: list[int]      = []
    max_speeds:   list[float]    = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_return   = 0.0
        ep_len      = 0
        gates        = 0
        crashed      = False
        out_of_bounds= False
        ep_max_speed = 0.0
        lap_finish_step = None
        done         = False

        while not done:
            if isinstance(actor, ExpertPolicy):
                action = actor.act(env)
            else:
                policy_obs = obs
                if obs_noise_std > 0.0:
                    policy_obs = (
                        obs + rng.normal(0.0, obs_noise_std, size=obs.shape).astype(np.float32)
                    )
                obs_t = obs_batch_to_tensor(policy_obs, device)
                img_t = None
                if getattr(actor, "expects_image", False):
                    img_t = image_batch_to_tensor(get_env_image(env), device)
                with torch.no_grad():
                    action = policy_act(actor, obs_t, img_t, deterministic=True).squeeze(0).cpu().numpy()
            if action_noise_std > 0.0:
                action = action + rng.normal(0.0, action_noise_std, size=action.shape).astype(np.float32)
                action = np.clip(action, -1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return    += float(reward)
            ep_len       += 1
            gates         = info.get("gates_passed", gates)
            crashed       = crashed  or info.get("collision", False)
            out_of_bounds = out_of_bounds or (
                terminated and not info.get("collision", False)
            )
            if lap_finish_step is None and gates >= n_gates_total:
                lap_finish_step = ep_len

            # Measure true speed from the simulator state so observation noise
            # does not contaminate robustness metrics.
            if hasattr(env, "get_full_state"):
                vel_world = env.get_full_state()["vel"]
                ep_max_speed = max(ep_max_speed, float(np.linalg.norm(vel_world)))
            else:
                single_obs_dim = int(getattr(env, "single_obs_dim", 12))
                newest_frame = obs[-single_obs_dim:]
                ep_max_speed = max(ep_max_speed, float(np.linalg.norm(newest_frame[:3])))

            if realtime:
                time.sleep(control_dt)

            done = terminated or truncated

        returns.append(ep_return)
        lengths.append(ep_len)
        gates_list.append(gates)
        max_speeds.append(ep_max_speed)

        if gates >= n_gates_total:
            completions += 1
            finish_times.append(lap_finish_step if lap_finish_step is not None else ep_len)
        if crashed:
            crashes += 1
        if out_of_bounds:
            oobs += 1
        if episode_delay > 0.0 and ep < n_episodes - 1:
            time.sleep(episode_delay)

    stats = {
        "episodes":        n_episodes,
        "completion_rate": completions / n_episodes,
        "crash_rate":      crashes     / n_episodes,
        "oob_rate":        oobs        / n_episodes,
        "mean_return":     float(np.mean(returns)),
        "std_return":      float(np.std(returns)),
        "mean_gates":      float(np.mean(gates_list)),
        "max_gates":       int(np.max(gates_list)),
        "mean_finish_steps": float(np.mean(finish_times)) if finish_times else None,
        "mean_max_speed":  float(np.mean(max_speeds)),
        "max_max_speed":   float(np.max(max_speeds)),
    }

    if print_stats:
        _print_stats(stats)
    return stats


def _print_stats(s: dict) -> None:
    print(f"\n  Episodes        : {s['episodes']}")
    print(f"  Completion rate : {s['completion_rate']*100:.1f}%")
    print(f"  Crash rate      : {s['crash_rate']*100:.1f}%")
    print(f"  OOB rate        : {s['oob_rate']*100:.1f}%")
    print(f"  Mean return     : {s['mean_return']:.2f} ± {s['std_return']:.2f}")
    print(f"  Gates passed    : {s['mean_gates']:.2f} avg  (max {s['max_gates']})")
    if s["mean_finish_steps"] is not None:
        print(f"  Time-to-finish  : {s['mean_finish_steps']:.1f} steps (first completed lap)")
    print(f"  Max speed       : {s['max_max_speed']:.2f} m/s  (mean {s['mean_max_speed']:.2f})")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/default.yaml")
    parser.add_argument("--ckpt",     default=None)
    parser.add_argument("--type",     default="bc", choices=["bc", "ppo", "expert"])
    parser.add_argument("--episodes", type=int,  default=20)
    parser.add_argument("--gui",      action="store_true")
    parser.add_argument("--camera-follow", action="store_true",
                        help="In GUI mode, keep the camera centered on the drone.")
    parser.add_argument("--realtime", action="store_true",
                        help="Sleep between control steps so GUI playback runs in real time.")
    parser.add_argument("--episode-delay", type=float, default=1.0,
                        help="Seconds to pause between episodes when visualizing.")
    parser.add_argument("--seed",     type=int,  default=0)
    parser.add_argument("--obs-noise-std", type=float, default=0.0,
                        help="Gaussian std added to policy inputs only during evaluation.")
    parser.add_argument("--action-noise-std", type=float, default=0.0,
                        help="Gaussian std added to actions before env.step during evaluation.")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    env        = make_env(cfg["env"], gui=args.gui, camera_follow=args.camera_follow)
    policy_cfg = cfg["policy"]
    obs_dim    = int(env.observation_space.shape[0])
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

    if args.type == "expert":
        actor = ExpertPolicy()
        print("[Eval] Actor: Expert (lookahead)")
    elif args.type == "bc":
        actor = build_deterministic_policy(policy_cfg, obs_dim=obs_dim)
        actor.load(args.ckpt, device=device).to(device).eval()
        print(f"[Eval] Actor: BC  ({args.ckpt})")
    elif args.type == "ppo":
        actor = build_actor_critic(policy_cfg)
        actor.load(args.ckpt, device=device).to(device).eval()
        print(f"[Eval] Actor: PPO ({args.ckpt})")

    if args.gui:
        print("[Eval] GUI enabled")
        if args.camera_follow:
            print("[Eval] Chase camera enabled")
        if args.realtime:
            print(f"[Eval] Real-time playback at {getattr(env, 'CTRL_FREQ', 20)} Hz")

    evaluate(
        env,
        actor,
        n_episodes=args.episodes,
        device=device,
        seed=args.seed,
        realtime=args.realtime,
        episode_delay=args.episode_delay if args.gui else 0.0,
        obs_noise_std=args.obs_noise_std,
        action_noise_std=args.action_noise_std,
    )
    env.close()


if __name__ == "__main__":
    main()
