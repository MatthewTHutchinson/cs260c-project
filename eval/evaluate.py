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
from policy.actor import DronePolicy, ActorCritic


def evaluate(
    env,
    actor,
    n_episodes: int = 20,
    device: str = "cpu",
    seed: int = 0,
    realtime: bool = False,
    episode_delay: float = 0.0,
    print_stats: bool = True,
) -> dict:
    """Roll out actor for n_episodes and return an aggregated stats dict."""
    n_gates_total = getattr(env, "n_gates_total", env.n_gates)
    control_dt = 1.0 / float(getattr(env, "CTRL_FREQ", 20))
    single_obs_dim = 12

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
                obs_t  = torch.from_numpy(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = actor.act(obs_t, deterministic=True).squeeze(0).cpu().numpy()

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

            # Use the newest stacked frame for speed.
            vel_body_norm = float(np.linalg.norm(obs[-single_obs_dim : -single_obs_dim + 3]))
            ep_max_speed  = max(ep_max_speed, vel_body_norm)

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
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    env        = make_env(cfg["env"], gui=args.gui, camera_follow=args.camera_follow)
    policy_cfg = cfg["policy"]

    if args.type == "expert":
        actor = ExpertPolicy()
        print("[Eval] Actor: Expert (spline)")
    elif args.type == "bc":
        actor = DronePolicy(obs_dim=policy_cfg["obs_dim"],
                            action_dim=policy_cfg["action_dim"],
                            hidden=policy_cfg["hidden"])
        actor.load(args.ckpt, device=device).to(device).eval()
        print(f"[Eval] Actor: BC  ({args.ckpt})")
    elif args.type == "ppo":
        actor = ActorCritic(obs_dim=policy_cfg["obs_dim"],
                            action_dim=policy_cfg["action_dim"],
                            hidden=policy_cfg["hidden"])
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
    )
    env.close()


if __name__ == "__main__":
    main()
