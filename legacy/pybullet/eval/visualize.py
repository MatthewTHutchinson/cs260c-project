"""Visualization entrypoint for the PyBullet drone racing simulator.

Usage
-----
    python3 -m eval.visualize --type expert
    python3 -m eval.visualize --type bc  --ckpt logs/bc/policy_bc.pt
    python3 -m eval.visualize --type ppo --ckpt logs/ppo/policy_ppo.pt
"""

import argparse

from eval.evaluate import main as evaluate_main

_PRESETS = {
    "state_champion": {
        "type": "ppo",
        "config": "configs/generalization_obs_v1.yaml",
        "ckpt": "logs/ppo_generalization_obs_v1/policy_ppo_best.pt",
    },
    "multimodal_v1": {
        "type": "ppo",
        "config": "configs/multimodal_obs_v1.yaml",
        "ckpt": "logs/ppo_multimodal_obs_v1/policy_ppo_best.pt",
    },
    "competition_multimodal_v1": {
        "type": "ppo",
        "config": "configs/competition_spec_multimodal_eval.yaml",
        "ckpt": "logs/ppo_multimodal_obs_v1/policy_ppo_best.pt",
    },
    "vision_bridge_baseline": {
        "type": "ppo",
        "config": "configs/vision_bridge_eval_v1.yaml",
        "ckpt": "logs/ppo_generalization_obs_v1/policy_ppo_best.pt",
    },
    "multimodal_v2_bc": {
        "type": "bc",
        "config": "configs/multimodal_obs_v2.yaml",
        "ckpt": "logs/bc_multimodal_obs_v2/policy_bc.pt",
    },
}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--type", default="expert", choices=["expert", "bc", "ppo"])
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-delay", type=float, default=1.0)
    parser.add_argument("--preset", choices=sorted(_PRESETS.keys()), default=None,
                        help="Load a common config/checkpoint pair for quick visualization.")
    parser.add_argument("--fixed-camera", action="store_true",
                        help="Disable the chase camera and leave PyBullet camera manual.")
    args = parser.parse_args()

    run_type = args.type
    config = args.config
    ckpt = args.ckpt
    if args.preset is not None:
        preset = _PRESETS[args.preset]
        run_type = preset["type"]
        config = preset["config"]
        ckpt = preset["ckpt"]

    forwarded = [
        "--type", run_type,
        "--config", config,
        "--episodes", str(args.episodes),
        "--seed", str(args.seed),
        "--gui",
        "--realtime",
        "--episode-delay", str(args.episode_delay),
    ]
    if ckpt is not None:
        forwarded.extend(["--ckpt", ckpt])
    if not args.fixed_camera:
        forwarded.append("--camera-follow")

    evaluate_main(forwarded)


if __name__ == "__main__":
    main()
