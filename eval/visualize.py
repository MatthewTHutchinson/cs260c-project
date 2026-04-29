"""Visualization entrypoint for the PyBullet drone racing simulator.

Usage
-----
    python3 -m eval.visualize --type expert
    python3 -m eval.visualize --type bc  --ckpt logs/bc/policy_bc.pt
    python3 -m eval.visualize --type ppo --ckpt logs/ppo/policy_ppo.pt
"""

import argparse

from eval.evaluate import main as evaluate_main


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--type", default="expert", choices=["expert", "bc", "ppo"])
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-delay", type=float, default=1.0)
    parser.add_argument("--fixed-camera", action="store_true",
                        help="Disable the chase camera and leave PyBullet camera manual.")
    args = parser.parse_args()

    forwarded = [
        "--type", args.type,
        "--config", args.config,
        "--episodes", str(args.episodes),
        "--seed", str(args.seed),
        "--gui",
        "--realtime",
        "--episode-delay", str(args.episode_delay),
    ]
    if args.ckpt is not None:
        forwarded.extend(["--ckpt", args.ckpt])
    if not args.fixed_camera:
        forwarded.append("--camera-follow")

    evaluate_main(forwarded)


if __name__ == "__main__":
    main()
