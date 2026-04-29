"""Convenience script: run BC → DAgger → PPO in sequence.

Usage
-----
    python train_all.py
    python train_all.py --config configs/default.yaml --skip-bc --skip-dagger
"""

import argparse
import subprocess
import sys

PYTHON = sys.executable


def run(cmd: list[str]) -> None:
    print(f"\n{'='*60}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/default.yaml")
    parser.add_argument("--bc-out",      default="logs/bc")
    parser.add_argument("--dagger-out",  default="logs/dagger")
    parser.add_argument("--ppo-out",     default="logs/ppo")
    parser.add_argument("--skip-bc",     action="store_true")
    parser.add_argument("--skip-dagger", action="store_true")
    parser.add_argument("--skip-ppo",    action="store_true")
    args = parser.parse_args()

    if not args.skip_bc:
        run([PYTHON, "-m", "training.bc",
             "--config", args.config,
             "--out",    args.bc_out])

    if not args.skip_dagger:
        run([PYTHON, "-m", "training.dagger",
             "--config",  args.config,
             "--bc-ckpt", f"{args.bc_out}/policy_bc.pt",
             "--bc-data", args.bc_out,
             "--out",     args.dagger_out])

    if not args.skip_ppo:
        run([PYTHON, "-m", "training.ppo",
             "--config",      args.config,
             "--dagger-ckpt", f"{args.dagger_out}/policy_dagger.pt",
             "--dagger-data", args.dagger_out,
             "--out",         args.ppo_out])

    print("\n[train_all] Pipeline complete.")


if __name__ == "__main__":
    main()
