"""Convenience script: run BC → DAgger → PPO in sequence.

Usage
-----
    python train_all.py
    python train_all.py --config configs/default.yaml --skip-bc --skip-dagger
"""

import argparse
import os
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


def _exists(path: str) -> bool:
    return os.path.exists(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/default.yaml")
    parser.add_argument("--bc-out",      default="logs/bc")
    parser.add_argument("--dagger-out",  default="logs/dagger")
    parser.add_argument("--ppo-out",     default="logs/ppo")
    parser.add_argument("--skip-bc",     action="store_true")
    parser.add_argument("--skip-dagger", action="store_true")
    parser.add_argument("--skip-ppo",    action="store_true")
    parser.add_argument("--resume",      action="store_true",
                        help="Resume from existing stage artifacts when possible.")
    args = parser.parse_args()

    bc_ckpt = os.path.join(args.bc_out, "policy_bc.pt")
    dagger_ckpt = os.path.join(args.dagger_out, "policy_dagger.pt")
    ppo_ckpt = os.path.join(args.ppo_out, "policy_ppo.pt")

    if not args.skip_bc:
        if args.resume and _exists(bc_ckpt):
            print(f"[train_all] Resume: skipping BC because {bc_ckpt} already exists.")
        else:
            run([PYTHON, "-m", "training.bc",
                 "--config", args.config,
                 "--out",    args.bc_out])

    if not args.skip_dagger:
        if args.resume and _exists(dagger_ckpt):
            print(f"[train_all] Resume: skipping DAgger because {dagger_ckpt} already exists.")
        else:
            cmd = [PYTHON, "-m", "training.dagger",
                   "--config",  args.config,
                   "--bc-ckpt", bc_ckpt,
                   "--bc-data", args.bc_out,
                   "--out",     args.dagger_out]
            if args.resume:
                cmd.append("--resume")
            run(cmd)

    if not args.skip_ppo:
        if args.resume and _exists(ppo_ckpt):
            print(f"[train_all] Resume: skipping PPO because {ppo_ckpt} already exists.")
        else:
            cmd = [PYTHON, "-m", "training.ppo",
                   "--config",      args.config,
                   "--dagger-ckpt", dagger_ckpt,
                   "--dagger-data", args.dagger_out,
                   "--out",         args.ppo_out]
            if args.resume:
                cmd.append("--resume")
            run(cmd)

    print("\n[train_all] Pipeline complete.")


if __name__ == "__main__":
    main()
