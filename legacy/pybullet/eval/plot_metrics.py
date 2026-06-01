"""Plot training curves from saved .npy loss arrays.

Usage
-----
    python -m eval.plot_metrics                       # all phases
    python -m eval.plot_metrics --phase bc            # BC only
    python -m eval.plot_metrics --out figs/curves.png
"""

import argparse
import os
import numpy as np


def _load_if_exists(path: str):
    return np.load(path) if os.path.exists(path) else None


def _metrics_column(metrics, key: str, key_names=None):
    if metrics is None or len(metrics) == 0:
        return None
    first = metrics[0]
    if isinstance(first, dict):
        return np.array([m.get(key, np.nan) for m in metrics], dtype=np.float32)
    if key_names is not None and key in key_names:
        idx = key_names.index(key)
        return metrics[:, idx]
    key_to_idx = {"pg": 0, "vf": 1, "ent": 2, "bc": 3, "total": 4}
    if key not in key_to_idx:
        return None
    return metrics[:, key_to_idx[key]]


def plot(bc_dir="logs/bc", dagger_dir="logs/dagger", ppo_dir="logs/ppo",
         phase="all", out=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed — skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Training curves", fontsize=13)

    # BC
    bc_loss = _load_if_exists(os.path.join(bc_dir, "bc_losses.npy"))
    ax = axes[0]
    if bc_loss is not None:
        ax.plot(bc_loss, color="steelblue")
        ax.set_title("BC — MSE loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE")
    else:
        ax.set_title("BC — no data")
    ax.grid(True, alpha=0.3)

    # DAgger
    dag_loss = _load_if_exists(os.path.join(dagger_dir, "dagger_losses.npy"))
    ax = axes[1]
    if dag_loss is not None:
        ax.plot(dag_loss, color="darkorange")
        ax.set_title("DAgger — MSE loss")
        ax.set_xlabel("epoch (all rounds)")
        ax.set_ylabel("MSE")
    else:
        ax.set_title("DAgger — no data")
    ax.grid(True, alpha=0.3)

    # PPO
    ppo_metrics = _load_if_exists(os.path.join(ppo_dir, "ppo_metrics.npy"))
    ppo_key_names = None
    ppo_keys = _load_if_exists(os.path.join(ppo_dir, "ppo_metric_keys.npy"))
    if ppo_keys is not None:
        ppo_key_names = [str(k) for k in ppo_keys.tolist()]
    ax = axes[2]
    if ppo_metrics is not None:
        updates = np.arange(len(ppo_metrics))
        pg = _metrics_column(ppo_metrics, "pg", ppo_key_names)
        vf = _metrics_column(ppo_metrics, "vf", ppo_key_names)
        bc = _metrics_column(ppo_metrics, "bc", ppo_key_names)
        crash = _metrics_column(ppo_metrics, "rollout_crash_rate", ppo_key_names)
        if pg is not None:
            ax.plot(updates, pg, label="pg", color="crimson")
        if vf is not None:
            ax.plot(updates, vf, label="vf", color="seagreen")
        if bc is not None:
            ax.plot(updates, bc, label="bc", color="purple", linestyle="--")
        if crash is not None and not np.isnan(crash).all():
            ax.plot(updates, crash, label="crash", color="black", linestyle=":")
        ax.set_title("PPO — losses per update")
        ax.set_xlabel("PPO update")
        ax.set_ylabel("loss")
        ax.legend(fontsize=8)
    else:
        ax.set_title("PPO — no data")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        plt.savefig(out, dpi=120)
        print(f"[plot] Saved to {out}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-dir",     default="logs/bc")
    parser.add_argument("--dagger-dir", default="logs/dagger")
    parser.add_argument("--ppo-dir",    default="logs/ppo")
    parser.add_argument("--phase",      default="all", choices=["all", "bc", "dagger", "ppo"])
    parser.add_argument("--out",        default=None, help="Save path (e.g. figs/curves.png)")
    args = parser.parse_args()
    plot(args.bc_dir, args.dagger_dir, args.ppo_dir, args.phase, args.out)


if __name__ == "__main__":
    main()
