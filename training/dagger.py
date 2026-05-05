"""Phase C — DAgger (Dataset Aggregation).

Iteratively rolls out the current policy, queries the expert on visited
states, aggregates the dataset, and retrains with BC loss.

Usage
-----
    python -m training.dagger --config configs/default.yaml \
        --bc-ckpt logs/bc/policy_bc.pt \
        --bc-data logs/bc \
        --out logs/dagger
"""

import argparse
import os
import numpy as np
import torch
import yaml

from env.gate_race_aviary import make_env
from expert.expert_policy import ExpertPolicy
from policy.actor import build_deterministic_policy, policy_uses_images
from policy.runtime import get_env_image, image_batch_to_tensor, obs_batch_to_tensor, policy_act, policy_forward
from training.bc import build_state_teacher, dataset_to_tensors, train_bc


def rollout_policy(
    env,
    policy,
    expert: ExpertPolicy,
    n_episodes: int,
    device: str,
    seed: int = 0,
    collect_images: bool = False,
):
    """Roll out policy, label each state with the expert action."""
    new_obs, new_acts = [], []
    new_imgs = []
    policy.eval()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            obs_t = obs_batch_to_tensor(obs, device)
            img_t = None
            if collect_images:
                img = get_env_image(env)
                img_t = image_batch_to_tensor(img, device)
            with torch.no_grad():
                policy_forward(policy, obs_t, img_t)  # keep parity with multimodal path
            expert_action = expert.act(env)
            new_obs.append(obs.copy())
            new_acts.append(expert_action.copy())
            if collect_images:
                new_imgs.append(np.transpose(img, (2, 0, 1)).copy())
            # Step with policy action so distribution shift is applied
            policy_action = policy_act(policy, obs_t, img_t).squeeze(0).cpu().numpy()
            obs, _, terminated, truncated, _ = env.step(policy_action)
            done = terminated or truncated
    img_arr = np.array(new_imgs, dtype=np.uint8) if collect_images else None
    return (
        np.array(new_obs, dtype=np.float32),
        np.array(new_acts, dtype=np.float32),
        img_arr,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--bc-ckpt", default="logs/bc/policy_bc.pt")
    parser.add_argument("--bc-data", default="logs/bc")
    parser.add_argument("--out", default="logs/dagger")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DAgger] device={device}")

    env = make_env(cfg["env"])
    expert = ExpertPolicy()
    policy_cfg = cfg["policy"]
    obs_dim = int(env.observation_space.shape[0])
    cfg_obs_dim = int(policy_cfg.get("obs_dim", obs_dim))
    if cfg_obs_dim != obs_dim:
        raise ValueError(f"policy.obs_dim={cfg_obs_dim} does not match env observation dim {obs_dim}")
    collect_images = policy_uses_images(policy_cfg)
    if collect_images:
        cfg_image_shape = tuple(int(v) for v in policy_cfg.get("image_shape", []))
        env_image_shape = tuple(int(v) for v in getattr(env, "policy_image_shape", ()))
        if cfg_image_shape != env_image_shape:
            raise ValueError(
                f"policy.image_shape={cfg_image_shape} does not match env image shape {env_image_shape}"
            )
    policy = build_deterministic_policy(policy_cfg, obs_dim=obs_dim)
    policy.load(args.bc_ckpt, device=device)

    distill_teacher = None
    distill_coef = float(policy_cfg.get("distill_coef", 0.0))
    distill_ckpt = policy_cfg.get("distill_teacher_ckpt")
    if distill_coef > 0.0 and distill_ckpt:
        distill_teacher = build_state_teacher(
            obs_dim=obs_dim,
            policy_cfg=policy_cfg,
            ppo_cfg=cfg.get("ppo", {}),
            ckpt=str(distill_ckpt),
            teacher_type=str(policy_cfg.get("distill_teacher_type", "ppo")),
            device=device,
        )
        print(f"[DAgger] Distilling from state teacher with coef={distill_coef:.3f}")

    dagger_cfg = cfg["dagger"]

    # Load BC dataset as the initial aggregate dataset
    obs_agg = np.load(os.path.join(args.bc_data, "dataset_obs.npy"))
    act_agg = np.load(os.path.join(args.bc_data, "dataset_act.npy"))
    img_agg = None
    img_path = os.path.join(args.bc_data, "dataset_img.npy")
    if collect_images:
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Expected multimodal BC dataset at {img_path}, but it was not found."
            )
        img_agg = np.load(img_path)
        if img_agg.ndim == 4 and img_agg.shape[-1] == 3:
            img_agg = np.transpose(img_agg, (0, 3, 1, 2))
    print(f"[DAgger] Initial dataset size: {len(obs_agg)}")

    all_losses = []
    for rnd in range(dagger_cfg["n_rounds"]):
        print(f"\n[DAgger] Round {rnd + 1}/{dagger_cfg['n_rounds']}")

        new_obs, new_acts, new_imgs = rollout_policy(
            env, policy, expert,
            n_episodes=dagger_cfg["n_rollout_episodes"],
            device=device,
            seed=args.seed + rnd * 1000,
            collect_images=collect_images,
        )
        obs_agg = np.concatenate([obs_agg, new_obs], axis=0)
        act_agg = np.concatenate([act_agg, new_acts], axis=0)
        if collect_images:
            img_agg = np.concatenate([img_agg, new_imgs], axis=0)
        print(f"[DAgger]   Aggregated dataset size: {len(obs_agg)}")

        if collect_images:
            dataset = list(zip(obs_agg, act_agg, img_agg))
        else:
            dataset = list(zip(obs_agg, act_agg))
        obs_t, act_t, img_t = dataset_to_tensors(dataset)
        losses = train_bc(
            policy, obs_t, act_t,
            img_t=img_t,
            teacher=distill_teacher,
            distill_coef=distill_coef,
            n_epochs=dagger_cfg["n_epochs"],
            lr=dagger_cfg["lr"],
            batch_size=dagger_cfg["batch_size"],
            device=device,
        )
        all_losses.extend(losses)

        ckpt = os.path.join(args.out, f"policy_dagger_r{rnd + 1:02d}.pt")
        policy.save(ckpt)

    final_ckpt = os.path.join(args.out, "policy_dagger.pt")
    policy.save(final_ckpt)
    np.save(os.path.join(args.out, "dagger_losses.npy"), np.array(all_losses))

    # Save final aggregated dataset for PPO phase
    np.save(os.path.join(args.out, "dataset_obs.npy"), obs_agg)
    np.save(os.path.join(args.out, "dataset_act.npy"), act_agg)
    if collect_images and img_agg is not None:
        np.save(os.path.join(args.out, "dataset_img.npy"), img_agg)

    env.close()
    print(f"[DAgger] Done. Checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
