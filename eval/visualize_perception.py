"""Inspect what the onboard camera and detector are seeing.

Shows:
- full-resolution onboard RGB with detector overlay
- resized policy image that multimodal policies actually consume

Usage
-----
python -m eval.visualize_perception --preset multimodal_v1
python -m eval.visualize_perception --config configs/multimodal_obs_v2.yaml --type bc --ckpt logs/bc_multimodal_obs_v2/policy_bc.pt
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import yaml

from env.gate_race_aviary import make_env
from expert.expert_policy import ExpertPolicy
from policy.actor import build_actor_critic, build_deterministic_policy, policy_uses_images
from policy.runtime import get_env_image, image_batch_to_tensor, obs_batch_to_tensor, policy_act

try:
    import cv2
except ImportError as exc:  # pragma: no cover - user-facing runtime dependency
    raise ImportError("visualize_perception requires opencv-python.") from exc


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
    "multimodal_v2_bc": {
        "type": "bc",
        "config": "configs/multimodal_obs_v2.yaml",
        "ckpt": "logs/bc_multimodal_obs_v2/policy_bc.pt",
    },
    "vision_bridge_baseline": {
        "type": "ppo",
        "config": "configs/vision_bridge_eval_v1.yaml",
        "ckpt": "logs/ppo_generalization_obs_v1/policy_ppo_best.pt",
    },
}


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


def _resize_for_panel(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_NEAREST)


def _draw_detector_overlay(rgb: np.ndarray, env, info: dict) -> np.ndarray:
    panel = rgb[:, :, ::-1].copy()  # RGB -> BGR for cv2 drawing
    detections = env.get_last_camera_detections() if hasattr(env, "get_last_camera_detections") else []
    for idx, det in enumerate(detections):
        cx, cy = int(round(det.pixel_centre[0])), int(round(det.pixel_centre[1]))
        cv2.circle(panel, (cx, cy), 10, (0, 255, 255), 2)
        label = f"{idx}: d={det.distance_est:.2f} c={det.confidence:.2f}"
        cv2.putText(panel, label, (cx + 8, max(16, cy - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    lines = [
        f"track={info.get('track_name', '?')}",
        f"next_gate={info.get('next_gate', '?')} passed={info.get('gates_passed', 0)}",
        f"source={info.get('observation_source', '?')} detections={len(detections)}",
    ]
    y = 20
    for line in lines:
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        y += 20
    return panel


def _compose_panel(fullres_rgb: np.ndarray, policy_rgb: np.ndarray, env, info: dict) -> np.ndarray:
    full_panel = _draw_detector_overlay(fullres_rgb, env, info)
    target_h = max(full_panel.shape[0], 360)
    full_panel = _resize_for_panel(full_panel, int(full_panel.shape[1] * target_h / full_panel.shape[0]), target_h)

    policy_up = _resize_for_panel(policy_rgb, 256, 192)
    policy_up = policy_up[:, :, ::-1]
    cv2.putText(policy_up, "policy input", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    note = np.full((target_h - policy_up.shape[0], policy_up.shape[1], 3), 24, dtype=np.uint8)
    note_lines = [
        "Multimodal policy trains on this resized RGB image.",
        "Detector overlay is for the weaker vision_bridge path.",
        "Press q to quit.",
    ]
    y = 24
    for line in note_lines:
        cv2.putText(note, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        y += 22
    right = np.vstack([policy_up, note])
    if right.shape[0] < full_panel.shape[0]:
        pad = np.full((full_panel.shape[0] - right.shape[0], right.shape[1], 3), 24, dtype=np.uint8)
        right = np.vstack([right, pad])
    return np.hstack([full_panel, right])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multimodal_obs_v2.yaml")
    parser.add_argument("--type", default="ppo", choices=["expert", "bc", "ppo"])
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--preset", choices=sorted(_PRESETS.keys()), default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action="store_true",
                        help="Also show the PyBullet GUI.")
    parser.add_argument("--fixed-camera", action="store_true",
                        help="Disable PyBullet chase camera if GUI is enabled.")
    parser.add_argument("--no-window", action="store_true",
                        help="Do not open the OpenCV debug window; useful when only saving frames.")
    parser.add_argument("--realtime", action="store_true",
                        help="Sleep between steps to play back near control rate.")
    parser.add_argument("--save-dir", default=None,
                        help="Optional directory to save debug frames as PNGs.")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save every Nth frame when --save-dir is set.")
    args = parser.parse_args(argv)

    if args.preset is not None:
        preset = _PRESETS[args.preset]
        args.type = preset["type"]
        args.config = preset["config"]
        args.ckpt = preset["ckpt"]

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env(cfg["env"], gui=args.gui, camera_follow=not args.fixed_camera)
    actor = _load_actor(args.type, args.ckpt, cfg, env, device)
    control_dt = 1.0 / float(getattr(env, "CTRL_FREQ", 20))

    frame_idx = 0
    window_name = "Perception Debug"
    if not args.no_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            done = False
            while not done:
                action = _step_actor(actor, obs, env, device)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                fullres = env.get_last_camera_frame_fullres()
                policy_rgb = env.get_last_camera_frame()
                if fullres is None:
                    fullres = env.render_onboard_camera()
                if policy_rgb is None:
                    policy_rgb = fullres
                if fullres is None or policy_rgb is None:
                    raise RuntimeError("No onboard camera frame available for perception visualization.")

                panel = _compose_panel(fullres, policy_rgb, env, info)
                if not args.no_window:
                    cv2.imshow(window_name, panel)

                if args.save_dir is not None and frame_idx % max(1, args.save_every) == 0:
                    out_path = os.path.join(args.save_dir, f"perception_{frame_idx:06d}.png")
                    cv2.imwrite(out_path, panel)

                frame_idx += 1
                if not args.no_window:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        return
                if args.realtime:
                    time.sleep(control_dt)
    finally:
        env.close()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
