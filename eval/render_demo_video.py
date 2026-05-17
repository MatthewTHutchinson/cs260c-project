"""Render a presentation-friendly simulation video to MP4.

This script runs a policy or expert in the PyBullet racing environment and
captures an offscreen third-person chase view. It is intended for class
presentations and portfolio demos where a live GUI recording is less reliable
than a deterministic exported clip.

Example
-------
python -m eval.render_demo_video --preset state_champion --output demo.mp4
"""

from __future__ import annotations

import argparse
import copy
import subprocess
from pathlib import Path

import numpy as np
import pybullet as p
import torch
import yaml

from env.gate_race_aviary import make_env
from expert.expert_policy import ExpertPolicy
from policy.actor import build_actor_critic, build_deterministic_policy, policy_uses_images
from policy.runtime import get_env_image, image_batch_to_tensor, obs_batch_to_tensor, policy_act

try:
    import cv2
except ImportError as exc:  # pragma: no cover - user-facing runtime dependency
    raise ImportError("render_demo_video requires opencv-python.") from exc


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
    "expert_obs_v1": {
        "type": "expert",
        "config": "configs/generalization_obs_v1.yaml",
        "ckpt": None,
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


def _start_ffmpeg(output_path: Path, width: int, height: int, fps: int) -> subprocess.Popen:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _draw_overlay(frame_rgb: np.ndarray, title: str, info: dict, episode_idx: int, step_idx: int) -> np.ndarray:
    panel = frame_rgb[:, :, ::-1].copy()  # RGB -> BGR for cv2 overlay
    lines = [
        title,
        f"track={info.get('track_name', '?')}  episode={episode_idx + 1}  step={step_idx}",
        f"next_gate={info.get('next_gate', '?')}  passed={info.get('gates_passed', 0)}",
    ]
    y = 30
    for line in lines:
        cv2.putText(panel, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
        y += 32
    return panel[:, :, ::-1]


def _camera_frame(
    env,
    width: int,
    height: int,
    distance: float,
    pitch_deg: float,
    yaw_bias_deg: float,
    lookahead: float,
    height_offset: float,
    fov_deg: float,
    shadow: bool,
    smooth: float,
    state_cache: dict,
) -> np.ndarray:
    st = env.get_full_state()
    pos = np.asarray(st["pos"], dtype=np.float64)
    quat = np.asarray(st["quat"], dtype=np.float64)
    rot = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    forward = rot @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    heading = forward.copy()
    heading[2] = 0.0
    norm = np.linalg.norm(heading)
    if norm < 1e-6:
        yaw = float(st["rpy"][2])
        heading = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)
    else:
        heading /= norm

    yaw_bias = np.deg2rad(yaw_bias_deg)
    rot2 = np.array(
        [
            [np.cos(yaw_bias), -np.sin(yaw_bias)],
            [np.sin(yaw_bias), np.cos(yaw_bias)],
        ],
        dtype=np.float64,
    )
    heading_xy = rot2 @ heading[:2]
    heading = np.array([heading_xy[0], heading_xy[1], 0.0], dtype=np.float64)
    heading /= max(np.linalg.norm(heading), 1e-9)

    side = np.array([-heading[1], heading[0], 0.0], dtype=np.float64)
    camera_target = pos + heading * lookahead + np.array([0.0, 0.0, 0.35], dtype=np.float64)
    base_eye = pos - heading * distance + side * 0.35 + np.array([0.0, 0.0, height_offset], dtype=np.float64)

    if "eye" not in state_cache:
        state_cache["eye"] = base_eye
        state_cache["target"] = camera_target
    else:
        alpha = float(np.clip(smooth, 0.0, 0.98))
        state_cache["eye"] = alpha * state_cache["eye"] + (1.0 - alpha) * base_eye
        state_cache["target"] = alpha * state_cache["target"] + (1.0 - alpha) * camera_target

    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    pitch = np.deg2rad(pitch_deg)
    elevated_target = state_cache["target"] + np.array([0.0, 0.0, np.tan(-pitch) * 0.35], dtype=np.float64)

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=state_cache["eye"].tolist(),
        cameraTargetPosition=elevated_target.tolist(),
        cameraUpVector=up.tolist(),
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=fov_deg,
        aspect=float(width) / float(height),
        nearVal=0.05,
        farVal=45.0,
    )
    render_kwargs = dict(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=env.CLIENT,
    )
    light_kwargs = {}
    if shadow:
        theme = getattr(env, "_visual_theme", {})
        light_kwargs = dict(
            shadow=1,
            lightDirection=np.asarray(theme.get("light_direction", [1.0, -0.3, -1.0]), dtype=np.float64).tolist(),
            lightColor=np.asarray(theme.get("light_color", [1.0, 1.0, 1.0]), dtype=np.float64).tolist(),
            lightAmbientCoeff=float(theme.get("ambient_coeff", 0.55)),
            lightDiffuseCoeff=float(theme.get("diffuse_coeff", 0.7)),
            lightSpecularCoeff=float(theme.get("specular_coeff", 0.15)),
        )

    try:
        _, _, rgba, _, _ = p.getCameraImage(**render_kwargs, **light_kwargs)
    except TypeError:
        _, _, rgba, _, _ = p.getCameraImage(**render_kwargs)
    rgba_np = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)
    return rgba_np[:, :, :3]


def _prepare_env_cfg(cfg: dict, args) -> dict:
    env_cfg = copy.deepcopy(cfg["env"])
    env_cfg["scene_visuals"] = bool(args.force_visuals)
    env_cfg["scene_randomization"] = bool(args.randomize_visuals)
    env_cfg["scene_clutter_count"] = int(args.clutter_count)
    if args.track_name is not None:
        env_cfg["track_name"] = args.track_name
        env_cfg.pop("track_names", None)
        env_cfg["sample_track_on_reset"] = False
    return env_cfg


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generalization_obs_v1.yaml")
    parser.add_argument("--type", default="ppo", choices=["expert", "bc", "ppo"])
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--preset", choices=sorted(_PRESETS.keys()), default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="CS260C Drone Racing Demo")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--distance", type=float, default=4.2)
    parser.add_argument("--pitch", type=float, default=-18.0)
    parser.add_argument("--yaw-bias", type=float, default=-18.0)
    parser.add_argument("--lookahead", type=float, default=1.2)
    parser.add_argument("--height-offset", type=float, default=1.7)
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--smooth", type=float, default=0.82)
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--force-visuals", action="store_true",
                        help="Enable richer scene visuals even for state-only configs.")
    parser.add_argument("--randomize-visuals", action="store_true",
                        help="Enable visual randomization for a more varied scene.")
    parser.add_argument("--clutter-count", type=int, default=10)
    parser.add_argument("--track-name", default=None,
                        help="Force a specific named track instead of config sampling.")
    args = parser.parse_args(argv)

    if args.preset is not None:
        preset = _PRESETS[args.preset]
        args.type = preset["type"]
        args.config = preset["config"]
        args.ckpt = preset["ckpt"]

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg = _prepare_env_cfg(cfg, args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env(env_cfg, gui=False, camera_follow=False)
    actor = _load_actor(args.type, args.ckpt, cfg, env, device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _start_ffmpeg(output_path, width=args.width, height=args.height, fps=args.fps)
    control_dt = 1.0 / float(getattr(env, "CTRL_FREQ", 20))
    step_stride = max(1, int(round((1.0 / control_dt) / max(1, args.fps))))

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            done = False
            step_idx = 0
            cam_cache: dict[str, np.ndarray] = {}
            while not done:
                if step_idx % step_stride == 0:
                    frame = _camera_frame(
                        env=env,
                        width=args.width,
                        height=args.height,
                        distance=args.distance,
                        pitch_deg=args.pitch,
                        yaw_bias_deg=args.yaw_bias,
                        lookahead=args.lookahead,
                        height_offset=args.height_offset,
                        fov_deg=args.fov,
                        shadow=args.shadow,
                        smooth=args.smooth,
                        state_cache=cam_cache,
                    )
                    frame = _draw_overlay(frame, args.title, info, ep, step_idx)
                    assert ffmpeg.stdin is not None
                    ffmpeg.stdin.write(frame.tobytes())

                action = _step_actor(actor, obs, env, device)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step_idx += 1
    finally:
        env.close()
        if ffmpeg.stdin is not None:
            ffmpeg.stdin.close()
        ffmpeg.wait()

    print(f"[render_demo_video] Wrote {output_path}")


if __name__ == "__main__":
    main()
