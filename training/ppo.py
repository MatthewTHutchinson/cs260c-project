"""Phase D — PPO fine-tune with BC auxiliary loss + clip-radius curriculum.

Fine-tunes an ActorCritic from a DAgger checkpoint using PPO.  A BC
regularisation term (MSE against the DAgger expert dataset) is added to
each update to prevent catastrophic forgetting.  The clip_radius ramps
via a configurable schedule to enable faster racing behaviour.

Usage
-----
    python -m training.ppo --config configs/default.yaml \
        --dagger-ckpt logs/dagger/policy_dagger.pt \
        --dagger-data logs/dagger \
        --out logs/ppo
"""

import argparse
import os
import shutil
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from env.gate_race_aviary import GateRaceAviary, make_env
from eval.evaluate import evaluate
from policy.actor import ActorCritic

_EVAL_STAT_KEYS = [
    "episodes",
    "completion_rate",
    "crash_rate",
    "oob_rate",
    "mean_return",
    "std_return",
    "mean_gates",
    "max_gates",
    "mean_finish_steps",
    "mean_max_speed",
    "max_max_speed",
]


# ------------------------------------------------------------------
# Rollout buffer
# ------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self, n_steps: int, obs_dim: int, action_dim: int, device: str):
        self.n      = n_steps
        self.device = device
        self.obs      = torch.zeros(n_steps, obs_dim,    device=device)
        self.actions  = torch.zeros(n_steps, action_dim, device=device)
        self.rewards  = torch.zeros(n_steps,             device=device)
        self.values   = torch.zeros(n_steps,             device=device)
        self.log_probs= torch.zeros(n_steps,             device=device)
        self.dones    = torch.zeros(n_steps,             device=device)
        self.ptr = 0

    def add(self, obs, action, reward, value, log_prob, done):
        i = self.ptr
        self.obs[i]       = obs
        self.actions[i]   = action
        self.rewards[i]   = reward
        self.values[i]    = value
        self.log_probs[i] = log_prob
        self.dones[i]     = done
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.n

    def reset(self):
        self.ptr = 0

    def compute_returns_and_advantages(
        self, last_value: torch.Tensor, gamma: float, gae_lambda: float
    ):
        advantages = torch.zeros_like(self.rewards)
        last_gae   = 0.0
        for t in reversed(range(self.n)):
            non_term = 1.0 - self.dones[t]
            next_val = last_value if t == self.n - 1 else self.values[t + 1]
            delta    = self.rewards[t] + gamma * next_val * non_term - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_term * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        return returns, advantages


# ------------------------------------------------------------------
# Clip-radius curriculum
# ------------------------------------------------------------------

def _clip_radius_at_step(schedule: list, step: int) -> float:
    """Return the clip radius from the schedule for the given global step."""
    r = schedule[0][1]
    for threshold, radius in schedule:
        if step >= threshold:
            r = radius
    return float(r)


def _validation_score(stats: dict) -> tuple[float, float, float, float]:
    finish = -float(stats["mean_finish_steps"]) if stats["mean_finish_steps"] is not None else -1.0e9
    return (
        float(stats["completion_rate"]),
        float(stats["mean_gates"]),
        finish,
        float(stats["mean_return"]),
    )


def _sanitize_suite_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or "suite"


def _aggregate_validation_stats(
    suite_rows: list[tuple[str, dict, float]],
) -> dict:
    total_weight = float(sum(weight for _, _, weight in suite_rows))
    if total_weight <= 0.0:
        total_weight = float(len(suite_rows)) or 1.0

    def weighted(key: str) -> float:
        return float(sum(float(stats[key]) * weight for _, stats, weight in suite_rows) / total_weight)

    finish_rows = [
        (float(stats["mean_finish_steps"]), weight)
        for _, stats, weight in suite_rows
        if stats["mean_finish_steps"] is not None
    ]
    finish_total = float(sum(weight for _, weight in finish_rows))
    mean_finish = None
    if finish_rows and finish_total > 0.0:
        mean_finish = float(sum(value * weight for value, weight in finish_rows) / finish_total)

    return {
        "episodes": float(sum(float(stats["episodes"]) for _, stats, _ in suite_rows)),
        "completion_rate": weighted("completion_rate"),
        "crash_rate": weighted("crash_rate"),
        "oob_rate": weighted("oob_rate"),
        "mean_return": weighted("mean_return"),
        "std_return": weighted("std_return"),
        "mean_gates": weighted("mean_gates"),
        "max_gates": float(max(float(stats["max_gates"]) for _, stats, _ in suite_rows)),
        "mean_finish_steps": mean_finish,
        "mean_max_speed": weighted("mean_max_speed"),
        "max_max_speed": float(max(float(stats["max_max_speed"]) for _, stats, _ in suite_rows)),
    }


# ------------------------------------------------------------------
# PPO trainer
# ------------------------------------------------------------------

class PPOTrainer:
    def __init__(
        self,
        env: GateRaceAviary,
        policy: ActorCritic,
        cfg: dict,
        bc_obs: torch.Tensor,
        bc_acts: torch.Tensor,
        device: str,
        val_envs: list[tuple[str, GateRaceAviary, float]] | None = None,
    ):
        self.env    = env
        self.policy = policy.to(device)
        self.device = device
        self.val_envs = val_envs or []

        ppo = cfg["ppo"]
        self.n_steps       = int(ppo["n_steps"])
        self.n_epochs      = int(ppo["n_epochs"])
        self.batch_size    = int(ppo["batch_size"])
        self.gamma         = float(ppo["gamma"])
        self.gae_lambda    = float(ppo["gae_lambda"])
        self.clip_range    = float(ppo["clip_range"])
        self.ent_coef      = float(ppo["ent_coef"])
        self.vf_coef       = float(ppo["vf_coef"])
        self.bc_coef       = float(ppo["bc_coef"])
        self.max_grad_norm = float(ppo["max_grad_norm"])
        self.total_steps   = int(ppo["total_timesteps"])
        self.clip_schedule = [(int(s), float(r)) for s, r in ppo.get("clip_schedule", [[0, 1.0]])]
        self.target_kl     = float(ppo.get("target_kl", 0.03))
        self.min_updates_before_clip_relax = int(ppo.get("min_updates_before_clip_relax", 5))
        self.validation_interval = int(ppo.get("validation_interval_updates", 5))
        self.validation_episodes = int(ppo.get("validation_episodes", 8))
        self.validation_seed = int(ppo.get("validation_seed", 1234))
        self.checkpoint_interval = int(ppo.get("checkpoint_interval_updates", 5))

        obs_dim    = cfg["policy"]["obs_dim"]
        action_dim = cfg["policy"]["action_dim"]
        self.buffer = RolloutBuffer(self.n_steps, obs_dim, action_dim, device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=float(ppo["lr"]))

        self.bc_obs  = bc_obs.to(device)
        self.bc_acts = bc_acts.to(device)

        self._obs, _ = env.reset()
        self._global_step = 0
        self._update_idx = 0
        self._best_val_score = None
        self._best_val_stats = None
        self._val_metric_placeholders = self._build_val_metric_placeholders()

    def _build_val_metric_placeholders(self) -> dict[str, float]:
        if not self.val_envs:
            return {}
        placeholders: dict[str, float] = {}
        if len(self.val_envs) == 1:
            for key in _EVAL_STAT_KEYS:
                placeholders[f"val_{key}"] = np.nan
            return placeholders

        for name, _, _ in self.val_envs:
            for key in _EVAL_STAT_KEYS:
                placeholders[f"val_{name}_{key}"] = np.nan
        for key in _EVAL_STAT_KEYS:
            placeholders[f"val_mix_{key}"] = np.nan
        return placeholders

    def collect_rollout(self) -> tuple[torch.Tensor, dict]:
        self.policy.eval()
        self.buffer.reset()
        rollout_return = 0.0
        rollout_len = 0
        rollout_gates = 0
        rollout_crashes = 0
        rollout_oobs = 0
        completed_eps = 0
        episode_returns = []
        episode_lengths = []
        episode_gates = []

        while not self.buffer.full():
            obs_t = torch.from_numpy(self._obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(obs_t)
            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            self.buffer.add(
                obs_t.squeeze(0),
                action.squeeze(0),
                torch.tensor(reward,       device=self.device),
                value.squeeze(0),
                log_prob.squeeze(0),
                torch.tensor(float(done),  device=self.device),
            )
            self._global_step += 1
            rollout_return += float(reward)
            rollout_len += 1
            rollout_gates = info.get("gates_passed", rollout_gates)

            if done:
                completed_eps += 1
                episode_returns.append(rollout_return)
                episode_lengths.append(rollout_len)
                episode_gates.append(rollout_gates)
                if info.get("collision", False):
                    rollout_crashes += 1
                elif terminated:
                    rollout_oobs += 1
                self._obs = self.env.reset()[0]
                rollout_return = 0.0
                rollout_len = 0
                rollout_gates = 0
            else:
                self._obs = next_obs

        with torch.no_grad():
            obs_t = torch.from_numpy(self._obs).unsqueeze(0).to(self.device)
            _, _, last_value = self.policy.get_action(obs_t)

        stats = {
            "episodes": float(completed_eps),
            "rollout_mean_return": float(np.mean(episode_returns)) if episode_returns else rollout_return,
            "rollout_mean_len": float(np.mean(episode_lengths)) if episode_lengths else rollout_len,
            "rollout_mean_gates": float(np.mean(episode_gates)) if episode_gates else rollout_gates,
            "rollout_crash_rate": (rollout_crashes / completed_eps) if completed_eps else 0.0,
            "rollout_oob_rate": (rollout_oobs / completed_eps) if completed_eps else 0.0,
        }
        return last_value.squeeze(0), stats

    def update(self, last_value: torch.Tensor) -> dict:
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.policy.train()
        metrics = {"pg": [], "vf": [], "ent": [], "bc": [], "total": [], "approx_kl": []}
        stop_early = False

        for _ in range(self.n_epochs):
            idx = torch.randperm(self.n_steps, device=self.device)
            for start in range(0, self.n_steps, self.batch_size):
                bi     = idx[start : start + self.batch_size]
                obs_b  = self.buffer.obs[bi]
                act_b  = self.buffer.actions[bi]
                ret_b  = returns[bi]
                adv_b  = advantages[bi]
                old_lp = self.buffer.log_probs[bi]

                log_prob, entropy, value = self.policy.evaluate_actions(obs_b, act_b)
                ratio = torch.exp(log_prob - old_lp)
                approx_kl = (old_lp - log_prob).mean()

                pg1     = -adv_b * ratio
                pg2     = -adv_b * ratio.clamp(1 - self.clip_range, 1 + self.clip_range)
                pg_loss = torch.max(pg1, pg2).mean()
                vf_loss = F.mse_loss(value, ret_b)
                ent_loss= -entropy.mean()

                # BC auxiliary loss: keep policy close to DAgger reference dataset.
                bc_idx  = torch.randint(len(self.bc_obs), (len(bi),), device=self.device)
                bc_feats= self.policy.trunk(self.bc_obs[bc_idx])
                bc_pred = self.policy._squash(self.policy.actor_mean(bc_feats))
                bc_loss = F.mse_loss(bc_pred, self.bc_acts[bc_idx])

                loss = (pg_loss
                        + self.vf_coef  * vf_loss
                        + self.ent_coef * ent_loss
                        + self.bc_coef  * bc_loss)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metrics["pg"].append(pg_loss.item())
                metrics["vf"].append(vf_loss.item())
                metrics["ent"].append(ent_loss.item())
                metrics["bc"].append(bc_loss.item())
                metrics["total"].append(loss.item())
                metrics["approx_kl"].append(approx_kl.item())

                if approx_kl.item() > self.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def validate(self) -> tuple[dict, dict | None, list[tuple[str, dict]]]:
        if not self.val_envs:
            return {}, None, []
        self.policy.eval()
        suite_rows: list[tuple[str, dict, float]] = []
        for idx, (name, env, weight) in enumerate(self.val_envs):
            stats = evaluate(
                env,
                self.policy,
                n_episodes=self.validation_episodes,
                device=self.device,
                seed=self.validation_seed + idx * 1000,
                print_stats=False,
            )
            suite_rows.append((name, stats, weight))

        if len(suite_rows) == 1:
            _, stats, _ = suite_rows[0]
            metrics = {
                f"val_{k}": float(v) if v is not None else np.nan
                for k, v in stats.items()
            }
            return metrics, stats, [(suite_rows[0][0], stats)]

        metrics = {}
        per_suite = []
        for name, stats, _ in suite_rows:
            per_suite.append((name, stats))
            metrics.update({
                f"val_{name}_{k}": float(v) if v is not None else np.nan
                for k, v in stats.items()
            })
        agg = _aggregate_validation_stats(suite_rows)
        metrics.update({
            f"val_mix_{k}": float(v) if v is not None else np.nan
            for k, v in agg.items()
        })
        return metrics, agg, per_suite

    def train(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        steps      = 0
        update_idx = 0
        all_metrics: list[dict] = []
        best_ckpt = os.path.join(out_dir, "policy_ppo_best.pt")

        while steps < self.total_steps:
            # --- Curriculum: update clip_radius ---
            new_clip = _clip_radius_at_step(self.clip_schedule, steps)
            if new_clip != self.env.clip_radius and self._update_idx >= self.min_updates_before_clip_relax:
                self.env.set_clip_radius(new_clip)
                print(f"  [curriculum] clip_radius → {new_clip:.2f}  (step {steps})")

            last_value, rollout_stats = self.collect_rollout()
            metrics    = self.update(last_value)
            steps      += self.n_steps
            update_idx += 1
            self._update_idx = update_idx
            merged_metrics = {**metrics, **rollout_stats, "clip_radius": float(self.env.clip_radius)}
            merged_metrics.update(self._val_metric_placeholders)

            val_score_stats = None
            per_suite_stats: list[tuple[str, dict]] = []
            improved = False
            if self.validation_interval > 0 and update_idx % self.validation_interval == 0:
                val_metrics, val_score_stats, per_suite_stats = self.validate()
                if val_score_stats is not None:
                    merged_metrics.update(val_metrics)
                    score = _validation_score(val_score_stats)
                    if self._best_val_score is None or score > self._best_val_score:
                        self._best_val_score = score
                        self._best_val_stats = val_score_stats
                        improved = True
                        self.policy.save(best_ckpt)

            merged_metrics["best_ckpt_improved"] = 1.0 if improved else 0.0
            all_metrics.append(merged_metrics)

            if self.checkpoint_interval > 0 and update_idx % self.checkpoint_interval == 0:
                print(
                    f"  PPO step {steps:7d}/{self.total_steps}"
                    f"  clip={self.env.clip_radius:.2f}"
                    f"  ret={rollout_stats['rollout_mean_return']:.2f}"
                    f"  gates={rollout_stats['rollout_mean_gates']:.2f}"
                    f"  crash={rollout_stats['rollout_crash_rate']:.2f}"
                    f"  pg={metrics['pg']:.4f}"
                    f"  vf={metrics['vf']:.4f}"
                    f"  bc={metrics['bc']:.4f}"
                    f"  kl={metrics['approx_kl']:.4f}"
                )
                if val_score_stats is not None:
                    finish = val_score_stats["mean_finish_steps"]
                    finish_str = "n/a" if finish is None else f"{finish:.1f}"
                    label = "val" if len(self.val_envs) == 1 else "val_mix"
                    print(
                        f"    [{label}] completion={val_score_stats['completion_rate']:.2f}"
                        f" gates={val_score_stats['mean_gates']:.2f}"
                        f" return={val_score_stats['mean_return']:.2f}"
                        f" finish={finish_str}"
                        f" improved={improved}"
                    )
                    if len(per_suite_stats) > 1:
                        parts = []
                        for name, stats in per_suite_stats:
                            suite_finish = stats["mean_finish_steps"]
                            suite_finish_str = "n/a" if suite_finish is None else f"{suite_finish:.1f}"
                            parts.append(
                                f"{name}:ret={stats['mean_return']:.2f}/finish={suite_finish_str}"
                            )
                        details = " ".join(parts)
                        print(f"    [val-suites] {details}")
                ckpt = os.path.join(out_dir, f"policy_ppo_{steps:07d}.pt")
                self.policy.save(ckpt)

        final = os.path.join(out_dir, "policy_ppo.pt")
        self.policy.save(final)
        if self._best_val_stats is None:
            shutil.copyfile(final, best_ckpt)
        metric_keys = list(all_metrics[0].keys()) if all_metrics else []
        metric_matrix = np.array(
            [[float(m[k]) for k in metric_keys] for m in all_metrics],
            dtype=np.float32,
        ) if all_metrics else np.zeros((0, 0), dtype=np.float32)
        np.save(os.path.join(out_dir, "ppo_metrics.npy"), metric_matrix)
        np.save(os.path.join(out_dir, "ppo_metric_keys.npy"), np.array(metric_keys))
        return final


def _build_validation_envs(cfg: dict) -> list[tuple[str, GateRaceAviary, float]]:
    suite_cfgs = cfg.get("validation_suites")
    if suite_cfgs is None:
        val_env_cfg = deepcopy(cfg.get("validation_env", cfg["env"]))
        return [("default", make_env(val_env_cfg), 1.0)]

    envs: list[tuple[str, GateRaceAviary, float]] = []
    for idx, suite in enumerate(suite_cfgs):
        name = _sanitize_suite_name(str(suite.get("name", f"suite_{idx + 1}")))
        weight = float(suite.get("weight", 1.0))
        env_cfg = deepcopy(suite.get("env", cfg.get("validation_env", cfg["env"])))
        envs.append((name, make_env(env_cfg), weight))
    return envs


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="configs/default.yaml")
    parser.add_argument("--dagger-ckpt",  default="logs/dagger/policy_dagger.pt")
    parser.add_argument("--dagger-data",  default="logs/dagger")
    parser.add_argument("--out",          default="logs/ppo")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[PPO] device={device}")

    # Start at the first curriculum clip radius
    env_cfg = dict(cfg["env"])
    env_cfg["clip_radius_start"] = cfg["ppo"]["clip_schedule"][0][1]
    env = make_env(env_cfg)

    val_envs = _build_validation_envs(cfg)

    policy_cfg = cfg["policy"]
    policy = ActorCritic(
        obs_dim=policy_cfg["obs_dim"],
        action_dim=policy_cfg["action_dim"],
        hidden=policy_cfg["hidden"],
        init_log_std=float(cfg["ppo"].get("init_log_std", -2.0)),
    )

    # Warm-start from DAgger checkpoint: remap DronePolicy keys → ActorCritic keys
    # net.0.* → trunk.0.*, net.2.* → trunk.2.*, net.4.* → actor_mean.*
    dagger_state = torch.load(args.dagger_ckpt, map_location=device)
    key_map = {}
    for k in dagger_state:
        if k.startswith("net.0."):
            key_map[k] = k.replace("net.0.", "trunk.0.")
        elif k.startswith("net.2."):
            key_map[k] = k.replace("net.2.", "trunk.2.")
        elif k.startswith("net.4."):
            key_map[k] = k.replace("net.4.", "actor_mean.")
    remapped = {key_map[k]: v for k, v in dagger_state.items() if k in key_map}
    ac_state = policy.state_dict()
    compatible = {k: v for k, v in remapped.items()
                  if k in ac_state and ac_state[k].shape == v.shape}
    ac_state.update(compatible)
    policy.load_state_dict(ac_state)
    print(f"[PPO] Warm-start: transferred {len(compatible)}/{len(dagger_state)} tensors from DAgger")

    bc_obs  = torch.from_numpy(np.load(os.path.join(args.dagger_data, "dataset_obs.npy")))
    bc_acts = torch.from_numpy(np.load(os.path.join(args.dagger_data, "dataset_act.npy")))

    trainer    = PPOTrainer(env, policy, cfg, bc_obs, bc_acts, device, val_envs=val_envs)
    final_ckpt = trainer.train(args.out)
    env.close()
    for _, val_env, _ in val_envs:
        val_env.close()
    print(f"[PPO] Done. Checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
