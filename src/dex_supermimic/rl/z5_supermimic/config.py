# SPDX-License-Identifier: Apache-2.0
"""Z5 pen-spin SuperMimic training configuration.

Defaults are the reverse-engineered checkpoint values (model_final.pt config
section, run z5-sysid-pen-spin-geometric-e8192-utd1-20000-20261113-000259).
A 24 GB GPU cannot hold 8192 envs with the (2048,2048,2048) WM at bf16 plus
imagination graphs; scale num_envs down (the checkpoint recipe ran on bigger
hardware). Everything else matches the original run.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(kw_only=True)
class Z5SuperMimicCfg:
  task: str = "z5-pen-spin"
  recipe_path: str | None = None
  clips_root: str | None = None
  seed: int = 42
  logger: str = "tensorboard"
  wandb_project: str = "dex-supermimic"
  wandb_entity: str | None = None
  wandb_mode: str | None = None
  device: str = "cuda:0"
  num_envs: int = 2048
  control_hz: int = 50
  max_iterations: int = 20000
  buffer_steps: int = 1000
  warmup_steps: int = 64
  utd: float = 1.0
  wm_only_iters: int = 500
  wm_window: int = 8
  wm_failure_frac: float = 0.25
  wm_failure_horizon: int = 16
  final_state_weight: float = 1.0
  contact_weight: float = 0.1
  model_checkpoint: str | None = None
  reset_model_statistics: bool = False
  variant: str = "fixed"
  wm_batches: int = 10
  wm_batch: int = 2048
  wm_hidden: tuple[int, ...] = (2048, 2048, 2048)
  pi_batch: int = 1024
  horizon: int = 32
  max_horizon: int = 32
  horizon_ramp_iters: int = 1000
  critic_epochs: int = 5
  critic_minibatches: int = 4
  gamma: float = 0.99
  lam: float = 0.95
  lr: float = 2e-4
  lr_decay_iters: int = 250_000
  lr_final_ratio: float = 0.03
  action_noise: float = 0.1
  failure_warmup_iters: int = 300
  failure_head: bool = True
  failure_clip: float = 0.1
  amp: bool = True
  compile: bool = True
  compile_policy: bool = True
  compile_policy_dynamic: bool = False
  profile: bool = False
  save_interval: int = 2000
  log_interval: int = 50
  episode_window: int = 1000
  holdout_fraction: float = 0.1
  log_dir: str = "logs/z5_supermimic"
  checkpoint: str | None = None

  # Z5 / fork-specific contract fields (checkpoint config keys).
  # dynamics picks the robot plant: "training" = scene.mjb truth (kp 2.0 /
  # kv 0.04, uniform), "sysid_baseline" = kv 0.10 (the real-hand command gain
  # used during identification), "sysid_fitted" = per-joint sysid fit (deploy
  # model, not the training plant). The checkpoint config records actuator_kv
  # 0.04, so "training" is the default.
  dynamics: str = "training"
  rig: bool = False
  hand_pen_contact: bool = True
  hand_pen_contact_weight: float = 0.1
  contact_sensor: str = "geometric"
  pen_context_layout: str = "reference-com"
  action_rate_weight: float = -1.0
  finger_collision_weight: float = -0.5
  action_alpha: float = 0.3
  actuator_kv: float = 0.04
  joint_armature: float = 0.0001
  joint_damping: float = 0.005

  def __post_init__(self):
    if self.task != "z5-pen-spin":
      raise ValueError(f"Z5 runner only supports task z5-pen-spin, got {self.task!r}")
    if self.control_hz != 50:
      raise ValueError("Z5 pen spin requires 50 Hz reference playback")
    if self.logger not in ("wandb", "tensorboard", "none"):
      raise ValueError("logger must be wandb, tensorboard, or none")
    if any(v < 0 for v in (self.final_state_weight, self.contact_weight)):
      raise ValueError("Loss weights must be nonnegative")
    if self.variant != "fixed":
      raise ValueError("Z5 runner only implements the fixed variant")
    if self.dynamics not in ("training", "sysid_baseline", "sysid_fitted"):
      raise ValueError(
        "dynamics must be training, sysid_baseline, or sysid_fitted"
      )
    if self.checkpoint and self.model_checkpoint:
      raise ValueError("Choose resume or model-only initialization")
    if not 0 <= self.wm_failure_frac < 1:
      raise ValueError("Invalid failure sampling mixture")
    if not math.isfinite(self.utd) or not 0 < self.utd <= 1:
      raise ValueError("utd must be in (0, 1]")
    if self.lr <= 0 or not 0 < self.lr_final_ratio <= 1:
      raise ValueError("Invalid learning-rate schedule")
    if not (0 <= self.holdout_fraction < 1):
      raise ValueError("holdout_fraction must be in [0, 1)")
    if self.num_envs < 2 or self.buffer_steps < self.wm_window or self.warmup_steps < max(
      self.horizon, self.wm_window
    ):
      raise ValueError("Need >=2 envs, buffer >= wm_window and warmup >= max(horizon, wm_window)")
    if self.max_horizon < self.horizon:
      raise ValueError("max_horizon must be >= horizon")
    if self.critic_minibatches > self.horizon * self.pi_batch:
      raise ValueError("critic_minibatches exceeds the imagined transition count")
    if self.failure_clip <= 0 or self.failure_clip > 1:
      raise ValueError("failure_clip must be in (0, 1]")

  def to_dict(self):
    return asdict(self)
