# SPDX-License-Identifier: Apache-2.0
"""Z5 pen-spin SuperMimic entry points (train / play).

    uv run z5-supermimic-train [--model-checkpoint .../model_final.pt] [--num-envs 1024]
    uv run z5-supermimic-play  --checkpoint .../model_final.pt
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tyro

from .config import Z5SuperMimicCfg


def train():
  cfg = tyro.cli(Z5SuperMimicCfg)
  train_config(cfg)


def train_config(cfg, folder=None):
  if not cfg.device.startswith("cuda") or not torch.cuda.is_available():
    raise RuntimeError("Training requires an NVIDIA CUDA GPU.")
  import warp as wp
  from mjlab.envs import ManagerBasedRlEnv

  from dex_supermimic.tasks.z5_pen_spin.task import make_env_cfg
  from .runner import Z5SuperMimicRunner

  random.seed(cfg.seed)
  np.random.seed(cfg.seed)
  torch.manual_seed(cfg.seed)
  torch.backends.cuda.matmul.allow_tf32 = True
  wp.set_device(cfg.device)
  env_cfg = make_env_cfg(
    cfg.num_envs, cfg.control_hz, play=False, recipe_path=cfg.recipe_path, clips_root=cfg.clips_root,
    dynamics=cfg.dynamics, rig=cfg.rig,
  )
  env_cfg.seed = cfg.seed
  env_cfg.auto_reset = False
  env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
  log_root = Path(cfg.log_dir) / cfg.task
  folder = Path(folder) if folder else log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  runner = None
  try:
    runner = Z5SuperMimicRunner(env, cfg, folder)
    try:
      runner.learn()
    except KeyboardInterrupt:
      runner.save(folder / "model_interrupted.pt")
      print(f"Saved interrupted checkpoint to {folder}", flush=True)
  finally:
    if runner is not None:
      runner.close()
    env.close()


@dataclass
class PlayConfig:
  checkpoint: str
  recipe_path: str | None = None
  clips_root: str | None = None
  num_envs: int = 1
  device: str = "cuda:0"
  steps: int = 0
  """Playback control steps. 0 = unlimited (Viser) or the default 400 (headless)."""
  viser: bool = False
  """Interactive multi-hand web viewer (browser). Spreads hands on a grid."""
  viewer: bool = True
  """Native single-window MuJoCo viewer (needs a local display)."""
  export_onnx: str | None = None
  dynamics: str = "training"
  rig: bool = False
  """Add the hardware rig (wrist tag mount + bench floor, pen-only collision)."""
  env_spacing: float = 0.4
  """Grid spacing between hands when num_envs > 1 (visualization only)."""
  wm_horizon: int = 8
  wm_opacity: float = 0.4
  seed: int = 42
  """Seed for reproducible trajectory-frame sampling (play mode is random otherwise)."""


def play():
  cfg = tyro.cli(PlayConfig)
  if cfg.viewer and cfg.viser:
    raise ValueError("Use --no-viewer --viser for the web viewer, or --viewer for native.")
  import os

  os.environ.setdefault("MUJOCO_GL", "egl")
  random.seed(cfg.seed)
  np.random.seed(cfg.seed)
  torch.manual_seed(cfg.seed)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers import SceneEntityCfg

  from dex_supermimic.rl.z5_supermimic.networks import ActorZ5
  from dex_supermimic.tasks.pen_spin.mdp.observations import PEN_SHAFT_AXIS, object_pose_wrist_local
  from dex_supermimic.tasks.z5_pen_spin.task import make_env_cfg
  from mjlab.utils.lab_api.math import quat_apply

  state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
  if state["config"]["task"] != "z5-pen-spin":
    raise ValueError("not a z5-pen-spin checkpoint")
  actor = ActorZ5(state["modules"]["actor"]["norm.mean"].numel(), 22)
  actor.load_state_dict(state["modules"]["actor"])
  actor = actor.eval().to(cfg.device)

  # Multi-hand visualization spreads the env origins on a grid; single-hand and
  # the numeric path keep them coincident (spacing irrelevant at num_envs=1).
  spacing = cfg.env_spacing if cfg.num_envs > 1 else 0.0
  env_cfg = make_env_cfg(
    cfg.num_envs, state["config"]["control_hz"], play=True,
    recipe_path=cfg.recipe_path or state["config"].get("recipe_path"),
    clips_root=cfg.clips_root or state["config"].get("clips_root"),
    dynamics=cfg.dynamics, rig=cfg.rig, env_spacing=spacing,
  )
  env_cfg.seed = cfg.seed
  if cfg.num_envs > 1:
    # Pull the viewer camera back far enough to frame the whole grid (the
    # default 0.45 m distance only shows the center hands).
    import math

    span = math.ceil(math.sqrt(cfg.num_envs)) * cfg.env_spacing
    env_cfg.viewer.distance = max(0.45, span * 1.5)
  # Viewers (viser/native) run continuously and rely on the env self-resetting
  # terminated worlds; only the headless numeric path below resets manually (for
  # reproducible error measurement), so it alone disables auto_reset. Note the
  # format-4 checkpoint has no WM tracker, so DexViserPlayViewer does NOT do its
  # own manual resets and must therefore have auto_reset=True.
  env_cfg.auto_reset = cfg.viser or cfg.viewer

  # --- Viser: interactive web viewer, many hands at once ---
  if cfg.viser:
    from dex_supermimic.rl.dex_supermimic.viser import DexViserPlayViewer

    def make_reader(play_env):
      from dex_supermimic.rl.z5_supermimic.sim import Z5SimReader

      return Z5SimReader(play_env)

    env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
    try:
      env.reset()
      viewer = DexViserPlayViewer(
        env, policy=lambda obs: actor(obs["policy"]), checkpoint=state,
        wm_horizon=cfg.wm_horizon, wm_opacity=cfg.wm_opacity, make_reader=make_reader,
      )
      viewer.run(num_steps=cfg.steps or None)
    finally:
      env.close()
    return

  env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
  obs, _ = env.reset()
  motion = env.command_manager.get_term("motion")
  palm_cfg, obj_cfg = SceneEntityCfg("robot"), SceneEntityCfg("object")

  # --- Native single-window MuJoCo viewer ---
  if cfg.viewer:
    from mjlab.viewer import NativeMujocoViewer

    native = NativeMujocoViewer(env, policy=lambda o: actor(o["policy"]))
    native.setup()
    try:
      while native.is_running():
        native.tick()
    except KeyboardInterrupt:
      pass
    finally:
      native.close()
      env.close()
    return

  def axis_err(cur, ref):
    ax = torch.zeros(cur.shape[0], 3, device=cur.device)
    ax[:, PEN_SHAFT_AXIS] = 1.0
    cos = (quat_apply(cur[:, 3:], ax) * quat_apply(ref[:, 3:], ax)).sum(-1).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cos))

  pos_errs, ang_errs = [], []
  steps = cfg.steps or 400
  try:
    for step in range(steps):
      with torch.no_grad():
        action = actor(obs["policy"])
      obs, reward, term, trunc, info = env.step(action)
      cur = object_pose_wrist_local(env, palm_cfg, obj_cfg)
      ref = motion.object_pose
      pos_errs.append((cur[:, :3] - ref[:, :3]).norm(dim=-1).max().item())
      ang_errs.append(axis_err(cur, ref).max().item())
      if (term.any() or trunc.any()) and step + 1 < steps:
        obs, _ = env.reset(env_ids=(term | trunc).nonzero(as_tuple=False).flatten())
        motion = env.command_manager.get_term("motion")
  except KeyboardInterrupt:
    pass
  pos = np.array(pos_errs)
  ang = np.array(ang_errs)
  print(f"\n=== Z5 warm-start playback: {len(pos)} steps ===")
  print(f"max pos err {pos.max() * 100:.3f} cm | mean {pos.mean() * 100:.3f} cm")
  print(f"max axis err {ang.max():.2f} deg | mean {ang.mean():.2f} deg")
  print(f"first 100 steps: pos {pos[:100].max() * 100:.3f} cm, axis {ang[:100].max():.2f} deg")
  print("deploy reference (mujoco 100 steps): pos 1.675 cm, axis 7.14 deg")
  env.close()
