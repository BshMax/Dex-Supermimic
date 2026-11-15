# SPDX-License-Identifier: Apache-2.0
"""Standalone Unitree Z5 pen-spin recipe for Dex SuperMimic."""

from pathlib import Path

from .config.unitree_z5.env_cfgs import make_z5_pen_spin_env_cfg


def make_env_cfg(
  num_envs=1024,
  control_hz=50,
  play=False,
  recipe_path=None,
  clips_root=None,
  dynamics="training",
  rig=False,
  env_spacing=0.0,
):
  if control_hz != 50:
    raise ValueError("Z5 pen spin requires 50 Hz reference playback")
  cfg = make_z5_pen_spin_env_cfg(play=play, dynamics=dynamics, rig=rig)
  cfg.scene.num_envs = num_envs
  cfg.scene.env_spacing = env_spacing
  motion = cfg.commands["motion"]
  if recipe_path:
    motion.recipe_path = str(Path(recipe_path).expanduser().resolve())
  if clips_root:
    motion.clips_root = str(Path(clips_root).expanduser().resolve())
  return cfg
