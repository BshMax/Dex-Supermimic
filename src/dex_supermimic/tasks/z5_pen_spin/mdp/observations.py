# SPDX-License-Identifier: Apache-2.0
"""Z5 site-based link observations and rewards.

The Z5 reference command's 63-D link part stores the 21 ``*_reference_site``
positions in the palm frame (deploy bundle policy_config.json "link_names").
The Wuji task uses body origins instead, so Z5 needs site-frame variants of
``link_pos_wrist_local``, ``link_position_error`` and the tracking reward.
"""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from dex_supermimic.tasks.pen_spin.mdp.observations import (
  _reference_frame_pose,
  world_points_to_wrist_local,
)


def site_pos_wrist_local(env, link_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
  """Palm-frame positions of the selected reference sites (num_envs, n_sites*3)."""
  robot, wrist = env.scene[link_cfg.name], env.scene[robot_cfg.name]
  wrist_pos, wrist_quat = _reference_frame_pose(wrist, robot_cfg)
  value = world_points_to_wrist_local(
    robot.data.site_pos_w[:, link_cfg.site_ids],
    wrist_pos=wrist_pos,
    wrist_quat=wrist_quat,
  )
  return value.reshape(env.num_envs, -1)


def site_position_error(env, link_cfg: SceneEntityCfg, robot_cfg=SceneEntityCfg("robot"), command_name="motion"):
  current = site_pos_wrist_local(env, link_cfg, robot_cfg)
  motion = env.command_manager.get_term(command_name)
  return current - motion.link_pos.reshape(env.num_envs, -1)


def link_position_tracking_reward(
  env,
  link_cfg: SceneEntityCfg,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  command_name: str = "motion",
  sigma: float = 0.025,
) -> torch.Tensor:
  """Gaussian reward on per-site palm-frame position error (matches Wuji link_pos)."""
  current = site_pos_wrist_local(env, link_cfg, robot_cfg)
  motion = env.command_manager.get_term(command_name)
  ref = motion.link_pos.reshape(env.num_envs, -1)
  error = torch.linalg.norm(
    current.reshape(env.num_envs, -1, 3) - ref.reshape(env.num_envs, -1, 3), dim=-1
  )
  return torch.exp(-torch.square(error / sigma)).mean(dim=-1)
