# SPDX-License-Identifier: Apache-2.0
"""Z5 pen-spin rewards: hand-pen contact shaping and self-collision penalty.

Two contact channels, matching the checkpoint dims (contact_dim=15,
hand_pen_contact_dim=23):

- ``hand_pen_contact``: 23 binary channels, one per Z5 hand body, whether that
  body touches the pen. Used as a shaped reward (encouraging stable multi-finger
  contact) and as extra WM input/output in the fork.
- ``finger_collision``: finger self-collision penalty, Z5 body layout.

The exact private-fork slot semantics for the 15-D WM contact head cannot be
recovered from the checkpoint alone; the penalty here reproduces the *reward
magnitude* contract (finger_collision_weight = -0.5 in the checkpoint config).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from mjlab.managers.manager_base import ManagerTermBase

from dex_supermimic.tasks.z5_pen_spin.z5_constants import (
  NUM_LINKS,
  REF_CONTACT_AXIAL_SIGMA,
  REF_CONTACT_AXIAL_TOLERANCE,
  REF_CONTACT_MIN_CONFIDENCE,
  REF_CONTACT_SURFACE_SIGMA,
  REF_CONTACT_SURFACE_TOLERANCE,
  Z5_HAND_PEN_CONTACT_BODIES,
  Z5_LINK_NAMES,
  Z5_SITE_CONTACT_BODIES,
)

PEN_RADIUS_M = 0.00745
PEN_HALF_LENGTH_M = 0.125
_SURFACE_PATH = (
  Path(__file__).resolve().parents[1] / "data" / "reference_contacts" / "z5_surface.npz"
)


def load_surface_table(path: Path = _SURFACE_PATH):
  """(axial, depth, conf) reference tables plus the site->hpc channel map."""
  if not path.is_file():
    return None
  data = np.load(path, allow_pickle=False)
  channel = torch.tensor(
    [Z5_HAND_PEN_CONTACT_BODIES.index(b) for b in Z5_SITE_CONTACT_BODIES], dtype=torch.long
  )
  return dict(
    axial=torch.as_tensor(data["axial"], dtype=torch.float32),
    depth=torch.as_tensor(data["depth"], dtype=torch.float32),
    conf=torch.as_tensor(data["confidence"], dtype=torch.float32),
    channel=channel,
  )


def reference_contact_terms(
  links: torch.Tensor,
  pen_pos: torch.Tensor,
  pen_rot: torch.Tensor,
  hpc: torch.Tensor,
  table: dict,
  frame: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """(match, axial, surface) reference-contact shaping, all in [0, 1]-ish.

  ``links`` (N, 21, 3) site positions and the pen pose in the palm frame;
  ``hpc`` (N, 23) current hand-pen contact channels. Only reference sites whose
  stored confidence clears the gate contribute, and only while the
  corresponding hand body is actually in contact (the fork shaped contact
  quality, not free-air proximity).
  """
  axis = pen_rot[..., :, 0]
  v = links - pen_pos.unsqueeze(-2)
  t = (v * axis.unsqueeze(-2)).sum(-1).clamp(-PEN_HALF_LENGTH_M, PEN_HALF_LENGTH_M)
  closest = pen_pos.unsqueeze(-2) + t.unsqueeze(-1) * axis.unsqueeze(-2)
  depth = (links - closest).norm(dim=-1) - PEN_RADIUS_M

  device = links.device
  axial_ref = table["axial"].to(device)[frame]
  depth_ref = table["depth"].to(device)[frame]
  conf = table["conf"].to(device)[frame]
  channel = table["channel"].to(device)
  gate = (conf >= REF_CONTACT_MIN_CONFIDENCE) & (hpc[:, channel] > 0)
  denom = gate.float().sum(-1).clamp_min(1.0).unsqueeze(-1)

  contact = hpc[:, channel] * gate.float() / denom
  axial_err = (t - axial_ref).abs() - REF_CONTACT_AXIAL_TOLERANCE
  axial = (
    gate.float() * torch.exp(-(axial_err.clamp_min(0.0) / REF_CONTACT_AXIAL_SIGMA).square())
  ).sum(-1) / denom.squeeze(-1)
  surface_err = (depth - depth_ref).abs() - REF_CONTACT_SURFACE_TOLERANCE
  surface = (
    gate.float() * torch.exp(-(surface_err.clamp_min(0.0) / REF_CONTACT_SURFACE_SIGMA).square())
  ).sum(-1) / denom.squeeze(-1)
  return contact.sum(-1), axial, surface


class reference_contact_reward(ManagerTermBase):
  """Simulator-side reference-contact match (weight 0.1 in the checkpoint)."""

  def __init__(self, cfg, env):
    super().__init__(env)
    table = load_surface_table()
    self._table = (
      None if table is None else {k: v.to(env.device) for k, v in table.items()}
    )
    self._sensor = env.scene["hand_pen_contact"]
    from mjlab.managers import SceneEntityCfg

    from dex_supermimic.tasks.pen_spin.mdp.observations import object_pose_wrist_local
    from dex_supermimic.tasks.z5_pen_spin.mdp.observations import site_pos_wrist_local

    self._site_fn = site_pos_wrist_local
    self._obj_fn = object_pose_wrist_local
    self._link_cfg = SceneEntityCfg("robot", site_names=Z5_LINK_NAMES, preserve_order=True)
    self._robot_cfg = SceneEntityCfg("robot")
    self._object_cfg = SceneEntityCfg("object")
    # Reward terms are constructed after the managers resolve params-based
    # configs, so resolve the ones built here by hand.
    self._link_cfg.resolve(env.scene)

  def _parts(self, env):
    if self._table is None:
      return None
    links = self._site_fn(env, link_cfg=self._link_cfg, robot_cfg=self._robot_cfg)
    pose = self._obj_fn(env, self._robot_cfg, self._object_cfg)
    hpc = (self._sensor.data.found > 0).float()
    motion = env.command_manager.get_term("motion")
    frame = (motion.time_steps + motion.trajectory_starts).long().squeeze(-1)
    return links.reshape(links.shape[0], NUM_LINKS, 3), pose[:, :3], _rotmat(pose[:, 3:]), hpc, frame

  def __call__(self, env) -> torch.Tensor:
    parts = self._parts(env)
    if parts is None:
      return torch.zeros(env.num_envs, device=env.device)
    links, pos, rot, hpc, frame = parts
    match, _, _ = reference_contact_terms(links, pos, rot, hpc, self._table, frame)
    return match


class reference_contact_axial_reward(reference_contact_reward):
  """Axial (along-shaft) contact-position shaping (weight 0.05)."""

  def __call__(self, env) -> torch.Tensor:
    parts = self._parts(env)
    if parts is None:
      return torch.zeros(env.num_envs, device=env.device)
    links, pos, rot, hpc, frame = parts
    _, axial, _ = reference_contact_terms(links, pos, rot, hpc, self._table, frame)
    return axial


class reference_contact_surface_reward(reference_contact_reward):
  """Radial (surface-depth) contact-position shaping (weight 0.05)."""

  def __call__(self, env) -> torch.Tensor:
    parts = self._parts(env)
    if parts is None:
      return torch.zeros(env.num_envs, device=env.device)
    links, pos, rot, hpc, frame = parts
    _, _, surface = reference_contact_terms(links, pos, rot, hpc, self._table, frame)
    return surface


def _rotmat(quat_wxyz: torch.Tensor) -> torch.Tensor:
  from dex_supermimic.rl.dex_supermimic.rotations import quat_to_mat

  return quat_to_mat(quat_wxyz)


class hand_pen_contact_reward(ManagerTermBase):
  """Reward maintaining pen contact across the hand bodies (checkpoint weight 0.1).

  ``hand_pen_contact`` is a 23-slot ContactSensor (one per hand body). The
  reward is the mean activated channel, so broad, stable contact is preferred
  over a single fingertip pinch.
  """

  def __init__(self, cfg, env):
    super().__init__(env)
    self._sensor = env.scene["hand_pen_contact"]

  def __call__(self, env) -> torch.Tensor:
    found = (self._sensor.data.found > 0).float()
    return found.sum(dim=-1).clamp(max=3.0) / 3.0


class finger_self_collision_penalty(ManagerTermBase):
  """Z5 finger self-collision penalty (checkpoint finger_collision_weight=-0.5).

  Counts active finger-link contacts within the hand subtree (the
  ``finger_collision`` sensor), excluding palm-adjacent pairs that the MJCF
  already excludes structurally.
  """

  def __init__(self, cfg, env):
    super().__init__(env)
    self._sensor = env.scene["finger_collision"]

  def __call__(self, env) -> torch.Tensor:
    found = (self._sensor.data.found > 0).float()
    return found.sum(dim=-1)
