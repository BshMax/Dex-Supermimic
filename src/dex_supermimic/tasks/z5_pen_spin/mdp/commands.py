# SPDX-License-Identifier: Apache-2.0
"""Z5 clip catalog reader: 22-DoF reference_qpos and site-based link positions.

Mirrors ``pen_spin.mdp.commands.SingleHandTrackingReader`` but with the Z5
widths (NUM_J=22) and the converted deploy-bundle catalog. ``reference_qvel``
is used when present (the bundle stores measured joint velocities); otherwise
velocities are reconstructed by differencing, exactly like the Wuji reader.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from mjlab.utils.lab_api.math import subtract_frame_transforms

from dex_supermimic.tasks.pen_spin.mdp.commands import (
  _diff,
  _quat,
  world_points_to_wrist_local,
)
from dex_supermimic.tasks.pen_spin.tooling.clip_store import ResolvedClipSet, TrackingClip
from dex_supermimic.tasks.z5_pen_spin.z5_constants import NUM_J, NUM_LINKS

_ARRAY_KEYS = ("reference_qpos", "wrist_pose_w", "object_pose_w", "link_pos_w")


class Z5TrackingReader:
  """Read the ordered clips selected by one z5-pen-spin recipe."""

  def __init__(self, clip_set: ResolvedClipSet, *, device: str | torch.device = "cpu") -> None:
    if not isinstance(clip_set, ResolvedClipSet):
      raise TypeError("clip_set must be a ResolvedClipSet")
    if not clip_set.clips:
      raise ValueError("ResolvedClipSet must contain at least one clip")
    if not np.isfinite(clip_set.frame_dt) or clip_set.frame_dt <= 0:
      raise ValueError("ResolvedClipSet frame_dt must be finite and positive")
    self._clip_set = clip_set
    self._device = torch.device(device)

  def read(self) -> dict[str, object]:
    trajectories = [
      self._read_trajectory(clip, self._clip_set.frame_dt) for clip in self._clip_set.clips
    ]
    lengths = torch.tensor(
      [trajectory["joint_pos"].shape[0] for trajectory in trajectories],
      dtype=torch.long,
      device=self._device,
    )

    def concatenate(field: str) -> torch.Tensor:
      return torch.cat([trajectory[field] for trajectory in trajectories]).to(self._device)

    return {
      "frame_dt": self._clip_set.frame_dt,
      "trajectory_lengths": lengths,
      "joint_pos": concatenate("joint_pos"),
      "joint_vel": concatenate("joint_vel"),
      "wrist_pose_w": concatenate("wrist_pose_w"),
      "object_pose": concatenate("object_pose"),
      "link_pos": concatenate("link_pos"),
    }

  def _read_trajectory(self, clip: TrackingClip, frame_dt: float) -> dict:
    clip_id = clip.clip_id
    motion_path = clip.npz_path
    if not motion_path.is_file():
      raise FileNotFoundError(f"clip {clip_id!r} motion does not exist: {motion_path}")
    with np.load(motion_path, allow_pickle=False) as data:
      for key in _ARRAY_KEYS:
        if key not in data:
          raise KeyError(f"clip {clip_id!r} is missing required array {key!r}")
      joint_pos = torch.as_tensor(data["reference_qpos"], dtype=torch.float32)
      joint_vel = (
        torch.as_tensor(data["reference_qvel"], dtype=torch.float32)
        if "reference_qvel" in data
        else None
      )
      wrist_pose_w = torch.as_tensor(data["wrist_pose_w"], dtype=torch.float32)
      object_pose_w = torch.as_tensor(data["object_pose_w"], dtype=torch.float32)
      link_pos_w = torch.as_tensor(data["link_pos_w"], dtype=torch.float32)

    frames = joint_pos.shape[0] if joint_pos.ndim else 0
    frame_arrays = {
      "reference_qpos": (joint_pos, (frames, NUM_J)),
      "wrist_pose_w": (wrist_pose_w, (frames, 7)),
      "object_pose_w": (object_pose_w, (frames, 7)),
      "link_pos_w": (link_pos_w, (frames, NUM_LINKS, 3)),
    }
    for name, (value, expected) in frame_arrays.items():
      if tuple(value.shape) != expected:
        raise ValueError(
          f"{name} in clip {clip_id!r} must have shape {expected}, got {tuple(value.shape)}"
        )
      if not torch.isfinite(value).all():
        raise ValueError(f"{name} in clip {clip_id!r} contains non-finite values")

    wrist_quat = _quat(wrist_pose_w[:, 3:], f"wrist_pose_w in clip {clip_id!r}")
    object_quat_w = _quat(object_pose_w[:, 3:], f"object_pose_w in clip {clip_id!r}")
    wrist_pose_w = torch.cat((wrist_pose_w[:, :3], wrist_quat), dim=-1)
    object_pose_w = torch.cat((object_pose_w[:, :3], object_quat_w), dim=-1)
    object_pos, object_quat = subtract_frame_transforms(
      wrist_pose_w[:, :3], wrist_quat, object_pose_w[:, :3], object_quat_w
    )
    object_pose = torch.cat((object_pos, object_quat), dim=-1)
    link_pos = world_points_to_wrist_local(
      link_pos_w, wrist_pos=wrist_pose_w[:, :3], wrist_quat=wrist_quat
    )
    return {
      "joint_pos": joint_pos,
      "joint_vel": joint_vel if joint_vel is not None else _diff(joint_pos, frame_dt),
      "wrist_pose_w": wrist_pose_w,
      "object_pose": object_pose,
      "link_pos": link_pos,
    }


from dex_supermimic.tasks.pen_spin.mdp import commands as _pen_commands
from dex_supermimic.tasks.pen_spin.mdp.commands import (  # noqa: E402
  SingleHandTrackingCommand,
  SingleHandTrackingCommandCfg,
)


class Z5TrackingCommandCfg(SingleHandTrackingCommandCfg):
  def build(self, env) -> "Z5TrackingCommand":
    return Z5TrackingCommand(self, env)


class Z5TrackingCommand(SingleHandTrackingCommand):
  """SingleHandTrackingCommand with the 22-DoF/site-link Z5 reader.

  The parent __init__ resolves clips through the module-global
  ``SingleHandTrackingReader``; swap that global for the duration of
  construction so the shared reader code (recipe resolution, adaptive
  sampling, RSI) is reused untouched.
  """

  cfg: Z5TrackingCommandCfg

  def __init__(self, cfg, env):
    original = _pen_commands.SingleHandTrackingReader
    _pen_commands.SingleHandTrackingReader = Z5TrackingReader
    try:
      super().__init__(cfg, env)
    finally:
      _pen_commands.SingleHandTrackingReader = original
