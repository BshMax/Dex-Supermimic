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

import torch
from mjlab.managers.manager_base import ManagerTermBase


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
