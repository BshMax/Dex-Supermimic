# SPDX-License-Identifier: Apache-2.0
"""Z5 pen-spin MDP components.

Re-exports the shared Wuji pen-spin terms (they are robot-agnostic: tracking
rewards, residual-EMA action, camera noise model, RSI reset, terminations) and
adds the Z5-specific site-based link observations and contact rewards.
"""

from dex_supermimic.tasks.pen_spin.mdp import *  # noqa: F401,F403

from .commands import (  # noqa: F401
  Z5TrackingCommand,
  Z5TrackingCommandCfg,
  Z5TrackingReader,
)
from .observations import (  # noqa: F401
  link_position_tracking_reward,
  site_pos_wrist_local,
  site_position_error,
)
from .rewards import (  # noqa: F401
  finger_self_collision_penalty,
  hand_pen_contact_reward,
  reference_contact_axial_reward,
  reference_contact_reward,
  reference_contact_surface_reward,
)
