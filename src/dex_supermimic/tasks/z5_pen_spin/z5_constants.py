# SPDX-License-Identifier: Apache-2.0
"""Unitree Z5 (Dex5) constants reverse-engineered from the z5-pen-spin bundle.

Sources (all cross-checked against model_final.pt and policy_config.json,
schema ``dex_supermimic_z5_pen_spin:1``):

- ``third_party/z5-pen-spin-deploy/generated/deploy/z5-pen-spin/policy_config.json``
- ``external assets/model_final.pt`` config section
- ``external assets/z5_right_fitted_params.json`` (sha256 matches the checkpoint)
"""

from __future__ import annotations

NUM_J = 22
NUM_LINKS = 21
OBS_DIM = 163
ACTION_DIM = 22

# Policy joint order == MJCF order == policy_config.json order (thumb first).
Z5_JOINT_NAMES = (
  "right_thumb_pitch_joint",
  "right_thumb_roll_joint",
  "right_thumb_yaw1_joint",
  "right_thumb_yaw2_joint",
  "right_thumb_yaw3_joint",
  "right_index_finger_roll1_joint",
  "right_index_finger_yaw_joint",
  "right_index_finger_roll2_joint",
  "right_index_finger_roll3_joint",
  "right_middle_finger_roll1_joint",
  "right_middle_finger_yaw_joint",
  "right_middle_finger_roll2_joint",
  "right_middle_finger_roll3_joint",
  "right_ring_finger_roll1_joint",
  "right_ring_finger_yaw_joint",
  "right_ring_finger_roll2_joint",
  "right_ring_finger_roll3_joint",
  "right_little_finger_pitch_joint",
  "right_little_finger_roll1_joint",
  "right_little_finger_yaw_joint",
  "right_little_finger_roll2_joint",
  "right_little_finger_roll3_joint",
)

# The 21 reference sites whose palm-frame positions form the 63-D part of the
# motion command (policy_config.json "link_names"; all exist in the Z5 MJCF).
Z5_LINK_NAMES = (
  "palm_reference_site",
  "right_thumb_roll_reference_site",
  "right_thumb_yaw2_reference_site",
  "right_thumb_yaw3_reference_site",
  "right_thumb_tip_reference_site",
  "right_index_finger_roll1_reference_site",
  "right_index_finger_roll2_reference_site",
  "right_index_finger_roll3_reference_site",
  "right_index_finger_tip_reference_site",
  "right_middle_finger_roll1_reference_site",
  "right_middle_finger_roll2_reference_site",
  "right_middle_finger_roll3_reference_site",
  "right_middle_finger_tip_reference_site",
  "right_ring_finger_roll1_reference_site",
  "right_ring_finger_roll2_reference_site",
  "right_ring_finger_roll3_reference_site",
  "right_ring_finger_tip_reference_site",
  "right_little_finger_roll1_reference_site",
  "right_little_finger_roll2_reference_site",
  "right_little_finger_roll3_reference_site",
  "right_little_finger_tip_reference_site",
)

# The 23 hand bodies of the checkpoint's hand_pen_contact term
# (model_final.pt "hand_pen_contact_names").
Z5_HAND_PEN_CONTACT_BODIES = (
  "palm_link",
  "right_thumb_pitch_link",
  "right_thumb_roll_link",
  "right_thumb_yaw1_Link",
  "right_thumb_yaw2_link",
  "right_thumb_yaw3_link",
  "right_index_finger_roll1_link",
  "right_index_finger_yaw_link",
  "right_index_finger_roll2_link",
  "right_index_finger_roll3_link",
  "right_middle_finger_roll1_link",
  "right_middle_finger_yaw_link",
  "right_middle_finger_roll2_link",
  "right_middle_finger_roll3_link",
  "right_ring_finger_roll1_link",
  "right_ring_finger_yaw_link",
  "right_ring_finger_roll2_link",
  "right_ring_finger_roll3_link",
  "right_little_finger_pitch_link",
  "right_little_finger_roll1_link",
  "right_little_finger_yaw_link",
  "right_little_finger_roll2_link",
  "right_little_finger_roll3_link",
)

# Per-body collision GEOM names, in the same order as Z5_HAND_PEN_CONTACT_BODIES.
# The hand_pen_contact sensor matches these geoms explicitly (not the bodies) so
# that, with --rig enabled, a pen<->wrist_mount contact is NOT attributed to the
# palm_link channel (the mount geom hangs off body palm_link; body-mode matching
# lit the palm slot 91% of mount-contact frames -- see /tmp/z5_hpc_contamination).
# Note geom names are lowercased (right_thumb_yaw1_link_collision) even though
# the BODY is right_thumb_yaw1_Link.
Z5_HAND_PEN_CONTACT_GEOMS = (
  "palm_link_collision",
  "right_thumb_pitch_link_collision",
  "right_thumb_roll_link_collision",
  "right_thumb_yaw1_link_collision",
  "right_thumb_yaw2_link_collision",
  "right_thumb_yaw3_link_collision",
  "right_index_finger_roll1_link_collision",
  "right_index_finger_yaw_link_collision",
  "right_index_finger_roll2_link_collision",
  "right_index_finger_roll3_link_collision",
  "right_middle_finger_roll1_link_collision",
  "right_middle_finger_yaw_link_collision",
  "right_middle_finger_roll2_link_collision",
  "right_middle_finger_roll3_link_collision",
  "right_ring_finger_roll1_link_collision",
  "right_ring_finger_yaw_link_collision",
  "right_ring_finger_roll2_link_collision",
  "right_ring_finger_roll3_link_collision",
  "right_little_finger_pitch_link_collision",
  "right_little_finger_roll1_link_collision",
  "right_little_finger_yaw_link_collision",
  "right_little_finger_roll2_link_collision",
  "right_little_finger_roll3_link_collision",
)

# The 15 finger self-collision geoms of the checkpoint's contact head
# (contact_dim=15): the 3 most-distal links per finger (5 fingers x 3).
# This is a documented best-effort mapping of the private fork's "geometric"
# contact sensor onto the Z5 chain (see Z5_PORT_SPEC.md): the exact slot
# semantics are not recoverable from the checkpoint, but the width (15) and
# the reward magnitude contract are preserved.
Z5_FINGER_COLLISION_GEOMS = (
  "right_thumb_yaw1_link_collision",
  "right_thumb_yaw2_link_collision",
  "right_thumb_yaw3_link_collision",
  "right_index_finger_yaw_link_collision",
  "right_index_finger_roll2_link_collision",
  "right_index_finger_roll3_link_collision",
  "right_middle_finger_yaw_link_collision",
  "right_middle_finger_roll2_link_collision",
  "right_middle_finger_roll3_link_collision",
  "right_ring_finger_yaw_link_collision",
  "right_ring_finger_roll2_link_collision",
  "right_ring_finger_roll3_link_collision",
  "right_little_finger_yaw_link_collision",
  "right_little_finger_roll2_link_collision",
  "right_little_finger_roll3_link_collision",
)

# Distal (fingertip) links: targets of the fingertip_random_force disturbance
# event, the Z5 analogue of the Wuji hand2 right_finger._link4 pattern.
Z5_DISTAL_LINK_NAMES = (
  "right_thumb_yaw3_link",
  "right_index_finger_roll3_link",
  "right_middle_finger_roll3_link",
  "right_ring_finger_roll3_link",
  "right_little_finger_roll3_link",
)

# policy_config.json joint_limits, matching the MJCF ranges exactly.
Z5_JOINT_LIMITS = (
  (0.0, 1.57), (-1.57, 0.13), (0.0, 2.61), (-1.62, 0.07), (-1.62, 0.07),
  (-0.05, 1.71), (-0.35, 0.35), (-1.62, 0.07), (-1.62, 0.07),
  (-0.05, 1.71), (-0.35, 0.35), (-1.62, 0.07), (-1.62, 0.07),
  (-0.05, 1.71), (-0.35, 0.35), (-1.62, 0.07), (-1.62, 0.07),
  (0.0, 0.43), (-0.05, 1.71), (-0.35, 0.35), (-1.62, 0.07), (-1.62, 0.07),
)

# Fixed palm world pose. policy_config.json fixed_palm_pose_world_xyzw and the
# scene.mjb mocap_base body_quat are the SAME rotation -- the palm_link world
# orientation of the trained model:
#   scene.mjb: mocap_base quat = palm_link world quat = (0.67154,-0.012451,
#              -0.135393,0.728387) wxyz, palm_link body quat = identity.
# assets/unitree_z5.py identity-folds the MJCF palm body quat (the URDF
# root_to_palm fold) and carries the WHOLE rotation in the entity init pose, so
# init rot must equal the scene.mjb palm world quat directly -- NOT root(x)fold.
# (An earlier revision wrongly composed the fold in again, tilting the hand
#  94.79 deg and putting gravity/table 80 deg off; verified against scene.mjb.)
FIXED_PALM_POSE_WORLD_XYZ = (-0.021896, -0.001084, 0.500659)
# xyzw of the root/palm pose (policy_config.json fixed_palm_pose_world_xyzw).
FIXED_PALM_POSE_WORLD_XYZW = (-0.012451, -0.135393, 0.728387, 0.67154)
# Palm BODY world orientation == scene.mjb palm world quat (wxyz).
PALM_BODY_QUAT_WXYZ = (0.67154, -0.012451, -0.135393, 0.728387)
PALM_BODY_QUAT_XYZW = (-0.012451, -0.135393, 0.728387, 0.67154)
PALM_BODY_POS_W = FIXED_PALM_POSE_WORLD_XYZ
# wxyz 7-tuple used by the tracking command and the reference converter.
FIXED_WRIST_POSE_W = (-0.021896, -0.001084, 0.500659) + PALM_BODY_QUAT_WXYZ

# Controller contract (policy_config.json + deploy lib/core.py).
CONTROL_DT = 0.02
ACTION_SCALE = 0.5
RESIDUAL_CLIP = (-0.6, 0.6)
EMA_ALPHA = 0.3
AXIS_POINT_RADIUS_M = 0.10
OBJECT_POSITION_TANH_SCALE_M = 0.15
JOINT_VELOCITY_SCALE = 0.25
NOMINAL_KP = 2.0
NOMINAL_KD = 0.04

# Batched-world contact/constraint buffer sizes (same as the Wuji pen-spin
# recipe; scene.mjb itself leaves them at -1/auto, which the warp batching
# cannot use). Pure buffer capacity, not physics.
Z5_NCONMAX = 192
Z5_NJMAX = 4096

# Observation slices (deploy README "与训练保持一致的部分").
OBS_TERMS = {"motion": 91, "joint_pos": 22, "joint_vel": 22, "object_pose": 6, "last_action": 22}
MOTION_SPLIT = {"joint_pos": 22, "axis_points": 6, "link_pos": 63}

# Clip catalog (converted from the deploy bundle reference.npz).
CLIP_GROUP_ID = "z5_pen_spin"
FRAME_DT = 0.02

# Deploy validation reference (deployment_checks.json, mujoco_100_steps).
DEPLOY_MUJOCO_100STEPS = {"max_position_error_m": 0.016748, "max_axis_error_deg": 7.1416}
