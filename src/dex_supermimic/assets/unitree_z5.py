# SPDX-License-Identifier: Apache-2.0
"""Unitree Z5 (Dex5) right-hand robot config for the z5-pen-spin task.

The MJCF is the deploy-side ``unitree_z5/z5_right_mujoco.xml`` (22 hinge
joints, 22 position actuators, 26 reference sites). Its raw defaults are NOT
the training plant: the MJCF ships kp=5/kv=0.05 and per-joint passive params,
and its ``actuatorfrcrange`` lives on the *joint*, so a plain compile leaves
``actuator_forcerange`` at [0, 0]. The training truth is baked into the deploy
bundle's ``scene.mjb`` (mujoco 3.15.0) and echoed by ``model_final.pt``'s
config section:

    actuator gainprm[0] = 2.0          (uniform kp)
    actuator biasprm[2] = -0.04        (uniform kv; checkpoint actuator_kv)
    actuator forcerange = +-1.5 | +-0.8 (per joint, == scene.mjb)
    dof armature        = 0.0001        (uniform; checkpoint joint_armature)
    dof damping         = 0.005         (uniform; checkpoint joint_damping)
    dof frictionloss    = 0.01          (uniform)
    collision geom      = condim 4, friction (0.7, 0.005, 0.0001),
                          solref (0.02, 1.5), solimp (0.9, 0.95, 0.001, 0.5, 2.0)

``dynamics`` selects the plant:

- ``"training"``  (default): the scene.mjb uniform plant above, kv=0.04. This is
  the model that produced ``model_final.pt`` and is what training/playback use.
- ``"sysid_baseline"``: identical, but kv=0.10 -- the gain the real hand was
  commanded with during system identification (``z5_right_fitted_params.json``
  recording_metadata.gains, ``sysid_schema.json`` per-finger kp/kd). Used to
  test whether the policy is sensitive to the kv=0.04-vs-0.10 discrepancy.
- ``"sysid_fitted"``: per-joint fitted params (deploy-side ``z5_apply_fitted.py``
  mapping). NOT the training plant; kept only for deploy/real-hand alignment.

The Stribeck static-friction overshoot has no MuJoCo representation and is
dropped in the fitted preset, exactly as in the deploy tool.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np

from dex_supermimic.tasks.z5_pen_spin.z5_constants import (
  PALM_BODY_POS_W,
  PALM_BODY_QUAT_WXYZ,
  PALM_BODY_QUAT_XYZW,
  Z5_JOINT_NAMES,
)

if TYPE_CHECKING:
  from mjlab.entity import EntityCfg

_RESOURCE_DIR = Path(__file__).resolve().parent / "resources" / "unitree_z5"
Z5_XML_PATH = _RESOURCE_DIR / "z5_right_mujoco.xml"
Z5_FITTED_PARAMS_PATH = _RESOURCE_DIR / "z5_right_fitted_params.json"
Z5_WRIST_MOUNT_MESH = _RESOURCE_DIR / "wrist_tag_mount.STL"

DynamicsPreset = Literal["training", "sysid_baseline", "sysid_fitted"]

# scene.mjb / model_final.pt training truth (uniform across all 22 dofs).
TRAINING_KP = 2.0
TRAINING_KV = 0.04  # checkpoint config actuator_kv; scene.mjb biasprm[2] = -0.04
SYSID_KV = 0.10  # real-hand command kd during identification (recording gains)
TRAINING_ARMATURE = 0.0001  # checkpoint joint_armature; scene.mjb dof_armature
TRAINING_DOF_DAMPING = 0.005  # checkpoint joint_damping; scene.mjb dof_damping
TRAINING_DOF_FRICTIONLOSS = 0.01  # scene.mjb dof_frictionloss
# scene.mjb hand collision geoms (condim 4, anisotropic friction, soft solref).
COLLISION_CONDIM = 4
COLLISION_FRICTION = (0.7, 0.005, 0.0001)
COLLISION_SOLREF = (0.02, 1.5)
COLLISION_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)

# --- Optional hardware rig (NOT in scene.mjb; deploy-side only) -------------
# The wrist tag mount is a PEDESTAL used as the bench stand: the counterbored
# flange face (4 countersunk holes, 52 cm2) bolts to the palm back, and the
# small plate face (4.7 cm2) rests on the bench. The pose below is that as-built
# orientation: flange face on the palm back plane (13.9 mm behind the palm
# origin, per the palm collision mesh bbox), plate face on the bench plane
# 97.409 mm below the palm origin, local z keeping the jig's horizontal
# direction. Derived from wrist_tag_mount.STL face centroids plus the fixed palm
# world pose; verified: mount world z 0.4032..0.4869, plate face coplanar with
# the bench to 5e-6 m, and the reference pen trajectory grazes both (pen lowest
# world z 0.4031 vs bench 0.4032) -- the real rig does touch the pen.
WRIST_MOUNT_POS = (-0.072845856, -0.047114208, -0.157080845)
WRIST_MOUNT_QUAT_WXYZ = (0.395386519, 0.511680391, 0.586213305, -0.488064175)
# Bench plane offset below the palm origin (palm world z 0.500659 -> bench
# world z 0.403250). Equals palm-back offset 0.0139 + flange-to-plate 0.083509.
RIG_TABLE_OFFSET_BELOW_PALM = 0.097409174
# Contype/conaffinity chosen so the rig collides ONLY with the pen (contype 1 /
# conaffinity 2), never with the hand (contype 1 / conaffinity 1) or the ground:
#   rig(2,0) vs pen(1,2):  2 & 2 = 2  -> contact
#   rig(2,0) vs hand(1,1): 2 & 1 = 0 and 0 & 1 = 0 -> no contact
# This is required because the mount is rigidly bolted to the palm; letting it
# touch the fingers would jam the hand against its own fixture.
RIG_CONTYPE = 2
RIG_CONAFFINITY = 0
RIG_FRICTION = (1.0, 0.005, 0.0001)
RIG_SOLREF = (0.02, 1.0)
RIG_CONDIM = 3


def load_z5_fitted_params(path: Path = Z5_FITTED_PARAMS_PATH) -> dict[str, dict]:
  params = json.loads(Path(path).read_text())
  if params.get("schema") != "dex_supermimic_z5_sysid_v1":
    raise ValueError(f"unsupported Z5 sysid schema: {path}")
  fitted = {entry["name"]: entry["fitted"] for entry in params["joints"]}
  if tuple(fitted) != Z5_JOINT_NAMES:
    raise ValueError(
      f"Z5 fitted-param joint order mismatch: expected {Z5_JOINT_NAMES}, got {tuple(fitted)}"
    )
  return fitted


def _apply_collision_geom_params(spec: mujoco.MjSpec) -> None:
  """Set hand collision geoms to the scene.mjb training contact params."""
  for geom in spec.geoms:
    if geom.name.endswith("_collision"):
      geom.condim = COLLISION_CONDIM
      geom.friction = COLLISION_FRICTION
      geom.solref = COLLISION_SOLREF
      geom.solimp = COLLISION_SOLIMP


def apply_z5_training_dynamics(spec: mujoco.MjSpec, *, kv: float = TRAINING_KV) -> mujoco.MjSpec:
  """Overwrite gains/passives to the uniform scene.mjb training plant.

  ``forcerange`` is copied per-joint from the MJCF's ``actuatorfrcrange``
  (a plain compile drops it onto the actuator as [0, 0]), so the torque and
  overcurrent rewards see the trained effort caps (+-1.5 proximal, +-0.8 rest).
  """
  for actuator in spec.actuators:
    name = actuator.target
    if name not in Z5_JOINT_NAMES:
      raise ValueError(f"Z5 actuator targets unknown joint {name!r}")
    actuator.gainprm[0] = TRAINING_KP
    actuator.biasprm[1] = -TRAINING_KP
    actuator.biasprm[2] = -kv
    lo, hi = spec.joint(name).actfrcrange
    actuator.forcerange = (float(lo), float(hi))
    actuator.forcelimited = True
    joint = spec.joint(name)
    joint.armature = TRAINING_ARMATURE
    joint.damping[0] = TRAINING_DOF_DAMPING
    joint.frictionloss = TRAINING_DOF_FRICTIONLOSS
  _apply_collision_geom_params(spec)
  return spec


def apply_z5_fitted_dynamics(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Overwrite actuator gains and joint passive params with sysid fits.

  Deploy-side only (matches wuji-mjlab ``z5_apply_fitted.py``): the fitted
  per-joint model reproduces the real hand's response to kp=2.0/kd=0.1 commands.
  This is NOT the plant that produced ``model_final.pt`` -- use
  ``apply_z5_training_dynamics`` for that.
  """
  fitted = load_z5_fitted_params()
  for actuator in spec.actuators:
    name = actuator.target
    if name not in fitted:
      raise ValueError(f"Z5 actuator targets unknown joint {name!r}")
    fit = fitted[name]
    actuator.gainprm[0] = fit["stiffness"]
    actuator.biasprm[1] = -fit["stiffness"]
    actuator.biasprm[2] = -fit["damping"]
    actuator.forcerange = (-fit["effort_limit"], fit["effort_limit"])
    actuator.forcelimited = True
    joint = spec.joint(name)
    joint.armature = fit["armature"]
    joint.frictionloss = fit["dynamic_friction"]
    joint.damping[0] = fit["viscous_friction"]
  _apply_collision_geom_params(spec)
  return spec


def _unroot_palm_fold(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Move the URDF root_to_palm fold out of the palm body into the mocap pose.

  The deploy MJCF bakes quat (0.5, 0.5, 0.5, 0.5) into palm_link, so the mocap
  root frame and the palm body frame differ by that rotation. The bundle
  reference (policy_config.json pose_frame="palm_link") is expressed in the
  palm BODY frame. Identity-ing the body quat and carrying the fold in the
  entity init pose makes root frame == palm frame, physically identical, and
  lets the shared Wuji tracking math (which uses the root frame) apply as-is.
  """
  palm = spec.body("palm_link")
  expected = (0.4999981634, 0.5, 0.5000018366, 0.5)
  if not all(abs(a - b) < 1e-6 for a, b in zip(palm.quat, expected)):
    raise ValueError(
      f"unexpected palm_link body quat {tuple(palm.quat)}; the root_to_palm "
      f"fold {expected} must be baked into the MJCF for this remap to hold"
    )
  palm.quat = (1.0, 0.0, 0.0, 0.0)
  return spec


def inject_z5_rig(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Attach the hardware rig: wrist tag mount + the bench plane.

  Off by default. The mount is a pedestal standing palm-back-to-bench: its
  counterbored flange face meets the palm back plane and its small plate face
  rests on the bench plane 97.409 mm below the palm origin (see WRIST_MOUNT_*
  constants). Both geoms get contype=2/conaffinity=0 so they touch ONLY the pen
  (contype 1 / conaffinity 2) -- the mount must never collide with the hand it
  is bolted to, and the bench must not fight the hand's own collision geoms.
  """
  filename = Z5_WRIST_MOUNT_MESH.name
  spec.add_mesh(name="wrist_tag_mount", content_type="model/stl", file=filename)

  palm = spec.body("palm_link")
  # group=2 so the mount is visible by default in the viewers (mjviser shows
  # geom groups 0-2 initially; 3 is the hidden collision group).
  palm.add_geom(
    name="wrist_mount_rig",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="wrist_tag_mount",
    pos=WRIST_MOUNT_POS,
    quat=WRIST_MOUNT_QUAT_WXYZ,
    contype=RIG_CONTYPE,
    conaffinity=RIG_CONAFFINITY,
    condim=RIG_CONDIM,
    friction=RIG_FRICTION,
    solref=RIG_SOLREF,
    group=2,
    rgba=(0.85, 0.8, 0.72, 1.0),
  )
  # Bench floor: a child of the palm body, but ORIENTED WORLD-HORIZONTAL by
  # pre-multiplying the inverse of the fixed palm world quaternion, and placed
  # RIG_TABLE_OFFSET_BELOW_PALM below the palm origin in world z (palm world z
  # 0.500659 -> bench world z 0.403764). Verified: world normal == [0,0,1] and
  # the mount's flange face is coplanar with it to 5e-6 m. The reference pen
  # trajectory dips to world z 0.4031, i.e. it grazes the bench -- matching the
  # real rig where the pen does touch the table.
  pq = np.asarray(PALM_BODY_QUAT_WXYZ, dtype=np.float64)
  pq_inv = (pq[0], -pq[1], -pq[2], -pq[3])
  R = _quat_to_mat(pq)
  floor_local_pos = R.T @ np.array((0.0, 0.0, -RIG_TABLE_OFFSET_BELOW_PALM))
  palm.add_geom(
    name="rig_floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    pos=tuple(float(v) for v in floor_local_pos),
    quat=pq_inv,
    size=(0.0, 0.0, 0.01),
    contype=RIG_CONTYPE,
    conaffinity=RIG_CONAFFINITY,
    condim=RIG_CONDIM,
    friction=RIG_FRICTION,
    solref=RIG_SOLREF,
    group=2,
    rgba=(0.45, 0.45, 0.5, 1.0),
  )
  return spec


def _quat_to_mat(q):
  w, x, y, z = q
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
  )


def _get_z5_spec(dynamics: DynamicsPreset = "training", rig: bool = False) -> mujoco.MjSpec:
  if not Z5_XML_PATH.is_file():
    raise FileNotFoundError(f"Z5 MJCF not found: {Z5_XML_PATH}")
  spec = mujoco.MjSpec.from_file(str(Z5_XML_PATH))
  spec.modelfiledir = str(Z5_XML_PATH.parent)
  _unroot_palm_fold(spec)
  if dynamics == "training":
    apply_z5_training_dynamics(spec, kv=TRAINING_KV)
  elif dynamics == "sysid_baseline":
    apply_z5_training_dynamics(spec, kv=SYSID_KV)
  elif dynamics == "sysid_fitted":
    apply_z5_fitted_dynamics(spec)
  else:
    raise ValueError(f"unknown Z5 dynamics preset {dynamics!r}")
  if rig:
    inject_z5_rig(spec)
  return spec


def get_unitree_z5_cfg(
  dynamics: DynamicsPreset = "training", rig: bool = False
) -> "EntityCfg":
  """Build the Z5 EntityCfg: fixed palm at the trained world pose.

  ``dynamics`` picks the plant preset; see the module docstring. The default
  ``"training"`` is the scene.mjb truth used by the checkpoint. ``rig=True``
  additionally bolts on the wrist tag mount and bench floor (hardware rig,
  collides with the pen only); leave it off to reproduce scene.mjb exactly.
  """
  from mjlab.actuator import XmlActuatorCfg
  from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=PALM_BODY_POS_W,
      rot=PALM_BODY_QUAT_XYZW,
      joint_pos={".*": 0.0},
      joint_vel={".*": 0.0},
    ),
    spec_fn=partial(_get_z5_spec, dynamics=dynamics, rig=rig),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        XmlActuatorCfg(
          target_names_expr=(".*_joint",),
        ),
      ),
      # The policy contract uses hard MJCF limits (policy_config.json).
      soft_joint_pos_limit_factor=1.0,
    ),
  )
