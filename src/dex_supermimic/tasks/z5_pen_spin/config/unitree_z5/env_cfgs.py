# SPDX-License-Identifier: Apache-2.0
"""Z5 pen-spin environment configuration (22-DoF Unitree Dex5 hand).

Observation/action/reward layout matches the checkpoint contract
(policy_config.json, schema dex_supermimic_z5_pen_spin:1):

    policy obs 163 = motion 91 (ref_q 22 + bounded axis points 6 + 21 site
                     positions 63) + joint_pos 22 + joint_vel 22 + object_pose
                     6 + last_action 22
    critic obs 311 = clean policy 163 + link_pos 63 + joint_pos_error 22 +
                     link_pos_error 63
    action 22      = reference + EMA residual (scale 0.5, alpha 0.3,
                     residual clip +-0.6)
    50 Hz          = physics 0.005 x decimation 4
"""

from __future__ import annotations

import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr, time_out
from mjlab.managers import (
  CurriculumTermCfg,
  EventTermCfg,
  MetricsTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  SceneEntityCfg,
  TerminationTermCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig

from dex_supermimic.assets.pen import get_pen_cfg
from dex_supermimic.assets.unitree_z5 import get_unitree_z5_cfg
from dex_supermimic.tasks.z5_pen_spin import mdp
from dex_supermimic.tasks.z5_pen_spin.z5_constants import (
  AXIS_POINT_RADIUS_M,
  REF_CONTACT_AXIAL_WEIGHT,
  REF_CONTACT_SURFACE_WEIGHT,
  REF_CONTACT_WEIGHT,
  Z5_DISTAL_LINK_NAMES,
  Z5_FINGER_COLLISION_GEOMS,
  Z5_HAND_PEN_CONTACT_BODIES,
  Z5_HAND_PEN_CONTACT_GEOMS,
  Z5_NCONMAX,
  Z5_NJMAX,
  EMA_ALPHA,
  FIXED_WRIST_POSE_W,
  OBJECT_POSITION_TANH_SCALE_M,
  RESIDUAL_CLIP,
  Z5_JOINT_NAMES,
  Z5_LINK_NAMES,
)

PEN_MASS_KG = 0.030
HARDWARE_PEN_LENGTH_M = 0.250
HARDWARE_PEN_DIAMETER_M = 0.0149

_PALM_LINK_NAME = "palm_link"


def _finger_joint_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=Z5_JOINT_NAMES, preserve_order=True)


def _palm_body_cfg() -> SceneEntityCfg:
  """Tracking-frame selector: body_names=None means the entity ROOT frame.

  The Z5 MJCF bakes the URDF root_to_palm rotation (0.5,0.5,0.5,0.5) into the
  palm_link body quat, so the palm_link *body* frame is rotated relative to the
  mocap root frame. The bundle reference.npz (and FIXED_WRIST_POSE_W, and the
  deploy palm_link contract) are expressed in the root frame -- verified
  numerically: reference pen/site positions equal root-frame values, and only
  differ from palm-body-frame values by the (x,y,z)->(z,x,y) fold rotation.
  _reference_frame_pose() returns root_link_pose_w when body_names is None.
  """
  return SceneEntityCfg("robot")


def _object_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("object")


def _pen_asset_cfg() -> SceneEntityCfg:
  return SceneEntityCfg(
    "object",
    geom_names=("cube", "A_bottom", "A_top"),
    body_names=("cube",),
    preserve_order=True,
  )


def _z5_link_cfg() -> SceneEntityCfg:
  # Site-based link points: the 21 reference sites carry the 63-D command part.
  return SceneEntityCfg("robot", site_names=Z5_LINK_NAMES, preserve_order=True)


def make_z5_pen_spin_env_cfg(
  *, play: bool = False, dynamics: str = "training", rig: bool = False
) -> ManagerBasedRlEnvCfg:
  """Build the full Z5 pen-spin env config (play=True for inference setups).

  ``dynamics`` selects the robot plant preset (see ``assets/unitree_z5.py``):
  ``"training"`` is the scene.mjb truth (uniform kp 2.0 / kv 0.04) that produced
  the checkpoint; ``"sysid_baseline"`` swaps in the real-hand command gain
  kv=0.10 for comparison; ``"sysid_fitted"`` uses the per-joint identification
  fit (deploy-side model, not the training plant).

  ``rig=True`` bolts on the hardware fixture (wrist tag mount + bench floor,
  sized from the STEP assembly / z5_jig_preview.xml). Those geoms collide with
  the pen only, never with the hand or the terrain plane. Leave it False to
  reproduce scene.mjb exactly -- the checkpoint was trained without the rig.
  """

  policy_terms = {
    "motion": ObservationTermCfg(mdp.generated_commands, params={"command_name": "motion"}),
    # biased=True: the hardware encoder offset convention, matching Wuji.
    "joint_pos": ObservationTermCfg(
      mdp.joint_pos_rel,
      params={"asset_cfg": _finger_joint_cfg(), "biased": True},
      noise=UniformNoiseCfg(n_min=-0.015, n_max=0.015),
    ),
    "joint_vel": ObservationTermCfg(
      mdp.joint_vel_rel,
      params={"asset_cfg": _finger_joint_cfg()},
      scale=0.25,
      noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
    ),
    "object_pose": ObservationTermCfg(
      mdp.ObjectPoseCameraModel,
      params={
        "command_name": "motion",
        "robot_cfg": _palm_body_cfg(),
        "object_cfg": _object_cfg(),
        "bias_position_m": (-0.008, 0.008),
        "bias_orientation_rad": (-0.05, 0.05),
        "noise_position_m": (-0.002, 0.002),
        "noise_orientation_rad": (-0.015, 0.015),
        "lag_ticks": (1, 1),
        "dropout_enter_prob": 0.03,
        "dropout_continue_prob": 0.78,
        "dropout_max_ticks": 40,
      },
    ),
    "last_action": ObservationTermCfg(
      mdp.last_action, params={"action_name": "joint_pos"}
    ),
  }
  critic_terms = {
    "motion": ObservationTermCfg(
      mdp.generated_commands, params={"command_name": "motion"}
    ),
    "joint_pos": ObservationTermCfg(
      mdp.joint_pos_rel, params={"asset_cfg": _finger_joint_cfg()}
    ),
    "joint_vel": ObservationTermCfg(
      mdp.joint_vel_rel, params={"asset_cfg": _finger_joint_cfg()}, scale=0.25
    ),
    "object_pose": ObservationTermCfg(
      mdp.object_axis_points_wrist_local_bounded,
      params={"command_name": "motion", "robot_cfg": _palm_body_cfg()},
    ),
    "last_action": ObservationTermCfg(
      mdp.last_action, params={"action_name": "joint_pos"}
    ),
    "link_pos": ObservationTermCfg(
      mdp.site_pos_wrist_local,
      params={"link_cfg": _z5_link_cfg(), "robot_cfg": _palm_body_cfg()},
    ),
    "joint_position_error": ObservationTermCfg(mdp.joint_position_error, params={"joint_cfg": _finger_joint_cfg()}),
    "link_position_error": ObservationTermCfg(
      mdp.site_position_error,
      params={"link_cfg": _z5_link_cfg(), "robot_cfg": _palm_body_cfg()},
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={
        "robot": get_unitree_z5_cfg(dynamics, rig=rig),
        "object": get_pen_cfg(PEN_MASS_KG),
      },
      num_envs=4096,
      env_spacing=0.0,
    ),
    observations={
      "policy": ObservationGroupCfg(
        policy_terms, concatenate_terms=True, enable_corruption=True
      ),
      "critic": ObservationGroupCfg(
        critic_terms, concatenate_terms=True, enable_corruption=False
      ),
    },
    actions={
      "joint_pos": mdp.SingleHandResidualEMAActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=0.5,
        residual_clip=RESIDUAL_CLIP,
        alpha=EMA_ALPHA,
      )
    },
    commands={
      "motion": mdp.Z5TrackingCommandCfg(
        recipe_path=str(Path(__file__).resolve().parents[2] / "data/recipe/z5_all.json"),
        clips_root=str(Path(__file__).resolve().parents[2] / "data/rawdata/clips"),
        axis_point_radius_m=AXIS_POINT_RADIUS_M,
        object_position_tanh_scale_m=OBJECT_POSITION_TANH_SCALE_M,
        joint_names=Z5_JOINT_NAMES,
        link_names=Z5_LINK_NAMES,
        fixed_wrist_pose_w=FIXED_WRIST_POSE_W,
        debug_vis=False,
      )
    },
    events={
      "reset_to_reference_state": EventTermCfg(
        func=mdp.reset_to_reference_state, mode="reset", params={"command_name": "motion"}
      ),
    },
    curriculum={},
    rewards={},
    metrics={},
    terminations={},
    viewer=ViewerConfig(
      lookat=(0.0, 0.0, 0.5),
      distance=0.45,
      elevation=-25.0,
      azimuth=-40.0,
      origin_type=ViewerConfig.OriginType.WORLD,
    ),
    sim=SimulationCfg(
      # Batched contact buffers (scene.mjb leaves them auto/-1, unusable by the
      # warp batching); same values as the Wuji pen-spin recipe.
      nconmax=Z5_NCONMAX,
      njmax=Z5_NJMAX,
      # scene.mjb truth: 0.005 s implicitfast, 100/50 solver iters, impratio 2.0.
      mujoco=MujocoCfg(
        timestep=0.005,
        integrator="implicitfast",
        iterations=100,
        ls_iterations=50,
        impratio=2.0,
      ),
    ),
    decimation=4,
    episode_length_s=60.0,
  )

  _configure_z5_domain_randomization(cfg, play=play)
  _configure_z5_rewards_sensors(cfg, play=play)
  _configure_z5_terminations(cfg, play=play)
  if play:
    _configure_play(cfg)
  return cfg


def _configure_play(cfg: ManagerBasedRlEnvCfg) -> None:
  motion = cfg.commands["motion"]
  motion.trajectory_sampling_mode = "uniform"
  motion.first_frame_hold_prob = 1.0
  motion.first_frame_hold_s = (1.0, 1.0)
  cfg.scene.num_envs = 1
  policy = cfg.observations["policy"]
  policy.terms["joint_pos"].noise = None
  policy.terms["joint_vel"].noise = None
  object_pose = policy.terms["object_pose"]
  object_pose.func = mdp.object_axis_points_wrist_local_bounded
  object_pose.params = {
    "command_name": object_pose.params["command_name"],
    "robot_cfg": object_pose.params["robot_cfg"],
    "object_cfg": object_pose.params["object_cfg"],
  }
  for group in cfg.observations.values():
    group.enable_corruption = False
  cfg.terminations["object_position_error"].params["threshold"] = 0.20
  cfg.terminations.pop("object_orientation_error")


def _configure_z5_domain_randomization(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
  """Startup DR mirroring the Wuji pen-spin recipe, adapted to Z5 geoms.

  The checkpoint's xi names confirm the DR surface: friction_ratio,
  object_mass_ratio, kp_ratio, kd_ratio, joint_damping_ratio,
  object_size_ratio, object_com_local.
  """
  reset = cfg.events["reset_to_reference_state"]
  if play:
    cfg.events = {"reset_to_reference_state": reset}
    cfg.events["pen_dimensions"] = EventTermCfg(
      func=mdp.randomize_pen_dimensions,
      mode="startup",
      params={
        "diameter_range_m": (HARDWARE_PEN_DIAMETER_M, HARDWARE_PEN_DIAMETER_M),
        "total_length_range_m": (HARDWARE_PEN_LENGTH_M, HARDWARE_PEN_LENGTH_M),
        "asset_cfg": _pen_asset_cfg(),
      },
    )
    return

  hand_geoms = SceneEntityCfg("robot", geom_names=(r".*_collision$",))
  robot_joints = SceneEntityCfg("robot", joint_names=".*")
  cfg.events = {
    "reset_to_reference_state": reset,
    "object_com_axial": EventTermCfg(
      func=dr.body_com_offset,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("object", body_names=("cube",)),
        "operation": "add",
        "ranges": (-0.04, 0.04),
        "axes": (0,),
      },
    ),
    "object_com_radial": EventTermCfg(
      func=dr.body_com_offset,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("object", body_names=("cube",)),
        "operation": "add",
        "ranges": (-0.0005, 0.0005),
        "axes": (1, 2),
      },
    ),
    "robot_friction": EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": hand_geoms,
        "operation": "abs",
        "distribution": "log_uniform",
        "ranges": (0.35, 1.4),
        "axes": [0],
      },
    ),
    "robot_torsional_friction": EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": hand_geoms,
        "operation": "scale",
        "distribution": "log_uniform",
        "ranges": (0.5, 2.0),
        "axes": [1],
      },
    ),
    "contact_params": EventTermCfg(
      func=mdp.randomize_contact_params,
      mode="startup",
      params={
        "robot_cfg": hand_geoms,
        "solref_timeconst_range": (1.0, 2.0),
        "solref_dampratio_range": (0.8, 1.2),
        "solimp_width_range": (1.0, 2.0),
        "solimp_dmin_range": (0.5, 1.0),
      },
    ),
    "object_mass": EventTermCfg(
      func=dr.pseudo_inertia,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("object", body_names=("cube",)),
        "alpha_range": (0.5 * math.log(0.4), 0.5 * math.log(1.6)),
      },
    ),
    "pd_gains": EventTermCfg(
      func=dr.pd_gains,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": (0.9, 1.1),
        "kd_range": (0.8, 1.1),
        "distribution": "log_uniform",
        "operation": "scale",
      },
    ),
    "encoder_bias": EventTermCfg(
      func=dr.encoder_bias,
      mode="startup",
      params={
        "asset_cfg": robot_joints,
        "bias_range": (-0.01, 0.01),
      },
    ),
    "robot_dof_armature": EventTermCfg(
      func=dr.dof_armature,
      mode="startup",
      params={
        "asset_cfg": robot_joints,
        "operation": "scale",
        "ranges": (0.5, 2.0),
        "distribution": "log_uniform",
      },
    ),
    "robot_joint_damping": EventTermCfg(
      func=dr.joint_damping,
      mode="startup",
      params={
        "asset_cfg": robot_joints,
        "operation": "scale",
        "ranges": (0.5, 2.0),
        "distribution": "log_uniform",
      },
    ),
    "robot_effort_limits": EventTermCfg(
      func=dr.effort_limits,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "effort_limit_range": (0.7, 1.05),
        "operation": "scale",
      },
    ),
    "pen_dimensions": EventTermCfg(
      func=mdp.randomize_pen_dimensions,
      mode="startup",
      params={
        "diameter_range_m": (0.014, 0.016),
        "total_length_range_m": (0.205, 0.265),
        "asset_cfg": _pen_asset_cfg(),
      },
    ),
    # Step-mode disturbances. The checkpoint's wm.norm has NONZERO variance on
    # the 150-dim wrench slice (max sd 0.052), so the fork trained with external
    # wrenches flowing through body_external_wrench -> disturbance(auxiliary).
    # Without these the wrench input is identically zero and the loaded WM sees
    # an out-of-distribution channel. Mirrors the Wuji pen-spin recipe.
    "object_disturbance": EventTermCfg(
      func=mdp.apply_object_disturbance_wrench,
      mode="step",
      params={
        "asset_cfg": SceneEntityCfg("object", body_names=("cube",)),
        "grav_mult_range": (1.0, 2.0),
        "angular_accel": 375.0,
        "active_s_range": (0.2, 0.4),
        "silence_s_range": (0.5, 1.2),
        "decay_to_frac": 0.02,
        "warmup_time_s": 1.0,
        "curriculum_term": "disturbance_success_rate",
      },
    ),
    "fingertip_random_force": EventTermCfg(
      func=mdp.apply_link_random_force,
      mode="step",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=Z5_DISTAL_LINK_NAMES),
        "grav_mult_range": (0.0, 4.0),
        "duration_s": (0.1, 0.3),
        "cooldown_s": (0.5, 1.5),
        "curriculum_term": "disturbance_success_rate",
      },
    ),
  }


def _configure_z5_rewards_sensors(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
  cfg.rewards = {
    "object_pos": RewardTermCfg(
      func=mdp.object_position_tracking_reward,
      weight=2.0,
      params={"robot_cfg": _palm_body_cfg(), "object_cfg": _object_cfg(), "sigma": 0.05},
    ),
    "object_rot": RewardTermCfg(
      func=mdp.object_axis_point_tracking_reward,
      weight=8.0,
      params={"robot_cfg": _palm_body_cfg(), "object_cfg": _object_cfg(), "sigma": 0.04},
    ),
    "link_pos": RewardTermCfg(
      func=mdp.link_position_tracking_reward,
      weight=1.0,
      params={
        "link_cfg": SceneEntityCfg("robot", site_names=Z5_LINK_NAMES, preserve_order=True),
        "robot_cfg": _palm_body_cfg(),
      },
    ),
    "joint_pos": RewardTermCfg(
      func=mdp.joint_position_tracking_reward,
      weight=1.0,
      params={"joint_cfg": _finger_joint_cfg()},
    ),
    # Checkpoint: action_rate_weight=-1.0 (vs Wuji -0.1), action_alpha=0.3.
    "action_rate": RewardTermCfg(func=mdp.action_rate_combined, weight=-1.0),
    "torque": RewardTermCfg(
      func=mdp.normalized_joint_torques_l2,
      weight=-1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",), preserve_order=True)
      },
    ),
    "overcurrent": RewardTermCfg(
      func=mdp.sustained_overcurrent,
      weight=-4.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",), preserve_order=True),
        "charge_s": 1.0,
        "leak_s": 2.0,
        "threshold": 0.8,
      },
    ),
  }

  # 15-slot finger self-collision sensor (checkpoint contact_dim=15).
  # Layout mirrors the Wuji convention (per finger: two middle links, then the
  # distal link), mapped onto Z5 chains: thumb yaw1/yaw2 + yaw3; each finger
  # yaw/roll2 + roll3. Documented best-effort analogue of the private fork's
  # "geometric" sensor; see Z5_PORT_SPEC.md.
  cfg.scene.sensors = tuple(cfg.scene.sensors) + (
    ContactSensorCfg(
      name="finger_collision",
      primary=ContactMatch(
        mode="geom", pattern=Z5_FINGER_COLLISION_GEOMS, entity="robot"
      ),
      secondary=ContactMatch(mode="subtree", pattern=_PALM_LINK_NAME, entity="robot"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    ),
    # 23-slot hand-pen contact (checkpoint hand_pen_contact_dim=23), one per
    # hand body in the checkpoint's hand_pen_contact_names order. Matched by the
    # per-body COLLISION GEOM (not the body) so that, with --rig, a pen touching
    # the wrist mount -- which hangs off body palm_link -- does NOT falsely light
    # the palm channel. In the no-rig training scene each body has exactly one
    # collidable geom, so geom-mode == body-mode there (bit-identical hpc).
    ContactSensorCfg(
      name="hand_pen_contact",
      primary=ContactMatch(
        mode="geom", pattern=Z5_HAND_PEN_CONTACT_GEOMS, entity="robot"
      ),
      secondary=ContactMatch(mode="body", pattern="cube", entity="object"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    ),
  )
  cfg.rewards["finger_collision"] = RewardTermCfg(
    func=mdp.finger_self_collision_penalty, weight=-0.5
  )
  # Checkpoint: hand_pen_contact_weight=0.1 (shaped stable-contact reward).
  cfg.rewards["hand_pen_contact"] = RewardTermCfg(
    func=mdp.hand_pen_contact_reward, weight=0.1
  )
  # Checkpoint: reference_contact 0.1 / axial 0.05 / surface 0.05, shaped off
  # the regenerated z5_surface.npz. The terms no-op (return 0) if the table is
  # missing, so play builds without the data stay valid.
  cfg.rewards["reference_contact"] = RewardTermCfg(
    func=mdp.reference_contact_reward, weight=REF_CONTACT_WEIGHT
  )
  cfg.rewards["reference_contact_axial"] = RewardTermCfg(
    func=mdp.reference_contact_axial_reward, weight=REF_CONTACT_AXIAL_WEIGHT
  )
  cfg.rewards["reference_contact_surface"] = RewardTermCfg(
    func=mdp.reference_contact_surface_reward, weight=REF_CONTACT_SURFACE_WEIGHT
  )

  if play:
    return

  cfg.metrics = {
    "object_pos_tracking_error_cm": MetricsTermCfg(
      mdp.object_pos_tracking_error_cm, params={"robot_cfg": _palm_body_cfg()}
    ),
    "object_ori_tracking_error_deg": MetricsTermCfg(
      mdp.object_ori_tracking_error_deg, params={"robot_cfg": _palm_body_cfg()}
    ),
    "action_rate_sq": MetricsTermCfg(mdp.action_rate_l2),
    "action_acc_sq": MetricsTermCfg(mdp.action_acc_l2),
    "joint_vel_rms_rad_s": MetricsTermCfg(
      mdp.joint_vel_rms, params={"joint_cfg": _finger_joint_cfg()}
    ),
    "joint_acc_rms_rad_s2": MetricsTermCfg(
      mdp.joint_acc_rms, params={"joint_cfg": _finger_joint_cfg()}
    ),
    "actuator_saturation_share": MetricsTermCfg(
      mdp.actuator_saturation_share,
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",), preserve_order=True)
      },
    ),
  }

  cfg.curriculum = {
    "disturbance_success_rate": CurriculumTermCfg(
      func=mdp.linear_step_curriculum,
      params={"total_env_steps": 240_000, "warmup_env_steps": 40_000},
    )
  }
  cfg.events["reset_to_reference_state"].params.update(
    object_reset_axial_slide_range=(-0.02, 0.02),
    object_reset_roll_range=(-math.radians(20.0), math.radians(20.0)),
    curriculum_term="disturbance_success_rate",
  )
  motion = cfg.commands["motion"]
  motion.trajectory_sampling_mode = "adaptive"
  motion.adaptive_sampling_uniform_mix = 0.2
  motion.adaptive_sampling_bin_seconds = 0.5
  motion.first_frame_hold_prob = 0.15
  motion.first_frame_hold_s = (0.5, 3.0)


def _configure_z5_terminations(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "unstable": TerminationTermCfg(
      func=mdp.numerical_instability, params={"object_cfg": _object_cfg()}
    ),
    "object_position_error": TerminationTermCfg(
      func=mdp.object_position_error_exceeded,
      params={"robot_cfg": _palm_body_cfg(), "object_cfg": _object_cfg(), "threshold": 0.05},
      time_out=False,
    ),
    "object_orientation_error": TerminationTermCfg(
      func=mdp.object_shaft_direction_error_exceeded,
      params={
        "robot_cfg": _palm_body_cfg(),
        "object_cfg": _object_cfg(),
        "threshold_deg": 60.0,
      },
      time_out=False,
    ),
    "reference_end": TerminationTermCfg(func=mdp.reference_end, time_out=True),
  }
