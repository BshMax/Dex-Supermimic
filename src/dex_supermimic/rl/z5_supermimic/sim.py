# SPDX-License-Identifier: Apache-2.0
"""Z5 state/context readers: 22-DoF hand, site links, geometric-style contacts.

Extends the Wuji PenSimReader contract with the Z5 layouts reverse-engineered
from model_final.pt:

- xi (53) = 6 DR ratios + object_com_local(3) + actuator kp(22) + kd(22)
  (checkpoint context_names, order preserved)
- contact (15) = finger self-collision slots (Z5_FINGER_COLLISION_GEOMS order)
- hand_pen_contact (23) = per-hand-body pen contact (checkpoint
  hand_pen_contact_names order)
- auxiliary = palm pos(3) + palm rot(9) + zeros(9) + wrench(24 robot bodies
  + 1 pen body) * 6 = 171; dynamics takes auxiliary[:, 21:] (150 wrench)
"""

import torch

from dex_supermimic.rl.dex_supermimic.kinematics import HandKinematics
from dex_supermimic.rl.dex_supermimic.pen import PenObservations, PenSimReader
from dex_supermimic.rl.dex_supermimic.rotations import quat_to_mat
from dex_supermimic.rl.z5_supermimic.mdp import Z5PenMDP
from dex_supermimic.rl.z5_supermimic.state import NUM_J, PEN_MEMORY_DIMS
from dex_supermimic.tasks.z5_pen_spin.z5_constants import (
  Z5_FINGER_COLLISION_GEOMS,
  Z5_HAND_PEN_CONTACT_BODIES,
  Z5_HAND_PEN_CONTACT_GEOMS,
)

# Checkpoint context_names (order defines xi).
Z5_CONTEXT_NAMES = [
  ("friction_ratio", 1),
  ("object_mass_ratio", 1),
  ("kp_ratio", 1),
  ("kd_ratio", 1),
  ("joint_damping_ratio", 1),
  ("object_size_ratio", 1),
  ("object_com_local", 3),
  ("actuator_kp_ratios", NUM_J),
  ("actuator_kd_ratios", NUM_J),
]


class Z5SimReader(PenSimReader):
  def __init__(self, env):
    self.env, self.device = env, env.device
    self.robot, self.obj = env.scene["robot"], env.scene["object"]
    self.tip_ids = self.robot.find_sites(".*_tip_reference_site")[0]
    self.kin = HandKinematics(
      env.sim.mj_model,
      self.robot.indexing.body_ids.cpu().tolist(),
      self.robot.indexing.joint_ids.cpu().tolist(),
      self.robot.indexing.site_ids[self.tip_ids].cpu().tolist(),
      env.device,
    )
    # Resolve the 21 command link sites to (FK body index, local offset).
    import numpy as np

    model = env.sim.mj_model
    body_ids = self.robot.indexing.body_ids.cpu().tolist()
    body_local = {int(b): i for i, b in enumerate(body_ids)}
    site_gids = [
      int(model.site(name).id) for name in ("robot/" + n for n in self._link_names(env))
    ]
    self.link_site_body_ids = [body_local[int(model.site_bodyid[g])] for g in site_gids]
    self.link_site_offsets = torch.as_tensor(
      np.stack([model.site_pos[g] for g in site_gids]),
      dtype=torch.float32,
      device=env.device,
    )
    self.refresh_frame()
    self.mdp = Z5PenMDP(self)
    self.actuator_defaults = self._read_actuator_defaults()
    for key, attr in (("kp", "kp_default"), ("kd", "kd_default"), ("force_limit", "force_limit"), ("gear", "gear")):
      setattr(self.mdp, attr, self.actuator_defaults[key].to(self.device))
    # 15-slot finger self-collision sensor: slots follow Z5_FINGER_COLLISION_GEOMS.
    sensor = env.scene["finger_collision"]
    names = sensor.primary_names
    expected = [g for g in Z5_FINGER_COLLISION_GEOMS]
    if sorted(names) != sorted(expected):
      raise ValueError(f"finger_collision sensor slots {names} != {expected}")
    self.finger_slot_order = [names.index(g) for g in expected]
    self.xi = self.context()
    self.context_names = list(Z5_CONTEXT_NAMES)
    self.reward_snapshot = None

  @staticmethod
  def _link_names(env):
    return tuple(env.command_manager.get_term("motion").cfg.link_names)

  def auxiliary_observations(self):
    return Z5PenObservations(self)

  def context(self):
    """xi (53): 6 DR ratios + pen COM local offset + per-actuator kp/kd ratios."""
    from mjlab.managers import SceneEntityCfg

    from dex_supermimic.tasks.dex_reorient.mdp.observations import dr_params_privileged

    dr = dr_params_privileged(
      self.env,
      robot_cfg=SceneEntityCfg(
        "robot",
        geom_names=(r".*_collision$",),
        actuator_names=(".*",),
        joint_names=(".*",),
      ),
      object_cfg=SceneEntityCfg(
        "object",
        body_names=("cube",),
        geom_names=("cube", "A_bottom", "A_top"),
      ),
    )
    n = self.env.num_envs
    values = [v.reshape(n, 1) for v in dr.unbind(-1)]
    # Pen COM local offset (body ipos of the pen body), per-env after DR.
    obj_body = self.obj.indexing.body_ids[0].long()
    ipos = self.env.sim.model.body_ipos[:, obj_body]
    if ipos.ndim == 1:
      ipos = ipos.unsqueeze(0).expand(n, -1)
    values.append(ipos.clone())
    self.context_names = [(name, dim) for (name, dim) in Z5_CONTEXT_NAMES[:7]]
    return self._append_actuator_context(values=values)

  def contact(self):
    """15-slot finger self-collision labels in Z5_FINGER_COLLISION_GEOMS order."""
    found = self.env.scene["finger_collision"].data.found > 0
    return found[:, self.finger_slot_order].float()

  def hand_pen_contact(self):
    """23-slot hand-pen contact labels (hand_pen_contact_names order).

    The sensor matches per-body collision GEOMS (see env_cfgs), so the columns
    are reordered with Z5_HAND_PEN_CONTACT_GEOMS -- which is listed in exactly
    the checkpoint's hand_pen_contact_names body order.
    """
    sensor = self.env.scene["hand_pen_contact"]
    names = sensor.primary_names
    order = [names.index(g) for g in Z5_HAND_PEN_CONTACT_GEOMS]
    return (sensor.data.found > 0)[:, order].float()

  def memory(self):
    env, motion = self.env, self.mdp.motion
    act = env.action_manager.get_term("joint_pos")
    scalar = lambda x: x.float().unsqueeze(-1).clone()
    z = torch.zeros(env.num_envs, 1, device=self.device)
    out = {k: z.expand(-1, d).clone() for k, d in PEN_MEMORY_DIMS.items()}
    out.update(
      c1=act.processed_action.clone(),
      c2=act.processed_action.clone(),
      a1=env.action_manager.action.clone(),
      a2=env.action_manager.prev_action.clone(),
      goal=torch.eye(3, device=self.device).flatten()[None].expand(env.num_envs, -1),
      cage=env.reward_manager.get_term_cfg("overcurrent").func._charge.clone(),
      age=scalar(env.episode_length_buf),
      time=scalar(motion.time_steps),
      start=scalar(motion.trajectory_starts),
      end=scalar(motion.trajectory_ends),
      limits=self.torque_limits(),
      encoder_bias=self.robot.data.encoder_bias.clone(),
      frozen_target=act._targets.clone(),
      mask=act._mask_dofs.float().clone(),
      mask_remaining=scalar(act._mask_remaining),
      mask_active=scalar(act._mask_active),
    )
    camera = env.observation_manager.get_term_cfg("policy", "object_pose").func
    if hasattr(camera, "_history"):
      from dex_supermimic.tasks.pen_spin.mdp.observations import (
        bounded_object_pose_to_points_wrist_local,
      )

      def points(age):
        return (
          bounded_object_pose_to_points_wrist_local(
            camera._history[age], motion.axis_points, position_tanh_scale_m=self.mdp.bound
          )
          .flatten(1)
          .clone()
        )

      out.update(
        camera_raw=points(torch.zeros_like(camera._age)),
        camera_obs=points(camera._age),
        camera_holding=scalar(camera._holding),
        camera_ticks=scalar(camera._hold_ticks),
        camera_bias_pos=camera._bias_pos.clone(),
        camera_bias_rot=quat_to_mat(camera._bias_quat).flatten(1),
      )
    return out

  def attach_capture(self):
    env = self.env
    self._original_reward = env.reward_manager.compute

    def reward(dt):
      physics, contact = self.state(), self.contact().clone()
      hpc = self.hand_pen_contact().clone()
      torque = self.robot.data.actuator_force.clone()
      nan = env.termination_manager.get_term("unstable").clone()
      drop = env.termination_manager.terminated.clone()
      result = self._original_reward(dt)
      self.reward_snapshot = {
        "physics": physics,
        "contact": contact,
        "hpc": hpc,
        "torque": torque,
        "nan": nan,
        "drop": drop,
        "reward": result.clone(),
        "terms": {
          name: env.reward_manager._step_reward[:, j].clone() * (dt if env.reward_manager._scale_by_dt else 1)
          for j, name in enumerate(env.reward_manager._term_names)
        },
      }
      return result

    env.reward_manager.compute = reward


class Z5PenObservations(PenObservations):
  """Z5 frame/context layout: frame 311, context 387, critic = frame."""

  base_terms = {
    "motion": 91,
    "joint_pos": 22,
    "joint_vel": 22,
    "object_pose": 6,
    "last_action": 22,
    "link_pos": 63,
    "joint_position_error": 22,
    "link_position_error": 63,
  }

  def __init__(self, reader):
    self.reader, self.mdp = reader, reader.mdp
    self.frame_dim = sum(self.base_terms.values())
    # context = frame(311) + charge(22) + time(1) + xi(53) = 387
    self.context_dim = reader.xi.shape[-1] + self.frame_dim + NUM_J + 1

  def frame(self, x, memory, auxiliary):
    from dex_supermimic.rl.z5_supermimic.state import unpack

    q, qd, *_ = unpack(x)
    ref_q, _, ref_links = self.mdp.reference(memory)
    links = self.mdp.link_positions(q, qd).flatten(-2)
    return torch.cat((self.mdp.observation(x, memory), links, q - ref_q, links - ref_links.flatten(-2)), -1)
