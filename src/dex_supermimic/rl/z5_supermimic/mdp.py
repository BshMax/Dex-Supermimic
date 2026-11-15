# SPDX-License-Identifier: Apache-2.0
"""Z5 analytic MDP: imagination-side observations, rewards, terminations.

Extends the Wuji PenMDP with:
- 22-DoF joints (NUM_J from rl.z5_supermimic.state)
- site-based link positions (21 ``*_reference_site`` sites, palm/root frame)
- hand-pen contact (23 channels) in the reward
- checkpoint reward weights (action_rate -1.0, finger_collision -0.5,
  hand_pen_contact +0.1; tracking sigmas match the Wuji recipe)

The reference_contact_* terms of the private fork (axial/surface shaping off
z5_surface.npz) are NOT yet included; see Z5_PORT_SPEC.md. The reward_parity
metric during collection quantifies any analytic-vs-simulator mismatch.
"""

import torch

from dex_supermimic.rl.dex_supermimic.kinematics import HandKinematics
from dex_supermimic.rl.dex_supermimic.mdp import DexMDP
from dex_supermimic.rl.dex_supermimic.pen import PenMDP
from dex_supermimic.rl.dex_supermimic.rotations import mm, mv, quat_to_mat
from dex_supermimic.rl.z5_supermimic.state import NUM_J, unpack


class Z5PenMDP(PenMDP):
  def __init__(self, reader):
    DexMDP.__init__(
      self,
      reader.kin,
      reader.robot.data.default_joint_pos[0].clone(),
      reader.robot.data.joint_pos_limits[0, :, 0].clone(),
      reader.robot.data.joint_pos_limits[0, :, 1].clone(),
      torch.zeros(3, device=reader.env.device),
      torch.eye(3, device=reader.env.device),
      50,
    )
    env = reader.env
    self.motion = env.command_manager.get_term("motion")
    self.alpha = env.cfg.actions["joint_pos"].alpha
    # The 21 command link points are SITES on Z5. Resolve each site to its
    # parent body (FK index) plus the constant local offset, so imagination can
    # reconstruct site positions differentiably from q.
    self.link_site_body_ids = reader.link_site_body_ids
    self.link_site_offsets = reader.link_site_offsets
    self.nominal_limits = torch.as_tensor(
      env.sim.mj_model.actuator_forcerange[reader.robot.indexing.ctrl_ids.cpu().numpy(), 1].copy(),
      device=env.device,
      dtype=torch.float32,
    )
    self.radius = self.motion.cfg.axis_point_radius_m
    self.bound = self.motion.cfg.object_position_tanh_scale_m
    self.camera_enabled = env.cfg.observations["policy"].enable_corruption
    self.mask_cfg = env.cfg.actions["joint_pos"]
    self.env = env
    import hashlib

    digest = hashlib.sha256()
    for field in ("joint_pos", "object_pose", "link_pos"):
      digest.update(self.motion._frames[field].cpu().numpy().tobytes())
    digest.update(self.motion.group.trajectory_end.cpu().numpy().tobytes())
    self.reference_signature = digest.hexdigest()

  def reference(self, memory):
    frame = (memory["start"] + memory["time"].clamp_min(0)).minimum(memory["end"] - 1).long().squeeze(-1)
    frames = self.motion._frames
    return frames["joint_pos"][frame], frames["object_pose"][frame], frames["link_pos"][frame]

  def pd_torque(self, x, u, context=None, *, limits=None):
    """Analytic position-actuator torque at the pre-substep state (22 DoF)."""
    q, qd, *_ = unpack(x)
    kp = self.kp_default * context[..., -2 * NUM_J : -NUM_J]
    kd = self.kd_default * context[..., -NUM_J:]
    if limits is None:
      limits = self.force_limit
    return (kp * (u - q) + kd * qd).clamp(-limits, limits)

  def termination(self, x, memory):
    """Soft pen-drop termination: position 5 cm / shaft 60 deg, sigmoid-smoothed."""
    _, _, pos, rot, *_ = unpack(x)
    _, ref, _ = self.reference(memory)
    position = torch.sigmoid(((pos - ref[..., :3]).norm(dim=-1) - 0.05) / 0.002)
    cosine = (rot[..., :, 0] * quat_to_mat(ref[..., 3:])[..., :, 0]).sum(-1)
    orientation = torch.sigmoid((0.5 - cosine) / 0.02)
    return 1 - (1 - position) * (1 - orientation)

  def link_positions(self, q, qd):
    """Palm(root)-frame positions of the 21 reference sites via differentiable FK."""
    bodies, rots, *_ = self.kin.forward(q, qd)
    body_pos = bodies[..., self.link_site_body_ids, :]
    body_rot = rots[..., self.link_site_body_ids, :, :]
    return body_pos + mv(body_rot, self.link_site_offsets)


  def rewards(self, x1, memory, action, residual, contact, torque, soft=False, hpc=None):
    q, qd, pos, rot, *_ = unpack(x1)
    ref_q, ref_pose, ref_links = self.reference(memory)
    links = self.link_positions(q, qd)
    point_error = (self.points(pos, rot) - self.reference_points(ref_pose)).norm(dim=-1).mean(-1)
    load = (torque.abs() / memory["limits"].clamp_min(1e-8)).clamp(max=1)
    charge = (memory["cage"] + self.dt * (load - memory["cage"] / 2)).clamp(0, 1)
    terms = {
      "object_pos": 2 * torch.exp(-((pos - ref_pose[..., :3]).norm(dim=-1) / 0.05).square()),
      "object_rot": 8 * torch.exp(-(point_error / 0.04).square()),
      "link_pos": torch.exp(-((links - ref_links).norm(dim=-1) / 0.025).square()).mean(-1),
      "joint_pos": torch.exp(
        -((q - ref_q).abs() / (self.upper - self.lower).clamp_min(1e-6) / 0.1).square()
      ).mean(-1),
      # Checkpoint config: action_rate_weight=-1.0 (combined first+second diff).
      "action_rate": -1.0
      * ((action - memory["a1"]).square() + (action - 2 * memory["a1"] + memory["a2"]).square()).sum(-1),
      "torque": -(torque / self.nominal_limits).square().sum(-1),
      "overcurrent": -4 * (charge - 0.8).clamp_min(0).sum(-1) / 0.2,
      # Checkpoint config: finger_collision_weight=-0.5 over the 15 slots.
      "finger_collision": -0.5 * contact.sum(-1),
    }
    if hpc is not None:
      # Checkpoint config: hand_pen_contact_weight=0.1, mirroring the simulator
      # term (mean activated channel, capped at 3 like the env reward).
      terms["hand_pen_contact"] = 0.1 * hpc.sum(-1).clamp(max=3.0) / 3.0
    terms = {name: value * self.dt for name, value in terms.items()}
    return sum(terms.values()), charge, terms

  def observation(self, x, memory):
    q, qd, pos, rot, *_ = unpack(x)
    ref_q, ref_pose, ref_links = self.reference(memory)
    command = torch.cat((ref_q, self.reference_points(ref_pose, True).flatten(-2), ref_links.flatten(-2)), -1)
    return torch.cat(
      (command, q - self.default, qd * 0.25, self.points(pos, rot, True).flatten(-2), memory["a1"]), -1
    )

  def policy_observation(self, x, memory, auxiliary):
    obs = self.observation(x, memory)
    if not self.camera_enabled:
      return obs
    q, qd, pos, rot, *_ = unpack(x)
    bias_rot = memory["camera_bias_rot"].reshape(-1, 3, 3)
    axis = torch.nn.functional.normalize(torch.randn_like(pos), dim=-1)
    angle = pos.new_empty(pos.shape[0]).uniform_(-0.015, 0.015)
    from dex_supermimic.rl.dex_supermimic.rotations import axis_angle_to_mat

    rotation = mm(axis_angle_to_mat(axis, angle), mm(bias_rot, rot))
    position = pos + memory["camera_bias_pos"] + torch.empty_like(pos).uniform_(-0.002, 0.002)
    raw = self.points(position, rotation, True).flatten(-2)
    roll = torch.rand_like(memory["camera_holding"])
    holding = torch.where(memory["camera_holding"] > 0, (roll < 0.78) & (memory["camera_ticks"] < 40), roll < 0.03)
    camera = torch.where(holding, memory["camera_obs"], memory["camera_raw"])
    memory.update(
      camera_raw=raw,
      camera_obs=camera,
      camera_holding=holding.float(),
      camera_ticks=torch.where(holding, memory["camera_ticks"] + 1, 0),
    )
    motion_dim = NUM_J + 6 + 21 * 3  # 91 for Z5
    return torch.cat(
      (
        obs[..., :motion_dim],
        q - self.default + memory["encoder_bias"] + torch.empty_like(q).uniform_(-0.015, 0.015),
        qd * 0.25 + torch.empty_like(qd).uniform_(-0.05, 0.05),
        camera,
        memory["a1"],
      ),
      -1,
    )

  def control_target(self, action, memory, bias):
    """residual = EMA(alpha=0.3) of clipped 0.5*a; target = ref_q + residual - bias."""
    residual = self.alpha * (0.5 * action).clamp(-0.6, 0.6) + (1 - self.alpha) * memory["c1"]
    reference = self.reference(memory)[0]
    target = reference + residual - bias
    cfg = self.mask_cfg
    if cfg.mask_prob > 0:
      from dex_supermimic.tasks.pen_spin.mdp.curriculum_utils import get_curriculum_value

      scale = get_curriculum_value(self.env, cfg.mask_curriculum_term, 0.0) if cfg.mask_curriculum_term else 1.0
      active_at_start = memory["mask_active"] > 0
      remaining = memory["mask_remaining"] - active_at_start.float()
      active = active_at_start & (remaining > 0)
      fire = (~active_at_start) & (torch.rand_like(remaining) < cfg.mask_prob * scale)
      chosen = torch.rand_like(action).topk(cfg.mask_num_dofs, dim=-1).indices
      selected = torch.zeros_like(action).scatter(-1, chosen, 1)
      mask = torch.where(fire, selected, memory["mask"] * active.float())
      maximum = max(
        round(cfg.mask_duration_min + (cfg.mask_duration_max - cfg.mask_duration_min) * scale),
        cfg.mask_duration_min,
      )
      duration = torch.randint(cfg.mask_duration_min, maximum + 1, remaining.shape, device=action.device).float()
      remaining = torch.where(fire, duration, remaining)
      active = active | fire
      target = torch.where((mask > 0) & (memory["age"] > 0), memory["frozen_target"], target)
      memory.update(mask=mask, mask_remaining=remaining, mask_active=active.float())
    memory["frozen_target"] = target
    return residual, target

  @staticmethod
  def flatten_history(history):
    return history[:, -1]
