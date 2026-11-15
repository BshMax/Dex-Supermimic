# SPDX-License-Identifier: Apache-2.0
"""Z5 SuperMimic runner: collect -> WM fit -> differentiable imagination.

Standalone runner for the z5-pen-spin task (it shares no __init__ with the Wuji
DexSuperMimicRunner because every dimension differs), but follows the same
three-phase MBRL loop. The extra hand-pen-contact (hpc, 23-D) channel threads
through collection, WM supervision and imagination: it is both a WM input
(measured or fed-back prediction) and a WM output (BCE-supervised, sigmoid-ed
into the shaped reward).

This file is the heart of "how the checkpoint was trained". Read it against
outputs/z5_reverse/Z5_PORT_SPEC.md in wuji-mjlab.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dex_supermimic.rl.dex_supermimic.common import RunningNorm
from dex_supermimic.rl.dex_supermimic.metrics import MetricLogger
from dex_supermimic.rl.dex_supermimic.networks import Critic
from dex_supermimic.rl.dex_supermimic.rotations import quat_to_mat
from dex_supermimic.rl.z5_supermimic.networks import ActorZ5, FailureHeadZ5, WorldModelZ5
from dex_supermimic.rl.z5_supermimic.sim import Z5PenObservations, Z5SimReader
from dex_supermimic.rl.z5_supermimic.state import (
  NUM_J,
  difference,
  flatten_pen_memory,
  unflatten_pen_memory,
)

HPC_DIM = 23


class Z5SuperMimicRunner:
  format_version_value = 4  # matches the private fork's z5 checkpoints

  def __init__(self, env, cfg, log_dir=None):
    if torch.distributed.is_initialized():
      raise ValueError("Z5SuperMimic currently supports one GPU/process")
    self.env = env
    self.cfg = cfg
    self.device = env.device
    self.obs = env.reset()[0]["policy"]
    self.flatten_memory = flatten_pen_memory
    self.unflatten_memory = unflatten_pen_memory
    self.history_from_obs = lambda obs: obs[:, None]

    self.reader = Z5SimReader(env)
    self.mdp = self.reader.mdp
    self.observations = Z5PenObservations(self.reader)

    initial_context = self.observations.context(
      self.reader.state(), self.reader.memory(), self.observations.auxiliary(), self.reader.xi
    )
    initial_dynamics = self.observations.dynamics(
      self.reader.state(), self.reader.xi, self.observations.disturbance(self.observations.auxiliary())
    )
    self.reader.attach_capture()

    self.actor = ActorZ5(self.obs.shape[-1], NUM_J).to(self.device)
    contact_dim = self.reader.contact().shape[-1]
    dynamics_dim = initial_dynamics.shape[-1]
    self.dynamics_dim = dynamics_dim
    self.context_dim = self.observations.context_dim
    self.contact_dim = contact_dim
    self.hand_pen_contact_dim = HPC_DIM
    self.wm = WorldModelZ5(dynamics_dim, contact_dim, HPC_DIM, cfg.wm_hidden, cfg.amp).to(self.device)
    self.fail_head = FailureHeadZ5(self.context_dim).to(self.device)
    self.torque_norm = RunningNorm(NUM_J, eps=1e-3).to(self.device)
    self.critic = Critic(self.observations.critic(initial_context).shape[-1], cfg.amp).to(self.device)

    self.optimizers = {
      name: torch.optim.Adam(module.parameters(), lr=cfg.lr) for name, module in self.modules().items()
    }
    self.optimizers["actor"] = torch.optim.AdamW(
      self.actor.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0
    )

    from dex_supermimic.rl.dex_supermimic.buffer import ReplayBuffer

    self.buffer = ReplayBuffer(cfg.buffer_steps, env.num_envs, self.device, cfg.holdout_fraction)

    self.log_dir = Path(log_dir or cfg.log_dir)
    self.log_dir.mkdir(parents=True, exist_ok=True)
    if cfg.wandb_mode:
      os.environ["WANDB_MODE"] = cfg.wandb_mode
    self.writer = MetricLogger(
      cfg.logger,
      log_dir=self.log_dir,
      config=cfg.to_dict(),
      run_name=f"z5 pen spin {self.log_dir.name}",
      project=cfg.wandb_project,
      entity=cfg.wandb_entity,
    )
    self.iteration = 0
    self.sim_steps = 0
    self.wall_offset = 0.0
    self.started_at = time.perf_counter()
    self.env.cfg.auto_reset = False
    self.env.max_common_steps = cfg.warmup_steps + int(cfg.max_iterations / cfg.utd)

    self.step_wm = torch.compile(self.wm) if cfg.compile and str(self.device).startswith("cuda") else self.wm
    self.step_actor = (
      torch.compile(self.actor) if cfg.compile and str(self.device).startswith("cuda") else self.actor
    )
    self.reward_step = self.mdp.rewards
    self.task_features = self.observations.context
    self.wm_features = self.observations.dynamics
    if cfg.compile_policy and str(self.device).startswith("cuda"):
      options = {"dynamic": True} if cfg.compile_policy_dynamic else {}
      self.reward_step = torch.compile(self.mdp.rewards, **options)
      self.task_features = torch.compile(self.observations.context, **options)
      self.wm_features = torch.compile(self.observations.dynamics, **options)

    if cfg.checkpoint:
      self.load(cfg.checkpoint)
    if cfg.model_checkpoint:
      self.load_model_only(cfg.model_checkpoint)
    (self.log_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (self.log_dir / "context.json").write_text(json.dumps(self.reader.context_names, indent=2))
    layout = {
      "base_terms": self.observations.base_terms,
      "current_terms": {},
      "wm_history": 1,
      "critic_history": 1,
      "wm_input_dim": self.wm.norm.mean.numel(),
      "wm_net_input_dim": dynamics_dim + NUM_J + HPC_DIM,
      "critic_input_dim": self.critic.norm.mean.numel(),
      "dynamics_input_dim": dynamics_dim,
      "hand_pen_contact_dim": HPC_DIM,
      "simulator_context": self.reader.context_names,
    }
    (self.log_dir / "input_layout.json").write_text(json.dumps(layout, indent=2))
    import dataclasses

    def serialize(value):
      if dataclasses.is_dataclass(value):
        return {f.name: serialize(getattr(value, f.name)) for f in dataclasses.fields(value)}
      if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in value.items()}
      if isinstance(value, (tuple, list)):
        return [serialize(v) for v in value]
      if callable(value):
        return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', repr(value))}"
      if isinstance(value, (str, int, float, bool)) or value is None:
        return value
      return repr(value)

    (self.log_dir / "environment_config.json").write_text(json.dumps(serialize(env.cfg), indent=2))
    import cloudpickle

    (self.log_dir / "environment_config.pkl").write_bytes(cloudpickle.dumps(env.cfg))

  # ---- plumbing ------------------------------------------------------------
  def modules(self):
    return dict(actor=self.actor, wm=self.wm, fail_head=self.fail_head, critic=self.critic)

  def critic_features(self, context):
    return self.observations.critic(context)

  def optimize(self, name, loss):
    optimizer = self.optimizers[name]
    optimizer.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
      raise FloatingPointError(f"Non-finite {name} loss")
    loss.backward()
    norm = nn.utils.clip_grad_norm_(self.modules()[name].parameters(), 1.0)
    if name == "actor":
      nn.utils.clip_grad_value_(self.actor.parameters(), 0.1)
    if not torch.isfinite(norm):
      if name == "actor":
        return
      raise FloatingPointError(f"Non-finite {name} gradient")
    optimizer.step()

  def lr_at(self, iteration):
    c = self.cfg
    progress = min(iteration / c.lr_decay_iters, 1.0)
    return c.lr * (c.lr_final_ratio + (1 - c.lr_final_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))

  def advance_memory(self, x1, memory, action, target, cage, next_goal, soft=True):
    updated = self.mdp.command_step(x1, memory, next_goal, soft=soft)
    updated.update(c1=target, c2=memory["c1"], a1=action, a2=memory["a1"], cage=cage, age=memory["age"] + 1)
    return updated

  def sample_starts(self):
    rows = (
      self.buffer.ptr - self.buffer.size + torch.arange(self.buffer.size, device=self.device)
    ) % self.buffer.capacity
    eligible = self.buffer.data["valid"][rows].nonzero(as_tuple=False)
    if not len(eligible):
      raise RuntimeError("Replay contains no finite imagination starts")
    selected = eligible[torch.randint(len(eligible), (self.cfg.pi_batch,), device=self.device)]
    row, env = rows[selected[:, 0]], selected[:, 1]
    out = {k: v[row, env] for k, v in self.buffer.data.items()}
    out.update(env=env, sim_context=self.reader.xi[env])
    return out

  @torch.no_grad()
  def replay_failure_features(self, w, simulator):
    batch, window = w["x1"].shape[:2]
    x = w["x1"].flatten(0, 1)
    context = self.observations.context(
      x,
      self.unflatten_memory(w["m"].flatten(0, 1)),
      w["aux"].flatten(0, 1),
      simulator[:, None].expand(-1, window, -1).reshape(batch * window, -1),
    )
    return self.fail_head.features(x, context).reshape(batch, window, -1)

  # ---- phase 1: collect from the real simulator ----------------------------
  @torch.no_grad()
  def collect(self, steps):
    rewards, drops, term_gaps, gaps = [], [], {}, []
    for _ in range(steps):
      self.reader.refresh_frame()
      x = self.reader.state()
      m = self.reader.memory()
      auxiliary = self.observations.auxiliary()
      context = self.observations.context(x, m, auxiliary, self.reader.xi)
      self.actor.norm.update(self.obs[torch.isfinite(self.obs).all(-1)])
      action = (self.actor(self.obs) + self.cfg.action_noise * torch.randn_like(m["a1"])).clamp(-1, 1)
      obs, reward, terminated, truncated, _ = self.env.step(action)
      cap = self.reader.reward_snapshot
      c = self.env.action_manager.get_term("joint_pos").processed_action.clone()
      u = self.reader.executed_target()
      next_x = self.reader.state()
      valid = (
        (~cap["nan"])
        & torch.isfinite(x).all(-1)
        & torch.isfinite(cap["physics"]).all(-1)
        & torch.isfinite(cap["contact"]).all(-1)
        & torch.isfinite(cap["hpc"]).all(-1)
        & torch.isfinite(cap["torque"]).all(-1)
        & torch.isfinite(u).all(-1)
        & torch.isfinite(c).all(-1)
        & torch.isfinite(action).all(-1)
        & torch.isfinite(self.flatten_memory(m)).all(-1)
        & torch.isfinite(next_x).all(-1)
        & torch.isfinite(self.obs).all(-1)
        & torch.isfinite(context).all(-1)
      )
      train_valid = valid.clone()
      train_valid[self.buffer.train_end :] = False
      self.torque_norm.update(cap["torque"][train_valid])
      dynamics = self.observations.dynamics(
        x[train_valid], self.reader.xi[train_valid], self.observations.disturbance(auxiliary[train_valid])
      )
      self.wm.norm.update(self.wm.inputs(dynamics, u[train_valid]))
      self.wm.delta_norm.update(difference(x[train_valid], cap["physics"][train_valid]))
      with torch.no_grad():
        selected_m = {key: value[train_valid] for key, value in m.items()}
        _, next_cage, _ = self.mdp.rewards(
          cap["physics"][train_valid], selected_m, action[train_valid], c[train_valid],
          cap["contact"][train_valid], cap["torque"][train_valid], hpc=cap["hpc"][train_valid],
        )
        next_memory = self.advance_memory(
          cap["physics"][train_valid], selected_m, action[train_valid], c[train_valid],
          next_cage, m["goal"][train_valid].reshape(-1, 3, 3), soft=False,
        )
        next_context = self.observations.context(
          cap["physics"][train_valid], next_memory, auxiliary[train_valid], self.reader.xi[train_valid]
        )
      self.fail_head.norm.update(self.fail_head.features(cap["physics"][train_valid], next_context))
      done = terminated | truncated
      self.buffer.add(
        dict(
          x=x, x1=cap["physics"], push=next_x - cap["physics"], u=u, c=c,
          action=action, m=self.flatten_memory(m), obs=self.obs, aux=auxiliary,
          contact=cap["contact"], hpc=cap["hpc"], torque=cap["torque"],
          reward=reward, done=done, terminated=terminated, valid=valid,
        )
      )
      # Reward parity: analytic (imagination-side) vs simulator reward terms.
      an, _, analytic_terms = self.mdp.rewards(
        cap["physics"], m, action, c, cap["contact"], cap["torque"], hpc=cap["hpc"]
      )
      denom = valid.float().sum().clamp_min(1)
      for name, value in analytic_terms.items():
        if name in cap["terms"]:
          error = (value - cap["terms"][name]).abs()
          term_gaps.setdefault(name, []).append(torch.where(valid, error, 0.0).sum() / denom)
      gaps.append(torch.where(valid, (an - cap["reward"]).abs(), 0.0).sum() / denom)
      rewards.append((reward * valid).sum() / denom)
      drops.append(cap["drop"].float().mean())
      self.obs = obs["policy"]
      self.reader.refresh_frame()
      ids = done.nonzero(as_tuple=False).flatten()
      if ids.numel():
        post, _ = self.env.reset(env_ids=ids)
        self.obs = torch.where(done[:, None], post["policy"], self.obs)
        self.reader.refresh_frame()
      self.sim_steps += self.env.num_envs
    return dict(
      **{"reward_parity/" + k: torch.stack(v).mean() for k, v in term_gaps.items()},
      norm_count=self.wm.norm.count.clone(),
      delta_norm_count=self.wm.delta_norm.count.clone(),
      reward=torch.stack(rewards).mean(),
      reward_parity=torch.stack(gaps).mean(),
      drop_rate=torch.stack(drops).mean(),
    )

  # ---- phase 2: supervised world-model + failure-head fit ------------------
  def fit_world_model(self, holdout=False):
    c = self.cfg
    window = c.wm_window
    w = self.buffer.sample(
      c.wm_batch, window, holdout,
      failure_frac=0.0 if holdout else c.wm_failure_frac,
      failure_horizon=c.wm_failure_horizon,
    )
    memory = self.unflatten_memory(w["m"][:, 0])
    auxiliary = w["aux"][:, 0]
    simulator = w["sim_context"] if "sim_context" in w else self.reader.xi[w["env"]]
    x = w["x"][:, 0]
    hpc_in = w["hpc"][:, 0]
    losses, state_losses, contact_losses, hpc_losses, failure_features = [], [], [], [], []
    std = torch.maximum(self.wm.delta_norm.std(), self.wm.error_floor)
    for j in range(window):
      dynamics = self.wm_features(x, simulator, self.observations.disturbance(auxiliary))
      x1, contact, hpc1 = self.step_wm(x, w["u"][:, j], dynamics, hpc_in)
      state_loss = (difference(x1, w["x1"][:, j]) / std).square().mean(-1)
      contact_loss = F.binary_cross_entropy_with_logits(contact, w["contact"][:, j], reduction="none").sum(-1)
      hpc_loss = F.binary_cross_entropy_with_logits(hpc1, w["hpc"][:, j], reduction="none").sum(-1)
      losses.append(c.final_state_weight * state_loss + c.contact_weight * (contact_loss + hpc_loss))
      state_losses.append(state_loss)
      contact_losses.append(contact_loss)
      hpc_losses.append(hpc_loss)
      tau = self.mdp.pd_torque(x, w["u"][:, j], simulator, limits=memory["limits"])
      _, cage, _ = self.reward_step(
        x1, memory, w["action"][:, j], w["c"][:, j], contact.sigmoid(), tau, soft=True, hpc=hpc1.sigmoid()
      )
      next_goal = memory["goal"].reshape(-1, 3, 3)
      memory = self.advance_memory(x1, memory, w["action"][:, j], w["c"][:, j], cage, next_goal)
      failure_features.append(
        self.fail_head.features(x1.detach(), self.observations.context(x1.detach(), memory, auxiliary, simulator))
      )
      auxiliary = w["aux"][:, min(j + 1, window - 1)]
      hpc_in = w["hpc"][:, min(j + 1, window - 1)]  # teacher-forced next measured hpc
      x = x1 + w["push"][:, j]
    mask = w["mask"]
    denom = mask.sum().clamp_min(1)
    wm_loss = (torch.stack(losses, 1) * mask).sum() / mask.shape[0]
    if not holdout:
      self.optimize("wm", wm_loss)
    true = self.replay_failure_features(w, simulator)
    predicted = torch.stack(failure_features, 1).detach()
    label = w["terminated"].float()
    failure_loss = (
      (
        F.binary_cross_entropy_with_logits(self.fail_head(true), label, reduction="none")
        + F.binary_cross_entropy_with_logits(self.fail_head(predicted), label, reduction="none")
      )
      * mask * w["importance"]
    ).sum() / (2 * denom)
    if not holdout:
      self.optimize("fail_head", failure_loss)
    return dict(
      supervision_window=window,
      wm_loss=wm_loss.detach(),
      failure_loss=failure_loss.detach(),
      state_loss=(torch.stack(state_losses, 1) * mask).sum() / denom,
      contact_digit_loss=(torch.stack(contact_losses, 1) * mask).sum() / denom,
      hpc_digit_loss=(torch.stack(hpc_losses, 1) * mask).sum() / denom,
    )

  # ---- phase 3: differentiable imagination ---------------------------------
  def policy_update(self):
    c = self.cfg
    w = self.sample_starts()
    x = w["x"]
    m = self.unflatten_memory(w["m"])
    history = self.history_from_obs(w["obs"])
    auxiliary = w["aux"]
    simulator = w["sim_context"]
    bias = self.reader.robot.data.encoder_bias[w["env"]]
    hpc = w["hpc"]
    horizon = c.horizon
    frozen = (self.wm, self.fail_head, self.critic)
    for mod in frozen:
      mod.requires_grad_(False)
    xs, rewards, continues = [], [], []
    survival = x.new_ones(x.shape[0])
    objective = x.new_zeros(x.shape[0])
    gamma_powers = [c.gamma**t for t in range(horizon + 1)]
    try:
      for t in range(horizon):
        context = self.task_features(x, m, auxiliary, simulator)
        xs.append(self.critic_features(context))
        a = self.step_actor(self.mdp.flatten_history(history))
        target, u = self.mdp.control_target(a, m, bias)
        x1, contact, hpc1 = self.step_wm(
          x, u, self.wm_features(x, simulator, self.observations.disturbance(auxiliary)), hpc
        )
        tau = self.mdp.pd_torque(x, u, simulator, limits=m["limits"])
        r, cage, _ = self.reward_step(x1, m, a, target, contact.sigmoid(), tau, soft=True, hpc=hpc1.sigmoid())
        d = self.mdp.termination(x1, m)
        reference_end = (m["time"] + 1 >= m["end"] - m["start"]).squeeze(-1)
        d = torch.where(reference_end, 1.0, d)
        next_goal = quat_to_mat(F.normalize(torch.randn(x.shape[0], 4, device=x.device), dim=-1))
        next_memory = self.advance_memory(x1, m, a, target, cage, next_goal)
        if c.failure_head and self.iteration >= c.wm_only_iters + c.failure_warmup_iters:
          next_context = self.task_features(x1, next_memory, auxiliary, simulator)
          p_fail = self.fail_head(self.fail_head.features(x1, next_context)).sigmoid().clamp(max=c.failure_clip)
          d = 1 - (1 - d) * (1 - p_fail)
        objective = objective + gamma_powers[t] * survival * r
        survival = survival * (1 - d)
        rewards.append(r.detach())
        continues.append((1 - d).detach())
        x = x1  # pen imagination: no exogenous velocity push
        # Feed the predicted contact PROBABILITY back as the next WM input.
        # fit_world_model teacher-forces binary 0/1 hpc labels, so raw logits
        # are out-of-distribution and imagination diverges (survival@32 -> 0);
        # sigmoid keeps the feedback in [0,1]. A/B verified: surv@32 0.00 -> 0.91.
        hpc = hpc1.sigmoid().detach()
        m = next_memory
        observation = self.mdp.policy_observation(x, m, auxiliary)
        history = torch.cat((history[:, 1:], observation[:, None]), 1)
      context = self.task_features(x, m, auxiliary, simulator)
      final = self.critic_features(context)
      xs.append(final)
      objective = objective + (c.gamma**horizon) * survival * self.critic(final)
      actor_loss = -objective.mean()
      self.optimize("actor", actor_loss)
    finally:
      for mod in frozen:
        mod.requires_grad_(True)
    inputs = torch.stack(xs).detach()
    rs, cs = torch.stack(rewards), torch.stack(continues)
    with torch.no_grad():
      self.critic.norm.update(inputs)
      values = self.critic(inputs)
      ret = values[-1]
      targets = []
      for t in reversed(range(horizon)):
        ret = rs[t] + c.gamma * cs[t] * ((1 - c.lam) * values[t + 1] + c.lam * ret)
        targets.append(ret)
      targets = torch.stack(targets[::-1])
    features = inputs[:-1].flatten(0, 1)
    targets = targets.flatten()
    count = features.shape[0]
    minibatch = count // c.critic_minibatches
    for _ in range(c.critic_epochs):
      permutation = torch.randperm(count, device=self.device)
      for i in range(c.critic_minibatches):
        ids = permutation[i * minibatch : (i + 1) * minibatch]
        critic_loss = (self.critic(features[ids]) - targets[ids]).square().mean()
        self.optimize("critic", critic_loss)
    return dict(
      actor_loss=actor_loss.detach(), critic_loss=critic_loss.detach(),
      survival=survival.detach().mean(), horizon=horizon,
    )

  # ---- training loop --------------------------------------------------------
  def learn(self):
    start = self.started_at
    self.collect(max(self.cfg.warmup_steps, self.cfg.wm_window))
    for it in range(self.iteration, self.cfg.max_iterations):
      self.iteration = it
      lr = self.lr_at(it)
      for optimizer in self.optimizers.values():
        for group in optimizer.param_groups:
          group["lr"] = lr
      # utd=1: one collect step per iteration.
      stats = self.collect(1)
      for _ in range(self.cfg.wm_batches):
        stats.update(self.fit_world_model())
      if it >= self.cfg.wm_only_iters:
        stats.update(self.policy_update())
      if it % self.cfg.log_interval == 0:
        if self.cfg.holdout_fraction:
          with torch.no_grad():
            stats.update({"holdout/" + k: v for k, v in self.fit_world_model(True).items()})
        stats.update(
          lr=lr,
          wall_seconds=self.wall_offset + time.perf_counter() - start,
          sim_steps=self.sim_steps,
        )
        scalar = {
          k: float(v) for k, v in stats.items()
          if torch.is_tensor(v) and v.numel() == 1 or isinstance(v, (int, float))
        }
        scalar["training/iteration"] = it
        self.writer.log_scalars(scalar, it)
        self._print_log(it, scalar)
      if (it + 1) % self.cfg.save_interval == 0:
        self.save(self.log_dir / f"model_{it + 1}.pt", it + 1)
    self.save(self.log_dir / "model_final.pt", self.cfg.max_iterations)
    self.writer.close()

  def _print_log(self, it, scalar):
    def g(key):
      return f"{scalar[key]:.6g}" if key in scalar else "--"

    print(
      f"\n=== Z5 iter {it:,}/{self.cfg.max_iterations:,} | {int(scalar.get('training/time', 0) if 'training/time' in scalar else scalar.get('wall_seconds', 0)) // 60}min"
      f" | reward={g('reward')} drop={g('drop_rate')} parity={g('reward_parity')}\n"
      f"    wm_loss={g('wm_loss')} (holdout {g('holdout/wm_loss')}) state={g('state_loss')} "
      f"contact={g('contact_digit_loss')} hpc={g('hpc_digit_loss')} fail={g('failure_loss')}\n"
      f"    actor={g('actor_loss')} critic={g('critic_loss')} survival={g('survival')} horizon={g('horizon')}",
      flush=True,
    )

  # ---- checkpointing --------------------------------------------------------
  def save(self, path, next_iteration=None):
    state = dict(
      format_version=self.format_version_value,
      simulation_backend="mjlab-mujoco-warp",
      reference_signature=getattr(self.mdp, "reference_signature", None),
      torque_norm=self.torque_norm.state_dict(),
      config=self.cfg.to_dict(),
      iteration=self.iteration if next_iteration is None else next_iteration,
      sim_steps=self.sim_steps,
      context_dim=self.context_dim,
      contact_dim=self.contact_dim,
      hand_pen_contact_dim=self.hand_pen_contact_dim,
      modules={k: v.state_dict() for k, v in self.modules().items()},
      optimizers={k: v.state_dict() for k, v in self.optimizers.items()},
      context_names=self.reader.context_names,
      joint_names=list(self.reader.robot.joint_names),
      wall_seconds=self.wall_offset + time.perf_counter() - self.started_at,
      common_step=self.env.common_step_counter,
      curriculum_scales={},
      cuda_rng=torch.cuda.get_rng_state_all() if str(self.device).startswith("cuda") else None,
      torch_rng=torch.get_rng_state(),
      numpy_rng=np.random.get_state(),
      python_rng=random.getstate(),
    )
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)

  def load(self, path):
    state = torch.load(path, map_location=self.device, weights_only=False)
    if state.get("format_version") != self.format_version_value:
      raise ValueError(f"Expected format_version {self.format_version_value}, got {state.get('format_version')}")
    if state["config"]["task"] != "z5-pen-spin":
      raise ValueError("Checkpoint task mismatch")
    if state.get("reference_signature") != self.mdp.reference_signature:
      raise ValueError("Z5 pen-spin reference data mismatch")
    if state["joint_names"] != list(self.reader.robot.joint_names):
      raise ValueError("Checkpoint joint-order mismatch")
    if "torque_norm" in state:
      self.torque_norm.load_state_dict(state["torque_norm"])
    for k, mod in self.modules().items():
      mod.load_state_dict(state["modules"][k])
    for k, opt in self.optimizers.items():
      if k in state.get("optimizers", {}):
        opt.load_state_dict(state["optimizers"][k])
    self.iteration = state["iteration"]
    self.sim_steps = state["sim_steps"]
    self.env.common_step_counter = state.get("common_step", 0)
    self.wall_offset = state.get("wall_seconds", 0.0)
    self.started_at = time.perf_counter()
    if state.get("cuda_rng") is not None and str(self.device).startswith("cuda"):
      torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])
    torch.set_rng_state(state["torch_rng"].cpu())
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])

  def load_model_only(self, path, strict_report=True):
    """Warm start: load actor/wm/fail_head/critic weights, fresh optimizers/iteration.

    Accepts the original private-fork checkpoint (isaaclab3-newton backend,
    format_version 4). Dimension layouts must match exactly; reference and
    context signatures are reported but do not block the load, because the
    simulator backend differs (documented gap in Z5_PORT_SPEC.md).
    """
    state = torch.load(path, map_location=self.device, weights_only=False)
    if state["config"]["task"] != "z5-pen-spin":
      raise ValueError("Not a z5-pen-spin checkpoint")
    report = {}
    if state.get("reference_signature") != self.mdp.reference_signature:
      report["reference_signature"] = (state.get("reference_signature"), self.mdp.reference_signature)
    if state["context_names"] != self.reader.context_names:
      report["context_names"] = (state["context_names"], self.reader.context_names)
    if state["joint_names"] != list(self.reader.robot.joint_names):
      raise ValueError("Checkpoint joint order differs from the Z5 entity")
    for name in ("wm", "fail_head", "critic", "actor"):
      missing, unexpected = self.modules()[name].load_state_dict(state["modules"][name], strict=False)
      if missing or unexpected:
        report[name] = dict(missing=list(missing), unexpected=list(unexpected))
    if "torque_norm" in state:
      self.torque_norm.load_state_dict(state["torque_norm"])
    if self.cfg.reset_model_statistics:
      for norm in (self.wm.norm, self.wm.delta_norm, self.fail_head.norm, self.torque_norm):
        norm.mean.zero_()
        norm.var.fill_(1.0)
        norm.count.zero_()
        norm._host_count = 0.0
    if strict_report and report:
      (self.log_dir / "warmstart_report.json").write_text(json.dumps(report, indent=2, default=str))
      print(f"[z5 warm-start] layout notes written to {self.log_dir / 'warmstart_report.json'}")
      print(f"[z5 warm-start] report keys: {list(report)}")
    return report

  def close(self):
    self.writer.close()
    self.reader.detach_capture()
