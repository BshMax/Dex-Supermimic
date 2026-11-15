# SPDX-License-Identifier: Apache-2.0
"""Z5 network variants matching the checkpoint state-dict layout exactly.

From model_final.pt (format_version 4, task z5-pen-spin):

- actor:     RunningNorm(163) + mlp(163,[512,256,128],22,ELU) + tanh
             (norm.mean/var/count -- RunningNorm buffers, NOT EmpiricalNorm)
- wm:        RunningNorm(287) + mlp(310,[2048]*3,94,SiLU,LayerNorm),
             delta_norm RunningNorm(56), error_floor(56),
             out = delta(56) + contact(15) + hand_pen_contact(23);
             net input = normed dynamics(287) + raw hpc(23) = 310
- critic:    RunningNorm(311) + mlp(311,[512,512,256,128],1)  (base class fits)
- fail_head: RunningNorm(449) + mlp(449,[256,128],1)           (base class fits)
"""

import torch
from torch import nn

from dex_supermimic.rl.dex_supermimic.common import RunningNorm, mlp
from dex_supermimic.rl.z5_supermimic.state import DELTA_DIM, integrate

NUM_J = 22
HPC_DIM = 23


class ActorZ5(nn.Module):
  """Deterministic tanh actor with RunningNorm (deploy policy.ts contract)."""

  def __init__(self, obs_dim: int = 163, action_dim: int = NUM_J):
    super().__init__()
    self.norm = RunningNorm(obs_dim, eps=1e-2)
    self.net = mlp(obs_dim, [512, 256, 128], action_dim)
    nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
    nn.init.zeros_(self.net[-1].bias)

  def forward(self, obs):
    return torch.tanh(self.net(self.norm(obs)))


class WorldModelZ5(nn.Module):
  """WM predicting state delta + finger self-collision + hand-pen contact.

  The hand-pen-contact state (23 binary channels) is an INPUT as well as an
  output: supervised fitting teacher-forces the measured channels; imagination
  feeds back the previous predicted sigmoid (see runner).
  """

  def __init__(self, dynamics_dim, contact_dim=15, hpc_dim=HPC_DIM, hidden=(2048, 2048, 2048), amp=True):
    super().__init__()
    self.contact_dim = contact_dim
    self.hpc_dim = hpc_dim
    self.amp = amp
    inp = dynamics_dim + NUM_J  # dynamics + control target u
    self.norm = RunningNorm(inp)
    self.net = mlp(inp + hpc_dim, list(hidden), DELTA_DIM + contact_dim + hpc_dim,
                   act=nn.SiLU, layer_norm=True)
    self.delta_norm = RunningNorm(DELTA_DIM, eps=1e-4)
    nn.init.zeros_(self.net[-1].weight)
    nn.init.zeros_(self.net[-1].bias)
    with torch.no_grad():
      self.net[-1].bias[DELTA_DIM:] = -3.0
    self.register_buffer(
      "error_floor",
      torch.tensor(
        [0.003] * NUM_J + [0.05] * NUM_J + [0.001] * 3 + [0.005] * 3 + [0.02] * 3 + [0.05] * 3
      ),
    )

  def inputs(self, dynamics, u):
    return torch.cat((dynamics, u), -1)

  def forward(self, x, u, dynamics, hpc):
    features = torch.cat((self.norm(self.inputs(dynamics, u)), hpc), -1)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and x.is_cuda):
      y = self.net(features)
    y = y.float()
    d1, c, h = y.split((DELTA_DIM, self.contact_dim, self.hpc_dim), -1)
    scale = self.delta_norm.std()
    return integrate(x, d1 * scale + self.delta_norm.mean), c, h


class FailureHeadZ5(nn.Module):
  """Dreamer-style continue head with Z5 dims: STATE_DIM(62) + context(387) = 449."""

  def __init__(self, context_dim):
    super().__init__()
    from dex_supermimic.rl.z5_supermimic.state import STATE_DIM

    dim = STATE_DIM + context_dim
    self.norm = RunningNorm(dim)
    self.net = mlp(dim, [256, 128], 1)

  def features(self, x1, context):
    return torch.cat((x1, context), -1)

  def forward(self, features):
    return self.net(self.norm(features)).squeeze(-1)
