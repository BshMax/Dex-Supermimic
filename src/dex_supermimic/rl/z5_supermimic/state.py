# SPDX-License-Identifier: Apache-2.0
"""Z5 (22-DoF) physical state and controller memory layout.

Mirror of ``rl/dex_supermimic/state.py`` with NUM_J=22, matching the checkpoint:
STATE_DIM=62, DELTA_DIM=56, error_floor=[0.003]*22+[0.05]*22+[0.001]*3+
[0.005]*3+[0.02]*3+[0.05]*3.
"""

import torch

from dex_supermimic.rl.dex_supermimic.rotations import mm, so3_exp, so3_log

NUM_J = 22
STATE_DIM = 2 * NUM_J + 18
DELTA_DIM = 2 * NUM_J + 12


def unpack(x):
    q, qd, p, r, v, w = x.split((NUM_J, NUM_J, 3, 9, 3, 3), -1)
    return q, qd, p, r.reshape(*r.shape[:-1], 3, 3), v, w


def pack(q, qd, p, r, v, w):
    return torch.cat((q, qd, p, r.flatten(-2), v, w), -1)


def integrate(x, delta):
    q, qd, p, r, v, w = unpack(x)
    dq, dqd, dp, dr, dv, dw = delta.split((NUM_J, NUM_J, 3, 3, 3, 3), -1)
    return pack(q + dq, qd + dqd, p + dp, mm(r, so3_exp(dr)), v + dv, w + dw)


def difference(x, target):
    q, qd, p, r, v, w = unpack(x)
    qt, qdt, pt, rt, vt, wt = unpack(target)
    return torch.cat(
        (
            qt - q,
            qdt - qd,
            pt - p,
            so3_log(mm(r.transpose(-1, -2), rt)),
            vt - v,
            wt - w,
        ),
        -1,
    )


# Controller / command memory (22-DoF fields). Pen task only: goal/hold/window
# fields are kept for schema compatibility but unused (the reference clip owns
# the target).
MEMORY_DIMS = {
    "c1": NUM_J,
    "c2": NUM_J,
    "a1": NUM_J,
    "a2": NUM_J,
    "goal": 9,
    "hold": 1,
    "goal_timer": 1,
    "window": 1,
    "window_timer": 1,
    "reward_window": 1,
    "reward_hold": 1,
    "reward_snapshot": 1,
    "within": 1,
    "cage": 1,
    "age": 1,
    "count": 1,
}

# Pen memory: the Z5 fork's replay memory (checkpoint PEN_MEMORY_DIMS=230 for
# Wuji; Z5 scales the 20-DoF fields to 22).
PEN_MEMORY_DIMS = dict(
    MEMORY_DIMS,
    cage=NUM_J,        # per-joint overcurrent charge integrator
    time=1,
    start=1,
    end=1,
    limits=NUM_J,      # per-env randomized torque limits
    camera_raw=6,
    camera_obs=6,
    camera_holding=1,
    camera_ticks=1,
    camera_bias_pos=3,
    camera_bias_rot=9,
    encoder_bias=NUM_J,
    frozen_target=NUM_J,
    mask=NUM_J,
    mask_remaining=1,
    mask_active=1,
)


def flatten_memory(m):
    return torch.cat([m[k] for k in MEMORY_DIMS], -1)


def unflatten_memory(x):
    return dict(zip(MEMORY_DIMS, x.split(tuple(MEMORY_DIMS.values()), -1)))


def flatten_pen_memory(memory):
    return torch.cat([memory[key] for key in PEN_MEMORY_DIMS], -1)


def unflatten_pen_memory(value):
    return dict(zip(PEN_MEMORY_DIMS, value.split(tuple(PEN_MEMORY_DIMS.values()), -1)))
