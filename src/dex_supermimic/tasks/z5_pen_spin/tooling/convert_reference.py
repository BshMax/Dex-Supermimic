"""Convert the Z5 deploy bundle's reference.npz into the engine clip catalog.

The deploy bundle stores one concatenated palm-frame reference
(command/joint_pos/joint_vel/object_pose, 9 clips / 8918 frames, 50 Hz).
The engine's SingleHandTrackingReader instead expects per-clip NPZ files in
world frame with keys reference_qpos / wrist_pose_w / object_pose_w /
link_pos_w plus a group index.json (schema wuji_pen_spin_clips:1).

This tool reconstructs that catalog losslessly: the wrist is fixed at
FIXED_WRIST_POSE_W, so world-frame arrays are the palm-frame arrays composed
with that one transform (the reader subtracts it again at load time).

Run from the Dex-Supermimic checkout:

    uv run python -m dex_supermimic.tasks.z5_pen_spin.tooling.convert_reference \
        --reference /path/to/generated/deploy/z5-pen-spin/reference.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dex_supermimic.tasks.z5_pen_spin.z5_constants import (
  CLIP_GROUP_ID,
  FIXED_WRIST_POSE_W,
  NUM_LINKS,
  Z5_LINK_NAMES,
)

FRAME_DT = 0.02
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "rawdata" / "clips" / CLIP_GROUP_ID


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
  bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
  return np.stack(
    (
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ),
    -1,
  ).astype(np.float32)


def quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
  # t = 2 * cross(q_vec, v)
  tx = 2 * (y * vz - z * vy)
  ty = 2 * (z * vx - x * vz)
  tz = 2 * (x * vy - y * vx)
  return np.stack(
    (
      vx + w * tx + (y * tz - z * ty),
      vy + w * ty + (z * tx - x * tz),
      vz + w * tz + (x * ty - y * tx),
    ),
    -1,
  ).astype(np.float32)


def local_to_world(points_local: np.ndarray, wrist_pos: np.ndarray, wrist_quat_wxyz: np.ndarray) -> np.ndarray:
  leading = points_local.shape[:-1]
  q = np.broadcast_to(wrist_quat_wxyz, leading + (4,))
  return quat_rotate_wxyz(q, points_local) + wrist_pos


def pose_local_to_world(pos_local: np.ndarray, quat_local_wxyz: np.ndarray, wrist_pos, wrist_quat):
  pos_w = local_to_world(pos_local[..., None, :], wrist_pos, wrist_quat)[..., 0, :]
  quat_w = quat_mul_wxyz(np.broadcast_to(wrist_quat, pos_local.shape[:-1] + (4,)), quat_local_wxyz)
  return np.concatenate((pos_w, quat_w), -1).astype(np.float32)


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--reference", type=Path, required=True, help="deploy bundle reference.npz")
  parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
  args = parser.parse_args()

  data = np.load(args.reference, allow_pickle=False)
  clip_ids = [str(v) for v in data["clip_ids"]]
  lengths = data["lengths"].astype(np.int64)
  joint_pos, joint_vel = data["joint_pos"], data["joint_vel"]
  object_pose, command = data["object_pose"], data["command"]
  if int(lengths.sum()) != joint_pos.shape[0] or joint_pos.shape[1] != 22:
    raise SystemExit("unexpected reference.npz layout")
  link_pos_local = command[:, 22 + 6 : 22 + 6 + NUM_LINKS * 3].reshape(-1, NUM_LINKS, 3)

  wrist_pos = np.asarray(FIXED_WRIST_POSE_W[:3], dtype=np.float32)
  wrist_quat = np.asarray(FIXED_WRIST_POSE_W[3:], dtype=np.float32)
  wrist_quat = wrist_quat / np.linalg.norm(wrist_quat)

  args.out.mkdir(parents=True, exist_ok=True)
  starts = np.r_[0, np.cumsum(lengths)[:-1]]
  entries = []
  for clip_id, start, end in zip(clip_ids, starts, np.cumsum(lengths)):
    n = int(end - start)
    local_id = clip_id.split("/")[-1]
    wrist_pose_w = np.tile(
      np.concatenate((wrist_pos, wrist_quat)).astype(np.float32), (n, 1)
    )
    object_pose_w = pose_local_to_world(
      object_pose[start:end, :3], object_pose[start:end, 3:], wrist_pos, wrist_quat
    )
    link_pos_w = local_to_world(link_pos_local[start:end], wrist_pos, wrist_quat)
    payload = {
      "reference_qpos": joint_pos[start:end].astype(np.float32),
      "reference_qvel": joint_vel[start:end].astype(np.float32),
      "wrist_pose_w": wrist_pose_w,
      "object_pose_w": object_pose_w,
      "link_pos_w": link_pos_w.astype(np.float32),
    }
    path = args.out / f"{local_id}.npz"
    np.savez(path, **payload)
    entries.append(
      {"clip_id": local_id, "file": f"{local_id}.npz", "sha256": sha256(path)}
    )
    print(f"{clip_id}: {n} frames -> {path.name}")

  index = {
    "schema": "wuji_pen_spin_clips:1",
    "group_id": CLIP_GROUP_ID,
    "group_name": "Unitree Z5 pen spin (deploy bundle reference)",
    "frame_dt": FRAME_DT,
    "clip_count": len(entries),
    "clips": entries,
  }
  (args.out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
  print(f"catalog written: {args.out} ({len(entries)} clips, {int(lengths.sum())} frames)")
  print(f"link order ({NUM_LINKS}): {Z5_LINK_NAMES[0]} ... {Z5_LINK_NAMES[-1]}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
