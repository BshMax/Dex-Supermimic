# SPDX-License-Identifier: Apache-2.0
"""Regenerate the Z5 reference surface-contact table (z5_surface.npz).

The private fork trained with two extra reward terms, ``reference_contact_axial``
and ``reference_contact_surface``, fed by ``data/reference_contacts/z5_surface.npz``
(checkpoint config ``reference_contact_path``). That file ships only with the
fork, so this tool reconstructs a physically equivalent table from the deploy
bundle's ``reference.npz`` (pen pose + 21 link-site positions, palm frame) and
the known pen geometry (cylinder r=7.45 mm, half length 125 mm):

per reference frame i and link site j:
    t        axial coordinate of the closest point on the pen axis
    depth    signed radial distance of the site to the pen surface (|w| - r)
    normal   radial unit vector at that closest point (pen frame -> palm frame)
    conf     exp(-(depth/4 mm)^2): 1 when the site rides the pen surface

``reference_contact_signature`` in the checkpoint hashes the FORK's exact bytes,
which used an unknown (private) algorithm, so a byte-exact match is impossible;
the tool prints both signatures for the record. What matters physically is that
the masked (conf >= 0.5) sites are exactly the load-bearing finger contacts of
each reference frame.

Run from the Dex-Supermimic checkout:

    uv run python -m dex_supermimic.tasks.z5_pen_spin.tooling.generate_surface_contacts \
        --reference .../generated/deploy/z5-pen-spin/reference.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PEN_RADIUS_M = 0.00745
PEN_HALF_LENGTH_M = 0.125
CONF_SIGMA_M = 0.004
NUM_LINKS = 21
DEFAULT_OUT = (
  Path(__file__).resolve().parents[1] / "data" / "reference_contacts" / "z5_surface.npz"
)


def quat_to_mat_wxyz(q: np.ndarray) -> np.ndarray:
  w, x, y, z = (q[..., i] for i in range(4))
  return np.stack(
    (
      np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), -1),
      np.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), -1),
      np.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), -1),
    ),
    -2,
  )


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--reference", type=Path, required=True)
  parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
  args = parser.parse_args()

  data = np.load(args.reference, allow_pickle=False)
  pose = data["object_pose"].astype(np.float64)          # (N,7) palm frame, wxyz
  links = data["command"][:, 28:28 + NUM_LINKS * 3].reshape(-1, NUM_LINKS, 3)
  pos, rot = pose[:, :3], quat_to_mat_wxyz(pose[:, 3:] / np.linalg.norm(pose[:, 3:], axis=-1, keepdims=True))
  axis = rot[..., 0]                                     # pen +X shaft axis, palm frame

  v = links - pos[:, None, :]                            # (N,21,3)
  t = np.clip((v * axis[:, None, :]).sum(-1), -PEN_HALF_LENGTH_M, PEN_HALF_LENGTH_M)
  closest = pos[:, None, :] + t[..., None] * axis[:, None, :]
  w = links - closest
  radial = np.linalg.norm(w, axis=-1)
  normal = w / np.maximum(radial, 1e-9)[..., None]
  depth = radial - PEN_RADIUS_M
  conf = np.exp(-((depth / CONF_SIGMA_M) ** 2))

  args.out.parent.mkdir(parents=True, exist_ok=True)
  payload = {
    "schema": np.array("dex_supermimic_z5_surface:1"),
    "pen_radius_m": np.float32(PEN_RADIUS_M),
    "pen_half_length_m": np.float32(PEN_HALF_LENGTH_M),
    "axial": t.astype(np.float32),
    "depth": depth.astype(np.float32),
    "normal": normal.astype(np.float32),
    "confidence": conf.astype(np.float32),
  }
  np.savez(args.out, **payload)

  digest = hashlib.sha256()
  for key in ("axial", "depth", "normal", "confidence"):
    digest.update(payload[key].tobytes())
  mine = digest.hexdigest()
  print(f"wrote {args.out}: frames={t.shape[0]} links={NUM_LINKS}")
  print(f"  masked (conf>=0.5) site-frames: {int((conf >= 0.5).sum())} "
        f"({(conf >= 0.5).mean() * 100:.1f}%), mean contacts/frame {(conf >= 0.5).sum(1).mean():.2f}")
  print(f"  regenerated signature {mine}")
  print("  checkpoint signature  6763a65ebbb1fc10ceed5e41fdc071d3c76d0d8487e5bbf5c443420dfaa5a994")
  print("  (byte-exact match impossible: the fork's generator is private; the table is")
  print("   a physically equivalent reconstruction, see module docstring)")
  (args.out.with_suffix(".json")).write_text(
    json.dumps({"schema": "dex_supermimic_z5_surface:1", "sha256": mine}, indent=2) + "\n"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
