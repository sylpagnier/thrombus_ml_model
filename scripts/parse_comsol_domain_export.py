"""Parse the raw COMSOL patient007 DOMAIN calibration export into a compact .npz.

51240 nodes x (2 static coords + 28 expressions x 4 timesteps).  Block order from the
header ``% Description``:

  0 x 1 y 2 u 3 v 4 p 5 sr 6 dsrx 7 dsry 8 mu 9 mu1 10 mu2 11 Omega 12 kpa_chem
  13 kpa_mech 14 k_pa 15 Gamma 16 step4 17 sr<lss 18 dsrx<sgt 19 rp 20 ap 21 apr
  22 aps 23 PT 24 th 25 at 26 fg 27 fi

COMSOL-native CGS.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

NBLK = 28
KEEP = {"x": 0, "y": 1, "u": 2, "v": 3, "p": 4, "sr": 5, "dsrx": 6, "dsry": 7,
        "mu": 8, "mu1": 9, "mu2": 10, "gate_low": 17, "gate_sep": 18, "rp": 19, "ap": 20}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/reference_local/comsol_calibration/patient007_calibration_domain.txt")
    ap.add_argument("--out", default="outputs/comsol_p007_domain.npz")
    args = ap.parse_args()
    times: list[float] = []
    rows = []
    with open(args.src) as fh:
        for line in fh:
            if line.startswith("%"):
                if "@ t=" in line and not times:
                    times = [float(m) for m in re.findall(r"@ t=([-\d.eE+]+)", line)]
                continue
            rows.append(np.fromstring(line, sep=" "))
    arr = np.asarray(rows, dtype=np.float64)
    n, ncol = arr.shape
    nt = (ncol - 2) // NBLK
    assert 2 + NBLK * nt == ncol, (ncol, nt)
    blk = arr[:, 2:].reshape(n, nt, NBLK)
    t = np.asarray(times, dtype=np.float64)[:nt * NBLK].reshape(nt, NBLK)[:, 0]
    out = {k: np.ascontiguousarray(blk[:, :, i].T) for k, i in KEEP.items()}
    out["t"] = t
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"nodes={n} times={nt} t={t}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
