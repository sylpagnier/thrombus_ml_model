"""Parse the raw COMSOL patient007 wall calibration export into a compact .npz.

The export is 876 wall nodes x (2 static coords + 41 expressions x 201 timesteps).
Column block order per timestep (from the header ``% Description`` line):

  0 x   1 y   2 u   3 v   4 p   5 sr  6 dsrx 7 dsry 8 mu 9 mu1 10 mu2
  11 Sat 12 Omega 13 kpa_chem 14 kpa_mech 15 k_pa 16 Gamma 17 step2t
  18 sr<lss 19 dsrx<sgt 20 M 21 Mas 22 Mat 23 rp 24 ap 25 apr 26 aps
  27 PT 28 th 29 at 30 fg 31 fi 32 d(M,t) 33 d(Mas,t) 34 d(Mat,t)
  35 J0_M 36 J0_Mas 37 J0_Mat 38 J0_rp 39 J0_ap 40 J0_th

Everything is COMSOL-native CGS (cm, plt/cm^2, plt/cm^3, 1/s, 1/(s*cm)).

Usage:  python scripts/parse_comsol_wall_export.py [--out outputs/comsol_p007_wall.npz]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

SRC = Path("data/reference_local/comsol_calibration/patient007_calibration_wall.txt")
NBLK = 41
KEEP = {
    "x": 0, "y": 1, "u": 2, "v": 3, "p": 4, "sr": 5, "dsrx": 6, "dsry": 7,
    "mu": 8, "mu1": 9, "Sat": 11, "step2t": 17, "gate_low": 18, "gate_sep": 19,
    "M": 20, "Mas": 21, "Mat": 22, "rp": 23, "ap": 24, "PT": 27, "th": 28,
    "fi": 31, "dMt": 32, "dMast": 33, "dMatt": 34,
    "J0_M": 35, "J0_Mas": 36, "J0_Mat": 37, "J0_th": 40,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default="outputs/comsol_p007_wall.npz")
    args = ap.parse_args()

    times: list[float] = []
    rows: list[np.ndarray] = []
    with open(args.src) as fh:
        for line in fh:
            if line.startswith("%"):
                if "@ t=" in line and not times:
                    times = [float(m) for m in re.findall(r"@ t=([-\d.eE+]+)", line)]
                continue
            rows.append(np.fromstring(line, sep=" "))
    arr = np.asarray(rows, dtype=np.float64)          # [N, 2 + 41*T]
    n, ncol = arr.shape
    nt = (ncol - 2) // NBLK
    assert 2 + NBLK * nt == ncol, (ncol, nt)
    blk = arr[:, 2:].reshape(n, nt, NBLK)             # [N, T, 41]

    t = np.asarray(times[:nt], dtype=np.float64) if len(times) >= nt else np.arange(nt) * 150.0
    # header repeats each expression per time; dedupe to unique block times
    if len(times) > nt:
        t = np.asarray(times, dtype=np.float64)[:nt * NBLK].reshape(nt, NBLK)[:, 0]

    out = {k: np.ascontiguousarray(blk[:, :, i].T) for k, i in KEEP.items()}  # [T, N]
    out["t"] = t
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"nodes={n} times={nt} t[0..3]={t[:4]} t[-1]={t[-1]}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
