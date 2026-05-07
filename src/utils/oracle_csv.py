from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_comsol_oracle_table(path: Path, *, expected_cols: int) -> pd.DataFrame:
    """Read COMSOL-exported oracle tables robustly (comments, whitespace/comma, junk rows)."""
    try:
        df = pd.read_csv(path, comment="%", sep=r"\s+", header=None)
        if df.shape[1] < expected_cols:
            df = pd.read_csv(path, comment="%", sep=",", header=None)
    except Exception:
        df = pd.read_csv(path, comment="%", sep=",", header=None)

    df = df.iloc[:, :expected_cols].copy()
    df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    return df

