"""Console / tqdm policy for biochem training (Windows PowerShell-safe)."""

from __future__ import annotations

import os
import sys
from typing import Any, Iterable, Iterator, TypeVar

T = TypeVar("T")

_ASCII_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("\u2192", "->"),
    ("\u2026", "..."),
    ("\u21b3", "-> "),
    ("\u2013", "-"),
    ("\u2014", "-"),
    ("\u03bc", "mu"),
    ("\u0394", "d"),
)


def sanitize_console_text(text: str) -> str:
    """Replace common Unicode console glyphs with ASCII (PowerShell cp437/UTF-8 safe)."""
    out = text
    for old, new in _ASCII_REPLACEMENTS:
        out = out.replace(old, new)
    return out


def configure_biochem_console() -> None:
    """Idempotent: prefer UTF-8 on Windows; does not wrap stdout (tqdm needs a TTY)."""
    if os.environ.get("BIOCHEM_CONSOLE_CONFIGURED") == "1":
        return
    os.environ["BIOCHEM_CONSOLE_CONFIGURED"] = "1"
    if sys.platform == "win32":
        os.environ.setdefault("PYTHONUTF8", "1")
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def biochem_quiet_logs() -> bool:
    return os.environ.get("BIOCHEM_QUIET", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def biochem_tqdm_enabled() -> bool:
    if biochem_quiet_logs():
        return False
    raw = os.environ.get("BIOCHEM_TQDM", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    try:
        return sys.stderr.isatty()
    except Exception:
        return True


def biochem_tqdm_compact() -> bool:
    raw = os.environ.get("BIOCHEM_TQDM_COMPACT", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return sys.platform == "win32"


def biochem_tqdm_refresh_stride() -> int:
    raw = os.environ.get("BIOCHEM_TQDM_REFRESH_STRIDE", "").strip()
    if raw:
        return max(1, int(raw))
    return 2 if biochem_tqdm_compact() else 1


def biochem_tqdm_mininterval() -> float:
    raw = os.environ.get("BIOCHEM_TQDM_MININTERVAL", "").strip()
    if raw:
        return max(0.1, float(raw))
    return 1.0 if biochem_tqdm_compact() else 0.5


class _NoOpTqdm:
    """Iterable wrapper when tqdm is disabled (quiet / non-TTY)."""

    def __init__(self, iterable: Iterable[T], *, desc: str = "") -> None:
        self._iter = iter(iterable)
        self.desc = desc

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        return next(self._iter)

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        return None

    def set_postfix_str(self, *args: Any, **kwargs: Any) -> None:
        return None

    def refresh(self) -> None:
        return None

    def close(self) -> None:
        return None


def biochem_tqdm(iterable: Iterable[T], *, desc: str, total: int | None = None) -> Any:
    """Single-line ASCII progress bar (avoid Unicode blocks; throttle refresh on Windows)."""
    if not biochem_tqdm_enabled():
        return _NoOpTqdm(iterable, desc=desc)

    from tqdm import tqdm

    ncols_raw = os.environ.get("BIOCHEM_TQDM_NCOLS", "").strip()
    ncols = int(ncols_raw) if ncols_raw else 96
    bar_format = (
        os.environ.get("BIOCHEM_TQDM_BAR_FORMAT", "").strip()
        or "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
    )
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        ascii=True,
        dynamic_ncols=False,
        ncols=ncols,
        mininterval=biochem_tqdm_mininterval(),
        maxinterval=5.0,
        file=sys.stderr,
        leave=True,
        bar_format=bar_format,
    )


def format_biochem_tqdm_postfix(
    *,
    ema_metrics: dict[str, float],
    metrics: dict[str, float],
    batch_dt: float,
    anchor_supervised_batches: int,
    pseudo_supervised_batches: int,
    total_batches: int,
    current_phys_ceiling: float,
    w_mu_log_ep: float,
    compact: bool | None = None,
) -> str:
    compact = biochem_tqdm_compact() if compact is None else compact
    if compact:
        parts = [
            f"L={ema_metrics['L_tot']:.2e}",
            f"bio={ema_metrics['L_Data_Bio']:.2e}",
            f"TF={metrics['TF_eff']:.2f}",
            f"{batch_dt:.1f}s",
            f"A={anchor_supervised_batches}/{total_batches}",
        ]
        if pseudo_supervised_batches:
            parts.append(f"P={pseudo_supervised_batches}/{total_batches}")
        return " ".join(parts)
    parts = [
        f"L_tot={ema_metrics['L_tot']:.2e}",
        f"L_Kine={ema_metrics['L_Data_Kine']:.2e}",
        f"L_Bio={ema_metrics['L_Data_Bio']:.2e}",
        f"L_ADR_F={ema_metrics['L_ADR_F']:.2e}",
        f"TF={metrics['TF_eff']:.2f}",
        f"ODE={int(metrics.get('ODE_Evals', 0))}",
        f"t={batch_dt:.2f}s",
        f"A={anchor_supervised_batches}/{total_batches}",
        f"P={pseudo_supervised_batches}/{total_batches}",
    ]
    if "L_MuSI_aux" in ema_metrics:
        parts.append(f"L_MuSI={ema_metrics['L_MuSI_aux']:.2e}")
    if w_mu_log_ep > 0.0 and "L_MuLog_aux" in ema_metrics:
        parts.append(f"L_MuLog={ema_metrics['L_MuLog_aux']:.2e}")
    return ", ".join(parts)
