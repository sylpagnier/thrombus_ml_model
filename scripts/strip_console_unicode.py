"""Replace emoji / special symbols in console prints with ASCII (PowerShell-safe)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

REPLACEMENTS: list[tuple[str, str]] = [
    ("🆕", "[NEW] "),
    ("⚡", ""),
    ("ℹ️", "[i] "),
    ("✅", "[OK] "),
    ("❌", "[ERR] "),
    ("🔥", ""),
    ("💾", "[save] "),
    ("🧪", ""),
    ("🚀", ""),
    ("👩‍🏫", ""),
    ("📐", ""),
    ("📂", ""),
    ("🎯", "[scope] "),
    ("❄️", ""),
    ("⚠️", "[WARN] "),
    ("🩺", ""),
    ("🧠", ""),
    ("🧭", ""),
    ("🧬", ""),
    ("🛑", ""),
    ("💉", ""),
    ("📊", ""),
    ("📋", "[log] "),
    ("📦", ""),
    ("🔮", ""),
    ("🎨", ""),
    ("⏳", ""),
    ("✂️", ""),
    ("↳", "-> "),
    ("→", "->"),
    ("⚖️", ""),
    ("⏭️", "[skip] "),
    ("⏱️", ""),
    ("🧊", ""),
    ("🔓", ""),
    ("🧾", ""),
    ("🧩", ""),
    ("🔁", ""),
    ("🔧", ""),
    ("🔎", ""),
    ("🔄", ""),
    ("🫧", ""),
    ("🔐", ""),
    ("🔒", ""),
    ("💥", "[ERR] "),
    ("🖥️", ""),
    ("🕸️", ""),
    ("♻️", ""),
]

TARGETS = [
    REPO / "src/training/train_biochem_corrector.py",
    REPO / "src/evaluation/visualize_pipeline.py",
]


def strip_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    if text != original:
        path.write_text(text, encoding="utf-8", newline="\n")
    return sum(1 for a, b in zip(original, text) if a != b)


def _resolve(path: Path) -> Path:
    p = path if path.is_absolute() else (REPO / path)
    return p.resolve()


def main() -> None:
    paths = TARGETS
    if len(sys.argv) > 1:
        paths = [_resolve(Path(p)) for p in sys.argv[1:]]
    for p in paths:
        if p.is_file():
            try:
                rel = p.relative_to(REPO)
            except ValueError:
                rel = p
            print(f"stripped {rel}")
            strip_file(p)


if __name__ == "__main__":
    main()
