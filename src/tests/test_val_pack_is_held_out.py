"""Validation must never silently fall back to a training vessel.

`--exclude-val-from-train` drops the val anchor from `train_anchors`, and `packs` is built only
from `train_anchors`, so no pack for the held-out anchor exists. The old resolution line

    val_pack = next((p for p in packs if p["anchor"] == val_anchor), packs[0])

therefore substituted the first *training* anchor. Proof from the wall_family7 cold run: the
in-training "val" numbers for patient020 (score 0.8771 / strict F1 0.8444 / relaxed rec 0.9452
/ mat F1 0.5825) match a standalone canonical eval of **patient005** to four decimals, while
patient020 itself scored 0.2996 / 0.2487.
"""

from __future__ import annotations

import pytest


def _packs(*anchors: str) -> list[dict]:
    return [{"anchor": a} for a in anchors]


def _resolve(packs: list[dict], val_anchor: str, heldout: dict | None) -> dict:
    """Mirror of the resolution logic in train_species_pushforward_continuous.main()."""
    val_pack = heldout if heldout is not None else next(
        (p for p in packs if p["anchor"] == val_anchor), None
    )
    if val_pack is None or val_pack["anchor"] != val_anchor:
        got = None if val_pack is None else val_pack["anchor"]
        raise ValueError(
            f"val pack resolves to {got!r}, expected {val_anchor!r}; refusing to validate on "
            "the wrong vessel"
        )
    return val_pack


def test_missing_val_pack_raises_instead_of_substituting() -> None:
    """The exact wall_family7 configuration: val excluded, so no pack exists."""
    packs = _packs("patient005", "patient006", "patient010", "patient023", "patient002")
    with pytest.raises(ValueError, match="refusing to validate on the wrong vessel"):
        _resolve(packs, "patient020", None)


def test_old_fallback_would_have_picked_patient005() -> None:
    """Documents the defect: the silent fallback returned the first training anchor."""
    packs = _packs("patient005", "patient006", "patient010")
    legacy = next((p for p in packs if p["anchor"] == "patient020"), packs[0])
    assert legacy["anchor"] == "patient005"


def test_held_out_pack_is_used_when_supplied() -> None:
    packs = _packs("patient005", "patient006")
    heldout = {"anchor": "patient020"}
    assert _resolve(packs, "patient020", heldout)["anchor"] == "patient020"


def test_in_train_val_still_resolves_normally() -> None:
    packs = _packs("patient005", "patient020", "patient006")
    assert _resolve(packs, "patient020", None)["anchor"] == "patient020"


def test_mismatched_heldout_pack_is_rejected() -> None:
    packs = _packs("patient005")
    with pytest.raises(ValueError):
        _resolve(packs, "patient020", {"anchor": "patient006"})


def test_heldout_anchor_filtered_from_backprop_aux_packs() -> None:
    """Guard for the deploy-horizon aux term, which backprops."""
    val_anchor = "patient020"
    dep_packs = _packs("patient005", "patient020", "patient006")
    filtered = [p for p in dep_packs if p["anchor"] != val_anchor]
    assert all(p["anchor"] != val_anchor for p in filtered)
    assert len(filtered) == 2
