"""Single source of truth for biochem species channel ordering.

WHY THIS FILE EXISTS
--------------------
Species channels are referenced in **several different coordinate systems** across
the codebase, and they are easy to confuse (humans *and* AI agents routinely grab
the wrong index). This module makes the ordering explicit, obvious, and
machine-checked so the wrong channel cannot be referenced silently.

THE THREE COORDINATE SYSTEMS (read this before indexing any species)
-------------------------------------------------------------------
There is one canonical species ordering. The *same* species appears at different
integer indices depending on which tensor you are slicing:

    state / label tensor ``y``  (16 channels, ``BIO_Y_SCHEMA``)
        col  0  u_nd          <- flow block (NOT a species)
        col  1  v_nd
        col  2  p_nd
        col  3  mu_eff_nd
        col  4  RP    \
        col  5  AP     |
        col  6  APR    |
        col  7  APS    |
        col  8  PT     |  bulk species (9)
        col  9  T      |
        col 10  AT     |
        col 11  FG     |
        col 12  FI    /     <- FI in FULL Y == 12
        col 13  M     \
        col 14  Mas    |  wall species (3)
        col 15  Mat   /     <- Mat in FULL Y == 15

    species block ``y[..., 4:16]`` (12 channels, flow stripped)
        idx  0  RP ... idx 8 FI ... idx 11 Mat
                            ^^ FI in BLOCK == 8, Mat in BLOCK == 11

    ``BulkSpecies`` enum (9 bulk only):  RP=0 ... FI=8
    ``WallSpecies`` enum (3 wall only):  M=0, Mas=1, Mat=2

So, depending on the tensor, "Mat" is index 15 (full y), 11 (species block) or 2
(wall enum). **Never hard-code these numbers.** Use the helpers below, e.g.::

    from src.utils import species_channels as sc

    fi_full   = sc.y_index("FI")          # -> 12   (column in y[..., :])
    fi_block  = sc.block_index("FI")      # ->  8   (column in y[..., 4:16])
    mat_full  = sc.y_index("Mat")         # -> 15
    mat_block = sc.block_index("Mat")     # -> 11
    sp        = y[..., sc.SPECIES_BLOCK]  # the 12-ch species block

This module validates at import time that it agrees with ``BulkSpecies``,
``WallSpecies``, ``BULK_SPECIES_ORDER`` (``src.config``) and ``BIO_Y_SCHEMA``
(``src.utils.channel_schema``). If any of those drift, importing fails loudly
rather than letting a silent off-by-N bug through.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from src.config import BULK_SPECIES_ORDER, BulkSpecies, WallSpecies


# --- Layout constants ---------------------------------------------------------
# Flow occupies the first FLOW_WIDTH channels of the full state/label tensor.
FLOW_WIDTH = 4
FLOW_NAMES: Tuple[str, ...] = ("u_nd", "v_nd", "p_nd", "mu_eff_nd")

# The species block is ``y[..., SPECIES_BLOCK_START : SPECIES_BLOCK_START + SPECIES_BLOCK_WIDTH]``.
SPECIES_BLOCK_START = FLOW_WIDTH          # 4
N_BULK = 9
N_WALL = 3
SPECIES_BLOCK_WIDTH = N_BULK + N_WALL     # 12
Y_WIDTH = FLOW_WIDTH + SPECIES_BLOCK_WIDTH  # 16

# Reusable slices (prefer these over literal slice(4, 16) etc.).
FLOW_SLICE = slice(0, FLOW_WIDTH)                                  # y[..., 0:4]
SPECIES_BLOCK = slice(SPECIES_BLOCK_START, Y_WIDTH)               # y[..., 4:16]
BULK_SLICE_IN_Y = slice(SPECIES_BLOCK_START, SPECIES_BLOCK_START + N_BULK)   # y[..., 4:13]
WALL_SLICE_IN_Y = slice(SPECIES_BLOCK_START + N_BULK, Y_WIDTH)              # y[..., 13:16]

GROUP_BULK = "bulk"
GROUP_WALL = "wall"


@dataclass(frozen=True)
class SpeciesChannel:
    """One species and its index in every coordinate system.

    Attributes
    ----------
    name:        canonical short token (e.g. ``"FI"``); matches the enums.
    full_name:   human-readable description.
    group:       ``"bulk"`` or ``"wall"``.
    y_index:     column in the full 16-ch state/label tensor (``BIO_Y_SCHEMA``).
    block_index: column in the 12-ch species block ``y[..., 4:16]``.
    group_index: index within its own enum (``BulkSpecies`` or ``WallSpecies``).
    """

    name: str
    full_name: str
    group: str
    y_index: int
    block_index: int
    group_index: int


# Canonical ordering (channels 4..15 of the full y tensor). DO NOT reorder.
_SPECIES: Tuple[SpeciesChannel, ...] = (
    SpeciesChannel("RP",  "Resting platelets",                      GROUP_BULK, 4,  0,  0),
    SpeciesChannel("AP",  "Activated platelets",                    GROUP_BULK, 5,  1,  1),
    SpeciesChannel("APR", "ADP agonist",                            GROUP_BULK, 6,  2,  2),
    SpeciesChannel("APS", "Thromboxane (TxA2) agonist",             GROUP_BULK, 7,  3,  3),
    SpeciesChannel("PT",  "Prothrombin",                            GROUP_BULK, 8,  4,  4),
    SpeciesChannel("T",   "Thrombin",                               GROUP_BULK, 9,  5,  5),
    SpeciesChannel("AT",  "Antithrombin",                           GROUP_BULK, 10, 6,  6),
    SpeciesChannel("FG",  "Fibrinogen",                             GROUP_BULK, 11, 7,  7),
    SpeciesChannel("FI",  "Fibrin (primary clot species)",          GROUP_BULK, 12, 8,  8),
    SpeciesChannel("M",   "Surface-bound resting platelets",        GROUP_WALL, 13, 9,  0),
    SpeciesChannel("Mas", "Surface-bound activated platelets",      GROUP_WALL, 14, 10, 1),
    SpeciesChannel("Mat", "Mature surface fibrin/clot (gelation)",  GROUP_WALL, 15, 11, 2),
)

SPECIES_NAMES: Tuple[str, ...] = tuple(s.name for s in _SPECIES)
BULK_NAMES: Tuple[str, ...] = tuple(s.name for s in _SPECIES if s.group == GROUP_BULK)
WALL_NAMES: Tuple[str, ...] = tuple(s.name for s in _SPECIES if s.group == GROUP_WALL)

_BY_NAME: Dict[str, SpeciesChannel] = {s.name: s for s in _SPECIES}
# Case-insensitive aliases for forgiving CLI / env parsing.
_ALIAS: Dict[str, str] = {
    "fibrin": "FI",
    "fi": "FI",
    "mature": "Mat",
    "mat": "Mat",
    "thrombin": "T",
    "th": "T",
    "t": "T",
    "prothrombin": "PT",
    "pt": "PT",
    "fibrinogen": "FG",
    "fg": "FG",
    "adp": "APR",
    "txa2": "APS",
    "antithrombin": "AT",
}


def _resolve_name(name: str) -> str:
    token = str(name).strip()
    if token in _BY_NAME:
        return token
    key = token.lower()
    if key in _ALIAS:
        return _ALIAS[key]
    # exact case-insensitive match against canonical names
    for canon in SPECIES_NAMES:
        if canon.lower() == key:
            return canon
    raise KeyError(
        f"Unknown species name {name!r}. Known: {', '.join(SPECIES_NAMES)}."
    )


def get(name: str) -> SpeciesChannel:
    """Return the :class:`SpeciesChannel` record for ``name`` (alias-tolerant)."""
    return _BY_NAME[_resolve_name(name)]


def y_index(name: str) -> int:
    """Column of ``name`` in the FULL 16-ch state/label tensor (``BIO_Y_SCHEMA``)."""
    return get(name).y_index


def block_index(name: str) -> int:
    """Column of ``name`` in the 12-ch species block ``y[..., 4:16]``."""
    return get(name).block_index


def block_index_to_y_index(i: int) -> int:
    """Map a species-block index (0..11) to a full-y column (4..15)."""
    i = int(i)
    if not 0 <= i < SPECIES_BLOCK_WIDTH:
        raise IndexError(f"block index {i} out of range [0, {SPECIES_BLOCK_WIDTH}).")
    return SPECIES_BLOCK_START + i


def y_index_to_block_index(i: int) -> int:
    """Map a full-y column (4..15) to a species-block index (0..11)."""
    i = int(i)
    if not SPECIES_BLOCK_START <= i < Y_WIDTH:
        raise IndexError(
            f"y index {i} is not a species column; species occupy "
            f"[{SPECIES_BLOCK_START}, {Y_WIDTH})."
        )
    return i - SPECIES_BLOCK_START


def name_at_block_index(i: int) -> str:
    """Species short name at species-block index ``i`` (0..11)."""
    return _SPECIES[int(i)].name  # block_index == positional index by construction


def name_at_y_index(i: int) -> str:
    """Species short name at full-y column ``i`` (4..15)."""
    return name_at_block_index(y_index_to_block_index(i))


def block_indices(names: Sequence[str]) -> List[int]:
    """Species-block indices for an ordered list of names."""
    return [block_index(n) for n in names]


def y_indices(names: Sequence[str]) -> List[int]:
    """Full-y columns for an ordered list of names."""
    return [y_index(n) for n in names]


def describe() -> str:
    """Human-readable table of every species across all coordinate systems."""
    rows = [
        f"{'name':<5} {'group':<5} {'y_idx':>5} {'block':>5} {'enum':>4}  full_name",
        "-" * 64,
    ]
    for s in _SPECIES:
        rows.append(
            f"{s.name:<5} {s.group:<5} {s.y_index:>5} {s.block_index:>5} "
            f"{s.group_index:>4}  {s.full_name}"
        )
    return "\n".join(rows)


# --- Import-time consistency validation ---------------------------------------
def _validate() -> None:
    """Fail loudly if this registry drifts from the enums / channel schema."""
    # 1) Positional invariant: block_index must equal list position.
    for pos, s in enumerate(_SPECIES):
        if s.block_index != pos:
            raise AssertionError(
                f"species_channels: {s.name} block_index {s.block_index} != position {pos}"
            )
        if s.y_index != SPECIES_BLOCK_START + pos:
            raise AssertionError(
                f"species_channels: {s.name} y_index {s.y_index} != {SPECIES_BLOCK_START + pos}"
            )

    # 2) Agreement with BulkSpecies / WallSpecies enums (src.config).
    if tuple(b.name for b in BULK_SPECIES_ORDER) != BULK_NAMES:
        raise AssertionError(
            "species_channels: BULK names "
            f"{BULK_NAMES} != BULK_SPECIES_ORDER {tuple(b.name for b in BULK_SPECIES_ORDER)}"
        )
    for s in _SPECIES:
        if s.group == GROUP_BULK:
            if int(BulkSpecies[s.name]) != s.group_index:
                raise AssertionError(
                    f"species_channels: BulkSpecies.{s.name}={int(BulkSpecies[s.name])} "
                    f"!= group_index {s.group_index}"
                )
        else:
            if int(WallSpecies[s.name]) != s.group_index:
                raise AssertionError(
                    f"species_channels: WallSpecies.{s.name}={int(WallSpecies[s.name])} "
                    f"!= group_index {s.group_index}"
                )

    # 3) Agreement with the persisted BIO_Y_SCHEMA channel names.
    #    Imported lazily to avoid any import-order surprises.
    from src.utils.channel_schema import BIO_Y_SCHEMA, Y_SCHEMAS

    bio_channels = Y_SCHEMAS[BIO_Y_SCHEMA].channels
    if len(bio_channels) != Y_WIDTH:
        raise AssertionError(
            f"species_channels: BIO_Y_SCHEMA width {len(bio_channels)} != {Y_WIDTH}"
        )

    def _strip(ch: str) -> str:
        return ch.split("_log1p_nd")[0].split("_nd")[0]

    for s in _SPECIES:
        token = _strip(bio_channels[s.y_index])
        if token != s.name:
            raise AssertionError(
                f"species_channels: BIO_Y_SCHEMA[{s.y_index}]={bio_channels[s.y_index]!r} "
                f"(={token!r}) does not match species {s.name!r}."
            )


_validate()
