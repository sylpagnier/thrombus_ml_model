"""Lock the canonical species channel ordering and cross-coordinate consistency.

These tests are the safety net behind ``src/utils/species_channels.py``: if any
of the parallel definitions (``BulkSpecies``/``WallSpecies`` enums, the
``BIO_Y_SCHEMA`` channel list, or the ``biochem_species_scope`` constants) drift
out of agreement, CI fails here instead of silently reading the wrong channel.
"""

from __future__ import annotations

import pytest

from src.config import BULK_SPECIES_ORDER, BulkSpecies, WallSpecies
from src.utils import species_channels as sc
from src.utils.channel_schema import BIO_Y_SCHEMA, Y_SCHEMAS


def test_canonical_order_is_frozen():
    assert sc.SPECIES_NAMES == (
        "RP", "AP", "APR", "APS", "PT", "T", "AT", "FG", "FI", "M", "Mas", "Mat",
    )
    assert sc.BULK_NAMES == ("RP", "AP", "APR", "APS", "PT", "T", "AT", "FG", "FI")
    assert sc.WALL_NAMES == ("M", "Mas", "Mat")


def test_layout_constants():
    assert sc.FLOW_WIDTH == 4
    assert sc.SPECIES_BLOCK_START == 4
    assert sc.SPECIES_BLOCK_WIDTH == 12
    assert sc.Y_WIDTH == 16
    assert sc.SPECIES_BLOCK == slice(4, 16)
    assert sc.BULK_SLICE_IN_Y == slice(4, 13)
    assert sc.WALL_SLICE_IN_Y == slice(13, 16)


@pytest.mark.parametrize(
    "name,y_idx,block_idx",
    [
        ("RP", 4, 0),
        ("FI", 12, 8),
        ("FG", 11, 7),
        ("T", 9, 5),
        ("PT", 8, 4),
        ("M", 13, 9),
        ("Mat", 15, 11),
    ],
)
def test_known_indices(name, y_idx, block_idx):
    assert sc.y_index(name) == y_idx
    assert sc.block_index(name) == block_idx


def test_block_full_roundtrip():
    for name in sc.SPECIES_NAMES:
        b = sc.block_index(name)
        y = sc.y_index(name)
        assert sc.block_index_to_y_index(b) == y
        assert sc.y_index_to_block_index(y) == b
        assert sc.name_at_block_index(b) == name
        assert sc.name_at_y_index(y) == name


def test_aliases_resolve():
    assert sc.get("fibrin").name == "FI"
    assert sc.get("thrombin").name == "T"
    assert sc.get("MAT").name == "Mat"
    with pytest.raises(KeyError):
        sc.get("nope")


def test_agrees_with_enums():
    assert tuple(b.name for b in BULK_SPECIES_ORDER) == sc.BULK_NAMES
    for name in sc.BULK_NAMES:
        assert int(BulkSpecies[name]) == sc.get(name).group_index
    for name in sc.WALL_NAMES:
        assert int(WallSpecies[name]) == sc.get(name).group_index


def test_agrees_with_bio_y_schema():
    channels = Y_SCHEMAS[BIO_Y_SCHEMA].channels
    assert len(channels) == sc.Y_WIDTH
    for s_name in sc.SPECIES_NAMES:
        col = sc.y_index(s_name)
        token = channels[col].split("_log1p_nd")[0].split("_nd")[0]
        assert token == s_name


def test_biochem_species_scope_constants_match_registry():
    from src.training import biochem_species_scope as scope

    assert scope.FI_CHANNEL == sc.block_index("FI")
    assert scope.MAT_CHANNEL == sc.block_index("Mat")
    assert scope.THROMBIN_CHANNEL == sc.block_index("T")
    assert scope.PROTHROMBIN_CHANNEL == sc.block_index("PT")
    assert scope.FIBRINOGEN_CHANNEL == sc.block_index("FG")
    for i in range(sc.SPECIES_BLOCK_WIDTH):
        assert scope.BULK_CHANNEL_NAMES[i] == sc.name_at_block_index(i)
