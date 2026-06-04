from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch

from src.config import BiochemNodeFeat, NodeFeat


SCHEMA_VERSION = 1

KINE_Y_SCHEMA = "kine_v1_5ch"
BIO_Y_SCHEMA = "biochem_v1_16ch"
KINE_X_SCHEMA = "kine_x_v1_18ch"
BIO_X_SCHEMA = "biochem_x_v1_15ch"


def coerce_graph_schema_token(schema) -> str:
    """Normalize ``x_schema`` / ``y_schema`` from PyG ``DataBatch`` (often length-1 lists)."""
    if schema is None:
        return ""
    if isinstance(schema, (list, tuple)):
        if len(schema) == 0:
            return ""
        first = str(schema[0])
        if all(str(s) == first for s in schema):
            return first
        return first
    return str(schema)


def normalize_graph_schema_attrs(data) -> None:
    """In-place fix for batched or legacy schema metadata on a graph or batch."""
    if hasattr(data, "x_schema"):
        data.x_schema = coerce_graph_schema_token(getattr(data, "x_schema", None))
    if hasattr(data, "y_schema"):
        data.y_schema = coerce_graph_schema_token(getattr(data, "y_schema", None))
    if hasattr(data, "x_biochem_schema"):
        data.x_biochem_schema = coerce_graph_schema_token(getattr(data, "x_biochem_schema", None))


@dataclass(frozen=True)
class ChannelSchema:
    name: str
    channels: Tuple[str, ...]

    @property
    def width(self) -> int:
        return len(self.channels)

    @property
    def encoded_names(self) -> str:
        return ",".join(self.channels)


Y_SCHEMAS: Dict[str, ChannelSchema] = {
    KINE_Y_SCHEMA: ChannelSchema(
        name=KINE_Y_SCHEMA,
        channels=("u_nd", "v_nd", "p_nd", "mu_eff_nd", "wss_nd"),
    ),
    BIO_Y_SCHEMA: ChannelSchema(
        name=BIO_Y_SCHEMA,
        channels=(
            "u_nd",
            "v_nd",
            "p_nd",
            "mu_eff_nd",
            "RP_log1p_nd",
            "AP_log1p_nd",
            "APR_log1p_nd",
            "APS_log1p_nd",
            "PT_log1p_nd",
            "T_log1p_nd",
            "AT_log1p_nd",
            "FG_log1p_nd",
            "FI_log1p_nd",
            "M_log1p_nd",
            "Mas_log1p_nd",
            "Mat_log1p_nd",
        ),
    ),
}

X_SCHEMAS: Dict[str, ChannelSchema] = {
    KINE_X_SCHEMA: ChannelSchema(
        name=KINE_X_SCHEMA,
        channels=(
            "x_nd",
            "y_nd",
            "sdf_nd",
            "shear_potential",
            "wall_normal_x",
            "wall_normal_y",
            "node_type_0",
            "node_type_1",
            "node_type_2",
            "node_type_3",
            "rheology_flag",
            "u_prior",
            "v_prior",
            "mu_prior_nd",
            "wss_prior_nd",
            "width_nd",
            "width_d1",
            "width_d2",
        ),
    ),
    BIO_X_SCHEMA: ChannelSchema(
        name=BIO_X_SCHEMA,
        channels=(
            "x_nd",
            "y_nd",
            "sdf_nd",
            "wall_normal_x",
            "wall_normal_y",
            "mask_inlet",
            "mask_outlet",
            "mask_wall",
            "u_bc",
            "v_bc",
            "p_bc",
            "uv_mask",
            "p_mask",
            "mu_bc_nd",
            "mu_mask",
        ),
    ),
}


def _ones_valid_mask_like(y: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(y, dtype=torch.bool)


def build_y_valid_mask(y: torch.Tensor, y_schema: str, mask_wall: Optional[torch.Tensor]) -> torch.Tensor:
    valid = _ones_valid_mask_like(y)
    if y_schema != KINE_Y_SCHEMA:
        return valid
    if mask_wall is None:
        return valid
    wall = mask_wall.view(-1).bool()
    if y.dim() == 2 and y.shape[0] == wall.shape[0]:
        valid[:, 4] = wall
    elif y.dim() == 3 and y.shape[1] == wall.shape[0]:
        valid[:, :, 4] = wall.unsqueeze(0).expand(y.shape[0], -1)
    return valid


def attach_channel_metadata(
    data,
    *,
    x_schema: str,
    y_schema: str,
    mask_wall: Optional[torch.Tensor] = None,
):
    if x_schema not in X_SCHEMAS:
        raise ValueError(f"Unknown x schema: {x_schema}")
    if y_schema not in Y_SCHEMAS:
        raise ValueError(f"Unknown y schema: {y_schema}")
    x_def = X_SCHEMAS[x_schema]
    y_def = Y_SCHEMAS[y_schema]

    if data.x.shape[-1] != x_def.width:
        raise ValueError(f"x width {data.x.shape[-1]} != expected {x_def.width} for {x_schema}")
    if data.y.shape[-1] != y_def.width:
        raise ValueError(f"y width {data.y.shape[-1]} != expected {y_def.width} for {y_schema}")

    data.channel_schema_version = torch.tensor([SCHEMA_VERSION], dtype=torch.int64)
    data.x_schema = x_schema
    data.y_schema = y_schema
    data.x_channel_names = x_def.encoded_names
    data.y_channel_names = y_def.encoded_names
    data.y_valid_mask = build_y_valid_mask(data.y, y_schema=y_schema, mask_wall=mask_wall)
    return data


def infer_missing_schema(data, phase_hint: Optional[str] = None):
    normalize_graph_schema_attrs(data)
    if getattr(data, "x_schema", None) and getattr(data, "y_schema", None):
        return data

    xw = int(data.x.shape[-1]) if hasattr(data, "x") and data.x is not None else -1
    yw = int(data.y.shape[-1]) if hasattr(data, "y") and data.y is not None else -1
    hint = (phase_hint or "").lower()

    if yw == 16:
        y_schema = BIO_Y_SCHEMA
    elif yw == 5:
        y_schema = KINE_Y_SCHEMA
    elif "biochem" in hint:
        y_schema = BIO_Y_SCHEMA
    else:
        y_schema = KINE_Y_SCHEMA

    if xw == 15:
        x_schema = BIO_X_SCHEMA
    elif xw == 18:
        x_schema = KINE_X_SCHEMA
    elif "biochem" in hint:
        x_schema = BIO_X_SCHEMA
    else:
        x_schema = KINE_X_SCHEMA

    wall = getattr(data, "mask_wall", None)
    return attach_channel_metadata(data, x_schema=x_schema, y_schema=y_schema, mask_wall=wall)


def assert_graph_schema(data, expected_y_schema: Optional[Iterable[str]] = None):
    if not hasattr(data, "x") or not hasattr(data, "y"):
        raise ValueError("Graph must include x and y tensors.")
    if not getattr(data, "x_schema", None) or not getattr(data, "y_schema", None):
        raise ValueError("Graph missing x_schema/y_schema metadata.")
    if data.x_schema not in X_SCHEMAS:
        raise ValueError(f"Unknown x_schema '{data.x_schema}'.")
    if data.y_schema not in Y_SCHEMAS:
        raise ValueError(f"Unknown y_schema '{data.y_schema}'.")

    x_def = X_SCHEMAS[data.x_schema]
    y_def = Y_SCHEMAS[data.y_schema]
    if int(data.x.shape[-1]) != x_def.width:
        raise ValueError(f"x width {int(data.x.shape[-1])} != expected {x_def.width} for {data.x_schema}.")
    if int(data.y.shape[-1]) != y_def.width:
        raise ValueError(f"y width {int(data.y.shape[-1])} != expected {y_def.width} for {data.y_schema}.")

    if expected_y_schema is not None and data.y_schema not in set(expected_y_schema):
        raise ValueError(f"Unexpected y_schema '{data.y_schema}', expected one of {tuple(expected_y_schema)}.")

    if not hasattr(data, "y_valid_mask") or data.y_valid_mask is None:
        data.y_valid_mask = build_y_valid_mask(data.y, data.y_schema, getattr(data, "mask_wall", None))
    elif tuple(data.y_valid_mask.shape) != tuple(data.y.shape):
        raise ValueError("y_valid_mask shape must match y shape.")


def migrate_tensor_last_dim(
    t: torch.Tensor,
    *,
    target_width: int,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Pad/trim the last dim to a target width (behavior-preserving when widths already match)."""
    if int(t.shape[-1]) == int(target_width):
        return t
    if int(t.shape[-1]) > int(target_width):
        return t[..., :target_width].contiguous()
    pad = torch.full(
        (*t.shape[:-1], int(target_width) - int(t.shape[-1])),
        float(fill_value),
        device=t.device,
        dtype=t.dtype,
    )
    return torch.cat([t, pad], dim=-1).contiguous()


def biochem_encoder_x(data) -> torch.Tensor:
    """Biochem-model node features (15ch ``BIO_X_SCHEMA``), never the kinematics ``data.x`` layout."""
    if hasattr(data, "x_biochem") and data.x_biochem is not None:
        xb = data.x_biochem
        if int(xb.shape[-1]) != X_SCHEMAS[BIO_X_SCHEMA].width:
            raise ValueError(
                f"x_biochem width {int(xb.shape[-1])} != {X_SCHEMAS[BIO_X_SCHEMA].width} for {BIO_X_SCHEMA}."
            )
        return xb
    schema = coerce_graph_schema_token(getattr(data, "x_schema", None))
    if schema == BIO_X_SCHEMA:
        return data.x
    if schema == KINE_X_SCHEMA:
        raise ValueError(
            "Graph has kinematics x_schema on data.x but no x_biochem; "
            "re-run PatientDataExtractor or migrate anchor graphs."
        )
    raise ValueError(
        f"Cannot resolve biochem encoder features (x_schema={schema!r}, "
        f"x.shape={tuple(data.x.shape)}, x_biochem missing)."
    )


def attach_patient_anchor_graph_metadata(data, *, mask_wall: Optional[torch.Tensor] = None):
    """Attach schemas for anchor graphs: ``data.x`` = kine 18ch, ``data.x_biochem`` = biochem 15ch."""
    if not hasattr(data, "x_biochem") or data.x_biochem is None:
        raise ValueError("attach_patient_anchor_graph_metadata requires data.x_biochem.")
    attach_channel_metadata(
        data,
        x_schema=KINE_X_SCHEMA,
        y_schema=BIO_Y_SCHEMA,
        mask_wall=mask_wall,
    )
    data.x_biochem_schema = BIO_X_SCHEMA
    assert_anchor_dual_x_aligned(data)
    return data


def attach_biochem_synthetic_graph_metadata(data, *, mask_wall: Optional[torch.Tensor] = None):
    """Dual-x biochem synthetic graphs: kine ``data.x`` + biochem ``data.x_biochem``."""
    if not hasattr(data, "x_biochem") or data.x_biochem is None:
        raise ValueError("attach_biochem_synthetic_graph_metadata requires data.x_biochem.")
    attach_channel_metadata(
        data,
        x_schema=KINE_X_SCHEMA,
        y_schema=BIO_Y_SCHEMA,
        mask_wall=mask_wall,
    )
    data.x_biochem_schema = BIO_X_SCHEMA
    assert_anchor_dual_x_aligned(data)
    return data


def assert_anchor_dual_x_aligned(data, *, atol: float = 1e-5) -> None:
    """Guard: shared geometry channels must agree between kine and biochem tensors."""
    xk = data.x
    xb = biochem_encoder_x(data)
    if int(xk.shape[0]) != int(xb.shape[0]):
        raise ValueError(f"node count mismatch: x {xk.shape[0]} vs x_biochem {xb.shape[0]}")
    for sl_k, sl_b in (
        (NodeFeat.XY, BiochemNodeFeat.XY),
        (NodeFeat.SDF, BiochemNodeFeat.SDF),
    ):
        if not torch.allclose(xk[:, sl_k], xb[:, sl_b], atol=atol, rtol=0.0):
            raise ValueError(f"anchor graph mismatch on slice {sl_k} vs {sl_b}")
    if not torch.allclose(
        xk[:, NodeFeat.WALL_NORMAL],
        xb[:, BiochemNodeFeat.WALL_NORMAL],
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError(
            "anchor graph wall_normal mismatch between data.x (kine layout) and data.x_biochem."
        )
    if getattr(data, "x_schema", None) != KINE_X_SCHEMA:
        raise ValueError(f"expected data.x_schema={KINE_X_SCHEMA!r}, got {getattr(data, 'x_schema', None)!r}")
    if getattr(data, "x_biochem_schema", None) != BIO_X_SCHEMA:
        raise ValueError(
            f"expected data.x_biochem_schema={BIO_X_SCHEMA!r}, got {getattr(data, 'x_biochem_schema', None)!r}"
        )


def migrate_graph_schema(
    data,
    *,
    x_schema: str,
    y_schema: str,
    fill_value: float = 0.0,
):
    """Opt-in migration: pad/trim x/y to match schemas, then attach metadata."""
    if x_schema not in X_SCHEMAS:
        raise ValueError(f"Unknown x schema: {x_schema}")
    if y_schema not in Y_SCHEMAS:
        raise ValueError(f"Unknown y schema: {y_schema}")
    x_def = X_SCHEMAS[x_schema]
    y_def = Y_SCHEMAS[y_schema]
    data.x = migrate_tensor_last_dim(data.x, target_width=x_def.width, fill_value=fill_value)
    data.y = migrate_tensor_last_dim(data.y, target_width=y_def.width, fill_value=fill_value)
    return attach_channel_metadata(data, x_schema=x_schema, y_schema=y_schema, mask_wall=getattr(data, "mask_wall", None))

