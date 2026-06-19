"""Train the local kinematic corrector on the COMSOL Patch Factory residuals.

The corrector learns the velocity diversion ``[dU, dV]`` a micro-clot induces, as a
residual on the frozen GINO-DEQ base flow. The Patch Factory (``patch_factory_comsol``)
writes structured-grid samples in **SI** units; this trainer non-dimensionalizes every
sample with the *same* ``PhysicsConfig`` convention the GINO-DEQ kine model uses, so the
corrector predicts ``dU_nd`` directly in the deploy domain:

  * length  : positions / channel height ``H`` (the patch's ``d_bar`` analog)
  * velocity: SI / ``u_ref = PhysicsConfig.get_u_ref(H)``
  * viscosity: ``PhysicsConfig.viscosity_si_to_nd`` (delta over fluid baseline)

Translation invariance is enforced by ``assemble_local_corrector_features`` (dx/dy are
centered on the clot center of mass, averaged over clot nodes only).

CLI:
    python -m src.training.train_local_kinematic_corrector \
        --patch-dir data/processed/cfd_results_patch_factory --epochs 100
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.config import PhysicsConfig
from src.core_physics.coupled_shear_gnn import (
    LOCAL_CORRECTOR_IN_CHANNELS,
    LocalKinematicCorrector,
    assemble_local_corrector_features,
    save_local_corrector,
)
from src.utils.paths import data_root, get_project_root


DEFAULT_PATCH_DIR = data_root() / "processed" / "cfd_results_patch_factory"
DEFAULT_OUT_DIR = get_project_root() / "outputs" / "kinematics" / "local_corrector"


@dataclass
class PatchNdConfig:
    """Cropping / subsampling controls for converting a patch grid into a subgraph."""

    crop_x_factor: float = 4.0   # keep |x - clot_x| <= factor * clot_width
    crop_y_frac: float = 0.5     # keep y <= frac * channel height
    crop_y_min_abs_factor: float = 8.0  # ...but at least factor * clot_height
    stride: int = 1              # grid subsample stride (>=2 for large patches)
    mu_thresh_si: float = 1e-4   # delta-mu above fluid baseline that flags a clot node


def build_grid_edge_index(ny: int, nx: int) -> torch.Tensor:
    """Bidirectional 4-neighbour connectivity for a regular ``(ny, nx)`` grid."""
    idx = np.arange(ny * nx).reshape(ny, nx)
    src: list[np.ndarray] = []
    dst: list[np.ndarray] = []
    # horizontal neighbours
    src.append(idx[:, :-1].ravel()); dst.append(idx[:, 1:].ravel())
    # vertical neighbours
    src.append(idx[:-1, :].ravel()); dst.append(idx[1:, :].ravel())
    s = np.concatenate(src); d = np.concatenate(dst)
    # symmetric edges
    ei = np.stack([np.concatenate([s, d]), np.concatenate([d, s])], axis=0)
    return torch.from_numpy(ei).long()


def _scalar(z: dict, key: str, default: float | None = None) -> float:
    if key not in z:
        if default is None:
            raise KeyError(key)
        return float(default)
    return float(np.asarray(z[key]).ravel()[0])


def patch_to_data(
    npz_path: Path,
    phys: PhysicsConfig,
    cfg: PatchNdConfig,
) -> Data | None:
    """Load one ``patch_*.npz`` and convert it to a non-dimensionalized subgraph ``Data``.

    Returns ``None`` for samples that are unusable (dry-run, missing viscosity, no clot,
    non-finite fields, or a degenerate crop).
    """
    with np.load(npz_path) as z:
        if "dry_run" in z.files and bool(np.asarray(z["dry_run"]).ravel()[0]):
            return None
        if "mu" not in z.files:
            return None  # need viscosity to locate the clot + form delta-mu
        nx = int(np.asarray(z["grid_nx"]).ravel()[0])
        ny = int(np.asarray(z["grid_ny"]).ravel()[0])
        x = np.asarray(z["x"], dtype=np.float64)
        y = np.asarray(z["y"], dtype=np.float64)
        u_base = np.asarray(z["u_base"], dtype=np.float64)
        du = np.asarray(z["du"], dtype=np.float64)
        dv = np.asarray(z["dv"], dtype=np.float64)
        mu = np.asarray(z["mu"], dtype=np.float64)
        H = _scalar(z, "height")
        cx = _scalar(z, "clot_x_center")
        cw = _scalar(z, "clot_width")
        ch = _scalar(z, "clot_height")

    if not (np.isfinite(du).all() and np.isfinite(dv).all() and np.isfinite(mu).all()):
        return None

    # --- ND scales (match the GINO-DEQ kine domain) ---
    length_scale = float(H)
    u_ref = float(phys.get_u_ref(length_scale))
    if not np.isfinite(u_ref) or u_ref <= 0.0:
        return None
    mu_inf = float(phys.mu_inf)

    # --- reshape to (ny, nx) so we can crop a contiguous rectangular block ---
    def g(a: np.ndarray) -> np.ndarray:
        return a.reshape(ny, nx)

    xs = g(x)[0, :]          # x varies along columns
    ys = g(y)[:, 0]          # y varies along rows (wall at y=0 -> row 0)

    col_keep = np.where(np.abs(xs - cx) <= cfg.crop_x_factor * cw)[0]
    y_max = min(float(H), max(cfg.crop_y_frac * H, cfg.crop_y_min_abs_factor * ch))
    row_keep = np.where(ys <= y_max)[0]
    if col_keep.size < 2 or row_keep.size < 2:
        return None
    r0, r1 = int(row_keep.min()), int(row_keep.max()) + 1
    c0, c1 = int(col_keep.min()), int(col_keep.max()) + 1
    st = max(1, int(cfg.stride))

    def crop(a: np.ndarray) -> np.ndarray:
        return g(a)[r0:r1:st, c0:c1:st]

    x_c = crop(x); y_c = crop(y)
    du_c = crop(du); dv_c = crop(dv); ub_c = crop(u_base); mu_c = crop(mu)
    ny_c, nx_c = x_c.shape
    if ny_c < 2 or nx_c < 2:
        return None

    flat = lambda a: torch.from_numpy(np.ascontiguousarray(a.ravel())).float()
    pos_nd = torch.stack([flat(x_c) / length_scale, flat(y_c) / length_scale], dim=-1)
    sdf_nd = flat(y_c) / length_scale            # distance to bottom wall (clot attaches at y=0)
    u0_nd = flat(ub_c) / u_ref                   # base flow = analytical shear baseline
    v0_nd = torch.zeros_like(u0_nd)              # baseline v = 0
    delta_mu_si = flat(mu_c) - mu_inf
    delta_mu_nd = delta_mu_si / float(phys.mu_viscosity_nd_scale)
    target = torch.stack([flat(du_c) / u_ref, flat(dv_c) / u_ref], dim=-1)

    clot_nodes = torch.where(delta_mu_si > float(cfg.mu_thresh_si))[0]
    if clot_nodes.numel() == 0:
        return None
    subset = torch.arange(pos_nd.shape[0])

    feats = assemble_local_corrector_features(
        pos_nd, sdf_nd, u0_nd, v0_nd, delta_mu_nd, clot_nodes, subset
    )
    edge_index = build_grid_edge_index(ny_c, nx_c)
    data = Data(x=feats, edge_index=edge_index, y=target)
    data.num_nodes = feats.shape[0]
    return data


class PatchFactoryDataset(torch.utils.data.Dataset):
    """Lazily convert ``patch_*.npz`` files into non-dimensionalized subgraphs."""

    def __init__(self, patch_dir: Path, phys: PhysicsConfig, cfg: PatchNdConfig):
        self.paths = sorted(
            glob.glob(str(Path(patch_dir) / "patch_*.npz")),
            key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
        )
        self.phys = phys
        self.cfg = cfg
        self._cache: dict[int, Data] = {}
        self._usable: list[int] = []
        for i, p in enumerate(self.paths):
            d = patch_to_data(Path(p), phys, cfg)
            if d is not None:
                self._cache[i] = d
                self._usable.append(i)

    def __len__(self) -> int:
        return len(self._usable)

    def __getitem__(self, i: int) -> Data:
        return self._cache[self._usable[i]]


def train_corrector(
    patch_dir: Path | str = DEFAULT_PATCH_DIR,
    *,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 64,
    val_frac: float = 0.1,
    device: torch.device | str | None = None,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    nd_cfg: PatchNdConfig | None = None,
    seed: int = 0,
) -> LocalKinematicCorrector:
    """Fit the corrector on Patch Factory residuals; save best-by-val checkpoint."""
    dev = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(seed)
    phys = PhysicsConfig(phase="kinematics")
    cfg = nd_cfg or PatchNdConfig()

    dataset = PatchFactoryDataset(Path(patch_dir), phys, cfg)
    n = len(dataset)
    if n == 0:
        raise RuntimeError(
            f"No usable patches in {patch_dir}. Generate them with "
            "`python -m src.data_gen.lib.patch_factory_comsol` (mu/viscosity required)."
        )
    n_val = max(1, int(round(val_frac * n))) if n > 1 else 0
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx = set(perm[:n_val])
    train_set = [dataset[i] for i in range(n) if i not in val_idx]
    val_set = [dataset[i] for i in range(n) if i in val_idx]
    print(f"[i] patches: {n} usable | train {len(train_set)} | val {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False) if val_set else None

    model = LocalKinematicCorrector(in_channels=LOCAL_CORRECTOR_IN_CHANNELS, hidden_dim=hidden_dim).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "local_kinematic_corrector_best.pth"
    last_path = out_dir / "local_kinematic_corrector_last.pth"
    best_val = float("inf")

    def _meta(epoch: int, train_mse: float, val_mse: float | None) -> dict[str, Any]:
        return {
            "model": "LocalKinematicCorrector",
            "in_channels": LOCAL_CORRECTOR_IN_CHANNELS,
            "hidden_dim": hidden_dim,
            "feature_names": ["dx", "dy", "dist_to_wall", "u0", "v0", "delta_mu"],
            "target": "delta_uv_nd",
            "normalization": {
                "length_scale": "channel_height_H",
                "velocity_scale": "PhysicsConfig.get_u_ref(H)",
                "viscosity_scale": float(phys.mu_viscosity_nd_scale),
            },
            "patch_dir": str(patch_dir),
            "n_patches_usable": n,
            "epoch": epoch,
            "train_mse_nd": train_mse,
            "val_mse_nd": val_mse,
            "nd_cfg": vars(cfg),
        }

    for epoch in range(epochs):
        model.train()
        total = 0.0
        nb = 0
        for batch in train_loader:
            batch = batch.to(dev)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index)
            loss = F.mse_loss(pred, batch.y)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            nb += 1
        train_mse = total / max(nb, 1)

        val_mse: float | None = None
        if val_loader is not None:
            model.eval()
            vt = 0.0
            vb = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(dev)
                    pred = model(batch.x, batch.edge_index)
                    vt += float(F.mse_loss(pred, batch.y).item())
                    vb += 1
            val_mse = vt / max(vb, 1)

        score = val_mse if val_mse is not None else train_mse
        if score < best_val:
            best_val = score
            save_local_corrector(best_path, model, _meta(epoch, train_mse, val_mse))

        if epoch % 10 == 0 or epoch == epochs - 1:
            vtxt = f" | val MSE {val_mse:.6e}" if val_mse is not None else ""
            print(f"[i] epoch {epoch:3d} | train MSE {train_mse:.6e}{vtxt}")

    save_local_corrector(last_path, model, _meta(epochs - 1, train_mse, val_mse))
    print(f"[OK] best val MSE_nd {best_val:.6e} -> {best_path}")
    return model


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the local kinematic corrector on Patch Factory residuals.")
    p.add_argument("--patch-dir", type=str, default=str(DEFAULT_PATCH_DIR))
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--stride", type=int, default=1, help="Grid subsample stride (>=2 for large patches).")
    p.add_argument("--crop-x-factor", type=float, default=4.0)
    p.add_argument("--crop-y-frac", type=float, default=0.5)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    nd_cfg = PatchNdConfig(
        crop_x_factor=args.crop_x_factor,
        crop_y_frac=args.crop_y_frac,
        stride=args.stride,
    )
    train_corrector(
        patch_dir=args.patch_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        val_frac=args.val_frac,
        device=args.device,
        nd_cfg=nd_cfg,
        seed=args.seed,
    )
