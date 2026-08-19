"""Per-node feature extraction from the t=0 state, plus the targets.

Deploy-legal inputs for this phase: mesh geometry/connectivity, ``u_ref``/``d_bar``, the
t=0 species initial condition, and the **GT velocity field at t=0** (the stated scope).
Nothing from t>0 ever enters ``X``; GT only appears in ``y``.

Feature groups, and why each one is here:

  geometry   the shell structure is topological (PHASE7 8.4) and off-wall Mat is a fixed
             attenuation of its owner's, so "which node row am I on" and "who owns me" are
             the two facts that make the off-wall problem learnable at all.
  flow       ``sr`` and ``d(sr,x)`` are the gate's own arguments.  The MARGINS
             (``sr/lss``, ``dsrx/sgt``) matter more than the indicators: a node just above
             ``lss`` at t=0 is the one most likely to gate later as the clot narrows the
             lumen, which is exactly the 25% creep the t=0 law misses.
  physics    the zero-parameter backbone's own ``Mat`` integral and mask, so the network
             can learn a residual rather than rediscover the law.
  context    multi-hop and flow-directional aggregates of the gate/shear fields.  The
             creep is flow-mediated, so upstream/downstream asymmetry is the right prior;
             PHASE6_RESULTS 3.4 measured that isotropic smoothing of the source makes the
             fit worse, i.e. the non-locality is advective.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree

M_TO_CM = 100.0
MAT_S = 7e10          # pack Mat_log1p_nd -> COMSOL model units
BULK_S = 2.5e14


def adjacency(ei: np.ndarray, n: int) -> sp.csr_matrix:
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A


def hop_distance(seed: np.ndarray, A: sp.csr_matrix, max_h: int = 24) -> np.ndarray:
    d = np.full(len(seed), max_h + 1, dtype=np.float32)
    cur = seed.copy()
    d[cur] = 0
    for h in range(1, max_h + 1):
        nxt = ((A @ cur.astype(np.float64)) > 0) & ~cur
        if not nxt.any():
            break
        d[nxt] = h
        cur = cur | nxt
    return d


def khop_stats(A: sp.csr_matrix, vals: np.ndarray, hops: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and max of ``vals`` over the k-hop neighbourhood (inclusive)."""
    cur = sp.eye(A.shape[0], format="csr") + A
    reach = cur.copy()
    for _ in range(hops - 1):
        reach = ((reach @ (sp.eye(A.shape[0], format="csr") + A)) > 0).astype(np.float64)
    cnt = np.asarray(reach.sum(axis=1)).reshape(-1)
    mean = np.asarray(reach @ vals).reshape(-1) / np.maximum(cnt, 1.0)
    R = reach.tocoo()
    mx = np.full(A.shape[0], -np.inf)
    np.maximum.at(mx, R.row, vals[R.col])
    return mean.astype(np.float32), np.where(np.isfinite(mx), mx, 0.0).astype(np.float32)


def directional_agg(ei: np.ndarray, pos: np.ndarray, u: np.ndarray, v: np.ndarray,
                    vals: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate ``vals`` separately from UPSTREAM and DOWNSTREAM neighbours.

    An edge j->i is upstream of i when the flow at j points toward i.  Isotropic smoothing
    of the source is measurably wrong here (PHASE6_RESULTS 3.4); the transport is advective,
    so the model needs the asymmetry as a feature and not just as extra hops.
    """
    src, dst = ei[0], ei[1]
    d = pos[dst] - pos[src]
    nrm = np.linalg.norm(d, axis=1) + 1e-12
    flow = np.stack([u[src], v[src]], 1)
    fn = np.linalg.norm(flow, axis=1) + 1e-12
    cos = (d[:, 0] * flow[:, 0] + d[:, 1] * flow[:, 1]) / (nrm * fn)
    up = np.zeros(n)
    dn = np.zeros(n)
    cu = np.zeros(n)
    cd = np.zeros(n)
    w_up = np.clip(cos, 0.0, None)          # source is upstream of dst
    w_dn = np.clip(-cos, 0.0, None)
    np.add.at(up, dst, w_up * vals[src])
    np.add.at(cu, dst, w_up)
    np.add.at(dn, dst, w_dn * vals[src])
    np.add.at(cd, dst, w_dn)
    return (up / np.maximum(cu, 1e-6)).astype(np.float32), (dn / np.maximum(cd, 1e-6)).astype(np.float32)


def midside_mask(pos: np.ndarray, A: sp.csr_matrix) -> np.ndarray:
    """Degree-2 nodes sitting at the midpoint of their two neighbours (P2 mid-edge)."""
    deg = np.asarray(A.sum(axis=1)).reshape(-1)
    out = np.zeros(A.shape[0], dtype=bool)
    cand = np.flatnonzero(deg == 2)
    Ac = A.tolil()
    for i in cand:
        nb = Ac.rows[i]
        if len(nb) != 2:
            continue
        mid = 0.5 * (pos[nb[0]] + pos[nb[1]])
        if np.linalg.norm(pos[i] - mid) < 0.05 * np.linalg.norm(pos[nb[0]] - pos[nb[1]]) + 1e-12:
            out[i] = True
    return out


def build_features(data, bio_cfg, phys_cfg, *, flow: str = "gt") -> dict:
    """Everything the models consume, for one vessel.  Returns arrays keyed by name."""
    from src.core_physics.mls_gradient import build_mls_gradient, node_positions, shear_rate_2d
    from src.core_physics.physics_lumen_model import resolve_offwall_shell
    from src.core_physics.physics_wall_model import integrate_mat_trajectory, t0_flow_fields
    from src.core_physics.ap_closure import SHIPPED, SHIPPED_DA_SCALE, make_rollout_hook
    from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index
    from src.core_physics.t0_mu_physics import gt_clot_phi_at_time

    n = int(data.num_nodes)
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    ei = data.edge_index.detach().cpu().numpy()
    A = adjacency(ei, n)
    pos_nd = node_positions(data)
    pos_xy = (data.siren_pos if hasattr(data, "siren_pos") else data.x[:, :2])
    pos_xy = pos_xy.detach().cpu().numpy().astype(np.float64)
    u_ref = float(data.u_ref.reshape(-1)[0])
    d_bar = float(data.d_bar.reshape(-1)[0])

    if flow == "pred":
        u = data.u0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64)
        v = data.v0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64)
        hops = 4
    else:
        u = data.y[0, :, 0].detach().cpu().numpy().astype(np.float64)
        v = data.y[0, :, 1].detach().cpu().numpy().astype(np.float64)
        hops = 3
    p = data.y[0, :, 2].detach().cpu().numpy().astype(np.float64)

    Dx, Dy = build_mls_gradient(pos_nd, ei, hops=hops)
    ux, uy, vx, vy = Dx @ u, Dy @ u, Dx @ v, Dy @ v
    scale = u_ref / d_bar
    sr = shear_rate_2d(ux, uy, vx, vy) * scale
    dsrx = (Dx @ sr) / (d_bar * M_TO_CM)
    dsry = (Dy @ sr) / (d_bar * M_TO_CM)
    vort = (vx - uy) * scale
    div = (ux + vy) * scale
    spd = np.hypot(u, v)

    lss = float(bio_cfg.lss)
    sgt = float(bio_cfg.sgt) / M_TO_CM
    coef = float(bio_cfg.L_char) * M_TO_CM / float(bio_cfg.gamma_m)
    g_low = (sr < lss).astype(np.float64)
    g_sep = (dsrx < sgt).astype(np.float64)
    A_branch = g_sep * coef * np.abs(dsrx)
    gate = A_branch + g_low

    # --- geometry / topology -------------------------------------------------
    shell = resolve_offwall_shell(pos_xy, wall, ei)
    ms = midside_mask(pos_xy, A)
    dist_w, owner = cKDTree(pos_xy[wall]).query(pos_xy)
    owner = np.flatnonzero(wall)[owner]
    h_edge = float(np.median(np.linalg.norm(pos_xy[ei[0]] - pos_xy[ei[1]], axis=1)))
    hop_w = hop_distance(wall, A)
    deg = np.asarray(A.sum(axis=1)).reshape(-1)
    xs = data.x.detach().cpu().numpy()
    ch = {c: i for i, c in enumerate(data.x_channel_names.split(","))}
    sdf = xs[:, ch["sdf_nd"]] if "sdf_nd" in ch else np.zeros(n, np.float32)
    width = xs[:, ch["width_nd"]] if "width_nd" in ch else np.zeros(n, np.float32)
    w_d1 = xs[:, ch["width_d1"]] if "width_d1" in ch else np.zeros(n, np.float32)
    w_d2 = xs[:, ch["width_d2"]] if "width_d2" in ch else np.zeros(n, np.float32)
    nx_ = xs[:, ch["wall_normal_x"]] if "wall_normal_x" in ch else np.zeros(n, np.float32)
    ny_ = xs[:, ch["wall_normal_y"]] if "wall_normal_y" in ch else np.zeros(n, np.float32)
    u_n = u * nx_ + v * ny_
    u_t = -u * ny_ + v * nx_

    # --- physics backbone ----------------------------------------------------
    f0 = t0_flow_fields(data, bio_cfg, hops=hops, flow_source=flow)
    hook = make_rollout_hook(SHIPPED, bio_cfg, f0.sr)
    traj, _ = integrate_mat_trajectory(data, bio_cfg, f0.gate * wall,
                                       da_scale=SHIPPED_DA_SCALE, ap_closure=hook)
    mat_phys = traj[-1]
    # onset index in the backbone rollout (-1 -> never); a timing prior for the network
    crit = float(bio_cfg.viscosity_mat_crit)
    hot = traj >= crit
    onset_phys = np.where(hot.any(axis=0), hot.argmax(axis=0), traj.shape[0]).astype(np.float32)
    onset_phys = onset_phys / float(traj.shape[0])

    # --- species IC ----------------------------------------------------------
    names = data.y_channel_names.split(",")
    rp0 = np.expm1(data.y[0, :, names.index("RP_log1p_nd")].double().numpy()) * BULK_S
    ap0 = np.expm1(data.y[0, :, names.index("AP_log1p_nd")].double().numpy()) * BULK_S

    # --- context aggregates --------------------------------------------------
    gate_bin = (gate > 0).astype(np.float64)
    feats = {}
    for k in (1, 2, 4, 8, 16):
        gm, gx = khop_stats(A, gate_bin, k)
        feats[f"gate_frac_h{k}"] = gm
        sm, _ = khop_stats(A, np.minimum(sr, 500.0), k)
        feats[f"sr_mean_h{k}"] = sm.astype(np.float32)
        pm, _ = khop_stats(A, spd, k)
        feats[f"spd_mean_h{k}"] = pm.astype(np.float32)
    d_gate = hop_distance(gate_bin > 0, A)
    up_gate, dn_gate = directional_agg(ei, pos_xy, u, v, gate_bin, n)
    up_sr, dn_sr = directional_agg(ei, pos_xy, u, v, np.minimum(sr, 500.0), n)
    up_spd, dn_spd = directional_agg(ei, pos_xy, u, v, spd, n)

    F = {
        # geometry / topology
        "is_wall": wall.astype(np.float32),
        "is_shell": shell.astype(np.float32),
        "is_midside": ms.astype(np.float32),
        "dist_wall_edges": (dist_w / max(h_edge, 1e-12)).astype(np.float32),
        "dist_wall_dbar": (dist_w).astype(np.float32),
        "hop_wall": np.minimum(hop_w, 12).astype(np.float32),
        "degree": deg.astype(np.float32),
        "sdf_nd": sdf.astype(np.float32),
        "width_nd": width.astype(np.float32),
        "width_d1": w_d1.astype(np.float32),
        "width_d2": w_d2.astype(np.float32),
        # flow
        "speed_nd": spd.astype(np.float32),
        "u_n": u_n.astype(np.float32),
        "u_t": u_t.astype(np.float32),
        "p_nd": p.astype(np.float32),
        "log_sr": np.log1p(np.maximum(sr, 0.0)).astype(np.float32),
        "sr_over_lss": np.clip(sr / lss, 0, 40).astype(np.float32),
        "log_absdsrx": np.log1p(np.abs(dsrx)).astype(np.float32),
        "dsrx_over_sgt": np.clip(dsrx / abs(sgt), -40, 40).astype(np.float32),
        "log_absdsry": np.log1p(np.abs(dsry)).astype(np.float32),
        "vort": np.sign(vort) * np.log1p(np.abs(vort)).astype(np.float32),
        "div": np.sign(div) * np.log1p(np.abs(div)).astype(np.float32),
        # gates
        "gate_low": g_low.astype(np.float32),
        "gate_sep": g_sep.astype(np.float32),
        "gate_A": np.log1p(A_branch).astype(np.float32),
        "gate_sum": np.log1p(gate).astype(np.float32),
        # physics backbone
        "log_mat_phys": np.log1p(np.maximum(mat_phys, 0.0) / crit).astype(np.float32),
        "onset_phys": onset_phys,
        "log_mat_owner": np.log1p(np.maximum(mat_phys[owner], 0.0) / crit).astype(np.float32),
        "gate_owner": np.log1p(gate[owner]).astype(np.float32),
        "sr_owner": np.log1p(np.maximum(sr[owner], 0.0)).astype(np.float32),
        # species IC
        "rp0": (rp0 / 2.5e8).astype(np.float32),
        "ap0": (ap0 / 1.25e7).astype(np.float32),
        # context
        "dist_to_gate": np.minimum(d_gate, 24).astype(np.float32),
        "up_gate": up_gate, "dn_gate": dn_gate,
        "up_sr": np.log1p(np.maximum(up_sr, 0)).astype(np.float32),
        "dn_sr": np.log1p(np.maximum(dn_sr, 0)).astype(np.float32),
        "up_spd": up_spd, "dn_spd": dn_spd,
    }
    F.update(feats)

    # --- targets -------------------------------------------------------------
    t_eval = resolve_deploy_eval_time_index(int(data.y.shape[0]))
    gt = (gt_clot_phi_at_time(data, t_eval, phys_cfg, device=torch.device("cpu"))
          .reshape(-1).numpy() > 0.5)
    mat_gt = np.expm1(data.y[t_eval, :, names.index("Mat_log1p_nd")].double().numpy()) * MAT_S
    return dict(F=F, y=gt.astype(np.float32),
                mat_gt=np.log1p(np.maximum(mat_gt, 0.0) / crit).astype(np.float32),
                wall=wall, shell=shell, owner=owner.astype(np.int64),
                edge_index=ei.astype(np.int64), pos=pos_xy.astype(np.float32),
                mat_phys=mat_phys.astype(np.float32), gate=gate.astype(np.float32),
                sr=sr.astype(np.float32), spd=spd.astype(np.float32),
                u=u.astype(np.float32), v=v.astype(np.float32), n=n)


FEATURE_ORDER: list[str] | None = None


def feature_matrix(F: dict) -> tuple[np.ndarray, list[str]]:
    global FEATURE_ORDER
    if FEATURE_ORDER is None:
        FEATURE_ORDER = sorted(F.keys())
    return (np.stack([np.asarray(F[k], dtype=np.float32) for k in FEATURE_ORDER], axis=1),
            list(FEATURE_ORDER))
