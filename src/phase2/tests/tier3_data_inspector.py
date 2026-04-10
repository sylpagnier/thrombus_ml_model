import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.widgets import Button, Slider, TextBox

from src.config import VesselConfig, BiochemConfig, STATE_CHANNEL_MU_EFF_ND
from src.utils.paths import get_project_root

# y channel layout in extracted Tier-3 graphs:
# 0:u, 1:v, 2:p, 3:mu, 4:rp, 5:ap, 6:apr, 7:aps, 8:PT, 9:thrombin, 10:AT, 11:FG, 12:FI, 13:M, 14:Mas, 15:Mat
THROMBIN_CHANNEL_IDX = 9
FIBRIN_CHANNEL_IDX = 12


def _compute_boundary_normals(edge_index, boundary_mask, pos_tensor, num_nodes):
    normals = torch.zeros((num_nodes, 2), dtype=torch.float32, device=pos_tensor.device)
    row, col = edge_index
    b_edges = boundary_mask[row] & boundary_mask[col]
    if not b_edges.any():
        return normals
    r = row[b_edges]
    c = col[b_edges]
    edge_vecs = pos_tensor[c] - pos_tensor[r]
    edge_normals = torch.stack([-edge_vecs[:, 1], edge_vecs[:, 0]], dim=1)
    normals.scatter_add_(0, r.unsqueeze(1).expand(-1, 2), edge_normals)
    normals.scatter_add_(0, c.unsqueeze(1).expand(-1, 2), edge_normals)
    norm_mag = torch.linalg.norm(normals, dim=1, keepdim=True) + 1e-9
    return normals / norm_mag


def _compute_wls_condition_numbers(data):
    try:
        pos = data.x[:, :2]
        edge_index = data.edge_index
        row, col = edge_index
        num_nodes = int(data.num_nodes)
        pos_diff = pos[col] - pos[row]
        dx = pos_diff[:, 0]
        dy = pos_diff[:, 1]
        dist_sq = dx ** 2 + dy ** 2 + 1e-8
        w = 1.0 / dist_sq
        v = torch.stack([dx, dy, 0.5 * dx ** 2, dx * dy, 0.5 * dy ** 2], dim=1)
        m_e = w.view(-1, 1, 1) * torch.bmm(v.unsqueeze(2), v.unsqueeze(1))
        m_flat = torch.zeros((num_nodes, 25), dtype=m_e.dtype, device=m_e.device)
        m_flat.scatter_add_(0, row.view(-1, 1).expand(-1, 25), m_e.view(-1, 25))
        m = m_flat.view(num_nodes, 5, 5)
        eps = 1e-6
        eye = torch.eye(5, dtype=m.dtype, device=m.device).unsqueeze(0).expand(num_nodes, 5, 5)
        m_reg = m + eps * eye
        cond = torch.linalg.cond(m_reg)
        return cond.detach().cpu().numpy()
    except Exception:
        return None


class Tier3DataInspector:
    """Interactive Tier-3 graph inspector with optional QC dashboard mode."""

    def __init__(self, proc_dir=None):
        self.root = get_project_root()
        self.proc_dir_override = Path(proc_dir) if proc_dir else None

    def _proc_dir_for_tier(self, tier):
        if self.proc_dir_override is not None:
            return self.proc_dir_override
        return self.root / VesselConfig(tier=tier).graph_output_dir

    def _list_stems_for_tier(self, tier):
        proc_dir = self._proc_dir_for_tier(tier)
        if not proc_dir.exists():
            return []
        return sorted([p.stem for p in proc_dir.glob("*.pt")])

    def _comsol_dir_for_tier(self, tier):
        if tier == "tier3":
            return self.root / VesselConfig(tier="tier3").output_dir
        return self.root / VesselConfig(tier="tier3_patients").output_dir

    def _pick_comsol_export_for_tier(self, tier, preferred_stem=None):
        comsol_dir = self._comsol_dir_for_tier(tier)
        if not comsol_dir.exists():
            return None
        if preferred_stem:
            p = comsol_dir / f"{preferred_stem}.txt"
            if p.exists():
                return p
        txts = sorted(
            [
                p for p in comsol_dir.glob("*.txt")
                if p.is_file() and not p.stem.endswith("_inlet")
                and not p.stem.endswith("_outlet")
                and not p.stem.endswith("_wall")
            ]
        )
        return txts[0] if txts else None

    def audit_comsol_export_units(self, export_path, sample_rows=50000):
        """
        Heuristic unit audit for one COMSOL wide-format export.
        Uses first timestep block and reports likely unit families + SI conversion hints.
        """
        export_path = Path(export_path)
        if not export_path.exists():
            raise FileNotFoundError(f"COMSOL export not found: {export_path}")

        # Extract the first timestep block exactly like extract_tier3_comsol_data.py layout.
        df_full = pd.read_csv(export_path, comment="%", sep=r"\s+", header=None, nrows=sample_rows)
        if df_full.shape[1] < 20:
            raise ValueError(
                f"Unexpected format in {export_path.name}: need >=20 columns, got {df_full.shape[1]}. "
                "Expected COMSOL wide export with [x y u v p mu + 12 species] per timestep."
            )

        first_block = df_full.iloc[:, 2:20].copy()
        first_block.columns = [
            "x", "y", "u", "v", "p", "mu_effective",
            "rp", "ap", "apr", "aps", "PT", "th", "at", "fg", "fi", "M", "Mas", "Mat",
        ]

        bio_cfg = BiochemConfig(tier="tier3")

        # Expected CGS-ish baseline magnitudes from COMSOL parameters.
        expected_cgs = {
            "rp": bio_cfg.c_RP0 / 1e6,     # plt/m^3 -> plt/ml
            "ap": (0.05 * bio_cfg.c_RP0) / 1e6,  # c_AP0 in COMSOL params screenshot
            "apr": bio_cfg.APRcrit * 1e3,  # mol/m^3 -> uM
            "aps": bio_cfg.APScrit * 1e3,  # mol/m^3 -> uM
            "PT": bio_cfg.c_pT0 * 1e3,     # mol/m^3 -> uM
            "th": bio_cfg.Tcrit * 1e3,     # mol/m^3 -> uM
            "at": bio_cfg.cAT0 * 1e3,      # mol/m^3 -> uM
            "fg": bio_cfg.c_Fg0 * 1e3,     # mol/m^3 -> uM
            "fi": bio_cfg.c_Fg0 * 1e3,     # proxy
            "M": bio_cfg.Minf / 1e4,       # plt/m^2 -> plt/cm^2
            "Mas": bio_cfg.Minf / 1e4,
            "Mat": bio_cfg.Minf / 1e4,
        }

        species_cols = ["rp", "ap", "apr", "aps", "PT", "th", "at", "fg", "fi", "M", "Mas", "Mat"]
        summary_rows = []
        flagged = []

        def _safe_pos_stats(series):
            vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return 0.0, 0.0, 0.0
            pos = vals[vals > 0.0]
            if pos.size == 0:
                return float(np.nanmedian(vals)), 0.0, 0.0
            return float(np.nanmedian(pos)), float(np.nanpercentile(pos, 95)), float(np.nanmax(pos))

        for col in species_cols:
            med, p95, vmax = _safe_pos_stats(first_block[col])
            ref = max(expected_cgs[col], 1e-18)
            ratio = p95 / ref if p95 > 0 else 0.0

            # Species-specific unit family guess.
            if col in ("rp", "ap"):
                if 0.1 <= ratio <= 10.0:
                    unit_guess = "likely plt/ml (CGS)"
                    to_si = "multiply by 1e6 -> plt/m^3"
                elif 1e5 <= ratio <= 1e7:
                    unit_guess = "likely already plt/m^3 (SI-like)"
                    to_si = "no 1e6 multiplier"
                else:
                    unit_guess = "uncertain platelet unit"
                    to_si = "check COMSOL unit display in export table"
            elif col in ("M", "Mas", "Mat"):
                if 0.1 <= ratio <= 10.0:
                    unit_guess = "likely plt/cm^2 (CGS)"
                    to_si = "multiply by 1e4 -> plt/m^2"
                elif 1e3 <= ratio <= 1e5:
                    unit_guess = "likely already plt/m^2 (SI-like)"
                    to_si = "no 1e4 multiplier"
                else:
                    unit_guess = "uncertain surface unit"
                    to_si = "check COMSOL wall species units"
            else:
                if 0.1 <= ratio <= 10.0:
                    unit_guess = "likely uM (CGS/COMSOL style)"
                    to_si = "multiply by 1e-3 -> mol/m^3"
                elif 1e-4 <= ratio <= 1e-2:
                    unit_guess = "likely already mol/m^3 (SI-like)"
                    to_si = "no 1e-3 multiplier"
                else:
                    unit_guess = "uncertain solute unit"
                    to_si = "check COMSOL concentration unit (uM vs mol/m^3)"

            if "uncertain" in unit_guess:
                flagged.append(col)

            summary_rows.append((col, med, p95, vmax, expected_cgs[col], ratio, unit_guess, to_si))

        print("\n=== Tier3 COMSOL Unit Audit ===")
        print(f"file: {export_path}")
        print(f"rows sampled: {len(first_block)}")
        print("columns: rp ap apr aps PT th at fg fi M Mas Mat")
        print("")
        print(
            f"{'col':<5} {'median+':>12} {'p95+':>12} {'max+':>12} {'ref(CG S)':>12} "
            f"{'p95/ref':>10}  {'guess':<32} {'suggested SI conversion'}"
        )
        for col, med, p95, vmax, ref, ratio, guess, to_si in summary_rows:
            print(
                f"{col:<5} {med:12.4g} {p95:12.4g} {vmax:12.4g} {ref:12.4g} "
                f"{ratio:10.3g}  {guess:<32} {to_si}"
            )

        # Specific high-risk warning for the extractor scale mismatch pattern.
        major_solutes = ["PT", "at", "fg"]
        likely_um = True
        for c in major_solutes:
            p95 = [row for row in summary_rows if row[0] == c][0][2]
            ref = expected_cgs[c]
            rr = p95 / max(ref, 1e-18) if p95 > 0 else 0.0
            if not (0.1 <= rr <= 10.0):
                likely_um = False
                break

        if likely_um:
            print("\n[WARN] Likely CGS export detected for major solutes (uM / plt-style).")
            print("    If extractor multiplies these raw columns by bulk_scale=1e6 directly,")
            print("    PT/AT/FG can be inflated by ~1e3 and dominate B_In / ADR_F residuals.")
            print("    Recommended: convert uM->mol/m^3 first (x1e-3) before ND/log transform.")
        elif len(flagged) > 0:
            print(f"\n[WARN] Uncertain unit columns: {', '.join(flagged)}")
            print("    Confirm exact unit in COMSOL Export table 'Unit' column and align extractor conversions.")
        else:
            print("\n[OK] Units look self-consistent against expected COMSOL CGS baselines.")

    def run_default_unit_audit(self, synthetic_stem=None, anchor_stem=None, sample_rows=50000):
        """
        Default audit pass: attempt one COMSOL export audit for synthetic and anchor tiers.
        This is best-effort and never raises hard failures during UI startup.
        """
        print("\n=== Default Tier3 Unit Audit (pre-inspection) ===")
        targets = [
            ("tier3", synthetic_stem),
            ("tier3_patients", anchor_stem),
        ]
        any_ran = False
        for tier, stem in targets:
            path = self._pick_comsol_export_for_tier(tier, preferred_stem=stem)
            if path is None:
                print(f"- {tier}: no COMSOL domain export found, skipping.")
                continue
            try:
                print(f"\n--- Auditing {tier}: {path.name} ---")
                self.audit_comsol_export_units(path, sample_rows=sample_rows)
                any_ran = True
            except Exception as exc:
                print(f"- {tier}: audit failed for {path.name}: {exc}")
        if not any_ran:
            print("No COMSOL exports were audited.")

    def _build_single_window(self, tier, stem=None, qc_mode=False):
        state = {"tier": tier, "stem": stem, "stems": [], "widgets": [], "key_cid": None}
        fig = plt.figure(figsize=(18, 11) if qc_mode else (14, 10))

        def _refresh_stem_list():
            state["stems"] = self._list_stems_for_tier(state["tier"])
            if len(state["stems"]) == 0:
                state["stem"] = None
            elif state["stem"] not in set(state["stems"]):
                state["stem"] = state["stems"][0]

        def _set_stem_and_redraw(new_stem):
            if new_stem not in set(state["stems"]):
                print(f"Stem {new_stem} not found in {state['tier']}.")
                return
            state["stem"] = new_stem
            _render_current()

        def _render_current():
            _refresh_stem_list()
            fig.clf()
            state["widgets"] = []
            fig.subplots_adjust(bottom=0.12, hspace=0.34, wspace=0.26, left=0.04, right=0.99, top=0.92)

            if len(state["stems"]) == 0:
                msg_ax = fig.add_subplot(111)
                msg_ax.axis("off")
                msg_ax.text(
                    0.5,
                    0.5,
                    f"No .pt files found for {state['tier']} in {self._proc_dir_for_tier(state['tier'])}",
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="crimson",
                )
                fig.suptitle(f"Tier 3 Inspector [{state['tier']}]", fontsize=18, fontweight="bold")
                fig.canvas.draw_idle()
                return

            filepath = self._proc_dir_for_tier(state["tier"]) / f"{state['stem']}.pt"
            if not filepath.exists():
                print(f"Error: Could not find processed graph file at {filepath}")
                fig.canvas.draw_idle()
                return

            print(f"Loading data for {state['tier']}::{state['stem']}...")
            data = torch.load(filepath, weights_only=False)
            x = data.x[:, 0].cpu().numpy()
            y = data.x[:, 1].cpu().numpy()
            sdf = data.x[:, 2].cpu().numpy()

            if hasattr(data, "t") and data.t is not None:
                time_steps = data.t.cpu().numpy()
            else:
                time_steps = np.array([0.0], dtype=np.float32)
            num_steps = len(time_steps)

            y_data = data.y
            if y_data.dim() == 2:
                y_data = y_data.unsqueeze(0)
            elif y_data.dim() != 3:
                raise ValueError(f"Unsupported y shape for inspection: {tuple(y_data.shape)}")

            y_channels = int(y_data.shape[-1])
            has_thrombin_channel = y_channels > THROMBIN_CHANNEL_IDX
            has_fibrin_channel = y_channels > FIBRIN_CHANNEL_IDX
            ia = getattr(data, "is_anchor", None)
            if ia is None:
                anchor_mask = np.zeros(int(data.num_nodes), dtype=bool)
                anchor_count = 0
            elif torch.is_tensor(ia):
                if ia.numel() == int(data.num_nodes):
                    anchor_mask = ia.bool().cpu().numpy()
                else:
                    anchor_mask = np.full(int(data.num_nodes), bool(ia.any().item()), dtype=bool)
                anchor_count = int(np.sum(anchor_mask))
            else:
                anchor_mask = np.full(int(data.num_nodes), bool(ia), dtype=bool)
                anchor_count = int(np.sum(anchor_mask))
            print(
                f"Loaded {state['stem']}: nodes={int(data.num_nodes)} | T={num_steps} | "
                f"y_channels={y_channels} | anchor_nodes={anchor_count}"
            )

            axs = fig.subplots(4, 4) if qc_mode else fig.subplots(4 if has_fibrin_channel else 3, 2)
            fig.suptitle(
                f"Tier 3 Inspector [{state['tier']}]: {state['stem']} (t={time_steps[0]:.4f}s)",
                fontsize=15,
                fontweight="bold",
            )

            def get_data_at_step(idx):
                u_comp = y_data[idx, :, 0].cpu().numpy()
                v_comp = y_data[idx, :, 1].cpu().numpy()
                vel_mag = np.sqrt(u_comp ** 2 + v_comp ** 2)
                p_rel = y_data[idx, :, 2].cpu().numpy()
                mu_eff = y_data[idx, :, STATE_CHANNEL_MU_EFF_ND].cpu().numpy()
                thrombin = (
                    y_data[idx, :, THROMBIN_CHANNEL_IDX].cpu().numpy()
                    if has_thrombin_channel
                    else np.zeros_like(vel_mag)
                )
                fibrin = (
                    y_data[idx, :, FIBRIN_CHANNEL_IDX].cpu().numpy()
                    if has_fibrin_channel
                    else np.zeros_like(vel_mag)
                )
                return vel_mag, p_rel, mu_eff, thrombin, fibrin

            wall_mask = data.mask_wall.cpu().numpy().astype(bool)
            inlet_mask = data.mask_inlet.cpu().numpy().astype(bool)
            outlet_mask = data.mask_outlet.cpu().numpy().astype(bool)
            init_idx = 0
            vel_mag, p_rel, mu_eff, thrombin, fibrin = get_data_at_step(init_idx)

            if not qc_mode:
                sc1 = axs[0, 0].scatter(x, y, c=vel_mag, cmap="viridis", s=2)
                axs[0, 0].set_title(r"Normalized Velocity ($|U| / u_{ref}$)")
                fig.colorbar(sc1, ax=axs[0, 0], label="ND")

                sc2 = axs[0, 1].scatter(x, y, c=p_rel, cmap="RdBu_r", s=2)
                axs[0, 1].set_title("Non-Dimensional Pressure (Relative)")
                fig.colorbar(sc2, ax=axs[0, 1], label="ND (p / p_ref)")

                sc3 = axs[1, 0].scatter(x, y, c=mu_eff, cmap="magma", s=2)
                axs[1, 0].set_title("ND effective viscosity (μ_si / μ_viscosity_nd_scale)")
                fig.colorbar(sc3, ax=axs[1, 0], label="ND Ratio")

                sc4 = axs[1, 1].scatter(x, y, c=thrombin, cmap="plasma", s=2)
                axs[1, 1].set_title(r"Thrombin $\ln(1 + \hat{T})$" if has_thrombin_channel else "Thrombin (missing)")
                fig.colorbar(sc4, ax=axs[1, 1], label="Transformed ND Units")

                sc5 = axs[2, 0].scatter(x, y, c=sdf, cmap="coolwarm", s=2)
                axs[2, 0].set_title("Wall Distance (SDF)")
                fig.colorbar(sc5, ax=axs[2, 0])

                axs[2, 1].scatter(x, y, c="gray", s=1, alpha=0.05, label="Internal")
                axs[2, 1].scatter(x[wall_mask], y[wall_mask], c="black", s=5, label="Wall")
                axs[2, 1].scatter(x[inlet_mask], y[inlet_mask], c="blue", s=8, label="Inlet")
                axs[2, 1].scatter(x[outlet_mask], y[outlet_mask], c="red", s=8, label="Outlet")
                axs[2, 1].set_title("Boundary Node Verification")
                axs[2, 1].legend(loc="upper right")

                sc_fi = None
                if has_fibrin_channel:
                    sc_fi = axs[3, 0].scatter(x, y, c=fibrin, cmap="cividis", s=2)
                    axs[3, 0].set_title(r"Fibrin $\ln(1 + \hat{FI})$")
                    fig.colorbar(sc_fi, ax=axs[3, 0], label="Transformed ND Units")
                    axs[3, 1].axis("off")

                for ax in axs.flat:
                    ax.axis("equal")
                    ax.axis("off")
            else:
                u_series = y_data[:, :, 0].cpu().numpy()
                v_series = y_data[:, :, 1].cpu().numpy()
                vel_series = np.sqrt(u_series ** 2 + v_series ** 2)
                p_series = y_data[:, :, 2].cpu().numpy()
                mu_series = y_data[:, :, STATE_CHANNEL_MU_EFF_ND].cpu().numpy()
                t_series = (
                    y_data[:, :, THROMBIN_CHANNEL_IDX].cpu().numpy()
                    if has_thrombin_channel
                    else np.zeros_like(vel_series)
                )
                fi_series = (
                    y_data[:, :, FIBRIN_CHANNEL_IDX].cpu().numpy()
                    if has_fibrin_channel
                    else np.zeros_like(vel_series)
                )
                vel_mean, vel_p95, vel_max = np.mean(vel_series, axis=1), np.percentile(vel_series, 95, axis=1), np.max(vel_series, axis=1)
                mu_mean, mu_p95, mu_max = np.mean(mu_series, axis=1), np.percentile(mu_series, 95, axis=1), np.max(mu_series, axis=1)
                t_mean, t_p95, t_max = np.mean(t_series, axis=1), np.percentile(t_series, 95, axis=1), np.max(t_series, axis=1)
                fi_mean, fi_p95, fi_max = np.mean(fi_series, axis=1), np.percentile(fi_series, 95, axis=1), np.max(fi_series, axis=1)

                cond = _compute_wls_condition_numbers(data)
                if cond is None:
                    cond = np.zeros_like(x)
                log_cond = np.log10(cond + 1.0)

                pos_t = data.x[:, :2].detach()
                edge_index = data.edge_index
                inlet_n = _compute_boundary_normals(edge_index, data.mask_inlet.bool(), pos_t, int(data.num_nodes)).cpu().numpy()
                outlet_n = _compute_boundary_normals(edge_index, data.mask_outlet.bool(), pos_t, int(data.num_nodes)).cpu().numpy()
                flux_in, flux_out, flux_imb = [], [], []
                for k in range(num_steps):
                    uv_k = np.stack([u_series[k], v_series[k]], axis=1)
                    fi = abs(float(np.sum(uv_k[inlet_mask] * inlet_n[inlet_mask])))
                    fo = abs(float(np.sum(uv_k[outlet_mask] * outlet_n[outlet_mask])))
                    flux_in.append(fi)
                    flux_out.append(fo)
                    flux_imb.append(abs(fi - fo) / (fi + 1e-8))
                flux_in = np.asarray(flux_in)
                flux_out = np.asarray(flux_out)
                flux_imb = np.asarray(flux_imb)

                axes = axs.flatten()
                ax_vel, ax_p, ax_mu, ax_t = axes[0], axes[1], axes[2], axes[3]
                ax_anchor, ax_wall_u, ax_bnd, ax_cond = axes[4], axes[5], axes[6], axes[7]
                ax_vel_ts, ax_mu_ts, ax_t_ts, ax_flux = axes[8], axes[9], axes[10], axes[11]
                ax_hist_main, ax_hist_bio, ax_sdf, ax_meta = axes[12], axes[13], axes[14], axes[15]

                sc1 = ax_vel.scatter(x, y, c=vel_mag, cmap="viridis", s=2)
                ax_vel.set_title("Velocity Magnitude")
                fig.colorbar(sc1, ax=ax_vel, fraction=0.046, pad=0.02)

                sc2 = ax_p.scatter(x, y, c=p_rel, cmap="RdBu_r", s=2)
                ax_p.set_title("Pressure (ND)")
                fig.colorbar(sc2, ax=ax_p, fraction=0.046, pad=0.02)

                sc3 = ax_mu.scatter(x, y, c=mu_eff, cmap="magma", s=2)
                ax_mu.set_title("Effective Viscosity (ND)")
                fig.colorbar(sc3, ax=ax_mu, fraction=0.046, pad=0.02)

                sc4 = ax_t.scatter(x, y, c=thrombin, cmap="plasma", s=2)
                ax_t.set_title("Thrombin Channel")
                fig.colorbar(sc4, ax=ax_t, fraction=0.046, pad=0.02)

                ax_anchor.scatter(x, y, c="lightgray", s=1, alpha=0.15)
                if anchor_mask.any():
                    ax_anchor.scatter(x[anchor_mask], y[anchor_mask], c="limegreen", s=4, label="anchor")
                ax_anchor.set_title(f"COMSOL-Matched Nodes ({100.0 * anchor_mask.mean():.1f}%)")
                if anchor_mask.any():
                    ax_anchor.legend(loc="upper right", fontsize="x-small")

                wall_speed = np.zeros_like(vel_mag)
                wall_speed[wall_mask] = vel_mag[wall_mask]
                sc_wall = ax_wall_u.scatter(x, y, c=wall_speed, cmap="inferno", s=2)
                ax_wall_u.set_title("No-Slip Check: Wall |U|")
                fig.colorbar(sc_wall, ax=ax_wall_u, fraction=0.046, pad=0.02)

                ax_bnd.scatter(x, y, c="gray", s=1, alpha=0.05)
                ax_bnd.scatter(x[wall_mask], y[wall_mask], c="black", s=4, label="wall")
                ax_bnd.scatter(x[inlet_mask], y[inlet_mask], c="blue", s=6, label="inlet")
                ax_bnd.scatter(x[outlet_mask], y[outlet_mask], c="red", s=6, label="outlet")
                ax_bnd.set_title("Boundary Masks")
                ax_bnd.legend(loc="upper right", fontsize="x-small")

                sc_cond = ax_cond.scatter(x, y, c=log_cond, cmap="cividis", s=2)
                ax_cond.set_title(r"Mesh Quality: $\log_{10}(cond(M)+1)$")
                fig.colorbar(sc_cond, ax=ax_cond, fraction=0.046, pad=0.02)

                ax_vel_ts.plot(time_steps, vel_mean, label="mean")
                ax_vel_ts.plot(time_steps, vel_p95, label="p95")
                ax_vel_ts.plot(time_steps, vel_max, label="max")
                ax_vel_ts.set_title("|U| Time Trace")
                ax_vel_ts.grid(alpha=0.25)
                ax_vel_ts.legend(fontsize="x-small")

                ax_mu_ts.plot(time_steps, mu_mean, label="mean")
                ax_mu_ts.plot(time_steps, mu_p95, label="p95")
                ax_mu_ts.plot(time_steps, mu_max, label="max")
                ax_mu_ts.set_title("mu_eff Time Trace")
                ax_mu_ts.grid(alpha=0.25)
                ax_mu_ts.legend(fontsize="x-small")

                ax_t_ts.plot(time_steps, t_mean, label="mean")
                ax_t_ts.plot(time_steps, t_p95, label="p95")
                ax_t_ts.plot(time_steps, t_max, label="max")
                if has_fibrin_channel:
                    ax_t_ts.plot(time_steps, fi_mean, label="FI mean", linestyle="--")
                    ax_t_ts.plot(time_steps, fi_p95, label="FI p95", linestyle="--")
                    ax_t_ts.plot(time_steps, fi_max, label="FI max", linestyle=":")
                ax_t_ts.set_title("Thrombin / Fibrin Time Trace")
                ax_t_ts.grid(alpha=0.25)
                ax_t_ts.legend(fontsize="x-small")

                ax_flux.plot(time_steps, flux_in, label="inlet")
                ax_flux.plot(time_steps, flux_out, label="outlet")
                ax_flux.plot(time_steps, flux_imb, label="imbalance")
                ax_flux.set_title("Flux Consistency vs Time")
                ax_flux.grid(alpha=0.25)
                ax_flux.legend(fontsize="x-small")

                def _draw_histograms(step_idx):
                    ax_hist_main.clear()
                    ax_hist_bio.clear()
                    u_now = y_data[step_idx, :, 0].cpu().numpy()
                    v_now = y_data[step_idx, :, 1].cpu().numpy()
                    p_now = y_data[step_idx, :, 2].cpu().numpy()
                    mu_now = y_data[step_idx, :, STATE_CHANNEL_MU_EFF_ND].cpu().numpy()
                    ax_hist_main.hist(u_now, bins=40, alpha=0.5, label="u")
                    ax_hist_main.hist(v_now, bins=40, alpha=0.5, label="v")
                    ax_hist_main.hist(p_now, bins=40, alpha=0.5, label="p")
                    ax_hist_main.hist(mu_now, bins=40, alpha=0.5, label="mu")
                    ax_hist_main.set_title("Current-Step Kinematic Dist.")
                    ax_hist_main.legend(fontsize="x-small")
                    ax_hist_main.grid(alpha=0.2)

                    if y_channels > 12:
                        apr = y_data[step_idx, :, 6].cpu().numpy()
                        aps = y_data[step_idx, :, 7].cpu().numpy()
                        t_ch = y_data[step_idx, :, 9].cpu().numpy() if y_channels > 9 else np.zeros_like(apr)
                        fi = y_data[step_idx, :, 12].cpu().numpy()
                        ax_hist_bio.hist(apr, bins=40, alpha=0.5, label="APR")
                        ax_hist_bio.hist(aps, bins=40, alpha=0.5, label="APS")
                        ax_hist_bio.hist(t_ch, bins=40, alpha=0.5, label="T")
                        ax_hist_bio.hist(fi, bins=40, alpha=0.5, label="FI")
                    ax_hist_bio.set_title("Current-Step Bio Dist.")
                    ax_hist_bio.legend(fontsize="x-small")
                    ax_hist_bio.grid(alpha=0.2)

                sdf_or_fi = fibrin if has_fibrin_channel else sdf
                sdf_or_fi_cmap = "cividis" if has_fibrin_channel else "coolwarm"
                sdf_or_fi_title = r"Fibrin $\ln(1+\hat{FI})$" if has_fibrin_channel else "Wall Distance (SDF)"
                sc5 = ax_sdf.scatter(x, y, c=sdf_or_fi, cmap=sdf_or_fi_cmap, s=2)
                ax_sdf.set_title(sdf_or_fi_title)
                fig.colorbar(sc5, ax=ax_sdf, fraction=0.046, pad=0.02)

                ax_meta.axis("off")
                wall_u_max = float(np.max(wall_speed[wall_mask])) if wall_mask.any() else 0.0
                ax_meta.text(
                    0.02,
                    0.98,
                    (
                        f"nodes: {int(data.num_nodes)}\n"
                        f"time steps: {num_steps}\n"
                        f"y channels: {y_channels}\n"
                        f"source: {'COMSOL trajectory (tier3_patients)' if state['tier'] == 'tier3_patients' else 'Synthetic prior (tier3)'}\n"
                        f"COMSOL-matched frac: {100.0 * anchor_mask.mean():.2f}%\n"
                        f"max log10(cond+1): {float(np.max(log_cond)):.2f}\n"
                        f"mean flux imbalance: {float(np.mean(flux_imb)):.3e}\n"
                        f"max wall |U| @t0: {wall_u_max:.3e}"
                    ),
                    ha="left",
                    va="top",
                    fontsize=10,
                    family="monospace",
                )

                for ax in [ax_vel, ax_p, ax_mu, ax_t, ax_anchor, ax_wall_u, ax_bnd, ax_cond, ax_sdf]:
                    ax.axis("equal")
                    ax.axis("off")
                _draw_histograms(init_idx)

            ax_slider = fig.add_axes([0.22, 0.03, 0.34, 0.03])
            time_slider = Slider(ax=ax_slider, label="Time Step", valmin=0, valmax=num_steps - 1, valinit=init_idx, valstep=1, color="teal")
            current_pos = state["stems"].index(state["stem"])
            prev_ax = fig.add_axes([0.58, 0.03, 0.08, 0.05])
            next_ax = fig.add_axes([0.67, 0.03, 0.08, 0.05])
            jump_ax = fig.add_axes([0.76, 0.03, 0.20, 0.05])
            prev_btn = Button(prev_ax, "Prev")
            next_btn = Button(next_ax, "Next")
            jump_box = TextBox(jump_ax, "Stem ", initial=state["stem"])

            def _go_prev(_event):
                _set_stem_and_redraw(state["stems"][(current_pos - 1) % len(state["stems"])])

            def _go_next(_event):
                _set_stem_and_redraw(state["stems"][(current_pos + 1) % len(state["stems"])])

            def _go_jump(text):
                target = text.strip()
                if target:
                    _set_stem_and_redraw(target)

            def _on_key(event):
                if event.key == "left":
                    _go_prev(None)
                elif event.key == "right":
                    _go_next(None)

            def _update_time(_):
                idx = int(time_slider.val)
                v_m, p_r, m_e, thr, fib = get_data_at_step(idx)
                sc1.set_array(v_m)
                sc2.set_array(p_r)
                sc3.set_array(m_e)
                sc4.set_array(thr)
                sc1.set_clim(vmin=v_m.min(), vmax=v_m.max())
                sc2.set_clim(vmin=p_r.min(), vmax=p_r.max())
                sc3.set_clim(vmin=m_e.min(), vmax=m_e.max())
                sc4.set_clim(vmin=thr.min(), vmax=thr.max())
                if (not qc_mode) and has_fibrin_channel and sc_fi is not None:
                    sc_fi.set_array(fib)
                    sc_fi.set_clim(vmin=fib.min(), vmax=fib.max())
                if qc_mode:
                    wall_now = np.zeros_like(v_m)
                    wall_now[wall_mask] = v_m[wall_mask]
                    sc_wall.set_array(wall_now)
                    sc_wall.set_clim(vmin=wall_now.min(), vmax=wall_now.max() if wall_now.max() > 0 else 1.0)
                    sc5.set_array(fib if has_fibrin_channel else sdf)
                    if has_fibrin_channel:
                        sc5.set_clim(vmin=fib.min(), vmax=fib.max())
                    _draw_histograms(idx)
                fig.suptitle(
                    f"Tier 3 Inspector [{state['tier']}]: {state['stem']} (t={time_steps[idx]:.4f}s)",
                    fontsize=15,
                    fontweight="bold",
                )
                fig.canvas.draw_idle()

            time_slider.on_changed(_update_time)
            prev_btn.on_clicked(_go_prev)
            next_btn.on_clicked(_go_next)
            jump_box.on_submit(_go_jump)
            state["widgets"].extend([time_slider, prev_btn, next_btn, jump_box])
            if state["key_cid"] is not None:
                fig.canvas.mpl_disconnect(state["key_cid"])
            state["key_cid"] = fig.canvas.mpl_connect("key_press_event", _on_key)
            fig.text(
                0.22,
                0.075,
                ("QC mode: expanded quality dashboard. " if qc_mode else "") +
                "Slide time, Prev/Next geometry, Left/Right keys, or type a stem.",
                fontsize=10,
            )
            fig.canvas.draw_idle()

        _render_current()
        return fig

    def inspect_dual_windows(self, synthetic_stem=None, anchor_stem=None, qc_mode=False):
        self._build_single_window("tier3", stem=synthetic_stem, qc_mode=qc_mode)
        self._build_single_window("tier3_patients", stem=anchor_stem, qc_mode=qc_mode)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Tier 3 graphs in dual windows (synthetic + anchor)")
    parser.add_argument("--synthetic-stem", type=str, default=None, help="Optional initial stem for synthetic window.")
    parser.add_argument("--anchor-stem", type=str, default=None, help="Optional initial stem for anchor/patient window.")
    parser.add_argument("--proc-dir", type=str, default=None, help="Optional directory containing processed .pt files.")
    parser.add_argument("--basic-mode", action="store_true", help="Use the lighter non-QC layout.")
    parser.add_argument(
        "--no-default-audit",
        action="store_true",
        help="Skip automatic COMSOL unit audit before launching the inspector.",
    )
    parser.add_argument(
        "--audit-comsol-file",
        type=str,
        default=None,
        help="Path to one COMSOL .txt wide export to audit species units.",
    )
    parser.add_argument(
        "--audit-sample-rows",
        type=int,
        default=50000,
        help="Rows to sample from COMSOL export for unit audit.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run audit(s) only and exit (no interactive windows).",
    )
    args = parser.parse_args()

    inspector = Tier3DataInspector(proc_dir=args.proc_dir)
    sample_rows = max(1000, args.audit_sample_rows)

    if args.audit_comsol_file:
        inspector.audit_comsol_export_units(args.audit_comsol_file, sample_rows=sample_rows)
    elif not args.no_default_audit:
        inspector.run_default_unit_audit(
            synthetic_stem=args.synthetic_stem,
            anchor_stem=args.anchor_stem,
            sample_rows=sample_rows,
        )

    if args.audit_only:
        raise SystemExit(0)

    inspector.inspect_dual_windows(
        synthetic_stem=args.synthetic_stem,
        anchor_stem=args.anchor_stem,
        qc_mode=not args.basic_mode,
    )
