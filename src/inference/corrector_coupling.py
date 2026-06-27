"""Couple the local kinematic corrector into the deploy biochem rollout.

Single source of truth for *dynamically* bending the frozen GINO-DEQ base flow around
nucleating micro-clots before that flow is consumed by the downstream biochem model
(species GraphSAGE teacher and/or clot-phi physics). This codifies the loop intercept
demonstrated in ``src/tools/verify_local_corrector_live.py`` so train, verify, and deploy
all share the *same* feature convention (``assemble_local_corrector_features``):

    Step A  base flow ``[u0, v0]``    -> frozen GINO-DEQ kine pass (ND)
    Step B  clot nodes                -> nodes where ``delta_mu_si > thresh`` (clot has formed)
    Step C  k-hop subgraph            -> ``k_hop_subgraph`` around the active clot nodes
    Step D  invariant features        -> ``assemble_local_corrector_features`` (clot-centered dx,dy)
    Step E  diversion ``[dU, dV]``    -> ``corrector(x_sub, sub_edge_index)`` patched onto base flow
    Step F  feed coupled flow         -> the biochem model sees ``u_coupled, v_coupled``

Everything stays in the GINO-DEQ non-dimensional convention: positions by the geometric
length scale (``data.x[:, 0:2]`` are already ND on patient graphs), velocity by ``u_ref``,
viscosity by ``PhysicsConfig.mu_viscosity_nd_scale``.

Enable with ``BIOCHEM_CORRECTOR_COUPLING=1``. The coupled velocity is published to a small
per-graph registry that the species rollout flow helpers consult, and can also be written
straight into ``data.y[:, :, 0:2]`` for physics consumers that read the velocity channels.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.utils import k_hop_subgraph

from src.config import NodeFeat, PhysicsConfig
from src.core_physics.clot_phi_simple import sdf_nd_from_data
from src.core_physics.coupled_shear_gnn import (
    LocalKinematicCorrector,
    assemble_local_corrector_features,
    load_local_corrector,
)
from src.utils.kinematics_inference import (
    load_kinematics_predictor,
    predict_kinematics,
    predict_kinematics_latent,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root

DEFAULT_CORRECTOR_REL = (
    "outputs/kinematics/local_corrector/local_kinematic_corrector_best.pth"
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or ("1" if default else "0")).strip().lower()
    return raw in ("1", "true", "yes", "on")


def corrector_coupling_enabled() -> bool:
    """Master switch for routing rollout flow through the local kinematic corrector."""
    return _env_bool("BIOCHEM_CORRECTOR_COUPLING", False)


def resolve_corrector_checkpoint(explicit: Path | str | None = None) -> Path:
    """Locate the trained corrector checkpoint (explicit arg > env > default path)."""
    raw = str(explicit or os.environ.get("BIOCHEM_CORRECTOR_CKPT") or "").strip()
    if not raw:
        raw = DEFAULT_CORRECTOR_REL
    p = Path(raw)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


def corrector_num_hops() -> int:
    """k for the k-hop subgraph extracted around clot nodes (Step C)."""
    raw = (os.environ.get("BIOCHEM_CORRECTOR_NUM_HOPS") or "4").strip()
    try:
        return max(int(float(raw)), 1)
    except ValueError:
        return 4


def corrector_min_delta_mu_si() -> float:
    """Min viscosity bump over the fluid baseline that flags a node as 'clot' (Step B).

    A small positive floor keeps numerical noise in the predicted ``mu_eff`` from spuriously
    triggering the corrector on the whole mesh.
    """
    raw = (os.environ.get("BIOCHEM_CORRECTOR_MU_THRESH") or "1e-3").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 1e-3


def corrector_max_delta_mu_si() -> float:
    """Clamp the Δμ feature to the corrector's training range (clot μ ~1.5-3 Pa.s).

    Patient gelation caps μ_eff near ~4 Pa.s -- beyond the patches the corrector saw -- so an
    unclamped Δμ drives it far out of distribution and it extrapolates unphysically large
    diversions (|dUV| ~ the full freestream). 0 disables the clamp.
    """
    raw = (os.environ.get("BIOCHEM_CORRECTOR_MAX_DELTA_MU") or "3.0").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 3.0


def corrector_local_clusters_enabled() -> bool:
    """Tile a macro-clot into micro-clot-scale patches and apply the corrector per patch.

    The corrector was trained on single small clots: one center of mass, a small clot-centered
    dx,dy span. Handing it a patient macro-clot (hundreds of connected nodes) as ONE subgraph is
    far out of distribution -- a single COM over the whole mass and dx,dy spanning the entire clot
    -- so it extrapolates huge, unphysical diversions (|dUV| ~ the full freestream). Tiling
    restores the training scale: each local cluster gets its own COM + small subgraph, and the
    per-node diversions are averaged where patches overlap.
    """
    return _env_bool("BIOCHEM_CORRECTOR_LOCAL_CLUSTERS", True)


def corrector_cluster_radius_nd() -> float:
    """Spatial radius (GINO-DEQ ND frame) of each local clot patch handed to the corrector."""
    raw = (os.environ.get("BIOCHEM_CORRECTOR_CLUSTER_RADIUS_ND") or "0.12").strip()
    try:
        return max(float(raw), 1e-4)
    except ValueError:
        return 0.12


def corrector_cluster_max_nodes() -> int:
    """Cap on clot nodes per local patch (keeps each call near the trained patch size)."""
    raw = (os.environ.get("BIOCHEM_CORRECTOR_CLUSTER_MAX_NODES") or "64").strip()
    try:
        return max(int(float(raw)), 1)
    except ValueError:
        return 64


def kine_resolve_enabled() -> bool:
    """Whether a clot burden may trigger a full GINO-DEQ re-solve (defaults to coupling state).

    The local corrector handles small clots cheaply, but it only patches ``u, v`` -- it never
    regenerates the DEQ latent ``z_kin`` that is the GraphSAGE teacher's *primary* flow input.
    Once enough clot has formed to genuinely reroute the global flow, the kine model must update
    itself (re-solve with the clot ``mu`` in ``MU_PRIOR``) so the latent reflects the new field.
    """
    raw = os.environ.get("BIOCHEM_KINE_RESOLVE_ON_CLOT")
    if raw is None:
        return corrector_coupling_enabled()
    return raw.strip().lower() in ("1", "true", "yes", "on")


def kine_resolve_min_clot_nodes() -> int:
    """Clot node count above which a full DEQ re-solve is warranted (global flow change)."""
    raw = (os.environ.get("BIOCHEM_KINE_RESOLVE_MIN_CLOT_NODES") or "40").strip()
    try:
        return max(int(float(raw)), 1)
    except ValueError:
        return 40


def kine_resolve_min_band_frac() -> float:
    """Alternative trigger: clot fraction of the mesh above which to re-solve (0 disables)."""
    raw = (os.environ.get("BIOCHEM_KINE_RESOLVE_MIN_BAND_FRAC") or "0.0").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.0


def kine_resolve_growth_factor() -> float:
    """Hysteresis: re-solve only once the clot has grown by this factor since the last solve.

    Avoids paying for a global DEQ solve every macro step when the clot is barely changing.
    """
    raw = (os.environ.get("BIOCHEM_KINE_RESOLVE_GROWTH_FACTOR") or "1.5").strip()
    try:
        return max(float(raw), 1.0)
    except ValueError:
        return 1.5


def clot_burden_significant(n_clot: int, n_total: int) -> bool:
    """True when the clot is large enough to materially reroute the global flow."""
    if not kine_resolve_enabled():
        return False
    if int(n_clot) >= kine_resolve_min_clot_nodes():
        return True
    frac = kine_resolve_min_band_frac()
    return frac > 0.0 and n_total > 0 and (float(n_clot) / float(n_total)) >= frac


def _graph_key(data) -> tuple[int, int, int]:
    n = int(data.num_nodes)
    e = int(data.edge_index.shape[1])
    ptr = 0
    if hasattr(data, "x") and torch.is_tensor(data.x) and data.x.numel() > 0:
        ptr = int(data.x.untyped_storage().data_ptr())
    return (n, e, ptr)


def clot_nodes_from_delta_mu(
    delta_mu_si: torch.Tensor,
    *,
    min_delta_mu_si: float | None = None,
) -> torch.Tensor:
    """Step B: indices where the dynamic viscosity exceeds the fluid baseline."""
    thr = corrector_min_delta_mu_si() if min_delta_mu_si is None else float(min_delta_mu_si)
    return torch.where(delta_mu_si.reshape(-1) > thr)[0]


@torch.no_grad()
def tile_clot_nodes(
    pos_nd: torch.Tensor,
    clot_nodes: torch.Tensor,
    *,
    radius_nd: float | None = None,
    max_per_cluster: int | None = None,
) -> list[torch.Tensor]:
    """Greedily partition clot nodes into local, micro-clot-scale clusters.

    Each cluster is a spatial ball (radius ``radius_nd`` in the GINO-DEQ ND frame) capped at
    ``max_per_cluster`` nodes, so every corrector call sees a patch the size it trained on rather
    than the full macro-clot. Returns a list of global node-index tensors that together cover all
    clot nodes (disjoint).
    """
    clot_nodes = clot_nodes.reshape(-1)
    if clot_nodes.numel() == 0:
        return []
    radius = corrector_cluster_radius_nd() if radius_nd is None else float(radius_nd)
    cap = corrector_cluster_max_nodes() if max_per_cluster is None else int(max_per_cluster)
    cpos = pos_nd[clot_nodes].to(dtype=torch.float32)
    remaining = torch.ones(clot_nodes.numel(), dtype=torch.bool, device=cpos.device)
    clusters: list[torch.Tensor] = []
    while bool(remaining.any()):
        seed = int(torch.nonzero(remaining, as_tuple=False)[0])
        dist = (cpos - cpos[seed]).norm(dim=1)
        sel = remaining & (dist <= radius)
        idx = torch.nonzero(sel, as_tuple=False).reshape(-1)
        if idx.numel() > cap:  # keep the cap closest to the seed; rest fall to later clusters
            order = torch.argsort(dist[idx])
            idx = idx[order[:cap]]
        clusters.append(clot_nodes[idx])
        remaining[idx] = False
    return clusters


@torch.no_grad()
def couple_flow_with_corrector(
    data,
    u0_nd: torch.Tensor,
    v0_nd: torch.Tensor,
    delta_mu_si: torch.Tensor,
    *,
    corrector: LocalKinematicCorrector,
    phys_cfg: PhysicsConfig,
    device: torch.device,
    num_hops: int | None = None,
    min_delta_mu_si: float | None = None,
    clot_nodes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the trained corrector's velocity diversion around the active clot (Steps B-E).

    ``u0_nd, v0_nd`` are the frozen GINO-DEQ base flow (ND); ``delta_mu_si`` is the per-node
    viscosity bump over the fluid baseline (SI), i.e. ``mu_eff_si - mu_inf``. Returns full-graph
    coupled ``(u, v)`` (ND); nodes outside the clot subgraph are left at the base flow. With no
    clot present (or an untrained near-identity corrector) the base flow is returned unchanged.
    """
    n = int(data.num_nodes)
    u_coupled = u0_nd.reshape(-1).clone()
    v_coupled = v0_nd.reshape(-1).clone()

    nodes = clot_nodes if clot_nodes is not None else clot_nodes_from_delta_mu(
        delta_mu_si, min_delta_mu_si=min_delta_mu_si
    )
    nodes = nodes.reshape(-1).to(device=device)
    if nodes.numel() == 0:
        return u_coupled, v_coupled

    hops = corrector_num_hops() if num_hops is None else int(num_hops)
    pos_nd = data.x[:, 0:2].to(device=device, dtype=torch.float32)
    sdf_nd = sdf_nd_from_data(data, device, n).reshape(-1)
    delta_mu_clamped = delta_mu_si.reshape(-1).to(device=device)
    max_delta = corrector_max_delta_mu_si()
    if max_delta > 0.0:
        delta_mu_clamped = delta_mu_clamped.clamp(max=max_delta)
    delta_mu_nd = phys_cfg.viscosity_si_to_nd(delta_mu_clamped)
    edge_index = data.edge_index.to(device)

    # Apply the (micro-clot) corrector *locally*: tile the clot into training-scale patches so
    # each call keeps a small clot-centered dx,dy span (in distribution), then accumulate and
    # average the diversions where the k-hop subgraphs overlap. The base flow ``u_coupled`` is
    # held fixed across clusters (each patch is a residual on the same frozen field).
    clusters = (
        tile_clot_nodes(pos_nd, nodes)
        if corrector_local_clusters_enabled()
        else [nodes]
    )
    du_sum = torch.zeros(n, device=device)
    dv_sum = torch.zeros(n, device=device)
    hits = torch.zeros(n, device=device)
    for cluster in clusters:
        if cluster.numel() == 0:
            continue
        # Step C: k-hop subgraph around this local cluster (compact relabelled index space).
        subset, sub_edge_index, _, _ = k_hop_subgraph(
            cluster,
            num_hops=hops,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=n,
        )
        if subset.numel() == 0:
            continue
        # Step D: translation-invariant features (dx, dy centered on this cluster's COM).
        x_sub = assemble_local_corrector_features(
            pos_nd, sdf_nd, u_coupled, v_coupled, delta_mu_nd, cluster, subset
        )
        # Step E: predict the diversion and accumulate it for this patch.
        delta_uv = corrector(x_sub, sub_edge_index.to(device))
        du_sum[subset] += delta_uv[:, 0]
        dv_sum[subset] += delta_uv[:, 1]
        hits[subset] += 1.0

    touched = hits > 0
    if bool(touched.any()):
        u_coupled[touched] = u_coupled[touched] + du_sum[touched] / hits[touched]
        v_coupled[touched] = v_coupled[touched] + dv_sum[touched] / hits[touched]
    return u_coupled, v_coupled


# --- per-graph coupled-flow registry (consumed by the species rollout flow helpers) ---
_COUPLED_FLOW: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def reset_coupled_flow_registry() -> None:
    _COUPLED_FLOW.clear()


def set_coupled_flow(data, u_nd: torch.Tensor, v_nd: torch.Tensor) -> None:
    _COUPLED_FLOW[_graph_key(data)] = (u_nd.detach().reshape(-1), v_nd.detach().reshape(-1))


def get_coupled_flow(
    data, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor] | None:
    entry = _COUPLED_FLOW.get(_graph_key(data))
    if entry is None:
        return None
    u, v = entry
    return u.to(device=device), v.to(device=device)


class CorrectorCoupledFlow:
    """Stateful provider: frozen base flow + trained corrector -> dynamic coupled flow.

    Loads the corrector and (lazily) the GINO-DEQ kine model once, caches the per-graph base
    flow ``[u0, v0]``, and exposes :meth:`couple` to produce the diverted flow for a given
    predicted viscosity field. Mirrors :class:`KinematicsUvProvider` but replaces the full DEQ
    re-solve with the cheap local diversion the corrector was trained to emulate.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        corrector_ckpt: Path | str | None = None,
        kine_ckpt: Path | str | None = None,
        phys_cfg: PhysicsConfig | None = None,
        num_hops: int | None = None,
        min_delta_mu_si: float | None = None,
    ) -> None:
        self.device = device
        self.phys_cfg = phys_cfg or PhysicsConfig(phase="kinematics")
        self.num_hops = corrector_num_hops() if num_hops is None else int(num_hops)
        self.min_delta_mu_si = (
            corrector_min_delta_mu_si() if min_delta_mu_si is None else float(min_delta_mu_si)
        )
        self._corrector_ckpt = resolve_corrector_checkpoint(corrector_ckpt)
        self._kine_ckpt = kine_ckpt
        self._corrector: LocalKinematicCorrector | None = None
        self._kine = None
        self._base_flow: tuple[torch.Tensor, torch.Tensor] | None = None
        self._base_key: tuple[int, int, int] | None = None

    def _ensure_corrector(self) -> LocalKinematicCorrector:
        if self._corrector is None:
            if not self._corrector_ckpt.is_file():
                raise FileNotFoundError(
                    f"Local corrector checkpoint missing: {self._corrector_ckpt}. Train it with "
                    "`python -m src.training.train_local_kinematic_corrector`."
                )
            self._corrector = load_local_corrector(self._corrector_ckpt, self.device)
        return self._corrector

    @torch.no_grad()
    def base_flow(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        """Step A: frozen GINO-DEQ base flow ``(u0, v0)`` (ND), cached per graph."""
        key = _graph_key(data)
        if self._base_flow is not None and self._base_key == key:
            return self._base_flow
        if self._kine is None:
            ckpt = resolve_kinematics_checkpoint(self._kine_ckpt)
            self._kine = load_kinematics_predictor(ckpt, self.device, phys_cfg=self.phys_cfg)
        pred = predict_kinematics(self._kine, data.to(self.device))
        u0 = pred[:, 0].contiguous()
        v0 = pred[:, 1].contiguous()
        self._base_flow = (u0, v0)
        self._base_key = key
        return self._base_flow

    @torch.no_grad()
    def couple_from_delta_mu(
        self, data, delta_mu_si: torch.Tensor, *, publish: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u0, v0 = self.base_flow(data)
        u, v = couple_flow_with_corrector(
            data,
            u0,
            v0,
            delta_mu_si,
            corrector=self._ensure_corrector(),
            phys_cfg=self.phys_cfg,
            device=self.device,
            num_hops=self.num_hops,
            min_delta_mu_si=self.min_delta_mu_si,
        )
        if publish:
            set_coupled_flow(data, u, v)
        return u, v

    def _delta_mu_si(
        self, mu_eff_si: torch.Tensor, mu_bulk_si: torch.Tensor | None
    ) -> torch.Tensor:
        """Clot viscosity elevation = ``mu_eff`` over the *clot-free* reference.

        On a real vessel the no-clot baseline is the non-Newtonian (Carreau) bulk, not ``mu_inf``,
        so passing ``mu_bulk_si`` keeps the feature (and the clot-node mask) ~0 away from the clot
        -- the distribution the corrector was trained on. Falling back to ``mu_inf`` (the Newtonian
        patch baseline) would flag all bulk blood as 'clot' and apply the diversion mesh-wide.
        """
        ref = (
            mu_bulk_si.reshape(-1).to(device=mu_eff_si.device)
            if mu_bulk_si is not None
            else float(self.phys_cfg.mu_inf)
        )
        return (mu_eff_si.reshape(-1) - ref).clamp(min=0.0)

    @torch.no_grad()
    def couple(
        self,
        data,
        mu_eff_si: torch.Tensor,
        *,
        mu_bulk_si: torch.Tensor | None = None,
        publish: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Step F input: predicted ``mu_eff_si`` -> coupled ``(u, v)`` (delta over the bulk ref)."""
        delta_mu_si = self._delta_mu_si(mu_eff_si, mu_bulk_si)
        return self.couple_from_delta_mu(data, delta_mu_si, publish=publish)


def inject_mu_prior(data, mu_eff_si: torch.Tensor, phys_cfg: PhysicsConfig):
    """Return a shallow graph clone with the predicted clot ``mu`` written into ``MU_PRIOR``.

    The GINO-DEQ reads the viscosity prior from ``data.x[:, NodeFeat.MU_PRIOR]``; injecting the
    clot ``mu_eff`` there makes a fresh solve (velocity *and* latent) clot-aware.
    """
    mu_nd = phys_cfg.viscosity_si_to_nd(mu_eff_si.reshape(-1, 1))
    x_new = data.x.clone()
    x_new[:, NodeFeat.MU_PRIOR] = mu_nd.to(device=x_new.device, dtype=x_new.dtype)
    data_k = data.clone()
    data_k.x = x_new
    return data_k


@dataclass
class ClotFlowState:
    """Result of one clot-aware flow refresh."""

    u: torch.Tensor
    v: torch.Tensor
    z_kin: torch.Tensor | None  # clot-aware DEQ latent if a full re-solve ran, else None
    mode: str                   # "frozen" | "corrector" | "resolved"
    n_clot: int


class ClotAwareFlow(CorrectorCoupledFlow):
    """Two-tier flow refresh: local-corrector diversion, escalating to a full DEQ re-solve.

    As the clot grows during the biochem rollout, :meth:`update` decides per call:

      * **frozen**    -- no clot yet -> frozen base flow, frozen latent.
      * **corrector** -- small clot  -> cheap local diversion on ``u, v`` (latent left frozen).
      * **resolved**  -- significant clot -> re-solve the GINO-DEQ with the clot ``mu`` in
        ``MU_PRIOR``, regenerating both the velocity field *and* the latent ``z_kin`` that feeds
        the GraphSAGE teacher. Hysteresis (``kine_resolve_growth_factor``) avoids re-solving
        every step.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_resolve_n = 0
        self._frozen_latent: torch.Tensor | None = None
        self._frozen_latent_key: tuple[int, int, int] | None = None

    @torch.no_grad()
    def frozen_latent(self, data) -> torch.Tensor:
        """Clot-free DEQ latent ``z_kin`` (the default GraphSAGE primary input), cached."""
        key = _graph_key(data)
        if self._frozen_latent is not None and self._frozen_latent_key == key:
            return self._frozen_latent
        self.base_flow(data)  # ensures the kine model is loaded
        assert self._kine is not None
        self._frozen_latent = predict_kinematics_latent(self._kine, data.to(self.device))
        self._frozen_latent_key = key
        return self._frozen_latent

    @torch.no_grad()
    def resolve_full(
        self, data, mu_eff_si: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full clot-aware DEQ re-solve -> ``(z_kin, u, v)`` (the kine model updating itself).

        Falls back to a CPU solve on CUDA OOM (a full DEQ solve on a large mesh + the live
        species rollout can exceed a small GPU); results are returned on ``self.device``.
        """
        self.base_flow(data)  # ensures the kine model is loaded
        assert self._kine is not None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            data_k = inject_mu_prior(data.to(self.device), mu_eff_si.to(self.device), self.phys_cfg)
            z_kin = predict_kinematics_latent(self._kine, data_k)
            pred = predict_kinematics(self._kine, data_k)
        except torch.cuda.OutOfMemoryError:
            # Fragmentation is the usual cause -- defrag and RETRY ON GPU (full speed) before CPU.
            torch.cuda.empty_cache()
            try:
                data_k = inject_mu_prior(data.to(self.device), mu_eff_si.to(self.device), self.phys_cfg)
                z_kin = predict_kinematics_latent(self._kine, data_k)
                pred = predict_kinematics(self._kine, data_k)
                print("[i] DEQ re-solve recovered on GPU after empty_cache.")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("[WARN] DEQ re-solve OOM on CUDA; retrying clot-aware solve on CPU.")
                kine_cpu = self._kine.to("cpu")
                data_k = inject_mu_prior(data.to("cpu"), mu_eff_si.to("cpu"), self.phys_cfg)
                z_kin = predict_kinematics_latent(kine_cpu, data_k)
                pred = predict_kinematics(kine_cpu, data_k)
                self._kine = kine_cpu.to(self.device)  # restore for the cached base-flow path
        return (
            z_kin.to(self.device),
            pred[:, 0].contiguous().to(self.device),
            pred[:, 1].contiguous().to(self.device),
        )

    def _grown_enough(self, n_clot: int) -> bool:
        if self._last_resolve_n <= 0:
            return True
        return n_clot >= self._last_resolve_n * kine_resolve_growth_factor()

    @torch.no_grad()
    def update(
        self,
        data,
        mu_eff_si: torch.Tensor,
        *,
        mu_bulk_si: torch.Tensor | None = None,
        publish: bool = True,
    ) -> ClotFlowState:
        """Pick the cheapest sufficient flow refresh for the current clot ``mu`` field.

        ``mu_bulk_si`` is the clot-free (Carreau bulk) reference; the clot mask is the elevation
        of ``mu_eff`` over it (see :meth:`_delta_mu_si`). A full DEQ re-solve still injects the
        *full* ``mu_eff`` into ``MU_PRIOR`` so the rerouted flow reflects the absolute viscosity.
        """
        delta_mu_si = self._delta_mu_si(mu_eff_si, mu_bulk_si)
        nodes = clot_nodes_from_delta_mu(delta_mu_si, min_delta_mu_si=self.min_delta_mu_si)
        n_clot = int(nodes.numel())
        n_total = int(data.num_nodes)

        if clot_burden_significant(n_clot, n_total) and self._grown_enough(n_clot):
            z_kin, u, v = self.resolve_full(data, mu_eff_si)
            self._last_resolve_n = n_clot
            state = ClotFlowState(u=u, v=v, z_kin=z_kin, mode="resolved", n_clot=n_clot)
        elif n_clot > 0:
            u, v = self.couple_from_delta_mu(data, delta_mu_si, publish=False)
            state = ClotFlowState(u=u, v=v, z_kin=None, mode="corrector", n_clot=n_clot)
        else:
            u0, v0 = self.base_flow(data)
            state = ClotFlowState(u=u0.clone(), v=v0.clone(), z_kin=None, mode="frozen", n_clot=0)

        if publish:
            set_coupled_flow(data, state.u, state.v)
        return state


@torch.no_grad()
def write_coupled_flow_into_y(
    data,
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    *,
    time_index: int | None = None,
) -> None:
    """Step F: overwrite the velocity channels in ``data.y`` so physics consumers see the diversion.

    Most deploy clot-phi / shear / nucleation helpers read ``[u, v]`` from ``data.y[ti][:, 0:2]``.
    Writing the coupled flow there (for one macro step or the whole timeline) feeds the diverted
    field downstream without touching the physics kernels. Operates in-place on ``data.y``.
    """
    u = u_nd.reshape(-1).to(device=data.y.device, dtype=data.y.dtype)
    v = v_nd.reshape(-1).to(device=data.y.device, dtype=data.y.dtype)
    if time_index is None:
        data.y[:, :, 0] = u.unsqueeze(0)
        data.y[:, :, 1] = v.unsqueeze(0)
    else:
        ti = max(0, min(int(time_index), int(data.y.shape[0]) - 1))
        data.y[ti, :, 0] = u
        data.y[ti, :, 1] = v
