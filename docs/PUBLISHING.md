# Publishing / public vs local policy (HemoRGP)

This repository is meant to be **publishable**: source, docs, and small reference manifests are versioned. Heavy data, checkpoints, and COMSOL models stay on the machine that trains.

## Track in git (public)

| Path | Why |
|------|-----|
| `src/` | All library / training / tools code |
| `scripts/` (active) + `scripts/README.md` | Supported launchers |
| `scripts/archive/` | Retired launchers kept for archaeology (not the default surface) |
| `docs/` (active) + `docs/archive/` | Design docs + historical chronicles |
| `data/reference/*.json` | Small baseline / architecture manifests |
| `customer_geometries/README.txt` | Inbox instructions only |
| `README.md`, `AGENTS.md`, `requirements.txt`, `pytest.ini` | Project entry |

## Keep local (never push)

| Path | Why |
|------|-----|
| `data/raw/`, `data/processed/`, `data/benchmark/` | Large meshes / graphs / CFD extracts |
| `outputs/` | Checkpoints, logs, viz PNGs, run dumps |
| `comsol_models/` | `.mph` sources |
| `customer_geometries/*` (except README) | User uploads (`.pt`, meshes, images) |
| `*.pth`, `*.pt`, `*.ckpt` | Model weights |
| `.venv/`, `__pycache__/`, `.pytest_cache/`, `.idea/` | Environment / IDE |

## Do not re-add junk

- Root dumps (`test_legend.png`, `check_nodes_out.txt`, probe `.txt` logs)
- One-off census / compare JSON under `outputs/`
- Personal notes under `notes/`

## After clone (developer setup)

1. Create a venv and `pip install -r requirements.txt`.
2. Place COMSOL / graph data under `data/` and `comsol_models/` as needed (not from git).
3. Optional: pull promoted checkpoints into `outputs/biochem/biochem_gnn/locked/` and `outputs/kinematics/` from your private artifact store.
4. Reference manifests in `data/reference/` describe which checkpoints are canonical.

## Script surface

- **Supported:** only what `scripts/README.md` lists.
- **Archived:** `scripts/archive/` — retired GNODE / clot-ML / T0 / graybox ladders.
- Prefer not adding new one-off `analyze_*.py` / `_print_*.py` to the repo root of `scripts/` unless documented in the README.
