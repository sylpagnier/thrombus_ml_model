# Publishing policy (HemoRGP)

This repository is meant to be **publicly pushable**: source, docs, and small reference manifests are versioned. Heavy data, checkpoints, and COMSOL models stay on the machine that trains.

## Track in git

| Path | Why |
|------|-----|
| `src/` | Library, training, tools, tests |
| `scripts/` (active) + `scripts/README.md` | Supported launchers |
| `scripts/archive/` | Retired launchers (not the default surface) |
| `docs/` (active) + `docs/archive/` | Design docs + historical notebooks |
| `docs/assets/` | Small README / paper figures (tracked) |
| `data/reference/` | Small baseline / architecture JSON + README |
| `customer_geometries/README.txt` | Inbox instructions only |
| `README.md`, `AGENTS.md`, `requirements.txt`, `pytest.ini` | Project entry |

## Keep local (never push)

| Path | Why |
|------|-----|
| `data/raw/`, `data/processed/`, `data/benchmark/` | Large meshes / graphs / CFD extracts |
| `data/reference_local/` | Sweep leftovers (gitignored) |
| `outputs/` | Checkpoints, logs, viz |
| `comsol_models/` | `.mph` sources |
| `customer_geometries/*` (except README) | User uploads |
| `*.pth`, `*.pt`, `*.ckpt` | Weights |
| `.venv/`, `__pycache__/`, `.pytest_cache/`, `.idea/` | Environment / IDE |

## Do not re-add

- Root dumps (`test_legend.png`, `check_nodes_out.txt`, probe logs)
- One-off census / compare JSON under `outputs/`
- Personal notes under `notes/`

## After clone

1. `pip install -r requirements.txt` (venv recommended).
2. Place COMSOL / graph data under `data/` and `comsol_models/` as needed.
3. Optional: copy promoted checkpoints into `outputs/biochem/biochem_gnn/locked/` and `outputs/kinematics/` from your private artifact store.
4. Use `data/reference/*.json` to see which runs are canonical.

## Script surface

- **Supported:** only what [`scripts/README.md`](../scripts/README.md) lists.
- **Archived:** `scripts/archive/`.
- Prefer not adding one-off `analyze_*.py` / `_print_*.py` to the active `scripts/` root unless documented in that README.
