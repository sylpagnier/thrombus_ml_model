"""Canonical import alias for the biochem deploy stack.

Implementation lives in ``src.biochem_gnn``; this package re-exports the public API
so ``from src.biochem_deploy import BiochemGNN`` matches docs/nomenclature.
"""

from src.biochem_gnn import *  # noqa: F403
from src.biochem_gnn import __all__  # noqa: F401
