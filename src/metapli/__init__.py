"""MetaPLI: MPNN and ANP for protein–ligand activity prediction."""

__version__ = "0.1.0"

__all__ = ["MetaPLI", "default_model_config"]


def __getattr__(name):
    if name in __all__:
        from .model import MetaPLI, default_model_config

        return {"MetaPLI": MetaPLI, "default_model_config": default_model_config}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
