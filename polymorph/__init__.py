from importlib import import_module

__version__ = "0.1.0"

from . import utils, io, sources, sims, pipeline

# eager load
__all__ = ["__version__", "utils", "io", "sources", "sims", "pipeline"]


# lazy load
def __getattr__(name: str):
    if name in {}:
        mod = import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(name)
