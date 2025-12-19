from importlib import import_module
from importlib.metadata import version
from types import ModuleType

__version__ = version("lattica-polymorph")

from . import pipeline, sources, utils

# Eager load
__all__ = ["__version__", "utils", "sources", "pipeline"]


# Lazy load
def __getattr__(name: str) -> ModuleType:
    if name in {}:
        mod = import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(name)
