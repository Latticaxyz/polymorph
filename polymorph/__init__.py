from importlib.metadata import version

__version__ = version("lattica-polymorph")

from . import pipeline, sources, utils

__all__ = ["__version__", "utils", "sources", "pipeline"]
