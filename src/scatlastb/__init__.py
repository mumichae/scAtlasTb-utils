from importlib.metadata import version

from . import io, metrics, pipeline, pl, pp, tl

me = metrics

__all__ = [
    "io",
    "pl",
    "pp",
    "tl",
    "metrics",
    "me",
    "pipeline",
]

__version__ = version("scatlastb")
