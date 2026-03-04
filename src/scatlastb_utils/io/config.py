import zarr
from dask import config as da_config

da_config.set(**{"array.slicing.split_large_chunks": False})
zarr.default_compressor = zarr.Blosc(shuffle=zarr.Blosc.SHUFFLE)

ALL_SLOTS = ["X", "obs", "var", "obsm", "varm", "obsp", "varp", "layers", "uns", "raw"]


def print_flushed(*args, **kwargs):
    """Print and flush output to ensure it appears immediately."""
    kwargs |= dict(flush=True)
    verbose = kwargs.pop("verbose", True)
    if verbose:
        print(*args, **kwargs)
