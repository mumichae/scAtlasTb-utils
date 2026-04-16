"""General-purpose utility functions shared across submodules."""

from __future__ import annotations

import re
from contextlib import nullcontext

import anndata as ad
import numpy as np
from dask import array as da
from tqdm.dask import TqdmCallback


def _sanitize_default_file_name(value):
    if isinstance(value, list):
        name = "_".join(str(entry) for entry in value)
    else:
        name = str(value)
    # Define unwanted characters to replace with '_'
    unwanted = '/\\[](){}\'"",:;?<>|=+*&^%$#@!~` \t\n\r'
    trans = str.maketrans(dict.fromkeys(unwanted, "_"))
    name = name.translate(trans)
    # Only allow alphanumeric, dash, and underscore
    name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("._")
    assert name, "Sanitized file name is empty"
    return name


def parse_gene_names(adata: ad.AnnData, gene_list: list) -> list:
    """Match gene names or regex patterns against ``adata.var_names``.

    Each entry in *gene_list* is first looked up as an exact match in
    ``adata.var_names``.  Entries that are not found exactly are treated as
    regex patterns and matched via ``str.contains`` against all variable names.

    Parameters
    ----------
    adata
        Annotated data matrix whose ``.var_names`` are searched.
    gene_list
        List of gene identifiers or regex patterns.

    Returns
    -------
    list
        Deduplicated list of variable names that match any entry in
        *gene_list*.
    """
    var_names = adata.var_names.astype(str)
    exact = [g for g in gene_list if g in var_names]
    patterns = [str(g) for g in gene_list if g not in var_names]
    if patterns:
        mask = var_names.str.contains(pat="|".join(re.escape(p) for p in patterns))
        exact += var_names[mask].tolist()
    return list(dict.fromkeys(exact))


def remove_outliers(
    adata: ad.AnnData, extrema: str = "max", factor: float = 10, rep: str = "X_umap", copy: bool = False
) -> ad.AnnData:
    """Remove outliers from an ``.obsm`` embedding representation.

    Cells whose embedding coordinate deviates more than *factor* times the
    mean absolute value of the extreme coordinates are excluded.  Set
    *factor* to ``0`` to disable outlier removal entirely.

    Parameters
    ----------
    adata
        Annotated data matrix.
    extrema
        Which extreme to use for outlier detection: ``"max"`` or ``"min"``.
    factor
        Multiplier applied to the mean absolute value to set the threshold.
        ``0`` disables filtering.
    rep
        Key in ``adata.obsm`` for the embedding (default ``"X_umap"``).
    copy
        Whether to return a new copy of the data or just a view

    Returns
    -------
    AnnData view with outlier cells removed.
    """
    if factor == 0:
        return adata
    coords = adata.obsm[rep]
    if extrema == "max":
        abs_values = np.abs(coords.max(axis=1))
    elif extrema == "min":
        abs_values = np.abs(coords.min(axis=1))
    else:
        raise ValueError(f"extrema must be 'max' or 'min', got {extrema!r}")

    outlier_mask = abs_values < factor * abs_values.mean()

    adata = adata[outlier_mask]

    if copy:
        adata = adata.copy()

    return adata


def apply_layers(
    adata: ad.AnnData,
    func: callable,
    layers: list | str | bool = None,
    verbose: bool = False,
    **kwargs,
) -> ad.AnnData:
    """Apply *func* to specified layers of an AnnData object.

    Parameters
    ----------
    adata
        Annotated data matrix.
    func
        Function to apply to each layer array.
    layers
        Which layers to process.  ``None`` / ``True`` → ``["X", "raw"]`` plus
        all named layers.  A string selects a single layer.  ``False`` is a
        no-op.
    verbose
        Print progress messages when ``True``.
    **kwargs
        Forwarded to *func*.
    """
    if layers is None or layers is True:
        layers = ["X", "raw"] + list(adata.layers.keys())
    elif isinstance(layers, str):
        layers = [layers]
    elif layers is False:
        return adata

    for layer in layers:
        if verbose:
            print(f"Apply function {func.__name__} to {layer}...", flush=True)
        if layer == "X":
            adata.X = func(adata.X, **kwargs)
        elif layer in adata.layers:
            adata.layers[layer] = func(adata.layers[layer], **kwargs)
        elif layer in adata.obsm:
            adata.obsm[layer] = func(adata.obsm[layer], **kwargs)
        elif layer == "raw":
            if adata.raw is None:
                continue
            adata_raw = adata.raw.to_adata()
            adata_raw.X = func(adata.raw.X, **kwargs)
            adata.raw = adata_raw
        elif verbose:
            print(f"Layer {layer} not found, skipping...", flush=True)
    return adata


def dask_compute(
    adata: ad.AnnData,
    layers: list | str = None,
    verbose: bool = True,
    **kwargs,
) -> ad.AnnData:
    """Compute Dask arrays inside an AnnData object.

    Parameters
    ----------
    adata
        Annotated data matrix.
    layers
        Layers to compute (see :func:`apply_layers` for accepted values).
        ``None`` computes ``"X"``, ``"raw"``, and all named layers.
    verbose
        Show a ``tqdm``-based Dask progress bar when ``True``.
    **kwargs
        Forwarded to :func:`apply_layers`.
    """
    if adata.is_view:
        adata = adata.copy()

    def compute_layer(x, persist=False):
        if not isinstance(x, da.Array):
            return x
        if any(dim == 0 for dim in x.shape):
            return np.empty(x.shape, dtype=x.dtype)
        context = TqdmCallback(desc="Dask compute", miniters=10, mininterval=5) if verbose else nullcontext()
        with context:
            if persist:
                x = x.persist()
            x = x.compute()
        return x

    return apply_layers(adata, func=compute_layer, layers=layers, verbose=verbose, **kwargs)
