import importlib
import logging
from collections.abc import Iterable, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from dask import array as da
from dask import config as dask_config
from scipy import sparse as sp

from scatlastb_utils.utils import ensure_sparse

logging.basicConfig(level=logging.INFO)


def _get_group_codes(series: pd.Series, order: Iterable):
    """Return integer group codes aligned to `order`.

    Parameters
    ----------
    series : pandas.Series
        Series of group labels.
    order : Iterable
        Desired ordering of groups (categories).

    Returns
    -------
    codes : ndarray
        Integer codes for `series` (same length as `series`), -1 for NA.
    n_groups : int
        Number of groups (len of `order`).
    order_list : list
        The provided `order` converted to a list.
    """
    order_list = list(order)
    grp = pd.Categorical(series, categories=order_list)
    codes = grp.codes.astype(np.int64)
    n_groups = len(order_list)
    return codes, n_groups, order_list


def _mode_for_column(ser: pd.Series, group_codes: np.ndarray, n_groups: int):
    """Compute per-group categorical mode for one categorical Series.

    Packs (group, value) pairs into 64-bit integers, counts unique pairs
    with np.unique, then selects the most frequent value per group.
    """
    ser_cat = ser.astype("category")
    cat_codes = ser_cat.cat.codes.to_numpy(dtype=np.int64)

    valid_mask = (group_codes >= 0) & (cat_codes >= 0)
    if not valid_mask.any():
        return np.array([None] * n_groups, dtype=object)

    group_codes_valid = group_codes[valid_mask].astype(np.int64)
    value_codes_valid = cat_codes[valid_mask]

    # Pack (group, value) into a single int64 for counting
    packed_keys = (group_codes_valid << 32) | value_codes_valid
    unique_packed, pair_counts = np.unique(packed_keys, return_counts=True)

    # np.unique returns sorted keys, so group IDs are already in order
    decoded_group_ids = unique_packed >> 32
    decoded_value_codes = unique_packed & 0xFFFFFFFF

    # Locate boundaries between groups
    present_group_ids, start_indices = np.unique(decoded_group_ids, return_index=True)
    end_indices = np.append(start_indices[1:], decoded_group_ids.size)

    # For each present group pick the value code with the highest count
    mode_value_codes = np.full(n_groups, -1, dtype=np.int64)
    for pg, start, end in zip(present_group_ids, start_indices, end_indices, strict=True):
        seg_counts = pair_counts[start:end]
        mode_value_codes[pg] = decoded_value_codes[start + seg_counts.argmax()]

    # Map integer codes back to category labels
    categories = ser_cat.cat.categories
    mode_values = np.where(mode_value_codes >= 0, categories[mode_value_codes], None)
    return mode_values


def _aggregate_numeric_bool(obs: pd.DataFrame, group_key: str, bool_columns: list, num_columns: list) -> pd.DataFrame:
    """Aggregate boolean and numeric columns by group (mean).

    Returns an empty DataFrame if no columns provided.
    """
    if not (bool_columns or num_columns):
        return pd.DataFrame()
    g = obs.groupby(group_key, observed=True)
    cols = bool_columns + num_columns
    return g[cols].mean()


def _aggregate_categorical(obs: pd.DataFrame, group_key: str, group_order: Iterable, cat_columns: list) -> pd.DataFrame:
    """Aggregate categorical columns by computing per-group mode for each column.

    Uses `_group_codes` and `_mode_for_column` helpers. Returns a DataFrame
    indexed by `group_order` with one column per categorical input column.
    """
    if not cat_columns:
        return pd.DataFrame()

    logging.info("Aggregating %d categorical columns...", len(cat_columns))
    group_codes, n_groups, order_list = _get_group_codes(obs[group_key], group_order)
    df = pd.DataFrame(index=order_list)
    for i, col in enumerate(cat_columns, 1):
        logging.debug("Processing categorical column %d/%d: %s", i, len(cat_columns), col)
        if obs[col].value_counts().max() == 1:
            logging.debug("Column '%s' is unique per group, skipping mode calculation", col)
            modes = obs.groupby(group_key)[col].first().reindex(order_list).values
        else:
            modes = _mode_for_column(obs[col], group_codes, n_groups)
        df[col] = pd.Series(modes, index=order_list)

    return df


def _aggregate_obs(
    obs: pd.DataFrame,
    group_key: str,
    group_order: Iterable,
    columns: list | None = None,
):
    """Aggregate observation metadata by group.

    Parameters
    ----------
    obs : pandas.DataFrame
        Observation dataframe (typically ``adata.obs``).
    group_key : str
        Column name in ``obs`` to group by.
    group_order : Iterable
        Ordered list of group labels to use as the result index.
    columns : list or None
        Which columns of ``obs`` to aggregate. If ``None``, all columns
        are used.

    Returns
    -------
    pandas.DataFrame
        Aggregated metadata indexed by ``group_order``. Numeric and
        boolean columns are averaged, categorical columns use per-group
        mode, and an ``n_agg`` column with per-group cell counts is
        added. Categorical dtypes are converted back to categories where
        possible.
    """
    if columns is None:
        columns = obs.columns.tolist()

    obs = obs[columns].copy()
    bool_columns = obs.select_dtypes(include=["bool"]).columns.tolist()
    num_columns = obs.select_dtypes(include=["number"]).columns.tolist()
    cat_columns = obs.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    cat_columns = [col for col in cat_columns if col != group_key]

    # Cast category-like columns to categorical for aggregation efficiency
    if cat_columns:
        obs[cat_columns] = obs[cat_columns].astype("category")

    if bool_columns or num_columns or cat_columns:
        numeric_cols = _aggregate_numeric_bool(obs, group_key, bool_columns, num_columns)
        cat_cols = _aggregate_categorical(obs, group_key, group_order, cat_columns)
        df = pd.concat([numeric_cols, cat_cols], axis=1)
    else:
        df = obs.groupby(group_key, observed=True).first()

    if bool_columns:
        df[bool_columns] = df[bool_columns] > 0.5  # mode for bool

    # Set aggregated metadata order
    df = df.loc[group_order]
    df.index.name = None
    df[group_key] = df.index.astype(str)

    # compute per-group counts (n_agg) and add to aggregated obs
    counts = obs.groupby(group_key, observed=True).size()
    df["n_agg"] = counts.reindex(group_order).fillna(0).astype(int)

    # Convert metadata to categorical
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(str).astype("category")

    return df


def _get_pseudobulk_matrix_dask_legacy(adata, group_key, agg, mask, layer, force_sparse, dtype=None, **kwargs):
    """Aggregate a dask-backed matrix by groups using a legacy path.

    This implementation sorts and rechunks the dask array so that groups
    are contiguous, then maps a block-wise aggregation function across
    the grouped chunks.

    Returns a tuple `(pseudobulks, groups)` where `pseudobulks` is a
    dask array or sparse array (group x features) and `groups` is the
    ordered list of group labels.
    """

    def aggregate(x, agg, force_sparse=True, dtype=None):
        if agg == "sum":
            result = x.sum(0)
        elif agg == "mean":
            result = x.mean(0)
        else:
            raise ValueError(f'invalid aggregation method "{agg}"')
        if force_sparse:
            return sp.csr_matrix(result, dtype=dtype)
        return np.asarray(result, dtype=dtype)

    dtype = dtype or np.float32

    # choose data source (prefer named layer when provided)
    X = adata.X if layer is None or layer == "X" else adata.layers[layer]
    if mask is None:
        group_series = adata.obs[group_key]
    else:
        group_series = adata.obs.loc[mask, group_key]
        X = X[mask.values]

    value_counts = group_series.value_counts(dropna=True)
    groups = value_counts.index.sort_values()  # sort alphabetically so argsort and chunk_sizes agree

    group_col = pd.Categorical(group_series, categories=groups, ordered=True)
    sorted_idx = np.argsort(group_col.codes, kind="stable")
    chunk_sizes = tuple(value_counts.reindex(groups).values)

    logging.info(f'Sort and rechunk dask array by "{group_key}"...')
    with dask_config.set(**{"array.slicing.split_large_chunks": False}):
        X = X[sorted_idx].rechunk((chunk_sizes, -1))
        meta = sp.csr_matrix((0, 0), dtype=dtype) if force_sparse else np.empty((0, 0), dtype=dtype)
        pseudobulks = X.map_blocks(
            aggregate,
            agg,
            force_sparse=force_sparse,
            dtype=dtype,
            chunks=((1,) * len(chunk_sizes), X.shape[1]),
            meta=meta,
        )
    assert pseudobulks.shape[0] == len(groups)
    return pseudobulks, groups


def _get_pseudobulk_matrix(adata, group_key, agg, mask, layer, force_sparse, dtype, use_legacy=False, **kwargs):
    """Dispatch to the appropriate pseudobulk matrix implementation.

    Chooses a legacy dask-based implementation when the input is a
    dask array and `use_legacy` is True (or Scanpy version is older).

    Returns
    -------
    matrix : array-like
        Aggregated matrix (groups x features). May be a dask array or
        sparse matrix depending on `force_sparse` and backend.
    groups : Index-like
        Ordered group labels corresponding to the rows of `matrix`.
    """
    use_legacy |= importlib.metadata.version("scanpy") < "1.12"

    matrix = adata.layers[layer] if layer is not None else adata.X
    if isinstance(matrix, da.Array) and use_legacy:
        return _get_pseudobulk_matrix_dask_legacy(
            adata, group_key, agg, mask, layer, force_sparse, dtype=dtype, **kwargs
        )

    pb_adata = sc.get.aggregate(adata, by=group_key, func=agg, mask=mask, layer=layer, axis=0, **kwargs)
    if force_sparse:
        pb_adata = ensure_sparse(pb_adata)
    return pb_adata.layers[agg], pb_adata.obs_names


def pseudobulk(
    adata: ad.AnnData,
    group_key: str | Sequence[str],
    agg: str = "sum",
    sep: str = "--",
    group_cols=None,
    layer: str | None = None,
    min_cells: int = 2,
    force_sparse: bool = True,
    dtype: str | np.dtype = "float32",
    use_legacy: bool = False,
    **kwargs,
) -> ad.AnnData:
    """Aggregate an ``AnnData`` object into pseudobulk samples.

    Parameters
    ----------
    adata : AnnData
        Input annotated data matrix (function works on a copy).
    group_key : str or sequence of str
        Column name(s) in ``adata.obs`` to group by. If a sequence is
        provided the keys are concatenated with ``sep`` to form a single
        group label column.
    agg : str, optional
        Aggregation function name forwarded to ``scanpy.get.aggregate``
        (common values: ``"sum"``, ``"mean"``).
    sep : str, optional
        Separator used when joining multiple group keys.
    group_cols : list-like, optional
        Which ``adata.obs`` columns to preserve/aggregate. Defaults to
        all columns.
    layer : str or None, optional
        Name of the layer to use for the expression matrix. If ``None``
        uses ``adata.X``. When provided, both legacy and modern paths
        prefer the named layer for aggregation.
    min_cells : int, optional
        Minimum number of cells required for a group to be kept.
    force_sparse : bool, optional
        If True, attempt to return the aggregated matrix in a sparse
        representation when appropriate.
    dtype : str or numpy.dtype, optional
        Data-type to use for aggregation results.
    use_legacy : bool, optional
        Force the legacy dask-backed aggregation path when True.
    **kwargs
        Forwarded to ``scanpy.get.aggregate``.

    Returns
    -------
    AnnData
        New ``AnnData`` whose ``X`` contains the aggregated matrix
        (groups x features) and whose ``obs`` contains aggregated
        metadata (including an ``n_agg`` column with per-group cell
        counts).

    Notes
    -----
    The function filters out groups with fewer than ``min_cells``
    before aggregation. If multiple ``group_key`` values are provided
    they are joined using ``sep`` and the combined label is used for
    grouping.
    """
    adata = adata.copy()

    # Normalize group_key to a single column name
    logging.info(f"Processing group key(s) {group_key}...")
    if isinstance(group_key, (list, tuple, Sequence)) and not isinstance(group_key, str):
        group_keys = list(group_key)
        missing = pd.Index(group_keys).difference(adata.obs.columns)
        if missing.any():
            raise KeyError(f"group key(s) {missing.tolist()} not found in adata.obs")
        group_key = sep.join(group_keys)
        adata.obs[group_key] = adata.obs[group_keys].astype(str).agg(sep.join, axis=1)

    # Determine which obs columns to keep, preserving order
    if group_cols is None:
        group_cols = adata.obs.columns.tolist()

    # Ensure group_key present and preserve original order
    group_cols = list(dict.fromkeys(group_cols + [group_key]))

    # filter groups with too few cells
    mask = None
    value_counts = adata.obs[group_key].value_counts(dropna=True)

    if min_cells > 0:
        logging.info(f"Filtering groups with at least {min_cells} cells...")
        value_counts = value_counts[value_counts >= min_cells]
        mask = adata.obs[group_key].isin(value_counts.index)
        logging.info(f"Remove {mask.sum()} cells from groups with fewer than {min_cells} cells")

    logging.info(f"Aggregate {value_counts.shape[0]} pseudobulks...")
    pseudobulks, groups = _get_pseudobulk_matrix(
        adata, group_key, agg, mask, layer, force_sparse, dtype, use_legacy=use_legacy, **kwargs
    )

    logging.info(f"Aggregate {len(group_cols)} metadata columns...")
    obs = _aggregate_obs(adata.obs, group_key, group_order=groups, columns=group_cols)
    logging.debug("Aggregated obs:\n%s", obs)

    return ad.AnnData(X=pseudobulks, obs=obs, var=adata.var.copy())
