import logging
from collections.abc import Iterable, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from scatlastb_utils.utils import ensure_sparse

logging.basicConfig(level=logging.INFO)


def _get_group_codes(series: pd.Series, order: Iterable):
    """Return integer codes for `series` aligned to `order`.

    Returns a tuple `(codes, n_groups, order_list)` where `codes` is an
    ndarray of integer group codes (-1 for NA), `n_groups` is the number
    of groups (len(order_list)), and `order_list` is the list form of
    `order`.
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
    df[group_key] = df.index.astype(str)

    # Convert metadata to categorical
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(str).astype("category")

    return df


def pseudobulk(
    adata: ad.AnnData,
    group_key: str | Sequence[str],
    agg: str = "sum",
    sep: str = "--",
    group_cols=None,
    min_cells: int = 0,
    force_sparse: bool = False,
    **kwargs,
) -> ad.AnnData:
    """Pseudobulk an AnnData object and its metadata.

    Parameters
    ----------
    adata : AnnData
        AnnData to aggregate (not modified in-place).
    group_key : str or list of str
        Column name or list of column names in `adata.obs` to group by.
    agg : str, optional
        Aggregation function name passed to `scanpy.get.aggregate` (e.g. 'sum').
    sep : str, optional
        Separator used when joining multiple group keys into a single group label.
    group_cols : list of str, optional
        List of obs columns to preserve/aggregate; defaults to all.
    min_cells : int, optional
        Minimum cells per group to keep (>=0).
    kwargs : dict
        Additional arguments passed to `scanpy.get.aggregate`.

    Returns
    -------
    AnnData
        Pseudobulked AnnData object with aggregated expression and metadata.
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

    logging.info(f"Aggregate {value_counts.shape[0]} pseudobulks...")
    pb_adata = sc.get.aggregate(adata, by=group_key, func=agg, mask=mask, axis=0, **kwargs)
    if force_sparse:
        pb_adata = ensure_sparse(pb_adata)

    logging.info(f"Aggregate {len(group_cols)} metadata columns...")
    obs = _aggregate_obs(adata.obs, group_key, group_order=pb_adata.obs_names, columns=group_cols)
    obs["n_agg"] = pb_adata.obs["n_obs_aggregated"].values
    obs = obs[group_cols + ["n_agg"]].copy()  # reorder columns
    logging.debug("Aggregated obs:\n%s", obs)

    return ad.AnnData(
        X=pb_adata.layers[agg],
        obs=obs,
        var=adata.var.copy(),
    )
