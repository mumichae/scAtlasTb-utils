import logging
from collections.abc import Iterable, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


def _categorical_mode(x: pd.Series):
    """Robust mode resolver for categorical-like series.

    Handles categorical and non-categorical dtypes and missing values.
    """
    # Drop NA for mode calculations
    x_non_na = x.dropna()
    if x_non_na.empty:
        return None

    n_vals = len(x_non_na)
    n_unique = x_non_na.nunique()

    # If all values are unique, return the first non-NA value
    if n_unique == n_vals:
        return x_non_na.iloc[0]

    # If series is categorical, leverage codes for performance
    if pd.api.types.is_categorical_dtype(x_non_na.dtype):
        codes = x_non_na.cat.codes.values
        codes = codes[codes >= 0]
        if len(codes) == 0:
            return None
        counts = np.bincount(codes)
        mode_code = int(np.argmax(counts))
        return x_non_na.cat.categories[mode_code]

    # Fallback to value_counts for general types
    return x_non_na.value_counts().index[0]


def _aggregate_obs(
    obs: pd.DataFrame,
    group_key: str,
    group_order: Iterable,
):
    obs = obs.copy()
    bool_columns = obs.select_dtypes(include=["bool"]).columns.tolist()
    num_columns = obs.select_dtypes(include=["number"]).columns.tolist()
    cat_columns = obs.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    cat_columns = [col for col in cat_columns if col != group_key]

    # Cast category-like columns to categorical for aggregation efficiency
    if cat_columns:
        obs[cat_columns] = obs[cat_columns].astype("category")

    if bool_columns or num_columns or cat_columns:
        g = obs.groupby(group_key, observed=True)
        df = pd.concat(
            [
                g[bool_columns + num_columns].mean(),
                g[cat_columns].agg(_categorical_mode),
            ],
            axis=1,
        )
    else:
        df = obs.groupby(group_key, observed=True).first()

    if bool_columns:
        df[bool_columns] = df[bool_columns] > 0.5  # mode for bool

    logging.info("Set aggregated metadata order...")
    df = df.loc[group_order]
    df[group_key] = df.index.astype(str)

    logging.info("Convert metadata to categorical...")
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col].dtype):
            df[col] = df[col].astype(str).astype("category")

    return df


def pseudobulk(
    adata: ad.AnnData,
    group_key: str | Sequence[str],
    agg: str = "sum",
    sep: str = "--",
    group_cols=None,
    min_cells: int = 0,
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
    if isinstance(group_key, (list, tuple)) or isinstance(group_key, Sequence) and not isinstance(group_key, str):
        group_keys = list(group_key)
        for k in group_keys:
            if k not in adata.obs.columns:
                raise KeyError(f"group key '{k}' not found in adata.obs")
        new_key = sep.join(group_keys)
        if new_key in adata.obs.columns:
            raise KeyError(f"generated group key '{new_key}' already exists in adata.obs")
        adata.obs[new_key] = adata.obs[group_keys].astype(str).apply(lambda x: sep.join(x), axis=1)
        group_key = new_key
    else:
        if group_key not in adata.obs.columns:
            raise KeyError(f"group key '{group_key}' not found in adata.obs")

    # Determine which obs columns to keep, preserving order
    if group_cols is None:
        group_cols = adata.obs.columns.tolist()

    # Ensure group_key present and preserve original order
    group_cols = list(dict.fromkeys(group_cols + [group_key]))

    # filter groups with too few cells
    value_counts = adata.obs[group_key].value_counts(dropna=True)
    value_counts = value_counts[value_counts >= min_cells]

    # subset data (safe copy)
    adata = adata[adata.obs[group_key].isin(value_counts.index)].copy()
    adata.obs = adata.obs[group_cols].copy()

    logging.info(f"Aggregate {value_counts.shape[0]} pseudobulks...")
    pb_adata = sc.get.aggregate(adata, by=group_key, func=agg, **kwargs)

    logging.info(f"Aggregate {adata.obs.shape[1]} metadata columns...")
    obs = _aggregate_obs(adata.obs, group_key, group_order=pb_adata.obs_names)
    obs["n_agg"] = pb_adata.obs["n_obs_aggregated"].values
    logging.debug("Aggregated obs:\n%s", obs)

    return ad.AnnData(
        X=pb_adata.layers[agg],
        obs=obs,
        var=adata.var.copy(),
    )
