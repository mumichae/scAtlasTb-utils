import logging
from collections.abc import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


def _categorical_mode(x):
    n_vals = len(x)
    n_unique = x.nunique()

    # If all values are unique, just return the first one
    if n_unique == n_vals:
        return x.iloc[0]

    if n_unique / n_vals < 0.8:  # threshold can be tuned
        codes = x.cat.codes.values
        # Handle missing values (-1 codes)
        codes = codes[codes >= 0]
        if len(codes) == 0:
            return None
        counts = np.bincount(codes)
        mode_code = np.argmax(counts)
        return x.cat.categories[mode_code]

    # If very high uniqueness ratio, use value_counts (avoid bincount)
    return x.value_counts().index[0]


def _aggregate_obs(
    obs: pd.DataFrame,
    group_key: str,
    group_order: Iterable,
):
    bool_columns = obs.select_dtypes("bool").columns.tolist()
    num_columns = obs.select_dtypes("number").columns.tolist()
    cat_columns = obs.select_dtypes(exclude=["number", "bool"]).columns
    cat_columns = [col for col in cat_columns if col != group_key]
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
    df[bool_columns] = df[bool_columns] > 0.5  # mode for bool

    logging.info("Set aggregated metadata order...")
    df = df.loc[group_order]
    df[group_key] = df.index.astype(str)

    logging.info("Convert metadata to categorical...")
    # convert category columns to string
    for col in df.columns:
        if df[col].dtype.name == "category":
            df[col] = df[col].astype(str).astype("category")

    return df


def pseudobulk(
    adata,
    group_key,
    agg="sum",
    dtype="float32",
    sep="--",
    group_cols=None,
    min_cells: int = 0,
):
    """Pseudobulk an AnnData object by aggregating the expression matrix and metadata according to a group key."""
    if isinstance(group_key, list):
        group_keys = group_key
        group_key = sep.join(group_keys)
        adata.obs[group_key] = adata.obs[group_keys].astype(str).apply(lambda x: sep.join(x), axis=1)

    if group_cols is None:
        group_cols = adata.obs.columns.tolist()
    group_cols = list(set(group_cols + [group_key]))

    # filter groups with too few cells
    value_counts = adata.obs[group_key].value_counts(dropna=True)
    value_counts = value_counts[value_counts >= min_cells]

    # subset data
    adata.obs = adata.obs[group_cols].copy()
    adata = adata[adata.obs[group_key].isin(value_counts.index)].copy()

    logging.info("Aggregate %d pseudobulks...", value_counts.shape[0])
    pb_adata = sc.get.aggregate(adata, by=group_key, func=agg)

    logging.info("Aggregate %d metadata columns...", adata.obs.shape[1])
    obs = _aggregate_obs(adata.obs, group_key, group_order=pb_adata.obs_names)
    obs["n_agg"] = pb_adata.obs["n_obs_aggregated"]
    logging.debug("Aggregated obs:\n%s", obs)

    return ad.AnnData(X=pb_adata.layers[agg].astype(dtype), obs=obs, var=adata.var.copy())
