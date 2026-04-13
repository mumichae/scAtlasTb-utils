"""General-purpose AnnData sampling utilities for preprocessing."""

import anndata as ad
import numpy as np
import scanpy as sc


def sample(
    adata: ad.AnnData,
    frac: float | None = None,
    n: int | None = None,
    stratify: str | None = None,
    random_state: int = 0,
) -> ad.AnnData:
    """
    Subsample AnnData object with optional stratification.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    stratify : str, optional
        Column in adata.obs to stratify by (for categorical stratified sampling).
    frac : float, optional
        Fraction of cells to sample (0 < frac <= 1).
    n : int, optional
        Number of cells to sample (overrides frac if both given).
    random_state : int, default 0
        Random seed for reproducibility.

    Returns
    -------
    AnnData
        Subsampled AnnData object.
    """
    obs = adata.obs
    rng = np.random.RandomState(random_state)

    if stratify is not None and stratify in obs.columns:
        mask = np.zeros(len(obs), dtype=bool)

        if n is not None:
            counts = obs[stratify].value_counts()
            n_per_cat_dict = (counts * (n / len(obs))).round().astype(int).clip(lower=1).to_dict()
        elif frac is not None and 0 < frac < 1:
            counts = obs[stratify].value_counts()
            n_per_cat_dict = (counts * frac).round().astype(int).clip(lower=1).to_dict()
        else:
            return adata

        grouped = obs.groupby(stratify, sort=False).indices
        for cat, cat_positions in grouped.items():
            n_cat = min(len(cat_positions), n_per_cat_dict.get(cat, 1))
            chosen = rng.choice(cat_positions, n_cat, replace=False)
            mask[chosen] = True

        return adata[mask]

    else:
        if n is not None and n < len(obs):
            return sc.pp.subsample(adata, n_obs=n, random_state=random_state, copy=True)
        elif frac is not None and 0 < frac < 1:
            return sc.pp.subsample(adata, fraction=frac, random_state=random_state, copy=True)
        else:
            return adata
