"""General-purpose AnnData sampling utilities for preprocessing."""

import anndata as ad
import numpy as np


def sample(
    adata: ad.AnnData,
    fraction: float | None = None,
    n: int | None = None,
    stratify: str | None = None,
    rng: int = 0,
    copy: bool = False,
    **kwargs,
) -> ad.AnnData:
    """
    Subsample AnnData object with optional stratification.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    stratify : str, optional
        Column in adata.obs to stratify by (for categorical stratified sampling).
    fraction : float, optional
        Fraction of cells to sample (0 < fraction <= 1).
    n : int, optional
        Number of cells to sample (overrides fraction if both given).
    rng : int, default 0
        Random seed for reproducibility.
    copy : bool, default False
        Whether to return a new copy of the data, otherwise returns a view
    kwargs : dict
        Parameters to pass to scanpy.pp.sample

    Returns
    -------
    AnnData
        Subsampled AnnData object.
    """
    obs = adata.obs
    rng = np.random.RandomState(rng)

    if stratify is not None:
        assert stratify in obs.columns, f'stratify column "{stratify}" not found in adata.obs'
        mask = np.zeros(len(obs), dtype=bool)

        if n is not None:
            counts = obs[stratify].value_counts()
            n_per_cat_dict = (counts * (n / len(obs))).round().astype(int).clip(lower=1).to_dict()
        elif fraction is not None and 0 < fraction < 1:
            counts = obs[stratify].value_counts()
            n_per_cat_dict = (counts * fraction).round().astype(int).clip(lower=1).to_dict()
        else:
            return adata

        grouped = obs.groupby(stratify, sort=False, observed=True).indices
        for cat, cat_positions in grouped.items():
            n_cat = min(len(cat_positions), n_per_cat_dict.get(cat, 1))
            chosen = rng.choice(cat_positions, n_cat, replace=False)
            mask[chosen] = True
        subset = adata[mask]
        return subset.copy() if copy else subset

    else:
        # Manual sampling to preserve view/copy semantics (scanpy.pp.sample returns a copy)
        total = len(obs)
        if n is not None and 0 < n < total:
            chosen = rng.choice(total, size=n, replace=False)
            subset = adata[chosen]
            return subset.copy() if copy else subset
        elif fraction is not None and 0 < fraction < 1:
            n_frac = int(round(fraction * total))
            # ensure at least one selected when fraction > 0
            n_frac = max(1, min(n_frac, total - 1))
            chosen = rng.choice(total, size=n_frac, replace=False)
            subset = adata[chosen]
            return subset.copy() if copy else subset
        return adata
