from typing import Any

import pandas as pd
from anndata import AnnData


def majority_reference(
    adata: AnnData, reference_key: str, cluster_key: str, crosstab_kwargs: dict[str, Any] | None = None
) -> AnnData:
    """Annotate a cluster by the most common label

    Annotate clusters in an AnnData object by assigning the most common reference label to each cluster.
    This function determines the majority label (from `reference_key`) for each cluster (from `cluster_key`)
    using a crosstabulation, and annotates each cell with its cluster's majority label. It also computes
    per-cluster counts of reference labels and the confidence (fraction of cells matching the majority label)
    for each cluster.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (typically single-cell data) with observations in `adata.obs`.
    reference_key : str
        Column name in `adata.obs` containing reference labels (e.g., cell type annotations).
    cluster_key : str
        Column name in `adata.obs` containing cluster assignments.
    crosstab_kwargs : dict, optional
        Additional keyword arguments to pass to `pd.crosstab` for customizing the crosstabulation.

    Returns
    -------
    The function modifies `adata.obs` in place by adding:
        - 'majority_reference': Categorical column with the majority reference label per cluster.

    Side Effects
    ------------
    - Adds a new column 'majority_reference' to `adata.obs` with the majority label for each cluster.
    - Computes per-cluster counts and confidence values (not returned, but can be used for further analysis).

    Notes
    -----
    - Cells with missing or NaN reference labels are handled by `pd.crosstab`, depending on the provided `crosstab_kwargs`.
    - The confidence per cluster is calculated as the fraction of cells in each cluster that match the majority label.

    Examples
    --------
    >>> majority_reference(adata, reference_key="cell_type", cluster_key="louvain")
    >>> adata.obs["majority_reference"].value_counts()
    """
    if crosstab_kwargs is None:
        crosstab_kwargs = {}


    crosstab = pd.crosstab(adata.obs[reference_key], adata.obs[cluster_key], **crosstab_kwargs)
    map_majority = crosstab.idxmax(axis=0)

    # get majority label per cluster
    adata.obs["majority_reference"] = pd.Categorical(
        adata.obs[cluster_key].map(map_majority),
        categories=map_majority.dropna().unique(),
    )

    # Calculate confidence: fraction of cells in each cluster matching the majority label
    # For each cluster, count how many cells have the majority label, divided by total cells in the cluster
    cluster_counts = adata.obs[cluster_key].value_counts()
    majority_label_counts = adata.obs.groupby(cluster_key)[reference_key].apply(
        lambda x: (x == map_majority[x.name]).sum()
    )
    confidence_per_cluster = (majority_label_counts / cluster_counts).fillna(0.0)
    adata.obs["majority_reference_confidence"] = adata.obs[cluster_key].map(confidence_per_cluster).astype(float)

    return adata
