from typing import Any

import pandas as pd
from anndata import AnnData


def majority_reference(
    adata: AnnData, reference_key: str, cluster_key: str, crosstab_kwargs: dict[str, Any] | None = None
) -> AnnData:
    """
    Annotate clusters in an AnnData object by assigning the most common reference label to each cluster.

    For each cluster (from ``cluster_key``), this function determines the majority label (from ``reference_key``)
    using a crosstabulation, and annotates each cell with its cluster's majority label. It also computes
    the confidence for each cluster, defined as the fraction of cells in the cluster that match the majority label.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (typically single-cell data) with observations in ``adata.obs``.
    reference_key : str
        Column name in ``adata.obs`` containing reference labels (e.g., cell type annotations).
    cluster_key : str
        Column name in ``adata.obs`` containing cluster assignments.
    crosstab_kwargs : dict, optional
        Additional keyword arguments to pass to ``pd.crosstab`` for customizing the crosstabulation.

    Returns
    -------
    AnnData
        The input AnnData object with two new columns added to ``adata.obs``:
            - ``adata.obs["majority_reference"]``: Categorical column with the majority reference label per cluster.
            - ``adata.obs["majority_reference_confidence"]``: Fraction of cells in each cluster matching the majority label.

    Notes
    -----
    - Cells with missing or NaN reference labels are handled by ``pd.crosstab``, depending on the provided ``crosstab_kwargs``.
    - The confidence per cluster is calculated as::

        confidence = (# cells in cluster with majority label) / (total # cells in cluster)

    - In case of a tie, pandas ``idxmax`` returns the first label encountered.

    Example
    -------
    >>> print(adata.obs)
       cell_type cluster
    0   T-cell    A
    1   T-cell    A
    2   B-cell    A
    3   B-cell    B
    4   B-cell    B
    5   T-cell    B
    6   NK-cell   C
    7   T-cell    C
    8   NK-cell   C
    9   NK-cell   C
    10  B-cell    D
    11  B-cell    D
    12  B-cell    D

    >>> adata = majority_reference(adata, reference_key="cell_type", cluster_key="cluster")
    >>> print(adata.obs)
       cell_type cluster majority_reference  majority_reference_confidence
    0   T-cell      A        T-cell              0.67
    1   T-cell      A        T-cell              0.67
    2   B-cell      A        T-cell              0.67
    3   B-cell      B        B-cell              0.67
    4   B-cell      B        B-cell              0.67
    5   T-cell      B        B-cell              0.67
    6   NK-cell     C        NK-cell             0.5
    7   T-cell      C        NK-cell             0.5
    8   NK-cell     C        NK-cell             0.5
    9   NK-cell     C        NK-cell             0.5
    10  B-cell     D        B-cell              1.0
    11  B-cell     D        B-cell              1.0
    12  B-cell     D        B-cell              1.0
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
