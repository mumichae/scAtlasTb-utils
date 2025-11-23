import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp


@pytest.fixture
def adata():
    from scipy.stats import nbinom

    np.random.seed(42)  # For reproducibility

    # Parameters for the negative binomial distribution
    n, p = 1000, 0.5
    counts = sp.csr_matrix(nbinom.rvs(n, p, size=(5, 10)))

    adata = ad.AnnData(
        X=counts,
        layers={"counts": counts},
        obs=pd.DataFrame(
            {
                "cell_type": np.random.choice(["T_cell", "B_cell", "Macrophage", "Neutrophil"], size=counts.shape[0]),
                "condition": np.random.choice(["control", "treated"], size=counts.shape[0]),
                "donor_id": np.random.choice(["donor_1", "donor_2", "donor_3"], size=counts.shape[0]),
                "quality_score": np.random.uniform(0.5, 1.0, size=counts.shape[0]),
                "total_counts": np.random.poisson(1000, size=counts.shape[0]),
                "n_genes": np.random.poisson(500, size=counts.shape[0]),
            },
            index=[f"cell_{i}" for i in range(counts.shape[0])],
        ),
    )

    # save raw data
    adata.raw = adata

    # normalize counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


@pytest.fixture
def adata_dask(adata):
    from dask import array as da

    adata.X = da.from_array(adata.X, chunks=(2, -1))
    adata.layers["counts"] = da.from_array(adata.layers["counts"], chunks=(2, -1))
    return adata
