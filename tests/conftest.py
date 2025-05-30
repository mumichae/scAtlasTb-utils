import anndata as ad
import numpy as np
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
