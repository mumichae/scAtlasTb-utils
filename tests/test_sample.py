import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scatlastb_utils.pp.sample import sample


@pytest.fixture
def adata_categorical():
    obs = pd.DataFrame({"celltype": ["A"] * 50 + ["B"] * 30 + ["C"] * 20, "batch": np.repeat([1, 2], 50)})
    X = np.random.randn(100, 5)
    return ad.AnnData(X=X, obs=obs)


def test_stratified_frac(adata_categorical):
    adata_sub = sample(adata_categorical, stratify="celltype", frac=0.2, random_state=42)
    counts = adata_sub.obs["celltype"].value_counts()
    # Should have at least 1 per category, and roughly 20% of each
    assert all(counts >= 1)
    assert set(counts.index) == {"A", "B", "C"}
    assert abs(counts["A"] - 10) <= 2
    assert abs(counts["B"] - 6) <= 2
    assert abs(counts["C"] - 4) <= 2


def test_stratified_n(adata_categorical):
    adata_sub = sample(adata_categorical, stratify="celltype", n=15, random_state=42)
    counts = adata_sub.obs["celltype"].value_counts()
    # Should have at least 1 per category, sum to 15
    assert all(counts >= 1)
    assert set(counts.index) == {"A", "B", "C"}
    assert adata_sub.n_obs == 15


def test_random_frac(adata_categorical):
    adata_sub = sample(adata_categorical, frac=0.1, random_state=42)
    assert adata_sub.n_obs == 10


def test_random_n(adata_categorical):
    adata_sub = sample(adata_categorical, n=7, random_state=42)
    assert adata_sub.n_obs == 7


def test_no_sampling(adata_categorical):
    adata_sub = sample(adata_categorical)
    assert adata_sub.n_obs == 100
