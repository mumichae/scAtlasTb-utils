import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scatlastb_utils.tl.majority_voting import majority_reference


def make_adata(cluster, cell_type):
    n = len(cell_type)
    obs = pd.DataFrame(
        {"cluster": np.array(cluster), "cell_type": np.array(cell_type)}, index=[f"cell_{i}" for i in range(n)]
    )
    # Minimal dummy matrix
    X = np.zeros((n, 1))
    return AnnData(X=X, obs=obs)


@pytest.mark.parametrize(
    "cluster,cell_type,expected_majority",
    [
        (
            [f"c{i}" for i in range(5)],
            ["A", "B", "A", "A", "B"],
            ["A", "B", "A", "A", "B"],
        ),  # each cell its own cluster
        (["c1"] * 5, ["A", "B", "A", "A", "B"], ["A"] * 5),  # all in one cluster, majority 'A'
        (["c1"] * 5, ["A", "B", "A", "B", "C"], ["A"] * 5),  # tie, pandas idxmax returns first
        (["c1"] * 5, ["A", "B", np.nan, "A", "B"], ["A"] * 5),  # NaN present, majority 'A'
        (["c1", "c1", "c2", "c2", "c2"], ["A", "A", "B", "B", "A"], ["A", "A", "B", "B", "B"]),  # multiple clusters
    ],
)
def test_majority_reference_labels(cluster, cell_type, expected_majority):
    adata = make_adata(cluster, cell_type)
    adata = majority_reference(adata, reference_key="cell_type", cluster_key="cluster")
    assert list(adata.obs["majority_reference"]) == expected_majority


def test_majority_reference_confidence():
    cluster = ["c1"] * 5
    cell_type = ["A", "B", "A", "A", "B"]
    adata = make_adata(cluster, cell_type)
    adata = majority_reference(adata, reference_key="cell_type", cluster_key="cluster")
    # Confidence should be 3/5 for all cells
    assert np.isclose(adata.obs["majority_reference_confidence"].iloc[0], 3 / 5), (
        f"Expected confidence 0.6, got {adata.obs['majority_reference_confidence'].iloc[0]}"
    )
    assert (np.isclose(adata.obs["majority_reference_confidence"], 3 / 5)).all(), (
        f"Expected all confidences to be 0.6, got {adata.obs['majority_reference_confidence'].tolist()}"
    )
