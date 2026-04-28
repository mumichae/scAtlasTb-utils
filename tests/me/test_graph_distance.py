import os
import sys

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.spatial import distance

# Ensure package importable from tests directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from scatlastb_utils.metrics import graph_distance as gd


@pytest.mark.parametrize(
    "matrix,expected",
    [
        (sp.csr_matrix([[0, 1], [1, 0]]), True),
        (sp.csr_matrix([[0, 1], [0, 0]]), False),
    ],
)
def test_is_symmetric(matrix, expected):
    assert bool(gd.is_symmetric(matrix)) is expected


def test_symmetrize_helpers():
    x = sp.csr_matrix([[0, 2, 0], [0, 0, 0], [0, 0, 0]])
    rows, cols = gd._symmetrize_mask(x)
    assert rows.size == cols.size and rows.size > 0
    sym = gd.symmetrize_if_needed(x)
    assert gd.is_symmetric(sym)


def test__compute_distances_basic():
    obsm = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
    rows = np.array([0, 1, 2, 3])
    cols = np.array([1, 2, 3, 4])
    dists = gd._compute_distances(rows, cols, obsm, n_jobs=1)
    expected = np.array([distance.cdist(obsm[[r]], obsm[[c]])[0, 0] for r, c in zip(rows, cols, strict=True)])
    np.testing.assert_allclose(dists, expected, atol=1e-8)


def test_sparse_spearman_and_edge_cases():
    # normal case
    m1 = sp.csr_matrix([[1, 2, 3], [4, 5, 6]])
    m2 = sp.csr_matrix([[1, 3, 2], [6, 5, 4]])
    mask = sp.csr_matrix([[1, 1, 0], [1, 1, 1]])
    res = gd.sparse_spearman(m1, m2, mask, n_jobs=1)
    np.testing.assert_allclose(res, np.array([1.0, -1.0]), atol=1e-8, equal_nan=True)

    # row with <=1 neighbor -> NaN
    mask2 = sp.csr_matrix([[1, 0, 0], [1, 1, 1]])
    res2 = gd.sparse_spearman(m1, m2, mask2, n_jobs=1)
    assert np.isnan(res2[0]) and not np.isnan(res2[1])


def test_compute_missing_distances(adata_pp):
    conn = adata_pp.obsp["connectivities"]
    res = gd.compute_missing_distances(
        adata_pp, obsp_key="distances", obsm_key="X_pca", conn_mask=conn, inplace=False, n_jobs=1
    )

    rows, cols = sp.triu(conn > 0).nonzero()
    coords = adata_pp.obsm["X_pca"]
    for r, c in zip(rows, cols, strict=True):
        assert np.isclose(res[r, c], distance.cdist(coords[[r]], coords[[c]])[0, 0])


def test_compare_distances(adata_pp):
    df = gd.compare_distances(
        adata_pp,
        obsp_connectivities_1="connectivities",
        obsp_distances_1="distances",
        obsm_key_1="X_pca",
        obsp_connectivities_2="connectivities",
        obsp_distances_2="distances",
        obsm_key_2="X_pca",
        n_jobs=1,
    )
    assert set(df.columns) == {
        "average_distance_1",
        "average_distance_2",
        "average_difference",
        "average_distance_diff",
        "spearman_correlation",
    }


def test_get_knn_skips_if_no_numba():
    pytest.importorskip("numba")
    x = np.array([[0, 3, 1], [4, 1, 0], [0, 2, 5]], dtype=float)
    k1 = gd.get_knn(x, k=1)
    assert all((k1.getrow(i).nnz <= 1) for i in range(k1.shape[0]))
