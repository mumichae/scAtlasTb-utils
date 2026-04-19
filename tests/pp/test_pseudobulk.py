import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from pytest import approx

from scatlastb_utils.pp.pseudobulk import _aggregate_obs, pseudobulk


def test_aggregate_obs(adata):
    groups = list(adata.obs["donor_id"].unique())
    out = _aggregate_obs(adata.obs, group_key="donor_id", group_order=groups)

    # index should match groups order
    assert list(out.index) == groups

    # check obs aggregation: n_agg should match group size, total_counts should match group mean
    expected_counts = adata.obs.groupby("donor_id").size()
    expected_total = adata.obs.groupby("donor_id")["total_counts"].mean()
    for name in groups:
        assert out.loc[name, "n_agg"] == expected_counts.loc[name], (
            f"n_agg mismatch for group {name}\n{adata.obs.query('donor_id == @name')}"
        )
        assert out.loc[name, "total_counts"] == expected_total.loc[name], (
            f"total_counts mismatch for group {name}\n{adata.obs.query('donor_id == @name')}"
        )


@pytest.mark.parametrize("adata_fixture", ["adata", "adata_dask"])
def test_pseudobulk(adata_fixture, request):
    ad = request.getfixturevalue(adata_fixture)
    groups = sc.get.aggregate(ad, "donor_id", "count_nonzero").obs_names
    out = pseudobulk(ad, group_key="donor_id", agg="sum")

    # index should match groups order
    assert all(out.obs_names == groups)

    # n_agg should equal counts per donor
    expected_counts = ad.obs.groupby("donor_id").size().reindex(groups).values
    assert list(out.obs["n_agg"].values) == list(expected_counts)

    # numeric aggregation: total_counts should match group mean
    expected_total = ad.obs.groupby("donor_id")["total_counts"].mean().reindex(groups)
    for g in groups:
        assert approx(out.obs.loc[g, "total_counts"]) == expected_total.loc[g]


def test_pseudobulk_with_dask_backed_read(adata):
    import tempfile
    from pathlib import Path

    import scatlastb_utils as sa

    # use donor_id as group key from fixture
    groups_list = list(adata.obs["donor_id"].unique())

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "tmp.zarr"
        adata.write_zarr(p)

        adata_dask = sa.io.read_anndata(p, dask=True, backed=True)
        print(adata_dask.obs)

        # run pseudobulk, should return an AnnData with one obs per group
        res = pseudobulk(adata_dask, group_key="donor_id", agg="sum")
        assert isinstance(res, AnnData)
        # expected number of pseudobulk samples equals number of unique donors
        assert res.n_obs == len(groups_list)


def test_pseudobulk_prefers_layer(adata):
    """Ensure pseudobulk uses a named layer created for the aggregation if present."""
    adata.layers["counts"] = adata.X.copy()  # create a layer to be used for aggregation
    out = pseudobulk(adata, group_key="donor_id", agg="sum", layer="counts")

    expected = sc.get.aggregate(adata, by="donor_id", func="sum", layer="counts").layers["sum"]
    assert np.allclose(np.asarray(out.X), expected)
