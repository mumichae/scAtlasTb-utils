import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from pytest import approx

from scatlastb_utils.pp.pseudobulk import _aggregate_obs, _categorical_mode, pseudobulk


@pytest.mark.parametrize(
    "series,expected",
    [
        (pd.Series(["a", "b", "c"]).astype("category"), "a"),
        (pd.Series(["x", "y", "x", "z"]).astype("category"), "x"),
    ],
)
def test_categorical_mode_param(series, expected):
    assert _categorical_mode(series) == expected


def test_aggregate_obs(adata):
    groups = list(adata.obs["donor_id"].unique())
    out = _aggregate_obs(adata.obs, group_key="donor_id", group_order=groups)

    # index should match groups order
    assert list(out.index) == groups

    # n_agg should equal counts per donor
    expected_counts = adata.obs.groupby("donor_id").size().reindex(groups).values
    assert list(out["n_agg"].values) == list(expected_counts)

    # numeric aggregation: total_counts should match group mean
    expected_total = adata.obs.groupby("donor_id")["total_counts"].mean().reindex(groups)
    for g in groups:
        assert approx(out.loc[g, "total_counts"]) == expected_total.loc[g]


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
