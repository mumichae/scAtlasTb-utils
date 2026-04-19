import numpy as np
import pytest
import scanpy as sc

from scatlastb_utils.pp.pseudobulk import _aggregate_obs, pseudobulk


def _to_dense(x):
    """Return a dense numpy array for sparse / dask / numpy-like inputs."""
    # Dask array -> compute
    if hasattr(x, "compute"):
        x = x.compute()
    # scipy sparse -> toarray
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def assert_allclose_dense(a, b, rtol=1e-6, atol=0, err_msg=None):
    """Assert that two arrays are equal up to tolerance, with helpful error message.

    Converts inputs to dense arrays first.
    """
    ad = _to_dense(a)
    bd = _to_dense(b)
    if ad.shape != bd.shape:
        raise AssertionError(f"shape mismatch: {ad.shape} != {bd.shape}")
    try:
        np.testing.assert_allclose(ad, bd, rtol=rtol, atol=atol, err_msg=err_msg)
    except AssertionError as e:
        max_abs = np.max(np.abs(ad - bd))
        max_rel = np.max(np.abs(ad - bd) / (np.abs(bd) + atol)) if np.any(bd) else np.inf
        raise AssertionError(
            f"Arrays not equal within rtol={rtol}, atol={atol}. max_abs={max_abs}, max_rel={max_rel}\n{e}"
        ) from e


def _agggregate_brute_force(adata, group_key, groups, agg="sum"):
    """Brute-force compute pseudobulk matrix (dense numpy)"""
    rows = []
    for g in groups:
        sub = adata.X[(adata.obs[group_key] == g).values]
        if sub.size == 0:
            rows.append(np.zeros(adata.X.shape[1], dtype=adata.X.dtype))
            continue
        if agg == "sum":
            rows.append(np.asarray(sub.sum(axis=0)).ravel())
        elif agg == "mean":
            rows.append(np.asarray(sub.mean(axis=0)).ravel())
        else:
            raise ValueError(f"unsupported agg: {agg}")

    return np.vstack(rows)


def _expected_pseudobulk(adata, groups, group_key, agg="sum", layer=None, impl="brute_force"):
    # choose matrix (prefer layer if given) and make dense numpy
    adata.X = _to_dense(adata.layers[layer] if layer is not None else adata.X)

    mask = adata.obs[group_key].isin(groups).values
    if impl == "brute_force":
        adata = adata[mask].copy()
        return _agggregate_brute_force(adata, group_key, groups, agg=agg)

    pseudobulk = sc.get.aggregate(adata, by=group_key, func=agg, mask=mask, axis=0)
    pseudobulk = pseudobulk[groups]  # ensure group order matches expected
    return pseudobulk.layers[agg]


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


@pytest.mark.parametrize(
    "adata_fixture,use_legacy,layer",
    [
        ("adata", False, None),
        ("adata", False, "counts"),
        ("adata_dask", False, None),
        ("adata_dask_backed", False, None),
        ("adata_dask_backed", True, None),
        ("adata_dask_backed", True, "counts"),
    ],
)
def test_pseudobulk(adata_fixture, use_legacy, layer, request):
    ad = request.getfixturevalue(adata_fixture)

    # if testing the named-layer case, create the layer and remove X to
    # replicate the original prefers-layer behaviour
    if layer is not None:
        ad.layers["counts"] = ad.X.copy()
        del ad.X

    out = pseudobulk(ad, group_key="donor_id", agg="sum", use_legacy=use_legacy, min_cells=1, layer=layer)

    # groups should match set of donors (order-insensitive)
    expected_groups = set(ad.obs["donor_id"].unique())
    assert set(out.obs_names) == set(expected_groups)

    # matrix aggregation: compute expected using output ordering
    expected_matrix = _expected_pseudobulk(
        ad,
        groups=out.obs_names,
        group_key="donor_id",
        agg="sum",
        layer=layer,
        impl="brute_force",
    )
    assert_allclose_dense(out.X, expected_matrix)

    expected_matrix = _expected_pseudobulk(
        ad,
        groups=out.obs_names,
        group_key="donor_id",
        agg="sum",
        layer=layer,
        impl="scanpy.get.aggregate",
    )
    assert_allclose_dense(out.X, expected_matrix)
