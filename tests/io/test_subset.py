import tempfile
from pathlib import Path

import numpy as np
import pytest

import scatlastb_utils as atl


@pytest.mark.parametrize("file_type", ["zarr", "h5ad"])
@pytest.mark.parametrize("dask", [False, True])
@pytest.mark.parametrize(
    "files_to_keep",
    ["obs", "var", "X", "obsm", "obsp", "varm", "varp", "layers"],
)
def test_subset_mask_basic(adata, file_type, dask, files_to_keep):
    """Test basic subset mask functionality"""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create masks
        n_obs, n_vars = adata.shape
        obs_mask = np.random.choice([True, False], size=n_obs, p=[0.7, 0.3])
        var_mask = np.random.choice([True, False], size=n_vars, p=[0.7, 0.3])
        subset_mask = (obs_mask, var_mask)

        # Write original data
        adata_path = Path(temp_dir) / "original"
        if file_type == "zarr":
            adata_path = adata_path.with_suffix(".zarr")
            adata.write_zarr(adata_path)
        else:
            adata_path = adata_path.with_suffix(".h5ad")
            adata.write_h5ad(adata_path)

        # Create subset with mask
        subset_path = Path(temp_dir) / "subset.zarr"
        adata_subset = adata[subset_mask].copy()
        atl.io.write_zarr_linked(
            adata_subset,
            in_dir=adata_path,
            out_dir=subset_path,
            subset_mask=subset_mask,
            files_to_keep=[files_to_keep],
        )

        # import subprocess
        # subprocess.run(['ls', '-l', str(subset_path / 'subset_mask')], check=True)
        # mask = np.load(subset_path / 'subset_mask' / 'obs' / 'mask.npy')
        # print('Loaded mask shape:', mask.shape, flush=True)

        # Read subset data
        adata_subset = atl.io.read_anndata(
            subset_path,
            dask=dask,
            backed=dask,
        )

    # Verify subset dimensions
    expected_obs = obs_mask.sum()
    expected_var = var_mask.sum()

    assert adata_subset.shape[0] == expected_obs, (
        f"Expected {expected_obs} obs, got {adata_subset.shape[0]}. Original shape: {adata.shape[0]}"
    )
    assert adata_subset.shape[1] == expected_var, (
        f"Expected {expected_var} var, got {adata_subset.shape[1]}. Original shape: {adata.shape[1]}"
    )

    # Verify data content matches expected subset
    assert np.array_equal(adata_subset.obs.index, adata.obs.index[obs_mask])
    assert np.array_equal(adata_subset.var.index, adata.var.index[var_mask])


def test_subset_mask_linking(adata):
    """Test linking subset masks between slots"""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create initial subset
        obs_mask = np.random.choice([True, False], size=adata.n_obs, p=[0.7, 0.3])
        var_mask = np.random.choice([True, False], size=adata.n_vars, p=[0.7, 0.3])
        subset_mask1 = (obs_mask, var_mask)

        original_path = Path(temp_dir) / "original.zarr"
        subset1_path = Path(temp_dir) / "subset1.zarr"
        subset2_path = Path(temp_dir) / "subset2.zarr"

        adata.write_zarr(original_path)

        # Create first subset
        atl.io.write_zarr_linked(
            adata[subset_mask1],
            in_dir=original_path,
            out_dir=subset1_path,
            subset_mask=subset_mask1,
            verbose=True,
        )
        adata_subset1 = atl.io.read_anndata(subset1_path)

        # Create second subset that links to first
        obs_mask2 = np.random.choice([True, False], size=adata_subset1.n_obs, p=[0.8, 0.2])
        var_mask2 = np.random.choice([True, False], size=adata_subset1.n_vars, p=[0.8, 0.2])
        subset_mask2 = (obs_mask2, var_mask2)

        atl.io.write_zarr_linked(
            adata_subset1[subset_mask2],
            in_dir=subset1_path,
            out_dir=subset2_path,
            subset_mask=subset_mask2,
            verbose=True,
        )

        # Verify final dimensions
        adata_subset2 = atl.io.read_anndata(subset2_path)
        expected_obs = obs_mask2.sum()
        expected_var = var_mask2.sum()

        assert adata_subset2.shape == (expected_obs, expected_var)


def test_subset_mask_raw_data(adata):
    """Test subset masks with raw data"""

    # Add raw data
    adata.raw = adata.copy()

    with tempfile.TemporaryDirectory() as temp_dir:
        obs_mask = np.random.choice([True, False], size=adata.n_obs, p=[0.6, 0.4])
        var_mask = np.random.choice([True, False], size=adata.n_vars, p=[0.6, 0.4])
        subset_mask = (obs_mask, var_mask)

        original_path = Path(temp_dir) / "original.zarr"
        subset_path = Path(temp_dir) / "subset.zarr"

        adata.write_zarr(original_path)
        atl.io.write_zarr_linked(
            adata[subset_mask],
            in_dir=original_path,
            out_dir=subset_path,
            subset_mask=subset_mask,
        )

        adata_subset = atl.io.read_anndata(subset_path)
        adata_raw = adata_subset.raw.to_adata()

    # Verify main data
    assert adata_subset.shape == (obs_mask.sum(), var_mask.sum())

    # Verify raw data
    assert adata_raw.shape == (obs_mask.sum(), var_mask.sum())
    assert np.array_equal(adata_raw.obs.index, adata.obs.index[obs_mask])
    assert np.array_equal(adata_raw.var.index, adata.var.index[var_mask])


def test_subset_mask_none_handling(adata):
    """Test that None masks are handled correctly"""

    with tempfile.TemporaryDirectory() as temp_dir:
        original_path = Path(temp_dir) / "original.zarr"
        subset_path = Path(temp_dir) / "subset.zarr"

        adata.write_zarr(original_path)

        # Test with None mask (should not subset)
        atl.io.write_zarr_linked(
            adata,
            in_dir=original_path,
            out_dir=subset_path,
            subset_mask=None,
        )

        adata_subset = atl.io.read_anndata(subset_path)
        assert adata_subset.shape == adata.shape

        # # Test with partial None masks
        # obs_mask = np.random.choice([True, False], size=adata.n_obs, p=[0.7, 0.3])

        # subset_path2 = Path(temp_dir) / "subset2.zarr"
        # atl.io.write_zarr_linked(
        #     adata,
        #     in_dir=original_path,
        #     out_dir=subset_path2,
        #     subset_mask=(obs_mask, None),  # Only subset observations
        # )

        # adata_subset2 = atl.io.read_anndata(subset_path2)
        # assert adata_subset2.shape == (obs_mask.sum(), adata.n_vars)\
