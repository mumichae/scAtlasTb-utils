import pytest

import scatlastb_utils as sa


@pytest.mark.parametrize(
    "slots",
    [
        None,
        {"X": "X"},
        {"obs": "obs"},
        {"X": "X", "obs": "obs", "var": "var"},
        {"X": "layers/counts", "obs": "obs", "var": "var"},
    ],
)
@pytest.mark.parametrize("file_type", ["zarr", "h5ad"])
@pytest.mark.parametrize("dask", [False, True])
def test_read_anndata(adata, file_type, slots, dask):
    import tempfile
    from pathlib import Path

    if slots is None:
        slots = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        adata_path = Path(temp_dir) / "test_adata"

        if file_type == "zarr":
            adata_path = adata_path.with_suffix(".zarr")
            adata.write_zarr(adata_path)
        elif file_type == "h5ad":
            adata_path = adata_path.with_suffix(".h5ad")
            adata.write_h5ad(adata_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        sa.io.read_anndata(
            adata_path,
            **slots,
            dask=dask,
            backed=dask,
        )


@pytest.mark.parametrize(
    "files_to_keep",
    [None, ["obs"], ["var"], ["X"], ["X", "obs", "var"]],
)
def test_write_linked_anndata(adata, files_to_keep):
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        in_path = f"{temp_dir}/input.zarr"
        out_path = f"{temp_dir}/output.zarr"

        # write input zarr for linking
        adata.write_zarr(in_path)

        # write linked zarr
        sa.io.write_zarr_linked(
            adata,
            in_dir=in_path,
            out_dir=out_path,
            files_to_keep=files_to_keep,
        )

        # print output
        os.system(f"ls -lah {out_path}")


# TODO: test subset masks
