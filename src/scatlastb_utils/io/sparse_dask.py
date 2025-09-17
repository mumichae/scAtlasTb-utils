"""Copied and adapted from https://gist.github.com/ivirshup/c29c9fb0b5b21a9c290cf621e4e68b18"""

import anndata as ad

try:
    from anndata.abc import CSCDataset, CSRDataset  # anndata >=0.11
except ModuleNotFoundError:
    from anndata.experimental import CSCDataset, CSRDataset  # anndata <0.11
import dask.array as da
import h5py
import numpy as np
import zarr

try:
    from anndata.io import read_elem, sparse_dataset  # anndata >=0.11
except ModuleNotFoundError:
    from anndata.experimental import read_elem, sparse_dataset  # anndata <0.11
from dask import delayed
from scipy import sparse


def read_as_dask_array(elem, chunks=("auto", -1), verbose=True):
    """Read an element as a dask array."""
    if isinstance(elem, zarr.storage.BaseStore):
        if verbose:
            print("Read dask array from zarr directly", flush=True)
        return da.from_zarr(elem, chunks=chunks)
    if verbose:
        print("Read and convert to dask array", flush=True)
    elem = read_elem(elem)
    if np.min(elem.shape) == 0:
        chunks = elem.shape
    return da.from_array(elem, chunks=chunks)


def csr_callable(shape: tuple[int, int], dtype) -> sparse.csr_matrix:  # noqa: D103
    if len(shape) == 0:
        shape = (0, 0)
    if len(shape) == 1:
        shape = (shape[0], 0)
    elif len(shape) == 2:
        pass
    else:
        raise ValueError(shape)
    sparse_matrix = sparse.csr_matrix(shape, dtype=dtype)
    sparse_matrix.indptr = sparse_matrix.indptr.astype(np.int64)
    return sparse_matrix


class CSRCallable:
    """Dummy class to bypass dask checks"""

    def __new__(cls, shape, dtype):  # noqa: D102
        return csr_callable(shape, dtype)


def make_dask_chunk(x: "SparseDataset", start: int, end: int) -> da.Array:  # noqa: F821
    """Create a dask array chunk from a sparse dataset."""

    def take_slice(x, idx):
        try:
            sliced = x[idx]
        except ValueError as e:
            print(f"Error slicing {x} with {idx}")
            raise e
        return sliced

    return da.from_delayed(
        delayed(take_slice)(x, slice(start, end)),
        dtype=x.dtype,
        shape=(end - start, x.shape[1]),
        meta=CSRCallable,
    )


def sparse_dataset_as_dask(x, stride: int = 1000):
    """Convert a sparse dataset to a dask array with specified chunk size."""
    if not isinstance(x, CSRDataset | CSCDataset | da.Array):
        return x
    n_chunks, rem = divmod(x.shape[0], stride)

    chunks = []
    cur_pos = 0
    for _ in range(n_chunks):
        chunks.append(make_dask_chunk(x, cur_pos, cur_pos + stride))
        cur_pos += stride
    if rem:
        chunks.append(make_dask_chunk(x, cur_pos, x.shape[0]))

    return da.concatenate(chunks, axis=0)


def read_w_sparse_dask(group: [h5py.Group, zarr.Group], obs_chunk: int = 1000) -> ad.AnnData:
    """Reads an AnnData object from a group with sparse data as dask arrays."""
    return ad.AnnData(
        X=sparse_dataset_as_dask(sparse_dataset(group["X"]), obs_chunk),
        **{
            k: read_elem(group[k]) if k in group else {}
            for k in ["layers", "obs", "var", "obsm", "varm", "uns", "obsp", "varp"]
        },
    )
