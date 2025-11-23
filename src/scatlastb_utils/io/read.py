import re
import warnings
from collections.abc import Callable
from pathlib import Path
from pprint import pformat
from typing import Any

import anndata as ad
import h5py
import numpy as np

try:
    from anndata.io import read_elem, sparse_dataset  # anndata >=0.11
except ModuleNotFoundError:
    from anndata.experimental import read_elem, sparse_dataset  # anndata <0.11
from scipy.sparse import csr_matrix
from tqdm import tqdm

from .config import print_flushed, zarr
from .sparse_dask import read_as_dask_array, sparse_dataset_as_dask
from .subset_slots import subset_slot


def get_file_reader(file: str | Path) -> tuple[Callable, str]:
    """
    Determine the file reader function based on the file extension.

    Parameters
    ----------
    file
        Path to the file.

    Returns
    -------
    func
        The function to open the file.
    file_type
        The type of the file ('zarr' or 'h5py').

    Raises
    ------
    ValueError
        If the file format is unknown.
    """
    file_path = Path(file)
    if file_path.suffix == ".zarr" or file_path.name.endswith(".zarr/raw"):
        func = zarr.open
        file_type = "zarr"
    elif file_path.suffix == ".h5ad":
        func = h5py.File
        file_type = "h5py"
    else:
        raise ValueError(f"Unknown file format: {file}")
    return func, file_type


def get_store(file: str | Path, return_file_type: bool = False) -> Any | tuple[Any, str]:
    """
    Get the store for the given file.

    Parameters
    ----------
    file
        Path to the file.
    return_file_type
        Whether to return the file type along with the store. Default is False.

    Returns
    -------
    store
        The opened file store.
    file_type
        The type of the file, if return_file_type is True.

    Raises
    ------
    FileNotFoundError
        If the file cannot be read.
    """
    func, file_type = get_file_reader(file)
    try:
        store = func(file, "r")
    except zarr.errors.PathNotFoundError as e:
        raise FileNotFoundError(f"Cannot read file {file}") from e
    if return_file_type:
        return store, file_type
    return store


def check_slot_exists(file: str | Path, slot: str) -> bool:
    """
    Check if a specific slot exists in the file without needing to read the file.

    Parameters
    ----------
    file
        Path to the file.
    slot
        Name of the slot to check.

    Returns
    -------
    exists
        True if the slot exists, False otherwise.
    """
    store = get_store(file)
    return slot in store


def read_anndata(
    file: str,
    dask: bool = False,
    backed: bool = False,
    fail_on_missing: bool = True,
    exclude_slots: list[str] | None = None,
    chunks: int | tuple | None = None,
    stride: int = 200_000,
    verbose: bool = True,
    dask_slots: list[str] | None = None,
    select_keys: str | list[str] | None = None,
    **kwargs: Any,
) -> ad.AnnData:
    """
    Read an AnnData file from zarr or h5ad format.

    Parameters
    ----------
    file
        Path to anndata file in zarr or h5ad format.
    dask
        Whether to read arrays as dask arrays. Default is False.
    backed
        Whether to read in backed mode. Default is False.
    fail_on_missing
        Whether to fail if a slot is missing. Default is True.
    exclude_slots
        Slots to exclude from reading. Default is None.
    chunks
        Chunk size for dask arrays. Default is None.
    stride
        Stride for dask arrays. Default is 200_000.
    verbose
        Whether to print verbose output. Default is True.
    dask_slots
        Slots to read as dask arrays. Default is None.
    select_keys
        Keys to select for reading. Default is None.
    **kwargs
        Additional keyword arguments mapping AnnData parameters to zarr group slots.

    Returns
    -------
    adata
        The loaded AnnData object.

    Raises
    ------
    ValueError
        If a required slot is missing and fail_on_missing is True.

    Examples
    --------
    Basic usage:

    >>> from scatlastb_utils.io import read_anndata
    >>> adata = read_anndata("example.h5ad", verbose=False)

    >>> adata = read_anndata("example.zarr")
    dask: False, backed: False
    Read slot "X", store as "X"...
    Read X and convert to csr matrix...
    Read slot "layers", store as "layers"...
    Read slot "obs", store as "obs"...
    Read slot "obsm", store as "obsm"...
    Read obsm slots as_dask=False: 100%|██████████████████████| 2/2 [00:00<00:00, 4080.06it/s]
    Read slot "obsp", store as "obsp"...
    Read obsp slots as_dask=False: 100%|██████████████████████| 2/2 [00:00<00:00, 1916.08it/s]
    Read slot "raw", store as "raw"...
    Read raw slots as_dask=False: 100%|███████████████████████| 3/3 [00:00<00:00, 1759.85it/s]
    Read slot "uns", store as "uns"...
    Read uns slots as_dask=False: 100%|███████████████████████| 6/6 [00:00<00:00, 1914.04it/s]
    Read slot "var", store as "var"...
    Read slot "varm", store as "varm"...
    Read slot "varp", store as "varp"...
    shape: (700, 765)

    >>> adata
    AnnData object with n_obs × n_vars = 700 × 765
        obs: 'bulk_labels', 'n_genes', 'percent_mito', 'n_counts', 'S_score', 'G2M_score', 'phase', 'louvain'
        var: 'n_counts', 'means', 'dispersions', 'dispersions_norm', 'highly_variable'
        uns: 'bulk_labels_colors', 'louvain', 'louvain_colors', 'neighbors', 'pca', 'rank_genes_groups'
        obsm: 'X_pca', 'X_umap'
        varm: 'PCs'
        obsp: 'connectivities', 'distances'

    Select custom slots:

    >>> read_anndata("example.zarr", X="X", obs="obs", var="var", uns="uns", verbose=False)
    AnnData object with n_obs × n_vars = 700 × 765
        obs: 'bulk_labels', 'n_genes', 'percent_mito', 'n_counts', 'S_score', 'G2M_score', 'phase', 'louvain'
        var: 'n_counts', 'means', 'dispersions', 'dispersions_norm', 'highly_variable'
        uns: 'bulk_labels_colors', 'louvain', 'louvain_colors', 'neighbors', 'pca', 'rank_genes_groups'

    >>> read_anndata("example.zarr", X="raw/X", obs="obs", var="var", uns="uns", verbose=False)
    AnnData object with n_obs × n_vars = 700 × 765
        obs: 'bulk_labels', 'n_genes', 'percent_mito', 'n_counts', 'S_score', 'G2M_score', 'phase', 'louvain'
        var: 'n_counts', 'means', 'dispersions', 'dispersions_norm', 'highly_variable'
        uns: 'bulk_labels_colors', 'louvain', 'louvain_colors', 'neighbors', 'pca', 'rank_genes_groups'

    >>> read_anndata("example.zarr", X="obsm/X_pca", obs="obs", uns="uns", verbose=False)
    AnnData object with n_obs × n_vars = 700 × 50
        obs: 'bulk_labels', 'n_genes', 'percent_mito', 'n_counts', 'S_score', 'G2M_score', 'phase', 'louvain'
        uns: 'bulk_labels_colors', 'louvain', 'louvain_colors', 'neighbors', 'pca', 'rank_genes_groups'

    Load slots with dask:

    >>> read_anndata("example.zarr", dask=True, backed=True)
    dask: True, backed: True
    chunks: (200000, -1)
    Read slot "X", store as "X"...
    Read X as dask array and convert blocks to csr_matrix...
    Read and convert to dask array
    Read slot "layers", store as "layers"...
    Read slot "obs", store as "obs"...
    Read slot "obsm", store as "obsm"...
    Read obsm slots as_dask=False: 100%|██████████████████████| 2/2 [00:00<00:00, 4359.98it/s]
    Read slot "obsp", store as "obsp"...
    Read obsp slots as_dask=False: 100%|██████████████████████| 2/2 [00:00<00:00, 2125.31it/s]
    Read slot "raw", store as "raw"...
    Read raw slots as_dask=True: 100%|████████████████████████| 3/3 [00:00<00:00, 1321.04it/s]
    Read slot "uns", store as "uns"...
    Read uns slots as_dask=False: 100%|███████████████████████| 6/6 [00:00<00:00, 1596.82it/s]
    Read slot "var", store as "var"...
    Read slot "varm", store as "varm"...
    Read slot "varp", store as "varp"...
    shape: (700, 765)
    AnnData object with n_obs × n_vars = 700 × 765
        obs: 'bulk_labels', 'n_genes', 'percent_mito', 'n_counts', 'S_score', 'G2M_score', 'phase', 'louvain'
        var: 'n_counts', 'means', 'dispersions', 'dispersions_norm', 'highly_variable'
        uns: 'bulk_labels_colors', 'louvain', 'louvain_colors', 'neighbors', 'pca', 'rank_genes_groups'
        obsm: 'X_pca', 'X_umap'
        varm: 'PCs'
        obsp: 'connectivities', 'distances'

    """
    if exclude_slots is None:
        exclude_slots = []
    elif exclude_slots == "all":
        exclude_slots = ["X", "layers", "raw"]

    assert Path(file).exists(), f"File not found: {file}"
    store, file_type = get_store(file, return_file_type=True)
    # set default kwargs
    kwargs = {x: x for x in store} if not kwargs else kwargs
    # set key == value if value is None
    kwargs |= {k: k for k, v in kwargs.items() if v is None}
    # exclude slots
    kwargs = {k: v for k, v in kwargs.items() if k not in exclude_slots + ["subset_mask"]}

    # return an empty AnnData object if no keys are available
    if len(store.keys()) == 0:
        return ad.AnnData()

    # check if keys are available
    for name, slot in kwargs.items():
        if slot not in store:
            message = f'Cannot find "{slot}" for AnnData parameter `{name}`'
            message += f"\nfile: {file}\navailable slots: {list(store)}"
            if fail_on_missing:
                raise ValueError(message)
            warnings.warn(f"{message}, will be skipped", stacklevel=2)
    adata = read_partial(
        file,
        store,
        dask=dask,
        backed=backed,
        chunks=chunks,
        stride=stride,
        verbose=verbose,
        dask_slots=dask_slots,
        select_keys=select_keys,
        **kwargs,
    )
    if not backed and file_type == "h5py":
        store.close()

    return adata


def read_partial(
    file: str,
    group: h5py.Group | zarr.Group,
    backed: bool = False,
    dask: bool = False,
    chunks: int | tuple | None = None,
    stride: int = 1000,
    force_sparse_types: str | list[str] | None = None,
    force_sparse_slots: str | list[str] | None = None,
    dask_slots: str | list[str] | None = None,
    verbose: bool = False,
    select_keys: str | list[str] | None = None,
    **kwargs: Any,
) -> ad.AnnData:
    """
    Partially read zarr or h5py groups into an AnnData object.

    Parameters
    ----------
    file
        Path to the file.
    group
        File group to read from.
    backed
        Whether to read sparse matrix as sparse_dataset. Default is False.
    dask
        Whether to read any matrix as dask array. Default is False.
    chunks
        Chunks parameter for creating dask array. Default is None.
    stride
        Stride parameter for creating backed dask array. Default is 1000.
    force_sparse_types
        Encoding types to convert to sparse_dataset via csr_matrix. Default is None.
    force_sparse_slots
        Slots to force as sparse. Default is None.
    dask_slots
        Slots to read as dask array when possible. Default is None.
    verbose
        Whether to print verbose output. Default is False.
    select_keys
        Regex pattern or list of keys to match slots that contain mappings. Default is None.
    **kwargs
        Mapping of to_slot: slot, by default use all available slots for the zarr file.

    Returns
    -------
    adata
        The loaded AnnData object.

    Raises
    ------
    ValueError
        If there is an error reading the file or slot shapes are incompatible.
    """
    if force_sparse_slots is None:
        force_sparse_slots = []
    elif isinstance(force_sparse_slots, str):
        force_sparse_slots = [force_sparse_slots]
    force_sparse_slots.extend(["X", "layers/", "raw/X"])

    if dask_slots is None:
        dask_slots = ["layers", "raw"]

    if chunks is None:
        chunks = (stride, -1)
    print_flushed(f"dask: {dask}, backed: {backed}", verbose=verbose)
    if dask and verbose:
        print_flushed("chunks:", chunks)

    slots = {}
    for to_slot, from_slot in kwargs.items():
        print_flushed(f'Read slot "{from_slot}", store as "{to_slot}"...', verbose=verbose)
        force_slot_sparse = any(from_slot.startswith((x, f"/{x}")) for x in force_sparse_slots)

        if from_slot in ["layers", "raw", "obsm", "varm", "obsp", "varp", "uns"]:
            keys = group[from_slot].keys()

            if isinstance(select_keys, str):
                print_flushed(f"select only slots that match {select_keys}")
                keys = [key for key in keys if re.match(select_keys, key)]
            elif isinstance(select_keys, list):
                keys = [key for key in keys if key in select_keys]

            if from_slot == "raw":
                keys = [key for key in keys if key in ["X", "var", "varm"]]

            if keys:
                as_dask = from_slot in dask_slots and dask
                slots[to_slot] = {
                    sub_slot: read_slot(
                        file=file,
                        group=group,
                        slot_name=f"{from_slot}/{sub_slot}",
                        force_sparse_types=force_sparse_types,
                        force_slot_sparse=force_slot_sparse,
                        backed=as_dask,
                        dask=as_dask,
                        chunks=chunks,
                        stride=stride,
                        fail_on_missing=False,
                        verbose=False,
                    )
                    for sub_slot in tqdm(keys, desc=f"Read {from_slot} slots as_dask={as_dask}", disable=not verbose)
                }
        else:
            slots[to_slot] = read_slot(
                file=file,
                group=group,
                slot_name=from_slot,
                force_sparse_types=force_sparse_types,
                force_slot_sparse=force_slot_sparse,
                backed=backed,
                dask=dask,
                chunks=chunks,
                stride=stride,
                fail_on_missing=False,
                verbose=verbose,
            )

    try:
        adata = ad.AnnData(**slots)
    except Exception as e:

        def _shape(value):
            if hasattr(value, "shape"):
                return value.shape
            if isinstance(value, dict):
                return {k: _shape(v) for k, v in value.items()}
            return None

        shapes = {k: _shape(v) for k, v in slots.items() if hasattr(v, "shape") or isinstance(v, dict)}
        message = f"Error reading {file}\nshapes: {pformat(shapes)}"
        raise ValueError(message) from e

    if verbose:
        print_flushed("shape:", adata.shape, verbose=verbose)

    return adata


def read_slot(
    file: str | Path,
    group: h5py.Group | zarr.Group | None,
    slot_name: str,
    force_sparse_types: str | list[str] | None = None,
    force_slot_sparse: bool = False,
    backed: bool = False,
    dask: bool = False,
    chunks: int | tuple | None = None,
    stride: int = 1000,
    fail_on_missing: bool = True,
    verbose: bool = True,
) -> Any:
    """
    Read a specific slot from a zarr or h5py group.

    Parameters
    ----------
    file
        Path to the zarr or h5ad file.
    group
        File group to read from.
    slot_name
        Name of the slot to read.
    force_sparse_types
        Encoding types to convert to sparse_dataset via csr_matrix. Default is None.
    force_slot_sparse
        Whether to force the slot to be read as sparse. Default is False.
    backed
        Whether to read sparse matrix as sparse_dataset. Default is False.
    dask
        Whether to read any matrix as dask array. Default is False.
    chunks
        Chunks parameter for creating dask array. Default is None.
    stride
        Stride parameter for creating backed dask array. Default is 1000.
    fail_on_missing
        Whether to raise an error if the slot is not found. Default is True.
    verbose
        Whether to print verbose output. Default is True.

    Returns
    -------
    slot
        The read slot as an AnnData object or a dask array.

    Raises
    ------
    ValueError
        If the slot is not found and fail_on_missing is True.
    """
    if group is None:
        group = get_store(file)

    if force_sparse_types is None:
        force_sparse_types = []
    elif isinstance(force_sparse_types, str):
        force_sparse_types = [force_sparse_types]

    if slot_name not in group:
        if fail_on_missing:
            raise ValueError(f'Slot "{slot_name}" not found in {file}')
        warnings.warn(f'Slot "{slot_name}" not found, skip...', stacklevel=2)
        return None

    if dask:
        slot = _read_slot_dask(
            group,
            slot_name,
            force_sparse_types,
            force_slot_sparse,
            stride=stride,
            chunks=chunks,
            backed=backed,
            verbose=verbose,
        )
    else:
        slot = _read_slot_default(
            group,
            slot_name,
            force_sparse_types,
            force_slot_sparse,
            backed=backed,
            verbose=verbose,
        )

    try:
        slot = subset_slot(
            slot_name=slot_name,
            slot=slot,
            mask_dir=Path(file) / "subset_mask",
            chunks=chunks,
        )
    except Exception as e:
        print_flushed(f"Error subsetting {slot_name}")
        raise e

    return slot


def _read_slot_dask(
    group: h5py.Group | zarr.Group,
    slot: str,
    force_sparse_types: str | list[str],
    force_slot_sparse: bool,
    stride: int,
    chunks: int | tuple,
    backed: bool,
    verbose: bool,
) -> Any:
    """
    Read a slot as dask array or backed sparse matrix.

    Parameters
    ----------
    group
        File group to read from.
    slot
        Slot name.
    force_sparse_types
        Encoding types to convert to sparse_dataset via csr_matrix.
    force_slot_sparse
        Whether to force the slot to be read as sparse.
    stride
        Stride parameter for creating backed dask array, ignored when backed=False.
    chunks
        Chunks parameter for creating dask array, ignored when backed=True.
    backed
        Whether to read sparse matrix as sparse_dataset.
    verbose
        Whether to print verbose output.

    Returns
    -------
    slot
        The read slot as a dask array or sparse matrix.
    """
    elem = group[slot]
    iospec = ad._io.specs.get_spec(elem)

    if iospec.encoding_type in ("csr_matrix", "csc_matrix"):
        if backed:
            print_flushed(f"Read {slot} as backed sparse dask array...", verbose=verbose)
            elem = sparse_dataset(elem)
            return sparse_dataset_as_dask(elem, stride=stride)
        print_flushed(f"Read {slot} as sparse dask array...", verbose=verbose)
        elem = read_as_dask_array(elem, chunks=chunks, verbose=verbose)
        return elem.map_blocks(csr_matrix_int64_indptr, dtype=elem.dtype)

    elif iospec.encoding_type in force_sparse_types or force_slot_sparse:
        print_flushed(
            f"Read {slot} as dask array and convert blocks to csr_matrix...",
            verbose=verbose,
        )
        elem = read_as_dask_array(elem, chunks=chunks, verbose=verbose)
        return elem.map_blocks(csr_matrix_int64_indptr, dtype=elem.dtype)

    elif iospec.encoding_type == "array":
        print_flushed(f"Read {slot} as dask array...", verbose=verbose)
        return read_as_dask_array(elem, chunks=chunks, verbose=verbose)

    return read_elem(elem)


def _read_slot_default(
    group: h5py.Group | zarr.Group,
    slot: str,
    force_sparse_types: str | list[str],
    force_slot_sparse: bool,
    backed: bool,
    verbose: bool,
) -> Any:
    """
    Read a slot using the default method (not dask).

    Parameters
    ----------
    group
        File group to read from.
    slot
        Slot name.
    force_sparse_types
        Encoding types to convert to sparse_dataset via csr_matrix.
    force_slot_sparse
        Whether to force the slot to be read as sparse.
    backed
        Whether to read sparse matrix as sparse_dataset.
    verbose
        Whether to print verbose output.

    Returns
    -------
    slot
        The read slot as a dense or sparse matrix.
    """
    elem = group[slot]
    iospec = ad._io.specs.get_spec(elem)

    if iospec.encoding_type in ("csr_matrix", "csc_matrix"):
        if backed:
            print_flushed(f"Read {slot} as backed sparse matrix...", verbose=verbose)
            return sparse_dataset(elem)
        return read_elem(elem)
    elif iospec.encoding_type in force_sparse_types or force_slot_sparse:
        print_flushed(f"Read {slot} and convert to csr matrix...", verbose=verbose)
        return csr_matrix(read_elem(elem))
    else:
        return read_elem(elem)


def csr_matrix_int64_indptr(x: Any) -> csr_matrix:
    """
    Convert a sparse matrix to csr_matrix with int64 indices and indptr.

    This is a workaround for automatic downcasting of scipy sparse matrices.

    Parameters
    ----------
    x
        Input sparse matrix or array.

    Returns
    -------
    csr
        Matrix with int64 indices and indptr.
    """
    if not isinstance(x, csr_matrix):
        x = csr_matrix(x)
    if x.indptr.dtype == np.int32:
        x.indptr = x.indptr.astype(np.int64)
    if x.indices.dtype == np.int32:
        # seems to be necessary to avoid "ValueError: Output dtype not compatible with inputs."
        x.indices = x.indices.astype(np.int64)
    return x
