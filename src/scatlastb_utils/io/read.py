import re
import warnings
from collections.abc import Callable
from pathlib import Path
from pprint import pformat
from typing import Any

import anndata as ad
import h5py
import numpy as np
from anndata.io import read_elem, sparse_dataset
from scipy.sparse import csr_matrix
from tqdm import tqdm

from .config import print_flushed, zarr
from .sparse_dask import read_as_dask_array, sparse_dataset_as_dask
from .subset_slots import subset_slot


def get_file_reader(file: str | Path) -> tuple[Callable, str]:
    """Determine the file reader function based on the file extension."""
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
    """Get the store for the given file."""
    func, file_type = get_file_reader(file)
    try:
        store = func(file, "r")
    except zarr.errors.PathNotFoundError as e:
        raise FileNotFoundError(f"Cannot read file {file}") from e
    if return_file_type:
        return store, file_type
    return store


def check_slot_exists(file: str | Path, slot: str) -> bool:
    """Check if a specific slot exists in the file without needing to read the file."""
    store = get_store(file)
    return slot in store


def read_anndata(
    file: str,
    dask: bool = False,
    backed: bool = False,
    fail_on_missing: bool = True,
    exclude_slots: list[str] = None,
    chunks: int | tuple = None,
    stride: int = 200_000,
    verbose: bool = True,
    dask_slots: list[str] | None = None,
    select_keys: str | list[str] | None = None,
    **kwargs: Any,
) -> ad.AnnData:
    """
    Read anndata file

    :param file: path to anndata file in zarr or h5ad format
    :param kwargs: AnnData parameter to zarr group mapping
    """
    # assert Path(file).exists(), f'File not found: {file}'
    if exclude_slots is None:
        exclude_slots = []
    elif exclude_slots == "all":
        exclude_slots = ["X", "layers", "raw"]

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
    chunks: int | tuple = None,
    stride: int = 1000,
    force_sparse_types: str | list[str] | None = None,
    force_sparse_slots: str | list[str] | None = None,
    dask_slots: str | list[str] | None = None,
    verbose: bool = False,
    select_keys: str | list[str] | None = None,
    **kwargs: Any,
) -> ad.AnnData:
    """
    Partially read zarr or h5py groups

    :params group: file group
    :params force_sparse_types: encoding types to convert to sparse_dataset via csr_matrix
    :params backed: read sparse matrix as sparse_dataset
    :params dask: read any matrix as dask array
    :params chunks: chunks parameter for creating dask array
    :params stride: stride parameter for creating backed dask array
    :params dask_slots: slots to read as dask array whenver possible
    :params select_keys: regex pattern or list of keys to match slots that contain mappings (e.g. layers, uns, obsm)
    :params **kwargs: dict of to_slot: slot, by default use all available slot for the zarr file
    :return: AnnData object
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

        if from_slot in ["layers", "raw", "obsm", "obsp", "uns"]:
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
        shapes = {slot: x.shape for slot, x in slots.items() if hasattr(x, "shape")}
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
    chunks: int | tuple = None,
    stride: int = 1000,
    fail_on_missing: bool = True,
    verbose: bool = True,
) -> Any:
    """
    Read a specific slot from a zarr or h5py group.

    :param file: path to the zarr or h5ad file
    :param group: h5py.Group or zarr.Group to read from
    :param slot_name: name of the slot to read
    :param force_sparse_types: list of encoding types to convert to sparse_dataset via csr_matrix
    :param force_slot_sparse: whether to force the slot to be read as sparse
    :param backed: read sparse matrix as sparse_dataset
    :param dask: read any matrix as dask array
    :param chunks: chunks parameter for creating dask array
    :param stride: stride parameter for creating backed dask array
    :param fail_on_missing: whether to raise an error if the slot is not found
    :param verbose: whether to print verbose output
    :return: the read slot as an AnnData object or a dask array
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

    :param group: h5py.Group or zarr.Group
    :param slot: slot name
    :stride: stride parameter for creating backed dask array, ignored when backed=False
    :chunks: chunks parameter for creating dask array ignored when backed=True
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

    Workaround to automatic downcasting of scipy sparse matrices.
    """
    if not isinstance(x, csr_matrix):
        x = csr_matrix(x)
    if isinstance(x.indptr, np.int32):
        x.indptr = x.indptr.astype(np.int64)
    if isinstance(x.indices, np.int32):
        # seems to be necessary to avoid "ValueError: Output dtype not compatible with inputs."
        x.indices = x.indices.astype(np.int64)
    return x
