import json
import os
import shutil
from collections.abc import MutableMapping
from pathlib import Path
from pprint import pformat

import anndata as ad
import pandas as pd
from dask import array as da

from scatlastb_utils.pipeline.misc import dask_compute

from .config import ALL_SLOTS, print_flushed
from .subset_slots import set_mask_per_slot


def write_zarr(adata: ad.AnnData, file: str | Path, compute: bool = False) -> None:
    """
    Write AnnData object to zarr file. Cleans up data types before writing.

    Parameters
    ----------
    adata
        AnnData object to write.
    file
        Path to output zarr file.
    compute
        Whether to compute dask arrays before writing. Default is False.
    """

    def sparse_coo_to_csr(matrix):
        import sparse

        if isinstance(matrix, da.Array) and isinstance(matrix._meta, sparse.COO):
            matrix = matrix.map_blocks(lambda x: x.tocsr(), dtype=matrix.dtype)
        return matrix

    adata.X = sparse_coo_to_csr(adata.X)
    for layer in adata.layers:
        adata.layers[layer] = sparse_coo_to_csr(adata.layers[layer])

    # fix dtype for NaN obs columns
    for col in adata.obs.columns:
        if adata.n_obs == 0:
            continue
        if adata.obs[col].isna().any() or adata.obs[col].dtype.name == "object":
            try:
                adata.obs[col] = pd.to_numeric(adata.obs[col])
            except (ValueError, TypeError):
                # Convert non-NaN entries to string, preserve NaN values
                adata.obs[col] = adata.obs[col].apply(lambda x: str(x) if pd.notna(x) else x).astype("category")

    if compute:
        adata = dask_compute(adata)

    adata.write_zarr(file)


def link_file(
    in_file: str | Path,
    out_file: str | Path,
    relative_path: bool = True,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Link an existing file to a new location.

    Parameters
    ----------
    in_file
        Path to the input file to link from.
    out_file
        Path to the output file to link to.
    relative_path
        Use relative path for the link. Default is True.
    overwrite
        Overwrite the output file if it exists. Default is False.
    verbose
        Whether to print verbose output. Default is True.
    """
    in_file = Path(in_file).resolve(True)
    out_file = Path(out_file)
    out_dir = out_file.parent.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if relative_path:
        in_file = Path(os.path.relpath(in_file, out_dir))

    if overwrite and out_file.exists():
        if out_file.is_dir() and not out_file.is_symlink():
            print_flushed(f"replace {out_file}...", verbose=verbose)
            shutil.rmtree(out_file)
        else:
            out_file.unlink()

    out_file.symlink_to(in_file)
    assert out_file.exists(), f"Linking failed for {out_file.resolve()} -> {in_file}"


def link_zarr(
    in_dir: str | Path,
    out_dir: str | Path,
    file_names: list[str] | None = None,
    overwrite: bool = False,
    relative_path: bool = True,
    slot_map: MutableMapping | None = None,
    in_dir_map: MutableMapping | None = None,
    subset_mask: tuple | None = None,
    verbose: bool = True,
) -> None:
    """
    Link to existing zarr file.

    Parameters
    ----------
    in_dir
        Path to existing zarr file or mapping of input slot in slot_map to input path.
    out_dir
        Path to output zarr file.
    file_names
        List of files to link. If None, link all files. Default is None.
    overwrite
        Overwrite existing output files. Default is False.
    relative_path
        Use relative path for link. Default is True.
    slot_map
        Custom mapping of output slot to input slots. Default is None.
    in_dir_map
        Input directory map for input slots. Default is None.
    subset_mask
        Mask to apply to slots. Default is None.
    verbose
        Whether to print verbose output. Default is True.
    """

    def prune_nested_links(slot_map, in_dir_map):
        # determine equivalence classes of slots (top hierarchy)
        eq_classes = {}
        for out_slot in slot_map:
            if "/" not in out_slot:
                continue
            eq = out_slot.rsplit("/", 1)[0]
            if eq not in slot_map:
                continue
            eq_classes.setdefault(eq, []).append(out_slot)

        for out_slot in eq_classes.keys():
            in_slot = slot_map[out_slot]
            # link all files of the equivalence class
            for f in (in_dir_map[in_slot] / in_slot).iterdir():
                new_out_slot = f"{out_slot}/{f.name}"
                new_in_slot = f"{in_slot}/{f.name}"
                # skip if already specified or .snakefile_timestamp
                if new_out_slot in slot_map or f.name == ".snakemake_timestamp":
                    continue
                # inherit in_dir
                in_dir_map[new_in_slot] = in_dir_map[in_slot]
                # update slot_map
                slot_map[new_out_slot] = new_in_slot
            # remove equivalence class once done to avoid overwriting
            del slot_map[out_slot]
        return slot_map, in_dir_map

    if file_names is None:
        file_names = [] if in_dir is None else [f.name for f in Path(in_dir).iterdir()]
    file_names = [file for file in file_names if file not in (".snakemake_timestamp")]

    if slot_map is None:
        slot_map = {}

    slot_map = {file: file for file in file_names} | slot_map
    slot_map |= {k: k for k, v in slot_map.items() if v is None}

    if in_dir_map is None:
        in_dir_map = {}

    in_dir_map = dict.fromkeys(slot_map.values(), in_dir) | in_dir_map
    in_dir_map = {slot: Path(path) for slot, path in in_dir_map.items()}
    for _dir in in_dir_map.values():
        assert _dir.exists(), f"Input directory {_dir} does not exist"

    # deal with nested mapping
    slot_map, in_dir_map = prune_nested_links(slot_map, in_dir_map)
    slots_to_link = slot_map.keys()

    # link all files
    out_dir = Path(out_dir)
    slot_map = sorted(
        slot_map.items(),
        key=lambda item: out_dir.name in str(in_dir_map[item[1]]),
        reverse=False,
    )
    print_flushed("slot_map:", pformat(slot_map), verbose=verbose)

    for out_slot, in_slot in slot_map:
        in_dir = in_dir_map[in_slot]
        in_file_name = str(in_dir).split(".zarr")[-1] + "/" + in_slot
        out_file_name = str(out_dir).split(".zarr")[-1] + "/" + out_slot
        print_flushed(f"Link {out_file_name} -> {in_file_name}", verbose=verbose)
        link_file(
            in_file=in_dir / in_slot,
            out_file=out_dir / out_slot,
            relative_path=relative_path,
            overwrite=overwrite,
            verbose=verbose,
        )

    for slot in ALL_SLOTS:
        if slot in slots_to_link or any(x.startswith(f"{slot}/") for x in slots_to_link):
            set_mask_per_slot(slot=slot, mask=subset_mask, out_dir=out_dir)
        else:
            set_mask_per_slot(slot=slot, mask=None, out_dir=out_dir)

    for out_slot, in_slot in slot_map:
        if (
            out_slot in ["subset_mask", ".zattrs", ".zgroup"]
            or out_slot.endswith(".zattrs")
            or out_slot.endswith(".zgroup")
        ):
            continue
        set_mask_per_slot(
            slot=out_slot,
            mask=subset_mask,
            in_slot=in_slot,
            in_dir=in_dir_map[in_slot],
            out_dir=out_dir,
        )


def write_zarr_linked(
    adata: ad.AnnData,
    in_dir: str | Path,
    out_dir: str | Path,
    relative_path: bool = True,
    files_to_keep: list[str] | None = None,
    slot_map: MutableMapping | None = None,
    in_dir_map: MutableMapping | None = None,
    verbose: bool = True,
    subset_mask: tuple | None = None,
    compute: bool = False,
) -> None:
    """
    Write AnnData object to a linked zarr file.

    Parameters
    ----------
    adata
        AnnData object to write.
    in_dir
        Path to existing zarr file.
    out_dir
        Path to output zarr file.
    relative_path
        Use relative path for link. Default is True.
    files_to_keep
        List of files to keep and not overwrite. Default is None.
    slot_map
        Custom mapping of output slot to input slot, for slots that are not in files_to_keep. Default is None.
    in_dir_map
        Input directory map for input slots. Default is None.
    verbose
        Whether to print verbose output. Default is True.
    subset_mask
        Mask to apply to slots. Default is None.
    compute
        Whether to compute dask arrays before writing. Default is False.
    """
    if in_dir is None:
        in_dirs = []
    else:
        in_dir = Path(in_dir)
        if in_dir.suffix != ".zarr" and not in_dir.name.endswith(".zarr/raw"):
            print_flushed(
                f"Warning: `{in_dir=!r}` is not a top-level zarr directory, not linking any files", verbose=True
            )
            write_zarr(adata, out_dir, compute=compute)
            return  # exit when in_dir is not a top-level zarr directory
        in_dirs = [f.name for f in in_dir.iterdir()]

    if files_to_keep is None:
        files_to_keep = []

    # Unique the list
    files_to_keep = list(set(files_to_keep))

    def get_top_level_path(file):
        if file.startswith("/"):
            return file.split("/")[1]  # take first directory after root
        return file.split("/")[0]  # take top level directory

    # Make a list of existing subpaths to keep
    file_to_link_clean = [get_top_level_path(f) for f in files_to_keep]

    # For those not keeping, link
    files_to_link = [f for f in in_dirs if get_top_level_path(f) not in file_to_link_clean]

    if slot_map is None:
        slot_map = {}
    extra_slots_to_link = list(slot_map.keys())

    # keep only slots that are not explicitly in files_to_keep
    slot_map = {in_slot: out_slot for in_slot, out_slot in slot_map.items() if in_slot not in files_to_keep}

    # remove slots that will be overwritten anyway
    for slot in set(files_to_link + extra_slots_to_link):
        if hasattr(adata, slot):
            print_flushed(f"remove slot to be linked: {slot}", verbose=verbose)
            delattr(adata, slot)

    # write zarr file
    write_zarr(adata, out_dir, compute=compute)

    # link files
    link_zarr(
        in_dir=in_dir,
        out_dir=out_dir,
        file_names=files_to_link,
        overwrite=True,
        relative_path=relative_path,
        slot_map=slot_map,
        in_dir_map=in_dir_map,
        subset_mask=subset_mask,
        verbose=verbose,
    )

    out_dir = Path(out_dir)

    # update .zattrs files
    for slot in ["obs", "var"]:
        zattrs_file = out_dir / slot / ".zattrs"

        if not zattrs_file.is_symlink():
            continue

        with open(zattrs_file) as file:
            zattrs = json.load(file)

        # add all columns (otherwise linked columns are not included)
        columns = [
            f.name for f in zattrs_file.parent.iterdir() if not f.name.startswith(".") and f.name != zattrs["_index"]
        ]
        zattrs["column-order"] = sorted(columns)

        zattrs_file.unlink()
        with open(zattrs_file, "w") as file:
            json.dump(zattrs, file, indent=4)
