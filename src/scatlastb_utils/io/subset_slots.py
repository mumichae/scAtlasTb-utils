import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from dask import array as da

from .config import ALL_SLOTS


## Writing subset masks
def set_mask_per_slot(slot: str, mask: np.array, out_dir: Path | str, in_slot: str = None, in_dir: Path | str = None):
    """
    Set the subset mask for a specific slot in the output directory.

    :param slot: The name of AnnData slot for which to set the mask (e.g., "X", "obs", "var", etc.)
    :param mask: The mask to apply, should be a boolean array or None
    """

    def _call_function_per_slot(func, path, *args, **kwargs):
        if path.is_dir() and (path / ".zattrs").exists():
            func(*args, **kwargs)

    link_slot = in_slot is not None

    if slot == "uns":
        # do not set mask for uns slots
        return

    out_dir = Path(out_dir)
    mask_dir = out_dir / "subset_mask"
    mask_dir = init_mask_dir(mask_dir, slot, in_slot, in_dir)

    if mask is not None:
        if slot.startswith("obs"):
            mask = mask[0]
        elif slot.startswith("var"):
            mask = mask[1]

    match slot:
        case _ if slot == "X" or slot.startswith("layers/"):
            save_feature_matrix_mask(mask_dir=mask_dir, mask=mask, link_slot=link_slot)

        case "obs" | "var":
            save_slot_mask(mask_dir=mask_dir, mask=mask, link_slot=link_slot)

        case "layers":
            for path in (out_dir / slot).iterdir():
                _call_function_per_slot(
                    func=save_feature_matrix_mask,
                    path=path,
                    mask_dir=mask_dir / path.name,
                    mask=mask,
                    link_slot=link_slot,
                )

        case "obsp" | "obsm" | "varp" | "varm":
            for path in (out_dir / slot).iterdir():
                _call_function_per_slot(
                    save_slot_mask,
                    path=path,
                    mask_dir=mask_dir / path.name,
                    mask=mask,
                    link_slot=link_slot,
                )

        case "raw":
            if not mask_dir.exists():
                return
            for path in mask_dir.iterdir():
                # recursive call for subdirectories
                _call_function_per_slot(
                    set_mask_per_slot,
                    path=path,
                    slot=slot,
                    mask=mask,
                    out_dir=out_dir / slot,
                    in_slot=in_slot,
                )

        case _:
            print(f"Unknown slot: {slot}", flush=True)


def remove_path(path, remove_file=False):
    """Remove a file, directory, or symlink at the specified path."""
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.is_file() and remove_file:
        path.unlink()


def init_mask_dir(mask_dir: Path | str, slot: str, in_slot: str, in_dir: Path | str):
    """Create if missing or copy files from symlinked directory."""
    mask_dir = Path(mask_dir)

    if not mask_dir.exists():
        mask_dir.mkdir(parents=True)
    elif mask_dir.is_symlink():
        link_dir = mask_dir.resolve()
        mask_dir.unlink()
        # Copy the folder from the original location
        shutil.copytree(link_dir, mask_dir)

    mask_dir_slot = mask_dir / slot

    if in_slot is not None:
        remove_path(mask_dir_slot)
        in_dir = (Path(in_dir) / "subset_mask" / in_slot).resolve()
        if in_dir.exists():
            # Copy the folder from the original location
            shutil.copytree(in_dir, mask_dir_slot)

    return mask_dir_slot


def save_feature_matrix_mask(mask_dir, mask, link_slot):
    """Save the subset mask for the feature matrices e.g. from X, raw/X or layers/*"""
    obs_mask_file = mask_dir / "obs.npy"
    var_mask_file = mask_dir / "var.npy"

    if mask is None:
        # remove any pre-existing mask files, since slot is being overwritten with correct shape
        # -> no subset mask required
        remove_mask_file(mask_dir, link_slot)
        return

    if not mask_dir.exists():
        mask_dir.mkdir(parents=True)

    if link_slot:
        # update subset masks
        mask = (
            update_mask(obs_mask_file, mask[0]),
            update_mask(var_mask_file, mask[1]),
        )

    np.save(obs_mask_file, mask[0])
    np.save(var_mask_file, mask[1])


def save_slot_mask(mask_dir: Path, mask: np.array, link_slot: bool):
    """
    Save the subset mask for a slot, e.g. obs, var, obsm, varp, etc.

    :param mask_dir: Directory where the mask will be saved
    :param mask: The mask to save, should be a boolean array or None
    :param link_slot: Whether this slot is a linked slot, in which case the mask will be updated
    """
    if mask is None:
        # # remove any pre-existing mask file, since slot is being overwritten with correct shape
        # # -> no subset mask required
        remove_mask_file(mask_dir, link_slot)
        return

    mask_file = mask_dir / "mask.npy"
    mask_dir.mkdir(parents=True, exist_ok=True)
    if mask_file.exists() and link_slot:
        # update mask for linked mask_file slot
        mask = update_mask(mask_file, mask)

    np.save(mask_file, mask)


def update_mask(mask_file, mask):
    """Update the existing mask with the new mask, assuming the new mask is a subset of the old mask."""
    if not mask_file.exists():
        return mask

    mask_old = np.load(mask_file)

    if mask_old.shape == mask.shape:
        # no need to update mask, since it has the same shape
        return mask

    # check if old mask can be subsetted with the new mask
    assert mask_old[mask_old].shape == mask.shape, (
        f"Shape mismatch\n mask_old: {mask_old.shape}, mask_old[mask_old]: "
        f"{mask_old[mask_old].shape}, mask: {mask.shape}\n{mask_file}"
    )

    mask_old[mask_old] &= mask
    return mask_old


def remove_mask_file(mask_file, link_slot):
    """Remove the mask file if it exists, unless it's a linked slot"""
    if link_slot:
        return
    remove_path(mask_file, remove_file=True)


## Functions for subsetting slots when reading them


def subset_slot(slot_name, slot, mask_dir, chunks=("auto", -1)):
    """
    Subset a slot based on the provided mask directory.

    :param slot_name: Name of the slot to subset
    :param slot: The slot object to subset
    :param mask_dir: Directory containing the mask files for subsetting
    :param chunks: Chunk size for dask arrays
    """
    if slot is None or not mask_dir.exists():
        return slot

    if isinstance(slot, dict):
        slot = {
            key: subset_slot(
                slot_name=f"{slot_name}/{key}",
                slot=value,
                mask_dir=mask_dir / key,
                chunks=chunks,
            )
            for key, value in slot.items()
        }
        return slot

    elif slot_name == "raw":
        # dead code
        mask_dir = mask_dir / "raw"
        for slot_name in ALL_SLOTS.items():
            slot.X = _subset_matrix(slot.X, mask_dir)
            slot.var = _subset_slot(slot_name, slot.var, mask_dir)
            slot.varm = _subset_slot(slot_name, slot.varm, mask_dir)

    elif slot_name.startswith("raw/"):
        slot_name = slot_name.split("/", 1)[-1]
        mask_dir = mask_dir.parent / "raw" / mask_dir.name
        slot = subset_slot(slot_name=slot_name, slot=slot, mask_dir=mask_dir)

    elif slot_name == "X":
        slot = _subset_matrix(slot, mask_dir / slot_name)

    elif slot_name.startswith("layers/"):
        slot = _subset_matrix(slot, mask_dir / slot_name)

    elif slot_name != "uns":
        slot = _subset_slot(slot_name, slot, mask_dir / slot_name)

    if isinstance(slot, da.Array):
        slot = slot.rechunk(chunks=chunks)

    if isinstance(slot, pd.DataFrame):
        for col in slot.columns:
            if slot[col].dtype.name != "category":
                continue
            slot[col] = slot[col].cat.remove_unused_categories()

    return slot


def _subset_matrix(slot, mask_dir):
    obs_mask_file = mask_dir / "obs.npy"
    var_mask_file = mask_dir / "var.npy"
    if not obs_mask_file.exists() or not var_mask_file.exists():
        return slot

    obs_mask = np.load(obs_mask_file)
    var_mask = np.load(var_mask_file)

    return slot[obs_mask, :][:, var_mask].copy()


def _subset_slot(slot_name, slot, mask_dir):
    mask_file = mask_dir / "mask.npy"

    if not mask_file.exists():
        return slot

    mask = np.load(mask_file)
    assert mask.shape[0] == slot.shape[0], (
        f"Mask shape {mask.shape} does not match slot shape {slot.shape} for slot {slot_name}"
    )

    if slot_name.startswith("obsp/") or slot_name.startswith("varp/"):
        # subset in both dimensions
        return slot[mask, :][:, mask].copy()

    return slot[mask].copy()
