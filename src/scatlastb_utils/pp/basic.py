from collections.abc import Callable, Iterable
from typing import Any, TypeVar

import numpy as np
from anndata import AnnData

MuData = TypeVar("MuData")
SpatialData = TypeVar("SpatialData")

ScverseDataStructures = AnnData | MuData | SpatialData

import pandas as pd
import numpy as np
import re


def basic_preproc(adata: AnnData) -> int:
    """Run a basic preprocessing on the AnnData object.

    Parameters
    ----------
    adata
        The AnnData object to preprocess.

    Returns
    -------
    Some integer value.
    """
    print("Implement a preprocessing function here.")
    return 0


def elaborate_example(
    items: Iterable[ScverseDataStructures],
    transform: Callable[[Any], str],
    *,  # functions after the asterix are key word only arguments
    layer_key: str | None = None,
    mudata_mod: (str | None) = "rna",  # Only specify defaults in the signature, not the docstring!
    sdata_table_key: str | None = "table1",
    max_items: int = 100,
) -> list[str]:
    """A method with a more complex docstring.

    This is where you add more details.
    Try to support general container classes such as Sequence, Mapping, or Collection
    where possible to ensure that your functions can be widely used.

    Parameters
    ----------
    items
        AnnData, MuData, or SpatialData objects to process.
    transform
        Function to transform each item to string.
    layer_key
        Optional layer key to access matrix to apply transformation on.
    mudata_mod
        Optional MuData modality key to apply transformation on.
    sdata_table_key
        Optional SpatialData table key to apply transformation on.

    Returns
    -------
    List of transformed string items.

    Examples
    --------
    >>> elaborate_example(
    ...     [adata, mudata, spatial_data],
    ...     lambda vals: f"Statistics: mean={vals.mean():.2f}, max={vals.max():.2f}",
    ...     {"var_key": "CD45", "modality": "rna", "min_value": 0.1},
    ... )
    ['Statistics: mean=1.24, max=8.75', 'Statistics: mean=0.86, max=5.42']
    """
    result: list[Any] = []

    for item in items:
        if isinstance(item, AnnData):
            matrix = item.X if not layer_key else item.layers[layer_key]
        elif isinstance(item, MuData):
            matrix = item.mod[mudata_mod].X if not layer_key else item.mod[mudata_mod].layers[layer_key]
        elif isinstance(item, SpatialData):
            matrix = item.tables[sdata_table_key].X if not layer_key else item.tables[sdata_table_key].layers[layer_key]
        else:
            raise ValueError(f"Item {item} must be of type AnnData, MuData, or SpatialData but is {item.__class__}.")

        if not isinstance(matrix, np.ndarray):
            raise ValueError(f"Item {item} matrix is not a Numpy matrix but of type {matrix.__class__}")

        result.append(transform(matrix.flatten()))

        if len(result) >= max_items:
            break

    return result


def strip_barcodes(obs_names):
    """
    Strip a list of cell names (e.g. from ``.obs_names`` of an object) to just 
    the 16 base pair 10X cell barcodes.
    
    Parameters
    ----------
    obs_names
        A list of cell names to strip down to just barcodes.
    
    Returns
    -------
    A list of the input cell names, stripped down to just the 16 base pair 10X 
    cell barcode.
    """
    #apply a simple regex to whatever the obs_names may be formatted as
    #find a 16-size block of [ACTG] in there
    #start by checking if the barcodes are present everywhere to throw an error
    #bool(re.search()) is false if no hits are found
    if not np.all([bool(re.search("[ACTG]{16}", i)) for i in obs_names]):
        raise ValueError("Not all input barcodes have a 16bp 10X barcode present")
    #return the first match - we've got the barcode
    return [re.search("[ACTG]{16}", i).group(0) for i in obs_names]


def find_library_obs(library_obs_names, obs, library_key="library_id"):
    """
    Group the input ``obs`` on the specified ``library_key``, identify the 
    best barcode match with the input ``library_obs_names``, and yield a 
    combination of the ``obs`` subset to the best matching library along with 
    summary statistics about the identified overlap.
    
    Parameters
    ----------
    library_obs_names
        A list of cell names, with the 16 base pair 10X barcode present, that 
        you are trying to find the best match for in the ``obs``.
    obs
        A ``pandas.DataFrame`` of cell level metadata, with the index having 
        the 16 base pair 10X barcode present, and ``library_key`` present as a 
        column.
    library_key
        The ``obs`` column holding library information. Default: ``"library_id"``
    
    Returns
    -------
    A tuple: a subset of ``obs`` for the best matching library, with the index 
    stripped to just the 16 base pair 10X cell barcode, and a ``pandas.Series`` 
    with information about the identified overlap: the library ID in ``obs``, 
    the input cell count (i.e. the length of ``library_obs_names``), the number 
    of cells in for the library in ``obs``, and the length of the overlap of the 
    two cell pools.
    """
    #is our library column in the data frame?
    if library_key not in obs.columns:
        raise KeyError(library_key+" missing from ``obs.columns``")
    #we want to stash info on the best match we find
    #the library name in the provided obs, the number of cells on input
    #the number of cells for the library in the obs, and the intersection
    found = pd.Series(index=["library", "input_cell_count", "library_cell_count", "overlap"], dtype="object")
    #the input cell count is fixed, and we need to initialise the overlap at 0
    found["input_cell_count"] = len(library_obs_names)
    found["overlap"] = 0
    #strip the input to just the barcodes, as that's what we'll be comparing to
    library_obs_names = strip_barcodes(library_obs_names)
    #iterate over the libraries in the obs
    for library in obs[library_key].unique():
        #pull out the barcodes for it, stripping anything else away
        obs_bcs = strip_barcodes(obs[obs[library_key] == library].index)
        #check intersection with input
        current_overlap = len(set(library_obs_names).intersection(set(obs_bcs)))
        #if we have a new best match, stash it as the best hit
        if current_overlap > found["overlap"]:
            found["library"] = library
            found["library_cell_count"] = len(obs_bcs)
            found["overlap"] = current_overlap
    #at this point we can pull out the subset and strip the barcodes
    sub = obs[obs[library_key] == found["library"]].copy(deep=True)
    sub.index = strip_barcodes(sub.index)
    #and now we have everything we want, yield both the subset and the match stats
    return sub, found