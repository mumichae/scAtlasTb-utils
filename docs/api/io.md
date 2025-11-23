(reading)=
(reading-and-writing)=

# Reading and Writing: `io`

```{eval-rst}
.. currentmodule:: scatlastb_utils.io
```

## Reading

Optimised reading wrapper functions for partial loading of `anndata.AnnData` slots and loading matrices as backed `dask` arrays:

```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: ../generated/

    read_anndata
    read_slot
```

## Writing

Optimised writing with linked zarr files to save storage space when input and output slots remain the same:

```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: ../generated/

    write_zarr_linked
```
