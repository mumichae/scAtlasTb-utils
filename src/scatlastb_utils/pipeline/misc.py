import hashlib
import types
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import sparse
from dask import array as da
from scipy import sparse as sp

# Re-exported for backwards compatibility — canonical definitions live in utils
from scatlastb_utils.utils import apply_layers


def get_use_gpu(config):
    """TODO: move to ModuleConfig?"""
    use_gpu = bool(config.get("use_gpu", False))
    if isinstance(use_gpu, str):
        use_gpu = use_gpu.lower() == "true"
    return use_gpu


def all_but(_list, is_not):
    """Returns a list with all elements except the specified one."""
    return [x for x in _list if x != is_not]


def unique_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from a pandas DataFrame.

    :param df: pandas DataFrame
    :return: DataFrame with unique rows
    """
    if df.empty:
        return df
    # hashable_columns = [
    #     col for col in df.columns
    #     if all(isinstance(df[col].iloc[i], typing.Hashable) for i in range(df.shape[0]))
    # ]
    # duplicated = df[hashable_columns].duplicated()
    duplicated = df.astype(str).duplicated()
    return df[~duplicated].reset_index(drop=True)


def expand_dict(_dict: dict) -> zip:
    """Create a cross-product on a dictionary with literals and lists

    :param _dict: dictionary with lists and literals as values
    :return: zip of wildcards and dictionaries
    """
    df = pd.DataFrame({k: [v] if isinstance(v, list) else [[v]] for k, v in _dict.items()})
    for col in df.columns:
        df = df.explode(col)
    dict_list = df.apply(lambda row: dict(zip(df.columns, row, strict=False)), axis=1)

    def remove_chars(s, chars="{} ',"):
        for c in chars:
            s = s.replace(c, "")
        return s

    wildcards = df.apply(
        lambda row: "-".join([remove_chars(f"{col[0]}:{x}") for col, x in zip(df.columns, row, strict=False)]), axis=1
    )
    return zip(wildcards, dict_list, strict=False)


def expand_dict_and_serialize(_dict: dict, do_not_expand: list = None) -> zip:
    """
    Create a cross-product on a dictionary with literals and lists

    :param _dict: dictionary with lists and literals as values
    :return: list of dictionaries with literals as values
    """
    import hashlib

    import jsonpickle

    if do_not_expand is None:
        do_not_expand = []

    df = pd.DataFrame({k: [v] if isinstance(v, list) and k not in do_not_expand else [[v]] for k, v in _dict.items()})
    for col in df.columns:
        df = df.explode(col)
    dict_list = df.apply(lambda row: dict(zip(df.columns, row, strict=False)), axis=1)

    wildcards = [hashlib.blake2b(jsonpickle.encode(d).encode("utf-8"), digest_size=5).hexdigest() for d in dict_list]

    return zip(wildcards, dict_list, strict=False)


def unlist_dict(_dict: dict) -> dict:
    """Unlist a dictionary, converting single-item lists to their values."""
    return {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in _dict.items()}


def unpack_dict_in_df(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Given a column in a pandas dataframe containing dictionaries, extract these to top level.

    :param df: pandas dataframe
    :param column: column name containing dictionaries
    """
    return df.drop(columns=column).assign(**df[column].dropna().apply(pd.Series, dtype=object))


def ifelse(statement, _if, _else):
    """Single-line if-else wrapper.

    :param statement: Condition to evaluate
    :param _if: Value to return if the condition is True
    :param _else: Value to return if the condition is False
    """
    if statement:
        return _if
    else:
        return _else


def check_sparse(matrix, sparse_type=None):
    """Check if a matrix is in sparse format."""
    if sparse_type is None:
        sparse_type = (sp.csr_matrix, sp.csc_matrix, ad.abc.CSRDataset, ad.abc.CSCDataset, sparse.COO)
    elif not isinstance(sparse_type, tuple):
        sparse_type = (sparse_type,)

    # convert to type for functions
    sparse_type = [type(x(0)) if isinstance(x, types.FunctionType) else x for x in sparse_type]
    sparse_type = tuple(sparse_type)

    if isinstance(matrix, da.Array):
        return isinstance(matrix._meta, sparse_type)
    return isinstance(matrix, sparse_type)


def check_sparse_equal(a: sp.spmatrix, b: sp.spmatrix):
    """Check if two matrices are equal in sparse format."""
    a = a if check_sparse(a) else sp.csr_matrix(a)
    b = b if check_sparse(b) else sp.csr_matrix(b)
    if a.shape != b.shape:
        warnings.warn(f"Shape mismatch: {a.shape} != {b.shape}", stacklevel=2)
    return a.shape == b.shape and (a != b).nnz == 0


def ensure_sparse(adata, layers: [str, list] = None, sparse_type=None, **kwargs):
    """Convert matrices in AnnData object to sparse format.

    This function also deals with Dask arrays, ensuring that the chunks are sparse.

    :param adata: AnnData object
    :param layers: List of layers to convert, or 'X', 'raw', or 'all' (default is None, which converts 'X', 'raw', and all layers)
    :param sparse_type: Type of sparse matrix to convert to (default is None, which uses csr_matrix)
    :param kwargs: Additional arguments passed to the apply_layers function
    """

    def to_sparse(matrix, sparse_type=None):
        if sparse_type is None:
            sparse_type = sp.csr_matrix

        if check_sparse(matrix, sparse_type):
            return matrix
        elif isinstance(matrix, da.Array):
            return matrix.map_blocks(sparse_type, dtype=matrix.dtype)
        return sparse_type(matrix)

    return apply_layers(adata, func=to_sparse, layers=layers, sparse_type=sparse_type, **kwargs)


def ensure_dense(adata: ad.AnnData, layers: [str, list] = None, **kwargs):
    """Convert sparse matrices in AnnData object to dense format.

    This function also deals with Dask arrays, ensuring that the chunks are dense.

    :param adata: AnnData object
    :param layers: List of layers to convert, or 'X', 'raw', or 'all' (default is None, which converts 'X', 'raw', and all layers)
    :param kwargs: Additional arguments passed to the apply_layers function
    """

    def to_dense(matrix):
        if isinstance(matrix, da.Array):
            return matrix.map_blocks(np.array)
        if check_sparse(matrix):
            return matrix.toarray()
        return matrix

    return apply_layers(adata, func=to_dense, layers=layers, **kwargs)


def merge(dfs: list, verbose: bool = True, **kwargs):
    """Merge list of dataframes

    :param dfs: list of dataframes
    :param kwargs: arguments passed to pd.merge
    :return: merged dataframe
    """
    from functools import reduce

    merged_df = reduce(lambda x, y: pd.merge(x, y, **kwargs), dfs)
    if verbose:
        print(merged_df)
    return merged_df


def create_hash(string: str, digest_size: int = 5):
    """Create a unique hash from a string using BLAKE2b hashing algorithm."""
    string = string.encode("utf-8")
    return hashlib.blake2b(string, digest_size=digest_size).hexdigest()
