import anndata as ad
import dask.array as da
import numpy as np
import sparse
from scipy import sparse as sp
from tqdm import dask, tqdm


def get_adjacency(
    X,
    weighted: bool,
    k: int = 1,
    _lambda: float = 1,
    tol: float = 1e-5,
    use_dask: bool = False,
    chunk_size: int = 50_000,
):
    """Retrieve adjacency matrix from distances or connectivities.

    :param weighted: whether to preserve weighting of adjacency or return unweighted adjacency
        Default: weighted=False - 1 when edge present, 0 when edge not present
    :param k: power of matrix multiplication. Computationally expensive and densifies the matrix.
        Default: 1, no multiplication performed.
    """
    assert sp.issparse(X) or isinstance(X, da.Array), "matrix must be scipy sparse matrix or dask array"

    if use_dask and not isinstance(X, da.Array):
        X = da.from_array(X, chunks=(-1, chunk_size)).map_blocks(sparse.COO)

    for _ in range(1, k):
        X = X.dot(X.T)

    if _lambda != 1:
        from decimal import Decimal  # for exact values

        X *= np.float32(Decimal(_lambda) ** k)

    if sp.issparse(X):
        X.data[X.data < tol] = 0
        X.eliminate_zeros()

    if not weighted:
        X = (X > 0).astype("float32")
    return X


def knn_overlap(
    adata: ad.AnnData,
    obsp_keys: list,
    operator="intersection",
    use_dask=True,
    chunk_size=50_000,
    power_k=1,
    _lambda=1,
    weighted=False,
    log_scale=False,
    max_scale=False,
    scale="min",
):
    """
    kNN-based graph similarity

    :params obsp_keys: list of
    :param operator: operator='intersection' -> kNN overlap, operator='union' -> number of edge occurrence for all embeddings
    """
    # set tqdm for non-dask computations
    obsp_keys = tqdm(obsp_keys) if not use_dask else obsp_keys

    x_diff = None

    for key in obsp_keys:
        x = get_adjacency(
            adata.obsp[key],
            weighted=weighted,
            k=power_k,
            _lambda=_lambda,
            use_dask=use_dask,
        )

        if x_diff is None:
            # assuming that obsp_keys references weighted adjacency matrix
            x_diff = x.copy()
            min_degrees = x_diff.sum(axis=0)
            union = x_diff.copy()
            continue

        # calculate global minimum degree
        min_degrees = np.minimum(min_degrees, x.sum(axis=0))

        # calculate union of edges
        union = da.maximum(union, x) if use_dask else union.maximum(x)

        if operator == "intersection":
            x_diff = get_adjacency(
                x_diff,
                weighted=weighted,
                k=power_k,
                _lambda=_lambda,
                use_dask=use_dask,
            )
            # element-wise multiplication (logical AND) -> intersection of same edges
            x_diff = da.multiply(x_diff, x) if use_dask else x_diff.multiply(x)

        # if operator == 'difference':
        #     # element-wise multiplication -> intersection of same edges
        #     x_diff = 1 - da.multiply(x_diff, x) if use_dask else 1 - x_diff.multiply(x)

        elif operator == "sum":
            # element-wise sum -> number of graphs that edge exists
            x_diff += x

        elif operator == "xor":
            # element-wise != -> highlight differences between edges
            x_diff = (x_diff != x).astype("float32")  # logical XOR

        elif operator == "product":
            # matrix multiplication
            x_diff = x_diff.dot(x)

            if not use_dask:
                x_diff.data[x_diff.data < 1e-5] = 0
                x_diff.eliminate_zeros()

        else:
            raise ValueError(f'unknown operator "{operator}"')

    # sum up intersecting edges per node
    x_diff = x_diff.sum(axis=0)

    if isinstance(x_diff, da.Array):
        with dask.TqdmCallback(desc=f"compute graph {operator}"):
            x_diff = x_diff.compute().todense()
    else:
        x_diff = x_diff.A1

    # determine scaling factors
    scaling_factor = 1
    if scale == "min":
        scaling_factor = min_degrees + 1
    elif scale in ["max", "union"]:
        scaling_factor = union.sum(axis=0) + 1

    if isinstance(scaling_factor, da.Array):
        with dask.TqdmCallback(desc=f"compute {scale} scaling factor"):
            scaling_factor = scaling_factor.compute().todense()
    elif isinstance(scaling_factor, np.matrix):
        scaling_factor = scaling_factor.A1

    # scale values
    x_diff /= scaling_factor

    if log_scale:
        x_diff = np.log1p(x_diff)

    if max_scale:
        x_diff /= x_diff.max()

    return x_diff
