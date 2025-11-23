"""Distance-based graph similarity."""

import multiprocessing

import dask.array as da
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from matplotlib import pyplot as plt
from scipy import sparse as sp
from scipy.spatial import distance
from tqdm import tqdm

multiprocessing.set_start_method("spawn", force=True)


def _symmetrize_mask(x, verbose=False):
    """Create a mask for symmetrizing a sparse matrix.

    This function identifies the entries in the upper triangular part of the matrix that are not mirrored in the lower triangular part.
    It returns two arrays: one for the row indices and one for the column indices of the entries that need to be symmetrized.
    The function assumes that the input matrix is sparse and that the entries are non-negative.

    :param x: sparse matrix to symmetrize
    :param verbose: if True, print the number of entries to be symmetrized
    :return: `(row_mask, col_mask)` containing the row and column indices of the entries to be symmetrized
    """
    row_mask, col_mask = [], []
    triangles = sp.triu(x > 0), sp.tril(x > 0).T
    diff_mask = triangles[0] != triangles[1]

    for i, tri in enumerate(triangles):
        indices = diff_mask.multiply(tri).nonzero()
        row_mask.append(indices[i])
        col_mask.append(indices[(i + 1) % 2])

    row_mask = np.concatenate(row_mask)
    col_mask = np.concatenate(col_mask)

    if verbose:
        print(f"{len(row_mask)} entries to be symmetrized")

    return row_mask, col_mask


def is_symmetric(x):
    """Check if a sparse matrix is symmetric.

    A matrix is symmetric if it is equal to its transpose.

    :param x: sparse matrix to check
    :return: True if the matrix is symmetric, False otherwise
    """
    # Check if the matrix is equal to its transpose
    # return (x != x.transpose()).nnz == 0
    # return sp.triu(x).nnz == sp.tril(x).nnz
    return (x.sum(0) % 2 == 0).all()
    # return (x != x.T).nnz == 0


def symmetrize_if_needed(x):
    """Symmetrize any matrix if it is not symmetric.

    Assumes that values that are 0 are not defined in the other direction.
    If the matrix is symmetric, it is returned as is.
    If the matrix is not symmetric, it is symmetrized by taking the maximum of the two directions.

    :param x: sparse matrix to symmetrize
    :return: symmetrized sparse matrix
    """
    if is_symmetric(x):
        return x
    return x.maximum(x.T)


def _compute_distances(rows, cols, obsm, n_jobs=-1, batch_size=100_000, **kwargs):
    """Compute distances for given rows and columns in obsm.

    :param rows: row indices for which distances should be computed
    :param cols: column indices for which distances should be computed
    :param obsm: embedding matrix from which distances are computed
    :param n_jobs: number of jobs to run in parallel, -1 for all available cores
    :param batch_size: batch size for distance computation
    :param kwargs: additional keyword arguments for distance computation, e.g. metric='euclidean'
    :return: distances for the given rows and columns
    """

    def _compute_row_distance(obsm, row, first, last):
        indices = np.arange(first, last)
        dist = distance.cdist(obsm[[row], :], obsm[cols[first:last], :], **kwargs)
        return indices, dist

    def _compute_batch(obsm, batch_rows, batch_firsts, batch_lasts):
        return [_compute_row_distance(obsm, *args) for args in zip(batch_rows, batch_firsts, batch_lasts, strict=False)]

    unique_rows, first_occurrence, counts = np.unique(rows, return_index=True, return_counts=True)
    last_occurrence = first_occurrence + counts
    n_rows = len(unique_rows)

    if n_jobs == 1:
        results = [
            _compute_row_distance(obsm, *args)
            for args in tqdm(
                zip(unique_rows, first_occurrence, last_occurrence, strict=False),
                desc="Computing distances",
                mininterval=1,
                total=n_rows,
            )
        ]

    else:
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "obsm.npy")
            np.save(filename, obsm)
            obsm_mmap = np.load(filename, mmap_mode="r")

            batched_args = (
                (
                    unique_rows[i : i + batch_size],
                    first_occurrence[i : i + batch_size],
                    last_occurrence[i : i + batch_size],
                )
                for i in range(0, n_rows, batch_size)
            )
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_compute_batch)(obsm_mmap, *args)
                for args in tqdm(batched_args, desc="Computing distances", total=int(n_rows / batch_size))
            )
            # Flatten the list of lists
            results = (item for sublist in results for item in sublist)

    distances = np.zeros(rows.shape[0])
    for indices, dist in results:
        distances[indices] = dist

    # # check if distances added to correct location
    # distances_correct = np.zeros(n_distances)
    # for i in tqdm(range(n_distances), desc='Compute missing distances', mininterval=1):
    #     dist = distance.cdist(obsm[[rows[i]], :], obsm[[cols[i]], :])[0][0]
    #     distances_correct[i] = dist
    # print(np.all(distances == distances_correct))

    return distances


def compute_missing_distances(
    adata,
    obsp_key_1,
    obsm_key,
    obsp_key_2,
    inplace=True,
    return_matrix=True,
    n_jobs=-1,
    batch_size=100_000,
    **kwargs,
):
    """Compute missing distances.

    Due to scanpy's heuristic, only distances for k-nearest neighbors are kept.
    In order to compare distances across embeddings, the corresponding distances of edges from one embedding
    must be present or recomputed for the other embedding.

    :param obsp_key_1: slot for pair-wise distances from embedding 1
    :param obsm_key: slot for embedding 1 used for missing distance computation
    :param obsp_key_2: slot slot for pair-wise distances from embedding 2
    :param inplace: if True, set distances inplace in adata.obsp[obsp_key_1]
    :param n_jobs: number of jobs to run in parallel, -1 for all available cores
    :param batch_size: batch size for distance computation
    :param kwargs: additional keyword arguments for distance computation, e.g. metric='euclidean'
    :param return_matrix: if True, return the distance matrix
    """
    x1 = adata.obsp[obsp_key_1]
    x2 = adata.obsp[obsp_key_2]
    obsm = adata.obsm[obsm_key]

    if isinstance(x1, da.Array):
        x1 = x1.compute()
    if isinstance(x2, da.Array):
        x2 = x2.compute()

    # fill in missing values to make distance matrix symmetric
    for _ in tqdm(range(1), desc="Symmetrize matrix"):
        x1 = symmetrize_if_needed(x1)
        x2 = symmetrize_if_needed(x2)

    # get neighbors that are not computed in the other graph
    print("Determine missing distances...")
    x1_adj, x2_adj = sp.triu(x1 > 0), sp.triu(x2 > 0)
    # XOR and x1 values > 0
    rows, cols = (x1_adj != x2_adj).multiply(x2_adj).nonzero()
    del x1_adj, x2_adj

    n_distances = len(rows)
    print(f"{n_distances} edges to recompute")

    if n_distances > 0:
        # compute distances
        new_distances = _compute_distances(
            rows,
            cols,
            obsm,
            n_jobs=n_jobs,
            batch_size=batch_size,
            **kwargs,
        )

        # add distances to distance matrix in place
        new_distances = sp.coo_matrix(
            (new_distances, (rows, cols)),
            shape=x1.shape,
            dtype="float32",
        )
        x1 = (x1.tocoo() + new_distances + new_distances.T).tocsr()

    if inplace:
        print("Set distances inplace...")
        adata.obsp[obsp_key_1] = x1

    if return_matrix:
        return x1


def plot_ranked_distances(adata, x1, x2):
    """Diagnostic plot for distance computation.

    This function plots the ranked average distances and differences between two embeddings.

    :param adata: AnnData object containing the distances in obs
    :param x1: name of the graph distances of the first embedding
    :param x2: name of the graph distances of the second embedding
    """
    # TODO: move to pl?

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    metrics = ["avg_distance_1", "avg_distance_2"]
    for x, metric in zip([x1, x2], metrics, strict=False):
        ax1.plot(adata.obs[f"{metric}:{x1}-vs-{x2}"].sort_values().values, label=x)
    ax1.set_xlabel("rank")
    ax1.set_ylabel("avg distance")
    ax1.set_title("Ranked average distance per node")
    ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))

    metrics = ["avg_difference", "avg_distance_diff"]
    for metric in metrics:
        ax2.plot(adata.obs[f"{metric}:{x1}-vs-{x2}"].sort_values().values, label=metric)
    ax2.set_xlabel("rank")
    ax2.set_ylabel("score")
    ax2.set_title("Ranked graph differences")
    ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))

    fig.subplots_adjust(wspace=1)
    plt.show()


def plot_distances_scatter(adata, x1, x2, **kwargs):
    """Diagnostic plot for distance computation.

    This function plots the average distances and differences between two embeddings in a scatter plot.

    :param adata: AnnData object containing the distances in obs
    :param x1: name of the graph distances of the first embedding
    :param x2: name of the graph distances of the second embedding
    :param kwargs: additional keyword arguments for the plt.scatter plot, e.g. `c`, `s`, `alpha`
    """
    # TODO: move to pl?

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(
        adata.obs[f"avg_distance_1:{x1}-vs-{x2}"],
        adata.obs[f"avg_distance_2:{x1}-vs-{x2}"],
        **kwargs,
    )
    ax1.set_xlabel(x1)
    ax1.set_ylabel(x2)
    ax1.set_title("Average distance per node")

    metrics = ["avg_difference", "avg_distance_diff"]
    for metric in metrics:
        ax2.scatter(
            adata.obs[f"{metric}:{x1}-vs-{x2}"],
            adata.obs[f"{metric}:{x2}-vs-{x1}"],
            label=metric,
            **kwargs,
        )
    ax2.set_xlabel(metrics[0])
    ax2.set_ylabel(metrics[1])
    ax2.set_title("Graph differences")
    ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))

    fig.subplots_adjust(wspace=0.5)
    plt.show()


def sparse_spearman(matrix1: sp.spmatrix, matrix2: sp.spmatrix, max_workers=None, n_jobs=-1):
    """Calculate Spearman correlation for each row in two sparse distance matrices.

    This function computes the Spearman correlation only for local neighborhoods.

    :param matrix1: First sparse matrix (scipy.sparse format).
    :param matrix2: Second sparse matrix (scipy.sparse format).
    :param max_workers: Maximum number of workers for parallel computation.
    :param n_jobs: Number of jobs to run in parallel. If -1, use all available cores.
    """

    def _sparse_spearman(row1, row2):
        def pad_zeros(x, shape):
            x_pad = np.zeros(shape, dtype=x.dtype)
            x_pad[-x.shape[0] :] = x
            return x_pad

        n, n2 = row1.shape[0], row2.shape[0]

        # Avoid division by zero for constant rows
        if n <= 1 or n2 <= 1 or np.all(row1 == row1[0]) or np.all(row2 == row2[0]):
            return np.nan

        # check if rows have same length, otherwise pad with zeros
        if n > n2:
            row2 = pad_zeros(row2, n)
        elif n < n2:
            row1 = pad_zeros(row1, n2)

        # Rank the data
        rank1 = np.argsort(np.argsort(row1)) + 1
        rank2 = np.argsort(np.argsort(row2)) + 1

        # Calculate Spearman correlation
        return 1 - (6 * np.sum((rank1 - rank2) ** 2)) / (n * (n**2 - 1))

    def get_row_data(x, i):
        return x.data[x.indptr[i] : x.indptr[i + 1]]

    assert matrix1.shape == matrix2.shape, f"shape mismatch: {matrix1.shape}, {matrix2.shape}"

    n_rows = matrix1.shape[0]
    row_data = [(get_row_data(matrix1, i), get_row_data(matrix2, i)) for i in range(n_rows)]

    with parallel_backend("loky"):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_sparse_spearman)(row1, row2)
            for row1, row2 in tqdm(row_data, desc="Spearman correlation", mininterval=1, total=n_rows)
        )

    return np.array(results)


def get_knn(x, k):
    """Get k-nearest neighbors for each row in a sparse matrix."""
    from numba import njit

    @njit
    def filter_row_knn(data, indices, indptr, k):
        new_data = []
        new_indices = []
        new_indptr = [0]

        for i in range(len(indptr) - 1):
            start = indptr[i]
            end = indptr[i + 1]

            row_data = data[start:end]
            row_indices = indices[start:end]

            if len(row_data) <= k:
                mask = np.ones(len(row_data), dtype=np.bool_)
            else:
                # get k-largest threshold
                threshold = np.partition(row_data, -k)[-k]
                mask = row_data >= threshold

            new_data.extend(row_data[mask])
            new_indices.extend(row_indices[mask])
            new_indptr.append(len(new_data))

        return np.array(new_data), np.array(new_indices), np.array(new_indptr)

    if isinstance(x, da.Array):
        x = x.compute()
    x = sp.csr_matrix(x)  # ensure CSR format

    data, indices, indptr = filter_row_knn(x.data, x.indices, x.indptr, k)
    return sp.csr_matrix((data, indices, indptr), shape=x.shape)


def compare_distances(
    adata,
    obsp_key_1,
    obsm_key_1,
    obsp_key_2,
    obsm_key_2,
    scale_distances=True,
    quantile=0.9,
    k_max=50,
    log_scale_diffs=False,
    **kwargs,
):
    """Compare distances of same edges but from different representations.

    This function computes:

    1. "average_distance_1": the average distances per k-nearest neighborhood for obsp_key_1
    2. "average_distance_2": the average distances per k-nearest neighborhood for obsp_key_2
    3. "average_difference": the difference of 1. and 2.
    4. "average_distance_diff": the average of the differences between the two embeddings
    5. "spearman_correlation": the Spearman correlation of the distance differences (of the k-nearest neighbors only)

    :param adata: AnnData object containing the distances in obsp
    :param obsp_key_1: slot for pair-wise distances from embedding 1
    :param obsm_key_1: slot for embedding 1 used for distance computation
    :param obsp_key_2: slot for pair-wise distances from embedding 2
    :param obsm_key_2: slot for embedding 2 used for distance computation
    :param scale_distances: if True, scale distances by the quantile of the distances
    :param quantile: quantile to scale distances by, default is 0.9
    :param k_max: maximum number of neighbors to consider, default is 50
    :param log_scale_diffs: if True, log scale the differences
    :param kwargs: additional keyword arguments for distance computation, e.g. metric='euclidean'
    :return: DataFrame with average distances and differences
    """
    if isinstance(adata.obsp[obsp_key_1], da.Array):
        adata.obsp[obsp_key_1] = adata.obsp[obsp_key_1].compute()

    if isinstance(adata.obsp[obsp_key_2], da.Array):
        adata.obsp[obsp_key_2] = adata.obsp[obsp_key_2].compute()

    x2 = compute_missing_distances(
        adata,
        obsp_key_1=obsp_key_2,
        obsm_key=obsm_key_2,
        obsp_key_2=obsp_key_1,
        inplace=False,
        return_matrix=True,
        **kwargs,
    ).copy()

    x1 = get_knn(adata.obsp[obsp_key_1], k=k_max)
    nnz_mask = x1 > 0
    x2 = x2.multiply(nnz_mask)
    x2.eliminate_zeros()

    degrees = nnz_mask.sum(axis=0)
    degrees[degrees == 0] = 1
    # print('zero degrees', (degrees == 0).sum())
    # print('zero degrees', ((x1 > 0).sum(axis=0) == 0).sum())
    # print('zero degrees', (x1.sum(axis=0) == 0).sum())

    if scale_distances:
        # x1.data = 1 / np.log10(x1.data + 1)
        # x2.data = 1 / np.log10(x2.data + 1)
        print(f"scale distances by {quantile} quantile...")
        x1.data /= np.quantile(adata.obsp[obsp_key_1].data, q=quantile)
        x2.data /= np.quantile(adata.obsp[obsp_key_2].data, q=quantile)

    print("Calculate differences...")
    diff_mtx = x1 - x2
    diff_mtx.data = np.abs(diff_mtx.data)
    avg_diff = (diff_mtx.sum(axis=0) / degrees).A1
    avg_dist_diff = np.abs((x1.sum(axis=0) - x2.sum(axis=0)) / degrees).A1

    # if scale:
    #     print(f'scale by {quantile} quantile...')
    #     avg_diff /= np.quantile(avg_diff, q=quantile)
    #     avg_dist_diff /= np.quantile(avg_dist_diff, q=quantile)

    if log_scale_diffs:
        print("log scale...")
        avg_diff = np.log10(avg_diff + 1)
        avg_dist_diff = np.log10(avg_dist_diff + 1)

    return pd.DataFrame(
        {
            # mean only for neighborhoods of interest
            "average_distance_1": (x1.sum(axis=0) / degrees).A1,
            "average_distance_2": (x2.sum(axis=0) / degrees).A1,
            "average_difference": avg_diff,
            "average_distance_diff": avg_dist_diff,
            "spearman_correlation": sparse_spearman(x1, x2),
        },
        index=adata.obs_names,
    )
