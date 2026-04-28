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


def is_symmetric(x, tol: float = 0.0) -> bool:
    """Check if a sparse matrix is symmetric.

    :param x: sparse matrix to check
    :param tol: tolerance for numeric differences (default 0.0)
    :return: True if the matrix is symmetric within `tol`, False otherwise
    """
    # Compute sparse difference; stays sparse and scales with nnz
    diff = x - x.T
    if diff.nnz == 0:
        return True
    if tol > 0.0:
        return np.max(np.abs(diff.data)) <= tol
    return False


def is_constant(arr):
    """Check if all values in an array are the same."""
    return arr[0] == arr[-1] and arr.min() == arr.max()


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
    obsp_key,
    obsm_key,
    conn_mask,
    inplace=True,
    return_matrix=True,
    n_jobs=-1,
    batch_size=100_000,
    **kwargs,
):
    """Recompute missing pairwise distances restricted to a neighbor mask.

    :param adata: AnnData-like object with `obsp` and `obsm` attributes.
    :param obsp_key: Key in ``adata.obsp`` containing the distance matrix to update.
    :param obsm_key: Key in ``adata.obsm`` containing the embedding used to compute distances.
    :param conn_mask: scipy.sparse binary mask of connectivities to determine missing distances.
    :param inplace: If True, store the updated distance matrix back to ``adata.obsp[obsp_key]``.
    :param return_matrix: If True, return the updated matrix.
    :param n_jobs: Number of parallel jobs to use for distance computation.
    :param batch_size: Batch size used for distance computation.
    :param kwargs: Passed to the distance computation routine (e.g., ``metric``).

    :returns: scipy.sparse matrix when ``return_matrix`` is True, otherwise None.
    """
    x = adata.obsp[obsp_key]
    obsm = adata.obsm[obsm_key]

    if isinstance(x, da.Array):
        x = x.compute()
    if isinstance(conn_mask, da.Array):
        conn_mask = conn_mask.compute()

    # fill in missing values to make distance matrix symmetric
    x = symmetrize_if_needed(x)

    # determine candidate neighbor edges from provided mask and select those missing in x
    rows, cols = sp.triu(conn_mask).nonzero()
    positive = x[rows, cols].A1 > 0
    rows, cols = rows[~positive], cols[~positive]

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
        new_distances = sp.coo_matrix((new_distances, (rows, cols)), shape=x.shape, dtype="float32")
        x = (x.tocoo() + new_distances + new_distances.T).tocsr()

    if inplace:
        print("Set distances inplace...")
        adata.obsp[obsp_key] = x

    if return_matrix:
        return x


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
    comp1 = f"{x1}-vs-{x2}"
    comp2 = f"{x2}-vs-{x1}"
    # TODO: move to pl?

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(
        adata.obs[f"avg_distance_1:{comp1}"],
        adata.obs[f"avg_distance_2:{comp1}"],
        **kwargs,
    )
    ax1.set_xlabel(x1)
    ax1.set_ylabel(x2)
    ax1.set_title("Average distance per node")

    metrics = ["avg_difference", "avg_distance_diff"]
    for metric in metrics:
        x = f"{metric}:{comp1}"
        y = f"{metric}:{comp2}"
        ax2.scatter(adata.obs[x], adata.obs[y], label=metric, **kwargs)
    ax2.set_xlabel(comp1)
    ax2.set_ylabel(comp2)
    ax2.set_title("Graph differences")
    ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))

    fig.subplots_adjust(wspace=0.5)
    plt.show()


def sparse_spearman(
    m1: sp.spmatrix,
    m2: sp.spmatrix,
    mask: sp.spmatrix,
    n_jobs: int = 1,
    batch_size: int = 10_000,
) -> np.ndarray:
    """Calculate Spearman correlation for each row, restricted to local neighborhoods.

    For each row i, only columns j where mask[i, j] != 0 are considered.
    Values absent in m1 or m2 but present in mask are treated as 0.
    Rows with fewer than 2 neighbors or constant values return NaN.

    :param m1: First sparse matrix (scipy.sparse CSR format preferred).
    :param m2: Second sparse matrix (scipy.sparse CSR format preferred).
    :param mask: Sparse binary mask defining local neighborhoods.
        Non-zero entries at [i, j] indicate that column j is a
        valid neighbor for row i.
    :param n_jobs: Number of parallel jobs. -1 uses all available cores.
    :param batch_size: Number of rows to process per batch. Controls the
        tradeoff between memory usage and parallelisation overhead.
    :return: 1D array of Spearman correlations, one per row. NaN where
        correlation is undefined (constant row or fewer than 2 neighbors).
    """
    from scipy.stats import rankdata

    assert m1.shape == m2.shape == mask.shape, f"Shape mismatch: {m1.shape}, {m2.shape}, {mask.shape}"

    # CSR for efficient row slicing
    m1 = m1.tocsr()
    m2 = m2.tocsr()
    mask = mask.tocsr()
    n_rows = m1.shape[0]

    def _get_row_values(i):
        cols = mask.indices[mask.indptr[i] : mask.indptr[i + 1]]
        if len(cols) <= 1:
            return None, None
        row1 = m1[i, cols].toarray().ravel()
        row2 = m2[i, cols].toarray().ravel()
        return row1, row2

    def _spearman(row1, row2) -> float:
        if row1 is None:
            return np.nan
        if is_constant(row1) or is_constant(row2):
            return np.nan
        n = len(row1)
        rank1 = rankdata(row1)
        rank2 = rankdata(row2)
        return 1.0 - (6.0 * np.sum((rank1 - rank2) ** 2)) / (n * (n**2 - 1))

    results = []
    batches = range(0, n_rows, batch_size)
    for start in tqdm(batches, desc="Spearman correlation", unit="batch"):
        end = min(start + batch_size, n_rows)
        batch = [_get_row_values(i) for i in range(start, end)]
        with parallel_backend("loky"):
            batch_results = Parallel(n_jobs=n_jobs)(delayed(_spearman)(row1, row2) for row1, row2 in batch)
        results.extend(batch_results)

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
    obsp_connectivities_1,
    obsp_distances_1,
    obsm_key_1,
    obsp_connectivities_2,
    obsp_distances_2,
    obsm_key_2,
    scale_distances=True,
    quantile=0.9,
    k_max=None,
    log_scale_diffs=False,
    n_jobs=1,
    batch_size=10_000,
    **kwargs,
):
    """Compare distances of same edges but from different representations.

    This function computes:

    1. "average_distance_1": the average distances per k-nearest neighborhood for obsp_key_1
    2. "average_distance_2": the average distances per k-nearest neighborhood for obsp_key_2
    3. "average_difference": the difference of 1. and 2.
    4. "average_distance_diff": the average of the differences between the two embeddings
    5. "spearman_correlation": the Spearman correlation of the distance differences (of the k-nearest neighbors only)

    Neighborhoods are inferred from connectivities of embedding 1. If k_max is None, all non-zero
    connectivities are used.

    :param adata: AnnData object containing the graph data in obsp
    :param obsp_connectivities_1: slot for connectivities from embedding 1
    :param obsp_distances_1: slot for pair-wise distances from embedding 1
    :param obsm_key_1: slot for embedding 1 used for missing distance computation
    :param obsp_connectivities_2: slot for connectivities from embedding 2
    :param obsp_distances_2: slot for pair-wise distances from embedding 2
    :param obsm_key_2: slot for embedding 2 used for missing distance computation
    :param scale_distances: if True, scale distances by the quantile of the distances
    :param quantile: quantile to scale distances by, default is 0.9
    :param k_max: maximum number of neighbors to consider, default is 50
    :param log_scale_diffs: if True, log scale the differences
    :param n_jobs: number of parallel jobs to run, default is 1
    :param batch_size: number of rows to process per batch, default is 10_000
    :param kwargs: additional keyword arguments for distance computation, e.g. metric='euclidean'
    :return: DataFrame with average distances and differences
    """
    if isinstance(adata.obsp[obsp_distances_1], da.Array):
        adata.obsp[obsp_distances_1] = adata.obsp[obsp_distances_1].compute()

    if isinstance(adata.obsp[obsp_distances_2], da.Array):
        adata.obsp[obsp_distances_2] = adata.obsp[obsp_distances_2].compute()

    if isinstance(adata.obsp[obsp_connectivities_1], da.Array):
        adata.obsp[obsp_connectivities_1] = adata.obsp[obsp_connectivities_1].compute()

    if isinstance(adata.obsp[obsp_connectivities_2], da.Array):
        adata.obsp[obsp_connectivities_2] = adata.obsp[obsp_connectivities_2].compute()

    if k_max is not None:
        conn_mask = get_knn(adata.obsp[obsp_connectivities_1], k=k_max) > 0
    else:
        conn_mask = adata.obsp[obsp_connectivities_1] > 0

    x2 = compute_missing_distances(
        adata,
        obsp_key=obsp_distances_2,
        obsm_key=obsm_key_2,
        conn_mask=conn_mask,
        inplace=False,
        return_matrix=True,
        n_jobs=n_jobs,
        batch_size=batch_size,
        **kwargs,
    ).copy()

    x1 = adata.obsp[obsp_distances_1].copy()
    x1 = x1.multiply(conn_mask)
    x1.eliminate_zeros()
    x2 = x2.multiply(conn_mask)
    x2.eliminate_zeros()

    degrees = conn_mask.sum(axis=0)
    degrees[degrees == 0] = 1

    if scale_distances:
        # x1.data = 1 / np.log10(x1.data + 1)
        # x2.data = 1 / np.log10(x2.data + 1)
        print(f"scale distances by {quantile} quantile...")
        x1.data /= np.quantile(adata.obsp[obsp_distances_1].data, q=quantile)
        x2.data /= np.quantile(adata.obsp[obsp_distances_2].data, q=quantile)

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
            "spearman_correlation": sparse_spearman(x1, x2, mask=conn_mask, n_jobs=n_jobs, batch_size=batch_size),
        },
        index=adata.obs_names,
    )
