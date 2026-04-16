import matplotlib
import pytest

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from scatlastb_utils.pl.embedding import _format_legend_labels, embedding


def make_adata():
    # 6 cells with 2D embedding
    X_umap = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.2],
            [0.2, 0.5],
        ]
    )
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(6)])

    # categories with explicit order (C, A, B)
    cats = pd.Categorical(
        ["C", "A", "A", "C", "A", "B"],
        categories=["C", "A", "B"],
        ordered=True,
    )
    obs["group"] = cats

    adata = AnnData(np.zeros((6, 1)))
    adata.obs = obs
    adata.obsm["X_umap"] = X_umap
    return adata


@pytest.mark.parametrize(
    "bold_labels,expected_fontweights",
    [
        ([], ["normal", "normal", "normal"]),
        (["A"], ["normal", "bold", "normal"]),
        (["C", "B"], ["bold", "normal", "bold"]),
    ],
)
def test_legend_bolding_parametrized(bold_labels, expected_fontweights):
    adata = make_adata()
    fig = sc.pl.embedding(adata, basis="X_umap", color=["group"], show=False, return_fig=True)
    ax = fig.get_axes()[0]
    legend = ax.get_legend()
    assert legend is not None, "Legend should be present"
    _format_legend_labels(legend=legend, obs=adata.obs, color="group", category_index_map=None, bold_labels=bold_labels)
    fontweights = [t.get_fontweight() for t in legend.get_texts()]
    assert fontweights == expected_fontweights


@pytest.mark.parametrize("downsample", [0.5, 0.25, 10])
def test_embedding_downsample_param(downsample):
    adata = make_adata()
    adata = adata.concatenate([adata] * 4)
    n_before = adata.n_obs
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        embedding(
            adata,
            basis="X_umap",
            color="group",
            downsample=downsample,
            output_dir=tmpdir,
            title="Downsample",
            dpi=80,
            figsize=(4, 4),
        )
        files = os.listdir(tmpdir)
        assert any(f.endswith(".png") for f in files)
        assert adata.n_obs == n_before
