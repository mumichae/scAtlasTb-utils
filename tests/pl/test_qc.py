import anndata as ad
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from matplotlib import pyplot as plt

# Import the function to be tested
from scatlastb_utils.pl.qc import qc_joint


@pytest.fixture
def qc_adata():
    """
    Fixture to create a simple AnnData object for qc_joint tests.
    """
    n_obs = 200
    np.random.seed(42)  # for reproducibility

    obs_data = {
        "n_genes": np.random.randint(500, 5000, n_obs),
        "n_counts": np.random.randint(1000, 10000, n_obs),
        "percent_mito": np.random.rand(n_obs) * 10,
        "cell_type": np.random.choice(["T cell", "B cell", "Macrophage", "NK cell"], n_obs),
        "batch": np.random.choice(["Batch1", "Batch2"], n_obs),
    }
    # Ensure some values are 0 for log1p testing
    obs_data["n_genes"][0] = 0
    obs_data["n_counts"][1] = 0

    return ad.AnnData(np.random.rand(n_obs, 10), obs=pd.DataFrame(obs_data, index=[f"cell_{i}" for i in range(n_obs)]))


def test_qc_joint_basic_plot_creation(qc_adata):
    """Test that qc_joint creates a JointGrid object and a figure."""
    g = qc_joint(qc_adata, x="n_genes", y="n_counts")
    assert isinstance(g, sns.JointGrid)
    assert g.fig is not None
    assert g.ax_joint.get_xlabel() == "n_genes"
    assert g.ax_joint.get_ylabel() == "n_counts"
    plt.close(g.fig)  # Close the figure to prevent it from showing up in test runs


def test_qc_joint_hue_parameter(qc_adata):
    """Test that the 'hue' parameter correctly applies color coding and creates a legend."""
    g = qc_joint(qc_adata, x="n_genes", y="n_counts", hue="cell_type")
    assert isinstance(g, sns.JointGrid)
    legend = g.ax_joint.get_legend()
    assert legend is not None
    assert len(legend.get_texts()) == qc_adata.obs["cell_type"].nunique()
    plt.close(g.fig)


def test_qc_joint_marginal_hue_parameter(qc_adata):
    """
    Test that 'marginal_hue' parameter correctly applies color coding to marginal plots.
    """
    g = qc_joint(qc_adata, x="n_genes", y="n_counts", marginal_hue="batch")
    assert isinstance(g, sns.JointGrid)
    n_unique_batches = qc_adata.obs["batch"].nunique()

    # Check that the marginal plots have the correct number of distinct line elements
    # when marginal_hue is applied with element='step' and fill=False.
    # Each unique hue category should result in a separate Line2D object in ax.lines.
    assert len(g.ax_marg_x.lines) == n_unique_batches
    assert len(g.ax_marg_y.lines) == n_unique_batches

    plt.close(g.fig)


def test_qc_joint_marginal_hue_ignored_for_many_unique_values(qc_adata):
    """
    Test that 'marginal_hue' is ignored if the column has too many unique values (>100).
    """
    n_obs_large = 101  # This will make nunique > 100
    large_adata = ad.AnnData(
        np.random.rand(n_obs_large, 10),
        obs=pd.DataFrame(
            {
                "x": np.random.rand(n_obs_large),
                "y": np.random.rand(n_obs_large),
                "many_unique": [f"id_{i}" for i in range(n_obs_large)],
            }
        ),
    )

    g = qc_joint(large_adata, x="x", y="y", marginal_hue="many_unique")
    assert isinstance(g, sns.JointGrid)
    # When marginal_hue is ignored, `use_marg_hue` is False.
    # This means `element='bars'` and `fill=True` (default for histplot).
    # So, the `Lines` object should be empty.
    assert len(g.ax_marg_x.lines) == 0
    assert len(g.ax_marg_y.lines) == 0
    plt.close(g.fig)


def test_qc_joint_custom_main_plot_function(qc_adata):
    """
    Test that a custom main_plot_function (e.g., sns.kdeplot) can be used.
    """
    g = qc_joint(qc_adata, x="n_genes", y="n_counts", main_plot_function=sns.kdeplot)
    assert isinstance(g, sns.JointGrid)
    # A basic check: ensure there are some artists in the joint plot.
    assert len(g.ax_joint.collections) > 0 or len(g.ax_joint.patches) > 0 or len(g.ax_joint.lines) > 0
    plt.close(g.fig)


def test_qc_joint_custom_marginal_kwargs(qc_adata):
    """
    Test that custom marginal_kwargs are passed to marginal plots.
    """
    custom_marg_kwargs = dict(bins=50, kde=True, color="red")
    g = qc_joint(qc_adata, x="n_genes", y="n_counts", marginal_kwargs=custom_marg_kwargs)
    assert isinstance(g, sns.JointGrid)
    # A basic check: ensure there are some artists in the marginal plots.
    assert len(g.ax_marg_x.collections) > 0 or len(g.ax_marg_x.patches) > 0 or len(g.ax_marg_x.lines) > 0
    assert len(g.ax_marg_y.collections) > 0 or len(g.ax_marg_y.patches) > 0 or len(g.ax_marg_y.lines) > 0
    plt.close(g.fig)


def test_qc_joint_kwargs_passed_to_main_plot_function(qc_adata):
    """
    Test that additional kwargs are passed to the main plot function (e.g., scatterplot 's' for size).
    """
    g = qc_joint(
        qc_adata,
        x="n_genes",
        y="n_counts",
        s=50,
        alpha=0.5,
    )
    assert isinstance(g, sns.JointGrid)

    # Check if scatterplot points have the specified size and alpha.
    # The scatterplot points are PathCollection objects.
    path_collections = [artist for artist in g.ax_joint.collections if isinstance(artist, plt.cm.ScalarMappable)]
    assert len(path_collections) > 0

    assert np.allclose(path_collections[0].get_sizes(), 50)
    assert np.isclose(path_collections[0].get_alpha(), 0.5)

    plt.close(g.fig)


def test_qc_joint_title(qc_adata):
    """
    Test that the title is correctly set.
    """
    test_title = "My Custom QC Plot"
    g = qc_joint(qc_adata, x="n_genes", y="n_counts", title=test_title)
    assert isinstance(g, sns.JointGrid)
    assert g.fig._suptitle.get_text() == test_title
    plt.close(g.fig)


def get_thresholds(g):
    """Extract vertical and horizontal line positions from a JointGrid."""
    x_lines, y_lines = [], []

    for ax in (g.ax_joint, g.ax_marg_x):
        for line in ax.get_lines():
            x = line.get_xdata()
            if np.allclose(x, x[0]):
                x_lines.append(x[0])

    for ax in (g.ax_joint, g.ax_marg_y):
        for line in ax.get_lines():
            y = line.get_ydata()
            if np.allclose(y, y[0]):
                y_lines.append(y[0])

    return {"x": x_lines, "y": y_lines}


@pytest.mark.parametrize(
    "x_thresh,y_thresh,log_base_x,log_base_y,expect_log",
    [
        ((1000, 4000), (2000, 8000), None, None, False),
        ((10, 1000), (50, 5000), 10, 2, True),
    ],
)
def test_qc_joint_thresholds(qc_adata, x_thresh, y_thresh, log_base_x, log_base_y, expect_log):
    """Test that thresholds are drawn correctly, with/without log transformation."""
    kwargs = dict(
        x="n_genes",
        y="n_counts",
        x_threshold=x_thresh,
        y_threshold=y_thresh,
    )
    if expect_log:
        kwargs.update(dict(log_x=log_base_x, log_y=log_base_y, return_df=True))
        g, df_out = qc_joint(qc_adata, **kwargs)

        # axis labels reflect log transform
        assert g.ax_joint.get_xlabel() == f"log{log_base_x} n_genes"
        assert g.ax_joint.get_ylabel() == f"log{log_base_y} n_counts"

        # new log-transformed columns exist
        assert f"log{log_base_x} n_genes" in df_out.columns
        assert f"log{log_base_y} n_counts" in df_out.columns

        # expected thresholds after log transform
        def log1p_base(x, base):
            return np.log1p(x) / np.log(base)

        expected_x = [log1p_base(v, log_base_x) for v in x_thresh]
        expected_y = [log1p_base(v, log_base_y) for v in y_thresh]
    else:
        g = qc_joint(qc_adata, **kwargs)
        expected_x, expected_y = x_thresh, y_thresh

    # check thresholds drawn on the plot
    found = get_thresholds(g)

    assert np.allclose(sorted(set(found["x"])), sorted(expected_x))
    assert np.allclose(sorted(set(found["y"])), sorted(expected_y))

    plt.close(g.fig)
