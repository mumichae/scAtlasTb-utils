import numpy as np
import seaborn as sns
from anndata import AnnData


def qc_joint(
    adata: AnnData,
    x: str,
    y: str,
    log_x: int = 1,
    log_y: int = 1,
    hue: str = None,
    main_plot_function=None,
    marginal_hue=None,
    x_threshold=None,
    y_threshold=None,
    title="",
    return_df=False,
    marginal_kwargs: dict = None,
    **kwargs,
):
    """Plot scatter plot with marginal histograms from df columns.

    Parameters
    ----------
    adata
        AnnData object containing axes to be plotted in joint plot.
    x
        Column in `adata.obs` for x axis.
    y
        Column in `adata.obs` for y axis.
    log_x
        Log base for transforming x axis before plotting. Default 1, no transformation.
    log_y
        Log base for transforming y axis before plotting. Default 1, no transformation.
    hue
        Column in `adata.obs` with annotations for color coding scatter plot points.
    main_plot_function
        Function to use for the main joint plot. Defaults to `seaborn.scatterplot`.
    marginal_hue
        Column in `adata.obs` with annotations for color coding marginal plot distributions.
    x_threshold
        Tuple of upper and lower filter thresholds for x axis.
    y_threshold
        Tuple of upper and lower filter thresholds for y axis.
    title
        Title text for plot.
    return_df
        If `True`, return the DataFrame used for plotting along with the JointGrid.
    marginal_kwargs
        Additional keyword arguments passed to the marginal plots (histograms/KDEs).
    **kwargs
        Additional keyword arguments passed to the main plot function.

    Returns
    -------
    g
        A `seaborn.axisgrid.JointGrid` object.
    out_df
        A `pandas.DataFrame` with updated values. Only returned if `return_df` is `True`.
    """
    columns = [col for col in adata.obs.columns if col in [x, y, hue, marginal_hue]]
    df = adata.obs[columns].copy()

    if main_plot_function is None:
        main_plot_function = sns.scatterplot
    if not x_threshold:
        x_threshold = (0, np.inf)
    if not y_threshold:
        y_threshold = (0, np.inf)

    def log1p_base(_x, base):
        return np.log1p(_x) / np.log(base)

    if log_x > 1:
        x_log = f"log{log_x} {x}"
        df[x_log] = log1p_base(df[x], log_x)
        x_threshold = log1p_base(x_threshold, log_x)
        x = x_log

    if log_y > 1:
        y_log = f"log{log_y} {y}"
        df[y_log] = log1p_base(df[y], log_y)
        y_threshold = log1p_base(y_threshold, log_y)
        y = y_log

    marginal_kwargs_defaults = dict(fill=False, bins=100, legend=False)
    marginal_kwargs = (marginal_kwargs or {}) | marginal_kwargs_defaults

    if marginal_hue in df.columns:
        marginal_hue = None if df[marginal_hue].nunique() > 100 else marginal_hue
    use_marg_hue = marginal_hue is not None

    if not use_marg_hue:
        marginal_kwargs.pop("palette", None)

    g = sns.JointGrid(
        data=df,
        x=x,
        y=y,
        xlim=(0, df[x].max()),
        ylim=(0, df[y].max()),
    )

    # main plot
    g.plot_joint(
        main_plot_function,
        data=df.sample(frac=1),
        hue=hue,
        **kwargs,
    )

    # marginal hist plot
    g.plot_marginals(
        sns.histplot,
        data=df,
        hue=marginal_hue,
        element="step" if use_marg_hue else "bars",
        **marginal_kwargs,
    )

    g.fig.suptitle(title, fontsize=12)
    # workaround for patchworklib
    g._figsize = g.fig.get_size_inches()

    # handles, labels = g.ax_joint.get_legend_handles_labels()
    markerscale = (80 / kwargs.get("s", 20)) ** 0.5
    g.ax_joint.legend(markerscale=markerscale)

    # x threshold
    for t, t_def in zip(x_threshold, (0, np.inf), strict=False):
        if t != t_def:
            g.ax_joint.axvline(x=t, color="red")
            g.ax_marg_x.axvline(x=t, color="red")

    # y threshold
    for t, t_def in zip(y_threshold, (0, np.inf), strict=False):
        if t != t_def:
            g.ax_joint.axhline(y=t, color="red")
            g.ax_marg_y.axhline(y=t, color="red")

    if return_df:
        return g, df
    return g
