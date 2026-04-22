from __future__ import annotations

import logging
import traceback
from collections.abc import Iterable
from pathlib import Path
from pprint import pformat

import matplotlib as mpl
import matplotlib.patheffects
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype
from tqdm import tqdm

from scatlastb_utils.pp.sample import sample
from scatlastb_utils.utils import _sanitize_default_file_name, dask_compute, parse_gene_names, remove_outliers


def _format_legend_labels(legend, obs, color, annotate_legend=True, category_index_map=None, bold_labels=None):
    """
    Unified handler for legend styling.

    Appends counts, applies SCTK-style numbering, and bolds specific labels.
    """
    if bold_labels is None:
        bold_labels = []
    if category_index_map is None:
        category_index_map = {}

    # Get group sizes
    if annotate_legend:
        counts = obs[color].value_counts(dropna=False)
        category_counts_str = {str(k): int(v) for k, v in counts.items()}
    else:
        category_counts_str = {}

    for text in legend.get_texts():
        label = text.get_text()
        # category numbers (SCTK-style)
        if category_index_map and label in category_index_map:
            display = f"{category_index_map[label]}: {label}"
        else:
            display = label
        # append counts to legend labels
        count = category_counts_str.get(label)
        text.set_text(f"{display} (n={count})" if count is not None else display)
        # set font weight for bold labels
        if label in bold_labels:
            text.set_fontweight("bold")
        else:
            text.set_fontweight("normal")


def _plot_centroids_on_embedding(
    ax, adata, color, basis, legend, category_index_map, legend_fontsize=10, bold_labels=None
):
    """Plot category numbers at centroid positions on the embedding."""
    # Only compute centroids for categories actually present in the current subset
    categories = [cat for cat in adata.obs[color].cat.categories if cat in adata.obs[color].unique()]

    # Compute centroids (median coordinates) for each category
    coords = adata.obsm[basis][:, :2]
    centroids = (
        pd.DataFrame(coords, index=adata.obs[color])
        .groupby(level=0, dropna=False, observed=True)
        .median()
        .reindex(categories)
    )

    # Extract colors from legend handles to match circles to categories
    color_map = {
        text.get_text(): handle.get_facecolor()[0]
        for handle, text in zip(legend.legend_handles, legend.get_texts(), strict=False)
        if hasattr(handle, "get_facecolor")
    }

    if bold_labels is None:
        bold_labels = []

    # Plot category numbers at centroids
    for cat, row in centroids.iterrows():
        bg_color = color_map.get(cat, "white")

        # Automatic text contrast: determine if white or black text is more readable
        try:
            r, g, b = mpl.colors.to_rgba(bg_color)[:3]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.4 else "black"
        except (ValueError, TypeError):
            text_color = "white"

        # Determine label for centroid
        label = category_index_map.get(cat, cat)
        ax.text(
            row.iloc[0],
            row.iloc[1],
            s=str(label),
            fontsize=legend_fontsize + 2 if cat in bold_labels else legend_fontsize,
            fontweight="bold" if cat in bold_labels else "normal",
            ha="center",
            va="center",
            color=text_color,
            bbox=dict(
                boxstyle="circle",
                facecolor="none",
                alpha=0.4,
                edgecolor="none",
            ),
        ).set_path_effects([mpl.patheffects.Stroke(linewidth=2, foreground=bg_color), mpl.patheffects.Normal()])


def _estimate_legend_width(categories, fontsize):
    if len(categories) == 0:
        return 0

    # from scanpy hardcoded defaults
    ncol = 1 if len(categories) <= 14 else 2 if len(categories) <= 30 else 3

    longest_cat = max(categories, key=len)
    # estimate text width
    avg_char_width_pts = fontsize * 0.6
    max_text_width_pts = len(longest_cat) * avg_char_width_pts
    # add handle (marker) and internal padding (in points)
    # handlelength (2.0) + handletextpad (0.8) = 2.8
    column_width_pts = max_text_width_pts + (2.8 * fontsize)
    total_width_pts = (column_width_pts * ncol) + (2.0 * fontsize * (ncol - 1))

    # Convert points to inches (72 pts = 1 inch)
    return (total_width_pts / 72) + 0.1


def _plot_color_axis(
    adata,
    color,
    basis,
    obs=None,
    annotate_legend=True,
    plot_centroids=False,
    centroid_label_bold=False,
    verbose=True,
    file_name=None,
    title="",
    output_dir=None,
    max_label_length=10,
    dpi=200,
    figsize=(6, 6),
    outline_thickness=2,
    bold_labels=None,
    warn_on_drop=True,
    **kwargs,
):
    """Plot a single color (or list of gene colors) and optionally save to disk."""
    if obs is None:
        obs = adata.obs.copy()

    palette = None
    if file_name is None:
        file_name = _sanitize_default_file_name(color)
    colors = color if isinstance(color, list) else [color]
    if bold_labels is None:
        bold_labels = []

    fig_params = dict(
        frameon=False,
        vector_friendly=True,
        fontsize=9,
        figsize=figsize,
        dpi=dpi,
        dpi_save=dpi,
        format="png",
    )

    if len(colors) > 4:
        fig_params = dict(frameon=False)
    sc.set_figure_params(**fig_params)

    # Select palette according to the first matching obs column's type and cardinality
    color = colors[0] if colors else None
    palette = None
    categorical_legend = False
    if len(colors) == 1 and color in adata.obs.columns:
        color_vec = adata.obs[color]

        if is_numeric_dtype(color_vec):
            palette = "coolwarm" if color_vec.min() < 0 else "plasma"

        elif is_categorical_dtype(color_vec):
            categorical_legend = True
            ncat = color_vec.nunique()
            if ncat > 102:
                if warn_on_drop:
                    logging.warning(
                        "Color '%s' has more than 102 categories, using 'turbo' palette and no legend", color
                    )
                palette = "turbo"
                categorical_legend = False
            elif ncat > 20:
                palette = sc.pl.palettes.godsnot_102

    legend_fontsize = kwargs.get("legend_fontsize", 10)
    legend_width = 0
    if categorical_legend:
        legend_width = _estimate_legend_width(color_vec.cat.categories, legend_fontsize)

    total_width = figsize[0] + legend_width
    fig = plt.figure(figsize=(total_width, figsize[1]), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[figsize[0], legend_width], wspace=0.1)
    ax = fig.add_subplot(gs[0, 0])

    try:
        sc.pl.embedding(
            adata,
            basis=basis,
            color=colors,
            show=False,
            ax=ax,
            palette=palette,
            legend_loc="right margin" if categorical_legend else None,
            **kwargs,
        )

        if categorical_legend:
            legend = ax.get_legend()
            category_index_map = None

            if plot_centroids:
                categories = [cat for cat in adata.obs[color].cat.categories if cat in adata.obs[color].unique()]
                category_index_map = {
                    cat: idx + 1 for idx, cat in enumerate(categories) if len(str(cat)) > max_label_length
                }
                _plot_centroids_on_embedding(
                    ax=ax,
                    adata=adata,
                    color=color,
                    basis=basis,
                    legend=legend,
                    category_index_map=category_index_map,
                    legend_fontsize=legend_fontsize,
                    bold_labels=bold_labels if centroid_label_bold else [],
                )

            _format_legend_labels(
                legend=legend,
                obs=obs,
                color=color,
                annotate_legend=annotate_legend,
                category_index_map=category_index_map,
                bold_labels=bold_labels,
            )

        # adjust figure layout to accommodate legend and title
        plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.1)
        ax.set_box_aspect(figsize[1] / figsize[0])
        fig.suptitle(f"{title}\nn={obs.shape[0]}", fontsize=12)

        if verbose:
            logging.info(f'Plotting color "{file_name}" successful.')

    except (ValueError, RuntimeError) as e:
        traceback.print_exc()
        logging.error(f'Failed to plot "{file_name}": {e}')
        fig = plt.figure()
    except Exception:
        raise

    if output_dir is not None:
        out_path = Path(output_dir) / f"{file_name}.png"
        try:
            fig.savefig(out_path, bbox_inches="tight")
        except (OSError, ValueError, RuntimeError) as e:
            logging.error(f'Failed to save plot "{file_name}" to {out_path}: {e}')
            traceback.print_exc()
        except Exception:
            raise
    else:
        plt.show(fig)
    plt.close(fig)


def embedding(
    adata,
    basis: str = "X_umap",
    color: str | list | None = None,
    plot_centroids: list | None = None,
    bold_labels: list | None = None,
    category_order: object = None,
    na_strings: str = ["NaN", "None", "", "nan", "unknown"],
    min_cells_per_category: float = 0,
    outlier_factor: float = 0,
    gene_chunk_size: int = 10,
    output_dir: Path | str | None = None,
    title: str = "",
    annotate_legend: bool = True,
    dpi: int = 200,
    n_jobs: int = 1,
    figsize: tuple = (6, 6),
    downsample: float | int | None = None,
    warn_on_drop: bool = True,
    inplace: bool = False,
    **kwargs,
):
    """
    Plot a scanpy embedding for one or more colors with preprocessing and post-processing.

    Wraps ``sc.pl.embedding`` with:
    - categorical column cleaning (NaN normalisation, rare-category removal)
    - embedding outlier removal
    - automatic palette selection
    - centroid labels with sctk-style numbering
    - group-size annotations in the legend
    - gene-panel chunking
    - parallel saving of per-color PNG files

    Parameters
    ----------
    adata
        Annotated data matrix.
    basis
        Key in ``adata.obsm`` for the embedding coordinates (e.g. ``"X_umap"``).
    color
        One or more ``adata.obs`` column names or gene names to color by.
    plot_centroids
        Subset of ``color`` values for which centroid labels are drawn on the
        embedding (sctk-style numbered circles).
    bold_labels
        List of category names to appear in bold in the legend.
    category_order
        Any iterable of strings for global order, or dict mapping {col_name: iterable of order}.
    na_strings
        For categorical or str columns, which strings to convert to NaN
    min_cells_per_category
        Minimum number of cells required per category.  Values in ``[0, 1)``
        are interpreted as a fraction of the total cell count.
    outlier_factor
        Cells whose embedding coordinate deviates more than this factor from
        the mean absolute value are removed.  Set to ``0`` to disable.
    gene_chunk_size
        Maximum number of genes rendered in a single gene-panel figure.
    output_dir
        Directory where output PNG files are written.  Set to ``None``
        (default) to display figures interactively instead of saving.
    title
        Prefix added to the figure suptitle (combined with the cell count).
    annotate_legend
        Whether to append category counts to legend labels (e.g. "CD4+ T (n=123)").
    dpi
        Resolution used for both rendering and saving figures.
    n_jobs
        Number of parallel threads for figure generation.
    figsize
        Tuple specifying figure size in inches, e.g. ``(8, 4)`` for wide aspect.
    downsample
        If float in (0, 1], randomly subsample that fraction of cells before plotting.
        If int > 1, randomly subsample up to that many cells. Default: None (no downsampling).
    warn_on_drop
        If True, log warnings when colors are dropped due to invalidity or low category counts.
    inplace
        If True, remove slots from adata inplace. This can be useful for scripts where the
        adata is not used afterwards, and memory footprint should be minimised to only what
        is essential for plotting.
    **kwargs
        Additional keyword arguments forwarded to ``_plot_color_axis`` and
        ultimately to ``sc.pl.embedding`` (e.g. ``legend_fontsize``, ``ncols``).
    """
    plot_centroids = list(plot_centroids) if plot_centroids else []
    obs_columns = list(adata.obs.columns)

    # Parse color list and ensure centroid colors are included
    colors: list = color if isinstance(color, list) else ([color] if color is not None else [])
    colors += [c for c in plot_centroids if c not in colors]
    colors = list(dict.fromkeys(colors))
    logging.info(f"Configured colors:\n{pformat(colors)}")

    # Separate gene colors from obs colors
    gene_colors = [c for c in colors if c not in obs_columns]
    gene_colors = parse_gene_names(adata, gene_colors)
    gene_colors.sort()

    # Filter obs colors: must exist and have more than one unique value
    dropped_colors = [c for c in colors if c not in obs_columns or adata.obs[c].nunique() <= 1]
    if warn_on_drop and dropped_colors:
        logging.warning(
            f"The following colors were dropped because they are not in obs or have <=1 unique value: {dropped_colors}"
        )
    colors = [c for c in colors if c in obs_columns and adata.obs[c].nunique() > 1]
    logging.info(f"Colors from obs after filtering:\n{pformat(colors)}")

    obs = adata.obs[colors].copy()  # full annotation needed for legend
    if not inplace or adata.is_view:
        logging.info("Convert view to copy...")
        adata = adata.copy()

    # Resolve min_cells_per_category threshold
    if min_cells_per_category < 1:
        min_cells_per_category *= adata.n_obs

    # Prepare each obs color column
    for col in [col for col in colors if is_categorical_dtype(adata.obs[col]) or is_string_dtype(adata.obs[col])]:
        # set NaNs
        column = adata.obs[col].astype(object).replace(na_strings, float("nan")).astype("category")
        # remove rare categories
        value_counts = column.value_counts()
        rare = value_counts[value_counts <= min_cells_per_category].index
        if warn_on_drop and len(rare) > 0:
            logging.warning(f"In color '{col}', the following rare categories were dropped: {list(rare)}")
        adata.obs[col] = column.cat.remove_categories(rare)

        # handle category ordering if specified
        order = None
        if isinstance(category_order, dict):
            order = category_order.get(col)
        elif category_order is not None and not isinstance(category_order, dict):
            # Accept any non-string iterable
            if isinstance(category_order, Iterable) and not isinstance(category_order, (str, bytes)):
                order = list(category_order)
            else:
                order = None
        if order:
            # Filter order to only include categories present in the data to avoid errors
            existing_order = [c for c in order if c in adata.obs[col].cat.categories]
            # Add any categories present in data but missing from order to the end
            missing = [c for c in adata.obs[col].cat.categories if c not in existing_order]
            adata.obs[col] = adata.obs[col].cat.reorder_categories(existing_order + missing)

    if not colors:
        logging.info("No valid colors, skip...")
        colors = [None]

    # Remove embedding outliers
    logging.info("Remove outliers...")
    adata = remove_outliers(adata, "max", factor=outlier_factor, rep=basis, copy=False)
    adata = remove_outliers(adata, "min", factor=outlier_factor, rep=basis, copy=False)

    if downsample is not None:
        # Find a categorical color column for stratification
        stratify_col = None
        for col in colors:
            if col in adata.obs.columns and is_categorical_dtype(adata.obs[col]):
                stratify_col = col
                break
        if isinstance(downsample, float) and 0 < downsample < 1:
            adata = sample(adata, fraction=downsample, stratify=stratify_col, copy=False)
        elif isinstance(downsample, int) and downsample > 1 and adata.n_obs > downsample:
            adata = sample(adata, n=downsample, stratify=stratify_col, copy=False)
        # else: ignore if not valid

    if adata.is_view:
        logging.info("Convert view to copy...")
        adata = adata.copy()

    # Subset to requested genes, or drop X/var if no genes needed
    if gene_colors:
        logging.info(f"Subset to {len(gene_colors)} requested genes...")
        adata = adata[:, adata.var_names.isin(gene_colors)]
        adata = dask_compute(adata, layers="X")
        logging.info(str(adata.var))
    else:
        del adata.X
        del adata.var

    # Compute point size (clamp between default and 200)
    default_size = max(1, 200_000 / adata.n_obs)
    size = kwargs.pop("size", default_size) or default_size
    kwargs["size"] = min(200, max(default_size, size))

    logging.info("Parameters:\n" + pformat(kwargs))

    # Plot obs colors in parallel
    list(
        tqdm(
            Parallel(return_as="generator", backend="threading", n_jobs=n_jobs)(
                delayed(_plot_color_axis)(
                    adata=adata,
                    color=col,
                    basis=basis,
                    obs=obs,
                    annotate_legend=annotate_legend,
                    plot_centroids=col in plot_centroids,
                    bold_labels=bold_labels,
                    title=title,
                    file_name=col,
                    dpi=dpi,
                    output_dir=output_dir,
                    figsize=figsize,
                    **kwargs,
                )
                for col in colors
            ),
            desc="Plotting colors",
            total=len(colors),
            miniters=1,
        )
    )

    # Chunk genes and plot each group in parallel
    gene_groups = {
        f"genes_group={idx}": gene_colors[i : i + gene_chunk_size]
        for idx, i in enumerate(range(0, len(gene_colors), gene_chunk_size))
    }
    if gene_groups:
        kwargs.setdefault("ncols", 4)
        list(
            tqdm(
                Parallel(return_as="generator", backend="threading", n_jobs=n_jobs)(
                    delayed(_plot_color_axis)(
                        adata=adata,
                        color=group_color,
                        basis=basis,
                        obs=obs,
                        annotate_legend=annotate_legend,
                        verbose=False,
                        title=title,
                        file_name=group_title,
                        dpi=dpi,
                        output_dir=output_dir,
                        figsize=figsize,
                        **kwargs,
                    )
                    for group_title, group_color in gene_groups.items()
                ),
                desc="Plotting genes",
                total=len(gene_groups),
                miniters=1,
            )
        )
