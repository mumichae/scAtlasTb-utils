from __future__ import annotations

import logging
import traceback
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
from scatlastb_utils.utils import dask_compute, parse_gene_names, remove_outliers


def _add_group_sizes_to_legend(legend, adata, color, label_formatter=None):
    """Append group sizes to legend labels for categorical columns."""
    category_counts_str = {
        str(category): int(count) for category, count in adata.obs[color].value_counts(dropna=False).items()
    }
    for text in legend.get_texts():
        label = text.get_text()
        count = category_counts_str.get(label)
        if count is not None:
            prefix = label if label_formatter is None else label_formatter(label)
            text.set_text(f"{prefix} (n={count})")


def _plot_centroids_on_embedding(
    ax, adata, color, basis, legend, category_numbers, legend_fontsize=10, outline_thickness=2
):
    """
    Plot category numbers at centroid positions on embedding.

    Args:
        ax: Matplotlib axes object
        adata: AnnData object
        color: Column name in adata.obs for grouping
        basis: Key in adata.obsm for coordinates
        legend: Matplotlib legend object
        category_numbers: Dict mapping categories to their numbers
        legend_fontsize: Font size for category labels
    """
    categories = [cat for cat in adata.obs[color].cat.categories if cat in adata.obs[color].unique()]

    # Compute centroids for each category
    coords = adata.obsm[basis][:, :2]
    centroids = (
        pd.DataFrame(coords, index=adata.obs[color])
        .groupby(level=0, dropna=False, observed=True)
        .median()
        .reindex(categories)
    )

    # Extract colors from legend handles
    color_map = {
        text.get_text(): handle.get_facecolor()[0]
        for handle, text in zip(legend.legend_handles, legend.get_texts(), strict=False)
        if hasattr(handle, "get_facecolor")
    }

    # Plot category numbers at centroids with matching colors
    for cat, row in centroids.iterrows():
        bg_color = color_map.get(cat, "white")

        # Convert color to RGB for luminance calculation
        try:
            r, g, b = mpl.colors.to_rgba(bg_color)[:3]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.4 else "black"
        except (ValueError, TypeError):
            text_color = "white"

        label_value = category_numbers[cat]
        ax.text(
            row.iloc[0],
            row.iloc[1],
            s=str(label_value),
            fontsize=legend_fontsize,
            fontweight="bold",
            ha="center",
            va="center",
            color=text_color,
            bbox=dict(
                boxstyle="circle",
                facecolor="none",
                alpha=0.4,
                edgecolor="none",
            ),
        ).set_path_effects(
            [mpl.patheffects.Stroke(linewidth=1.5, foreground=color_map.get(cat, "white")), mpl.patheffects.Normal()]
        )

    # Add category numbers and group sizes to legend labels.
    category_numbers_str = {str(k): v for k, v in category_numbers.items()}

    def formatter(label):
        mapped = category_numbers_str.get(label)
        return f"{mapped}: {label}" if isinstance(mapped, int) else label

    _add_group_sizes_to_legend(legend=legend, adata=adata, color=color, label_formatter=formatter)


def _plot_single_color(
    adata,
    color,
    basis,
    n_cells=None,
    plot_centroids=False,
    verbose=True,
    file_name=None,
    title="",
    output_dir=None,
    max_label_length=10,
    dpi=200,
    figsize=(6, 6),
    outline_thickness=2,
    **kwargs,
):
    """Plot a single color (or list of gene colors) and optionally save to disk."""
    if n_cells is None:
        n_cells = adata.n_obs

    palette = None
    if file_name is None:
        file_name = str(color)
    colors = color if isinstance(color, list) else [color]

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
    mpl.rcParams["figure.constrained_layout.use"] = True

    # Select palette according to obs column type and cardinality
    for col in colors:
        if col in adata.obs.columns:
            color_vec = adata.obs[col]
            if is_categorical_dtype(color_vec):
                if color_vec.nunique() > 102:
                    palette = "turbo"
                elif color_vec.nunique() > 20:
                    palette = sc.pl.palettes.godsnot_102
            elif is_numeric_dtype(color_vec):
                palette = "coolwarm" if color_vec.min() < 0 else "plasma"

    # Centroid plotting setup (sctk-like numbering)
    color = colors[0] if colors else None
    plot_centroids = (
        plot_centroids
        and len(colors) == 1
        and color in adata.obs.columns
        and is_categorical_dtype(adata.obs[color])
        and adata.obs[color].nunique() <= 102
    )

    try:
        fig = sc.pl.embedding(
            adata,
            basis=basis,
            color=colors,
            show=False,
            return_fig=True,
            palette=palette,
            **kwargs,
        )
        fig.suptitle(f"{title}\nn={n_cells}", fontsize=12)

        ax = fig.get_axes()[0]
        legend = ax.get_legend()

        if palette == "turbo":
            if legend:
                legend.remove()
                legend = None

        elif legend and plot_centroids:
            categories = [cat for cat in adata.obs[color].cat.categories if cat in adata.obs[color].unique()]
            category_numbers = {
                cat: idx + 1 if len(str(cat)) > max_label_length else cat for idx, cat in enumerate(categories)
            }
            _plot_centroids_on_embedding(
                ax=ax,
                adata=adata,
                color=color,
                basis=basis,
                legend=legend,
                category_numbers=category_numbers,
                legend_fontsize=kwargs.get("legend_fontsize", 10),
            )

        elif legend and len(colors) == 1 and is_categorical_dtype(adata.obs[color]):
            _add_group_sizes_to_legend(legend=legend, adata=adata, color=color)

        if legend:
            legend_bbox = legend.get_window_extent()
            fig_width, fig_height = fig.get_size_inches()
            fig.set_size_inches((fig_width + legend_bbox.width / fig.dpi, fig_height))

        if verbose:
            logging.info(f'Plotting color "{file_name}" successful.')

    except (ValueError, RuntimeError) as e:
        traceback.print_exc()
        logging.error(f'Failed to plot "{file_name}": {e}')
        plt.plot([])
    except Exception:
        raise

    out_path = output_dir / f"{file_name}.png" if output_dir is not None else None
    if out_path is not None:
        try:
            plt.savefig(out_path, bbox_inches="tight")
        except (OSError, ValueError, RuntimeError) as e:
            logging.error(f'Failed to save plot "{file_name}" to {out_path}: {e}')
            traceback.print_exc()
        except Exception:
            raise
    else:
        plt.show()
    plt.close("all")


def embedding(
    adata,
    basis: str = "X_umap",
    color: str | list | None = None,
    plot_centroids: list | None = None,
    min_cells_per_category: float = 1e-4,
    outlier_factor: float = 0,
    gene_chunk_size: int = 10,
    output_dir: Path | None = None,
    title: str = "",
    dpi: int = 200,
    n_jobs: int = 1,
    figsize: tuple = (6, 6),
    downsample: float | int | None = None,
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
    dpi
        Resolution used for both rendering and saving figures.
    n_jobs
        Number of parallel threads for figure generation.
    figsize
        Tuple specifying figure size in inches, e.g. ``(8, 4)`` for wide aspect.
    downsample
        If float in (0, 1], randomly subsample that fraction of cells before plotting.
        If int > 1, randomly subsample up to that many cells. Default: None (no downsampling).
    **kwargs
        Additional keyword arguments forwarded to ``_plot_single_color`` and
        ultimately to ``sc.pl.embedding`` (e.g. ``legend_fontsize``, ``ncols``).
    """
    # Find a categorical color column for stratification
    stratify_col = None
    if color is not None:
        color_list = color if isinstance(color, list) else [color]
        for col in color_list:
            if col in adata.obs.columns and is_categorical_dtype(adata.obs[col]):
                stratify_col = col
                break

    n_cells = adata.n_obs  # Preserve original number of cells before downsampling
    if downsample is not None:
        if isinstance(downsample, float) and 0 < downsample < 1:
            adata = sample(adata, frac=downsample, stratify=stratify_col)
        elif isinstance(downsample, int) and downsample > 1 and adata.n_obs > downsample:
            adata = sample(adata, n=downsample, stratify=stratify_col)
        # else: ignore if not valid
    plot_centroids = list(plot_centroids) if plot_centroids else []
    obs_columns = list(adata.obs.columns)

    # Resolve min_cells_per_category threshold
    if min_cells_per_category < 1:
        min_cells_per_category *= n_cells
    logging.info(f"Remove categories with fewer than {min_cells_per_category:.1f} cells")

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
    colors = [c for c in colors if c in obs_columns and adata.obs[c].nunique() > 1]
    logging.info(f"Colors from obs after filtering:\n{pformat(colors)}")

    # Clean categorical columns (normalise NaN strings, drop rare categories)
    for col in colors:
        column = adata.obs[col]
        if is_categorical_dtype(column) or is_string_dtype(column):
            column = (
                column.astype(object).replace(["NaN", "None", "", "nan", "unknown"], float("nan")).astype("category")
            )
            value_counts = column.value_counts()
            rare = value_counts[value_counts <= min_cells_per_category].index
            adata.obs[col] = column.cat.remove_categories(rare)

    if not colors:
        logging.info("No valid colors, skip...")
        colors = [None]

    # Remove embedding outliers
    logging.info("Remove outliers...")
    adata = remove_outliers(adata, "max", factor=outlier_factor, rep=basis)
    adata = remove_outliers(adata, "min", factor=outlier_factor, rep=basis)

    # Subset to requested genes, or drop X/var if no genes needed
    if gene_colors:
        logging.info(f"Subset to {len(gene_colors)} requested genes...")
        adata = adata[:, adata.var_names.isin(gene_colors)].copy()
        logging.info(str(adata))
        adata = dask_compute(adata, layers="X")
        logging.info(str(adata.var))
    else:
        del adata.X
        del adata.var

    # Subsample very large datasets
    logging.info("Shuffle cells...")
    if n_cells > 1e6:
        adata = adata[adata.obs.sample(frac=0.7).index]

    if adata.is_view:
        logging.info("Convert view to copy...")
        adata = adata.copy()

    # Compute point size (clamp between default and 200)
    default_size = max(1, 200_000 / adata.n_obs)
    size = kwargs.pop("size", default_size) or default_size
    kwargs["size"] = min(200, max(default_size, size))

    logging.info("Parameters:\n" + pformat(kwargs))

    # Plot obs colors in parallel
    list(
        tqdm(
            Parallel(return_as="generator", backend="threading", n_jobs=n_jobs)(
                delayed(_plot_single_color)(
                    adata=adata,
                    color=col,
                    basis=basis,
                    n_cells=n_cells,
                    plot_centroids=col in plot_centroids,
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
                    delayed(_plot_single_color)(
                        adata=adata,
                        color=group_color,
                        basis=basis,
                        n_cells=n_cells,
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
