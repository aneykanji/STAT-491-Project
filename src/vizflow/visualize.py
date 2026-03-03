"""
visualize.py
------------
Function 2: plot_umap_trajectories
Function 3: plot_marker_expression
"""

from __future__ import annotations

import warnings
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import umap
from anndata import AnnData


# ---------------------------------------------------------------------------
# Function 2
# ---------------------------------------------------------------------------

def plot_umap_trajectories(
    traj: np.ndarray,
    t_span: np.ndarray,
    adata: AnnData,
    time_col: str = "time",
    n_cells: int = 500,
    n_trajectory_lines: int = 20,
    figsize: tuple[float, float] = (10, 8),
    umap_kwargs: dict | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Show how the cell population shifts in UMAP space over time.

    Fits a UMAP model on the real expression data stored in *adata*, then
    projects the simulated trajectories into that 2-D space and plots them
    on top of the real data background.

    Parameters
    ----------
    traj : np.ndarray, shape (n_timepoints, n_cells, n_markers)
        Output of :func:`simulate_trajectories`.
    t_span : np.ndarray, shape (n_timepoints,)
        Time values corresponding to each slice of *traj* — also the output
        of :func:`simulate_trajectories`.
    adata : AnnData
        Original single-cell dataset.  Must have ``adata.obsm["X_umap"]`` and
        ``adata.obs[time_col]``.
    time_col : str, default "time"
        Column in ``adata.obs`` that holds the time-point label.
    n_cells : int, default 500
        Number of simulated cells to plot (subsampled from *traj*).
    n_trajectory_lines : int, default 20
        Number of individual cell paths to draw as lines.
    figsize : tuple, default (10, 8)
        Matplotlib figure size.
    umap_kwargs : dict or None
        Extra keyword arguments forwarded to ``umap.UMAP``.
    save_path : str or None
        If provided, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    umap_kwargs = umap_kwargs or {}
    n_cells = min(n_cells, traj.shape[1])

    # --- Fit UMAP on real expression data -----------------------------------
    if hasattr(adata.X, "toarray"):
        expr = adata.X.toarray()
    else:
        expr = np.asarray(adata.X)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        umap_model = umap.UMAP(
            n_neighbors=umap_kwargs.pop("n_neighbors", 15),
            n_components=2,
            metric=umap_kwargs.pop("metric", "euclidean"),
            random_state=umap_kwargs.pop("random_state", 42),
            **umap_kwargs,
        )
        umap_model.fit(expr)

    # --- Project simulated trajectories to 2D --------------------------------
    traj_sub = traj[:, :n_cells, :]  # (T, n_cells, markers)
    T, C, M = traj_sub.shape
    traj_flat = traj_sub.reshape(T * C, M)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traj_2d = umap_model.transform(traj_flat).reshape(T, C, 2)

    # --- Plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    time_labels = sorted(adata.obs[time_col].unique())
    colors_real = plt.cm.Set2(np.linspace(0, 1, len(time_labels)))

    for color, label in zip(colors_real, time_labels):
        mask = adata.obs[time_col] == label
        ax.scatter(
            adata.obsm["X_umap"][mask, 0],
            adata.obsm["X_umap"][mask, 1],
            s=2,
            alpha=0.25,
            color=color,
            label=f"Real – {label}",
            rasterized=True,
        )

    # Trajectory scatter (all time points collapsed)
    ax.scatter(
        traj_2d[:, :, 0].ravel(),
        traj_2d[:, :, 1].ravel(),
        s=0.5,
        alpha=0.15,
        color="olive",
        label="Simulated cells",
        rasterized=True,
    )

    # Individual trajectory lines
    n_lines = min(n_trajectory_lines, n_cells)
    for i in range(n_lines):
        ax.plot(
            traj_2d[:, i, 0],
            traj_2d[:, i, 1],
            alpha=0.7,
            color="crimson",
            linewidth=1.2,
        )
    # Add a dummy line for the legend
    ax.plot([], [], color="crimson", linewidth=1.2, label="Trajectory lines")

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title("Simulated trajectories in UMAP space", fontsize=14, fontweight="bold")
    ax.legend(markerscale=4, fontsize=9)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Function 3
# ---------------------------------------------------------------------------

def plot_marker_expression(
    traj: np.ndarray,
    t_span: np.ndarray,
    marker_names: Sequence[str],
    adata: AnnData | None = None,
    time_col: str = "time",
    n_bins: int = 40,
    figsize_per_panel: tuple[float, float] = (4.0, 3.0),
    save_path: str | None = None,
) -> plt.Figure:
    """Show how every marker's expression distribution evolves over simulated time.

    Creates a grid of histograms: **rows = markers**, **columns = time points**.
    If *adata* is provided, the real data distribution (closest observed time
    point) is shown as an orange overlay for reference.

    Parameters
    ----------
    traj : np.ndarray, shape (n_timepoints, n_cells, n_markers)
        Output of :func:`simulate_trajectories`.
    t_span : np.ndarray, shape (n_timepoints,)
        Corresponding time values — also output of :func:`simulate_trajectories`.
    marker_names : sequence of str
        Name of each marker in the same order as the feature dimension of
        *traj* (i.e. ``adata.var_names``).
    adata : AnnData or None
        If provided, real data distributions are overlaid.  Must have
        ``adata.obs[time_col]``.
    time_col : str, default "time"
        Column in ``adata.obs`` with time labels.
    n_bins : int, default 40
        Number of histogram bins.
    figsize_per_panel : tuple, default (4.0, 3.0)
        Size of a single subplot panel (width, height) in inches.
    save_path : str or None
        If provided, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_markers = len(marker_names)
    n_timepoints = len(t_span)

    if traj.shape[2] != n_markers:
        raise ValueError(
            f"traj has {traj.shape[2]} features but {n_markers} marker names were given."
        )

    fig_width = figsize_per_panel[0] * n_timepoints
    fig_height = figsize_per_panel[1] * n_markers
    fig, axes = plt.subplots(
        n_markers,
        n_timepoints,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    # Pre-compute real data matrices keyed by observed time label
    real_by_label: dict[str, np.ndarray] = {}
    obs_time_labels: list[str] = []
    if adata is not None:
        obs_time_labels = sorted(adata.obs[time_col].unique())
        expr = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        for label in obs_time_labels:
            mask = adata.obs[time_col] == label
            real_by_label[label] = expr[mask]

    def _closest_obs_label(t_val: float) -> str | None:
        """Return the observed time label whose index is closest to t_val."""
        if not obs_time_labels:
            return None
        n = len(obs_time_labels)
        boundaries = np.linspace(0, n - 1, n + 1)
        idx = int(np.clip(np.searchsorted(boundaries, t_val) - 1, 0, n - 1))
        return obs_time_labels[idx]

    for row, marker in enumerate(marker_names):
        for col, t_val in enumerate(t_span):
            ax = axes[row][col]
            generated = traj[col, :, row]

            combined_min = generated.min()
            combined_max = generated.max()

            obs_label = _closest_obs_label(t_val)
            if obs_label is not None:
                real = real_by_label[obs_label][:, row]
                combined_min = min(combined_min, real.min())
                combined_max = max(combined_max, real.max())
                bins = np.linspace(combined_min, combined_max, n_bins)
                ax.hist(real, bins=bins, alpha=0.45, color="tab:orange",
                        density=True, label=f"Real ({obs_label})")
            else:
                bins = np.linspace(combined_min, combined_max, n_bins)

            ax.hist(generated, bins=bins, alpha=0.7, color="tab:blue",
                    density=True, label="Simulated")

            # Row label (marker name) on the leftmost column
            if col == 0:
                ax.set_ylabel(marker, fontsize=10, fontweight="bold")

            # Column label (time value) on the top row
            if row == 0:
                ax.set_title(f"t = {t_val:.2f}", fontsize=9)

            if row == 0 and col == n_timepoints - 1:
                ax.legend(fontsize=7, loc="upper right")

            ax.tick_params(labelsize=7)
            ax.set_xlabel("Expression", fontsize=7)

    fig.suptitle(
        "Marker expression distributions across simulated time",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
