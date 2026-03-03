"""
vizflow — Visualization tools for flow matching on single-cell / flow cytometry data.

Public API
----------
simulate_trajectories   : Run a trained flow matching model forward through time.
plot_umap_trajectories  : Show how the cell population shifts in UMAP space over time.
plot_marker_expression  : Show how every marker's distribution evolves over time.
"""

from vizflow.simulate import simulate_trajectories
from vizflow.visualize import plot_umap_trajectories, plot_marker_expression

__all__ = [
    "simulate_trajectories",
    "plot_umap_trajectories",
    "plot_marker_expression",
]
