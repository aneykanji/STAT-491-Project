"""
simulate.py
-----------
Function 1: simulate_trajectories

Given a trained flow matching model (an MLP that takes [x_t, t] and predicts
velocity), integrate the ODE forward through time to produce cell trajectories.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper


def simulate_trajectories(
    model: nn.Module,
    initial_cells: np.ndarray,
    n_timepoints: int = 10,
    t_start: float = 0.0,
    t_end: float = 2.0,
    device: str | None = None,
    solver: str = "dopri5",
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate cell trajectories using a trained flow matching model.

    Integrates the learned ODE  dX/dt = model(X_t, t)  forward from *t_start*
    to *t_end* at *n_timepoints* evenly-spaced time steps.

    Parameters
    ----------
    model : torch.nn.Module
        Trained MLP that accepts a concatenated (x_t, t) tensor and returns
        the velocity.  Must already be on the target device.
    initial_cells : np.ndarray, shape (n_cells, n_markers)
        Starting cell states, typically taken from the earliest observed time
        point (e.g. time point A).
    n_timepoints : int, default 10
        Number of time points at which to record the trajectory.  Must be
        >= 10 per the project specification.
    t_start : float, default 0.0
        Start of the integration interval.
    t_end : float, default 2.0
        End of the integration interval.  For a dataset with 3 time points
        (A=0, B=1, C=2) this covers the full range.
    device : str or None, default None
        Torch device string (e.g. ``"cuda"`` or ``"cpu"``).  If *None*,
        uses CUDA when available, otherwise CPU.
    solver : str, default "dopri5"
        ODE solver passed to torchdyn.  ``"dopri5"`` (Dormand-Prince 4/5) is
        a good default; use ``"euler"`` for speed during debugging.

    Returns
    -------
    traj : np.ndarray, shape (n_timepoints, n_cells, n_markers)
        Simulated cell states at each recorded time point.
    t_span : np.ndarray, shape (n_timepoints,)
        The time values corresponding to each slice of *traj*.

    Examples
    --------
    >>> traj, t_span = simulate_trajectories(trained_model, X[0], n_timepoints=10)
    >>> traj.shape   # (10, n_cells, n_markers)
    """
    if n_timepoints < 10:
        raise ValueError(
            f"n_timepoints must be >= 10 (got {n_timepoints})."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    node = NeuralODE(
        torch_wrapper(model),
        solver=solver,
        sensitivity="adjoint",
    )

    x0 = torch.from_numpy(initial_cells).float().to(device)
    t_span_torch = torch.linspace(t_start, t_end, n_timepoints)

    with torch.no_grad():
        traj_tensor = node.trajectory(x0, t_span=t_span_torch)

    traj = traj_tensor.cpu().numpy()           # (n_timepoints, n_cells, n_markers)
    t_span_np = t_span_torch.numpy()

    return traj, t_span_np
