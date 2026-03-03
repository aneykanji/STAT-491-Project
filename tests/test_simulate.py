"""Tests for vizflow.simulate"""

import numpy as np
import pytest
import torch
from torch import nn


class _DummyModel(nn.Module):
    """Tiny MLP that mimics the (x_t, t) → velocity interface."""

    def __init__(self, dim: int = 4):
        super().__init__()
        self.fc = nn.Linear(dim + 1, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_simulate_trajectories_shape():
    from vizflow.simulate import simulate_trajectories

    dim = 4
    n_cells = 20
    n_timepoints = 10

    model = _DummyModel(dim)
    x0 = np.random.randn(n_cells, dim).astype(np.float32)

    traj, t_span = simulate_trajectories(
        model, x0, n_timepoints=n_timepoints, t_start=0.0, t_end=2.0, device="cpu"
    )

    assert traj.shape == (n_timepoints, n_cells, dim), (
        f"Expected shape ({n_timepoints}, {n_cells}, {dim}), got {traj.shape}"
    )
    assert t_span.shape == (n_timepoints,)
    assert float(t_span[0]) == pytest.approx(0.0)
    assert float(t_span[-1]) == pytest.approx(2.0)


def test_simulate_trajectories_min_timepoints():
    from vizflow.simulate import simulate_trajectories

    model = _DummyModel(4)
    x0 = np.random.randn(5, 4).astype(np.float32)

    with pytest.raises(ValueError, match="n_timepoints must be >= 10"):
        simulate_trajectories(model, x0, n_timepoints=5, device="cpu")
