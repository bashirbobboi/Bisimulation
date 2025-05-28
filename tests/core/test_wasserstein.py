"""
Unit tests for wasserstein_distance (optimal transport/Wasserstein metric).
Covers identical distributions, simple cases, rectangular cases, and infeasibility.
"""

import numpy as np
import pytest

from probisim.bisimdistance import wasserstein_distance


def test_identical_distributions_zero_cost():
    """Test that identical distributions with zero cost yield zero distance and identity coupling."""
    # Distributions and cost matrix of size 1
    p = np.array([1.0])
    q = np.array([1.0])
    C = np.array([[0.0]])
    dist, coupling = wasserstein_distance(p, q, C)
    # Distance should be zero and coupling should be [[1]]
    assert pytest.approx(dist, rel=1e-6) == 0.0
    assert coupling.shape == (1, 1)
    assert pytest.approx(coupling[0, 0], rel=1e-6) == 1.0


def test_simple_two_point_distribution():
    """Test two-point distributions with unit costs and check coupling properties."""
    # Two-point distributions with unit costs
    p = np.array([0.3, 0.7])
    q = np.array([0.5, 0.5])
    # Cost 0 for matching indices, 1 otherwise
    C = np.array([[0.0, 1.0], [1.0, 0.0]])
    dist, coupling = wasserstein_distance(p, q, C)
    # Expected distance: move 0.2 mass at cost 1 => 0.2
    assert pytest.approx(dist, rel=1e-6) == pytest.approx(0.2, rel=1e-6)
    # Coupling should satisfy row sums = p and col sums = q
    assert coupling.shape == (2, 2)
    assert np.allclose(coupling.sum(axis=1), p, atol=1e-6)
    assert np.allclose(coupling.sum(axis=0), q, atol=1e-6)
    # Check coupling non-negativity
    assert np.all(coupling >= -1e-8)


def test_rectangular_distributions_conservation():
    """Test rectangular distributions and conservation of mass in coupling."""
    # Rectangular case: p length 2, q length 3
    p = np.array([0.5, 0.5])
    q = np.array([1 / 3, 1 / 3, 1 / 3])
    # Cost matrix of increasing costs
    C = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    dist, coupling = wasserstein_distance(p, q, C)
    # Validate distance is non-negative
    assert dist >= 0.0
    # Coupling should be shape (2,3)
    assert coupling.shape == (2, 3)
    # Row sums and column sums conservation
    assert np.allclose(coupling.sum(axis=1), p, atol=1e-6)
    assert np.allclose(coupling.sum(axis=0), q, atol=1e-6)


def test_infeasible_distributions_raise():
    """Test that infeasible distributions (mismatched mass) raise ValueError."""
    # Mismatched total mass should be infeasible
    p = np.array([1.0])
    q = np.array([0.5])
    C = np.array([[0.0, 1.0]])  # Shape 1x2
    with pytest.raises(ValueError):
        wasserstein_distance(p, q, C)


if __name__ == "__main__":
    pytest.main()
