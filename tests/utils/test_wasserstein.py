import pytest
import numpy as np
from probisim.bisimdistance import wasserstein_distance

def default_cost(p, q):
    return np.abs(np.subtract.outer(np.arange(len(p)), np.arange(len(q))))

def test_identical_distributions():
    """Test that identical distributions have zero distance."""
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.5, 0.3, 0.2])
    C = default_cost(p, q)
    assert wasserstein_distance(p, q, C) == 0.0

def test_diagonal_distance():
    """Test distance between distributions on diagonal."""
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 0.0, 1.0])
    C = default_cost(p, q)
    assert wasserstein_distance(p, q, C) == 2.0  # Distance of 2 steps

def test_partial_overlap():
    """Test distance between distributions with partial overlap."""
    p = np.array([0.5, 0.5, 0.0])
    q = np.array([0.0, 0.5, 0.5])
    C = default_cost(p, q)
    # Expected: 0.5 * 1 + 0.5 * 1 = 1.0
    assert wasserstein_distance(p, q, C) == 1.0

def test_normalized_distributions():
    """Test that distributions are properly normalized."""
    p = np.array([0.6, 0.4, 0.0])
    q = np.array([0.0, 0.4, 0.6])
    C = default_cost(p, q)
    # Should normalize to [0.6, 0.4, 0.0] and [0.0, 0.4, 0.6]
    assert wasserstein_distance(p, q, C) == 1.2

def test_single_state():
    """Test with single state distributions."""
    p = np.array([1.0])
    q = np.array([1.0])
    C = default_cost(p, q)
    assert wasserstein_distance(p, q, C) == 0.0

def test_empty_distributions():
    """Test with empty distributions."""
    p = np.array([])
    q = np.array([])
    C = np.zeros((0, 0))
    assert wasserstein_distance(p, q, C) == 0.0

def test_invalid_inputs():
    """Test that invalid inputs raise appropriate errors."""
    # Different lengths
    with pytest.raises(ValueError):
        wasserstein_distance(np.array([0.5, 0.5]), np.array([0.3, 0.3, 0.4]), default_cost(np.array([0.5, 0.5]), np.array([0.3, 0.3, 0.4])))
    
    # Negative probabilities
    with pytest.raises(ValueError):
        wasserstein_distance(np.array([-0.5, 1.5]), np.array([0.5, 0.5]), default_cost(np.array([-0.5, 1.5]), np.array([0.5, 0.5])))
    
    # Non-probability distributions
    with pytest.raises(ValueError):
        wasserstein_distance(np.array([0.3, 0.3]), np.array([0.3, 0.3]), default_cost(np.array([0.3, 0.3]), np.array([0.3, 0.3])))

def test_symmetry():
    """Test that distance is symmetric."""
    p = np.array([0.7, 0.3])
    q = np.array([0.3, 0.7])
    C = default_cost(p, q)
    assert wasserstein_distance(p, q, C) == wasserstein_distance(q, p, C)

def test_triangle_inequality():
    """Test triangle inequality property."""
    p = np.array([0.5, 0.5, 0.0])
    q = np.array([0.0, 0.5, 0.5])
    r = np.array([0.0, 0.0, 1.0])
    C = default_cost(p, q)
    d_pq = wasserstein_distance(p, q, C)
    d_qr = wasserstein_distance(q, r, C)
    d_pr = wasserstein_distance(p, r, C)
    assert d_pr <= d_pq + d_qr 