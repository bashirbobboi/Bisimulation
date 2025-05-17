import numpy as np
import pytest
from probisim.bisimdistance import bisimulation_distance_matrix

@pytest.mark.parametrize("T,Term,expected", [
    # Single state, non-terminating
    (np.array([[1.0]]), np.array([0]), np.array([[0.0]])),
    
    # Two identical states
    (np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([0, 0]), np.array([[0.0, 0.0], [0.0, 0.0]])),
    
    # Two different states (one terminating)
    (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1, 0]), np.array([[0.0, 1.0], [1.0, 0.0]])),
    
    # Three states with partial similarity
    (np.array([
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 1.0]
    ]), np.array([0, 0, 1]), np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0]
    ]))
])
def test_simple_distance(T, Term, expected):
    """Test bisimulation distance on simple, hand-computed examples."""
    D = bisimulation_distance_matrix(T, Term, max_iter=10)
    assert np.allclose(D, expected, atol=1e-6)

def test_distance_properties():
    """Test general properties of the distance matrix."""
    # Create a random PTS
    n = 5
    T = np.random.dirichlet(np.ones(n), size=n)
    Term = np.random.choice([0, 1], size=n)
    
    D = bisimulation_distance_matrix(T, Term)
    
    # Test properties
    assert np.all(np.diag(D) == 0)  # Diagonal is zero
    assert np.all(D >= 0)  # Non-negative
    assert np.all(D <= 1)  # Bounded by 1
    assert np.allclose(D, D.T)  # Symmetric
    
    # Test triangle inequality on a few random triples
    for _ in range(10):
        i, j, k = np.random.choice(n, 3, replace=False)
        assert D[i,j] <= D[i,k] + D[k,j]

@pytest.mark.parametrize("T,Term,expected_classes", [
    # Two identical states
    (np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([0, 0]), [{0, 1}]),
    
    # Two different states
    (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1, 0]), [{0}, {1}]),
    
    # Three states with two classes
    (np.array([
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 1.0]
    ]), np.array([0, 0, 1]), [{0, 1}, {2}])
])
def test_equivalence_classes(T, Term, expected_classes):
    """Test that states with zero distance are in the same equivalence class."""
    D = bisimulation_distance_matrix(T, Term)
    n = len(T)
    
    # Group states by zero distance
    classes = []
    remaining = set(range(n))
    while remaining:
        s = remaining.pop()
        class_s = {s}
        for t in list(remaining):
            if D[s,t] == 0:
                class_s.add(t)
                remaining.remove(t)
        classes.append(class_s)
    
    # Compare with expected classes
    assert len(classes) == len(expected_classes)
    for c in classes:
        assert c in expected_classes 