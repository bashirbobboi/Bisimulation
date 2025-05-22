"""
Unit tests for the refine_relation function (partition refinement for bisimulation).
Covers identical states, termination mismatch, iterative refinement, and edge cases.
"""
import numpy as np
import pytest

from probisim.bisimdistance import refine_relation

def make_full_relation(n):
    """Helper to create the full initial relation R0 = S x S."""
    return {(i, j) for i in range(n) for j in range(n)}

def test_refine_relation_identical_nonterminating_states():
    """Test that identical non-terminating states remain fully related after refinement."""
    # States with identical transitions and non-terminating remain fully related
    T = np.array([[0.5, 0.5], [0.5, 0.5]])
    Term = np.array([0, 0], dtype=int)
    R0 = make_full_relation(2)
    R = refine_relation(R0, T, Term)
    assert R == R0

def test_refine_relation_termination_mismatch():
    """Test that states with different termination vector become unrelated."""
    # States with different termination vector become unrelated
    T = np.eye(2)
    Term = np.array([1, 0], dtype=int)
    R0 = make_full_relation(2)
    R = refine_relation(R0, T, Term)
    # Only reflexive pairs should remain
    assert R == {(0, 0), (1, 1)}

def test_refine_relation_full_relation_when_probabilities_differ():
    """Test that relation remains full even if outgoing probabilities differ (for this scenario)."""
    # Even if outgoing probabilities differ, initial classes are merged, so relation remains full
    T = np.array([[1.0, 0.0], [0.0, 1.0]])
    Term = np.array([0, 0], dtype=int)
    R0 = make_full_relation(2)
    R = refine_relation(R0, T, Term)
    # Relation remains full for this scenario
    assert R == R0

def test_refine_relation_iterative_refinement():
    """Test iterative refinement on a 3-state system where only two states are bisimilar."""
    # Three-state example where only two states are bisimilar
    # State 0 -> {1,2} equally; States 1 and 2 are identical self-loops
    T = np.array([
        [0.0, 0.5, 0.5],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    Term = np.array([0, 1, 1], dtype=int)
    R0 = make_full_relation(3)
    R = refine_relation(R0, T, Term)
    # Expected related pairs: (0,0), (1,1), (2,2), (1,2), (2,1)
    expected = {(0,0), (1,1), (2,2), (1,2), (2,1)}
    assert R == expected

def test_refine_relation_empty_relation():
    """Test that starting with an empty relation returns only reflexive pairs."""
    # If starting with empty R, should return only reflexive pairs
    T = np.array([[1.0]])
    Term = np.array([0], dtype=int)
    R0 = set()
    R = refine_relation(R0, T, Term)
    assert R == {(0, 0)}

if __name__ == "__main__":
    pytest.main()
