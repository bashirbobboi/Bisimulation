import numpy as np
import pytest

from probisim.bisimdistance import refine_relation

def make_full_relation(n):
    return {(i, j) for i in range(n) for j in range(n)}

def test_refine_relation_identical_nonterminating_states():
    # States with identical transitions and non-terminating remain fully related
    T = np.array([[0.5, 0.5], [0.5, 0.5]])
    Term = np.array([0, 0], dtype=int)
    R0 = make_full_relation(2)
    R = refine_relation(R0, T, Term)
    assert R == R0

def test_refine_relation_termination_mismatch():
    # States with different termination vector become unrelated
    T = np.eye(2)
    Term = np.array([1, 0], dtype=int)
    R0 = make_full_relation(2)
    R = refine_relation(R0, T, Term)
    # Only reflexive pairs should remain
    assert R == {(0, 0), (1, 1)}

def test_refine_relation_full_relation_when_probabilities_differ():
    # Even if outgoing probabilities differ, initial classes are merged, so relation remains full
    T = np.array([[1.0, 0.0], [0.0, 1.0]])
    Term = np.array([0, 0], dtype=int)
    R0 = make_full_relation(2)
    R = refine_relation(R0, T, Term)
    # Relation remains full for this scenario
    assert R == R0

def test_refine_relation_iterative_refinement():
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
    # If starting with empty R, should return empty immediately
    T = np.array([[1.0]])
    Term = np.array([0], dtype=int)
    R0 = set()
    R = refine_relation(R0, T, Term)
    assert R == set()

if __name__ == "__main__":
    pytest.main()
