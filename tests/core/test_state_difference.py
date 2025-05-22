"""
Unit tests for analyze_state_differences (state difference explanations).
Covers termination mismatch, no contributions, and single contribution cases.
"""
import numpy as np
import pytest

from probisim.bisimdistance import analyze_state_differences

def test_termination_mismatch():
    """Test that a termination mismatch is explained correctly."""
    # Termination status differs
    T = np.array([[1.0, 0.0], [0.0, 1.0]])
    Term = np.array([1, 0], dtype=int)
    # D_classes irrelevant for termination mismatch
    D_classes = np.zeros((2, 2))

    explanations = analyze_state_differences(0, 1, T, Term, D_classes, {0: {0}, 1: {1}}, T, {0: True, 1: False})
    expected = (
        "Termination mismatch: State 1 is terminating, "
        "while State 2 is non-terminating."
    )
    assert explanations == [expected]


def test_no_contributions_when_no_costs():
    """Test that no explanations are given when there are no positive costs."""
    # Same termination, but no positive costs => no explanations
    T = np.array([[0.5, 0.5], [0.5, 0.5]])
    Term = np.array([0, 0], dtype=int)
    # Zero matrix => no D_classes[i,j] > 0
    D_classes = np.zeros((2, 2))

    explanations = analyze_state_differences(0, 1, T, Term, D_classes, {0: {0}, 1: {1}}, T, {0: False, 1: False})
    assert explanations == []


def test_single_contribution_flow():
    """Test that a single flow contribution is explained correctly."""
    # Simple case with one flow contributing
    T = np.array([[1.0, 0.0], [0.0, 1.0]])
    Term = np.array([0, 0], dtype=int)
    # All costs = 1
    D_classes = np.ones((2, 2))

    explanations = analyze_state_differences(0, 1, T, Term, D_classes, {0: {0}, 1: {1}}, T, {0: False, 1: False})
    # Should have exactly one contribution explanation
    assert len(explanations) == 1
    exp = explanations[0]
    assert exp.startswith("  Class 1 → Class 1")
    assert "vs Class 2 → Class 2" in exp
    assert exp.endswith("contributes 1.000000 to the distance")

if __name__ == "__main__":
    pytest.main()
