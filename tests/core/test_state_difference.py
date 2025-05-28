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
    equivalence_classes = {0: {0}, 1: {1}}
    minimized_T = T.copy()
    class_termination = {0: True, 1: False}

    explanations = analyze_state_differences(
        0, 1, T, Term, D_classes, equivalence_classes, minimized_T, class_termination
    )
    expected = (
        "Termination mismatch: State 1 is terminating, "
        "while State 2 is non-terminating."
    )
    assert explanations[0] == expected


def test_no_contributions_when_no_costs():
    """Test that no explanations are given when there are no positive costs."""
    # Same termination, but no positive costs => no explanations
    T = np.array([[0.5, 0.5], [0.5, 0.5]])
    Term = np.array([0, 0], dtype=int)
    # Zero matrix => no D_classes[i,j] > 0
    D_classes = np.zeros((2, 2))
    equivalence_classes = {0: {0}, 1: {1}}
    minimized_T = T.copy()
    class_termination = {0: False, 1: False}

    explanations = analyze_state_differences(
        0, 1, T, Term, D_classes, equivalence_classes, minimized_T, class_termination
    )
    # Should only have the header and the 'no contributions' line
    assert explanations[0] == "Contributions to distance:"
    assert explanations[1].strip() == "(No nonzero contributions to the distance.)"


def test_single_contribution_flow():
    """Test that a single flow contribution is explained correctly."""
    # Simple case with one flow contributing
    T = np.array([[1.0, 0.0], [0.0, 1.0]])
    Term = np.array([0, 0], dtype=int)
    # All costs = 1
    D_classes = np.ones((2, 2))
    equivalence_classes = {0: {0}, 1: {1}}
    minimized_T = T.copy()
    class_termination = {0: False, 1: False}

    explanations = analyze_state_differences(
        0, 1, T, Term, D_classes, equivalence_classes, minimized_T, class_termination
    )
    # Should have the header and one contribution
    assert explanations[0] == "Contributions to distance:"
    assert explanations[1].startswith(
        "  Class 1 → Class 1 (p=1.00) vs Class 2 → Class 2 (p=1.00) "
    )
    assert explanations[1].endswith("contributes 1.0000 to the distance")


if __name__ == "__main__":
    pytest.main()
