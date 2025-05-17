import numpy as np
import pytest

from probisim.bisimdistance import analyze_state_differences

def test_termination_mismatch():
    # Termination status differs
    T = np.array([[1.0, 0.0], [0.0, 1.0]])
    Term = np.array([1, 0], dtype=int)
    # D_prev irrelevant for termination mismatch
    D_prev = np.zeros((2, 2))

    explanations = analyze_state_differences(0, 1, T, Term, D_prev)
    expected = (
        "Termination mismatch: State 1 is terminating, "
        "while State 2 is non-terminating."
    )
    assert explanations == [expected]


def test_no_contributions_when_no_costs():
    # Same termination, but no positive costs => no explanations
    T = np.array([[0.5, 0.5], [0.5, 0.5]])
    Term = np.array([0, 0], dtype=int)
    # Zero matrix => no D_prev[i,j] > 0
    D_prev = np.zeros((2, 2))

    explanations = analyze_state_differences(0, 1, T, Term, D_prev)
    assert explanations == []


def test_single_contribution_flow():
    # Simple case with one flow contributing
    T = np.array([[1.0, 0.0], [0.0, 1.0]])
    Term = np.array([0, 0], dtype=int)
    # All costs = 1
    D_prev = np.ones((2, 2))

    explanations = analyze_state_differences(0, 1, T, Term, D_prev)
    # Should have exactly one contribution explanation
    assert len(explanations) == 1
    exp = explanations[0]
    assert exp.startswith("Transition from State 1 to State 1")
    assert "vs From State 2 to State 2" in exp
    assert exp.endswith("→ this contributes 1.000 to their distance")

if __name__ == "__main__":
    pytest.main()
