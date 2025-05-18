"""
Unit tests for minimization and equivalence class computation in bisimulation.
Covers relation refinement, class computation, and minimized matrix/label logic.
"""
import numpy as np
import pytest

from probisim.bisimdistance import (
    refine_relation,
    compute_equivalence_classes,
    compute_minimized_transition_matrix
)


def make_full_relation(n):
    """Helper to create the full initial relation R0 = S x S."""
    return {(i, j) for i in range(n) for j in range(n)}


def test_refine_relation_identical_states():
    """Test that two identical non-terminating states remain fully related after refinement."""
    # Two identical non-terminating states should remain fully related
    T = np.array([[0.5, 0.5], [0.5, 0.5]])
    Term = np.array([0, 0], dtype=int)
    R0 = make_full_relation(2)
    R = refine_relation(R0, T, Term)
    assert R == R0


def test_refine_relation_termination_mismatch():
    """Test that states with different termination status are not related after refinement."""
    # States with different termination status should not relate
    T = np.eye(2)
    Term = np.array([1, 0], dtype=int)
    R0 = make_full_relation(2)
    R = refine_relation(R0, T, Term)
    assert R == {(0, 0), (1, 1)}


def test_compute_equivalence_classes_full_relation():
    """Test that a full relation on two states yields one equivalence class."""
    # Full relation on two states yields one equivalence class
    n = 2
    R_mat = np.ones((n, n), dtype=int)
    classes, state_map, class_term = compute_equivalence_classes(
        R_mat, n, np.array([0, 0], dtype=int)
    )
    assert len(classes) == 1
    assert classes[0] == {0, 1}
    assert state_map[0] == state_map[1] == 0
    assert class_term[0] is False


def test_compute_minimized_transition_matrix_full_relation():
    """Test that minimized matrix is 1x1 with probability 1 when both states are equivalent."""
    # When both states are equivalent, minimized matrix is 1x1 with probability 1
    T = np.array([[0.5, 0.5], [0.5, 0.5]])
    eq_classes = {0: {0, 1}}
    state_map = {0: 0, 1: 0}
    minimized_T, minimized_labels = compute_minimized_transition_matrix(
        T, eq_classes, state_map, transition_labels={}
    )
    assert minimized_T.shape == (1, 1)
    assert pytest.approx(minimized_T[0, 0], rel=1e-6) == 1.0
    assert minimized_labels == {}


def test_full_minimization_chain_example():
    """Test full minimization on a 3-state chain where two states are bisimilar."""
    # Three-state chain where states 2 and 3 are bisimilar
    T = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.0, 0.5, 0.5]
    ])
    Term = np.array([1, 0, 0], dtype=int)
    R0 = make_full_relation(3)

    # Refine relation
    Rn = refine_relation(R0, T, Term)
    expected_pairs = {(0, 0), (1, 1), (2, 2), (1, 2), (2, 1)}
    assert Rn == expected_pairs

    # Convert to matrix form for equivalence-class computation
    R_mat = np.zeros((3, 3), dtype=int)
    for i, j in Rn:
        R_mat[i, j] = 1

    classes, state_map, class_term = compute_equivalence_classes(R_mat, 3, Term)
    # Should have two classes: {0} and {1,2}
    assert len(classes) == 2
    # Identify class ID for {1,2}
    found = False
    for cid, members in classes.items():
        if members == {1, 2}:
            assert state_map[1] == cid and state_map[2] == cid
            found = True
    assert found, "Equivalence class {1,2} not found"

    # Test minimized transition matrix
    minimized_T, minimized_labels = compute_minimized_transition_matrix(
        T, classes, state_map, transition_labels={}
    )
    assert minimized_T.shape == (2, 2)
    # Class containing {1,2} should have a self-loop probability of 1
    for cid, members in classes.items():
        if members == {1, 2}:
            assert pytest.approx(minimized_T[cid, cid], rel=1e-6) == 1.0


def test_label_propagation_single_class():
    """Test that labels are combined correctly when all states are in one class."""
    # Two original states (0,1) in one class, both transition to state 2 with different labels
    T = np.array([
        [0.0, 0.0, 1.0],  # state 0
        [0.0, 0.0, 1.0],  # state 1
        [0.0, 0.0, 1.0],  # state 2 (self-loop)
    ])
    # All states in same class
    equiv = {0: {0, 1, 2}}
    state_map = {0: 0, 1: 0, 2: 0}
    # Labels on transitions from 0->2 and 1->2
    labels = {(0, 2): 'a', (1, 2): 'b'}

    minimized_T, minimized_labels = compute_minimized_transition_matrix(T, equiv, state_map, labels)
    # Only one class so shape is (1,1) and since each original state self-loop prob
    # Minimized_T[0,0] = sum of all transitions / |class| = (1+1+1)/3 = 1.0
    assert minimized_T.shape == (1,1)
    assert pytest.approx(minimized_T[0,0], rel=1e-6) == 1.0
    # Labels should combine 'a' and 'b'
    assert (0,0) in minimized_labels
    merged = minimized_labels[(0,0)]
    assert set(merged) == {'a','b'}


def test_label_deduplication():
    """Test that duplicate labels are deduplicated in the minimized label output."""
    # Two states with same label on transitions to different targets but same class
    T = np.array([
        [0.5, 0.5],  # state 0 transitions to 0 and 1
        [1.0, 0.0],  # state 1 self-loop
    ])
    # Classes: state0 and state1 separate
    equiv = {0: {0}, 1: {1}}
    state_map = {0: 0, 1: 1}
    # Labels: both transitions from state0->0 and state1->0 have same label 'x'
    labels = {(0, 0): 'x', (1, 0): 'x'}

    minimized_T, minimized_labels = compute_minimized_transition_matrix(T, equiv, state_map, labels)
    # minimized_labels should have entries for (0,0) and (1,0)
    assert (0,0) in minimized_labels
    assert (1,0) in minimized_labels
    # Each list should contain only one 'x'
    assert minimized_labels[(0,0)] == ['x']
    assert minimized_labels[(1,0)] == ['x']


def test_no_labels_leads_to_empty_dict():
    """Test that no labels in input leads to an empty minimized label dict."""
    # No transitions_labels provided
    T = np.array([[1.0]])
    equiv = {0: {0}}
    state_map = {0: 0}
    minimized_T, minimized_labels = compute_minimized_transition_matrix(T, equiv, state_map, {})
    assert minimized_T.shape == (1,1)
    assert minimized_labels == {}


