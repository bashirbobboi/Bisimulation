import pytest
import numpy as np
from probisim.bisimdistance import refine_relation

def relation_set_to_array(refined, n):
    arr = np.zeros((n, n), dtype=int)
    for (i, j) in refined:
        arr[i, j] = 1
    return arr

def test_identical_states():
    """Test that identical states are in the same equivalence class."""
    T = np.array([[0.5, 0.5], [0.5, 0.5]])
    Term = np.array([0, 0])
    relation = np.array([[1, 1], [1, 1]])
    n = T.shape[0]
    refined = refine_relation(relation, T, Term)
    arr = relation_set_to_array(refined, n)
    assert np.array_equal(arr, relation)

def test_different_termination():
    """Test that states with different termination status are separated."""
    T = np.array([[0.5, 0.5], [0.5, 0.5]])
    Term = np.array([0, 1])
    relation = np.array([[1, 1], [1, 1]])
    expected = np.array([[1, 0], [0, 1]])
    n = T.shape[0]
    refined = refine_relation(relation, T, Term)
    arr = relation_set_to_array(refined, n)
    assert np.array_equal(arr, expected)

def test_different_transitions():
    """Test that states with different transition probabilities are separated."""
    T = np.array([[0.5, 0.5], [0.7, 0.3]])
    Term = np.array([0, 0])
    relation = np.array([[1, 1], [1, 1]])
    expected = np.array([[1, 0], [0, 1]])
    n = T.shape[0]
    refined = refine_relation(relation, T, Term)
    arr = relation_set_to_array(refined, n)
    assert np.array_equal(arr, expected)

def test_partial_equivalence():
    """Test partial equivalence between states."""
    T = np.array([
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 1.0]
    ])
    Term = np.array([0, 0, 1])
    relation = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    expected = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    n = T.shape[0]
    refined = refine_relation(relation, T, Term)
    arr = relation_set_to_array(refined, n)
    assert np.array_equal(arr, expected)

def test_empty_relation():
    """Test with an empty relation."""
    T = np.array([[0.5, 0.5], [0.5, 0.5]])
    Term = np.array([0, 0])
    relation = np.array([[0, 0], [0, 0]])
    n = T.shape[0]
    refined = refine_relation(relation, T, Term)
    arr = relation_set_to_array(refined, n)
    assert np.array_equal(arr, relation)

def test_single_state():
    """Test with a single state."""
    T = np.array([[1.0]])
    Term = np.array([0])
    relation = np.array([[1]])
    n = T.shape[0]
    refined = refine_relation(relation, T, Term)
    arr = relation_set_to_array(refined, n)
    assert np.array_equal(arr, relation)

def test_invalid_inputs():
    """Test that invalid inputs raise appropriate errors."""
    # Mismatched dimensions
    with pytest.raises(ValueError):
        refine_relation(
            np.array([[1, 1], [1, 1]]),
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            np.array([0])
        )
    
    # Invalid probabilities
    with pytest.raises(ValueError):
        refine_relation(
            np.array([[1, 1], [1, 1]]),
            np.array([[1.5, -0.5], [0.5, 0.5]]),
            np.array([0, 0])
        )
    
    # Invalid relation matrix
    with pytest.raises(ValueError):
        refine_relation(
            np.array([[1, 1], [1, 2]]),
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            np.array([0, 0])
        )

def test_refinement_properties():
    """Test that refinement maintains important properties."""
    T = np.array([
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 1.0]
    ])
    Term = np.array([0, 0, 1])
    relation = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    n = T.shape[0]
    refined = refine_relation(relation, T, Term)
    arr = relation_set_to_array(refined, n)
    # Reflexivity
    assert np.all(np.diag(arr) == 1)
    # Symmetry
    assert np.array_equal(arr, arr.T)
    # Transitivity (if a~b and b~c then a~c)
    for i in range(len(arr)):
        for j in range(len(arr)):
            for k in range(len(arr)):
                if arr[i,j] and arr[j,k]:
                    assert arr[i,k] 