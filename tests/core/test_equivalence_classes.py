import pytest
import numpy as np
from probisim.bisimdistance import compute_equivalence_classes

def classes_to_sorted_list(classes_dict):
    # Convert dict of sets to list of sets, sorted by smallest element in each set
    return [set(classes_dict[k]) for k in sorted(classes_dict, key=lambda x: min(classes_dict[x]) if classes_dict[x] else -1)]

def test_single_class():
    """Test when all states are in the same equivalence class."""
    relation = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    num_states = 3
    Term = np.zeros(num_states, dtype=int)
    classes_dict, _, _ = compute_equivalence_classes(relation, num_states, Term)
    classes = classes_to_sorted_list(classes_dict)
    assert len(classes) == 1
    assert set(classes[0]) == {0, 1, 2}

def test_disjoint_classes():
    """Test when states are in completely separate classes."""
    relation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    num_states = 3
    Term = np.zeros(num_states, dtype=int)
    classes_dict, _, _ = compute_equivalence_classes(relation, num_states, Term)
    classes = classes_to_sorted_list(classes_dict)
    assert len(classes) == 3
    assert set(classes[0]) == {0}
    assert set(classes[1]) == {1}
    assert set(classes[2]) == {2}

def test_partial_classes():
    """Test when some states are equivalent and others are not."""
    relation = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    num_states = 3
    Term = np.zeros(num_states, dtype=int)
    classes_dict, _, _ = compute_equivalence_classes(relation, num_states, Term)
    classes = classes_to_sorted_list(classes_dict)
    assert len(classes) == 2
    assert set(classes[0]) == {0, 1}
    assert set(classes[1]) == {2}

def test_empty_relation():
    """Test with an empty relation."""
    relation = np.array([])
    num_states = 0
    Term = np.zeros(num_states, dtype=int)
    classes_dict, _, _ = compute_equivalence_classes(relation, num_states, Term)
    classes = classes_to_sorted_list(classes_dict)
    assert len(classes) == 0

def test_single_state():
    """Test with a single state."""
    relation = np.array([[1]])
    num_states = 1
    Term = np.zeros(num_states, dtype=int)
    classes_dict, _, _ = compute_equivalence_classes(relation, num_states, Term)
    classes = classes_to_sorted_list(classes_dict)
    assert len(classes) == 1
    assert set(classes[0]) == {0}

def test_invalid_inputs():
    """Test that invalid inputs raise appropriate errors."""
    num_states = 2
    Term = np.zeros(num_states, dtype=int)
    # Non-symmetric relation
    with pytest.raises(ValueError):
        compute_equivalence_classes(np.array([[1, 0], [1, 1]]), num_states, Term)
    # Non-reflexive relation
    with pytest.raises(ValueError):
        compute_equivalence_classes(np.array([[0, 1], [1, 1]]), num_states, Term)
    # Invalid values
    with pytest.raises(ValueError):
        compute_equivalence_classes(np.array([[1, 2], [2, 1]]), num_states, Term)

def test_class_properties():
    """Test that computed classes have the expected properties."""
    relation = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ])
    num_states = 4
    Term = np.zeros(num_states, dtype=int)
    classes_dict, _, _ = compute_equivalence_classes(relation, num_states, Term)
    classes = classes_to_sorted_list(classes_dict)
    # All states should be in exactly one class
    all_states = set()
    for class_ in classes:
        all_states.update(class_)
    assert all_states == {0, 1, 2, 3}
    # Classes should be disjoint
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            assert not set(classes[i]) & set(classes[j])
    # States in the same class should be related
    for class_ in classes:
        for i in class_:
            for j in class_:
                assert relation[i,j] == 1
    # States in different classes should not be related
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            for state_i in classes[i]:
                for state_j in classes[j]:
                    assert relation[state_i,state_j] == 0

def test_class_ordering():
    """Test that classes are ordered by their smallest element."""
    relation = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ])
    num_states = 4
    Term = np.zeros(num_states, dtype=int)
    classes_dict, _, _ = compute_equivalence_classes(relation, num_states, Term)
    classes = classes_to_sorted_list(classes_dict)
    # Classes should be ordered by their smallest element
    for i in range(len(classes) - 1):
        assert min(classes[i]) < min(classes[i + 1]) 