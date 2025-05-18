"""
Unit tests for input_probabilistic_transition_system (legacy TXT parser).
Covers valid parsing, error handling for row length, row sum, and file fallback.
"""
import numpy as np
import pytest

def write_file(tmp_path, content):
    """Helper to write content to a file in tmp_path and return its path."""
    file_path = tmp_path / "model.txt"
    file_path.write_text(content)
    return str(file_path)

from probisim.bisimdistance import input_probabilistic_transition_system
from textwrap import dedent

# === Tests for input_probabilistic_transition_system ===

def test_input_system_valid(tmp_path):
    """Test parsing a valid 3-state system with labels and comments."""
    # Create a valid 3-state system with labels and comments
    content = dedent("""
# Example PTS
3
0.5 0.5 0.0
0.0 1.0 0.0
0.0 0.0 1.0
0
1
1
1 2 a12
2 3 b23
# trailing comment ignored
""")
    file_path = write_file(tmp_path, content)

    T, Term, labels = input_probabilistic_transition_system(filename=file_path, use_file=True)

    # Verify matrix
    assert isinstance(T, np.ndarray)
    assert T.shape == (3, 3)
    expected_T = np.array([
        [0.5, 0.5, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    assert np.allclose(T, expected_T)

    # Verify termination vector
    assert isinstance(Term, np.ndarray)
    assert Term.tolist() == [0, 1, 1]

    # Verify labels dictionary
    assert labels == {(0, 1): 'a12', (1, 2): 'b23'}


def test_input_system_missing_values(tmp_path):
    """Test that a row with the wrong number of entries raises a ValueError."""
    # Row has wrong number of entries
    content = dedent("""
2
0.5 0.5 0.0
1.0 0.0
0 1
""")
    file_path = write_file(tmp_path, content)
    with pytest.raises(ValueError) as excinfo:
        input_probabilistic_transition_system(filename=file_path, use_file=True)
    assert "must have exactly 2 values" in str(excinfo.value)


def test_input_system_row_sum_error(tmp_path):
    """Test that a row not summing to 1 raises a ValueError."""
    # Row sums not equal to 1
    content = dedent("""
2
0.3 0.3
0.5 0.5
0 0
""")
    file_path = write_file(tmp_path, content)
    with pytest.raises(ValueError) as excinfo:
        input_probabilistic_transition_system(filename=file_path, use_file=True)
    assert "must sum to 1" in str(excinfo.value)


def test_input_system_no_file_fallback():
    """Test that use_file=False raises NotImplementedError (not supported in Streamlit)."""
    # use_file=False should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        input_probabilistic_transition_system(filename=None, use_file=False)
