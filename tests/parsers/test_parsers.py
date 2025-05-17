import pytest
import numpy as np
from probisim.parsers import parse_model
import os
import tempfile
from probisim.bisimdistance import input_probabilistic_transition_system

# Test data
VALID_TXT = """3
0.5 0.5 0.0
0.0 0.5 0.5
0.0 0.0 1.0
0 0 1
a b c"""

VALID_PRISM = """[0] -> 0.5 : (state' = 1) + 0.5 : (state' = 2);
[1] -> 0.5 : (state' = 1) + 0.5 : (state' = 2);
[2] [term];"""

VALID_JSON = """{
    "states": 3,
    "transitions": [
        {"from": 0, "to": 1, "prob": 0.5, "label": "a"},
        {"from": 0, "to": 2, "prob": 0.5, "label": "b"},
        {"from": 1, "to": 1, "prob": 0.5, "label": "c"},
        {"from": 1, "to": 2, "prob": 0.5, "label": "d"}
    ],
    "terminating": [2]
}"""

INVALID_TXT = """2
0.5 0.5
0.0 1.0
1 0
a b
extra line"""

INVALID_PRISM = """[0] -> 0.5 : (state' = 1) + 0.6 : (state' = 2);"""

INVALID_JSON = """{
    "states": 2,
    "transitions": [
        {"from": 0, "to": 1, "prob": 1.5}
    ]
}"""

@pytest.mark.parametrize("content,format,expected", [
    (VALID_TXT, "txt", {
        "T": np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]),
        "Term": np.array([0, 0, 1]),
        "labels": {(0, 1): "a", (1, 1): "b", (1, 2): "c"}
    }),
    (VALID_PRISM, "prism", {
        "T": np.array([[0.0, 0.5, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]),
        "Term": np.array([0, 0, 1]),
        "labels": {}  # PRISM parser does not support labels
    }),
    (VALID_JSON, "json", {
        "T": np.array([[0.0, 0.5, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]),
        "Term": np.array([0, 0, 1]),
        "labels": {(0, 1): "a", (0, 2): "b", (1, 1): "c", (1, 2): "d"}
    })
])
def test_valid_parsers(content, format, expected):
    """Test parsing of valid input files."""
    T, Term, labels = parse_model(content, format)
    assert np.allclose(T, expected["T"])
    assert np.array_equal(Term, expected["Term"])
    # For labels, compare only keys that exist in both
    for k in expected["labels"]:
        assert k in labels
        assert labels[k] == expected["labels"][k]
    for k in labels:
        if k in expected["labels"]:
            assert labels[k] == expected["labels"][k]

@pytest.mark.parametrize("content,format,error_type", [
    (INVALID_TXT, "txt", ValueError),  # Extra line
    (INVALID_PRISM, "prism", ValueError),  # Probabilities sum to > 1
    (INVALID_JSON, "json", ValueError),  # Probability > 1
    ("invalid content", "txt", ValueError),  # Malformed content
    ("invalid content", "prism", ValueError),
    ("invalid content", "json", ValueError)
])
def test_invalid_parsers(content, format, error_type):
    """Test that invalid inputs raise appropriate errors."""
    with pytest.raises(error_type):
        parse_model(content, format)

def test_parser_edge_cases():
    """Test edge cases for parsers."""
    # Empty system
    empty_txt = "0\n"
    T, Term, labels = parse_model(empty_txt, "txt")
    assert T.shape == (0, 0)
    assert len(Term) == 0
    assert len(labels) == 0
    
    # Single state, terminating
    single_txt = "1\n1.0\n1"
    T, Term, labels = parse_model(single_txt, "txt")
    assert T.shape == (1, 1)
    assert Term[0] == 1
    assert len(labels) == 0
    
    # Single state, non-terminating
    single_txt = "1\n1.0\n0"
    T, Term, labels = parse_model(single_txt, "txt")
    assert T.shape == (1, 1)
    assert Term[0] == 0
    assert len(labels) == 0

def test_input_probabilistic_transition_system_valid(tmp_path):
    content = "2\n0.5 0.5\n0.0 1.0\n0 1\n0 1 a"
    file_path = tmp_path / "pts.txt"
    file_path.write_text(content)
    T, Term, labels = input_probabilistic_transition_system(str(file_path))
    assert np.allclose(T, [[0.5, 0.5], [0.0, 1.0]])
    assert np.array_equal(Term, [0, 1])
    assert labels == {(0, 1): "a"}

def test_input_probabilistic_transition_system_invalid_matrix(tmp_path):
    content = "2\n0.5 0.6\n0.0 1.0\n0 1"
    file_path = tmp_path / "bad.txt"
    file_path.write_text(content)
    with pytest.raises(ValueError):
        input_probabilistic_transition_system(str(file_path))

def test_input_probabilistic_transition_system_unsupported():
    with pytest.raises(NotImplementedError):
        input_probabilistic_transition_system(use_file=False) 