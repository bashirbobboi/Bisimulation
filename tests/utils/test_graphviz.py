"""
Unit tests for Graphviz DOT generation from PTS models.
Checks node/edge formatting, label handling, and minimized/class output.
"""

import numpy as np
import pytest

from probisim.bisimdistance import generate_graphviz_source


def normalize_dot(dot_src):
    """Remove whitespace differences for easier comparison of DOT source."""
    return "\n".join(line.strip() for line in dot_src.strip().splitlines())


def test_basic_nodes_and_edges_no_labels():
    """Test DOT output for a simple 2-state system with no labels."""
    # Simple 2-state system: state 1 non-terminating transitions to itself; state 2 is terminating
    T = np.array([[1.0, 0.0], [0.0, 1.0]])
    Term = np.array([0, 1], dtype=int)
    labels = {}
    dot_src = generate_graphviz_source(T, Term, labels, is_minimized=False)
    norm = normalize_dot(dot_src)

    # Check node 1 attributes
    assert '"State 1" [label="State 1"' in norm
    assert "shape=circle" in norm
    assert "color=lightgreen" in norm

    # Check node 2 (terminating) attributes
    assert '"State 2" [label="State 2"' in norm
    assert "peripheries=2" in norm
    assert "color=lightblue" in norm

    # Check edge for state1->state1 with probability label
    assert '"State 1" -> "State 1"' in norm
    assert "label=1.00" in norm


def test_string_label_edge():
    """Test DOT output for a transition with a string label."""
    # Single transition with a string label
    T = np.array([[0.0, 1.0], [0.0, 0.0]])
    Term = np.array([0, 1], dtype=int)
    labels = {(0, 1): "action"}
    dot_src = generate_graphviz_source(T, Term, labels, is_minimized=False)
    norm = normalize_dot(dot_src)

    # Edge should include label text and probability
    assert '"State 1" -> "State 2"' in norm
    assert "action (1.00)" in norm


def test_list_label_edge():
    """Test DOT output for transitions with list labels (joined by comma)."""
    # Transition label is a list -> joined by comma
    T = np.array([[0.0, 0.5, 0.5], [0, 0, 0], [0, 0, 0]])
    Term = np.array([0, 1, 1], dtype=int)
    labels = {(0, 1): ["a", "b"], (0, 2): ["c"]}
    dot_src = generate_graphviz_source(T, Term, labels, is_minimized=False)
    norm = normalize_dot(dot_src)

    # Check both edges with list labels
    assert '"State 1" -> "State 2"' in norm
    assert "a, b (0.50)" in norm
    assert '"State 1" -> "State 3"' in norm
    assert "c (0.50)" in norm


def test_minimized_prefix_and_color():
    """Test DOT output for minimized/class prefix and color handling."""
    # Check prefix changes to 'Class' and non-terminating classes colored lightgreen
    T = np.array([[1.0]])
    Term = np.array([0], dtype=int)
    labels = {}
    dot_src = generate_graphviz_source(T, Term, labels, is_minimized=True)
    norm = normalize_dot(dot_src)

    # Node should be named Class 1
    assert '"Class 1" [label="Class 1"' in norm
    # Non-terminating, so color lightgreen and no peripheries=2
    assert "color=lightgreen" in norm
    assert "peripheries" not in norm


if __name__ == "__main__":
    pytest.main()
