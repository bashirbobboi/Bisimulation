import pytest
import numpy as np
import json

from probisim.parsers import PrismParser, JsonLTSParser, TxtParser, parse_model


def test_parse_model_dispatch_valid_formats():
    json_str = '{"states": 1, "transitions": [], "terminating": [0]}'
    for fmt in ['txt', 'json', 'prism']:
        # For formats that expect content, we provide minimal valid inputs
        if fmt == 'txt':
            input_str = '1\n1.0\n1'
        elif fmt == 'json':
            input_str = json_str
        else:  # prism
            input_str = '[] s=0 -> (s\'=0);'
        T, Term, labels = parse_model(input_str, fmt)
        assert isinstance(T, np.ndarray)
        assert isinstance(Term, np.ndarray)
        assert isinstance(labels, dict)


def test_parse_model_dispatch_invalid_format():
    with pytest.raises(ValueError):
        parse_model('anything', 'unsupported')


def test_txt_parser_basic():
    # 2-state system: state0->state1 with prob 1, state1 terminates
    txt_input = '''
2
0 1
0 1
0 1
lbl01
'''.strip()

    parser = TxtParser()
    T, Term, labels = parser.parse(txt_input)
    expected_T = np.array([[0.0, 1.0], [0.0, 1.0]])
    expected_Term = np.array([0, 1])
    assert np.allclose(T, expected_T)
    assert np.array_equal(Term, expected_Term)
    assert labels == {(0, 1): 'lbl01'}


def test_txt_parser_row_length_mismatch():
    # Row length not matching n
    bad_input = '2\n0.5 0.5 0.0\n0 1\n0 1'
    parser = TxtParser()
    with pytest.raises(ValueError):
        parser.parse(bad_input)


def test_txt_parser_term_length_mismatch():
    # Termination vector wrong length
    bad_input = '2\n0 1\n1 0\n0'
    parser = TxtParser()
    with pytest.raises(ValueError):
        parser.parse(bad_input)


def test_txt_parser_row_sum_error():
    # Non-terminating row does not sum to 1
    bad_input = '2\n0.3 0.3\n0 1\n0 1'
    parser = TxtParser()
    with pytest.raises(ValueError):
        parser.parse(bad_input)


def test_prism_parser_basic():
    prism_input = '''
// simple chain
[] s=0 -> 1.0 : (s'=1);
[] s=1 -> 0.5 : (s'=0) + 0.5 : (s'=2);
[] s=2 -> (s'=2);
'''.strip()
    parser = PrismParser()
    T, Term, labels = parser.parse(prism_input)
    expected_T = np.array([[0.0, 1.0, 0.0],
                           [0.5, 0.0, 0.5],
                           [0.0, 0.0, 1.0]])
    expected_Term = np.array([0, 0, 1])
    assert np.allclose(T, expected_T)
    assert np.array_equal(Term, expected_Term)
    assert labels == {}


def test_prism_parser_row_sum_error():
    # Row with sum !=1 and not a pure self-loop
    bad_input = '[] s=0 -> 0.5 : (s\'=1);'
    parser = PrismParser()
    with pytest.raises(ValueError):
        parser.parse(bad_input)


def test_json_parser_basic():
    import json
    # Now explicitly mark states 1 and 2 as terminating to allow rows summing to zero
    data = {
        "states": 3,
        "transitions": [
            {"from": 0, "to": 1, "prob": 0.7, "label": "a"},
            {"from": 0, "to": 2, "prob": 0.3}
        ],
        "terminating": [1, 2]
    }
    input_str = json.dumps(data)
    parser = JsonLTSParser()
    T, Term, labels = parser.parse(input_str)
    expected_T = np.array([[0.0, 0.7, 0.3], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    expected_Term = np.array([0, 1, 1])
    assert np.allclose(T, expected_T)
    assert np.array_equal(Term, expected_Term)
    assert labels == {(0, 1): 'a'}

def test_json_parser_row_sum_error():
    # Non-terminating row does not sum to 1
    data = {"states": 2, "transitions": [{"from": 0, "to": 1, "prob": 0.5}], "terminating": []}
    input_str = json.dumps(data)
    parser = JsonLTSParser()
    with pytest.raises(ValueError):
        parser.parse(input_str)
