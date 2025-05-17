import abc
import numpy as np

class ModelParser(abc.ABC):
    """
    Abstract base class for all model parsers.
    All parsers must implement parse(input_str) and return (T, Term, labels),
    where T is a 0-based numpy array, Term is a 0/1 numpy array, and labels is a dict.
    """
    @abc.abstractmethod
    def parse(self, input_str):
        pass

class PrismParser(ModelParser):
    """
    PRISM .pm parser (DTMC subset, PRISM-style syntax).
    - Input: 0-based state numbering, lines like: [] s=3 -> 0.5 : (s'=1) + 0.5 : (s'=7);
    - Absorbing/terminating: inferred as states with no outgoing transitions or only self-loop with probability 1.
    - Labels: not supported (ignored)
    - Output: 0-based numpy arrays
    """
    def parse(self, input_str):
        import re
        lines = [l.strip() for l in input_str.splitlines() if l.strip() and not l.strip().startswith('//')]
        n = 0
        transitions = {}
        for line in lines:
            m = re.match(r'\[\]\s*s\s*=\s*(\d+)\s*->\s*(.*);', line)
            if not m:
                continue
            state = int(m.group(1))
            rhs = m.group(2)
            parts = [p.strip() for p in rhs.split('+')]
            for part in parts:
                # Try "p : (s'=j)"
                pm = re.match(r"([0-9.]+)\s*:\s*\(s'\s*=\s*(\d+)\)", part)
                if pm:
                    prob, tgt = float(pm.group(1)), int(pm.group(2))
                else:
                    # Fallback: just "(s'=j)"
                    m2 = re.match(r"\(s'\s*=\s*(\d+)\)", part)
                    if m2:
                        prob, tgt = 1.0, int(m2.group(1))
                    else:
                        continue
                transitions.setdefault(state, []).append((tgt, prob))
                n = max(n, state+1, tgt+1)
        T = np.zeros((n, n))
        for s in range(n):
            if s in transitions:
                for tgt, prob in transitions[s]:
                    T[s, tgt] += prob
        # Termination: states with no outgoing transitions or only self-loop with probability 1
        Term = np.array([
            1 if (s not in transitions or (len(transitions[s]) == 1 and transitions[s][0][0] == s and abs(transitions[s][0][1] - 1.0) < 1e-6))
            else 0
            for s in range(n)
        ], dtype=int)
        labels = {}  # Not supported
        # Validation
        for s in range(n):
            if Term[s] == 0 and abs(T[s].sum() - 1.0) > 1e-6:
                raise ValueError(f"Row {s} must sum to 1 for non-terminating states")
        return T, Term, labels

class JsonLTSParser(ModelParser):
    """
    JSON LTS parser.
    - Input: 0-based state numbering
    - Format: {"states": n, "transitions": [{"from": i, "to": j, "prob": p, "label": "a"}, ...], "terminating": [i, ...]}
    - Output: 0-based numpy arrays
    """
    def parse(self, input_str):
        import json
        data = json.loads(input_str)
        n = data['states']
        T = np.zeros((n, n))
        labels = {}
        for tr in data['transitions']:
            i, j, p = tr['from'], tr['to'], tr['prob']
            T[i, j] += p
            if 'label' in tr:
                labels[(i, j)] = tr['label']
        Term = np.zeros(n, dtype=int)
        for idx in data.get('terminating', []):
            Term[idx] = 1
        # Validation
        for s in range(n):
            if Term[s] == 0 and abs(T[s].sum() - 1.0) > 1e-6:
                raise ValueError(f"Row {s} must sum to 1 for non-terminating states")
        return T, Term, labels

# Registry for pluggable parsers
PARSER_REGISTRY = {
    'prism': PrismParser(),
    'json': JsonLTSParser(),
}

def parse_model(input_str, fmt):
    """
    Dispatch to the correct parser based on format string.
    fmt: 'prism' or 'json'
    Returns (T, Term, labels) as 0-based numpy arrays.
    """
    if fmt not in PARSER_REGISTRY:
        raise ValueError(f"Unknown model format: {fmt}")
    return PARSER_REGISTRY[fmt].parse(input_str)

# To add a new format:
# 1. Implement a new class XParser(ModelParser) with a parse() method.
# 2. Add it to PARSER_REGISTRY, e.g. 'x': XParser(), 