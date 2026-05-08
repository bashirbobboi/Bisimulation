# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Web UI
streamlit run app.py

# Tests (run from repo root)
PYTHONPATH=. pytest --cov=. --cov-report=term-missing

# Single test file
PYTHONPATH=. pytest tests/core/test_wasserstein.py -v

# Benchmarks
python benchmarks/run_benchmarks.py
```

## Architecture

All core logic lives in `probisim/`:

- **`bisimdistance.py`** — the entire algorithm stack:
  - `refine_relation` — Paige-Tarjan partition refinement for bisimulation
  - `compute_equivalence_classes` — connected components from relation matrix
  - `compute_minimized_transition_matrix` — collapse bisimilar states
  - `bisimulation_distance_matrix` — iterative LP (Wasserstein) over minimized system; returns per-state distance matrix plus intermediate data
  - `wasserstein_distance` — single LP solve via `scipy.optimize.linprog` (HiGHS)
  - `analyze_state_differences` — human-readable LP coupling explanation
  - `generate_graphviz_source` — DOT string for PTS visualization

- **`parsers.py`** — pluggable `ModelParser` ABC with `PARSER_REGISTRY`. Three formats: `prism` (DTMC subset), `json` (JSON LTS), `txt` (legacy matrix format). All parsers output 0-based `(T, Term, labels)` tuples.

- **`cli.py`** — Typer CLI. Internal format is JSON (`save_internal_json` / `load_internal_json`). All commands load this JSON, so the workflow is always: **parse → save internal JSON → run analysis commands**.

- **`app.py`** — Streamlit web interface wrapping the same `probisim` functions.

## Key data conventions

- States are **0-based internally**, **1-based in all user-facing output/CLI args**.
- `T` is an `np.ndarray` of shape `(n, n)`; rows are probability distributions (must sum to 1 for non-terminating states).
- `Term` is a 0/1 `np.ndarray` of length `n`.
- `labels` is a `dict` mapping `(i, j)` tuples to label strings (optional).
- `bisimulation_distance_matrix` returns `(D, equivalence_classes, minimized_T, class_termination, D_classes)` — most CLI commands unpack all five.

## Adding a new parser format

Subclass `ModelParser` in `parsers.py`, implement `parse(input_str) -> (T, Term, labels)`, and add to `PARSER_REGISTRY`.

## NumPy version constraint

`requirements.txt` pins `numpy>=1.24.0,<2.0.0`. NumPy 2.x removed `np.unicode_` which breaks pandas; do not upgrade past 1.x.
