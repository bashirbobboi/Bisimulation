# Probabilistic Bisimulation Analysis Tool

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Interface

Launch the Streamlit web interface:
```bash
streamlit run app.py
```

The web interface provides:
- Interactive PTS visualization
- Distance computation and analysis
- State difference explanations
- System metrics and statistics

---

## Command-Line Interface

All commands are exposed via the `cli.py` script. Use the `--help` flag to explore options for each command.

### Model Parsing and Entry

- **Parse a model (to internal JSON):**
  ```bash
  python cli.py parse path/to/model.pm prism --to internal.json
  ```
  <sub>Supported formats: `prism`, `json`, `txt`</sub>

- **Manual entry of a PTS:**
  ```bash
  python cli.py manual --to internal.json
  ```
  <sub>Enter a PTS interactively from the command line.</sub>

---

### Bisimulation and Analysis

- **Run bisimulation minimization and show statistics:**
  ```bash
  python cli.py bisim internal.json
  ```

- **Compute distance matrix and metrics:**
  ```bash
  python cli.py dist internal.json
  ```

- **Explain state differences:**
  ```bash
  python cli.py explain internal.json 1 2
  ```
  <sub>Explains level of similarity between state 1 and state 2 .</sub>

- **Get equivalence class of a state:**
  ```bash
  python cli.py classof internal.json 3
  ```

- **List all equivalence classes:**
  ```bash
  python cli.py classes internal.json
  ```

---

### Simulation

- **Simulate random runs from a starting state:**
  ```bash
  python cli.py simulate internal.json --start-state 1 --num-simulations 100 --max-steps 100 --show-runs
  ```
  <sub>Simulates random runs and reports statistics. `--show-runs` is optional.</sub>

- **Comparative simulation from two starting states:**
  ```bash
  python cli.py compare-sim internal.json --state1 1 --state2 2 --num-runs 20 --max-steps 100 --show-runs
  ```
  <sub>Compares simulation statistics and sample runs from two different starting states.</sub>

---

### General Help

- **Show help for any command:**
  ```bash
  python cli.py --help
  python cli.py <command> --help
  ```

---

**Tip:** All commands support `--help` for detailed usage and options.

## Testing and Coverage

A comprehensive test suite is provided to ensure correctness, robustness, and reliability of the tool.

- **Test Coverage:**
  - Achieved **over 90% coverage** on all files related to the core logic (including bisimulation distance, minimization, equivalence classes, simulation, and metrics).
  - Tests cover core algorithms, parsers, Wasserstein distance, refinement, equivalence classes, simulation, state difference explanations, minimization, CLI, Streamlit app, visualization, and metrics.

- **Testing Frameworks:**
  - [pytest](https://docs.pytest.org/): For unit and integration tests

- **How to Run Tests:**
  1. Make sure all dependencies are installed:
     ```bash
     pip install -r requirements.txt
     ```
  2. Run the test suite from root directory:
     ```bash
     $env:PYTHONPATH="."
     pytest --cov=. --cov-report=term-missing
     ```
     This will show a coverage report in the terminal.

- **Test Organization:**
  - Tests are organized by module in the `tests/` directory (e.g., `tests/core/`, `tests/app/`, `tests/cli/`, `tests/utils/`, `tests/parsers/`).
  - Each test file includes clear docstrings and comments for clarity and reproducibility.

## Theoretical Background

### Bisimulation Distance

The tool implements a metric for quantifying behavioral similarity between states in a PTS. The distance is computed iteratively using the following algorithm:

1. **Initialization**
   - Set initial distances based on termination status
   - For states with different termination status: distance = 1
   - For states with same termination status: distance = 0

2. **Iterative Refinement**
   - For each pair of states (x, y):
     - Compute Wasserstein distance between their transition distributions
     - Use current distances as ground costs
   - Repeat until convergence

3. **Convergence**
   - The algorithm converges to a fixed point
   - The resulting distance matrix satisfies the triangle inequality
   - Distance of 0 indicates bisimilar states

### Implementation Details

- **Core Components**
  - `bisimdistance.py`: Core algorithms for distance computation
  - `parsers.py`: Pluggable parser system for different input formats
  - `cli.py`: Command-line interface using Typer
  - `app.py`: Streamlit web interface

- **Testing**
  - Comprehensive test suite with >90% coverage
  - Property-based testing using Hypothesis
  - Unit tests for all core components
  - Integration tests for CLI and web interface

## Author

Mohammed BA Bobboi
The University of Sheffield
Grade: Distinction / First Class

