# Probabilistic Bisimulation Analysis Tool

A comprehensive tool for analyzing probabilistic transition systems (PTS), focusing on bisimulation distance, minimization, and system metrics. This project was developed as part of a dissertation on probabilistic bisimulation and behavioral equivalence.

## Features

- **Bisimulation Analysis**
  - Compute bisimulation distances between states
  - Identify bisimilar states and equivalence classes
  - Minimize PTS while preserving behavioral equivalence
  - Analyze state differences and explain discrepancies

- **Multiple Input Methods**
  - File upload support for various formats
  - Manual PTS entry through interactive interface
  - Built-in benchmark examples
  - Command-line interface

- **Visualization & Analysis**
  - Interactive state space visualization
  - Distance matrix heatmaps
  - Transition probability graphs
  - Comparative analysis of different metrics

- **Theoretical Background**
  - Implementation of Wasserstein distance for probability distributions
  - Support for various distance metrics (Euclidean, KL-divergence, bisimulation)
  - Simulation and comparative simulation analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bashirbobboi/Bisimulation.git
cd Bisimulation
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

> **⚠️ Important:** While you can skip this step, it's strongly recommended to use a virtual environment to:
> - Avoid conflicts with other Python packages
> - Ensure reproducible results
> - Prevent system-wide package installation
> 
> If you skip this step, you might encounter:
> - Package version conflicts
> - Issues with other Python projects
> - Difficulty reproducing results

3. Install dependencies:
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

### Command Line Interface

The tool provides a comprehensive CLI for batch processing and automation:

```bash
# Compute bisimulation distances
probisim distance input.pts --output distances.csv

# Find equivalence classes
probisim equivalence input.pts --output classes.txt

# Minimize PTS
probisim minimize input.pts --output minimized.pts

# Compare systems
probisim compare system1.pts system2.pts --metric bisimulation
```

For full CLI documentation:
```bash
probisim --help
```

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

## Contributing

This is a dissertation project, but suggestions and improvements are welcome.
- Report issues
- Suggest improvements
- Submit pull requests


## Author

Mohammed BA Bobboi
The University of Sheffield

