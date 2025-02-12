import numpy as np
from graphviz import Digraph

# Load the transition matrix from the CSV file
file_path_50 = "transition_matrix_50.csv"  # Replace with your file path
file_path_100 = "transition_matrix_100.csv"  # Replace with your file path

transition_matrix_50 = np.loadtxt(file_path_50, delimiter=",")
transition_matrix_100 = np.loadtxt(file_path_100, delimiter=",")

print("50-State Transition Matrix:")
print(transition_matrix_50)

print("100-State Transition Matrix:")
print(transition_matrix_100)

def visualize_with_graphviz(matrix, state_names, output_file="markov_chain"):
    """
    Visualize a Markov chain using Graphviz.
    :param matrix: Transition matrix (2D numpy array).
    :param state_names: List of state names corresponding to the rows/columns of the matrix.
    :param output_file: Output file name for the Graphviz visualization.
    """
    dot = Digraph(format='png')
    
    # Add nodes (states)
    for state in state_names:
        dot.node(state, shape='circle', style='filled', color='lightblue')
    
    # Add edges (transitions)
    for i, state_from in enumerate(state_names):
        for j, state_to in enumerate(state_names):
            if matrix[i][j] > 0:  # Only add transitions with non-zero probabilities
                dot.edge(state_from, state_to, label=f"{matrix[i][j]:.2f}")
    
    # Render the graph
    dot.render(output_file, view=True)  # Opens the generated visualization in the default viewer

# Generate state names
state_names_50 = [f"State {i}" for i in range(transition_matrix_50.shape[0])]
state_names_100 = [f"State {i}" for i in range(transition_matrix_100.shape[0])]

# Visualize 50-state and 100-state Markov chains
visualize_with_graphviz(transition_matrix_50, state_names_50, "markov_chain_50_states")
visualize_with_graphviz(transition_matrix_100, state_names_100, "markov_chain_100_states")
