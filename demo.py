import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph

def input_markov_chain():
    """
    Input the transition matrix from the user.
    """
    num_states = int(input("Enter the number of states: "))
    print("Enter the transition matrix row by row (space-separated, each row must sum to 1):")
    
    matrix = []
    for i in range(num_states):
        row = list(map(float, input(f"Row {i + 1}: ").split()))
        if len(row) != num_states:
            raise ValueError(f"Row {i + 1} must have exactly {num_states} values.")
        if abs(sum(row) - 1) > 1e-6:
            raise ValueError(f"Row {i + 1} must sum to 1.")
        matrix.append(row)
    
    return np.array(matrix)

def visualize_with_graphviz(matrix, state_names):
    """
    Visualize a Markov chain using Graphviz for clean layouts and aesthetics.
    :param matrix: Transition matrix (2D numpy array).
    :param state_names: List of state names corresponding to the rows/columns of the matrix.
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
    dot.render("markov_chain", view=True)  # Opens the generated visualization in the default viewer

if __name__ == "__main__":
    # Ask the user to input the Markov chain
    transition_matrix = input_markov_chain()
    state_names = [f"State {i}" for i in range(len(transition_matrix))]
    
    # Visualize the Markov chain
    visualize_with_graphviz(transition_matrix, state_names)


