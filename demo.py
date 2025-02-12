import numpy as np
from graphviz import Digraph

def input_markov_chain():
    """
    Input the transition matrix and terminating states.
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
    
    # Input terminating states as a column vector
    print("Enter a column vector (0s and 1s) to indicate which states are terminating:")
    terminating_vector = []
    for i in range(num_states):
        val = int(input(f"Is state {i} terminating? (1 for yes, 0 for no): "))
        if val not in [0, 1]:
            raise ValueError("Only 0 or 1 is allowed.")
        terminating_vector.append(val)
    
    return np.array(matrix), np.array(terminating_vector)

def visualize_with_graphviz(matrix, state_names, terminating_vector):
    """
    Visualize a Markov chain using Graphviz.
    :param matrix: Transition matrix (2D numpy array).
    :param state_names: List of state names corresponding to the rows/columns of the matrix.
    :param terminating_vector: 1D array indicating which states are terminating.
    """
    dot = Digraph(format='png')
    
    # Add nodes (states)
    for i, state in enumerate(state_names):
        if terminating_vector[i] == 1:
            dot.node(state, shape='circle', style='filled', peripheries='2', color='lightblue')
        else:
            dot.node(state, shape='circle', style='filled', color='lightgreen')
    
    # Add edges (transitions)
    for i, state_from in enumerate(state_names):
        for j, state_to in enumerate(state_names):
            if matrix[i][j] > 0:  # Only add transitions with non-zero probabilities
                dot.edge(state_from, state_to, label=f"{matrix[i][j]:.2f}")
    
    # Render the graph
    dot.render("markov_chain_v2", view=True)  # Opens the generated visualization in the default viewer

if __name__ == "__main__":
    # Ask the user to input the Markov chain and terminating states
    transition_matrix, terminating_vector = input_markov_chain()
    state_names = [f"State {i}" for i in range(len(transition_matrix))]
    
    # Visualize the Markov chain with updated representation
    visualize_with_graphviz(transition_matrix, state_names, terminating_vector)
