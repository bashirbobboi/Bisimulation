# Author: Mohammed Bashir Ahmed Bobboi 
# Updated: April 3rd, 2025
# Description: Computes bisimulation distances using Wasserstein metric via LP

import numpy as np
from graphviz import Digraph
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import seaborn as sns

def input_probabilistic_transition_system(filename=None, use_file=True):
    if use_file and filename:
        with open(filename, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]

        num_states = int(lines[0])
        matrix = []
        for i in range(1, num_states + 1):
            row = list(map(float, lines[i].split()))
            if len(row) != num_states:
                raise ValueError(f"State {i} must have exactly {num_states} values.")
            if abs(sum(row) - 1) > 1e-6:
                raise ValueError(f"State {i} must sum to 1.")
            matrix.append(row)

        terminating_vector = list(map(int, lines[num_states + 1:num_states + 1 + num_states]))

        transition_labels = {}
        for line in lines[num_states + 1 + num_states:]:
            parts = line.split()
            if len(parts) == 3:
                from_state, to_state, label = int(parts[0]) - 1, int(parts[1]) - 1, parts[2]
                transition_labels[(from_state, to_state)] = label

        return np.array(matrix), np.array(terminating_vector), transition_labels

    else:
        return input_probabilistic_transition_system_commandline()

def input_probabilistic_transition_system_commandline():
    num_states = int(input("Enter the number of states: "))

    print("Enter the transition matrix row by row (space-separated, each row must sum to 1):")
    matrix = []
    for i in range(num_states):
        row = list(map(float, input(f"State {i + 1}: ").split()))
        if len(row) != num_states:
            raise ValueError(f"State {i + 1} must have exactly {num_states} values.")
        if abs(sum(row) - 1) > 1e-6:
            raise ValueError(f"State {i + 1} must sum to 1.")
        matrix.append(row)

    print("Enter a column vector (0s and 1s) to indicate which states are terminating:")
    terminating_vector = []
    for i in range(num_states):
        val = int(input(f"Is state {i + 1} terminating? (1 for yes, 0 for no): "))
        if val not in [0, 1]:
            raise ValueError("Only 0 or 1 is allowed.")
        terminating_vector.append(val)

    transition_labels = {}
    for i in range(num_states):
        for j in range(num_states):
            if matrix[i][j] > 0:
                label = input(f"Enter label for transition from State {i + 1} to State {j + 1} (or press Enter to skip): ").strip()
                if label:
                    transition_labels[(i, j)] = label

    return np.array(matrix), np.array(terminating_vector), transition_labels

def wasserstein_distance(p, q, C):
    n, m = len(p), len(q)
    c = C.flatten()
    A_eq = []
    b_eq = []

    for i in range(n):
        row = np.zeros(n * m)
        row[i * m:(i + 1) * m] = 1
        A_eq.append(row)
        b_eq.append(p[i])

    for j in range(m):
        col = np.zeros(n * m)
        for i in range(n):
            col[i * m + j] = 1
        A_eq.append(col)
        b_eq.append(q[j])

    bounds = [(0, None)] * (n * m)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.success:
        coupling = res.x.reshape((n, m))
        return res.fun, coupling
    else:
        raise ValueError("Wasserstein LP did not converge: " + res.message)

def analyze_state_differences(x, y, T, Term, D_prev):
    """
    Analyze why two states are different by examining their transition probabilities and coupling.
    
    Args:
        x, y: State indices
        T: Transition matrix
        Term: Termination vector
        D_prev: Previous distance matrix
    
    Returns:
        List of explanations for why the states differ
    """
    explanations = []
    
    # Check termination status
    if Term[x] != Term[y]:
        explanations.append(f"Termination mismatch: State {x+1} is {'terminating' if Term[x] else 'non-terminating'}, "
                          f"while State {y+1} is {'terminating' if Term[y] else 'non-terminating'}")
    
    # Get transition probabilities and coupling
    dist, coupling = wasserstein_distance(T[x], T[y], D_prev)
    
    # Find significant transitions that contribute to the distance
    significant_moves = []
    for i in range(len(T)):
        for j in range(len(T)):
            if coupling[i,j] > 1e-8:  # Non-zero coupling
                contribution = coupling[i,j] * D_prev[i,j]
                if contribution > 0.01:  # Only consider significant contributions
                    significant_moves.append((i, j, coupling[i,j], D_prev[i,j], contribution))
    
    # Sort by contribution
    significant_moves.sort(key=lambda x: x[4], reverse=True)
    
    # Add explanations for top transitions
    for i, j, flow, cost, contribution in significant_moves[:3]:  # Top 3 contributions
        if T[x,i] > 0 or T[y,j] > 0:  # Only explain if there's an actual transition
            # Add transition difference explanation
            explanation = (f"Transition difference: Transition from State {x+1} to State {i+1} "
                         f"(probability = {T[x,i]:.2f}) vs From State {y+1} to State {j+1} "
                         f"(probability = {T[y,j]:.2f}) → this contributes {contribution:.3f} to their distance")


            explanations.append(explanation)
    
    return explanations

def bisimulation_distance_matrix(T, Term, max_iter=100):
    n = T.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i][j] = 0 if Term[i] == Term[j] else 1

    for _ in range(max_iter):
        D_new = np.zeros_like(D)
        for x in range(n):
            for y in range(n):
                if Term[x] != Term[y]:
                    D_new[x][y] = 1.0
                else:
                    D_new[x][y] = wasserstein_distance(T[x], T[y], D)[0]  # Only use distance value
        if np.array_equal(D_new, D):
            break
        D = D_new
    return D

def generate_graphviz_source(matrix, terminating_vector, transition_labels, is_minimized=False):
    dot = Digraph()
    prefix = "Class" if is_minimized else "State"
    
    for i in range(len(matrix)):
        label = f"{prefix} {i+1}"  # Start from 1
        style = {'shape': 'circle'}
        if terminating_vector[i]:
            dot.node(label, label, shape='circle', color='lightblue', style='filled', peripheries='2')
        else:
            dot.node(label, label, shape='circle', color='lightgreen', style='filled')

    # Add all transitions, not just those with labels
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] > 0:  # If there is a transition
                prob = matrix[i][j]
                # Get label if it exists, otherwise use empty string
                label = transition_labels.get((i, j), "")
                label_text = label if isinstance(label, str) else ", ".join(label)
                # Only show label if it exists
                edge_label = f"{label_text} ({prob:.2f})" if label_text else f"{prob:.2f}"
                dot.edge(f"{prefix} {i+1}", f"{prefix} {j+1}", label=edge_label)  # Start from 1

    return dot.source  # This is key!

def visualize_probabilistic_transition_system(matrix, terminating_vector, transition_labels, filename):
    """
    Visualize the Probabilistic Transition System (PTS) using Graphviz.
    
    Args:
        matrix (np.ndarray): Transition probability matrix
        terminating_vector (np.ndarray): Vector indicating terminating states (1) and non-terminating states (0)
        transition_labels (dict): Dictionary mapping (from_state, to_state) tuples to transition labels
        filename (str): Name of the output file (without extension)
    """
    dot = Digraph(format='png')
    
    # Add nodes
    for i in range(len(matrix)):
        node_label = f"State {i}"
        if terminating_vector[i]:
            dot.node(node_label, node_label, shape='circle', style='filled', 
                    peripheries='2', color='lightblue')
        else:
            dot.node(node_label, node_label, shape='circle', style='filled', 
                    color='lightgreen')
    
    # Add edges with transition probabilities and labels
    for (i, j), label in transition_labels.items():
        prob = matrix[i][j]
        # Handle different label formats
        if isinstance(label, str):
            label_text = label
        elif isinstance(label, list):
            label_text = ", ".join(label)
        else:
            label_text = str(label)
        
        dot.edge(f"State {i}", f"State {j}", 
                label=f"{label_text} ({prob:.2f})")
    
    dot.render(filename, view=True)

def visualize_distance_matrix(distance_matrix, filename="distance_matrix"):
    """
    Create a heatmap visualization of the distance matrix.
    
    Args:
        distance_matrix (np.ndarray): Matrix of bisimulation distances
        filename (str): Name of the output file (without extension)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, annot=True, cmap="YlOrRd", fmt=".3f",
                xticklabels=[f"State {i}" for i in range(len(distance_matrix))],
                yticklabels=[f"State {i}" for i in range(len(distance_matrix))])
    plt.title("Bisimulation Distance Matrix")
    plt.tight_layout()
    plt.savefig(f"{filename}.png")
    plt.close()

def analyze_distances(distance_matrix):
    """
    Analyze and print statistics about the distance matrix.
    
    Args:
        distance_matrix (np.ndarray): Matrix of bisimulation distances
    """
    print("\nAnalysis:")
    print(f"Minimum distance: {np.min(distance_matrix):.3f}")
    print(f"Maximum distance: {np.max(distance_matrix):.3f}")
    print(f"Average distance: {np.mean(distance_matrix):.3f}")
    
    # Find most similar and most different states (excluding self-distances)
    dist_copy = distance_matrix.copy()
    np.fill_diagonal(dist_copy, np.inf)
    min_idx = np.unravel_index(np.argmin(dist_copy), dist_copy.shape)
    max_idx = np.unravel_index(np.argmax(dist_copy), dist_copy.shape)
    
    print(f"\nMost similar states: {min_idx[0]} and {min_idx[1]} "
          f"(distance: {dist_copy[min_idx]:.3f})")
    print(f"Most different states: {max_idx[0]} and {max_idx[1]} "
          f"(distance: {dist_copy[max_idx]:.3f})")

if __name__ == "__main__":
    try:
        filename = input("Enter the filename for input data: ").strip()
        transition_matrix, terminating_vector, transition_labels = input_probabilistic_transition_system(
            filename=filename, use_file=True)
        
        distance_matrix = bisimulation_distance_matrix(transition_matrix, terminating_vector)
        
        print("\nBisimulation Distance Matrix:")
        print(np.round(distance_matrix, 3))
        
        # Generate visualizations
        visualize_probabilistic_transition_system(transition_matrix, terminating_vector, 
                                                transition_labels, "original_PTS")
        visualize_distance_matrix(distance_matrix)
        
        # Analyze distances
        analyze_distances(distance_matrix)
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")



