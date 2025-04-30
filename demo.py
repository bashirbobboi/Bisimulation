# Author: Mohammed Bashir Ahmed Bobboi 
# Updated: April 3rd, 2025
# Description: Computes bisimulation distances using Wasserstein metric via LP

import numpy as np
from graphviz import Digraph
from scipy.optimize import liimport matplotlib.pyplot as plt
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
        return res.fun
    else:
        raise ValueError("Wasserstein LP did not converge: " + res.message)

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
                    D_new[x][y] = wasserstein_distance(T[x], T[y], D)
        if np.array_equal(D_new, D):
            break
        D = D_new
    return D

def generate_graphviz_source(matrix, terminating_vector, transition_labels):
    dot = Digraph()
    for i in range(len(matrix)):
        label = f"State {i}"
        style = {'shape': 'circle'}
        if terminating_vector[i]:
            dot.node(label, label, shape='circle', color='lightblue', style='filled', peripheries='2')
        else:
            dot.node(label, label, shape='circle', color='lightgreen', style='filled')

    for (i, j), label in transition_labels.items():
        prob = matrix[i][j]
        label_text = label if isinstance(label, str) else ", ".join(label)
        dot.edge(f"State {i}", f"State {j}", label=f"{label_text} ({prob:.2f})")

    return dot.source  # This is key!


def visualize_distance_matrix(distance_matrix, filename="distance_matrix"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, annot=True, cmap="YlOrRd", fmt=".3f",
                xticklabels=[f"State {i+1}" for i in range(len(distance_matrix))],
                yticklabels=[f"State {i+1}" for i in range(len(distance_matrix))])
    plt.title("Bisimulation Distance Matrix")
    plt.tight_layout()
    plt.savefig(f"{filename}.png")
    plt.close()

if __name__ == "__main__":
    filename = input("Enter the filename for input data: ").strip()
    transition_matrix, terminating_vector, transition_labels = input_probabilistic_transition_system(filename=filename, use_file=True)
    distance_matrix = bisimulation_distance_matrix(transition_matrix, terminating_vector)

    print("\nBisimulation Distance Matrix:")
    print(np.round(distance_matrix, 3))

    # Visualize both the PTS and the distance matrix
    visualize_probabilistic_transition_system(transition_matrix, terminating_vector, transition_labels, "original_PTS")
    visualize_distance_matrix(distance_matrix, "distance_matrix")

    # Print some analysis
    print("\nAnalysis:")
    print(f"Minimum distance: {np.min(distance_matrix):.3f}")
    print(f"Maximum distance: {np.max(distance_matrix):.3f}")
    print(f"Average distance: {np.mean(distance_matrix):.3f}")
    
    # Find most similar and most different states
    np.fill_diagonal(distance_matrix, np.inf)  # Ignore diagonal
    min_idx = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
    max_idx = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
    print(f"\nMost similar states: {min_idx[0]+1} and {min_idx[1]+1} (distance: {distance_matrix[min_idx]:.3f})")
    print(f"Most different states: {max_idx[0]+1} and {max_idx[1]+1} (distance: {distance_matrix[max_idx]:.3f})")
ce_matrix, 3))

    visualize_probabilistic_transition_system(transition_matrix, terminating_vector, transition_labels, "original_PTS")
