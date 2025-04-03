# Author: Mohammed Bashir Ahmed Bobboi 
# This is the working version 
# Last Updated: 4th March 2025
# Last Update Made: Added Reading from file input and optional command line input
# Status: Active

import numpy as np
from graphviz import Digraph
from scipy.optimize import linprog


def input_probabilistic_transition_system(filename=None, use_file=True):
    """
    Reads the transition matrix, terminating states, and transition labels from a file if use_file=True.
    Otherwise, it reads input from the command line.
    """
    if use_file and filename:
        with open(filename, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]

        # Read the number of states
        num_states = int(lines[0])

        # Read the transition matrix
        matrix = []
        for i in range(1, num_states + 1):
            row = list(map(float, lines[i].split()))
            if len(row) != num_states:
                raise ValueError(f"State {i} must have exactly {num_states} values.")
            if abs(sum(row) - 1) > 1e-6:
                raise ValueError(f"State {i} must sum to 1.")
            matrix.append(row)

        # Read terminating states
        terminating_vector = list(map(int, lines[num_states + 1:num_states + 1 + num_states]))

        # Read transition labels
        transition_labels = {}
        for line in lines[num_states + 1 + num_states:]:
            parts = line.split()
            if len(parts) == 3:
                from_state, to_state, label = int(parts[0]) - 1, int(parts[1]) - 1, parts[2]
                transition_labels[(from_state, to_state)] = label

        return np.array(matrix), np.array(terminating_vector), transition_labels

    else:
        # Fallback to command-line input
        return input_probabilistic_transition_system_commandline()

def input_probabilistic_transition_system_commandline():
    """
    Fallback: Reads input from the command line.
    """
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

    # Input terminating states
    print("Enter a column vector (0s and 1s) to indicate which states are terminating:")
    terminating_vector = []
    for i in range(num_states):
        val = int(input(f"Is state {i + 1} terminating? (1 for yes, 0 for no): "))  # Start from 1
        if val not in [0, 1]:
            raise ValueError("Only 0 or 1 is allowed.")
        terminating_vector.append(val)

    # Input transition labels
    print("Enter a label for each nonzero transition (e.g., 'a, b', 'click, reset'):")
    transition_labels = {}
    for i in range(num_states):
        for j in range(num_states):
            if matrix[i][j] > 0:
                label = input(f"Enter label for transition from State {i + 1} to State {j + 1} (or press Enter to skip): ").strip()
                if label:
                    transition_labels[(i, j)] = label

    return np.array(matrix), np.array(terminating_vector), transition_labels

def compute_kantorovich_distance(P, Q, D):
    """
    Solves the Kantorovich LP between two probability distributions P and Q,
    using D as the current ground distance matrix.
    """
    n = len(P)
    c = D.flatten()

    A_eq = []
    b_eq = []

    # Row sums = P
    for i in range(n):
        row = np.zeros((n, n))
        row[i, :] = 1
        A_eq.append(row.flatten())
        b_eq.append(P[i])

    # Column sums = Q
    for j in range(n):
        col = np.zeros((n, n))
        col[:, j] = 1
        A_eq.append(col.flatten())
        b_eq.append(Q[j])

    bounds = [(0, None)] * (n * n)

    result = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if result.success:
        return result.fun
    else:
        raise RuntimeError("Kantorovich LP failed.")




def refine_relation(R, transition_matrix, terminating_vector):
    """
    Apply the refinement function to iteratively reduce the relation R.
    """
    num_states = len(transition_matrix)
    
    def transition_prob(x, equivalence_classes):
        """
        Compute transition probabilities to equivalence classes.
        """
        return {class_id: sum(transition_matrix[x, y] for y in equivalence_classes[class_id])
                for class_id in equivalence_classes}
    
    while True:
        new_R = set()
        equivalence_classes = {i: {j for j in range(num_states) if (i, j) in R} for i in range(num_states)}

        for x in range(num_states):
            for y in range(num_states):
                if (x, y) in R:
                    # Check transition probability condition
                    x_trans = transition_prob(x, equivalence_classes)
                    y_trans = transition_prob(y, equivalence_classes)
                    
                    if x_trans != y_trans or terminating_vector[x] != terminating_vector[y]:
                        continue  # (x, y) is removed if conditions are not met
                    
                    new_R.add((x, y))

        if new_R == R:  # If R stabilizes, stop
            break
        R = new_R

    return R

def compute_equivalence_classes(R, num_states, terminating_vector):
    """
    Construct equivalence classes from the final bisimulation relation.
    Determine if an equivalence class is terminating.
    """
    equivalence_classes = {}
    state_class_map = {}
    class_termination_status = {}

    for x in range(num_states):
        found = False
        for class_id, class_states in equivalence_classes.items():
            if (x, next(iter(class_states))) in R:  # If x is related to an existing class
                equivalence_classes[class_id].add(x)
                state_class_map[x] = class_id
                found = True
                break
        if not found:
            new_class_id = len(equivalence_classes)
            equivalence_classes[new_class_id] = {x}
            state_class_map[x] = new_class_id

    # Determine which equivalence classes are terminating
    for class_id, class_states in equivalence_classes.items():
        class_termination_status[class_id] = any(terminating_vector[state] == 1 for state in class_states)

    return equivalence_classes, state_class_map, class_termination_status


def compute_minimized_transition_matrix(transition_matrix, equivalence_classes, state_class_map, transition_labels):
    num_classes = len(equivalence_classes)
    minimized_T = np.zeros((num_classes, num_classes))
    minimized_labels = {}  # Store transition labels for the minimized system

    for class_id, class_states in equivalence_classes.items():
        for x in class_states:
            for y in range(len(transition_matrix)):
                if transition_matrix[x, y] > 0:
                    target_class = state_class_map[y]
                    minimized_T[class_id, target_class] += transition_matrix[x, y] / len(class_states)

                    # Preserve transition labels
                    if (x, y) in transition_labels:
                        action = transition_labels[(x, y)]
                        if (class_id, target_class) not in minimized_labels:
                            minimized_labels[(class_id, target_class)] = []
                        if action not in minimized_labels[(class_id, target_class)]:  # Prevent duplicates
                            minimized_labels[(class_id, target_class)].append(action)


    return minimized_T, minimized_labels


def visualize_probabilistic_transition_system(matrix, terminating_classes, transition_labels, filename):
    """
    Visualize the Probabilistic Transition System (PTS) with correct label formatting.
    """
    dot = Digraph(format='png')

    # Add nodes
    for i in range(len(matrix)):
        if terminating_classes[i]:
            dot.node(f"Class {i}", f"Class {i}", shape='circle', style='filled', peripheries='2', color='lightblue')  
        else:
            dot.node(f"Class {i}", f"Class {i}", shape='circle', style='filled', color='lightgreen')  

    # Add edges with transition probabilities and labels
    for (i, j), label in transition_labels.items():
        prob = matrix[i][j]
        
        # Ensure label is treated correctly (single string or list of strings)
        if isinstance(label, str):  # If it's a string, use as is
            label_text = label
        elif isinstance(label, list):  # If it's a list, join properly
            label_text = ", ".join(label)
        else:  # Fallback conversion
            label_text = str(label)

        dot.edge(f"Class {i}", f"Class {j}", label=f"{label_text} ({prob:.2f})")

    dot.render(filename, view=True)

def compute_distance_matrix(T, Term, epsilon=1e-4, max_iter=50):
    """
    Computes the bisimulation distance matrix using fixed-point iteration.
    """
    n = len(T)
    D = np.zeros((n, n))

    # Step 1: Initialize
    for x in range(n):
        for y in range(n):
            if Term[x] != Term[y]:
                D[x][y] = 1  # Max dissimilarity

    for _ in range(max_iter):
        new_D = np.copy(D)
        for x in range(n):
            for y in range(n):
                if Term[x] != Term[y]:
                    continue
                P = T[x]
                Q = T[y]
                new_D[x][y] = compute_kantorovich_distance(P, Q, D)

        if np.max(np.abs(new_D - D)) < epsilon:
            break
        D = new_D

    return D


if __name__ == "__main__":
    # Step 1: Input the Probabilistic Transition System
    filename = input("Enter the filename for input data: ").strip()
    
    transition_matrix, terminating_vector, transition_labels = input_probabilistic_transition_system(filename=filename, use_file=True)
    num_states = len(transition_matrix)

    # Step 2: Compute Initial Relation R_0
    R_0 = {(x, y) for x in range(num_states) for y in range(num_states)}

    # Step 3: Apply Refinement
    R_n = refine_relation(R_0, transition_matrix, terminating_vector)

    # Step 4: Compute Equivalence Classes and Termination Status
    equivalence_classes, state_class_map, class_termination_status = compute_equivalence_classes(R_n, num_states, terminating_vector)

    # Step 5: Compute Minimized Transition Matrix (Now with Labels)
    minimized_T, minimized_labels = compute_minimized_transition_matrix(transition_matrix, equivalence_classes, state_class_map, transition_labels)

    # Step 6: Visualization
    visualize_probabilistic_transition_system(transition_matrix, terminating_vector, transition_labels, "original_PTS")
    visualize_probabilistic_transition_system(minimized_T, list(class_termination_status.values()), minimized_labels, "minimized_PTS")


    print("\nOriginal Transition Matrix:")
    print(transition_matrix)

    print("\nEquivalence Classes:")
    for class_id, class_states in equivalence_classes.items():
        print(f"Class {class_id}: {class_states}, Terminating: {class_termination_status[class_id]}")

    print("\nMinimized Transition Matrix:")
    print(minimized_T)

    print("\nMinimized Transition Labels:")
    for (i, j), actions in minimized_labels.items():
        print(f"Class {i} → Class {j}: {', '.join(actions) if actions else 'No Label'}")

    print("\nComputing Distance Matrix using Kantorovich Metric...")
    distance_matrix = compute_distance_matrix(transition_matrix, terminating_vector)
    print("\nDistance Matrix:")
    print(np.round(distance_matrix, 3))

