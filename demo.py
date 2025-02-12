# Author: Mohammed Bashir Ahmed Bobboi 
# This is the working version 
# Last Updated: 12th February 2025
# Last Update Made: Added the Bismulation Algorithm (from my pseudocode)

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

def compute_equivalence_classes(R, num_states):
    """
    Construct equivalence classes from the final bisimulation relation.
    """
    equivalence_classes = {}
    state_class_map = {}

    for x in range(num_states):
        found = False
        for class_id, class_states in equivalence_classes.items():
            if (x, next(iter(class_states))) in R:  # Check if x is related to an existing class
                equivalence_classes[class_id].add(x)
                state_class_map[x] = class_id
                found = True
                break
        if not found:
            new_class_id = len(equivalence_classes)
            equivalence_classes[new_class_id] = {x}
            state_class_map[x] = new_class_id

    return equivalence_classes, state_class_map

def compute_minimized_transition_matrix(transition_matrix, equivalence_classes, state_class_map):
    """
    Compute the minimized transition matrix using the equivalence classes.
    """
    num_classes = len(equivalence_classes)
    minimized_T = np.zeros((num_classes, num_classes))

    for class_id, class_states in equivalence_classes.items():
        for x in class_states:
            for y in range(len(transition_matrix)):
                if transition_matrix[x, y] > 0:
                    target_class = state_class_map[y]
                    minimized_T[class_id, target_class] += transition_matrix[x, y] / len(class_states)

    return minimized_T

def visualize_markov_chain(matrix, state_names, terminating_vector, filename):
    """
    Visualize a Markov chain using Graphviz.
    """
    dot = Digraph(format='png')

    # Add nodes
    for i, state in enumerate(state_names):
        if terminating_vector[i] == 1:
            dot.node(state, shape='circle', style='filled', peripheries='2', color='lightblue')
        else:
            dot.node(state, shape='circle', style='filled', color='lightgreen')
    
    # Add edges
    for i, state_from in enumerate(state_names):
        for j, state_to in enumerate(state_names):
            if matrix[i][j] > 0:
                dot.edge(state_from, state_to, label=f"{matrix[i][j]:.2f}")

    dot.render(filename, view=True)

if __name__ == "__main__":
    # Step 1: User Input
    transition_matrix, terminating_vector = input_markov_chain()
    num_states = len(transition_matrix)

    # Step 2: Compute Initial Relation R_0
    R_0 = {(x, y) for x in range(num_states) for y in range(num_states)}

    # Step 3: Apply Refinement
    R_n = refine_relation(R_0, transition_matrix, terminating_vector)

    # Step 4: Compute Equivalence Classes
    equivalence_classes, state_class_map = compute_equivalence_classes(R_n, num_states)

    # Step 5: Compute Minimized Transition Matrix
    minimized_T = compute_minimized_transition_matrix(transition_matrix, equivalence_classes, state_class_map)

    # Step 6: Visualization
    original_state_names = [f"State {i}" for i in range(num_states)]
    visualize_markov_chain(transition_matrix, original_state_names, terminating_vector, "original_markov_chain")

    minimized_state_names = [f"Class {i}" for i in range(len(equivalence_classes))]
    visualize_markov_chain(minimized_T, minimized_state_names, np.zeros(len(equivalence_classes)), "minimized_markov_chain")

    print("\nOriginal Transition Matrix:")
    print(transition_matrix)

    print("\nEquivalence Classes:")
    for class_id, class_states in equivalence_classes.items():
        print(f"Class {class_id}: {class_states}")

    print("\nMinimized Transition Matrix:")
    print(minimized_T)

