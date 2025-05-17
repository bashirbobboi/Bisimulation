import numpy as np

# --- File parsing and minimization helpers ---
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
        # Fallback to command-line input (not needed for Streamlit app)
        raise NotImplementedError("Command-line input is not supported in the Streamlit app.")

def refine_relation(R, transition_matrix, terminating_vector):
    num_states = len(transition_matrix)
    def transition_prob(x, equivalence_classes):
        return {class_id: sum(transition_matrix[x, y] for y in equivalence_classes[class_id])
                for class_id in equivalence_classes}
    while True:
        new_R = set()
        equivalence_classes = {i: {j for j in range(num_states) if (i, j) in R} for i in range(num_states)}
        for x in range(num_states):
            for y in range(num_states):
                if (x, y) in R:
                    x_trans = transition_prob(x, equivalence_classes)
                    y_trans = transition_prob(y, equivalence_classes)
                    if x_trans != y_trans or terminating_vector[x] != terminating_vector[y]:
                        continue
                    new_R.add((x, y))
        if new_R == R:
            break
        R = new_R
    return R

def compute_equivalence_classes(R, num_states, terminating_vector):
    equivalence_classes = {}
    state_class_map = {}
    class_termination_status = {}
    for x in range(num_states):
        found = False
        for class_id, class_states in equivalence_classes.items():
            if (x, next(iter(class_states))) in R:
                equivalence_classes[class_id].add(x)
                state_class_map[x] = class_id
                found = True
                break
        if not found:
            new_class_id = len(equivalence_classes)
            equivalence_classes[new_class_id] = {x}
            state_class_map[x] = new_class_id
    for class_id, class_states in equivalence_classes.items():
        class_termination_status[class_id] = any(terminating_vector[state] == 1 for state in class_states)
    return equivalence_classes, state_class_map, class_termination_status

def compute_minimized_transition_matrix(transition_matrix, equivalence_classes, state_class_map, transition_labels):
    num_classes = len(equivalence_classes)
    minimized_T = np.zeros((num_classes, num_classes))
    minimized_labels = {}
    for class_id, class_states in equivalence_classes.items():
        for x in class_states:
            for y in range(len(transition_matrix)):
                if transition_matrix[x, y] > 0:
                    target_class = state_class_map[y]
                    minimized_T[class_id, target_class] += transition_matrix[x, y] / len(class_states)
                    if (x, y) in transition_labels:
                        action = transition_labels[(x, y)]
                        if (class_id, target_class) not in minimized_labels:
                            minimized_labels[(class_id, target_class)] = []
                        if action not in minimized_labels[(class_id, target_class)]:
                            minimized_labels[(class_id, target_class)].append(action)
    return minimized_T, minimized_labels

# --- Bisimulation distance and visualization helpers ---
from scipy.optimize import linprog
from graphviz import Digraph
import matplotlib.pyplot as plt
import seaborn as sns

def bisimulation_distance_matrix(T, Term, tol=1e-6, max_iter=100):
    n = len(T)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if Term[i] != Term[j]:
                D[i, j] = 1.0
    for iteration in range(max_iter):
        D_prev = D.copy()
        for i in range(n):
            for j in range(n):
                if Term[i] != Term[j]:
                    continue
                c = D_prev.flatten()
                A_eq = []
                b_eq = []
                for k in range(n):
                    row = np.zeros((n, n))
                    row[k, :] = 1
                    A_eq.append(row.flatten())
                    b_eq.append(T[i, k])
                for l in range(n):
                    col = np.zeros((n, n))
                    col[:, l] = 1
                    A_eq.append(col.flatten())
                    b_eq.append(T[j, l])
                bounds = [(0, None)] * (n * n)
                res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
                if res.success:
                    D[i, j] = res.fun
                else:
                    D[i, j] = 1.0
        if np.max(np.abs(D - D_prev)) < tol:
            break
    return D

def generate_graphviz_source(T, Term, labels, is_minimized=False):
    prefix = "Class" if is_minimized else "State"
    dot = Digraph(format='png')
    for i in range(len(T)):
        if Term[i]:
            dot.node(f"{prefix} {i+1}", f"{prefix} {i+1}", shape='circle', style='filled', peripheries='2', color='lightblue')
        else:
            dot.node(f"{prefix} {i+1}", f"{prefix} {i+1}", shape='circle', style='filled', color='lightgreen')
    for i in range(len(T)):
        if Term[i]:
            continue
        for j in range(len(T)):
            if T[i, j] > 0:
                label = labels.get((i, j), "") if labels else ""
                if label:
                    if isinstance(label, list):
                        label_text = ", ".join(label)
                    else:
                        label_text = str(label)
                    dot.edge(f"{prefix} {i+1}", f"{prefix} {j+1}", label=f"{label_text} ({T[i, j]:.2f})")
                else:
                    dot.edge(f"{prefix} {i+1}", f"{prefix} {j+1}", label=f"{T[i, j]:.2f}")
    return dot.source

def analyze_state_differences(idx1, idx2, T, Term, D):
    explanations = []
    if Term[idx1] != Term[idx2]:
        explanations.append(f"Termination mismatch: State {idx1+1} is {'terminating' if Term[idx1] else 'non-terminating'}, State {idx2+1} is {'terminating' if Term[idx2] else 'non-terminating'}.")
    if Term[idx1] == Term[idx2]:
        diffs = []
        for j in range(len(T)):
            if not np.isclose(T[idx1, j], T[idx2, j]):
                diffs.append((j, T[idx1, j], T[idx2, j], abs(T[idx1, j] - T[idx2, j])))
        diffs.sort(key=lambda x: -x[3])
        for j, p1, p2, diff in diffs[:3]:
            explanations.append(f"Transition difference: State {idx1+1} → State {j+1}: {p1:.2f} vs {p2:.2f} (Δ={diff:.2f})")
    return explanations 