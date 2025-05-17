# cli.py
import json
import typer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from graphviz import Source
from probisim.parsers import parse_model
from probisim.bisimdistance import (
    bisimulation_distance_matrix,
    refine_relation,
    compute_equivalence_classes,
    compute_minimized_transition_matrix,
    generate_graphviz_source,
    input_probabilistic_transition_system,
    analyze_state_differences
)

app = typer.Typer()

def save_internal_json(T, Term, labels, filename):
    data = {
        'T': T.tolist(),
        'Term': Term.tolist(),
        'labels': {f"{i},{j}": v for (i, j), v in labels.items()}
    }
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_internal_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    T = np.array(data['T'])
    Term = np.array(data['Term'])
    labels = {tuple(map(int, k.split(','))): v for k, v in data['labels'].items()}
    return T, Term, labels

@app.command()
def parse(input_file: str, fmt: str = typer.Argument(..., help="Model format: prism or json"), to: str = typer.Option(..., help="Output file (internal JSON format)")):
    """
    Parse a model file and save as internal JSON.
    """
    if fmt == "txt":
        T, Term, labels = input_probabilistic_transition_system(filename=input_file, use_file=True)
    else:
        with open(input_file, 'r') as f:
            content = f.read()
        T, Term, labels = parse_model(content, fmt)
    save_internal_json(T, Term, labels, to)
    typer.echo(f"Parsed {input_file} as {fmt} and saved to {to}")

@app.command()
def bisim(input_file: str, minimize: bool = typer.Option(True, "--minimize", help="Run minimization and show statistics")):
    """
    Run bisimulation minimization and print statistics.
    """
    T, Term, labels = load_internal_json(input_file)
    n = len(T)
    # Minimization
    R_0 = {(x, y) for x in range(n) for y in range(n)}
    R_n = refine_relation(R_0, T, Term)
    equivalence_classes, state_class_map, class_termination_status = compute_equivalence_classes(R_n, n, Term)
    minimized_T, minimized_labels = compute_minimized_transition_matrix(T, equivalence_classes, state_class_map, labels)
    # Print statistics
    num_classes = len(equivalence_classes)
    compression_ratio = num_classes / n
    class_sizes = [len(states) for states in equivalence_classes.values()]
    min_size = min(class_sizes)
    max_size = max(class_sizes)
    mean_size = sum(class_sizes) / len(class_sizes)
    median_size = sorted(class_sizes)[len(class_sizes) // 2]
    typer.echo(f"\n📊 Bisimulation Statistics:")
    typer.echo(f"  Number of Equivalence Classes: {num_classes}")
    typer.echo(f"  Compression Ratio: {compression_ratio:.2%}")
    typer.echo(f"  Class Sizes: min={min_size}, max={max_size}, mean={mean_size:.2f}, median={median_size}")
    typer.echo(f"\nEquivalence Classes:")
    for class_id, class_states in equivalence_classes.items():
        term = 'Terminating' if class_termination_status[class_id] else 'Non-Terminating'
        typer.echo(f"  Class {class_id}: {sorted([s+1 for s in class_states])} ({term})")
    typer.echo(f"\nMinimized Transition Matrix:")
    np.set_printoptions(precision=3, suppress=True)
    typer.echo(str(minimized_T))
    # Save minimized PTS visualization
    minimized_dot = generate_graphviz_source(minimized_T, list(class_termination_status.values()), minimized_labels, is_minimized=True)
    minimized_src = Source(minimized_dot)
    minimized_src.format = "png"
    minimized_src.render("minimized_PTS", cleanup=True)
    typer.echo("Minimized PTS image saved to minimized_PTS.png")
    # Save original PTS visualization
    orig_dot = generate_graphviz_source(T, Term, labels, is_minimized=False)
    orig_src = Source(orig_dot)
    orig_src.format = "png"
    orig_src.render("original_PTS", cleanup=True)
    typer.echo("Original PTS image saved to original_PTS.png")

@app.command()
def dist(input_file: str):
    """
    Compute and print the distance matrix, heatmap, and distance metrics.
    """
    T, Term, labels = load_internal_json(input_file)
    D = bisimulation_distance_matrix(T, Term)
    n = len(D)
    typer.echo("\nDistance Matrix:")
    np.set_printoptions(precision=3, suppress=True)
    typer.echo(str(D))
    # Save heatmap
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(D, annot=True, cmap="YlOrRd", fmt=".3f",
                xticklabels=[f"S{i+1}" for i in range(n)],
                yticklabels=[f"S{i+1}" for i in range(n)],
                annot_kws={"size": 8})
    ax.set_title("Bisimulation Distance Heatmap", pad=8, size=12)
    plt.tight_layout()
    plt.savefig("distance_heatmap.png")
    typer.echo("Distance heatmap saved to distance_heatmap.png")
    # Distance metrics
    D_flat = D[np.triu_indices_from(D, k=1)]
    min_dist = np.min(D_flat)
    max_dist = np.max(D_flat)
    mean_dist = np.mean(D_flat)
    median_dist = np.median(D_flat)
    std_dist = np.std(D_flat)
    total_pairs = n * (n-1) / 2
    zero_pairs = np.sum(D_flat == 0)
    zero_prop = zero_pairs / total_pairs
    typer.echo("\nDistance Metrics:")
    typer.echo(f"  Minimum Distance: {min_dist:.3f}")
    typer.echo(f"  Mean Distance: {mean_dist:.3f}")
    typer.echo(f"  Standard Deviation: {std_dist:.3f}")
    typer.echo(f"  Maximum Distance: {max_dist:.3f}")
    typer.echo(f"  Median Distance: {median_dist:.3f}")
    typer.echo(f"  Zero Distance Pairs: {zero_pairs} ({zero_prop:.1%})")
    # Find most similar and most different state pairs
    D_copy = D.copy()
    np.fill_diagonal(D_copy, np.inf)  # Exclude self-comparisons for min
    min_distance = np.min(D_copy)
    max_distance = np.max(D)

    min_pairs = np.where(np.abs(D_copy - min_distance) < 1e-10)
    max_pairs = np.where(np.abs(D - max_distance) < 1e-10)

    min_pairs = list(zip(min_pairs[0], min_pairs[1]))
    max_pairs = list(zip(max_pairs[0], max_pairs[1]))

    # Filter out duplicate pairs (e.g., (1,2) and (2,1))
    min_pairs = [(s1, s2) for s1, s2 in min_pairs if s1 < s2]
    max_pairs = [(s1, s2) for s1, s2 in max_pairs if s1 < s2]

    typer.echo(f"\nMost Similar State Pairs (distance={min_distance:.3f}):")
    for s1, s2 in min_pairs:
        typer.echo(f"  S{s1+1} and S{s2+1}")

    typer.echo(f"\nMost Different State Pairs (distance={max_distance:.3f}):")
    for s1, s2 in max_pairs:
        typer.echo(f"  S{s1+1} and S{s2+1}")

@app.command()
def explain(input_file: str, state1: int = typer.Argument(...), state2: int = typer.Argument(...)):
    """
    Explain why two states are similar or different.
    """
    T, Term, labels = load_internal_json(input_file)
    D = bisimulation_distance_matrix(T, Term)
    idx1, idx2 = state1 - 1, state2 - 1
    explanations = analyze_state_differences(idx1, idx2, T, Term, D)
    typer.echo(f"Distance between S{state1} and S{state2}: {D[idx1, idx2]:.3f}")
    for line in explanations:
        typer.echo(f"- {line}")

@app.command()
def classof(input_file: str, state: int = typer.Argument(...)):
    """
    Show the equivalence class for a given state.
    """
    T, Term, labels = load_internal_json(input_file)
    n = len(T)
    R_0 = {(x, y) for x in range(n) for y in range(n)}
    R_n = refine_relation(R_0, T, Term)
    equivalence_classes, state_class_map, class_termination_status = compute_equivalence_classes(R_n, n, Term)
    class_id = state_class_map[state - 1]
    class_states = sorted([s+1 for s in equivalence_classes[class_id]])
    term = 'Terminating' if class_termination_status[class_id] else 'Non-Terminating'
    typer.echo(f"State S{state} is in equivalence class {class_id+1}: {class_states} ({term})")

@app.command()
def classes(input_file: str):
    """
    List all equivalence classes and their members.
    """
    T, Term, labels = load_internal_json(input_file)
    n = len(T)
    R_0 = {(x, y) for x in range(n) for y in range(n)}
    R_n = refine_relation(R_0, T, Term)
    equivalence_classes, state_class_map, class_termination_status = compute_equivalence_classes(R_n, n, Term)
    typer.echo("Equivalence Classes:")
    for class_id, class_states in equivalence_classes.items():
        class_states_sorted = sorted([s+1 for s in class_states])
        term = 'Terminating' if class_termination_status[class_id] else 'Non-Terminating'
        typer.echo(f"  Class {class_id+1}: {class_states_sorted} ({term})")

@app.command()
def manual(to: str = typer.Option(..., help="Output file (internal JSON format)")):
    """
    Enter a PTS interactively from the command line and save as internal JSON.
    """
    typer.echo("Manual Probabilistic Transition System Entry")
    n = typer.prompt("How many states?", type=int)
    T = np.zeros((n, n))
    Term = np.zeros(n, dtype=int)
    labels = {}

    # Terminating states
    for i in range(n):
        term = typer.prompt(f"Is state {i+1} terminating? (y/n)", type=str).strip().lower()
        while term not in ("y", "n"):
            term = typer.prompt(f"Please enter 'y' or 'n' for state {i+1}:", type=str).strip().lower()
        Term[i] = 1 if term == "y" else 0

    # Transitions and labels
    for i in range(n):
        if Term[i]:
            typer.echo(f"State {i+1} is terminating. No outgoing transitions.")
            continue
        # Ask for outgoing transitions
        out_str = typer.prompt(
            f"Enter destination states and probabilities for state {i+1} (e.g. '2 0.5, 3 0.5' for 0.5 to state 2 and 0.5 to state 3), or leave blank for no transitions:"
        ).strip()
        if out_str:
            try:
                pairs = [p.strip() for p in out_str.split(',') if p.strip()]
                total_prob = 0.0
                for pair in pairs:
                    dest, prob = pair.split()
                    dest = int(dest) - 1
                    prob = float(prob)
                    if dest < 0 or dest >= n:
                        typer.echo(f"  Invalid destination state: {dest+1}")
                        return
                    T[i, dest] = prob
                    total_prob += prob
                    # Optional: ask for label
                    label = typer.prompt(f"Label for transition S{i+1}→S{dest+1} (or leave blank)", default="").strip()
                    if label:
                        labels[(i, dest)] = label
                if not np.isclose(total_prob, 1.0):
                    typer.echo(f"Warning: Probabilities from state {i+1} sum to {total_prob}, not 1.0!")
            except Exception as e:
                typer.echo(f"Error parsing transitions for state {i+1}: {e}")
                return
        else:
            typer.echo(f"State {i+1} will have no outgoing transitions.")

    save_internal_json(T, Term, labels, to)
    typer.echo(f"Manual PTS entry saved to {to}")

if __name__ == "__main__":
    app()
