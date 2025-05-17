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
    input_probabilistic_transition_system
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
    with open("minimized_PTS.dot", "w") as f:
        f.write(minimized_dot)
    typer.echo("Minimized PTS Graphviz source saved to minimized_PTS.dot")
    # Save original PTS visualization
    orig_dot = generate_graphviz_source(T, Term, labels, is_minimized=False)
    with open("original_PTS.dot", "w") as f:
        f.write(orig_dot)
    typer.echo("Original PTS Graphviz source saved to original_PTS.dot")

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

if __name__ == "__main__":
    app()
