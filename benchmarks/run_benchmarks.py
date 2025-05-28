#!/usr/bin/env python3
import json
import time
import statistics
from pathlib import Path
import csv
import numpy as np

from probisim.bisimdistance import (
    refine_relation,
    bisimulation_distance_matrix,
    input_probabilistic_transition_system,
)

# Configuration
LIT_DIR = Path(__file__).parent / "txt"
MODELS = [
    ("small", "pts_3_states.txt"),
    ("medium", "pts_20_states.txt"),
    ("large", "pts_50_states.txt"),
]
REPEATS = 10


def time_func(f):
    """Run f() once and return elapsed seconds."""
    t0 = time.perf_counter()
    f()
    return time.perf_counter() - t0


def bench_parse(fname):
    return time_func(
        lambda: input_probabilistic_transition_system(filename=fname, use_file=True)
    )


def bench_refine(T, Term):
    R0 = {(i, i) for i in range(len(T))}
    return time_func(lambda: refine_relation(R0, T, Term))


def bench_distance(T, Term):
    return time_func(lambda: bisimulation_distance_matrix(T, Term)[0])


def bench_simulate(T, Term, num_runs=100, max_steps=100):
    n = len(T)

    def simulate():
        for _ in range(num_runs):
            state = 0  # always start from state 0
            steps = 0
            while steps < max_steps and not Term[state]:
                state = np.random.choice(n, p=T[state])
                steps += 1

    return time_func(simulate)


def main():
    fieldnames = ["model", "operation", "mean_s", "stdev_s"]
    writer = csv.DictWriter(
        open("benchmarks_summary.csv", "w", newline=""), fieldnames=fieldnames
    )
    writer.writeheader()

    print(
        f"{'Model':>6} | {'Parse(ms)':>10} | {'Refine(ms)':>11} | {'Dist(ms)':>10} | {'Sim(ms)':>10}"
    )
    print("-" * 70)

    for name, fname in MODELS:
        fpath = LIT_DIR / fname
        # parse once to get T,Term
        T, Term, _ = input_probabilistic_transition_system(
            filename=str(fpath), use_file=True
        )

        # parse benchmarks
        times_parse = [bench_parse(str(fpath)) for _ in range(REPEATS)]
        # refine benchmarks
        times_ref = [bench_refine(T, Term) for _ in range(REPEATS)]
        # distance benchmarks
        times_dist = [bench_distance(T, Term) for _ in range(REPEATS)]
        # simulation benchmarks
        times_sim = [bench_simulate(T, Term) for _ in range(REPEATS)]

        # compute stats (convert to ms)
        mp, sp = statistics.mean(times_parse) * 1e3, statistics.stdev(times_parse) * 1e3
        mr, sr = statistics.mean(times_ref) * 1e3, statistics.stdev(times_ref) * 1e3
        md, sd = statistics.mean(times_dist) * 1e3, statistics.stdev(times_dist) * 1e3
        ms, ss = statistics.mean(times_sim) * 1e3, statistics.stdev(times_sim) * 1e3

        # print table row
        print(f"{name:>6} | {mp:10.2f} | {mr:11.2f} | {md:10.2f} | {ms:10.2f}")

        # write CSV rows
        writer.writerow(
            {
                "model": name,
                "operation": "parse",
                "mean_s": mp / 1000,
                "stdev_s": sp / 1000,
            }
        )
        writer.writerow(
            {
                "model": name,
                "operation": "refine",
                "mean_s": mr / 1000,
                "stdev_s": sr / 1000,
            }
        )
        writer.writerow(
            {
                "model": name,
                "operation": "dist",
                "mean_s": md / 1000,
                "stdev_s": sd / 1000,
            }
        )
        writer.writerow(
            {
                "model": name,
                "operation": "simulate",
                "mean_s": ms / 1000,
                "stdev_s": ss / 1000,
            }
        )

    print("\nResults also written to benchmarks_summary.csv")


if __name__ == "__main__":
    main()
