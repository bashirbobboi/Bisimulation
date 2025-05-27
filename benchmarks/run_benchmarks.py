#!/usr/bin/env python3
import json, time, statistics
from pathlib import Path
import csv

from probisim.parsers import parse_model
from probisim.bisimdistance import refine_relation, bisimulation_distance_matrix

# Configuration
LIT_DIR   = Path(__file__).parent / "json"
MODELS    = [("small",  "pts_3_states.json"),
             ("medium", "pts_20_states.json"),
             ("large",  "pts_50_states.json")]
REPEATS   = 10

def time_func(f):
    """Run f() once and return elapsed seconds."""
    t0 = time.perf_counter()
    f()
    return time.perf_counter() - t0

def bench_parse(content):
    return time_func(lambda: parse_model(content, "json"))

def bench_refine(T, Term):
    R0 = {(i,i) for i in range(len(T))}
    return time_func(lambda: refine_relation(R0, T, Term))

def bench_distance(T, Term):
    return time_func(lambda: bisimulation_distance_matrix(T, Term)[0])

def main():
    fieldnames = ["model","operation","mean_s","stdev_s"]
    writer = csv.DictWriter(open("benchmarks_summary.csv","w",newline=""), fieldnames=fieldnames)
    writer.writeheader()

    print(f"{'Model':>6} | {'Parse(ms)':>10} | {'Refine(ms)':>11} | {'Dist(ms)':>10}")
    print("-"*48)

    for name, fname in MODELS:
        fpath = LIT_DIR/fname
        content = fpath.read_text()
        # parse once to get T,Term
        T, Term, _ = parse_model(content, "json")

        # parse benchmarks
        times_parse = [ bench_parse(content) for _ in range(REPEATS) ]
        # refine benchmarks
        times_ref  = [ bench_refine(T,Term) for _ in range(REPEATS) ]
        # distance benchmarks
        times_dist = [ bench_distance(T,Term) for _ in range(REPEATS) ]

        # compute stats (convert to ms)
        mp, sp = statistics.mean(times_parse)*1e3, statistics.stdev(times_parse)*1e3
        mr, sr = statistics.mean(times_ref)*1e3,   statistics.stdev(times_ref)*1e3
        md, sd = statistics.mean(times_dist)*1e3,  statistics.stdev(times_dist)*1e3

        # print table row
        print(f"{name:>6} | {mp:10.2f} | {mr:11.2f} | {md:10.2f}")

        # write CSV rows
        writer.writerow({"model":name,"operation":"parse",  "mean_s":mp/1000,"stdev_s":sp/1000})
        writer.writerow({"model":name,"operation":"refine", "mean_s":mr/1000,"stdev_s":sr/1000})
        writer.writerow({"model":name,"operation":"dist",   "mean_s":md/1000,"stdev_s":sd/1000})

    print("\nResults also written to benchmarks_summary.csv")

if __name__ == "__main__":
    main()
