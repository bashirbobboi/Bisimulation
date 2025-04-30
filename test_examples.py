# Example 1: Simple 2-state Markov chain
example1 = """
2
0.7 0.3
0.4 0.6
1 0
1 2 a
2 1 b
"""

# Example 2: 3-state Markov chain with absorbing state
example2 = """
3
0.5 0.3 0.2
0.2 0.6 0.2
0.0 0.0 1.0
0 0 1
1 2 a
2 1 b
2 3 c
"""

# Example 3: 4-state Markov chain with cycles
example3 = """
4
0.4 0.3 0.2 0.1
0.2 0.5 0.2 0.1
0.1 0.2 0.6 0.1
0.1 0.1 0.1 0.7
0 0 0 1
1 2 a
2 3 b
3 4 c
4 1 d
"""

def save_example(example, filename):
    with open(filename, 'w') as f:
        f.write(example.strip())

# Save examples to files
save_example(example1, 'example1.txt')
save_example(example2, 'example2.txt')
save_example(example3, 'example3.txt')