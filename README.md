# Bisimulation
This is for my probabilistic bisimulation dissertation project

# Algorithm for Computing Distance
Input:
    - Transition matrix T representing the Markov chain. i.e Where T[x][·] gives the distribution over next states from x
    - Set of states X (state space)
    - Termination vector Term[x] ∈ {0, 1}, where x = 1 if its a terminating state, else 0 
    - Ground distance function d₀(x, y) (e.g., 0 if x = y, 1 otherwise)

Output:
    - Distance matrix D of size n × n
      where D[x][y] ∈ [0, 1] represents the behavioral distance between state x and state y.
      A value of:
        - 0 means x and y are behaviorally identical (bisimilar),
        - 1 means x and y are maximally different
        - Values between 0 and 1 reflect partial similarity.

Algorithm:

1. Initialize:
    - For all states x, y in X:
       - D₀[x][y] ← d₀(x, y)      # Initial distance 
       - If Term[x] ≠ Term[y]:
            D₀[x][y] ← 1     # Hard max distance for termination mismatch
        Else:
            D₀[x][y] ← d₀(x, y)
    - Set n ← 0

2. Define update rule for behavioral distance:
    - For every pair of states x, y ∈ X:
        D_{n+1}[x][y] ← Wasserstein_Distance(T[x], T[y], ground_cost = Dₙ)
        # i.e., solve LP to find minimal cost of coupling between T[x] and T[y]
        # where the cost of moving from state i to state j is Dₙ[i][j]

3. Iterate to convergence:
    - repeat:
        a. For all x, y ∈ X:
              Compute D_{n+1}[x][y] using step 2
        b. If D_{n+1} == Dₙ:
              break
        c. Else:
              n ← n + 1

4. Return D = Dₙ as the final bisimulation distance matrix
