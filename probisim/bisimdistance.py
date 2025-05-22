import numpy as np
from scipy.optimize import linprog
from graphviz import Digraph
import matplotlib.pyplot as plt
import seaborn as sns

# --- File parsing and minimization helpers ---
def input_probabilistic_transition_system(filename=None, use_file=True):
    """
    Reads the transition matrix, terminating states, and transition labels from a file if use_file=True.
    Otherwise, it reads input from the command line.
    Args:
        filename: str, path to the input file.
        use_file: bool, whether to read from file (True) or prompt (False).
    Returns:
        matrix: np.ndarray, transition matrix.
        terminating_vector: np.ndarray, 0/1 vector for termination.
        transition_labels: dict, transition labels.
    Raises:
        ValueError: If the file format is invalid or matrix rows do not sum to 1.
        NotImplementedError: If use_file is False (not supported in Streamlit app).
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

def build_predecessor_lists(transition_matrix):
    """
    Build predecessor lists for each state in the transition matrix.
    Returns a list where preds[y] contains all states x with T[x,y] > 0.
    """
    n = len(transition_matrix)
    preds = [[] for _ in range(n)]
    for x in range(n):
        for y in range(n):
            if transition_matrix[x, y] > 0:
                preds[y].append(x)
    return preds

def refine_relation(R, transition_matrix, terminating_vector):
    """
    Efficient partition refinement algorithm for probabilistic bisimulation using Paige-Tarjan style.
    Properly handles probability masses when splitting blocks.
    Args:
        R: Set of (i, j) pairs representing the current relation.
        transition_matrix: np.ndarray, the transition matrix of the PTS.
        terminating_vector: np.ndarray, 0/1 vector indicating terminating states.
    Returns:
        Set of (i, j) pairs representing the coarsest bisimulation relation.
    """
    n = len(transition_matrix)
    
    # Build predecessor lists for efficient lookup
    preds = build_predecessor_lists(transition_matrix)
    
    # Initialize partition: split by termination status
    blocks = {}
    state_to_block = {}
    worklist = []
    
    # Create initial blocks based on termination status
    term_block = set()
    nonterm_block = set()
    for i in range(n):
        if terminating_vector[i]:
            term_block.add(i)
            state_to_block[i] = 0
        else:
            nonterm_block.add(i)
            state_to_block[i] = 1
    
    blocks[0] = term_block
    blocks[1] = nonterm_block
    
    # Add smaller block to worklist
    if len(term_block) <= len(nonterm_block):
        worklist.append(0)
    else:
        worklist.append(1)
    
    # Main refinement loop
    while worklist:
        splitter_id = worklist.pop()
        splitter = blocks[splitter_id]
        
        # For each state in splitter, process its predecessors
        incoming_states = set()
        for y in splitter:
            incoming_states.update(preds[y])
        
        # Group incoming states by their current block and probability mass
        block_to_states = {}
        for x in incoming_states:
            # Calculate total probability mass from x into splitter, with rounding for stability
            eps = 1e-8
            w_x = sum(transition_matrix[x, y] for y in splitter)
            # Round to 8 decimal places (integer parameter)
            w_x = round(w_x, 8)
            block_id = state_to_block[x]
            
            if block_id not in block_to_states:
                block_to_states[block_id] = {}
            
            # Group by probability mass
            bucket = block_to_states[block_id].setdefault(w_x, set())
            bucket.add(x)
        
        # Split blocks that have states leading into splitter
        for block_id, mass_groups in block_to_states.items():
            if len(mass_groups) > 1:  # Only split if there are different probability masses
                # Snapshot the original block before any splitting
                original_block = blocks[block_id].copy()
                
                # For each probability mass group
                for mass, states in mass_groups.items():
                    if len(states) < len(original_block):  # Compare against original block size
                        # Create new block for this probability mass
                        new_block = states
                        old_block = original_block - new_block
                        
                        # Update block assignments
                        blocks[block_id] = old_block
                        new_block_id = len(blocks)
                        blocks[new_block_id] = new_block
                        
                        for s in new_block:
                            state_to_block[s] = new_block_id
                        
                        # Add smaller block to worklist
                        if len(old_block) <= len(new_block):
                            worklist.append(block_id)
                        else:
                            worklist.append(new_block_id)
    
    # Convert partition back to relation
    R_new = set()
    for block_id, states in blocks.items():
        for x in states:
            for y in states:
                R_new.add((x, y))
    
    return R_new

def compute_equivalence_classes(R, num_states, terminating_vector):
    """
    Given a 0/1 relation matrix R of shape (n,n), partition {0,…,n-1}
    into connected components under R.  Also compute a map from each
    state to its class ID and record whether any member of each class
    is terminating.
    """
    R = np.asarray(R)

    # --- 1) Handle the empty-system corner --- #
    if num_states == 0:
        return {}, {}, {}

    # --- 2) Basic sanity checks on R --- #
    if R.shape != (num_states, num_states):
        raise ValueError(f"Relation matrix must be shape ({num_states},{num_states})")
    if not np.all((R == 0) | (R == 1)):
        raise ValueError("Relation matrix must contain only 0 or 1 values")
    # Reflexivity
    if not np.all(np.diag(R) == 1):
        raise ValueError("Relation must be reflexive (all diagonal entries = 1)")
    # Symmetry
    if not np.all(R == R.T):
        raise ValueError("Relation must be symmetric")

    # --- 3) Find connected components #
    visited = set()
    equivalence_classes = {}
    state_class_map    = {}
    class_termination_status = {}

    class_id = 0
    for i in range(num_states):
        if i in visited:
            continue

        # grow a new component starting from i
        comp = set()
        stack = [i]
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            # neighbors are all j with R[u,j] == 1
            for j in np.where(R[u] == 1)[0]:
                if j not in comp:
                    stack.append(j)

        # register this class
        equivalence_classes[class_id] = comp
        for s in comp:
            state_class_map[s] = class_id
        # any member terminating?
        class_termination_status[class_id] = any(terminating_vector[s] == 1 for s in comp)

        visited |= comp
        class_id += 1

    return equivalence_classes, state_class_map, class_termination_status

def compute_minimized_transition_matrix(transition_matrix, equivalence_classes, transition_labels):
    """
    Compute the minimized transition matrix and labels based on equivalence classes.
    Args:
        transition_matrix: np.ndarray, original transition matrix.
        equivalence_classes: dict, mapping class_id to set of states.
        transition_labels: dict, original transition labels.
    Returns:
        minimized_T: np.ndarray, minimized transition matrix.
        minimized_labels: dict, minimized transition labels.
    """
    num_classes = len(equivalence_classes)
    minimized_T = np.zeros((num_classes, num_classes))
    minimized_labels = {}
    
    # Derive state_class_map from equivalence_classes
    state_class_map = {s: c for c, states in equivalence_classes.items() for s in states}
    
    # Compute transition probabilities between classes
    for x in range(len(transition_matrix)):
        if x not in state_class_map:
            continue
        source_class = state_class_map[x]
        for y in range(len(transition_matrix)):
            if y not in state_class_map:
                continue
            target_class = state_class_map[y]
            minimized_T[source_class, target_class] += transition_matrix[x, y]
    
    # Normalize rows to ensure they sum to 1
    row_sums = minimized_T.sum(axis=1)
    minimized_T = np.divide(minimized_T, row_sums[:, np.newaxis], where=row_sums[:, np.newaxis] != 0)
    
    # Compute minimized labels
    for (x, y), label in transition_labels.items():
        if x in state_class_map and y in state_class_map:
            source_class = state_class_map[x]
            target_class = state_class_map[y]
            if (source_class, target_class) not in minimized_labels:
                minimized_labels[(source_class, target_class)] = set()
            minimized_labels[(source_class, target_class)].add(label)
    
    return minimized_T, minimized_labels

def analyze_state_differences(idx1, idx2, T, Term, D_classes, equivalence_classes, minimized_T, class_termination):
    """
    Provides a human-readable explanation of why two states differ in behavior, based on their bisimulation distance and transition structure.
    Uses the minimized system to analyze differences between equivalence classes.
    Args:
        idx1: int, index of the first state.
        idx2: int, index of the second state.
        T: np.ndarray, transition matrix.
        Term: np.ndarray, termination vector.
        D_classes: np.ndarray, class-level distance matrix.
        equivalence_classes: dict, mapping class_id to set of states.
        minimized_T: np.ndarray, minimized transition matrix.
        class_termination: dict, mapping class_id to termination status.
    Returns:
        List of strings explaining the main sources of difference between the two states.
    """
    # Get the equivalence classes for the states we're comparing
    state_class_map = {s: c for c, states in equivalence_classes.items() for s in states}
    class1 = state_class_map[idx1]
    class2 = state_class_map[idx2]
    
    explanations = []
    
    # 1) Termination
    if Term[idx1] != Term[idx2]:
        explanations.append(
            f"Termination mismatch: State {idx1+1} is "
            f"{'terminating' if Term[idx1] else 'non-terminating'}, "
            f"while State {idx2+1} is "
            f"{'terminating' if Term[idx2] else 'non-terminating'}."
        )

    # 2) If same termination status, do the LP coupling on minimized system
    if Term[idx1] == Term[idx2]:
        # Compute distance between the two classes we're interested in
        num_classes = len(equivalence_classes)
        
        # Use the class-level distance matrix for the cost
        c = D_classes.flatten()
        
        A_eq = []
        b_eq = []
        
        # Row constraints for class1
        for k in range(num_classes):
            row = np.zeros((num_classes, num_classes))
            row[k, :] = 1
            A_eq.append(row.flatten())
            b_eq.append(minimized_T[class1, k])
        
        # Column constraints for class2
        for l in range(num_classes):
            col = np.zeros((num_classes, num_classes))
            col[:, l] = 1
            A_eq.append(col.flatten())
            b_eq.append(minimized_T[class2, l])
        
        # Convert to numpy arrays and ensure correct shapes
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        
        # Verify dimensions match
        if len(c) != A_eq.shape[1]:
            raise ValueError(f"Dimension mismatch: c length ({len(c)}) != A_eq columns ({A_eq.shape[1]})")
        
        # Set up bounds for the LP variables
        bounds = [(0, None)] * len(c)
        
        # Solve the LP
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if res.success:
            dist = res.fun
            coupling = res.x.reshape((num_classes, num_classes))
            
            # Analyze the coupling to explain differences
            explanations.append("Contributions to distance:")
            moves = []
            # Collect all contributions, even small ones
            for i in range(num_classes):
                for j in range(num_classes):
                    flow = coupling[i,j]
                    contrib = flow * D_classes[i,j]
                    if flow > 1e-8 and contrib > 1e-8:  # Only keep nonzero contributions
                        moves.append((i, j, minimized_T[class1,i], minimized_T[class2,j], contrib))
            # Sort by contribution and show top 3
            moves.sort(key=lambda x: x[4], reverse=True)
            for i, j, p1, p2, contrib in moves[:3]:
                explanations.append(
                    f"  Class {class1+1} → Class {i+1} (p={p1:.2f}) vs "
                    f"Class {class2+1} → Class {j+1} (p={p2:.2f}) "
                    f"contributes {contrib:.4f} to the distance"
                )
            if not moves:
                explanations.append("  (No nonzero contributions to the distance.)")
            explanations.append(f"Total distance between states: {dist:.6f}")
    
    return explanations

def bisimulation_distance_matrix(T, Term, tol=1e-6, max_iter=100):
    """
    Compute the bisimulation distance matrix via iterative LP solves (Wasserstein metric).
    First minimizes the system, then computes distances between equivalence classes.
    Args:
        T: np.ndarray, transition matrix.
        Term: np.ndarray, termination vector.
        tol: float, convergence tolerance.
        max_iter: int, maximum number of iterations.
    Returns:
        D: np.ndarray, bisimulation distance matrix for the original system.
    """
    n = len(T)
    
    # First, compute the minimized system
    R = refine_relation(set((i,i) for i in range(n)), T, Term)
    R_matrix = np.zeros((n, n), dtype=int)
    for i, j in R:
        R_matrix[i, j] = 1
    
    equivalence_classes, _, class_termination = compute_equivalence_classes(R_matrix, n, Term)
    minimized_T, _ = compute_minimized_transition_matrix(T, equivalence_classes, {})
    
    # Compute distances between equivalence classes
    num_classes = len(equivalence_classes)
    D_minimized = np.zeros((num_classes, num_classes))
    
    # Initialize distances based on termination
    for i in range(num_classes):
        for j in range(num_classes):
            if class_termination[i] != class_termination[j]:
                D_minimized[i, j] = 1.0
    
    # Iteratively update distances using the minimized system
    for iteration in range(max_iter):
        D_prev = D_minimized.copy()
        for i in range(num_classes):
            for j in range(num_classes):
                if class_termination[i] != class_termination[j]:
                    continue
                
                # Set up the LP for the minimized system
                c = D_prev.flatten()

                A_eq = []
                b_eq = []
                
                # Row constraints for class i
                for k in range(num_classes):
                    row = np.zeros((num_classes, num_classes))
                    row[k, :] = 1
                    A_eq.append(row.flatten())
                    b_eq.append(minimized_T[i, k])
                
                # Column constraints for class j
                for l in range(num_classes):
                    col = np.zeros((num_classes, num_classes))
                    col[:, l] = 1
                    A_eq.append(col.flatten())
                    b_eq.append(minimized_T[j, l])
                
                # Convert to numpy arrays and ensure correct shapes
                A_eq = np.array(A_eq)
                b_eq = np.array(b_eq)
                
                # Reshape c to match A_eq if needed
                if len(c) != A_eq.shape[1]:
                    c = c.reshape(-1)
                
                # Set up bounds for the LP variables
                bounds = [(0, None)] * len(c)
                
                # Solve the LP
                res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
                if res.success:
                    D_minimized[i, j] = res.fun
                else:
                    D_minimized[i, j] = 1.0
                    
        if np.max(np.abs(D_minimized - D_prev)) < tol:
            break
    
    # Map minimized distances back to original system
    D = np.zeros((n, n))
    state_class_map = {s: c for c, states in equivalence_classes.items() for s in states}
    for i in range(n):
        for j in range(n):
            D[i, j] = D_minimized[state_class_map[i], state_class_map[j]]
    
    D_classes = D_minimized.copy()
    
    return D, equivalence_classes, minimized_T, class_termination, D_classes

def bisimulation_distance_matrix_cached(T, Term, tol=1e-6, max_iter=100):
    """
    Same as bisimulation_distance_matrix, but caches each Wasserstein solve between two class-level distributions.
    """
    n = len(T)

    # 1) minimize system
    R = refine_relation({(i,i) for i in range(n)}, T, Term)
    R_mat = np.zeros((n,n),int)
    for i,j in R:
        R_mat[i,j] = 1
    eq_classes, _, class_term = compute_equivalence_classes(R_mat, n, Term)
    Tmin, _ = compute_minimized_transition_matrix(T, eq_classes, {})

    k = len(eq_classes)
    # init class distances
    Dk = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            if class_term[i] != class_term[j]:
                Dk[i,j] = 1.0

    # a simple cache: maps ((p_tuple),(q_tuple)) -> (distance, coupling_matrix)
    coupling_cache = {}

    def cached_wasserstein(p, q, C):
        # round probabilities to avoid float‐hash issues
        key = (tuple(np.round(p,8)), tuple(np.round(q,8)))
        # exploit symmetry: W(p,q)=W(q,p) if C symmetric
        if key not in coupling_cache and key[::-1] in coupling_cache:
            key = key[::-1]
        if key not in coupling_cache:
            dist, coup = wasserstein_distance(p, q, C)
            coupling_cache[key] = (dist, coup)
        return coupling_cache[key]

    # 2) iterative refinement on classes
    for _ in range(max_iter):
        D_prev = Dk.copy()
        for i in range(k):
            for j in range(k):
                if class_term[i] != class_term[j]:
                    continue
                p = Tmin[i]
                q = Tmin[j]
                C = D_prev  # flattened inside wasserstein_distance
                dist, _ = cached_wasserstein(p, q, C)
                Dk[i,j] = dist

        if np.max(np.abs(Dk - D_prev)) < tol:
            break

    # expand back to states...
    D = np.zeros((n,n))
    state_to_cls = {s:c for c,sts in eq_classes.items() for s in sts}
    for u in range(n):
        for v in range(n):
            D[u,v] = Dk[state_to_cls[u], state_to_cls[v]]

    return D, eq_classes, Tmin, class_term, Dk

def generate_graphviz_source(T, Term, labels, is_minimized=False):
    """
    Generate a Graphviz DOT source string for a probabilistic transition system (PTS).
    Args:
        T: np.ndarray, transition matrix.
        Term: np.ndarray, termination vector.
        labels: dict, transition labels (optional).
        is_minimized: bool, whether to use 'Class' or 'State' as node prefix.
    Returns:
        str: Graphviz DOT source representing the PTS.
    """
    prefix = "Class" if is_minimized else "State"
    dot = Digraph(format='png')
    # Add nodes with different styles for terminating/non-terminating
    for i in range(len(T)):
        if Term[i]:
            dot.node(f"{prefix} {i+1}", f"{prefix} {i+1}", shape='circle', style='filled', peripheries='2', color='lightblue')
        else:
            dot.node(f"{prefix} {i+1}", f"{prefix} {i+1}", shape='circle', style='filled', color='lightgreen')
    # Add edges for all nonzero transitions, including labels if present
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

def wasserstein_distance(p, q, C):
    """
    Compute the Wasserstein (earth mover's) distance between two distributions p and q with cost matrix C.
    Args:
        p: np.ndarray, source probability vector.
        q: np.ndarray, target probability vector.
        C: np.ndarray, cost matrix.
    Returns:
        distance: float, optimal transport cost.
        coupling: np.ndarray, optimal transport plan.
    Raises:
        ValueError: If the LP does not converge.
    """
    # Solve the optimal transport problem as a linear program; the coupling matrix gives the transport plan
    # The LP minimizes total cost c^T x subject to:
    #   - Each row of the coupling sums to p (source marginals)
    #   - Each column of the coupling sums to q (target marginals)
    #   - All entries are non-negative
    n, m = len(p), len(q)
    c = C.flatten()  # Cost vector for the LP
    A_eq = []
    b_eq = []

    # Row constraints: sum of each row in the coupling equals p[i]
    for i in range(n):
        row = np.zeros(n * m)
        row[i * m:(i + 1) * m] = 1
        A_eq.append(row)
        b_eq.append(p[i])

    # Column constraints: sum of each column in the coupling equals q[j]
    for j in range(m):
        col = np.zeros(n * m)
        for i in range(n):
            col[i * m + j] = 1
        A_eq.append(col)
        b_eq.append(q[j])

    bounds = [(0, None)] * (n * m)  # All transport variables must be non-negative
    # Use scipy.optimize.linprog to solve the LP efficiently
    # 'highs' is a modern, fast LP solver
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.success:
        # The optimal coupling matrix (transport plan) tells how much mass to move from p[i] to q[j]
        coupling = res.x.reshape((n, m))
        return res.fun, coupling
    else:
        raise ValueError("Wasserstein LP did not converge: " + res.message)

