import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
from bisimdistance import (
    input_probabilistic_transition_system,
    refine_relation,
    compute_equivalence_classes,
    compute_minimized_transition_matrix,
    bisimulation_distance_matrix,
    generate_graphviz_source,
    analyze_state_differences
)
import pandas as pd
import time

# Create necessary directories if they don't exist
Path("txt").mkdir(exist_ok=True)
Path("images").mkdir(exist_ok=True)

st.set_page_config(
    page_title="PTS Bisimulation Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .stSelectbox>div>div>select {
        border-radius: 5px;
    }
    .stRadio>div {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
    }
    .stExpander {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Probabilistic Bisimulation Tool")

# Add a description
st.markdown("""
    This tool helps you analyze and visualize probabilistic transition systems using bisimulation.
    Upload a file or manually input your transition matrix to get started.
""")

# Sidebar input controls
with st.sidebar:
    st.title("⚙️ Input Settings")
    st.markdown("---")
    input_mode = st.radio("Select Input Mode:", ["Upload File", "Manual Input", "Benchmark Datasets"])

T = Term = labels = None
all_labels_filled = True  # Ensure initialized

if input_mode == "Benchmark Datasets":
    st.sidebar.markdown("### 📚 Benchmark Datasets")
    dataset_type = st.sidebar.radio("Choose dataset type:", ["Pre-built Systems", "Random Generator"])
    
    # --- Pre-built Benchmark Systems ---
    benchmark_systems = [
        {
            "name": "Three-State Markov Chain (No Termination)",
            "matrix": [
                [0, 1, 0],
                [0, 0.5, 0.5],
                [1/3, 0, 2/3]
            ],
            "termination": [0, 0, 0],
            "labels": {},
            "description": (
                "This simple example (from Kemeny et al. 1966) has 3 states and no termination (each state's outgoing probabilities sum to 1). "
                "State 1 always transitions to State 2; State 2 stays in 2 or goes to 3 with equal probability; "
                "State 3 stays in 3 with probability 2/3 or goes to 1 with probability 1/3."
            ),
            "citation": ("Kemeny et al., as quoted in a Markov chains tutorial", "https://www.ssc.wisc.edu/~jmontgom/markovchains.pdf")
        },
        {
            "name": "Six-State Absorbing Random Walk (Gambler's Ruin)",
            "matrix": [
                [1, 0, 0, 0, 0, 0],
                [0.5, 0, 0.5, 0, 0, 0],
                [0, 0.5, 0, 0.5, 0, 0],
                [0, 0, 0.5, 0, 0.5, 0],
                [0, 0, 0, 0.5, 0, 0.5],
                [0, 0, 0, 0, 0, 1]
            ],
            "termination": [1, 0, 0, 0, 0, 1],
            "labels": {},
            "description": (
                "This 6-state PTS is a small gambler's ruin random walk with two absorbing termination states. "
                "States 1–4 move 'left' or 'right' with equal probability 0.5 toward the absorbing boundaries."
            ),
            "citation": ("Topics in Probability blog (absorbing chain example)", "https://probabilitytopics.wordpress.com/2018/01/08/absorbing-markov-chains/")
        },
        {
            "name": "Eight-State Fair Die Simulation (Knuth–Yao Coin Flip Model)",
            "matrix": [
                [0, 0.5, 0.5, 0, 0, 0, 0, 0],
                [0, 0, 0, 0.5, 0.5, 0, 0, 0],
                [0, 0, 0, 0, 0, 0.5, 0.5, 0],
                [0.5, 0, 0, 0, 0, 0, 0, 0.5],
                [0, 0.5, 0, 0, 0, 0, 0, 0.5],
                [0, 0, 0, 0, 0, 0, 0, 1.0],
                [0, 0.5, 0, 0, 0, 0, 0, 0.5],
                [0, 0, 0, 0, 0, 0, 0, 1.0]
            ],
            "termination": [0, 0, 0, 0, 0, 0, 0, 1],
            "labels": {},
            "description": (
                "This example comes from a PRISM model of Knuth & Yao's fair die simulation. "
                "State 0 is the start; states 1–6 represent intermediate coin-flip outcomes; "
                "state 7 is an absorbing termination state where the die value is decided."
            ),
            "citation": ("PRISM case study (Knuth–Yao die model)", "https://www.prismmodelchecker.org/casestudies/dice.php")
        }
    ]
    
    if dataset_type == "Pre-built Systems":
        st.sidebar.markdown("#### Select a pre-built system:")
        system_names = ["Select a system..."] + [sys["name"] for sys in benchmark_systems]
        selected_idx = st.sidebar.selectbox(
            "Choose a system:",
            list(range(len(system_names))),
            format_func=lambda i: system_names[i],
            help="Select a pre-built probabilistic transition system from the literature"
        )
        if selected_idx == 0:
            # No system selected yet; do not load or display anything
            T = Term = labels = None
            all_labels_filled = False
            st.sidebar.info("Please select a system to load its details.")
        else:
            selected_system = benchmark_systems[selected_idx - 1]
            st.sidebar.info(selected_system["description"])
            st.sidebar.markdown(
                f"**Source:** <a href='{selected_system['citation'][1]}' target='_blank'>{selected_system['citation'][0]}</a>",
                unsafe_allow_html=True
            )
            T = np.array(selected_system["matrix"], dtype=float)
            Term = np.array(selected_system["termination"], dtype=int)
            labels = selected_system["labels"]
            all_labels_filled = True
    else:  # Random Generation
        st.sidebar.markdown("#### Generate Random System")
        num_states = st.sidebar.number_input(
            "Number of States",
            min_value=2,
            max_value=10,
            value=4,
            help="Choose how many states your random system should have"
        )
        
        num_terminating = st.sidebar.number_input(
            "Number of Terminating States",
            min_value=0,
            max_value=num_states,
            value=1,
            help="Choose how many states should be terminating"
        )
        
        include_labels = st.sidebar.checkbox(
            "Include Transition Labels",
            value=False,
            help="If unchecked, generates a pure Markov chain without labels"
        )
        
        if st.sidebar.button("Generate Random System"):
            try:
                # Generate random transition matrix
                T = np.random.dirichlet(np.ones(num_states), size=num_states)
                
                # Generate random terminating states
                Term = np.zeros(num_states, dtype=int)
                terminating_indices = np.random.choice(num_states, num_terminating, replace=False)
                Term[terminating_indices] = 1
                
                # For terminating states, set their outgoing transitions to 0
                for i in terminating_indices:
                    T[i] = np.zeros(num_states)
                
                # Generate labels only if requested
                labels = {}
                if include_labels:
                    for i in range(num_states):
                        if not Term[i]:  # Only process non-terminating states
                            for j in range(num_states):
                                if T[i,j] > 0:
                                    labels[(i,j)] = f"a{i}{j}"
                
                st.success("✅ Random system generated successfully!")
                all_labels_filled = True
            except Exception as e:
                st.error(f"❌ Error generating random system: {str(e)}")
                T = Term = labels = None
                all_labels_filled = False

elif input_mode == "Upload File":
    uploaded_file = st.file_uploader("📁 Upload your transition matrix file (.txt)", type="txt")
    if uploaded_file:
        content = uploaded_file.read().decode("utf-8")
        temp_path = os.path.join("txt", "temp_input.txt")
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(content)
        try:
            T, Term, labels = input_probabilistic_transition_system(filename=temp_path, use_file=True)
            st.success("✅ File successfully loaded and parsed.")
            # Set session state to indicate file was uploaded
            st.session_state.file_uploaded = True
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            st.session_state.file_uploaded = False
    else:
        st.session_state.file_uploaded = False

    # Display help section for file upload
    with st.expander("📝 File Format Help", expanded=not st.session_state.get('file_uploaded', False)):
        st.markdown("""
        ### Expected File Format
        Your input file should follow this format:
        
        1. First line: Number of states (n)
        2. Next n lines: Transition matrix (each row must sum to 1)
        3. Next n lines: Terminating states (0 or 1)
        4. Remaining lines: Transition labels (optional)
        
        ### Example:
        ```
        3
        0.5 0.3 0.2
        0.2 0.6 0.2
        0.1 0.4 0.5
        0
        1
        0
        1 2 a
        2 3 b
        ```
        
        ### Notes:
        - Each row of the transition matrix must sum to 1
        - Terminating states are marked with 1, non-terminating with 0
        - Transition labels are optional and in format: `from_state to_state label`
        - States are numbered starting from 1 in the file
        """)
    

elif input_mode == "Manual Input":
    with st.sidebar:
        st.markdown("### 🛠️ Define Transition Matrix")
        n = st.number_input("Number of states", min_value=2, max_value=15, step=1)
        matrix = []
        valid = True

    # Create columns for matrix input
    cols = st.columns(min(n, 3))  # Show max 3 columns at a time
    for i in range(n):
        col_idx = i % len(cols)
        with cols[col_idx]:
            if n > 1:
                base = round(1.0/n, 2)
                first = round(1.0 - (n-1)*base, 2)
                placeholder = "e.g. " + " ".join([f"{first:.2f}"] + [f"{base:.2f}"]*(n-1))
            else:
                placeholder = "e.g. 1.00"
            row_input = st.text_input(
                f"State {i+1} transitions",
                key=f"row_{i}",
                placeholder=placeholder
            )
            try:
                row_vals = list(map(float, row_input.strip().split()))
                if len(row_vals) != n or not np.isclose(sum(row_vals), 1.0):
                    raise ValueError()
                matrix.append(row_vals)
            except:
                st.error(f"Row {i+1} must have {n} numbers summing to 1.")
                valid = False

    with st.sidebar:
        st.markdown("### 🎯 Terminating States")
        Term = [st.selectbox(f"State {i+1}", [0, 1], key=f"term_{i}") for i in range(n)]

    # Only show label inputs and create button if matrix and Term are valid
    if valid:
        T = None
        labels = {}
        st.markdown("### 🏷️ Transition Labels (Optional)")
        label_cols = st.columns(min(n, 3))
        for i in range(n):
            for j in range(n):
                if matrix[i][j] > 0:
                    col_idx = (i * n + j) % len(label_cols)
                    with label_cols[col_idx]:
                        label = st.text_input(f"S{i+1} → S{j+1}", key=f"label_{i}_{j}")
                        if label:
                            labels[(i, j)] = label
        st.warning("⚠️ You may optionally fill in transition labels. Leave blank for unlabeled transitions.")
        # Add Create System button
        if st.button("Create System"):
            T = np.array(matrix)
            Term = np.array(Term)
            st.success("✅ Manual input accepted. System created.")
            all_labels_filled = True
        else:
            T = None
            Term = None
            labels = None
            all_labels_filled = False

# ---- Main Calculation ---- #
if T is not None and Term is not None and (input_mode == "Upload File" or input_mode == "Manual Input" or (input_mode == "Benchmark Datasets" and all_labels_filled)):
    try:
        # --- Distance computation timing ---
        start_dist = time.time()
        D = bisimulation_distance_matrix(T, Term)
        end_dist = time.time()
        dist_time = end_dist - start_dist

        # --- Minimization timing ---
        start_min = time.time()
        R_0 = {(x, y) for x in range(len(T)) for y in range(len(T))}
        R_n = refine_relation(R_0, T, Term)
        equivalence_classes, state_class_map, class_termination_status = compute_equivalence_classes(R_n, len(T), Term)
        minimized_T, minimized_labels = compute_minimized_transition_matrix(T, equivalence_classes, state_class_map, labels)
        end_min = time.time()
        min_time = end_min - start_min

        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "📏 Distance Analysis", 
            "🔄 Minimization",
            "📈 System Analysis", 
            "📚 Theory"
        ])
        
        with tab1:
            st.markdown("### Distance Analysis")
            st.info(f"Distance computation time: {dist_time:.2f} seconds")
            
            st.markdown("#### Distance Matrix")
            with st.expander("Show Distance Table", expanded=True):
                df = pd.DataFrame(np.round(D, 3), 
                                index=[f"State {i+1}" for i in range(len(D))],
                                columns=[f"State {i+1}" for i in range(len(D))])
                st.dataframe(df)

            st.markdown("#### Distance Heatmap")
            with st.expander("Show Distance Heatmap", expanded=True):
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(D, annot=True, cmap="YlOrRd", fmt=".3f",
                            xticklabels=[f"State {i+1}" for i in range(len(D))],
                            yticklabels=[f"State {i+1}" for i in range(len(D))],
                            annot_kws={"size": 8})
                ax.tick_params(axis='both', which='major', labelsize=8)
                plt.title("Bisimulation Distance Heatmap", pad=8, size=12)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                
                st.markdown("### 🔍 Analyze State Differences")
                st.markdown("""
                This section helps you understand why any two states behave differently. Select two states below to see:
                - Their overall distance
                - Whether they terminate differently
                - Which differences contribute most to their distance
                """)
                
                state_options = ["Select a state..."] + [f"State {i+1}" for i in range(len(D))]
                col1, col2 = st.columns(2)
                with col1:
                    state1 = st.selectbox("Select first state", state_options, key=f"state1_{len(D)}")
                with col2:
                    state2 = st.selectbox("Select second state", state_options, key=f"state2_{len(D)}")
                
                compare_clicked = st.button("Compare", key=f"compare_{len(D)}")

                if compare_clicked:
                    if state1 != "Select a state..." and state2 != "Select a state...":
                        idx1 = int(state1.split()[1]) - 1
                        idx2 = int(state2.split()[1]) - 1
                        
                        explanations = analyze_state_differences(idx1, idx2, T, Term, D)
                        
                        st.markdown(f"#### Distance between {state1} and {state2}: {D[idx1, idx2]:.3f}")
                        
                        if idx1 == idx2:
                            st.info(f"You're comparing {state1} with itself! The distance is always 0 because:")
                            st.markdown("""
                            - It's the same state
                            - States always have identical behavior to themselves
                            - This is called the reflexive property of bisimulation
                            """)
                        elif D[idx1, idx2] == 0:
                            st.success(f"These states are bisimilar (identical behavior)! They have:")
                            st.markdown("""
                            - The same termination behavior
                            - Identical transition probabilities to all states
                            - No behavioral differences
                            """)
                        else:
                            st.markdown("##### Why these states differ:")
                            for line in explanations:
                                st.markdown(f"- {line}")
                            # footnote if more than 3 transitions exist
                            total_diffs = sum(
                                1 for j in range(len(T))
                                if not np.isclose(T[idx1,j], T[idx2,j])
                            )
                            if total_diffs > 0 and D[idx1,idx2] < 1.0:
                                st.markdown("*Note: Only the top 3 contributing transitions are shown here for clarity.*")
                                                        # termination‐only note
                            if explanations and explanations[0].startswith("Termination mismatch") and D[idx1,idx2] == 1.0:
                                st.markdown("*Note: This alone contributes the full distance of 1.0*")
                    else:
                        st.info("Please select two states to compare.")

            st.markdown("#### 📊 Metrics Breakdown")
            # Add state analysis in expanders
            with st.expander("🔍 State Analysis", expanded=True):
                st.markdown("""
                This section shows you the most similar and most different states in your system.
                - **Most Similar States**: These states behave the most similar
                - **Most Different States**: These states have the most different behavior                
                """)
                
                # Find all pairs with minimum and maximum distances
                D_copy = D.copy()  # Create a copy to avoid modifying the original
                np.fill_diagonal(D_copy, np.inf)  # Exclude self-comparisons for minimum distance
                min_distance = np.min(D_copy)
                
                # For maximum distance, use the original matrix
                max_distance = np.max(D)
                
                # Find all pairs with minimum distance and filter duplicates
                min_pairs = np.where(np.abs(D_copy - min_distance) < 1e-10)
                min_pairs = list(zip(min_pairs[0], min_pairs[1]))
                # Filter out duplicate pairs (e.g., if (1,2) exists, remove (2,1))
                min_pairs = [(s1, s2) for s1, s2 in min_pairs if s1 < s2]
                
                # Find all pairs with maximum distance and filter duplicates
                max_pairs = np.where(np.abs(D - max_distance) < 1e-10)
                max_pairs = list(zip(max_pairs[0], max_pairs[1]))
                # Filter out duplicate pairs
                max_pairs = [(s1, s2) for s1, s2 in max_pairs if s1 < s2]
                
                # Use tabs for similar and different states
                similar_tab, different_tab = st.tabs([
                    f"🎯 Most Similar States ({len(min_pairs)} pairs)",
                    f"🎯 Most Different States ({len(max_pairs)} pairs)"
                ])
                
                with similar_tab:
                    if len(min_pairs) == 1:
                        st.success(f"States S{min_pairs[0][0]+1} and S{min_pairs[0][1]+1} are most similar with distance {min_distance:.3f}")
                        # Add explanation for this pair
                        explanations = analyze_state_differences(min_pairs[0][0], min_pairs[0][1], T, Term, D)
                        st.markdown("##### Why these states are similar:")
                        for explanation in explanations[:3]:  # Show only top 3
                            st.write(explanation)
                        st.markdown("*Note: Only the top 3 contributing transitions are shown here for clarity.*")
                    else:
                        st.success(f"Found {len(min_pairs)} pairs of most similar states (distance: {min_distance:.3f})")
                        # Create a table for similar states
                        similar_data = {
                            "Pair": [f"{i+1}" for i in range(len(min_pairs))],
                            "State 1": [f"S{s1+1}" for s1, _ in min_pairs],
                            "State 2": [f"S{s2+1}" for _, s2 in min_pairs],
                            "Distance": [f"{min_distance:.3f}" for _ in min_pairs]
                        }
                        similar_df = pd.DataFrame(similar_data)
                        st.dataframe(
                            similar_df,
                            hide_index=True,
                            use_container_width=True
                        )
                
                with different_tab:
                    if len(max_pairs) == 1:
                        st.error(f"States S{max_pairs[0][0]+1} and S{max_pairs[0][1]+1} are most different with distance {max_distance:.3f}")
                        # Add explanation for this pair
                        explanations = analyze_state_differences(max_pairs[0][0], max_pairs[0][1], T, Term, D)
                        st.markdown("##### Why these states differ:")
                        for explanation in explanations[:3]:  # Show only top 3
                            st.write(explanation)
                        st.markdown("*Note: Only the top 3 contributing transitions are shown here for clarity.*")
                    else:
                        st.error(f"Found {len(max_pairs)} pairs of most different states (distance: {max_distance:.3f})")
                        # Create a table for different states
                        different_data = {
                            "Pair": [f"{i+1}" for i in range(len(max_pairs))],
                            "First State Pair": [f"S{s1+1}" for s1, _ in max_pairs],
                            "Second State Pair": [f"S{s2+1}" for _, s2 in max_pairs],
                            "Distance": [f"{max_distance:.3f}" for _ in max_pairs]
                        }
                        different_df = pd.DataFrame(different_data)
                        st.dataframe(
                            different_df,
                            hide_index=True,
                            use_container_width=True
                        )
            with st.expander("Show Distance Metrics", expanded=True):
                # Calculate basic statistics
                D_flat = D[np.triu_indices_from(D, k=1)]  # Get upper triangle excluding diagonal
                min_dist = np.min(D_flat)
                max_dist = np.max(D_flat)
                mean_dist = np.mean(D_flat)
                median_dist = np.median(D_flat)
                std_dist = np.std(D_flat)
                
                # Calculate proportions
                n = len(D)
                total_pairs = n * (n-1) / 2  # Number of unique state pairs
                zero_pairs = np.sum(D_flat == 0)
                zero_prop = zero_pairs / total_pairs
                
                # Display basic statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Minimum Distance",
                        f"{min_dist:.3f}",
                        help="The smallest distance between any two states. A value of 0 indicates exactly bisimilar states."
                    )
                    st.metric(
                        "Mean Distance",
                        f"{mean_dist:.3f}",
                        help="The average distance between all state pairs. Indicates the overall level of behavioral similarity in the system."
                    )
                    st.metric(
                        "Standard Deviation",
                        f"{std_dist:.3f}",
                        help="Measures how spread out the distances are. A high value indicates states have very different behaviors, while a low value suggests more uniform behavior."
                    )
                with col2:
                    st.metric(
                        "Maximum Distance",
                        f"{max_dist:.3f}",
                        help="The largest distance between any two states. Indicates the most behaviorally different states in the system."
                    )
                    st.metric(
                        "Median Distance",
                        f"{median_dist:.3f}",
                        help="The middle value of all distances. Less affected by outliers than the mean, giving a better sense of typical state differences."
                    )
                    st.metric(
                        "Zero Distance Pairs",
                        f"{zero_pairs} ({zero_prop:.1%})",
                        help="Number and proportion of state pairs that are exactly bisimilar (distance = 0). These states have identical behavior."
                    )
                
                # Interactive ε threshold analysis
                st.markdown("### 📊 Proportion Below Threshold Analysis")
                
                # Add threshold interpretation guide
                st.markdown("""
                #### ℹ️ Understanding Threshold Values
                The threshold value helps you analyze how similar or different states are. Here's a general guide:
                
                - **0.0**: States are exactly bisimilar (identical behavior)
                - **0.0 - 0.1**: States are very similar in behavior
                - **0.1 - 0.3**: States show moderate similarity
                - **0.3 - 0.5**: States show moderate difference
                - **0.5 - 0.8**: States are quite different
                - **0.8 - 1.0**: States are very different in behavior
                
                Note: These ranges are general guidelines. The actual interpretation may vary depending on your specific system.
                """)
                
                epsilon = st.slider(
                    "Select distance threshold",
                    min_value=0.0,
                    max_value=float(max_dist),
                    value=0.1,
                    step=0.01,
                    help="States with distance ≤ threshold are grouped together. This helps analyze how many state pairs fall within different distance ranges."
                )
                
                # Calculate proportion below ε
                below_epsilon = np.sum(D_flat <= epsilon)
                below_epsilon_prop = below_epsilon / total_pairs
                
                # Display ε analysis
                st.markdown(f"""
                #### Analysis for threshold = {epsilon:.3f}
                - **Exactly Bisimilar States** (distance = 0): {zero_pairs} pairs ({zero_prop:.1%})
                - **States Below Threshold** (distance ≤ {epsilon:.3f}): {below_epsilon} pairs ({below_epsilon_prop:.1%})
                - **States Above Threshold** (distance > {epsilon:.3f}): {total_pairs - below_epsilon} pairs ({(1 - below_epsilon_prop):.1%})
                """)
                
                # Visualize distribution
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(D_flat, bins=20, ax=ax)
                ax.axvline(x=epsilon, color='r', linestyle='--', label=f'Threshold = {epsilon:.3f}')
                ax.axvline(x=0, color='g', linestyle='--', label='Exactly Bisimilar')
                ax.set_title('Distribution of State Distances')
                ax.set_xlabel('Distance')
                ax.set_ylabel('Number of State Pairs')
                ax.legend()
                st.pyplot(fig)


        with tab2:
            st.markdown("### 🧠 Probabilistic Transition System Comparison")
            st.info(f"Bisimulation minimization time: {min_time:.2f} seconds")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎨 Original PTS")
                if not labels:
                    graphviz_src = f"""
digraph G {{
  rankdir=LR;
  node [shape=circle, style=filled];
  {{
    {chr(10).join([
        f'"{i}" [label="State {i+1}", ' +
        ("color=lightblue, peripheries=2" if Term[i] else "color=lightgreen") + "]" for i in range(len(T))
    ])}
  }}
  {chr(10).join([
      f'"{i}" -> "{j}" [label="{T[i,j]:.2f}"]' for i in range(len(T)) for j in range(len(T)) if T[i,j] > 0 and not Term[i]
  ])}
}}
"""
                else:
                    graphviz_src = generate_graphviz_source(T, Term, labels, is_minimized=False)
                st.graphviz_chart(graphviz_src)
            
            with col2:
                st.markdown("#### 🎨 Minimized PTS")
                if not minimized_labels:
                    minimized_graphviz_src = f"""
digraph G {{
  rankdir=LR;
  node [shape=circle, style=filled];
  {{
    {chr(10).join([
        f'"{i}" [label="Class {i+1}", ' +
        ("color=lightblue, peripheries=2" if list(class_termination_status.values())[i] else "color=lightgreen") + "]" for i in range(len(minimized_T))
    ])}
  }}
  {chr(10).join([
      f'"{i}" -> "{j}" [label="{minimized_T[i,j]:.2f}"]' for i in range(len(minimized_T)) for j in range(len(minimized_T)) if minimized_T[i,j] > 0 and not list(class_termination_status.values())[i]
  ])}
}}
"""
                else:
                    minimized_graphviz_src = generate_graphviz_source(
                        minimized_T, 
                        list(class_termination_status.values()), 
                        minimized_labels,
                        is_minimized=True
                    )
                st.graphviz_chart(minimized_graphviz_src)

            # Add bisimulation statistics in an expander
            with st.expander("📊 Bisimulation Statistics", expanded=True):
                # Calculate statistics
                num_classes = len(equivalence_classes)
                num_states = len(T)
                compression_ratio = num_classes / num_states
                
                # Calculate class size statistics
                class_sizes = [len(states) for states in equivalence_classes.values()]
                min_size = min(class_sizes)
                max_size = max(class_sizes)
                mean_size = sum(class_sizes) / len(class_sizes)
                median_size = sorted(class_sizes)[len(class_sizes) // 2]
                
                # Create columns for statistics display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Number of Equivalence Classes",
                        f"{num_classes}",
                        help="Total number of equivalence classes after minimization"
                    )
                    st.metric(
                        "Compression Ratio",
                        f"{compression_ratio:.2%}",
                        help="Ratio of classes to original states (|Classes| / |S|)"
                    )
                
                with col2:
                    st.markdown("#### Distribution of Class Sizes")
                    
                    # Create metrics in a grid layout
                    metrics_col1, metrics_col2 = st.columns(2)
                    
                    with metrics_col1:
                        st.metric(
                            "Smallest Class Size",
                            min_size,
                            help="The minimum number of states in any equivalence class"
                        )
                        st.metric(
                            "Largest Class Size",
                            max_size,
                            help="The maximum number of states in any equivalence class"
                        )
                    
                    with metrics_col2:
                        st.metric(
                            "Average Class Size",
                            f"{mean_size:.2f}",
                            help="The mean number of states across all equivalence classes"
                        )
                        st.metric(
                            "Typical Class Size",
                            median_size,
                            help="The middle value of class sizes (half of classes are smaller, half are larger)"
                        )
                    
                    # Add interpretation
                    if max_size > 2 * mean_size:
                        st.info("The system has some large equivalence classes with many singletons")
                    else:
                        st.info("The equivalence classes are relatively uniform in size")

            # Display equivalence classes in an expander
            with st.expander("📑 Equivalence Classes", expanded=False):
                # Group classes by termination status
                terminating_classes = []
                non_terminating_classes = []
                
                for class_id, class_states in equivalence_classes.items():
                    # Convert state numbers to 1-based indexing
                    states_1based = [s + 1 for s in class_states]
                    # Format states list based on number of states
                    if len(states_1based) == 1:
                        states_str = f"state {states_1based[0]}"
                    else:
                        states_str = ", ".join(f"state {s}" for s in states_1based[:-1]) + f" & state {states_1based[-1]}"
                    
                    class_info = {
                        'id': class_id + 1,
                        'states': states_str,
                        'terminating': class_termination_status[class_id]
                    }
                    
                    if class_termination_status[class_id]:
                        terminating_classes.append(class_info)
                    else:
                        non_terminating_classes.append(class_info)
                
                # Use tabs instead of nested expanders
                term_tab, non_term_tab = st.tabs([
                    f"🔴 Terminating Classes ({len(terminating_classes)})",
                    f"⚪ Non-Terminating Classes ({len(non_terminating_classes)})"
                ])
                
                with term_tab:
                    for class_info in terminating_classes:
                        st.markdown(f"""
                        <div style='padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin: 5px 0;'>
                            <strong>Class {class_info['id']}</strong>: {class_info['states']}
                        </div>
                        """, unsafe_allow_html=True)
                
                with non_term_tab:
                    for class_info in non_terminating_classes:
                        st.markdown(f"""
                        <div style='padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin: 5px 0;'>
                            <strong>Class {class_info['id']}</strong>: {class_info['states']}
                        </div>
                        """, unsafe_allow_html=True)

            # Display minimized transition matrix in an expander
            with st.expander("📊 Minimized Transition Matrix", expanded=False):
                # Create a DataFrame with 1-based indexing
                minimized_df = pd.DataFrame(np.round(minimized_T, 3),
                                         index=[f"Class {i+1}" for i in range(len(minimized_T))],
                                         columns=[f"Class {i+1}" for i in range(len(minimized_T))])
                st.dataframe(minimized_df)

        with tab3:
            st.markdown("### 📊 Summary Statistics")
            
            # Calculate metrics
            num_states = len(T)
            num_terminating = sum(Term)
            # Only count transitions from non-terminating states
            num_transitions = sum(sum(row > 0) for i, row in enumerate(T) if not Term[i])
            avg_transitions = num_transitions / (num_states - num_terminating)  # Average per non-terminating state
            sparsity = 1 - (num_transitions / ((num_states - num_terminating) * num_states))  # Sparsity excluding terminating states
            
            # Create metrics in a grid layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Number of States",
                    f"{num_states}",
                    help="Total number of states in the system"
                )
                st.metric(
                    "Terminating States",
                    f"{num_terminating}",
                    f"{num_terminating/num_states:.1%} of total",
                    help="Number of states that terminate"
                )
            
            with col2:
                st.metric(
                    "Total Transitions",
                    f"{num_transitions}",
                    help="Total number of non-zero transitions"
                )
                st.metric(
                    "Avg. Outgoing Transitions",
                    f"{avg_transitions:.2f}",
                    help="Average number of outgoing transitions per state"
                )
            
            with col3:
                st.metric(
                    "Transition Sparsity",
                    f"{sparsity:.1%}",
                    help="Percentage of zero transitions in the matrix"
                )
                st.metric(
                    "Density",
                    f"{1-sparsity:.1%}",
                    help="Percentage of non-zero transitions in the matrix"
                )

            # Add a visual representation of the metrics
            st.markdown("### 📈 System Characteristics")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Pie chart for state types
            ax1.pie([num_terminating, num_states - num_terminating],
                   labels=['Terminating', 'Non-terminating'],
                   autopct='%1.1f%%',
                   colors=['lightblue', 'lightgreen'])
            ax1.set_title('State Types Distribution')
            
            # Bar chart for transition metrics
            metrics = ['Total States', 'Terminating States', 'Total Transitions', 'Avg. Transitions']
            values = [num_states, num_terminating, num_transitions, avg_transitions]
            ax2.bar(metrics, values, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'])
            ax2.set_title('System Metrics')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)

            # Move Comparative Discrepancy Metrics to the end in an expander
            with st.expander("Show Comparative Discrepancy Metrics", expanded=False):
                st.markdown("#### ▶️ Comparative Discrepancy Metrics")
                n = len(T)
                # Compute all pairwise metrics (excluding diagonal)
                eucl_dists = []
                kl_dists = []
                bisim_dists = []
                for i in range(n):
                    for j in range(i+1, n):
                        # Euclidean distance between transition rows
                        eucl = np.linalg.norm(T[i] - T[j])
                        eucl_dists.append(eucl)
                        # KL divergence (symmetrized, add small epsilon to avoid log(0))
                        eps = 1e-12
                        p, q = T[i] + eps, T[j] + eps
                        kl1 = np.sum(p * np.log(p / q))
                        kl2 = np.sum(q * np.log(q / p))
                        kl = 0.5 * (kl1 + kl2)
                        kl_dists.append(kl)
                        # Bisimulation distance
                        bisim_dists.append(D[i, j])
                mean_eucl = np.mean(eucl_dists)
                mean_kl = np.mean(kl_dists)
                mean_bisim = np.mean(bisim_dists)
                st.markdown("""
| Metric                | Mean over all state-pairs |
|-----------------------|--------------------------:|
| Euclidean distance    | {:.3f}                   |
| KL-Divergence         | {:.3f}                   |
| Bisimulation distance | {:.3f}                   |
""".format(mean_eucl, mean_kl, mean_bisim))
                st.markdown("""
These metrics capture different aspects of state similarity. Euclidean and KL-divergence measure direct differences in transition probabilities, while bisimulation distance accounts for structural equivalence. For example, some state pairs may have high Euclidean distance but zero bisimulation distance, indicating they are structurally equivalent despite differing transitions. Conversely, bisimulation distance highlights behavioral differences that other metrics may miss.
""")
                # Bar chart
                st.markdown("##### Mean Metric Comparison")
                fig, ax = plt.subplots(figsize=(5, 3))
                metrics = ['Euclidean', 'KL-Div', 'Bisimulation']
                means = [mean_eucl, mean_kl, mean_bisim]
                sns.barplot(x=metrics, y=means, ax=ax, palette='pastel')
                ax.set_ylabel('Mean Distance')
                st.pyplot(fig)
                # Scatter plot
                st.markdown("##### Euclidean vs Bisimulation Distance (all pairs)")
                df_metrics = pd.DataFrame({'Euclidean': eucl_dists, 'Bisimulation': bisim_dists})
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                sns.regplot(x='Euclidean', y='Bisimulation', data=df_metrics, ax=ax2, scatter_kws={'s': 20}, line_kws={'color': 'red'})
                ax2.set_xlabel('Euclidean Distance')
                ax2.set_ylabel('Bisimulation Distance')
                st.pyplot(fig2)

        with tab4:
            st.markdown("## 📚 Theoretical Foundations")
            
            st.markdown("""
            ### Probabilistic Bisimulation
            Probabilistic bisimulation is an equivalence relation that captures behavioral equivalence 
            between states in probabilistic transition systems. Two states are bisimilar if:
            
            1. They have the same termination behavior
            2. For each equivalence class, they have the same probability of transitioning to that class
            
            ### Bisimulation Distance
            The bisimulation distance extends the notion of bisimulation by providing a quantitative 
            measure of how different two states are. It is computed using the Wasserstein metric, 
            which measures the minimum cost of transforming one probability distribution into another.
            
            #### Mathematical Definition
            For states s and t, their bisimulation distance d(s,t) is defined as:
            
            d(s,t) = max{
                |T(s) - T(t)|,
                sup_a min_π ∑(s',t') π(s',t') * d(s',t')
            }
            
            where:
            - T(s) is the termination probability of state s
            - π is a coupling between the probability distributions of s and t
            - a ranges over all actions
            """)
            
            st.markdown("### 🔍 Key Properties")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                #### Bisimulation Properties
                - Reflexivity: s ~ s
                - Symmetry: if s ~ t then t ~ s
                - Transitivity: if s ~ t and t ~ u then s ~ u
                - Compositionality: preserved under parallel composition
                """)
            
            with col2:
                st.markdown("""
                #### Distance Properties
                - Non-negativity: d(s,t) ≥ 0
                - Symmetry: d(s,t) = d(t,s)
                - Triangle inequality: d(s,t) ≤ d(s,u) + d(u,t)
                - Continuity: small changes in probabilities lead to small changes in distance
                """)
            
            st.markdown("### 📊 Interpretation of Results")
            st.markdown("""
            The distance matrix and heatmap show:
            - 0 distance: States are bisimilar
            - Small distance: States are behaviorally similar
            - Large distance: States have significantly different behavior
            
            The minimization process:
            1. Identifies equivalent states
            2. Merges them into equivalence classes
            3. Preserves the behavioral properties
            4. Reduces the system size while maintaining its essential characteristics
            """)

    except Exception as e:
        st.error(f"❌ Computation error: {e}")
