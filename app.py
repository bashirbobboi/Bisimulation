import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
from probisim.bisimdistance import (
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
from probisim.parsers import parse_model  # <-- NEW IMPORT

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

T = st.session_state.get('T')
Term = st.session_state.get('Term')
labels = st.session_state.get('labels')
all_labels_filled = st.session_state.get('all_labels_filled', True)

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
        with st.sidebar.form("rand"):
            num_states = st.number_input(
                "Number of States",
                min_value=2,
                max_value=10,
                value=4,
                help="Choose how many states your random system should have"
            )
            num_terminating = st.number_input(
                "Number of Terminating States",
                min_value=0,
                max_value=num_states,
                value=1,
                help="Choose how many states should be terminating"
            )
            include_labels = st.checkbox(
                "Include Transition Labels",
                value=False,
                help="If unchecked, generates a pure Markov chain without labels"
            )
            submit_rand = st.form_submit_button("Generate Random System")
            if submit_rand:
                try:
                    T = np.random.dirichlet(np.ones(num_states), size=num_states)
                    Term = np.zeros(num_states, dtype=int)
                    terminating_indices = np.random.choice(num_states, num_terminating, replace=False)
                    Term[terminating_indices] = 1
                    for i in terminating_indices:
                        T[i] = np.zeros(num_states)
                    labels = {}
                    if include_labels:
                        for i in range(num_states):
                            if not Term[i]:
                                for j in range(num_states):
                                    if T[i, j] > 0:
                                        labels[(i, j)] = f"a{i}{j}"
                    st.session_state.T = T
                    st.session_state.Term = Term
                    st.session_state.labels = labels
                    st.success("✅ Random system generated successfully!")
                    st.session_state.all_labels_filled = True
                except Exception as e:
                    st.error(f"❌ Error generating random system: {str(e)}")
                    st.session_state.T = None
                    st.session_state.Term = None
                    st.session_state.labels = None
                    st.session_state.all_labels_filled = False

elif input_mode == "Upload File":
    # --- Model format selection ---
    st.markdown("#### Model Format")
    if 'model_format' not in st.session_state:
        st.session_state.model_format = "txt (legacy)"

    model_format = st.selectbox(
        "Choose model format:",
        ["txt (legacy)", "prism", "json"],
        index=["txt (legacy)", "prism", "json"].index(st.session_state.model_format),
        help="Select the format of your uploaded model file. 'txt' is the legacy format; others use the pluggable parser."
    )

    # If the user changes the format, update session state
    if model_format != st.session_state.model_format:
        st.session_state.model_format = model_format
        # Clear the parsed content but not the raw upload
        if 'uploaded_content' in st.session_state:
            del st.session_state['uploaded_content']
        st.rerun()

    # Use a fixed key for the file uploader
    uploaded_file = st.file_uploader(
        "📁 Upload your model file",
        type=["txt", "pm", "json"],
        key="uploaded_file"
    )

    # Handle new file uploads
    if uploaded_file is not None:
        raw = uploaded_file.read().decode()
        # stash it
        st.session_state["uploaded_content"] = raw
        st.session_state["uploaded_name"] = uploaded_file.name

        # pick up extension → target_format
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        newfmt = {".pm": "prism", ".json": "json", ".txt": "txt (legacy)"}.get(ext)
        if newfmt and newfmt != st.session_state.model_format:
            st.session_state.model_format = newfmt
            st.rerun()

    # Process the uploaded content if it exists
    content = st.session_state.get("uploaded_content")
    if content:
        try:
            if st.session_state.model_format == "txt (legacy)":
                temp_path = os.path.join("txt", "temp_input.txt")
                with open(temp_path, "w", encoding="utf-8") as f:
                    f.write(content)
                T, Term, labels = input_probabilistic_transition_system(filename=temp_path, use_file=True)
            else:
                T, Term, labels = parse_model(content, st.session_state.model_format)
            st.success("✅ File successfully loaded and parsed.")
            st.session_state.file_uploaded = True
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            st.session_state.file_uploaded = False
    elif not uploaded_file:
        st.info("Please upload a file in the selected format.")

    # Display help section for file upload
    with st.expander("📝 File Format Help", expanded=not st.session_state.get('file_uploaded', False)):
        st.markdown("""
        ### Supported Model Formats
        - **txt (legacy):**
            - First line: Number of states (n)
            - Next n lines: Transition matrix (each row must sum to 1)
            - Next n lines: Terminating states (0 or 1)
            - Remaining lines: Transition labels (optional)
            - States are 1-based in the file
        - **prism:**
            - PRISM .pm DTMC subset, 0-based states, e.g.:
            ```
            [0] -> 0.5 : (state' = 1) + 0.5 : (state' = 2);
            [1] -> 1.0 : (state' = 0);
            [2] [term];
            ```
            - Use [term] annotation or omit outgoing transitions for termination
        - **json:**
            - JSON LTS, 0-based states, e.g.:
            ```json
            {
              "states": 3,
              "transitions": [
                {"from": 0, "to": 1, "prob": 0.5, "label": "a"},
                {"from": 0, "to": 2, "prob": 0.5}
              ],
              "terminating": [2]
            }
            ```
        """)

elif input_mode == "Manual Input":
    # --- collect raw inputs ---
    n = st.sidebar.number_input("Number of states", 2, 15, 2)
    matrix = []
    valid = True
    for i in range(n):
        row = st.sidebar.text_input(f"State {i+1} transitions", key=f"row_{i}")
        # parse & validate...
        try:
            vals = list(map(float, row.split()))
            assert len(vals)==n and np.isclose(sum(vals),1.0)
            matrix.append(vals)
        except:
            st.sidebar.error(f"Row {i+1} needs {n} numbers summing to 1.")
            valid = False

    Term = [
      1 if st.sidebar.radio(f"State {i+1} terminating?", ["No","Yes"], key=f"term_{i}")=="Yes" else 0
      for i in range(n)
    ]

    # --- only show this when your matrix+Term are valid ---
    if valid:
        # wrap create in its own form
        with st.form("manual_create_form"):
            st.markdown("🏷️ Optional Labels")
            labels = {}
            for i in range(n):
                for j in range(n):
                    if matrix[i][j]>0:
                        lab = st.text_input(f"S{i+1}→S{j+1}", key=f"lab_{i}_{j}")
                        if lab: labels[(i,j)] = lab

            create = st.form_submit_button("Create System")
            if create:
                # persist into session_state
                st.session_state["T_manual"]     = np.array(matrix)
                st.session_state["Term_manual"]  = np.array(Term)
                st.session_state["labels_manual"]= labels
                st.success("✅ System created!")

    # --- read back from session_state ---
    if "T_manual" in st.session_state:
        T      = st.session_state["T_manual"]
        Term   = st.session_state["Term_manual"]
        labels = st.session_state["labels_manual"]
        all_labels_filled = True
    else:
        T = Term = labels = None
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
        n = len(T)
        R_mat = np.zeros((n, n), dtype=int)
        for i, j in R_n:
            R_mat[i, j] = 1
        equivalence_classes, state_class_map, class_termination_status = compute_equivalence_classes(R_mat, n, Term)
        minimized_T, minimized_labels = compute_minimized_transition_matrix(T, equivalence_classes, state_class_map, labels)
        end_min = time.time()
        min_time = end_min - start_min

        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📏 Distance Analysis", 
            "🔄 Minimization",
            "📈 System Analysis", 
            "🎲 Simulation",
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

                # Use a form to group the pickers and button
                with st.form("state_compare_form"):
                    col1, col2 = st.columns(2)
                    with col1:
                        state1 = st.selectbox("Select first state", state_options, key=f"state1_{len(D)}")
                    with col2:
                        state2 = st.selectbox("Select second state", state_options, key=f"state2_{len(D)}")
                    compare_clicked = st.form_submit_button("Compare")

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
            st.markdown("## 🎲 System Simulation")
            
            # Show original PTS visualization
            st.markdown("### 📊 Original Probabilistic Transition System")
            if not labels:
                graphviz_src = f"""
digraph G {{
  rankdir=LR;
  size="4,4";
  node [shape=circle, style=filled, width=0.5, height=0.5];
  {{
    {chr(10).join([
        f'"{i}" [label="State {i+1}", ' +
        ("color=lightblue, peripheries=2" if Term[i] else "color=lightgreen") + "]" for i in range(len(T))
    ])}
  }}
  {chr(10).join([
      f'"{i}" -> "{j}" [label="{T[i,j]:.2f}", fontsize=8]' for i in range(len(T)) for j in range(len(T)) if T[i,j] > 0 and not Term[i]
  ])}
}}
"""
            else:
                graphviz_src = generate_graphviz_source(T, Term, labels, is_minimized=False)
            st.graphviz_chart(graphviz_src, use_container_width=False)
            
            st.markdown("---")
            st.markdown("### 🎮 Simulation Controls")
            
            # Simulation controls
            col1, col2 = st.columns(2)
            with col1:
                initial_state = st.selectbox(
                    "Select Initial State",
                    [f"State {i+1}" for i in range(len(T))],
                    help="Choose the starting state for the simulation"
                )
                max_steps = st.number_input(
                    "Maximum Steps",
                    min_value=1,
                    max_value=1000,
                    value=100,
                    help="Maximum number of steps before stopping the simulation"
                )
            
            with col2:
                num_simulations = st.number_input(
                    "Number of Simulations",
                    min_value=1,
                    max_value=1000,
                    value=100,
                    help="Number of independent simulation runs to perform"
                )
                show_individual_runs = st.checkbox(
                    "Show Individual Runs",
                    value=False,
                    help="Display the sequence of states for each simulation run"
                )
            
            if st.button("Run Simulation"):
                # Convert initial state to 0-based index
                current_state = int(initial_state.split()[1]) - 1
                
                # Store results
                all_runs = []
                steps_to_termination = []
                state_visits = np.zeros(len(T))
                
                # Run simulations
                for sim in range(num_simulations):
                    run = [current_state]
                    steps = 0
                    state = current_state
                    
                    while steps < max_steps:
                        # Count state visit
                        state_visits[state] += 1
                        
                        # Check if terminating state
                        if Term[state]:
                            break
                        
                        # Choose next state based on transition probabilities
                        next_state = np.random.choice(len(T), p=T[state])
                        run.append(next_state)
                        state = next_state
                        steps += 1
                    
                    all_runs.append(run)
                    steps_to_termination.append(steps)
                
                # Display results
                st.markdown("### 📊 Simulation Results")
                
                # Basic statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Average Steps to Termination",
                        f"{np.mean(steps_to_termination):.1f}",
                        help="Average number of steps before reaching a terminating state"
                    )
                with col2:
                    st.metric(
                        "Termination Rate",
                        f"{np.mean([s < max_steps for s in steps_to_termination]):.1%}",
                        help="Percentage of runs that reached a terminating state"
                    )
                with col3:
                    st.metric(
                        "Max Steps Reached",
                        f"{max(steps_to_termination)}",
                        help="Maximum number of steps taken in any run"
                    )
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram of steps to termination
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.histplot(steps_to_termination, bins=20, ax=ax)
                    ax.set_title('Distribution of Steps to Termination')
                    ax.set_xlabel('Steps')
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)
                
                with col2:
                    # State visit frequency
                    fig, ax = plt.subplots(figsize=(6, 4))
                    visit_freq = state_visits / num_simulations
                    df_visits = pd.DataFrame({
                        'State': [f"State {i+1}" for i in range(len(T))],
                        'Visits': visit_freq
                    })
                    sns.barplot(data=df_visits, x='State', y='Visits', hue='State', legend=False, ax=ax)
                    ax.set_title('State Visit Frequency')
                    ax.set_xlabel('State')
                    ax.set_ylabel('Average Visits per Run')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                
                # Show individual runs if requested
                if show_individual_runs:
                    st.markdown("### 📝 Individual Run Sequences")
                    for i, run in enumerate(all_runs):
                        # Convert to 1-based indexing and add arrows
                        run_str = " → ".join([f"S{s+1}" for s in run])
                        if Term[run[-1]]:
                            run_str += " (Terminated)"
                        else:
                            run_str += " (Max Steps)"
                        st.text(f"Run {i+1}: {run_str}")
            
            st.markdown("---")
            st.markdown("### 🔄 Comparative Simulation")
            st.markdown("Compare runs from two different initial states side by side")
            
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                state1 = st.selectbox(
                    "First Initial State",
                    [f"State {i+1}" for i in range(len(T))],
                    help="Choose the first starting state for comparison"
                )
                num_comparative_runs = st.number_input(
                    "Number of Comparative Runs",
                    min_value=1,
                    max_value=50,
                    value=10,
                    help="Number of runs to show for each state"
                )
            
            with comp_col2:
                state2 = st.selectbox(
                    "Second Initial State",
                    [f"State {i+1}" for i in range(len(T))],
                    help="Choose the second starting state for comparison"
                )
                run_speed = st.slider(
                    "Animation Speed (ms)",
                    min_value=100,
                    max_value=2000,
                    value=500,
                    step=100,
                    help="Time between steps in milliseconds"
                )
            
            if st.button("Run Comparative Simulation"):
                # Convert states to 0-based indices
                state1_idx = int(state1.split()[1]) - 1
                state2_idx = int(state2.split()[1]) - 1
                
                # Create containers for the runs
                run_container1 = st.container()
                run_container2 = st.container()
                
                # Run simulations
                for run in range(num_comparative_runs):
                    # Initialize runs
                    run1 = [state1_idx]
                    run2 = [state2_idx]
                    steps1 = steps2 = 0
                    state1_current = state1_idx
                    state2_current = state2_idx
                    
                    # Run until both reach termination or max steps
                    while (steps1 < max_steps or steps2 < max_steps) and \
                          (not Term[state1_current] or not Term[state2_current]):
                        
                        # Update first run if not terminated
                        if not Term[state1_current] and steps1 < max_steps:
                            next_state1 = np.random.choice(len(T), p=T[state1_current])
                            run1.append(next_state1)
                            state1_current = next_state1
                            steps1 += 1
                        
                        # Update second run if not terminated
                        if not Term[state2_current] and steps2 < max_steps:
                            next_state2 = np.random.choice(len(T), p=T[state2_current])
                            run2.append(next_state2)
                            state2_current = next_state2
                            steps2 += 1
                    
                    # Display the runs side by side
                    with run_container1:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Run {run+1} from {state1}:**")
                            run_str1 = " → ".join([f"S{s+1}" for s in run1])
                            if Term[run1[-1]]:
                                run_str1 += " (Terminated)"
                            else:
                                run_str1 += " (Max Steps)"
                            st.text(run_str1)
                        
                        with col2:
                            st.markdown(f"**Run {run+1} from {state2}:**")
                            run_str2 = " → ".join([f"S{s+1}" for s in run2])
                            if Term[run2[-1]]:
                                run_str2 += " (Terminated)"
                            else:
                                run_str2 += " (Max Steps)"
                            st.text(run_str2)
                    
                    # Add a small delay between runs
                    time.sleep(run_speed / 1000)
                
                # Add comparative statistics and visualizations
                st.markdown("### 📊 Comparative Analysis")
                
                # Collect statistics across all runs
                all_runs1 = []
                all_runs2 = []
                termination_steps1 = []
                termination_steps2 = []
                state_visits1 = np.zeros(len(T))
                state_visits2 = np.zeros(len(T))
                
                # Run additional simulations for statistics
                for _ in range(100):  # Run 100 simulations for better statistics
                    # Run from first state
                    run1 = [state1_idx]
                    steps1 = 0
                    state1_current = state1_idx
                    while steps1 < max_steps and not Term[state1_current]:
                        state_visits1[state1_current] += 1
                        next_state1 = np.random.choice(len(T), p=T[state1_current])
                        run1.append(next_state1)
                        state1_current = next_state1
                        steps1 += 1
                    all_runs1.append(run1)
                    termination_steps1.append(steps1)
                    
                    # Run from second state
                    run2 = [state2_idx]
                    steps2 = 0
                    state2_current = state2_idx
                    while steps2 < max_steps and not Term[state2_current]:
                        state_visits2[state2_current] += 1
                        next_state2 = np.random.choice(len(T), p=T[state2_current])
                        run2.append(next_state2)
                        state2_current = next_state2
                        steps2 += 1
                    all_runs2.append(run2)
                    termination_steps2.append(steps2)
                
                # Calculate statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Average Steps to Termination",
                        f"{np.mean(termination_steps1):.1f} vs {np.mean(termination_steps2):.1f}",
                        f"{np.mean(termination_steps2) - np.mean(termination_steps1):.1f}",
                        help="Average number of steps before termination for each state"
                    )
                with col2:
                    term_rate1 = np.mean([s < max_steps for s in termination_steps1])
                    term_rate2 = np.mean([s < max_steps for s in termination_steps2])
                    st.metric(
                        "Termination Rate",
                        f"{term_rate1:.1%} vs {term_rate2:.1%}",
                        f"{term_rate2 - term_rate1:.1%}",
                        help="Percentage of runs that reached termination"
                    )
                with col3:
                    st.metric(
                        "Max Steps Reached",
                        f"{max(termination_steps1)} vs {max(termination_steps2)}",
                        f"{max(termination_steps2) - max(termination_steps1)}",
                        help="Maximum steps taken in any run"
                    )
                
                # Visualizations
                st.markdown("#### 📈 Comparative Visualizations")
                
                # Create tabs for different visualizations
                viz_tab1, viz_tab2, viz_tab3 = st.tabs([
                    "Steps Distribution",
                    "State Visit Patterns",
                    "Path Divergence"
                ])
                
                with viz_tab1:
                    # Steps to termination distribution
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.kdeplot(data=termination_steps1, label=f"From {state1}", ax=ax)
                    sns.kdeplot(data=termination_steps2, label=f"From {state2}", ax=ax)
                    ax.set_title('Distribution of Steps to Termination')
                    ax.set_xlabel('Steps')
                    ax.set_ylabel('Density')
                    ax.legend()
                    st.pyplot(fig)
                
                with viz_tab2:
                    # State visit patterns
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Normalize visit counts
                    visits1 = state_visits1 / len(all_runs1)
                    visits2 = state_visits2 / len(all_runs2)
                    
                    # Plot visit frequencies
                    df_visits = pd.DataFrame({
                        'State': [f"State {i+1}" for i in range(len(T))] * 2,
                        'Visits': np.concatenate([visits1, visits2]),
                        'Source': [state1] * len(T) + [state2] * len(T)
                    })
                    
                    sns.barplot(data=df_visits, x='State', y='Visits', hue='Source', ax=ax1)
                    ax1.set_title('State Visit Frequency Comparison')
                    ax1.set_xlabel('State')
                    ax1.set_ylabel('Average Visits per Run')
                    plt.xticks(rotation=45)
                    
                    # Plot visit difference
                    visit_diff = visits2 - visits1
                    sns.barplot(x=[f"State {i+1}" for i in range(len(T))], y=visit_diff, ax=ax2)
                    ax2.set_title('Difference in Visit Frequency')
                    ax2.set_xlabel('State')
                    ax2.set_ylabel('Visit Difference (State 2 - State 1)')
                    plt.xticks(rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with viz_tab3:
                    # Path divergence analysis
                    # Calculate average path length for each state
                    avg_len1 = np.mean([len(run) for run in all_runs1])
                    avg_len2 = np.mean([len(run) for run in all_runs2])
                    
                    # Calculate state transition probabilities
                    trans1 = np.zeros((len(T), len(T)))
                    trans2 = np.zeros((len(T), len(T)))
                    
                    for run in all_runs1:
                        for i in range(len(run)-1):
                            trans1[run[i], run[i+1]] += 1
                    for run in all_runs2:
                        for i in range(len(run)-1):
                            trans2[run[i], run[i+1]] += 1
                    
                    # Normalize transition matrices
                    trans1 = trans1 / np.sum(trans1, axis=1, keepdims=True)
                    trans2 = trans2 / np.sum(trans2, axis=1, keepdims=True)
                    
                    # Calculate transition difference
                    trans_diff = trans2 - trans1
                    
                    # Plot transition difference heatmap
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(trans_diff, 
                              cmap='RdBu_r',
                              center=0,
                              annot=True,
                              fmt='.2f',
                              xticklabels=[f"S{i+1}" for i in range(len(T))],
                              yticklabels=[f"S{i+1}" for i in range(len(T))],
                              ax=ax)
                    ax.set_title('Difference in Transition Probabilities\n(State 2 - State 1)')
                    st.pyplot(fig)
                    
                    # Add interpretation
                    st.markdown("""
                    **Interpretation:**
                    - Positive values (red) indicate higher transition probability from State 2
                    - Negative values (blue) indicate higher transition probability from State 1
                    - The intensity of the color shows the magnitude of the difference
                    """)

        with tab5:
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
