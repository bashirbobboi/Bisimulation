import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
from commandline import (
    refine_relation,
    compute_equivalence_classes,
    compute_minimized_transition_matrix,
    input_probabilistic_transition_system
)
from demo import (
    bisimulation_distance_matrix,
    generate_graphviz_source,
    analyze_state_differences
)
import pandas as pd

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
    input_mode = st.radio("Select Input Mode:", ["Upload File", "Manual Input"])

T = Term = labels = None
all_labels_filled = True  # Ensure initialized

if input_mode == "Upload File":
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
        n = st.number_input("Number of states", min_value=2, max_value=10, step=1)
        matrix = []
        valid = True

    # Create columns for matrix input
    cols = st.columns(min(n, 3))  # Show max 3 columns at a time
    for i in range(n):
        col_idx = i % 3
        with cols[col_idx]:
            row_input = st.text_input(f"State {i+1} transitions", key=f"row_{i}")
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

    if valid:
        T = np.array(matrix)
        labels = {}
        st.markdown("### 🏷️ Transition Labels")
        label_cols = st.columns(min(n, 3))
        for i in range(n):
            for j in range(n):
                if T[i][j] > 0:
                    col_idx = (i * n + j) % 3
                    with label_cols[col_idx]:
                        label = st.text_input(f"S{i+1} → S{j+1}", key=f"label_{i}_{j}")
                        if not label:
                            all_labels_filled = False
                        else:
                            labels[(i, j)] = label

        if not all_labels_filled:
            st.warning("⚠️ Please fill in all transition labels to view results.")
        else:
            st.success("✅ Manual input accepted.")

# ---- Main Calculation ---- #
if T is not None and Term is not None and (input_mode == "Upload File" or all_labels_filled):
    try:
        D = bisimulation_distance_matrix(T, Term)

        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "📏 Distance Analysis", 
            "📈 System Analysis", 
            "🔄 Minimization",
            "📚 Theory"
        ])
        
        with tab1:
            st.markdown("### Distance Analysis")
            
            st.markdown("#### Distance Matrix")
            with st.expander("Show Distance Table", expanded=True):
                # Create a DataFrame with 1-based indexing
                df = pd.DataFrame(np.round(D, 3), 
                                index=[f"State {i+1}" for i in range(len(D))],
                                columns=[f"State {i+1}" for i in range(len(D))])
                st.dataframe(df)

            st.markdown("#### Distance Heatmap")
            with st.expander("Show Distance Heatmap", expanded=True):
                fig, ax = plt.subplots(figsize=(6, 4))  # Increased figure size
                sns.heatmap(D, annot=True, cmap="YlOrRd", fmt=".3f",
                            xticklabels=[f"State {i+1}" for i in range(len(D))],
                            yticklabels=[f"State {i+1}" for i in range(len(D))],
                            annot_kws={"size": 8})  # Increased annotation size
                # Set tick label sizes after creating the heatmap
                ax.tick_params(axis='both', which='major', labelsize=8)  # Increased tick label size
                plt.title("Bisimulation Distance Heatmap", pad=8, size=12)  # Increased title size
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                
                # Add interactive state pair analysis
                st.markdown("### 🔍 Analyze State Differences")
                st.markdown("""
                This section helps you understand why any two states behave differently. Select two states below to see:
                - Their overall distance
                - Whether they terminate differently
                - Which differences contribute most to their distance
                """)
                
                col1, col2 = st.columns(2)
                with col1:
                    state1 = st.selectbox("Select first state", 
                                        [f"State {i+1}" for i in range(len(D))],
                                        key="state1",
                                        help="Choose the first state to compare")
                with col2:
                    state2 = st.selectbox("Select second state", 
                                        [f"State {i+1}" for i in range(len(D))],
                                        key="state2",
                                        help="Choose the second state to compare")
                
                # Get state indices (subtract 1 because states are 1-indexed in UI)
                idx1 = int(state1.split()[1]) - 1
                idx2 = int(state2.split()[1]) - 1
                
                # Get explanations
                explanations = analyze_state_differences(idx1, idx2, T, Term, D)
                
                # Display the distance and explanations
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
                    # Display explanations directly without parsing
                    has_termination_mismatch = False
                    for explanation in explanations:
                        if "Termination mismatch" in explanation:
                            has_termination_mismatch = True
                            st.markdown(f"- {explanation}")
                            if D[idx1, idx2] == 1.0:
                                st.markdown("  *(This alone contributes the full distance of 1.0)*")
                        elif not has_termination_mismatch or D[idx1, idx2] < 1.0:
                            st.markdown(f"- {explanation}")
                    if not has_termination_mismatch or D[idx1, idx2] < 1.0:
                        st.markdown("*Note: Only the top 3 contributing transitions are shown here for clarity.*")

            st.markdown("#### 📊 Metrics Breakdown")
            # Add state analysis in expanders
            with st.expander("🔍 State Analysis", expanded=True):
                st.markdown("""
                This section shows you the most similar and most different states in your system.
                - **Most Similar States**: These states behave almost identically
                - **Most Different States**: These states have the most different behavior
                
                For each pair, you'll see:
                - Their distance (0 = identical, 1 = completely different)
                - Why they are similar or different
                - How their transition probabilities compare
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
                        for explanation in explanations:
                            if "Termination mismatch" in explanation:
                                st.markdown(f"- {explanation}")
                            else:
                                # Parse the transition difference explanation
                                parts = explanation.split(": ")[1].split(" vs ")
                                state1_trans = parts[0].split(" (p=")
                                state2_trans = parts[1].split(" (p=")
                                contribution = explanation.split("contribution: ")[1].replace("contribution", "").strip()
                                
                                # Create a more readable explanation
                                st.markdown(f"""
                                - **Transition Difference:**
                                  - State {min_pairs[0][0]+1} has a {state1_trans[1]} chance of going to {state1_trans[0]}
                                  - State {min_pairs[0][1]+1} has a {state2_trans[1]} chance of going to {state2_trans[0]}
                                  - This difference contributes {contribution} to their overall distance
                                """)
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
                        for explanation in explanations:
                            if "Termination mismatch" in explanation:
                                st.markdown(f"- {explanation}")
                            else:
                                # Parse the transition difference explanation
                                parts = explanation.split(": ")[1].split(" vs ")
                                state1_trans = parts[0].split(" (p=")
                                state2_trans = parts[1].split(" (p=")
                                contribution = explanation.split("contribution: ")[1].replace("contribution =", "").strip()
                                
                                # Create a more readable explanation
                                st.markdown(f"""
                                - **Transition Difference:**
                                  - State {max_pairs[0][0]+1} has a {state1_trans[1]} chance of going to {state1_trans[0]}
                                  - State {max_pairs[0][1]+1} has a {state2_trans[1]} chance of going to {state2_trans[0]}
                                  - This difference contributes {contribution} to their overall distance
                                """)
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
                - **0.3 - 0.5**: States show significant differences
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
            st.markdown("### 📊 Summary Statistics")
            
            # Calculate metrics
            num_states = len(T)
            num_terminating = sum(Term)
            num_transitions = sum(sum(row > 0) for row in T)
            avg_transitions = num_transitions / num_states
            sparsity = 1 - (num_transitions / (num_states * num_states))  # Sparsity of transition matrix
            
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

        with tab3:
            st.markdown("### 🧠 Probabilistic Transition System Comparison")
            
            # Compute the minimized PTS
            R_0 = {(x, y) for x in range(len(T)) for y in range(len(T))}
            R_n = refine_relation(R_0, T, Term)
            equivalence_classes, state_class_map, class_termination_status = compute_equivalence_classes(R_n, len(T), Term)
            minimized_T, minimized_labels = compute_minimized_transition_matrix(T, equivalence_classes, state_class_map, labels)

            # Show visualizations first
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎨 Original PTS")
                graphviz_src = generate_graphviz_source(T, Term, labels, is_minimized=False)
                st.graphviz_chart(graphviz_src)
            
            with col2:
                st.markdown("#### 🎨 Minimized PTS")
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
