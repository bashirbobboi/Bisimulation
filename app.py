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
    generate_graphviz_source
)

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
        with open(temp_path, "w") as f:
            f.write(content)
        try:
            T, Term, labels = input_probabilistic_transition_system(filename=temp_path, use_file=True)
            st.success("✅ File successfully loaded and parsed.")
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
    # Display help section for file upload
    with st.expander("📝 File Format Help", expanded=True):
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
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Distance Matrix", 
            "🌡️ Heatmap", 
            "📈 Analysis", 
            "🔄 Minimized PTS",
            "📚 Theory"
        ])
        
        with tab1:
            st.markdown("### Distance Matrix")
            with st.expander("Show Distance Table", expanded=True):
                st.dataframe(np.round(D, 3))

        with tab2:
            st.markdown("### Heatmap Visualization")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(D, annot=True, cmap="YlOrRd", fmt=".3f",
                        xticklabels=[f"S{i+1}" for i in range(len(D))],
                        yticklabels=[f"S{i+1}" for i in range(len(D))])
            plt.title("Bisimulation Distance Heatmap")
            st.pyplot(fig)

        with tab3:
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

            st.markdown("### 🔍 State Analysis")
            
            # Find all pairs with minimum and maximum distances
            D_copy = D.copy()  # Create a copy to avoid modifying the original
            np.fill_diagonal(D_copy, np.inf)  # Exclude self-comparisons for minimum distance
            min_distance = np.min(D_copy)
            
            # For maximum distance, use the original matrix
            max_distance = np.max(D)
            
            # Find all pairs with minimum distance
            min_pairs = np.where(np.abs(D_copy - min_distance) < 1e-10)
            min_pairs = list(zip(min_pairs[0], min_pairs[1]))
            
            # Find all pairs with maximum distance
            max_pairs = np.where(np.abs(D - max_distance) < 1e-10)
            max_pairs = list(zip(max_pairs[0], max_pairs[1]))
            
            # Create columns for the analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Most Similar States")
                if len(min_pairs) == 1:
                    st.success(f"States S{min_pairs[0][0]+1} and S{min_pairs[0][1]+1} are most similar with distance {min_distance:.3f}")
                else:
                    st.success(f"Found {len(min_pairs)} pairs of most similar states (distance: {min_distance:.3f}):")
                    for i, (s1, s2) in enumerate(min_pairs, 1):
                        st.write(f"{i}. States S{s1+1} and S{s2+1}")
                
                # Visualize the most similar states
                if len(min_pairs) > 0:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    similar_states = set()
                    for s1, s2 in min_pairs:
                        similar_states.add(s1)
                        similar_states.add(s2)
                    
                    # Create a subgraph of the most similar states
                    subgraph = T[list(similar_states)][:, list(similar_states)]
                    sns.heatmap(subgraph, annot=True, cmap="YlGn", fmt=".2f",
                              xticklabels=[f"S{i+1}" for i in similar_states],
                              yticklabels=[f"S{i+1}" for i in similar_states])
                    plt.title("Transition Probabilities\nBetween Most Similar States")
                    st.pyplot(fig)
            
            with col2:
                st.markdown("#### 🎯 Most Different States")
                if len(max_pairs) == 1:
                    st.error(f"States S{max_pairs[0][0]+1} and S{max_pairs[0][1]+1} are most different with distance {max_distance:.3f}")
                else:
                    st.error(f"Found {len(max_pairs)} pairs of most different states (distance: {max_distance:.3f}):")
                    for i, (s1, s2) in enumerate(max_pairs, 1):
                        st.write(f"{i}. States S{s1+1} and S{s2+1}")
                
                # Visualize the most different states
                if len(max_pairs) > 0:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    different_states = set()
                    for s1, s2 in max_pairs:
                        different_states.add(s1)
                        different_states.add(s2)
                    
                    # Create a subgraph of the most different states
                    subgraph = T[list(different_states)][:, list(different_states)]
                    sns.heatmap(subgraph, annot=True, cmap="YlOrRd", fmt=".2f",
                              xticklabels=[f"S{i+1}" for i in different_states],
                              yticklabels=[f"S{i+1}" for i in different_states])
                    plt.title("Transition Probabilities\nBetween Most Different States")
                    st.pyplot(fig)

        with tab4:
            st.markdown("### 🧠 Probabilistic Transition System Comparison")
            
            # Compute the minimized PTS
            R_0 = {(x, y) for x in range(len(T)) for y in range(len(T))}
            R_n = refine_relation(R_0, T, Term)
            equivalence_classes, state_class_map, class_termination_status = compute_equivalence_classes(R_n, len(T), Term)
            minimized_T, minimized_labels = compute_minimized_transition_matrix(T, equivalence_classes, state_class_map, labels)

            # Display equivalence classes
            st.markdown("#### 📑 Equivalence Classes")
            for class_id, class_states in equivalence_classes.items():
                st.write(f"Class {class_id}: States {class_states}, Terminating: {class_termination_status[class_id]}")

            # Display minimized transition matrix
            st.markdown("#### 📊 Minimized Transition Matrix")
            st.dataframe(np.round(minimized_T, 3))

            # Create two columns for side-by-side visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎨 Original PTS")
                graphviz_src = generate_graphviz_source(T, Term, labels)
                st.graphviz_chart(graphviz_src)
            
            with col2:
                st.markdown("#### 🎨 Minimized PTS")
                minimized_graphviz_src = generate_graphviz_source(
                    minimized_T, 
                    list(class_termination_status.values()), 
                    minimized_labels
                )
                st.graphviz_chart(minimized_graphviz_src)

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
