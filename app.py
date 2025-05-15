import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
        temp_path = "temp_input.txt"
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
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Distance Matrix", "🌡️ Heatmap", "📈 Analysis", "🔄 Minimized PTS"])
        
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
            col1, col2, col3 = st.columns(3)
            col1.metric("Minimum Distance", f"{np.min(D):.3f}")
            col2.metric("Maximum Distance", f"{np.max(D):.3f}")
            col3.metric("Average Distance", f"{np.mean(D):.3f}")

            st.markdown("### 🔍 State Analysis")
            max_idx = np.unravel_index(np.argmax(D), D.shape)
            st.info(f"Most different states: S{max_idx[0]+1} and S{max_idx[1]+1} (distance: {D[max_idx]:.3f})")

            np.fill_diagonal(D, np.inf)
            min_idx = np.unravel_index(np.argmin(D), D.shape)
            st.success(f"Most similar states: S{min_idx[0]+1} and S{min_idx[1]+1} (distance: {D[min_idx]:.3f})")

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

    except Exception as e:
        st.error(f"❌ Computation error: {e}")
