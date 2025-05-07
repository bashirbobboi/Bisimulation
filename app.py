import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from demo import (
    bisimulation_distance_matrix,
    input_probabilistic_transition_system,
    generate_graphviz_source,
)

st.set_page_config(page_title="PTS Bisimulation Tool", layout="wide")
st.title("📊 Distance Calculator for Diss")

input_mode = st.radio("Select Input Mode:", ["Upload File", "Manual Input"])

T = Term = labels = None

if input_mode == "Upload File":
    uploaded_file = st.file_uploader("Upload your transition matrix file (.txt)", type="txt")
    if uploaded_file:
        content = uploaded_file.read().decode("utf-8")
        temp_path = "temp_input.txt"
        with open(temp_path, "w") as f:
            f.write(content)
        try:
            T, Term, labels = input_probabilistic_transition_system(filename=temp_path, use_file=True)
            st.success("File successfully loaded and parsed.")
        except Exception as e:
            st.error(f"Error loading file: {e}")

elif input_mode == "Manual Input":
    st.markdown("### Define Transition Matrix")
    n = st.number_input("Number of states", min_value=2, max_value=10, step=1)
    matrix = []
    valid = True
    all_labels_filled = True  # <-- declare early to avoid NameError

    for i in range(n):
        row_input = st.text_input(f"Transition probabilities from State {i+1}", key=f"row_{i}")
        try:
            row_vals = list(map(float, row_input.strip().split()))
            if len(row_vals) != n or not np.isclose(sum(row_vals), 1.0):
                raise ValueError()
            matrix.append(row_vals)
        except:
            st.error(f"Row {i+1} must have {n} numbers summing to 1.")
            valid = False

    Term = [st.selectbox(f"Is State {i+1} terminating?", [0, 1], key=f"term_{i}") for i in range(n)]

    if valid:
        T = np.array(matrix)
        labels = {}
        for i in range(n):
            for j in range(n):
                if T[i][j] > 0:
                    label = st.text_input(f"Label for transition from S{i+1} to S{j+1}", key=f"label_{i}_{j}")
                    if not label:
                        all_labels_filled = False
                    else:
                        labels[(i, j)] = label

        if not all_labels_filled:
            st.warning("Please fill in all transition labels to view distance matrix and visualization.")
        else:
            st.success("Manual input accepted.")


# ---- Main Calculation ---- #
if T is not None and Term is not None and (input_mode == "Upload File" or all_labels_filled):
    try:
        D = bisimulation_distance_matrix(T, Term)

        st.subheader("Distance Matrix")
        st.dataframe(np.round(D, 3))

        st.subheader("Heatmap Visualization")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(D, annot=True, cmap="YlOrRd", fmt=".3f",
                    xticklabels=[f"S{i+1}" for i in range(len(D))],
                    yticklabels=[f"S{i+1}" for i in range(len(D))])
        st.pyplot(fig)

        st.subheader("Summary")
        st.write(f"Minimum Distance: {np.min(D):.3f}")
        st.write(f"Maximum Distance: {np.max(D):.3f}")
        st.write(f"Average Distance: {np.mean(D):.3f}")

        # Find most different states before modifying diagonal
        max_idx = np.unravel_index(np.argmax(D), D.shape)
        st.write(f"Most different states: S{max_idx[0]+1} and S{max_idx[1]+1} (distance: {D[max_idx]:.3f})")

        # Now modify diagonal for finding most similar states
        np.fill_diagonal(D, np.inf)
        min_idx = np.unravel_index(np.argmin(D), D.shape)
        st.write(f"Most similar states: S{min_idx[0]+1} and S{min_idx[1]+1} (distance: {D[min_idx]:.3f})")

        st.markdown("### Bisimulation Structure Visualization")
        graphviz_src = generate_graphviz_source(T, Term, labels)
        st.graphviz_chart(graphviz_src)

    except Exception as e:
        st.error(f"Computation error: {e}")
