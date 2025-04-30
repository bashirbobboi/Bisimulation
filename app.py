import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
from demo import (
    bisimulation_distance_matrix,
    input_probabilistic_transition_system,
    generate_graphviz_source,
)

st.set_page_config(page_title="PTS Bisimulation Tool", layout="wide")
st.title("📊 Probabilistic Bisimulation Distance Calculator")

uploaded_file = st.file_uploader("Upload your transition matrix file (.txt)", type="txt")

if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    temp_path = "temp_input.txt"
    with open(temp_path, "w") as f:
        f.write(content)

    try:
        T, Term, labels = input_probabilistic_transition_system(filename=temp_path, use_file=True)

        st.success("File successfully loaded and parsed.")
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

        np.fill_diagonal(D, np.inf)
        min_idx = np.unravel_index(np.argmin(D), D.shape)
        max_idx = np.unravel_index(np.argmax(D), D.shape)

        st.write(f"Most similar states: S{min_idx[0]+1} and S{min_idx[1]+1} (distance: {D[min_idx]:.3f})")
        st.write(f"Most different states: S{max_idx[0]+1} and S{max_idx[1]+1} (distance: {D[max_idx]:.3f})")

        st.markdown("### Bisimulation Structure Visualization")
        graphviz_src = generate_graphviz_source(T, Term, labels)
        st.graphviz_chart(graphviz_src)

    except Exception as e:
        st.error(f"Error parsing or computing: {e}")
