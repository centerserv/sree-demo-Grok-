import streamlit as st
import pandas as pd
from main import preprocess_data, ppp_loop, plot_results

st.title("SREE Demo: PPP Loop")
uploaded_file = st.file_uploader("Upload labeled CSV", type="csv")
target_column = st.text_input("Target column name (e.g., DEATH_EVENT)")
if uploaded_file and target_column:
    df = pd.read_csv(uploaded_file)
    X, y = preprocess_data(df, target_column)
    accuracies, trust_scores = ppp_loop(X, y)
    st.write(f"Final Accuracy: {accuracies[-1]:.3f}, Final Trust: {trust_scores[-1]:.3f}")
    plot_results(accuracies, trust_scores)
    st.image("sree_results.png")
    st.download_button("Download Results", "sree_results.csv")
