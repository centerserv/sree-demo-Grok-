import streamlit as st
import pandas as pd
from main import preprocess_data, ppp_loop, plot_results

st.title("SREE Demo: PPP Loop")
st.write("Welcome! Upload a labeled CSV with numeric features and a binary target (0 or 1). Results will show accuracy and trust over 10 iterations.")

uploaded_file = st.file_uploader("Upload labeled CSV", type="csv", help="Max 200MB. Ensure numeric columns and a 0/1 target column.")
target_column = st.text_input("Binary Target Column Name (example: result, target...)", help="Must contain only 0s and 1s.")

if st.button("Try Sample Dataset"):
    uploaded_file = open("UCI_heart_failure_clinical_records_dataset.csv", "rb")
    target_column = "DEATH_EVENT"

if uploaded_file and target_column:
    df = pd.read_csv(uploaded_file)
    try:
        X, y = preprocess_data(df, target_column)
        accuracies, trust_scores = ppp_loop(X, y)
        st.write(f"Final Accuracy: {accuracies[-1]:.3f}, Final Trust: {trust_scores[-1]:.3f}")
        plot_results(accuracies, trust_scores)
        st.image("sree_results.png")
        st.download_button("Download Results", "sree_results.csv")
    except ValueError as e:
        st.error(f"Error: {e}. Check target column contains only 0s and 1s.")
