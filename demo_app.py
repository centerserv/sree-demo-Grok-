import streamlit as st
import pandas as pd
from main import preprocess_data, ppp_loop, plot_results

st.title("SREE Demo: PPP Loop")
st.write("Welcome! Upload a labeled CSV with numeric features and a binary target (0 or 1). Results will show accuracy and trust over 10 iterations.")

dataset_option = st.selectbox("Choose a sample dataset or upload your own:", ["Upload your own"] + ["UCI_heart_failure_clinical_records_dataset.csv", "heart_disease_dataset.csv", "Cardiovascular_Disease_Dataset.csv"])
uploaded_file = st.file_uploader("Upload labeled CSV", type="csv", help="Max 200MB. Ensure numeric columns and a 0/1 target column.") if dataset_option == "Upload your own" else None
target_column = st.text_input("Binary Target Column (e.g., DEATH_EVENT or target)", help="Must contain only 0s and 1s.") if dataset_option == "Upload your own" else None

if dataset_option != "Upload your own":
    target_column = "DEATH_EVENT" if dataset_option == "UCI_heart_failure_clinical_records_dataset.csv" else "target"
    uploaded_file = open(dataset_option, "rb")

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
