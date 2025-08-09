import streamlit as st
import pandas as pd
from main import preprocess_data, ppp_loop, plot_results

st.title("SREE Demo: PPP Loop")
st.write("Welcome! This demo refines labeled datasets using the Pattern-Presence-Permanence (PPP) loop, computing accuracy and trust (0-1 scale). You have two options: use one of our sample datasets or upload your own CSV with numeric features and a binary target (0 or 1).")

st.write("### Download Sample Datasets:")
st.download_button("Download UCI Heart Failure", data=open("UCI_heart_failure_clinical_records_dataset.csv", "rb").read(), file_name="UCI_heart_failure_clinical_records_dataset.csv", help="12.2KB, target: DEATH_EVENT")
st.download_button("Download Heart Disease", data=open("heart_disease_dataset.csv", "rb").read(), file_name="heart_disease_dataset.csv", help="~48KB, target: target")
st.download_button("Download Cardiovascular Disease", data=open("Cardiovascular_Disease_Dataset.csv", "rb").read(), file_name="Cardiovascular_Disease_Dataset.csv", help="42.6KB, target: target")

dataset_option = st.selectbox("Choose a sample dataset or upload your own:", ["Upload your own"] + ["UCI_heart_failure_clinical_records_dataset.csv", "heart_disease_dataset.csv", "Cardiovascular_Disease_Dataset.csv"])
uploaded_file = st.file_uploader("Upload your own labeled CSV", type="csv", help="Max 200MB. Ensure numeric columns and a 0/1 target column.") if dataset_option == "Upload your own" else None
target_column = st.text_input("Binary Target Column (e.g., DEATH_EVENT or target)", help="Must contain only 0s and 1s.") if dataset_option == "Upload your own" else None

if dataset_option != "Upload your own":
    target_column = "DEATH_EVENT" if dataset_option == "UCI_heart_failure_clinical_records_dataset.csv" else "target"
    uploaded_file = open(dataset_option, "rb")

if uploaded_file and target_column:
    df = pd.read_csv(uploaded_file)
    try:
        X, y = preprocess_data(df, target_column)
        accuracies, trust_scores = ppp_loop(X, y)
        st.write("### Results:")
        st.write(f"Dataset: {dataset_option if dataset_option != 'Upload your own' else uploaded_file.name}, Rows: {len(df)}, Columns: {len(df.columns)}")
        st.write(f"Preprocessing: Handled NaN with median, scaled features, applied SMOTE if imbalance (>0.7 or <0.3).")
        st.write(f"Final Accuracy: {accuracies[-1]:.3f}, Final Trust: {trust_scores[-1]:.3f}")
        plot_results(accuracies, trust_scores)
        st.image("sree_results.png")
        st.download_button("Download Results", "sree_results.csv")
    except ValueError as e:
        st.error(f"Error: {e}. Check target column contains only 0s and 1s.")
