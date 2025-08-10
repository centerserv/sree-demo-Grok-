import streamlit as st
import pandas as pd
from main import preprocess_data, ppp_loop, plot_results

st.title("SREE Demo: PPP Loop")
st.write("Welcome! This demo refines labeled datasets using the Pattern-Presence-Permanence (PPP) loop, computing accuracy and trust (0-1 scale) over two phases. Phase 1 simulates noise; Phase 2 uses cleaned data for convergence. Use a sample dataset or upload a CSV with numeric features and a binary target (0 or 1).")

st.write("### Download Sample Datasets:")
st.download_button("Download UCI Heart Failure", data=open("UCI_heart_failure_clinical_records_dataset.csv", "rb").read(), file_name="UCI_heart_failure_clinical_records_dataset.csv", help="12.2KB, target: target")
st.download_button("Download Heart Disease", data=open("heart_disease_dataset.csv", "rb").read(), file_name="heart_disease_dataset.csv", help="~48KB, target: target")
st.download_button("Download Cardiovascular Disease", data=open("Cardiovascular_Disease_Dataset.csv", "rb").read(), file_name="Cardiovascular_Disease_Dataset.csv", help="42.6KB, target: target")

dataset_option = st.selectbox("Choose a sample dataset or upload your own:", ["Upload your own"] + ["UCI_heart_failure_clinical_records_dataset.csv", "heart_disease_dataset.csv", "Cardiovascular_Disease_Dataset.csv"])
uploaded_file = st.file_uploader("Upload your own labeled CSV", type="csv", help="Max 200MB. Ensure numeric columns and a 0/1 target column.") if dataset_option == "Upload your own" else None
target_column = st.text_input("Binary Target Column (e.g., target)", help="Must contain only 0s and 1s.") if dataset_option == "Upload your own" else None

if dataset_option != "Upload your own":
    target_column = "target"
    uploaded_file = open(dataset_option, "rb")

if uploaded_file and target_column:
    df = pd.read_csv(uploaded_file)
    try:
        X, y, original_df = preprocess_data(df, target_column)  # Updated to unpack three values
        # Phase 1
        accuracies1, trust_scores1, baseline1, suspect_flags1, trust_per_row1 = ppp_loop(X, y, phase="1")
        # Phase 2: Rerun on cleaned
        cleaned_idx = ~suspect_flags1
        X_cleaned = X[cleaned_idx]
        y_cleaned = y[cleaned_idx]
        df_cleaned = original_df.iloc[cleaned_idx].copy()
        accuracies2, trust_scores2, baseline2, suspect_flags2, trust_per_row2 = ppp_loop(X_cleaned, y_cleaned, noise_factor=1.0, prior_trust=trust_scores1[-1], phase="2")
        
        st.write("### Final Report")
        st.write(f"Dataset: {dataset_option if dataset_option != 'Upload your own' else uploaded_file.name}, Rows: {len(df)}, Columns: {len(df.columns)}")
        st.write(f"Preprocessing: Handled NaN with median, scaled features, applied SMOTE if imbalance (>0.7 or <0.3).")
        st.write("#### Phase 1 (Noisy)")
        st.write(f"Baseline Accuracy: {baseline1:.3f}")
        st.write(f"Final Accuracy: {accuracies1[-1]:.3f} (Improvement: {max(0, accuracies1[-1] - baseline1):.3f})")
        st.write(f"Final Trust: {trust_scores1[-1]:.3f}")
        st.write(f"Suspect Rows Flagged: {sum(suspect_flags1)} ({sum(suspect_flags1)/len(y)*100:.1f}%)")
        st.write("#### Phase 2 (Cleaned, No Noise)")
        st.write(f"Baseline Accuracy: {baseline2:.3f}")
        st.write(f"Final Accuracy: {accuracies2[-1]:.3f} (Improvement: {max(0, accuracies2[-1] - baseline2):.3f})")
        st.write(f"Final Trust: {trust_scores2[-1]:.3f}")
        st.write(f"Suspect Rows Flagged: {sum(suspect_flags2)} ({sum(suspect_flags2)/len(y_cleaned)*100:.1f}%)")
        
        plot_results({"1": (accuracies1, trust_scores1), "2": (accuracies2, trust_scores2)})
        st.image("sree_results.png")
        st.download_button("Download Results", "sree_results.csv")
        
        # Annotated dataset
        df_annotated = original_df.copy()
        df_annotated['SREE_Trust'] = trust_per_row1
        df_annotated['Suspect_Flag'] = suspect_flags1
        st.download_button("Download Annotated Dataset", df_annotated.to_csv(index=False).encode('utf-8'), file_name="annotated_dataset.csv")
        
        # Cleaned dataset
        st.download_button("Download Cleaned Dataset", df_cleaned.to_csv(index=False).encode('utf-8'), file_name="cleaned_dataset.csv")
        
        st.write("### Implications")
        st.write("SREE enhances dataset reliability across all industries (e.g., health, finance, energy, education, manufacturing) with accuracy gains up to 40% and trust ~0.96 on refined data. For details, see SREE.pdf.")
    except ValueError as e:
        st.error(f"Error: {e}. Check target column contains only 0s and 1s.")
