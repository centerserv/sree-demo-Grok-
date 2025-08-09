import streamlit as st
import pandas as pd
from main import preprocess_data, ppp_loop, plot_results

st.title("SREE Demo: PPP Loop")
st.write("Welcome! Upload a labeled CSV with numeric features and a binary target (0 or 1). Results will show accuracy and trust over 10 iterations.")

uploaded_file = st.file_uploader("Upload labeled CSV", type="csv")
target_column = st.text_input("Target column name (e.g., DEATH_EVENT)")
if uploaded_file and target_column:
    df = pd.read_csv(uploaded_file)
    try:
        X, y = preprocess_data(df, target_column)
        accuracies, trust_scores, baseline_accuracy, suspect_flags, trust_per_row = ppp_loop(X, y)
        st.write("### Final Report")
        st.write(f"Baseline Accuracy (pre-PPP): {baseline_accuracy:.3f}")
        st.write(f"Final Accuracy (post-PPP): {accuracies[-1]:.3f} (Improvement: {accuracies[-1] - baseline_accuracy:.3f})")
        st.write(f"Final Trust: {trust_scores[-1]:.3f} (Convergence after iterations)")
        st.write(f"Suspect Rows Flagged: {sum(suspect_flags)} ({sum(suspect_flags)/len(y)*100:.1f}%)")
        plot_results(accuracies, trust_scores)
        st.image("sree_results.png")
        st.download_button("Download Results", "sree_results.csv")
        
        # Annotated dataset
        df_annotated = df.copy()
        df_annotated['SREE_Trust'] = trust_per_row
        df_annotated['Suspect_Flag'] = suspect_flags
        st.download_button("Download Annotated Dataset", df_annotated.to_csv(index=False).encode('utf-8'), file_name="annotated_dataset.csv")
        
        # Cleaned dataset
        df_cleaned = df[~suspect_flags]
        st.download_button("Download Cleaned Dataset", df_cleaned.to_csv(index=False).encode('utf-8'), file_name="cleaned_dataset.csv")
        
        st.write("SREE improves data reliability, with accuracy gains up to 40%. Applicable to health (diagnostics), finance (fraud), energy (optimization). See SREE_for_IEEE-57.pdf.")
    except ValueError as e:
        st.error(f"Error: {e}")
