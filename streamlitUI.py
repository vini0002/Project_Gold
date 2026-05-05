import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

st.title("🏥 HITL Fraud Detection System")
st.markdown("Explainable AI-powered healthcare claim review dashboard")

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("mimic_fraud_dataset.csv")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Controls")

threshold = st.sidebar.slider("Fraud Probability Threshold", 0.0, 1.0, 0.5)
show_uncertain = st.sidebar.checkbox("Show Only Uncertain Cases")

# Simulated probability
df['prob'] = df['fraud_score']

# Decision logic
def decision(prob):
    if prob > 0.8:
        return "High Risk"
    elif prob < 0.4:
        return "Low Risk"
    else:
        return "Needs Review"

df['decision'] = df['prob'].apply(decision)

# Filter
filtered_df = df[df['prob'] >= threshold]

if show_uncertain:
    filtered_df = filtered_df[filtered_df['decision'] == "Needs Review"]

# -----------------------------
# Case Table
# -----------------------------
st.subheader("📊 Flagged Cases")

st.dataframe(
    filtered_df[['prob', 'decision', 'fraud_reason']].head(50),
    use_container_width=True
)

# -----------------------------
# Case Selection
# -----------------------------
case_idx = st.selectbox("Select Case ID", filtered_df.index)

case = df.loc[case_idx]

# -----------------------------
# Layout Columns
# -----------------------------
col1, col2 = st.columns([2, 1])

# -----------------------------
# LEFT: Case Details
# -----------------------------
with col1:
    st.subheader("🧾 Clinical Narrative")
    st.write(case['TEXT'])

    st.subheader("📌 Key Information")
    st.write(f"**Fraud Probability:** {round(case['prob'], 2)}")
    st.write(f"**Decision:** {case['decision']}")
    st.write(f"**Reason (Synthetic Ground Truth):** {case['fraud_reason']}")

# -----------------------------
# RIGHT: Explanation + Actions
# -----------------------------
with col2:
    st.subheader("🔍 Model Explanation")

    # Placeholder for SHAP (replace later)
    st.info("Top contributing factors:")
    st.write("- High lab count")
    st.write("- Low abnormal results")
    st.write("- Suspicious narrative pattern")

    st.subheader("👩‍⚕️ Reviewer Action")

    if st.button("✅ Confirm Fraud"):
        st.success("Case marked as Fraud")

    if st.button("❌ Mark as Legitimate"):
        st.warning("Case marked as Not Fraud")

    if st.button("🔁 Send for Review"):
        st.info("Escalated for further review")
