"""
app.py - Streamlit Web Application
====================================
Hospital Patient Readmission Prediction Dashboard

This module provides an interactive web UI for:
- Predicting patient readmission risk
- Visualizing model performance metrics
- Exploring dataset statistics
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path so we can import from src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.predict import ReadmissionPredictor

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "patient.csv")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
PLOTS_DIR = os.path.join(BASE_DIR, "output", "plots")


# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_predictor():
    """Load the prediction model (cached)."""
    return ReadmissionPredictor()


@st.cache_data
def load_raw_data():
    """Load raw dataset for exploration."""
    return pd.read_csv(RAW_DATA_PATH)


@st.cache_data
def load_metrics():
    """Load saved model metrics."""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🔮 Predict Readmission", "📊 Model Performance", "📈 Data Explorer"],
)
st.sidebar.markdown("---")
st.sidebar.info(
    "This app predicts whether a hospital patient is likely to be **readmitted** "
    "based on clinical and demographic features."
)


# ──────────────────────────────────────────────
# PAGE 1: PREDICTION
# ──────────────────────────────────────────────
if page == "🔮 Predict Readmission":
    st.title("🔮 Patient Readmission Prediction")
    st.markdown(
        "Enter patient details below to predict the likelihood of hospital readmission."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Patient Info")
        age = st.slider("Age", 18, 95, 55)
        gender = st.selectbox("Gender", ["Male", "Female"])
        region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])

    with col2:
        st.subheader("🩺 Clinical Details")
        primary_diagnosis = st.selectbox(
            "Primary Diagnosis",
            [
                "Diabetes",
                "Hypertension",
                "Stroke",
                "Fracture",
                "Appendicitis",
                "Sepsis",
                "Kidney Disease",
                "Heart Failure",
                "COPD",
                "Pneumonia",
                "Influenza",
            ],
        )
        comorbidities_count = st.slider("Comorbidities Count", 1, 10, 4)
        length_of_stay = st.slider("Length of Stay (days)", 1, 30, 7)
        treatment_type = st.selectbox(
            "Treatment Type",
            ["Medical", "Surgical", "Interventional", "Conservative"],
        )
        medications_count = st.slider("Medications Count", 1, 15, 6)

    with col3:
        st.subheader("📋 History & Insurance")
        followup_visits = st.slider("Follow-up Visits (last year)", 0, 10, 3)
        prev_readmissions = st.slider("Previous Readmissions", 0, 5, 1)
        insurance_type = st.selectbox(
            "Insurance Type", ["Medicare", "Medicaid", "Private", "Uninsured"]
        )
        discharge_disposition = st.selectbox(
            "Discharge Disposition",
            ["Home", "Home Health", "Skilled Nursing", "Rehab"],
        )
        readmission_risk_score = st.slider(
            "Readmission Risk Score", 0.0, 1.0, 0.5, 0.01
        )
        season = st.selectbox("Season of Admission", ["Spring", "Summer", "Fall", "Winter"])

    st.markdown("---")

    if st.button("🚀 Predict Readmission Risk", type="primary", use_container_width=True):
        patient_data = {
            "admission_date": "01-01-2024",
            "season": season,
            "age": age,
            "gender": gender,
            "region": region,
            "primary_diagnosis": primary_diagnosis,
            "comorbidities_count": comorbidities_count,
            "length_of_stay": length_of_stay,
            "treatment_type": treatment_type,
            "medications_count": medications_count,
            "followup_visits_last_year": followup_visits,
            "prev_readmissions": prev_readmissions,
            "insurance_type": insurance_type,
            "discharge_disposition": discharge_disposition,
            "readmission_risk_score": readmission_risk_score,
        }

        try:
            predictor = load_predictor()
            result = predictor.predict(patient_data)

            st.markdown("---")

            # Result display
            risk_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
            risk_icon = risk_colors.get(result["risk_level"], "⚪")

            r1, r2, r3 = st.columns(3)
            r1.metric("Prediction", result["label"])
            r2.metric("Probability", f"{result['probability']:.2%}")
            r3.metric("Risk Level", f"{risk_icon} {result['risk_level']}")

            # Progress bar for probability
            st.progress(result["probability"])

            if result["risk_level"] == "High":
                st.error(
                    "⚠️ **High Risk**: This patient has a high probability of readmission. "
                    "Consider enhanced follow-up care and discharge planning."
                )
            elif result["risk_level"] == "Medium":
                st.warning(
                    "⚡ **Medium Risk**: Moderate readmission risk. "
                    "Standard follow-up protocols recommended."
                )
            else:
                st.success(
                    "✅ **Low Risk**: This patient has a low probability of readmission."
                )

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("Make sure you have trained the model first (run `python src/train.py`).")


# ──────────────────────────────────────────────
# PAGE 2: MODEL PERFORMANCE
# ──────────────────────────────────────────────
elif page == "📊 Model Performance":
    st.title("📊 Model Performance Metrics")

    metrics = load_metrics()

    if metrics is None:
        st.warning(
            "No model metrics found. Train the model first by running `python src/train.py`."
        )
    else:
        # Best model summary
        best = metrics["best_model"]
        st.subheader(f"🏆 Best Model: {best['model_name']}")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{best['accuracy']:.2%}")
        m2.metric("Precision", f"{best['precision']:.2%}")
        m3.metric("Recall", f"{best['recall']:.2%}")
        m4.metric("F1 Score", f"{best['f1_score']:.2%}")
        m5.metric("AUC-ROC", f"{best['roc_auc']:.2%}")

        st.markdown("---")

        # All models comparison
        st.subheader("📋 All Models Comparison")
        models_df = pd.DataFrame(metrics["all_models"])
        models_df = models_df.set_index("model_name")

        st.dataframe(
            models_df.style.highlight_max(axis=0, color="#90EE90"),
            use_container_width=True,
        )

        # Bar chart comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        models_df.plot(kind="bar", ax=ax, colormap="viridis")
        ax.set_title("Model Metrics Comparison", fontsize=14)
        ax.set_ylabel("Score")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")

        # Show saved plots
        st.subheader("📊 Evaluation Plots")
        plot_files = [f for f in os.listdir(PLOTS_DIR) if f.endswith(".png")] if os.path.exists(PLOTS_DIR) else []

        if plot_files:
            cols = st.columns(2)
            for i, pf in enumerate(sorted(plot_files)):
                with cols[i % 2]:
                    st.image(
                        os.path.join(PLOTS_DIR, pf),
                        caption=pf.replace("_", " ").replace(".png", "").title(),
                        use_container_width=True,
                    )
        else:
            st.info("No plots found. Train the model to generate evaluation plots.")


# ──────────────────────────────────────────────
# PAGE 3: DATA EXPLORER
# ──────────────────────────────────────────────
elif page == "📈 Data Explorer":
    st.title("📈 Dataset Explorer")

    try:
        df = load_raw_data()

        st.subheader("📋 Dataset Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Features", df.shape[1])
        c3.metric("Readmitted", f"{df['label'].sum():,}")
        c4.metric("Not Readmitted", f"{(df['label'] == 0).sum():,}")

        st.markdown("---")

        # Data preview
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.markdown("---")

        # Target distribution
        st.subheader("🎯 Readmission Distribution")
        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            labels = ["Readmitted", "Not Readmitted"]
            counts = df["label"].value_counts().sort_index(ascending=False)
            colors = ["#FF6B6B", "#4ECDC4"]
            ax1.pie(counts, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
            ax1.set_title("Readmission Distribution")
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, x="label", palette=colors, ax=ax2)
            ax2.set_xticklabels(["Not Readmitted", "Readmitted"])
            ax2.set_title("Readmission Count")
            ax2.set_xlabel("")
            ax2.set_ylabel("Count")
            st.pyplot(fig2)

        st.markdown("---")

        # Feature distributions
        st.subheader("📊 Feature Distributions")
        selected_feature = st.selectbox(
            "Select a feature to visualize",
            ["age", "gender", "region", "primary_diagnosis", "treatment_type",
             "insurance_type", "comorbidities_count", "length_of_stay",
             "medications_count", "readmission_risk_score"],
        )

        fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

        if df[selected_feature].dtype == "object":
            sns.countplot(data=df, x=selected_feature, hue="label",
                          palette=["#4ECDC4", "#FF6B6B"], ax=axes[0])
            axes[0].set_title(f"{selected_feature} by Readmission Status")
            axes[0].tick_params(axis="x", rotation=45)

            ct = pd.crosstab(df[selected_feature], df["label"], normalize="index")
            ct.plot(kind="bar", stacked=True, color=["#4ECDC4", "#FF6B6B"], ax=axes[1])
            axes[1].set_title(f"{selected_feature} — Readmission Rate")
            axes[1].set_ylabel("Proportion")
            axes[1].tick_params(axis="x", rotation=45)
            axes[1].legend(["No Readmission", "Readmission"])
        else:
            sns.histplot(data=df, x=selected_feature, hue="label", kde=True,
                         palette=["#4ECDC4", "#FF6B6B"], ax=axes[0])
            axes[0].set_title(f"{selected_feature} Distribution by Readmission")

            sns.boxplot(data=df, x="label", y=selected_feature,
                        palette=["#4ECDC4", "#FF6B6B"], ax=axes[1])
            axes[1].set_xticklabels(["No Readmission", "Readmission"])
            axes[1].set_title(f"{selected_feature} — Box Plot")

        plt.tight_layout()
        st.pyplot(fig3)

        st.markdown("---")

        # Statistical summary
        st.subheader("📐 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")


# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Hospital Patient Readmission**  \n"
    "Data Analysis & Prediction  \n"
    "© 2026"
)
