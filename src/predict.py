"""
predict.py - Prediction Module
================================
Hospital Patient Readmission Prediction

This module handles:
- Loading the trained model and fitted transformers
- Preprocessing new patient data
- Making readmission predictions
- Providing prediction probabilities and risk levels
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib


# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "models", "encoders.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# ──────────────────────────────────────────────
# COLUMN DEFINITIONS (must match preprocess.py)
# ──────────────────────────────────────────────
CATEGORICAL_COLS = [
    "season",
    "gender",
    "region",
    "primary_diagnosis",
    "treatment_type",
    "insurance_type",
    "discharge_disposition",
]

NUMERICAL_COLS = [
    "age",
    "comorbidities_count",
    "length_of_stay",
    "medications_count",
    "followup_visits_last_year",
    "prev_readmissions",
    "readmission_risk_score",
]


class ReadmissionPredictor:
    """
    Predicts hospital patient readmission risk.

    Loads the trained model, label encoders, and scaler,
    then provides methods to preprocess new data and make predictions.
    """

    def __init__(self):
        """Load model and transformers."""
        print("[INFO] Loading model and transformers...")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train.py first."
            )

        self.model = joblib.load(MODEL_PATH)
        self.encoders = joblib.load(ENCODERS_PATH) if os.path.exists(ENCODERS_PATH) else {}
        self.scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

        print("[INFO] Model and transformers loaded successfully.")

    def preprocess_input(self, data: dict) -> pd.DataFrame:
        """
        Preprocess a single patient record for prediction.

        Parameters
        ----------
        data : dict
            Patient record with keys matching expected feature names.

        Returns
        -------
        pd.DataFrame
            Preprocessed feature row ready for prediction.
        """
        df = pd.DataFrame([data])

        # Feature engineering
        bins = [0, 30, 45, 60, 75, 100]
        labels_age = [0, 1, 2, 3, 4]
        df["age_group"] = pd.cut(
            df["age"], bins=bins, labels=labels_age, include_lowest=True
        ).astype(int)

        df["high_risk"] = (df["readmission_risk_score"] > 0.8).astype(int)

        df["total_care_intensity"] = (
            df["medications_count"] + df["comorbidities_count"] + df["length_of_stay"]
        )

        # Parse admission date if provided
        if "admission_date" in df.columns:
            df["admission_date"] = pd.to_datetime(
                df["admission_date"], format="%d-%m-%Y", errors="coerce"
            )
            df["admission_month"] = df["admission_date"].dt.month
            df["admission_year"] = df["admission_date"].dt.year
            df.drop(columns=["admission_date"], inplace=True)
        else:
            df["admission_month"] = 1
            df["admission_year"] = 2023

        # Drop non-feature columns
        for col in ["patient_id"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # Encode categorical variables
        for col in CATEGORICAL_COLS:
            if col in df.columns and col in self.encoders:
                le = self.encoders[col]
                # Handle unseen labels gracefully
                val = df[col].values[0]
                if val in le.classes_:
                    df[col] = le.transform(df[col].astype(str))
                else:
                    print(f"[WARN] Unknown value '{val}' for {col}. Using 0.")
                    df[col] = 0

        # Scale numerical features
        if self.scaler is not None:
            cols_to_scale = NUMERICAL_COLS + [
                "total_care_intensity",
                "admission_month",
                "admission_year",
            ]
            cols_to_scale = [c for c in cols_to_scale if c in df.columns]
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])

        return df

    def predict(self, data: dict) -> dict:
        """
        Predict readmission for a single patient record.

        Parameters
        ----------
        data : dict
            Patient record dict. Required keys:
            - age, gender, season, region, primary_diagnosis,
              comorbidities_count, length_of_stay, treatment_type,
              medications_count, followup_visits_last_year,
              prev_readmissions, insurance_type, discharge_disposition,
              readmission_risk_score

        Returns
        -------
        dict
            Prediction result with keys:
            - prediction: 0 (No Readmission) or 1 (Readmission)
            - probability: float (probability of readmission)
            - risk_level: str (Low / Medium / High)
            - label: str (human-readable label)
        """
        df = self.preprocess_input(data)
        prediction = int(self.model.predict(df)[0])
        probability = float(self.model.predict_proba(df)[0][1])

        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        result = {
            "prediction": prediction,
            "label": "Readmission" if prediction == 1 else "No Readmission",
            "probability": round(probability, 4),
            "risk_level": risk_level,
        }

        return result

    def predict_batch(self, data_list: list) -> list:
        """
        Predict readmission for multiple patient records.

        Parameters
        ----------
        data_list : list of dict
            List of patient records.

        Returns
        -------
        list of dict
            List of prediction results.
        """
        return [self.predict(record) for record in data_list]


def demo_prediction():
    """Run a demo prediction with sample patient data."""
    sample_patient = {
        "patient_id": "P99999",
        "admission_date": "15-05-2026",
        "season": "Spring",
        "age": 72,
        "gender": "Male",
        "region": "South",
        "primary_diagnosis": "Diabetes",
        "comorbidities_count": 5,
        "length_of_stay": 8,
        "treatment_type": "Medical",
        "medications_count": 9,
        "followup_visits_last_year": 3,
        "prev_readmissions": 2,
        "insurance_type": "Medicare",
        "discharge_disposition": "Home Health",
        "readmission_risk_score": 0.85,
    }

    predictor = ReadmissionPredictor()
    result = predictor.predict(sample_patient)

    print("\n" + "=" * 50)
    print("  PREDICTION RESULT")
    print("=" * 50)
    print(f"  Patient ID:    {sample_patient['patient_id']}")
    print(f"  Prediction:    {result['label']}")
    print(f"  Probability:   {result['probability']:.2%}")
    print(f"  Risk Level:    {result['risk_level']}")
    print("=" * 50)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    demo_prediction()
