"""
preprocess.py - Data Preprocessing Pipeline
=============================================
Hospital Patient Readmission Data Analysis

This module handles all data preprocessing steps:
- Loading raw data
- Handling missing values
- Encoding categorical variables
- Feature engineering
- Scaling numerical features
- Saving processed data and fitted transformers
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "patient.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "patient_cleaned.csv")
ENCODERS_PATH = os.path.join(BASE_DIR, "models", "encoders.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# ──────────────────────────────────────────────
# COLUMN DEFINITIONS
# ──────────────────────────────────────────────
ID_COL = "patient_id"
DATE_COL = "admission_date"
TARGET_COL = "label"

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


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw patient data from CSV."""
    print(f"[INFO] Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    print("[INFO] Handling missing values...")

    missing_before = df.isnull().sum().sum()

    # Fill numerical columns with median
    for col in NUMERICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"   -> Filled {col} missing values with median: {median_val}")

    # Fill categorical columns with mode
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"   -> Filled {col} missing values with mode: {mode_val}")

    missing_after = df.isnull().sum().sum()
    print(f"[INFO] Missing values: {missing_before} -> {missing_after}")

    return df


def encode_categorical(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """
    Encode categorical variables using LabelEncoder.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    fit : bool
        If True, fit new encoders and save them.
        If False, load previously fitted encoders.

    Returns
    -------
    pd.DataFrame
        Dataframe with encoded categorical columns.
    """
    print("[INFO] Encoding categorical variables...")
    encoders = {}

    if fit:
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
                print(f"   -> Encoded {col}: {list(le.classes_)}")

        # Save encoders for later use in prediction
        os.makedirs(os.path.dirname(ENCODERS_PATH), exist_ok=True)
        joblib.dump(encoders, ENCODERS_PATH)
        print(f"[INFO] Encoders saved to: {ENCODERS_PATH}")
    else:
        encoders = joblib.load(ENCODERS_PATH)
        for col in CATEGORICAL_COLS:
            if col in df.columns and col in encoders:
                le = encoders[col]
                df[col] = le.transform(df[col].astype(str))

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing columns.

    New Features:
    - age_group: Binned age categories
    - high_risk: Binary flag for high readmission risk score
    - total_care_intensity: Composite of medications + comorbidities + length of stay
    - admission_month / admission_year: Extracted from admission_date
    """
    print("[INFO] Engineering features...")

    # Age group binning
    bins = [0, 30, 45, 60, 75, 100]
    labels_age = [0, 1, 2, 3, 4]  # Young, Middle, Senior, Elderly, Very Elderly
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels_age, include_lowest=True).astype(int)
    print("   -> Created age_group (5 bins)")

    # High risk flag (readmission_risk_score > 0.8)
    df["high_risk"] = (df["readmission_risk_score"] > 0.8).astype(int)
    print("   -> Created high_risk flag")

    # Total care intensity score
    df["total_care_intensity"] = (
        df["medications_count"] + df["comorbidities_count"] + df["length_of_stay"]
    )
    print("   -> Created total_care_intensity")

    # Extract date features
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%d-%m-%Y", errors="coerce")
        df["admission_month"] = df[DATE_COL].dt.month
        df["admission_year"] = df[DATE_COL].dt.year
        print("   -> Extracted admission_month and admission_year")

    return df


def scale_features(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """
    Scale numerical features using StandardScaler.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    fit : bool
        If True, fit a new scaler and save it.
        If False, load a previously fitted scaler.

    Returns
    -------
    pd.DataFrame
        Dataframe with scaled numerical features.
    """
    print("[INFO] Scaling numerical features...")

    # Columns to scale (numerical + engineered)
    cols_to_scale = NUMERICAL_COLS + ["total_care_intensity", "admission_month", "admission_year"]
    cols_to_scale = [c for c in cols_to_scale if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        joblib.dump(scaler, SCALER_PATH)
        print(f"[INFO] Scaler saved to: {SCALER_PATH}")
    else:
        scaler = joblib.load(SCALER_PATH)
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df


def get_feature_columns() -> list:
    """Return the list of feature columns used for modeling."""
    return (
        CATEGORICAL_COLS
        + NUMERICAL_COLS
        + ["age_group", "high_risk", "total_care_intensity", "admission_month", "admission_year"]
    )


def preprocess_pipeline(fit: bool = True) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline.

    Parameters
    ----------
    fit : bool
        If True, fit transformers from scratch (training mode).
        If False, use previously fitted transformers (prediction mode).

    Returns
    -------
    pd.DataFrame
        Fully preprocessed dataframe ready for modeling.
    """
    print("=" * 60)
    print("  PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Load data
    df = load_data()

    # Step 2: Handle missing values
    df = handle_missing_values(df)

    # Step 3: Feature engineering (before encoding so date column is still available)
    df = engineer_features(df)

    # Step 4: Drop non-feature columns
    drop_cols = [ID_COL, DATE_COL]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    print(f"[INFO] Dropped columns: {drop_cols}")

    # Step 5: Encode categorical variables
    df = encode_categorical(df, fit=fit)

    # Step 6: Scale numerical features
    df = scale_features(df, fit=fit)

    # Step 7: Save processed data
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"[INFO] Processed data saved to: {PROCESSED_DATA_PATH}")

    print("=" * 60)
    print(f"  DONE  -- Final shape: {df.shape}")
    print("=" * 60)

    return df


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    processed_df = preprocess_pipeline(fit=True)
    print("\nPreview of processed data:")
    print(processed_df.head())
    print(f"\nTarget distribution:\n{processed_df[TARGET_COL].value_counts()}")
