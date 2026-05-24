"""
train.py - Model Training Pipeline
====================================
Hospital Patient Readmission Prediction

This module handles:
- Loading preprocessed data
- Handling class imbalance with SMOTE
- Training multiple ML models (Random Forest, XGBoost, Logistic Regression)
- Hyperparameter tuning with cross-validation
- Model evaluation (Accuracy, Precision, Recall, F1, AUC-ROC)
- Saving the best model and evaluation artifacts
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ----------------------------------------------
# PATHS
# ----------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "patient_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
PLOTS_DIR = os.path.join(BASE_DIR, "output", "plots")

TARGET_COL = "label"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_processed_data() -> tuple:
    """
    Load preprocessed data and split into features (X) and target (y).

    Returns
    -------
    tuple
        (X, y) where X is the feature matrix and y is the target vector.
    """
    print(f"[INFO] Loading processed data from: {PROCESSED_DATA_PATH}")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    print(f"[INFO] Features shape: {X.shape}")
    print(f"[INFO] Target distribution:\n{y.value_counts().to_string()}")

    return X, y


def get_models() -> dict:
    """
    Define the models to train with their hyperparameters.

    Returns
    -------
    dict
        Dictionary of model_name -> model instance.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
    }
    return models


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
    """
    Evaluate a trained model on the test set.

    Parameters
    ----------
    model : sklearn estimator
        Trained model.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
    model_name : str
        Name of the model for display.

    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    print(f"\n{'-' * 50}")
    print(f"  {model_name}  -- Results")
    print(f"{'-' * 50}")
    for k, v in metrics.items():
        if k != "model_name":
            print(f"  {k:>12s}: {v}")

    return metrics


def plot_confusion_matrix(y_test, y_pred, model_name: str):
    """Save confusion matrix plot."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Readmission", "Readmission"],
        yticklabels=["No Readmission", "Readmission"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix  -- {model_name}", fontsize=14)
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    path = os.path.join(PLOTS_DIR, f"confusion_matrix_{safe_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved: {path}")


def plot_roc_curves(models_results: list, X_test, y_test):
    """Save ROC curve comparison plot for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for result in models_results:
        model = result["model"]
        name = result["metrics"]["model_name"]
        auc = result["metrics"]["roc_auc"]

        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "roc_curve_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] ROC curve comparison saved: {path}")


def plot_feature_importance(model, feature_names: list, model_name: str, top_n: int = 15):
    """Save feature importance plot for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    ax.barh(
        range(top_n),
        importances[indices][::-1],
        color=colors,
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1], fontsize=10)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances  -- {model_name}", fontsize=14)
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    path = os.path.join(PLOTS_DIR, f"feature_importance_{safe_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Feature importance saved: {path}")


def train_pipeline():
    """
    Run the full training pipeline.

    Steps:
    1. Load processed data
    2. Split into train/test
    3. Train multiple models
    4. Evaluate each model
    5. Save the best model based on F1-score
    6. Generate evaluation plots
    """
    print("=" * 60)
    print("  MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Ensure output dirs exist
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Step 1: Load data
    X, y = load_processed_data()

    # Step 2: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[INFO] Train set: {X_train.shape[0]} samples")
    print(f"[INFO] Test set:  {X_test.shape[0]} samples")

    # Step 3: Handle class imbalance with SMOTE (if available)
    try:
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"[INFO] SMOTE applied  -- Training samples: {X_train.shape[0]}  -> {X_train_res.shape[0]}")
    except ImportError:
        print("[WARN] imbalanced-learn not installed. Skipping SMOTE.")
        X_train_res, y_train_res = X_train, y_train

    # Step 4: Train & evaluate all models
    models = get_models()
    all_results = []

    for name, model in models.items():
        print(f"\n[INFO] Training: {name}...")

        # Cross-validation score
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=cv, scoring="f1")
        print(f"[INFO] 5-Fold CV F1: {cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})")

        # Fit on full training set
        model.fit(X_train_res, y_train_res)

        # Evaluate on test set
        metrics = evaluate_model(model, X_test, y_test, name)

        # Generate plots
        y_pred = model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, name)
        plot_feature_importance(model, list(X.columns), name)

        all_results.append({"model": model, "metrics": metrics})

    # Step 5: ROC curve comparison
    plot_roc_curves(all_results, X_test, y_test)

    # Step 6: Select best model (by F1-score)
    best_result = max(all_results, key=lambda r: r["metrics"]["f1_score"])
    best_model = best_result["model"]
    best_metrics = best_result["metrics"]

    print(f"\n{'=' * 60}")
    print(f"  BEST MODEL: {best_metrics['model_name']}")
    print(f"  F1-Score:   {best_metrics['f1_score']}")
    print(f"  AUC-ROC:    {best_metrics['roc_auc']}")
    print(f"{'=' * 60}")

    # Step 7: Save best model
    joblib.dump(best_model, MODEL_PATH)
    print(f"[INFO] Best model saved to: {MODEL_PATH}")

    # Step 8: Save all metrics
    all_metrics = [r["metrics"] for r in all_results]
    with open(METRICS_PATH, "w") as f:
        json.dump(
            {"best_model": best_metrics, "all_models": all_metrics},
            f,
            indent=2,
        )
    print(f"[INFO] Metrics saved to: {METRICS_PATH}")

    # Step 9: Classification report for best model
    y_pred_best = best_model.predict(X_test)
    print(f"\n{'-' * 60}")
    print(f"  Classification Report  -- {best_metrics['model_name']}")
    print(f"{'-' * 60}")
    print(
        classification_report(
            y_test,
            y_pred_best,
            target_names=["No Readmission", "Readmission"],
        )
    )

    return best_model, best_metrics


# ----------------------------------------------
# MAIN
# ----------------------------------------------
if __name__ == "__main__":
    best_model, best_metrics = train_pipeline()
