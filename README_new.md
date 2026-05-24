# 🏥 Hospital Patient Readmission Data Analysis

A machine learning project to predict hospital patient readmissions using clinical and demographic data.

---

## 📁 Project Structure

```
HospitalPatientDataAnalysis/
│
├── data/
│   ├── raw/                    # Original dataset
│   │   └── patient.csv
│   └── processed/              # Cleaned & preprocessed data
│       └── patient_cleaned.csv
│
├── notebook/
│   └── EDA.ipynb               # Exploratory Data Analysis notebook
│
├── src/
│   ├── preprocess.py           # Data preprocessing pipeline
│   ├── train.py                # Model training pipeline
│   └── predict.py              # Prediction module
│
├── app/
│   └── app.py                  # Streamlit web dashboard
│
├── models/
│   ├── model.pkl               # Trained model (best performer)
│   ├── encoders.pkl            # Fitted label encoders
│   ├── scaler.pkl              # Fitted StandardScaler
│   └── metrics.json            # Evaluation metrics for all models
│
├── output/
│   └── plots/                  # Generated visualizations
│       ├── readmission_distribution.png
│       ├── confusion_matrix_*.png
│       ├── roc_curve_comparison.png
│       └── feature_importance_*.png
│
├── requirements_new.txt        # Python dependencies
└── README_new.md               # This file
```

---

## 📊 Dataset

| Property       | Value                              |
|----------------|------------------------------------|
| **Records**    | 8,000 patients                     |
| **Features**   | 17 columns                         |
| **Target**     | `label` (1 = Readmitted, 0 = Not) |
| **Imbalance**  | 77.3% Readmitted / 22.7% Not      |

### Features

| Feature                    | Type        | Description                           |
|----------------------------|-------------|---------------------------------------|
| patient_id                 | Identifier  | Unique patient ID                     |
| admission_date             | Date        | Date of hospital admission            |
| season                     | Categorical | Spring / Summer / Fall / Winter       |
| age                        | Numerical   | Patient age (18–95)                   |
| gender                     | Categorical | Male / Female                         |
| region                     | Categorical | North / South / East / West / Central |
| primary_diagnosis          | Categorical | 11 diagnosis categories               |
| comorbidities_count        | Numerical   | Number of comorbidities (1–10)        |
| length_of_stay             | Numerical   | Hospital stay duration (days)         |
| treatment_type             | Categorical | Medical / Surgical / Interventional / Conservative |
| medications_count          | Numerical   | Number of medications prescribed      |
| followup_visits_last_year  | Numerical   | Follow-up visits in past year         |
| prev_readmissions          | Numerical   | Previous readmission count            |
| insurance_type             | Categorical | Medicare / Medicaid / Private / Uninsured |
| discharge_disposition      | Categorical | Home / Home Health / Skilled Nursing / Rehab |
| readmission_risk_score     | Numerical   | Clinical risk score (0.0–1.0)        |
| label                      | Target      | 1 = Readmitted, 0 = Not              |

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements_new.txt
```

### 2. Preprocess Data
```bash
python src/preprocess.py
```
This will:
- Load raw data from `data/raw/patient.csv`
- Handle missing values
- Engineer new features (age groups, risk flags, care intensity)
- Encode categorical variables
- Scale numerical features
- Save processed data and fitted transformers

### 3. Train Models
```bash
python src/train.py
```
This will:
- Train Logistic Regression, Random Forest, and Gradient Boosting
- Apply SMOTE to handle class imbalance
- Run 5-fold cross-validation
- Save the best model (by F1-score)
- Generate confusion matrices, ROC curves, and feature importance plots

### 4. Make Predictions
```bash
python src/predict.py
```
Runs a demo prediction with sample patient data.

### 5. Launch Dashboard
```bash
streamlit run app/app.py
```
Opens a web dashboard with:
- **Predict**: Enter patient details → get readmission risk
- **Model Performance**: View accuracy, precision, recall, F1, AUC-ROC
- **Data Explorer**: Interactive dataset visualizations

---

## 🧠 Models Trained

| Model                  | Approach                                    |
|------------------------|---------------------------------------------|
| Logistic Regression    | Baseline with balanced class weights        |
| Random Forest          | 200 trees, max_depth=15, balanced weights   |
| Gradient Boosting      | 200 estimators, learning_rate=0.1           |

Best model is selected based on **F1-score** on the test set.

---

## 🛠️ Tools & Technologies

- **Python 3.12+**
- **Pandas, NumPy** — Data manipulation
- **Scikit-learn** — ML models & evaluation
- **Imbalanced-learn** — SMOTE oversampling
- **Matplotlib, Seaborn** — Visualizations
- **Streamlit** — Interactive web app
- **Joblib** — Model serialization

---

## 📜 License

This project is for educational and research purposes.

---

*Built with ❤️ by Jeevan*
