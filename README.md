# Heart Attack Prediction System

A comprehensive machine learning system for predicting 10-year heart attack risk using the Framingham Heart Study dataset. Includes EDA, supervised and unsupervised models, and an interactive GUI.

## Features

### 1. **Exploratory Data Analysis (EDA)**
- Comprehensive data exploration: distributions, correlations, missing values
- Class imbalance analysis
- EDA visualizations saved to `eda_output/`

### 2. **Supervised Learning Models**
- **Logistic Regression + Random Forest** with GridSearchCV hyperparameter optimization
- Preprocessing pipeline with:
  - Imputation (median for numeric, most frequent for categorical)
  - StandardScaler for numeric features
  - OneHotEncoder for categorical features
  - SMOTE for class imbalance handling
- Model evaluation: ROC-AUC, classification report, confusion matrix, ROC curve

### 3. **Unsupervised Learning Model**
- **K-Means Clustering** to discover natural patient groupings
- Optimal cluster selection using Silhouette and Davies-Bouldin scores
- Cluster analysis with CHD distribution per cluster
- Helps understand patient segments and risk profiles

### 4. **Interactive GUI**
- User-friendly Tkinter interface for heart attack risk prediction
- Input 16 patient health features
- Real-time probability prediction and risk classification
- Patient cluster assignment (optional)
- Information tab with feature descriptions

## Installation

### Prerequisites
- Python 3.7+

### Setup

1. Clone or navigate to the project directory:
```bash
cd /path/to/Heart-Prediction-system-project-using-ML
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Run EDA
Explore the dataset and generate visualization:
```bash
python 01_eda.py
```
This generates plots in `eda_output/` directory (missing values, class distribution, correlations, etc.)

### Step 2: Train Supervised Model
Train Logistic Regression and Random Forest models:
```bash
python 02_train_supervised_model.py
```
This saves:
- `models/supervised_model.pkl` — best fitted pipeline
- `models/supervised_metadata.json` — model performance metrics
- `models/supervised_roc_curve.png` — ROC curve

### Step 3: Train Unsupervised Model
Train K-Means clustering model:
```bash
python 03_train_unsupervised_model.py
```
This saves:
- `models/unsupervised_model.pkl` — fitted K-Means model
- `models/kmeans_scaler.pkl` — feature scaler
- `models/cluster_analysis.txt` — cluster characteristics
- `models/cluster_distribution.png` — visualization

### Step 4: Launch GUI
Run the interactive prediction system:
```bash
python app_gui.py
```

The GUI will:
- Display input fields for 16 patient features
- Show real-time heart attack probability prediction
- Classify risk level (Low, Moderate, High)
- Assign patient to a cluster (if K-Means available)

### Alternative: CLI Prediction
Test the predictor from command line:
```bash
python heart_predictor.py
```

## Dataset

**Framingham Heart Study** (`framingham.csv`)
- 4,240 patient records
- 16 features (clinical and demographic)
- Target: `TenYearCHD` (10-year coronary heart disease risk)
- Class distribution: ~85% No CHD, ~15% CHD (imbalanced)

### Features

**Numeric:**
- age, cigsPerDay, totChol, sysBP, diaBP, BMI, heartRate, glucose

**Categorical (Binary):**
- male, currentSmoker, prevalentStroke, prevalentHyp, diabetes, BPMeds, education

## Model Performance

### Supervised Model (Logistic Regression + Random Forest)
- **Test ROC-AUC:** 0.6994
- **Accuracy:** 0.66
- **CHD Recall:** 0.61 (identifies 61% of actual CHD cases)
- **CHD Precision:** 0.25 (25% of predicted CHD cases are true positives)

**Note:** Precision can be improved by raising the decision threshold. See `models/supervised_metadata.json` for detailed metrics.

### Unsupervised Model (K-Means)
- **Optimal Clusters:** 3
- **Silhouette Score:** 0.217
- **Cluster 0:** ~50% CHD rate (high-risk patients)
- **Cluster 1:** ~17% CHD rate (moderate-risk patients)
- **Cluster 2:** ~8% CHD rate (low-risk patients)

## Output Files

```
eda_output/
  ├── 01_missing_values.png
  ├── 02_class_distribution.png
  ├── 03_feature_distributions.png
  ├── 04_violin_plots.png
  ├── 05_correlation_heatmap.png
  ├── 06_target_correlations.png
  └── summary_statistics.txt

models/
  ├── supervised_model.pkl
  ├── supervised_metadata.json
  ├── supervised_roc_curve.png
  ├── unsupervised_model.pkl
  ├── kmeans_scaler.pkl
  ├── unsupervised_metadata.json
  ├── cluster_analysis.txt
  ├── cluster_evaluation.png
  └── cluster_distribution.png
```

## Risk Assessment Guide

The GUI displays risk levels based on predicted probability:
- **Low Risk (< 30%):** Monitor lifestyle factors; no immediate intervention needed
- **Moderate Risk (30-60%):** Consider preventive measures; consult healthcare provider
- **High Risk (> 60%):** Seek medical consultation; may require aggressive prevention/treatment

## Important Notes

1. **Clinical Use:** This model is for educational and research purposes only. Always consult healthcare professionals for medical decisions.

2. **Model Limitations:**
   - Based on historical Framingham data; may not generalize to all populations
   - Class imbalance means lower positive-class precision; consider using different thresholds based on use case
   - Feature engineering opportunities (interaction terms, domain knowledge) could improve performance

3. **Threshold Tuning:** The default decision threshold is 0.5. For higher precision (fewer false alarms), increase threshold; for higher recall (catch more cases), decrease threshold.

## Files Description

- `01_eda.py` — Exploratory data analysis script
- `02_train_supervised_model.py` — Supervised model training (LogisticRegression + RandomForest)
- `03_train_unsupervised_model.py` — Unsupervised model training (K-Means)
- `heart_predictor.py` — Prediction module (loads models, makes predictions)
- `app_gui.py` — Interactive Tkinter GUI application
- `framingham.csv` — Dataset
- `requirements.txt` — Python dependencies

## Troubleshooting

**Models not found when running GUI:**
- Ensure you've run steps 2 and 3 (train supervised and unsupervised models)
- Check that `models/` folder contains `.pkl` files

**Import errors:**
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check Python version (3.7+): `python --version`

**GUI doesn't start:**
- On macOS, try: `python app_gui.py`
- Ensure Tkinter is installed (usually included with Python)

## License

Educational use only. Dataset from Framingham Heart Study (public).

## Contact & Support

For issues or questions, please refer to:
- Framingham Heart Study documentation
- scikit-learn documentation (https://scikit-learn.org)
- Imbalanced-learn documentation (https://imbalanced-learn.org)

---

**Version:** 1.0 | **Date:** May 2026
