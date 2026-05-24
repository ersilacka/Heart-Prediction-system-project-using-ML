"""
Train supervised models (Logistic Regression and Random Forest) with GridSearchCV.
Evaluate both models and save the best one to models/ folder.
"""

import os
import logging
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

os.makedirs('models', exist_ok=True)


def load_and_preprocess(df_path='framingham.csv'):
    """Load data and handle missing values."""
    logger.info('Loading data')
    df = pd.read_csv(df_path)
    
    # Define columns
    num_cols = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    cat_cols = ['male', 'currentSmoker', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'BPMeds', 'education']
    
    # Impute missing values
    num_imputer = SimpleImputer(strategy='median')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    
    X = df.drop('TenYearCHD', axis=1)
    y = df['TenYearCHD']
    
    return X, y, num_cols, cat_cols


def build_pipeline(clf):
    """Build preprocessing + SMOTE + classifier pipeline."""
    num_cols = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    cat_cols = ['male', 'currentSmoker', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'BPMeds', 'education']
    
    numeric_transformer = StandardScaler()
    try:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols),
        ],
        remainder='drop',
    )
    
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', clf)
    ])
    
    return pipe


def train_models(X, y):
    """Train and evaluate supervised models with GridSearchCV."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f'Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}')
    
    # Define parameter grid
    param_grid = [
        {
            'clf': [LogisticRegression(random_state=42, max_iter=2000)],
            'clf__C': [0.01, 0.1, 1.0, 10],
            'clf__penalty': ['l2'],
            'clf__solver': ['lbfgs'],
        },
        {
            'clf': [RandomForestClassifier(random_state=42)],
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [5, 10, None],
            'clf__min_samples_split': [2, 5],
        },
    ]
    
    # Build pipeline with dummy classifier for GridSearch
    pipe = build_pipeline(LogisticRegression())
    
    # GridSearchCV
    logger.info('Starting GridSearchCV')
    gs = GridSearchCV(
        pipe,
        param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    
    gs.fit(X_train, y_train)
    logger.info(f'Best CV score: {gs.best_score_:.4f}')
    
    best_pipe = gs.best_estimator_
    
    # Evaluate on test set
    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    y_pred = best_pipe.predict(X_test)
    test_auc = roc_auc_score(y_test, y_proba)
    
    logger.info(f'Test ROC-AUC: {test_auc:.4f}')
    
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, target_names=['No CHD', 'CHD']))
    
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {test_auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Supervised Model')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('models/supervised_roc_curve.png', dpi=150)
    plt.close()
    logger.info('Saved ROC curve')
    
    # Save model and metadata
    model_path = 'models/supervised_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_pipe, f)
    logger.info(f'Saved model to {model_path}')
    
    metadata = {
        'cv_score': float(gs.best_score_),
        'test_auc': float(test_auc),
        'best_params': str(gs.best_params_),
        'roc_auc': float(test_auc),
    }
    
    with open('models/supervised_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info('Saved metadata')
    
    return best_pipe, X_test, y_test, y_proba


def main():
    logger.info('Training supervised models')
    X, y, num_cols, cat_cols = load_and_preprocess()
    best_pipe, X_test, y_test, y_proba = train_models(X, y)
    logger.info('Supervised model training complete')


if __name__ == '__main__':
    main()
