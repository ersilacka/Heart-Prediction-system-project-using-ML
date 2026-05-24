

import logging
import os
from typing import Tuple, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

import joblib

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def load_data(path: str = 'framingham.csv') -> pd.DataFrame:
    logger.info('Loading data from %s', path)
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Columns assumed present in dataset - keep same convention as original script
    num_cols = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    cat_cols = ['male', 'currentSmoker', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'BPMeds', 'education']

    logger.info('Imputing missing values')
    num_imputer = SimpleImputer(strategy='median')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # Final check
    if df.isnull().sum().sum() != 0:
        raise ValueError('There are still missing values after imputation')

    X = df.drop('TenYearCHD', axis=1)
    y = df['TenYearCHD']
    return X, y


def train_and_select_model(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict]:
    """Build an imblearn Pipeline with scaling and SMOTE, then run GridSearchCV to choose best model.

    Returns the best pipeline (with fitted estimator) and the cv results dict.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    logger.info('Train/test split: %d train, %d test', X_train.shape[0], X_test.shape[0])

    # Build a ColumnTransformer that scales numeric features and one-hot-encodes categoricals
    num_cols = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    cat_cols = ['male', 'currentSmoker', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'BPMeds', 'education']

    numeric_transformer = StandardScaler()
    # use sparse_output for newer scikit-learn versions
    try:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        # fallback for older versions
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols),
        ],
        remainder='drop',
    )

    smote = SMOTE(random_state=random_state)

    # Pipeline: preprocessing -> SMOTE -> classifier
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('smote', smote), ('clf', LogisticRegression())])

    param_grid = [
        {
            'clf': [LogisticRegression(random_state=random_state, max_iter=2000)],
            'clf__C': [0.01, 0.1, 1.0, 10],
            'clf__penalty': ['l2'],
            'clf__solver': ['lbfgs'],
        },
        {
            'clf': [RandomForestClassifier(random_state=random_state)],
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [5, 10, None],
            'clf__min_samples_split': [2, 5],
        },
    ]

    gs = GridSearchCV(
        pipe,
        param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    logger.info('Starting GridSearchCV')
    gs.fit(X_train, y_train)

    logger.info('GridSearchCV done. Best score: %.4f', gs.best_score_)

    # Evaluate on test set
    best_pipe = gs.best_estimator_
    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)
    logger.info('Test ROC AUC for best estimator: %.4f', test_auc)

    cv_results = {
        'best_params': gs.best_params_,
        'best_score': gs.best_score_,
        'test_auc': test_auc,
        'X_test': X_test,
        'y_test': y_test,
        'y_proba': y_proba,
        'best_estimator': best_pipe,
    }

    return best_pipe, cv_results


def evaluate_and_plot(cv_results: Dict, output_dir: str = '.') -> None:
    os.makedirs(output_dir, exist_ok=True)
    X_test = cv_results['X_test']
    y_test = cv_results['y_test']
    y_proba = cv_results['y_proba']
    best = cv_results['best_estimator']

    y_pred = best.predict(X_test)

    auc = roc_auc_score(y_test, y_proba)
    logger.info('Final model AUC: %.4f', auc)

    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, target_names=['No CHD', 'CHD']))
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Best Model (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()
    logger.info('ROC curve saved to %s', roc_path)

    # If classifier has feature_importances_, save top features
    try:
        clf = best.named_steps['clf']
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            feat_imp = pd.DataFrame({'Feature': X_test.columns, 'Importance': importances}).sort_values(
                'Importance', ascending=False
            )
            fig_path = os.path.join(output_dir, 'feature_importance.png')
            plt.figure(figsize=(10, 8))
            plt.barh(feat_imp.head(10)['Feature'], feat_imp.head(10)['Importance'], color='steelblue')
            plt.xlabel('Importance Score')
            plt.title('Top 10 Feature Importances')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
            logger.info('Feature importance plot saved to %s', fig_path)
            print('\nTop features:')
            print(feat_imp.head(10))
    except Exception:
        logger.debug('No feature importance available for the selected estimator')


def precision_recall_analysis(cv_results: Dict, output_dir: str = '.') -> Dict:
    """Compute precision-recall curve, find thresholds, plot, and evaluate at selected thresholds.

    Returns a dict with selected thresholds and corresponding metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    X_test = cv_results['X_test']
    y_test = cv_results['y_test']
    y_proba = cv_results['y_proba']
    best = cv_results['best_estimator']

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    # thresholds length is len(precision)-1; append 1.0 for plotting ease
    # Compute F1 for each threshold
    # For mapping, use thresholds array; for each threshold t, get p/r where y_proba >= t
    f1_scores = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
    # (compute precision and recall manually below)
        # compute precision and recall manually
        tp = int(((preds == 1) & (y_test.values == 1)).sum())
        fp = int(((preds == 1) & (y_test.values == 0)).sum())
        fn = int(((preds == 0) & (y_test.values == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1_scores.append(f1)

    f1_scores = np.array(f1_scores)

    # Best F1 threshold
    if len(f1_scores) > 0:
        best_idx = f1_scores.argmax()
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
    else:
        best_threshold = 0.5
        best_f1 = None

    # Threshold that achieves precision >= 0.5 (if any) — pick the one with highest recall
    target_precision = 0.5
    candidate_idxs = []
    for i, t in enumerate(thresholds):
        preds = (y_proba >= t).astype(int)
        tp = int(((preds == 1) & (y_test.values == 1)).sum())
        fp = int(((preds == 1) & (y_test.values == 0)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        if p >= target_precision:
            candidate_idxs.append(i)

    if candidate_idxs:
        # pick candidate with highest recall
        recalls = []
        for i in candidate_idxs:
            t = thresholds[i]
            preds = (y_proba >= t).astype(int)
            tp = int(((preds == 1) & (y_test.values == 1)).sum())
            fn = int(((preds == 0) & (y_test.values == 1)).sum())
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls.append(r)
        best_prec_idx = candidate_idxs[int(np.argmax(recalls))]
        threshold_for_prec = thresholds[best_prec_idx]
    else:
        threshold_for_prec = None

    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP = {ap:.3f}', linewidth=2)
    if best_threshold is not None:
        # Get precision/recall at best_threshold
        preds_b = (y_proba >= best_threshold).astype(int)
        tp = int(((preds_b == 1) & (y_test.values == 1)).sum())
        fp = int(((preds_b == 1) & (y_test.values == 0)).sum())
        fn = int(((preds_b == 0) & (y_test.values == 1)).sum())
        p_b = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_b = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        plt.scatter(r_b, p_b, marker='o', color='red', label=f'Best F1 (t={best_threshold:.3f})')
    if threshold_for_prec is not None:
        preds_p = (y_proba >= threshold_for_prec).astype(int)
        tp = int(((preds_p == 1) & (y_test.values == 1)).sum())
        fp = int(((preds_p == 1) & (y_test.values == 0)).sum())
        fn = int(((preds_p == 0) & (y_test.values == 1)).sum())
        p_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_p = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        plt.scatter(r_p, p_p, marker='x', color='green', label=f'Precision≥{target_precision} (t={threshold_for_prec:.3f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    pr_path = os.path.join(output_dir, 'precision_recall_curve.png')
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150)
    plt.close()
    logger.info('Precision-Recall curve saved to %s', pr_path)

    # Evaluate at selected thresholds and print
    results = {}
    if best_threshold is not None:
        preds = (y_proba >= best_threshold).astype(int)
        results['best_f1_threshold'] = best_threshold
        results['best_f1_score'] = best_f1
        results['best_f1_classification_report'] = classification_report(y_test, preds, target_names=['No CHD', 'CHD'])
        results['best_f1_confusion_matrix'] = confusion_matrix(y_test, preds).tolist()

    if threshold_for_prec is not None:
        preds = (y_proba >= threshold_for_prec).astype(int)
        results['precision_target_threshold'] = threshold_for_prec
        results['precision_target_classification_report'] = classification_report(y_test, preds, target_names=['No CHD', 'CHD'])
        results['precision_target_confusion_matrix'] = confusion_matrix(y_test, preds).tolist()

    # Save results to a small text file
    out_txt = os.path.join(output_dir, 'pr_thresholds.txt')
    with open(out_txt, 'w') as f:
        f.write(f'Average precision (AP): {ap:.4f}\n')
        if 'best_f1_threshold' in results:
            f.write(f"Best F1 threshold: {results['best_f1_threshold']:.4f} (F1 approx: {results['best_f1_score']:.4f})\n\n")
            f.write('Classification report at best F1 threshold:\n')
            f.write(results['best_f1_classification_report'] + '\n')
            f.write('Confusion matrix:\n')
            f.write(str(results['best_f1_confusion_matrix']) + '\n\n')
        if 'precision_target_threshold' in results:
            f.write(f"Threshold for precision≥{target_precision}: {results['precision_target_threshold']:.4f}\n\n")
            f.write('Classification report at precision target threshold:\n')
            f.write(results['precision_target_classification_report'] + '\n')
            f.write('Confusion matrix:\n')
            f.write(str(results['precision_target_confusion_matrix']) + '\n')

    logger.info('Saved precision-recall thresholds and reports to %s', out_txt)
    return results


def save_pipeline(pipeline: Pipeline, output_path: str = 'heart_attack_pipeline.pkl') -> None:
    joblib.dump(pipeline, output_path)
    logger.info('Saved pipeline to %s', output_path)


def main():
    logger.info('Starting EDA + modeling pipeline')
    df = load_data('framingham.csv')

    print('=' * 50)
    print('INITIAL DATASET OVERVIEW')
    print('=' * 50)
    print(f'Original shape: {df.shape[0]} rows, {df.shape[1]} columns')
    print(f'\nMissing values before imputation:\n{df.isnull().sum()}')
    print(f'Total missing values: {df.isnull().sum().sum()}')

    X, y = preprocess(df)

    best_pipe, cv_results = train_and_select_model(X, y)

    evaluate_and_plot(cv_results, output_dir='outputs')

    pr_results = precision_recall_analysis(cv_results, output_dir='outputs')
    # Print brief summary of PR threshold results
    if 'best_f1_threshold' in pr_results:
        print(f"\nBest F1 threshold: {pr_results['best_f1_threshold']:.4f}")
    if 'precision_target_threshold' in pr_results:
        print(f"Threshold achieving precision≥0.5: {pr_results['precision_target_threshold']:.4f}")

    save_pipeline(best_pipe, output_path=os.path.join('outputs', 'heart_attack_pipeline.pkl'))

    # Save scaler separately if needed
    if 'scaler' in best_pipe.named_steps:
        joblib.dump(best_pipe.named_steps['scaler'], os.path.join('outputs', 'scaler.pkl'))

    print('\nDone. Artifacts saved to ./outputs')


if __name__ == '__main__':
    main()