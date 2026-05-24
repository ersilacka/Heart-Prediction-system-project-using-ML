"""
Comprehensive Exploratory Data Analysis (EDA) for Framingham Heart Study data.
Analyzes distributions, correlations, missing values, and class imbalance.
Outputs plots and summary statistics to eda_output/ folder.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

OUTPUT_DIR = 'eda_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path='framingham.csv'):
    """Load the Framingham dataset."""
    logger.info(f'Loading data from {path}')
    df = pd.read_csv(path)
    return df


def plot_missing_values(df, output_dir=OUTPUT_DIR):
    """Plot missing values."""
    fig, ax = plt.subplots(figsize=(12, 6))
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        missing.plot(kind='barh', ax=ax, color='coral')
        ax.set_title('Missing Values by Feature')
        ax.set_xlabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '01_missing_values.png'), dpi=150)
        plt.close()
        logger.info(f'Saved missing values plot to {output_dir}/01_missing_values.png')


def plot_class_distribution(df, target_col='TenYearCHD', output_dir=OUTPUT_DIR):
    """Plot target class distribution."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot
    counts = df[target_col].value_counts()
    counts.plot(kind='bar', ax=ax[0], color=['steelblue', 'coral'])
    ax[0].set_title('10-Year CHD Event Count')
    ax[0].set_xticklabels(['No CHD', 'CHD'], rotation=0)
    ax[0].set_ylabel('Count')
    
    # Pie chart
    pct = df[target_col].value_counts(normalize=True) * 100
    ax[1].pie(pct, labels=['No CHD', 'CHD'], autopct='%1.1f%%', colors=['steelblue', 'coral'])
    ax[1].set_title('Class Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_class_distribution.png'), dpi=150)
    plt.close()
    logger.info(f'Saved class distribution plot to {output_dir}/02_class_distribution.png')


def plot_feature_distributions(df, target_col='TenYearCHD', output_dir=OUTPUT_DIR):
    """Plot histograms and violin plots for numeric features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]
    
    # Histograms
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    axes = axes.flatten()
    for idx, col in enumerate(numeric_cols):
        axes[idx].hist(df[col].dropna(), bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'{col}')
        axes[idx].set_ylabel('Frequency')
    for idx in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[idx])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_feature_distributions.png'), dpi=150)
    plt.close()
    logger.info(f'Saved feature distributions to {output_dir}/03_feature_distributions.png')
    
    # Violin plots (split by target class)
    n_cols = 4
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes = axes.flatten() if len(numeric_cols) > 1 else [axes]
    for idx, col in enumerate(numeric_cols):
        data_to_plot = [df[df[target_col] == 0][col].dropna(), df[df[target_col] == 1][col].dropna()]
        axes[idx].violinplot(data_to_plot, positions=[0, 1], showmeans=True)
        axes[idx].set_xticks([0, 1])
        axes[idx].set_xticklabels(['No CHD', 'CHD'])
        axes[idx].set_title(f'{col} by CHD Status')
        axes[idx].set_ylabel('Value')
    for idx in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[idx])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_violin_plots.png'), dpi=150)
    plt.close()
    logger.info(f'Saved violin plots to {output_dir}/04_violin_plots.png')


def plot_correlation_heatmap(df, target_col='TenYearCHD', output_dir=OUTPUT_DIR):
    """Plot correlation heatmap."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                linewidths=0.5, cbar_kws={'label': 'Correlation'})
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_correlation_heatmap.png'), dpi=150)
    plt.close()
    logger.info(f'Saved correlation heatmap to {output_dir}/05_correlation_heatmap.png')
    
    # Top correlations with target
    if target_col in corr_matrix.columns:
        target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        target_corr.plot(kind='barh', color='steelblue')
        plt.title(f'Feature Correlations with {target_col}')
        plt.xlabel('Correlation Coefficient')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '06_target_correlations.png'), dpi=150)
        plt.close()
        logger.info(f'Saved target correlations to {output_dir}/06_target_correlations.png')


def print_summary_statistics(df, target_col='TenYearCHD', output_dir=OUTPUT_DIR):
    """Print and save summary statistics."""
    summary_file = os.path.join(output_dir, 'summary_statistics.txt')
    with open(summary_file, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('FRAMINGHAM HEART STUDY - EDA SUMMARY\n')
        f.write('=' * 80 + '\n\n')
        
        f.write(f'Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n')
        
        f.write('Missing Values:\n')
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing) == 0:
            f.write('No missing values\n\n')
        else:
            for col, count in missing.items():
                pct = count / len(df) * 100
                f.write(f'  {col}: {count} ({pct:.2f}%)\n')
            f.write('\n')
        
        f.write('Class Distribution (Target):\n')
        counts = df[target_col].value_counts()
        for label, count in counts.items():
            pct = count / len(df) * 100
            label_str = 'CHD' if label == 1 else 'No CHD'
            f.write(f'  {label_str}: {count} ({pct:.2f}%)\n')
        f.write('\n')
        
        f.write('Feature Statistics:\n')
        f.write(df.describe().to_string())
        f.write('\n\n')
        
        f.write('Numeric Feature Correlations with Target:\n')
        numeric_df = df.select_dtypes(include=[np.number])
        if target_col in numeric_df.columns:
            target_corr = numeric_df.corr()[target_col].drop(target_col).sort_values(ascending=False)
            for feat, corr in target_corr.items():
                f.write(f'  {feat}: {corr:.4f}\n')
    
    logger.info(f'Saved summary statistics to {summary_file}')


def main():
    logger.info('Starting EDA for Heart Prediction System')
    df = load_data('framingham.csv')
    
    # Print initial info
    print('\n' + '=' * 80)
    print('DATASET OVERVIEW')
    print('=' * 80)
    print(f'Shape: {df.shape[0]} rows, {df.shape[1]} columns')
    print(f'Columns: {list(df.columns)}')
    print(f'\nFirst few rows:\n{df.head()}')
    print(f'\nData types:\n{df.dtypes}')
    print(f'\nMissing values:\n{df.isnull().sum()}')
    print(f'Total missing: {df.isnull().sum().sum()}')
    
    # EDA plots
    plot_missing_values(df)
    plot_class_distribution(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    print_summary_statistics(df)
    
    print('\n' + '=' * 80)
    print('EDA COMPLETE')
    print('=' * 80)
    print(f'Plots and statistics saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
