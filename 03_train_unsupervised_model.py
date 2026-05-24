"""
Train unsupervised model (K-Means clustering) to discover natural patient groupings.
Analyze cluster characteristics and CHD distribution within clusters.
"""

import os
import logging
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns

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
    
    return X, y, df


def find_optimal_clusters(X, max_clusters=10):
    """Use elbow method and silhouette score to find optimal cluster count."""
    logger.info('Finding optimal number of clusters')
    
    inertias = []
    silhouette_scores = []
    db_scores = []
    K_range = range(2, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        db_scores.append(davies_bouldin_score(X, kmeans.labels_))
        logger.info(f'k={k}: Silhouette={silhouette_scores[-1]:.3f}, DB={db_scores[-1]:.3f}')
    
    # Plot elbow curve
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score (Higher is Better)')
    axes[1].grid(alpha=0.3)
    
    axes[2].plot(K_range, db_scores, 'ro-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Number of Clusters (k)')
    axes[2].set_ylabel('Davies-Bouldin Score')
    axes[2].set_title('Davies-Bouldin Score (Lower is Better)')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/cluster_evaluation.png', dpi=150)
    plt.close()
    logger.info('Saved cluster evaluation plot')
    
    # Find best k (max silhouette)
    best_k = list(K_range)[np.argmax(silhouette_scores)]
    logger.info(f'Optimal k (by Silhouette): {best_k}')
    
    return best_k


def train_kmeans(X, k=3):
    """Train K-Means clustering model."""
    logger.info(f'Training K-Means with k={k}')
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    silhouette = silhouette_score(X, clusters)
    db_score = davies_bouldin_score(X, clusters)
    
    logger.info(f'Silhouette Score: {silhouette:.4f}')
    logger.info(f'Davies-Bouldin Score: {db_score:.4f}')
    
    return kmeans, clusters


def analyze_clusters(X, y, clusters, df_orig):
    """Analyze cluster characteristics and CHD distribution."""
    logger.info('Analyzing clusters')
    
    analysis_file = 'models/cluster_analysis.txt'
    with open(analysis_file, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('K-MEANS CLUSTERING ANALYSIS\n')
        f.write('=' * 80 + '\n\n')
        
        for cluster_id in np.unique(clusters):
            mask = clusters == cluster_id
            cluster_size = mask.sum()
            chd_count = y[mask].sum()
            chd_pct = (chd_count / cluster_size * 100) if cluster_size > 0 else 0
            
            f.write(f'Cluster {cluster_id}:\n')
            f.write(f'  Size: {cluster_size} ({cluster_size / len(clusters) * 100:.1f}%)\n')
            f.write(f'  CHD Cases: {chd_count} ({chd_pct:.1f}%)\n')
            f.write(f'  Feature means:\n')
            
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                mean_val = X.loc[mask, col].mean()
                f.write(f'    {col}: {mean_val:.2f}\n')
            f.write('\n')
    
    logger.info(f'Saved cluster analysis to {analysis_file}')
    
    # Plot cluster distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cluster_sizes = pd.Series(clusters).value_counts().sort_index()
    cluster_sizes.plot(kind='bar', ax=axes[0], color='steelblue')
    axes[0].set_title('Cluster Sizes')
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Number of Samples')
    
    # CHD distribution by cluster
    chd_by_cluster = []
    for cluster_id in np.unique(clusters):
        mask = clusters == cluster_id
        chd_count = y[mask].sum()
        total = mask.sum()
        chd_by_cluster.append(chd_count / total if total > 0 else 0)
    
    pd.Series(chd_by_cluster).plot(kind='bar', ax=axes[1], color='coral')
    axes[1].set_title('CHD Rate by Cluster')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('CHD Rate')
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('models/cluster_distribution.png', dpi=150)
    plt.close()
    logger.info('Saved cluster distribution plot')


def main():
    logger.info('Training unsupervised model (K-Means)')
    
    # Load data
    X, y, df = load_and_preprocess()
    
    # Standardize features for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal clusters
    best_k = find_optimal_clusters(X_scaled, max_clusters=10)
    
    # Train K-Means with optimal k
    kmeans, clusters = train_kmeans(X_scaled, k=best_k)
    
    # Analyze clusters
    analyze_clusters(X, y, clusters, df)
    
    # Save model
    model_path = 'models/unsupervised_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(kmeans, f)
    logger.info(f'Saved K-Means model to {model_path}')
    
    # Save scaler
    scaler_path = 'models/kmeans_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f'Saved scaler to {scaler_path}')
    
    # Save metadata
    metadata = {
        'n_clusters': int(best_k),
        'algorithm': 'K-Means',
    }
    with open('models/unsupervised_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info('Unsupervised model training complete')


if __name__ == '__main__':
    main()
