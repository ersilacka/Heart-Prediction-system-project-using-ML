"""
Heart Attack Prediction Module
Loads the trained supervised model and provides prediction interface.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Define feature columns
NUM_COLS = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
CAT_COLS = ['male', 'currentSmoker', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'BPMeds', 'education']


class HeartAttackPredictor:
    """Predict 10-year heart attack risk using trained supervised model."""
    
    def __init__(self, model_path='models/supervised_model.pkl'):
        """Initialize predictor with trained model."""
        self.model = None
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained pipeline model."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f'Model not found at {model_path}')
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f'Loaded model from {model_path}')
    
    def predict_proba(self, features_dict, threshold=0.5):
        """
        Predict heart attack probability for a single patient.
        
        Args:
            features_dict: Dictionary with feature names as keys and values
            threshold: Decision threshold (default 0.5)
        
        Returns:
            Dictionary with probability and risk assessment
        """
        # Create DataFrame from features
        df = pd.DataFrame([features_dict])
        
        # Get probability prediction
        proba = self.model.predict_proba(df)[0, 1]
        
        # Get binary prediction
        pred = self.model.predict(df)[0]
        
        # Risk assessment
        if proba < 0.3:
            risk_level = 'Low'
        elif proba < 0.6:
            risk_level = 'Moderate'
        else:
            risk_level = 'High'
        
        return {
            'probability': float(proba),
            'risk_level': risk_level,
            'predicted_class': int(pred),
            'threshold': threshold,
        }
    
    def get_feature_info(self):
        """Return information about required features."""
        return {
            'numeric': NUM_COLS,
            'categorical': CAT_COLS,
        }


class KMeansClusterAnalyzer:
    """Analyze patient clusters using trained K-Means model."""
    
    def __init__(self, model_path='models/unsupervised_model.pkl', 
                 scaler_path='models/kmeans_scaler.pkl'):
        """Initialize analyzer with trained models."""
        self.kmeans = None
        self.scaler = None
        self.load_models(model_path, scaler_path)
    
    def load_models(self, model_path, scaler_path):
        """Load K-Means model and scaler."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f'Model not found at {model_path}')
        if not Path(scaler_path).exists():
            raise FileNotFoundError(f'Scaler not found at {scaler_path}')
        
        with open(model_path, 'rb') as f:
            self.kmeans = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f'Loaded K-Means model and scaler')
    
    def get_cluster(self, features_dict):
        """Get cluster assignment for a single patient."""
        df = pd.DataFrame([features_dict])
        # Scale numeric features using the same scaler as training
        numeric_features = df[NUM_COLS]
        scaled = self.scaler.transform(numeric_features)
        cluster = self.kmeans.predict(scaled)[0]
        return int(cluster)


def create_sample_features():
    """Create a sample feature dictionary for testing."""
    return {
        'male': 1,
        'age': 50,
        'education': 2.0,
        'currentSmoker': 0,
        'cigsPerDay': 0.0,
        'BPMeds': 0.0,
        'prevalentStroke': 0,
        'prevalentHyp': 0,
        'diabetes': 0,
        'totChol': 200.0,
        'sysBP': 120.0,
        'diaBP': 80.0,
        'BMI': 25.0,
        'heartRate': 70.0,
        'glucose': 100.0,
    }


if __name__ == '__main__':
    # Test the predictor
    try:
        predictor = HeartAttackPredictor()
        sample = create_sample_features()
        result = predictor.predict_proba(sample)
        print(f'\nSample Prediction Result:')
        print(f'  Heart Attack Probability: {result["probability"]:.2%}')
        print(f'  Risk Level: {result["risk_level"]}')
        print(f'  Predicted Class: {result["predicted_class"]}')
    except FileNotFoundError as e:
        print(f'Error: {e}')
        print('Please train the model first using 02_train_supervised_model.py')
