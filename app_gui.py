"""
Heart Attack Prediction GUI
A user-friendly Tkinter interface for heart attack risk prediction.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import sys

# Add current directory to path to import predictor
sys.path.insert(0, str(Path(__file__).parent))

from heart_predictor import HeartAttackPredictor, KMeansClusterAnalyzer, NUM_COLS, CAT_COLS


class HeartPredictionGUI:
    """GUI for heart attack prediction system."""
    
    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title('Heart Attack Risk Predictor')
        self.root.geometry('900x700')
        
        # Try to load models
        self.predictor = None
        self.analyzer = None
        self.init_models()
        
        if self.predictor is None:
            messagebox.showerror('Error', 'Failed to load prediction model. Please train models first.')
            self.root.destroy()
            return
        
        # Create main frame
        self.create_widgets()
    
    def init_models(self):
        """Initialize prediction models."""
        try:
            self.predictor = HeartAttackPredictor('models/supervised_model.pkl')
            try:
                self.analyzer = KMeansClusterAnalyzer('models/unsupervised_model.pkl', 
                                                       'models/kmeans_scaler.pkl')
            except Exception as e:
                print(f'Could not load K-Means model: {e}')
        except FileNotFoundError as e:
            print(f'Error loading models: {e}')
    
    def create_widgets(self):
        """Create GUI widgets."""
        # Title
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        
        title_label = ttk.Label(title_frame, text='10-Year Heart Attack Risk Predictor', 
                                font=('Arial', 14, 'bold'))
        title_label.pack()
        
        subtitle = ttk.Label(title_frame, text='Enter patient information to predict heart attack risk')
        subtitle.pack()
        
        # Create notebook (tabbed interface)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Prediction
        pred_frame = ttk.Frame(notebook)
        notebook.add(pred_frame, text='Prediction')
        self.create_prediction_tab(pred_frame)
        
        # Tab 2: Information
        info_frame = ttk.Frame(notebook)
        notebook.add(info_frame, text='Information')
        self.create_info_tab(info_frame)
    
    def create_prediction_tab(self, parent):
        """Create prediction input tab."""
        # Create scrollable frame
        canvas = tk.Canvas(parent, bg='white')
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            '<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all'))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Input fields
        self.entries = {}
        
        # Numeric fields
        numeric_frame = ttk.LabelFrame(scrollable_frame, text='Numeric Features', padding=10)
        numeric_frame.pack(fill=tk.X, padx=5, pady=5)
        
        defaults = {
            'age': 50, 'cigsPerDay': 0, 'totChol': 200, 'sysBP': 120,
            'diaBP': 80, 'BMI': 25, 'heartRate': 70, 'glucose': 100
        }
        
        for i, col in enumerate(NUM_COLS):
            frame = ttk.Frame(numeric_frame)
            frame.pack(fill=tk.X, padx=5, pady=3)
            
            label = ttk.Label(frame, text=col, width=15)
            label.pack(side=tk.LEFT)
            
            entry = ttk.Entry(frame, width=15)
            entry.insert(0, str(defaults.get(col, 0)))
            entry.pack(side=tk.LEFT, padx=5)
            
            self.entries[col] = entry
        
        # Categorical fields
        cat_frame = ttk.LabelFrame(scrollable_frame, text='Categorical Features (0=No, 1=Yes)', padding=10)
        cat_frame.pack(fill=tk.X, padx=5, pady=5)
        
        for i, col in enumerate(CAT_COLS):
            frame = ttk.Frame(cat_frame)
            frame.pack(fill=tk.X, padx=5, pady=3)
            
            label = ttk.Label(frame, text=col, width=15)
            label.pack(side=tk.LEFT)
            
            var = tk.IntVar(value=0)
            combo = ttk.Combobox(frame, textvariable=var, values=[0, 1], state='readonly', width=12)
            combo.pack(side=tk.LEFT, padx=5)
            
            self.entries[col] = var
        
        # Buttons frame
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        predict_btn = ttk.Button(button_frame, text='Predict', command=self.make_prediction)
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = ttk.Button(button_frame, text='Clear', command=self.clear_inputs)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Pack canvas and scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Results frame
        self.results_frame = ttk.LabelFrame(parent, text='Prediction Results', padding=10)
        self.results_frame.pack(fill=tk.X, padx=5, pady=5, after=canvas)
        
        self.results_text = tk.Text(self.results_frame, height=10, width=80, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True)
    
    def create_info_tab(self, parent):
        """Create information tab."""
        text = tk.Text(parent, wrap=tk.WORD, padx=10, pady=10)
        text.pack(fill=tk.BOTH, expand=True)
        
        info_text = """
HEART ATTACK RISK PREDICTOR

This application predicts the 10-year risk of heart attack based on patient health data.

FEATURES:

Numeric Inputs:
- Age: Patient age in years
- Cigarettes Per Day: Number of cigarettes smoked daily
- Total Cholesterol: Total cholesterol level (mg/dL)
- Systolic BP: Systolic blood pressure (mmHg)
- Diastolic BP: Diastolic blood pressure (mmHg)
- BMI: Body Mass Index (kg/m²)
- Heart Rate: Resting heart rate (beats/min)
- Glucose: Fasting glucose level (mg/dL)

Categorical Inputs (0=No, 1=Yes):
- Male: Patient gender
- Current Smoker: Is patient currently smoking
- Prevalent Stroke: History of stroke
- Prevalent Hypertension: History of hypertension
- Diabetes: Presence of diabetes
- BP Meds: Currently taking blood pressure medication
- Education: Education level

RISK LEVELS:
- Low: < 30% probability
- Moderate: 30-60% probability
- High: > 60% probability

MODEL INFO:
- Supervised Model: Logistic Regression + Random Forest (GridSearchCV optimized)
- Unsupervised Model: K-Means clustering for patient segmentation
- Data: Framingham Heart Study dataset

For questions or issues, contact your healthcare provider.
        """
        text.insert('1.0', info_text)
        text.config(state=tk.DISABLED)
    
    def make_prediction(self):
        """Make a prediction based on user inputs."""
        try:
            # Collect inputs
            features = {}
            
            for col in NUM_COLS:
                val = float(self.entries[col].get())
                features[col] = val
            
            for col in CAT_COLS:
                val = int(self.entries[col].get())
                features[col] = val
            
            # Make prediction
            result = self.predictor.predict_proba(features)
            
            # Get cluster if available
            cluster_info = ''
            if self.analyzer:
                try:
                    cluster = self.analyzer.get_cluster(features)
                    cluster_info = f'\nPatient Cluster: {cluster}'
                except Exception as e:
                    cluster_info = f'\n(Could not determine cluster: {e})'
            
            # Display results
            results_text = f"""
PREDICTION RESULTS
{'=' * 50}

Heart Attack Probability (10-year): {result['probability']:.1%}
Risk Level: {result['risk_level']}
Predicted Class: {'CHD Risk' if result['predicted_class'] == 1 else 'No CHD Risk'}

Risk Assessment Guide:
- Low Risk (< 30%): Monitor lifestyle factors
- Moderate Risk (30-60%): Consider preventive measures
- High Risk (> 60%): Consult healthcare provider
{cluster_info}

Note: This is a predictive model. Always consult with healthcare professionals for medical decisions.
            """
            
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete('1.0', tk.END)
            self.results_text.insert('1.0', results_text)
            self.results_text.config(state=tk.DISABLED)
            
        except ValueError as e:
            messagebox.showerror('Error', f'Invalid input: {e}')
    
    def clear_inputs(self):
        """Clear all input fields."""
        defaults = {
            'age': 50, 'cigsPerDay': 0, 'totChol': 200, 'sysBP': 120,
            'diaBP': 80, 'BMI': 25, 'heartRate': 70, 'glucose': 100
        }
        
        for col in NUM_COLS:
            self.entries[col].delete(0, tk.END)
            self.entries[col].insert(0, str(defaults.get(col, 0)))
        
        for col in CAT_COLS:
            self.entries[col].set(0)


def main():
    """Launch the GUI."""
    root = tk.Tk()
    app = HeartPredictionGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
