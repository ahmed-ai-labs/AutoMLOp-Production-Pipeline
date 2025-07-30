# src/models/train.py
"""
MLflow Model Training Pipeline
Production-grade model training with experiment tracking
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import yaml
import joblib
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLOpsModelTrainer:
    """Production-grade model training with MLflow integration"""
    
    def __init__(self, config_path="configs/config.yaml"):
        """Initialize trainer with configuration"""
        self.config = self._load_config(config_path)
        self.model = None
        self.preprocessors = {}
        self.metrics = {}
        
        # Set up MLflow
        self._setup_mlflow()
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._default_config()
    
    def _default_config(self):
        """Default configuration"""
        return {
            'data': {
                'raw_data_path': 'data/raw/',
                'processed_data_path': 'data/processed/',
                'test_size': 0.2,
                'random_state': 42
            },
            'model': {
                'algorithm': 'RandomForest',
                'n_estimators': 100,
                'max_depth': 10
            },
            'mlflow': {
                'experiment_name': 'customer_churn_prediction',
                'tracking_uri': 'http://localhost:5000'
            }
        }
    
    def _setup_mlflow(self):
        """Setup MLflow experiment tracking"""
        try:
            # Set tracking URI (local for now)
            mlflow.set_tracking_uri("file:./mlruns")
            
            # Set or create experiment
            experiment_name = self.config['mlflow']['experiment_name']
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new MLflow experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {experiment_name}")
            
            mlflow.set_experiment(experiment_name)
            
        except Exception as e:
            logger.error(f"MLflow setup failed: {e}")
            logger.info("Continuing without MLflow tracking...")
    
    def load_data(self, filename="customer_churn.csv"):
        """
        Load training data
        
        Args:
            filename (str): Data filename
            
        Returns:
            pd.DataFrame: Loaded data
        """
        filepath = Path(self.config['data']['raw_data_path']) / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        logger.info(f"Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess data for model training
        
        Args:
            df (pd.DataFrame): Raw data
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        logger.info("Starting data preprocessing...")
        
        # Separate features and target
        X = df.drop(['customerID', 'Churn'], axis=1)
        y = df['Churn']
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        
        # Encode categorical variables
        X_encoded = X.copy()
        for col in categorical_columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
            self.preprocessors[f'{col}_encoder'] = le
        
        # Scale numerical features
        scaler = StandardScaler()
        X_encoded[numerical_columns] = scaler.fit_transform(X[numerical_columns])
        self.preprocessors['scaler'] = scaler
        
        # Encode target variable
        y_encoder = LabelEncoder()
        y_encoded = y_encoder.fit_transform(y)
        self.preprocessors['target_encoder'] = y_encoder
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y_encoded
        )
        
        logger.info(f"Data preprocessing completed.")
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, model_type=None):
        """
        Train machine learning model
        
        Args:
            X_train: Training features
            y_train: Training target
            model_type (str): Type of model to train
            
        Returns:
            trained model
        """
        if model_type is None:
            model_type = self.config['model']['algorithm']
        
        logger.info(f"Training {model_type} model...")
        
        if model_type == 'RandomForest':
            model = RandomForestClassifier(
                n_estimators=self.config['model']['n_estimators'],
                max_depth=self.config['model']['max_depth'],
                random_state=self.config['data']['random_state'],
                n_jobs=-1
            )
        elif model_type == 'LogisticRegression':
            model = LogisticRegression(
                random_state=self.config['data']['random_state'],
                max_iter=1000
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train model
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.model = model
        self.metrics['training_time'] = training_time
        
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        return model
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate trained model
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        logger.info("Evaluating model performance...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'training_time': self.metrics['training_time']
        }
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            metrics['feature_importance'] = self.model.feature_importances_.tolist()
        
        self.metrics.update(metrics)
        
        # Print results
        print("\n📊 MODEL EVALUATION RESULTS")
        print("=" * 50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        print(f"Training Time: {metrics['training_time']:.2f} seconds")
        
        logger.info("Model evaluation completed")
        return metrics
    
    def save_model(self, model_name=None):
        """
        Save trained model and preprocessors
        
        Args:
            model_name (str): Name for saved model
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        if model_name is None:
            model_name = f"churn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Ensure models directory exists
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = models_dir / f"{model_name}.joblib"
        joblib.dump(self.model, model_path)
        
        # Save preprocessors
        preprocessors_path = models_dir / f"{model_name}_preprocessors.joblib"
        joblib.dump(self.preprocessors, preprocessors_path)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Preprocessors saved to: {preprocessors_path}")
        
        return model_path, preprocessors_path
    
    def run_experiment(self, run_name=None):
        """
        Run complete training experiment with MLflow tracking
        
        Args:
            run_name (str): Name for MLflow run
        """
        if run_name is None:
            run_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name) as run:
            try:
                # Log parameters
                mlflow.log_params({
                    'algorithm': self.config['model']['algorithm'],
                    'n_estimators': self.config['model'].get('n_estimators', 'N/A'),
                    'max_depth': self.config['model'].get('max_depth', 'N/A'),
                    'test_size': self.config['data']['test_size'],
                    'random_state': self.config['data']['random_state']
                })
                
                # Load and preprocess data
                df = self.load_data()
                X_train, X_test, y_train, y_test = self.preprocess_data(df)
                
                # Train model
                model = self.train_model(X_train, y_train)
                
                # Evaluate model
                metrics = self.evaluate_model(X_test, y_test)
                
                # Log metrics to MLflow
                mlflow.log_metrics({
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'roc_auc': metrics['roc_auc'],
                    'training_time': metrics['training_time']
                })
                
                # Save and log model
                model_paths = self.save_model()
                mlflow.sklearn.log_model(model, "model")
                
                # Log artifacts
                mlflow.log_artifact(str(model_paths[0]))
                mlflow.log_artifact(str(model_paths[1]))
                
                print(f"\n🚀 MLflow Run ID: {run.info.run_id}")
                print(f"📊 View results: mlflow ui")
                
                return run.info.run_id, metrics
                
            except Exception as e:
                logger.error(f"Experiment failed: {e}")
                raise

def main():
    """Main execution function"""
    print("🚀 Starting MLOps Model Training Pipeline...")
    
    # Initialize trainer
    trainer = MLOpsModelTrainer()
    
    # Run experiment
    run_id, metrics = trainer.run_experiment()
    
    print("\n🎉 TRAINING PIPELINE COMPLETED!")
    print("=" * 50)
    print(f"✅ Model Accuracy: {metrics['accuracy']:.2%}")
    print(f"✅ F1 Score: {metrics['f1_score']:.4f}")
    print(f"✅ ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"✅ Training Time: {metrics['training_time']:.2f}s")
    print(f"✅ MLflow Run ID: {run_id}")
    
    print("\n🎯 Next Steps:")
    print("1. Run: mlflow ui")
    print("2. Open: http://localhost:5000")
    print("3. Explore experiment results")
    print("4. Compare different model runs")
    
    return trainer, metrics

if __name__ == "__main__":
    trainer, metrics = main()