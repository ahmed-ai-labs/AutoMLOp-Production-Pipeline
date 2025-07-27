import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import mlflow
import mlflow.sklearn
import yaml
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path="configs/config.yaml"):
        """Initialize the model trainer with configuration"""
        self.config = self.load_config(config_path)
        self.model = None
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        return {}
    
    def load_data(self, data_path="data/processed/train.csv"):
        """Load training data"""
        try:
            if os.path.exists(data_path):
                data = pd.read_csv(data_path)
                logger.info(f"Data loaded successfully from {data_path}")
                return data
            else:
                # Generate sample data if no data file exists
                logger.warning(f"Data file not found at {data_path}. Generating sample data.")
                return self.generate_sample_data()
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return self.generate_sample_data()
    
    def generate_sample_data(self, n_samples=1000):
        """Generate sample data for demonstration"""
        np.random.seed(42)
        X = np.random.randn(n_samples, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        # Create feature names
        feature_names = [f"feature_{i}" for i in range(10)]
        
        # Create DataFrame
        data = pd.DataFrame(X, columns=feature_names)
        data['target'] = y
        
        logger.info(f"Generated sample data with {n_samples} samples and {len(feature_names)} features")
        return data
    
    def prepare_data(self, data):
        """Prepare data for training"""
        model_config = self.config.get('model', {})
        target_column = model_config.get('target_column', 'target')
        
        # Separate features and target
        if target_column in data.columns:
            X = data.drop(columns=[target_column])
            y = data[target_column]
        else:
            # If target column not specified, assume last column is target
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
        
        # Split data
        training_config = model_config.get('training', {})
        test_size = training_config.get('test_size', 0.2)
        random_state = training_config.get('random_state', 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train the model"""
        model_config = self.config.get('model', {})
        hyperparameters = model_config.get('hyperparameters', {})
        
        # Initialize model with hyperparameters
        self.model = RandomForestClassifier(
            n_estimators=hyperparameters.get('n_estimators', 100),
            max_depth=hyperparameters.get('max_depth', 10),
            random_state=42
        )
        
        # Train model
        logger.info("Starting model training...")
        self.model.fit(X_train, y_train)
        logger.info("Model training completed")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score']
        }
        
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
    
    def save_model(self, model_path="models/model.pkl"):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        try:
            # Set up MLflow
            mlflow_config = self.config.get('mlflow', {})
            mlflow.set_tracking_uri(mlflow_config.get('tracking_uri', 'http://localhost:5000'))
            mlflow.set_experiment(mlflow_config.get('experiment_name', 'default'))
            
            with mlflow.start_run():
                # Load and prepare data
                data = self.load_data()
                X_train, X_test, y_train, y_test = self.prepare_data(data)
                
                # Log dataset info
                mlflow.log_param("n_samples", len(data))
                mlflow.log_param("n_features", X_train.shape[1])
                
                # Train model
                model = self.train_model(X_train, y_train)
                
                # Log hyperparameters
                hyperparameters = self.config.get('model', {}).get('hyperparameters', {})
                for param, value in hyperparameters.items():
                    mlflow.log_param(param, value)
                
                # Evaluate model
                metrics = self.evaluate_model(X_test, y_test)
                
                # Log metrics
                for metric, value in metrics.items():
                    mlflow.log_metric(metric, value)
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                # Save model locally
                self.save_model()
                
                logger.info("Training pipeline completed successfully")
                return metrics
                
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            raise

if __name__ == "__main__":
    trainer = ModelTrainer()
    metrics = trainer.run_training_pipeline()
    print(f"Training completed with metrics: {metrics}")
