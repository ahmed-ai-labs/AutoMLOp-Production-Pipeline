"""
Model deployment and serving utilities.
"""

import json
import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelDeployment:
    """Utilities for model deployment and serving."""
    
    def __init__(self, model_path: str = "models"):
        self.model_path = Path(model_path)
        self.model = None
        self.metadata = {}
        
    def load_model(self, model_filename: str) -> Any:
        """Load a trained model from disk."""
        model_file = self.model_path / model_filename
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        try:
            if model_filename.endswith('.pkl'):
                self.model = joblib.load(model_file)
            elif model_filename.endswith('.json'):
                with open(model_file, 'r') as f:
                    self.model = json.load(f)
            else:
                # Try pickle as fallback
                with open(model_file, 'rb') as f:
                    self.model = pickle.load(f)
            
            logger.info(f"Model loaded successfully from {model_file}")
            
            # Load metadata if available
            metadata_file = model_file.parent / f"{model_file.stem}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def save_model(self, model: Any, model_filename: str, metadata: Dict = None) -> None:
        """Save a model to disk with metadata."""
        self.model_path.mkdir(parents=True, exist_ok=True)
        model_file = self.model_path / model_filename
        
        try:
            if model_filename.endswith('.pkl'):
                joblib.dump(model, model_file)
            else:
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
            
            # Save metadata
            if metadata:
                metadata_file = model_file.parent / f"{model_file.stem}_metadata.json"
                metadata['saved_at'] = datetime.now().isoformat()
                metadata['model_file'] = str(model_file)
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Model saved to {model_file}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def predict(self, features: Union[List, np.ndarray, pd.DataFrame]) -> Union[float, np.ndarray]:
        """Make prediction using loaded model."""
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        try:
            # Convert input to appropriate format
            if isinstance(features, list):
                features = np.array(features).reshape(1, -1)
            elif isinstance(features, pd.DataFrame):
                features = features.values
            elif isinstance(features, np.ndarray) and features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(features)
            
            # Log prediction
            self.log_prediction(features, prediction)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def predict_proba(self, features: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Get prediction probabilities if model supports it."""
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability prediction")
        
        try:
            # Convert input to appropriate format
            if isinstance(features, list):
                features = np.array(features).reshape(1, -1)
            elif isinstance(features, pd.DataFrame):
                features = features.values
            elif isinstance(features, np.ndarray) and features.ndim == 1:
                features = features.reshape(1, -1)
            
            probabilities = self.model.predict_proba(features)
            return probabilities
            
        except Exception as e:
            logger.error(f"Error getting prediction probabilities: {str(e)}")
            raise
    
    def log_prediction(self, features: np.ndarray, prediction: np.ndarray) -> None:
        """Log prediction for monitoring."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'features': features.tolist() if hasattr(features, 'tolist') else features,
            'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
            'model_metadata': self.metadata
        }
        
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Append to predictions log file
        log_file = logs_dir / "predictions.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry, default=str) + '\n')
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "No model loaded"}
        
        info = {
            "model_type": type(self.model).__name__,
            "model_loaded": True,
            "metadata": self.metadata,
            "has_predict_proba": hasattr(self.model, 'predict_proba'),
            "model_attributes": [attr for attr in dir(self.model) if not attr.startswith('_')]
        }
        
        # Add model-specific information
        if hasattr(self.model, 'feature_importances_'):
            info["has_feature_importances"] = True
        
        if hasattr(self.model, 'n_features_in_'):
            info["n_features_in"] = self.model.n_features_in_
        
        return info

class ModelVersionManager:
    """Manage different versions of models."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
    
    def list_model_versions(self) -> List[Dict[str, Any]]:
        """List all available model versions."""
        versions = []
        
        for model_file in self.models_dir.glob("*.pkl"):
            metadata_file = model_file.parent / f"{model_file.stem}_metadata.json"
            
            version_info = {
                "filename": model_file.name,
                "path": str(model_file),
                "created": datetime.fromtimestamp(model_file.stat().st_ctime).isoformat(),
                "size_mb": round(model_file.stat().st_size / (1024 * 1024), 2)
            }
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    version_info["metadata"] = metadata
                except:
                    pass
            
            versions.append(version_info)
        
        # Sort by creation time (newest first)
        versions.sort(key=lambda x: x["created"], reverse=True)
        return versions
    
    def get_latest_model(self) -> str:
        """Get the filename of the latest model."""
        versions = self.list_model_versions()
        if not versions:
            raise FileNotFoundError("No models found")
        
        return versions[0]["filename"]
    
    def delete_old_versions(self, keep_count: int = 5) -> List[str]:
        """Delete old model versions, keeping only the specified number."""
        versions = self.list_model_versions()
        deleted_files = []
        
        if len(versions) > keep_count:
            for version in versions[keep_count:]:
                try:
                    model_file = Path(version["path"])
                    model_file.unlink()
                    
                    # Also delete metadata file if exists
                    metadata_file = model_file.parent / f"{model_file.stem}_metadata.json"
                    if metadata_file.exists():
                        metadata_file.unlink()
                    
                    deleted_files.append(version["filename"])
                    logger.info(f"Deleted old model version: {version['filename']}")
                    
                except Exception as e:
                    logger.error(f"Error deleting {version['filename']}: {str(e)}")
        
        return deleted_files

def create_model_artifact(model: Any, model_name: str, version: str, 
                         metrics: Dict[str, float] = None, 
                         feature_names: List[str] = None) -> Dict[str, Any]:
    """Create a complete model artifact with metadata."""
    
    # Create deployment instance
    deployment = ModelDeployment()
    
    # Prepare metadata
    metadata = {
        "model_name": model_name,
        "version": version,
        "created_at": datetime.now().isoformat(),
        "model_type": type(model).__name__,
        "metrics": metrics or {},
        "feature_names": feature_names or [],
        "deployment_ready": True
    }
    
    # Save model with metadata
    model_filename = f"{model_name}_v{version}.pkl"
    deployment.save_model(model, model_filename, metadata)
    
    return {
        "model_filename": model_filename,
        "metadata": metadata,
        "deployment": deployment
    }
