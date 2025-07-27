import pytest
import requests
import numpy as np
import json
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.train import ModelTrainer
from src.api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

class TestAPI:
    """Test cases for the ML API"""
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data
    
    def test_predict_endpoint_without_model(self):
        """Test prediction endpoint when no model is loaded"""
        test_data = {
            "features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "model_version": "test"
        }
        response = client.post("/predict", json=test_data)
        # Should return 503 if no model is loaded
        assert response.status_code in [503, 200]  # 200 if sample model exists
    
    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = client.get("/model/info")
        assert response.status_code in [503, 200]  # Depends on if model is loaded

class TestModelTrainer:
    """Test cases for the model trainer"""
    
    def test_model_trainer_initialization(self):
        """Test model trainer can be initialized"""
        trainer = ModelTrainer()
        assert trainer is not None
        assert trainer.config is not None
    
    def test_sample_data_generation(self):
        """Test sample data generation"""
        trainer = ModelTrainer()
        data = trainer.generate_sample_data(100)
        assert len(data) == 100
        assert 'target' in data.columns
        assert len(data.columns) == 11  # 10 features + 1 target
    
    def test_data_preparation(self):
        """Test data preparation"""
        trainer = ModelTrainer()
        data = trainer.generate_sample_data(100)
        X_train, X_test, y_train, y_test = trainer.prepare_data(data)
        
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        assert len(X_train) + len(X_test) == 100
    
    def test_model_training(self):
        """Test model training"""
        trainer = ModelTrainer()
        data = trainer.generate_sample_data(100)
        X_train, X_test, y_train, y_test = trainer.prepare_data(data)
        
        model = trainer.train_model(X_train, y_train)
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_model_evaluation(self):
        """Test model evaluation"""
        trainer = ModelTrainer()
        data = trainer.generate_sample_data(100)
        X_train, X_test, y_train, y_test = trainer.prepare_data(data)
        
        trainer.train_model(X_train, y_train)
        metrics = trainer.evaluate_model(X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1

class TestUtilities:
    """Test utility functions"""
    
    def test_config_loading(self):
        """Test configuration loading"""
        # Test with existing config
        if os.path.exists("configs/config.yaml"):
            trainer = ModelTrainer("configs/config.yaml")
            assert trainer.config is not None
        
        # Test with non-existent config
        trainer = ModelTrainer("non_existent_config.yaml")
        assert trainer.config == {}
    
    def test_prediction_format(self):
        """Test prediction data format"""
        test_features = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        # Test feature array shape
        features_array = np.array(test_features).reshape(1, -1)
        assert features_array.shape == (1, 10)
        
        # Test JSON serialization
        test_data = {
            "features": test_features,
            "model_version": "test"
        }
        json_str = json.dumps(test_data)
        parsed_data = json.loads(json_str)
        assert parsed_data["features"] == test_features

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
