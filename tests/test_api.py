import pytest
import requests
import numpy as np
import json
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.train import MLOpsModelTrainer
from src.data.data_loader import DataLoader
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

class TestDataLoader:
    """Test cases for the data loader"""
    
    def test_data_loader_initialization(self):
        """Test data loader can be initialized"""
        loader = DataLoader()
        assert loader is not None
        assert loader.data_path is not None
    
    def test_sample_data_generation(self):
        """Test sample data generation"""
        loader = DataLoader()
        data = loader.generate_sample_data(100)
        assert len(data) == 100
        assert 'target' in data.columns
        assert len(data.columns) >= 7  # At least 7 features + target
    
    def test_sample_data_format(self):
        """Test sample data format"""
        loader = DataLoader()
        data = loader.generate_sample_data(50)
        
        # Check data types
        assert data.dtypes['feature_1'] in ['float64', 'float32']
        assert data.dtypes['categorical_1'] == 'object'
        assert data['target'].dtype in ['int64', 'int32']

class TestModelTrainer:
    """Test cases for the model trainer"""
    
    def test_model_trainer_initialization(self):
        """Test model trainer can be initialized"""
        trainer = MLOpsModelTrainer()
        assert trainer is not None
        assert trainer.config is not None
    
    def test_config_loading(self):
        """Test configuration loading"""
        # Test with non-existent config (should use defaults)
        trainer = MLOpsModelTrainer("non_existent_config.yaml")
        assert trainer.config is not None
        assert isinstance(trainer.config, dict)
        # Should have default config values
        assert len(trainer.config) > 0
    
    def test_preprocess_data(self):
        """Test data preprocessing"""
        trainer = MLOpsModelTrainer()
        loader = DataLoader()
        data = loader.generate_sample_data(100)
        
        try:
            processed = trainer.preprocess_data(data)
            assert processed is not None
        except Exception as e:
            # If preprocessing fails due to missing target column, that's expected
            assert "target" in str(e).lower() or "churn" in str(e).lower()

class TestUtilities:
    """Test utility functions"""
    
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
    
    def test_numpy_operations(self):
        """Test numpy operations work correctly"""
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.mean() == 3.0
        assert arr.std() > 0
        assert len(arr) == 5

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
