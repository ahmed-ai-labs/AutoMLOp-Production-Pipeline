from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from datetime import datetime
import os
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLOps Production Pipeline API",
    description="Production-ready ML model serving API",
    version="1.0.0"
)

# Load configuration
def load_config():
    config_path = "configs/config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    return {}

config = load_config()

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    features: list
    model_version: str = "latest"

class PredictionResponse(BaseModel):
    prediction: float
    probability: float = None
    model_version: str
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    model_loaded: bool

# Global model variable
model = None

def load_model(model_path: str = "models/model.pkl"):
    """Load the trained model"""
    global model
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            return True
        else:
            logger.warning(f"Model file not found at {model_path}")
            return False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        model_loaded=model is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction using the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to numpy array
        features = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get probability if available (for classification models)
        probability = None
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(features)[0]
            probability = float(max(prob))
        
        logger.info(f"Prediction made: {prediction}")
        
        return PredictionResponse(
            prediction=float(prediction),
            probability=probability,
            model_version=request.model_version,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "model_loaded": True,
        "timestamp": datetime.now()
    }

@app.post("/model/reload")
async def reload_model():
    """Reload the model"""
    success = load_model()
    if success:
        return {"message": "Model reloaded successfully", "timestamp": datetime.now()}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
