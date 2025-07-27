# MLOps Production Pipeline

A comprehensive MLOps (Machine Learning Operations) production pipeline that provides end-to-end machine learning lifecycle management including data processing, model training, deployment, monitoring, and CI/CD.

## 🏗️ Architecture

```
├── src/                    # Source code
│   ├── api/               # FastAPI application
│   ├── data/              # Data processing modules
│   ├── models/            # Model training and inference
│   ├── monitoring/        # Model and data monitoring
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── data/                  # Data storage
│   ├── raw/              # Raw data
│   ├── processed/        # Processed data
│   └── reference/        # Reference data for drift detection
├── models/               # Trained model artifacts
├── tests/                # Test files
├── docker/               # Docker configurations
├── k8s/                  # Kubernetes manifests
├── monitoring/           # Monitoring configurations
├── logs/                 # Application logs
├── notebooks/            # Jupyter notebooks
└── docs/                 # Documentation
```

## 🚀 Features

### Core ML Pipeline
- **Data Processing**: Automated data validation and preprocessing
- **Model Training**: Configurable training pipeline with hyperparameter tuning
- **Model Evaluation**: Comprehensive metrics and validation
- **Model Registry**: MLflow integration for model versioning

### Production Deployment
- **REST API**: FastAPI-based model serving with automatic documentation
- **Containerization**: Docker support for consistent deployments
- **Orchestration**: Kubernetes manifests for scalable deployment
- **Load Balancing**: Built-in load balancing and auto-scaling

### Monitoring & Observability
- **Model Performance**: Real-time model accuracy and performance monitoring
- **Data Drift Detection**: Statistical tests for data distribution changes
- **System Health**: API health checks and uptime monitoring
- **Metrics**: Prometheus integration for custom metrics
- **Alerting**: Configurable alerts for model degradation

### DevOps & CI/CD
- **Infrastructure as Code**: Kubernetes and Docker configurations
- **Testing**: Comprehensive test suite with pytest
- **Configuration Management**: YAML-based configuration system
- **Logging**: Structured logging with multiple output formats

## 🛠️ Quick Start

### Prerequisites
- Python 3.9+
- Docker (optional, for containerization)
- Kubernetes (optional, for orchestration)

### Installation

1. **Clone and setup the project:**
```powershell
# Navigate to project directory
cd "c:\Users\ahmed\Desktop\utoMLOp-Production-Pipeline"

# Install dependencies
pip install -r requirements.txt
```

2. **Train a model:**
```powershell
python src\models\train.py
```

3. **Start the API server:**
```powershell
python src\api\main.py
```

4. **Test the API:**
```powershell
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"features\": [1,2,3,4,5,6,7,8,9,10]}"
```

### Docker Deployment

1. **Build the container:**
```powershell
cd docker
docker build -t mlops-pipeline -f Dockerfile ..
```

2. **Run with Docker Compose:**
```powershell
docker-compose up -d
```

This will start:
- MLOps API (port 8000)
- MLflow (port 5000)
- PostgreSQL database
- Prometheus (port 9090)
- Grafana (port 3000)

### Kubernetes Deployment

1. **Create namespace and deploy:**
```powershell
kubectl apply -f k8s\namespace.yaml
kubectl apply -f k8s\deployment.yaml
```

## 📊 Monitoring

### Access Monitoring Dashboards

- **API Documentation**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### Run Monitoring Scripts

```powershell
# Generate monitoring report
python src\monitoring\monitor.py

# Start continuous monitoring
python -c "from src.monitoring.monitor import ModelMonitor; ModelMonitor().run_continuous_monitoring()"
```

## 🧪 Testing

Run the test suite:

```powershell
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests\ -v

# Run with coverage
pytest tests\ --cov=src --cov-report=html
```

## 📝 Configuration

The pipeline is configured via `configs/config.yaml`. Key sections include:

- **Project Settings**: Basic project information
- **Data Configuration**: Data paths and validation settings
- **Model Parameters**: Training hyperparameters and model settings
- **MLflow Integration**: Experiment tracking configuration
- **Monitoring**: Drift detection and alerting thresholds
- **Deployment**: API and infrastructure settings

Example configuration:

```yaml
project:
  name: "MLOps Production Pipeline"
  version: "1.0.0"

model:
  type: "classification"
  hyperparameters:
    n_estimators: 100
    max_depth: 10

monitoring:
  data_drift:
    drift_detection_method: "ks_test"
    threshold: 0.1
```

## 🔧 API Endpoints

### Core Endpoints

- `GET /health` - Health check
- `POST /predict` - Make predictions
- `GET /model/info` - Model information
- `POST /model/reload` - Reload model

### Example Usage

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Make prediction
data = {
    "features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "model_version": "latest"
}
response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## 📈 Model Training

The training pipeline supports:

- **Automated Data Loading**: CSV files or generated sample data
- **Data Preprocessing**: Feature scaling and validation
- **Model Training**: Configurable algorithms and hyperparameters
- **Evaluation**: Multiple metrics (accuracy, precision, recall, F1)
- **MLflow Integration**: Automatic experiment logging
- **Model Persistence**: Local and MLflow model registry

### Custom Training

```python
from src.models.train import ModelTrainer

trainer = ModelTrainer("configs/config.yaml")
metrics = trainer.run_training_pipeline()
print(f"Training completed: {metrics}")
```

## 🔍 Monitoring Features

### Data Drift Detection
- Kolmogorov-Smirnov statistical tests
- Configurable reference datasets
- Automated alert generation

### Performance Monitoring
- Real-time prediction tracking
- Model confidence scoring
- API response time monitoring

### System Health
- Service availability checks
- Resource utilization monitoring
- Error rate tracking

## 🚨 Alerting

Configure alerts in `configs/config.yaml`:

```yaml
monitoring:
  alerts:
    email: "admin@company.com"
    slack_webhook: "https://hooks.slack.com/your-webhook"
  model_performance:
    threshold: 0.85
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports in `configs/config.yaml`
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Docker issues**: Ensure Docker is running and accessible
4. **Model not loading**: Check model path in configuration

### PowerShell Specific Commands

Since you're using Windows PowerShell, here are the correct commands:

```powershell
# Create directories
New-Item -ItemType Directory -Force -Path directory_name

# Set environment variables
$env:VARIABLE_NAME = "value"

# Run Python scripts
python script.py

# Docker commands
docker build -t image_name .
docker run -p 8000:8000 image_name
```

For any issues, check the logs in the `logs/` directory or enable debug logging in the configuration.
