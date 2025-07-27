import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
import os
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter('ml_predictions_total', 'Total number of predictions made')
PREDICTION_LATENCY = Histogram('ml_prediction_duration_seconds', 'Time spent on predictions')
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Current model accuracy')
DATA_DRIFT_SCORE = Gauge('ml_data_drift_score', 'Data drift detection score')

class ModelMonitor:
    def __init__(self, config_path="configs/config.yaml"):
        """Initialize the model monitor"""
        self.config = self.load_config(config_path)
        self.reference_data = None
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        return {}
    
    def load_reference_data(self, reference_path="data/reference/reference.csv"):
        """Load reference data for drift detection"""
        try:
            if os.path.exists(reference_path):
                self.reference_data = pd.read_csv(reference_path)
                logger.info(f"Reference data loaded from {reference_path}")
            else:
                logger.warning(f"Reference data not found at {reference_path}")
                # Generate sample reference data
                self.reference_data = self.generate_sample_reference_data()
        except Exception as e:
            logger.error(f"Error loading reference data: {str(e)}")
            self.reference_data = self.generate_sample_reference_data()
    
    def generate_sample_reference_data(self, n_samples=1000):
        """Generate sample reference data"""
        np.random.seed(42)
        data = pd.DataFrame({
            f'feature_{i}': np.random.randn(n_samples) 
            for i in range(10)
        })
        logger.info("Generated sample reference data")
        return data
    
    def detect_data_drift(self, current_data):
        """Detect data drift using statistical tests"""
        if self.reference_data is None:
            self.load_reference_data()
        
        drift_scores = {}
        
        for column in current_data.columns:
            if column in self.reference_data.columns:
                # Kolmogorov-Smirnov test
                from scipy import stats
                
                ref_values = self.reference_data[column].dropna()
                curr_values = current_data[column].dropna()
                
                if len(ref_values) > 0 and len(curr_values) > 0:
                    ks_stat, p_value = stats.ks_2samp(ref_values, curr_values)
                    drift_scores[column] = {
                        'ks_statistic': ks_stat,
                        'p_value': p_value,
                        'drift_detected': p_value < 0.05
                    }
        
        # Calculate overall drift score
        overall_drift_score = np.mean([score['ks_statistic'] for score in drift_scores.values()])
        DATA_DRIFT_SCORE.set(overall_drift_score)
        
        logger.info(f"Data drift analysis completed. Overall drift score: {overall_drift_score:.4f}")
        return drift_scores, overall_drift_score
    
    def monitor_model_performance(self, predictions_log_path="logs/predictions.jsonl"):
        """Monitor model performance from prediction logs"""
        try:
            if not os.path.exists(predictions_log_path):
                logger.warning(f"Predictions log file not found at {predictions_log_path}")
                return {}
            
            # Read recent predictions
            recent_predictions = []
            with open(predictions_log_path, 'r') as f:
                for line in f:
                    try:
                        pred_data = json.loads(line.strip())
                        recent_predictions.append(pred_data)
                    except json.JSONDecodeError:
                        continue
            
            if not recent_predictions:
                logger.info("No recent predictions found")
                return {}
            
            # Calculate metrics
            total_predictions = len(recent_predictions)
            PREDICTION_COUNTER._value._value = total_predictions
            
            # Calculate average prediction confidence (if available)
            confidences = [p.get('probability', 0) for p in recent_predictions if p.get('probability')]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            metrics = {
                'total_predictions': total_predictions,
                'average_confidence': avg_confidence,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Performance monitoring completed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error monitoring model performance: {str(e)}")
            return {}
    
    def check_model_health(self, api_url="http://localhost:8000"):
        """Check if the model API is healthy"""
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"Model API health check passed: {health_data}")
                return True, health_data
            else:
                logger.warning(f"Model API health check failed with status {response.status_code}")
                return False, {}
        except Exception as e:
            logger.error(f"Error checking model health: {str(e)}")
            return False, {}
    
    def generate_monitoring_report(self):
        """Generate a comprehensive monitoring report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_results': {}
        }
        
        # Check model health
        is_healthy, health_data = self.check_model_health()
        report['monitoring_results']['model_health'] = {
            'is_healthy': is_healthy,
            'details': health_data
        }
        
        # Monitor performance
        performance_metrics = self.monitor_model_performance()
        report['monitoring_results']['performance'] = performance_metrics
        
        # Generate sample current data for drift detection
        current_data = pd.DataFrame({
            f'feature_{i}': np.random.randn(100) + 0.1  # Slight shift to simulate drift
            for i in range(10)
        })
        
        # Detect data drift
        drift_scores, overall_drift = self.detect_data_drift(current_data)
        report['monitoring_results']['data_drift'] = {
            'overall_drift_score': overall_drift,
            'drift_threshold': 0.1,
            'drift_detected': overall_drift > 0.1,
            'feature_drift_scores': drift_scores
        }
        
        # Save report
        report_path = f"logs/monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Monitoring report saved to {report_path}")
        return report
    
    def start_prometheus_server(self, port=8001):
        """Start Prometheus metrics server"""
        try:
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {str(e)}")
    
    def run_continuous_monitoring(self, interval_minutes=60):
        """Run continuous monitoring"""
        logger.info(f"Starting continuous monitoring with {interval_minutes} minute intervals")
        
        # Start Prometheus server
        self.start_prometheus_server()
        
        while True:
            try:
                logger.info("Running monitoring cycle...")
                report = self.generate_monitoring_report()
                
                # Check for alerts
                self.check_alerts(report)
                
                logger.info(f"Monitoring cycle completed. Next check in {interval_minutes} minutes.")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring cycle: {str(e)}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def check_alerts(self, report):
        """Check for alert conditions and send notifications"""
        monitoring_config = self.config.get('monitoring', {})
        
        # Check model health
        if not report['monitoring_results']['model_health']['is_healthy']:
            self.send_alert("Model health check failed", report)
        
        # Check data drift
        drift_result = report['monitoring_results']['data_drift']
        if drift_result['drift_detected']:
            self.send_alert(f"Data drift detected: {drift_result['overall_drift_score']:.4f}", report)
        
        # Check performance thresholds
        performance = report['monitoring_results']['performance']
        if performance.get('average_confidence', 1.0) < 0.7:
            self.send_alert(f"Low model confidence: {performance['average_confidence']:.4f}", report)
    
    def send_alert(self, message, report_data):
        """Send alert notification"""
        alert_message = f"MLOps Alert: {message}"
        logger.warning(alert_message)
        
        # Here you would implement actual alert sending (email, Slack, etc.)
        # For now, just log the alert
        alert_log = {
            'timestamp': datetime.now().isoformat(),
            'message': alert_message,
            'report_data': report_data
        }
        
        alert_log_path = "logs/alerts.jsonl"
        os.makedirs(os.path.dirname(alert_log_path), exist_ok=True)
        
        with open(alert_log_path, 'a') as f:
            f.write(json.dumps(alert_log, default=str) + '\n')

if __name__ == "__main__":
    monitor = ModelMonitor()
    
    # Run one-time monitoring report
    report = monitor.generate_monitoring_report()
    print(f"Monitoring report generated: {json.dumps(report, indent=2, default=str)}")
    
    # Uncomment to run continuous monitoring
    # monitor.run_continuous_monitoring(interval_minutes=30)
