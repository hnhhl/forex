"""
Application Metrics Implementation
Ultimate XAU Super System V4.0
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import functools
from typing import Callable, Any

# Define metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections'
)

TRADING_OPERATIONS = Counter(
    'trading_operations_total',
    'Total trading operations',
    ['operation_type', 'symbol', 'status']
)

AI_PREDICTIONS = Counter(
    'ai_predictions_total',
    'Total AI predictions made',
    ['model_type', 'prediction_type']
)

AI_MODEL_ACCURACY = Gauge(
    'ai_model_accuracy',
    'Current AI model accuracy',
    ['model_name']
)

PORTFOLIO_VALUE = Gauge(
    'portfolio_value_usd',
    'Current portfolio value in USD'
)

RISK_METRICS = Gauge(
    'risk_metrics',
    'Risk management metrics',
    ['metric_type']
)

class MetricsCollector:
    """Centralized metrics collection"""
    
    def __init__(self):
        self.start_time = time.time()
        
    def start_metrics_server(self, port: int = 8000):
        """Start Prometheus metrics server"""
        start_http_server(port)
        
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics"""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
        
    def record_trading_operation(self, operation: str, symbol: str, status: str):
        """Record trading operation"""
        TRADING_OPERATIONS.labels(
            operation_type=operation,
            symbol=symbol,
            status=status
        ).inc()
        
    def record_ai_prediction(self, model_type: str, prediction_type: str):
        """Record AI prediction"""
        AI_PREDICTIONS.labels(
            model_type=model_type,
            prediction_type=prediction_type
        ).inc()
        
    def update_ai_accuracy(self, model_name: str, accuracy: float):
        """Update AI model accuracy"""
        AI_MODEL_ACCURACY.labels(model_name=model_name).set(accuracy)
        
    def update_portfolio_value(self, value: float):
        """Update portfolio value"""
        PORTFOLIO_VALUE.set(value)
        
    def update_risk_metric(self, metric_type: str, value: float):
        """Update risk metric"""
        RISK_METRICS.labels(metric_type=metric_type).set(value)

def monitor_endpoint(endpoint: str):
    """Decorator to monitor endpoint performance"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                status = 200
                return result
            except Exception as e:
                status = 500
                raise
            finally:
                duration = time.time() - start_time
                REQUEST_DURATION.labels(
                    method='GET',  # Default, can be extracted from request
                    endpoint=endpoint
                ).observe(duration)
                REQUEST_COUNT.labels(
                    method='GET',
                    endpoint=endpoint,
                    status=status
                ).inc()
        return wrapper
    return decorator

# Global metrics collector
metrics_collector = MetricsCollector()
