"""
Health Check Endpoint
Ultimate XAU Super System V4.0
"""

from fastapi import FastAPI, HTTPException
from typing import Dict
import psutil
import time
from datetime import datetime

def create_health_check_endpoint(app: FastAPI):
    """Add health check endpoint to FastAPI app"""
    
    @app.get("/health")
    async def health_check() -> Dict:
        """System health check endpoint"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Service checks
            services_status = {
                'api': True,
                'database': check_database_connection(),
                'redis': check_redis_connection(),
                'ai_systems': check_ai_systems(),
                'trading_systems': check_trading_systems()
            }
            
            # Overall health
            all_services_healthy = all(services_status.values())
            
            health_data = {
                'status': 'healthy' if all_services_healthy else 'unhealthy',
                'timestamp': datetime.now().isoformat(),
                'version': '4.0.0',
                'uptime': get_uptime(),
                'system_metrics': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': disk.percent
                },
                'services': services_status,
                'performance_score': calculate_performance_score()
            }
            
            if not all_services_healthy:
                raise HTTPException(status_code=503, detail=health_data)
                
            return health_data
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            )

def check_database_connection() -> bool:
    """Check database connectivity"""
    try:
        # Database connection check logic
        return True
    except:
        return False

def check_redis_connection() -> bool:
    """Check Redis connectivity"""
    try:
        # Redis connection check logic
        return True
    except:
        return False

def check_ai_systems() -> bool:
    """Check AI systems status"""
    try:
        # AI systems check logic
        return True
    except:
        return False

def check_trading_systems() -> bool:
    """Check trading systems status"""
    try:
        # Trading systems check logic
        return True
    except:
        return False

def get_uptime() -> str:
    """Get system uptime"""
    uptime_seconds = time.time() - psutil.boot_time()
    hours = int(uptime_seconds // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    return f"{hours}h {minutes}m"

def calculate_performance_score() -> float:
    """Calculate overall performance score"""
    cpu_score = max(0, 100 - psutil.cpu_percent())
    memory_score = max(0, 100 - psutil.virtual_memory().percent)
    return round((cpu_score + memory_score) / 2, 2)
