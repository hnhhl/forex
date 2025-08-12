"""
Performance Optimizer System
Ultimate XAU Super System V4.0 - Day 19 Implementation

Advanced performance optimization and fine-tuning system:
- AI System Response Time Optimization
- Memory Usage Optimization  
- Decision Making Speed Enhancement
- Resource Management Optimization
- Parallel Processing Optimization
"""

import numpy as np
import pandas as pd
import time
import threading
import multiprocessing
import asyncio
import concurrent.futures
import gc
import psutil
import cProfile
import pstats
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
import sys
import os

# Import optimization libraries
try:
    import numba
    from numba import jit, cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import cython
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Performance optimization levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXTREME = "extreme"


class OptimizationTarget(Enum):
    """Optimization targets"""
    RESPONSE_TIME = "response_time"
    MEMORY_USAGE = "memory_usage"
    CPU_UTILIZATION = "cpu_utilization"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    ALL = "all"


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization"""
    
    # Optimization settings
    level: OptimizationLevel = OptimizationLevel.ADVANCED
    target: OptimizationTarget = OptimizationTarget.ALL
    max_workers: int = 4
    enable_parallel_processing: bool = True
    enable_memory_optimization: bool = True
    enable_cpu_optimization: bool = True
    
    # Performance targets
    target_response_time_ms: float = 30.0  # Target <30ms response
    target_memory_usage_mb: float = 512.0  # Target <512MB memory
    target_cpu_usage_percent: float = 70.0  # Target <70% CPU
    target_throughput_ops: float = 1000.0  # Target 1000 ops/sec
    
    # Caching settings
    enable_result_caching: bool = True
    cache_size: int = 10000
    cache_ttl_seconds: int = 300
    
    # Threading settings
    max_thread_pool_size: int = 8
    async_processing: bool = True
    
    # Monitoring settings
    enable_profiling: bool = True
    profile_output_dir: str = "performance_profiles"
    monitoring_interval_seconds: float = 1.0


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    timestamp: datetime
    
    # Response time metrics
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    
    # System resource metrics
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_usage_percent: float
    
    # Throughput metrics
    operations_per_second: float
    total_operations: int
    successful_operations: int
    failed_operations: int
    
    # Quality metrics
    accuracy_score: float
    success_rate: float
    
    # System health
    system_health_score: float
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class PerformanceProfiler:
    """Advanced performance profiling system"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.profiles = {}
        self.active_profiles = {}
        
        # Create profile output directory
        os.makedirs(config.profile_output_dir, exist_ok=True)
    
    def start_profiling(self, profile_name: str):
        """Start profiling a specific operation"""
        if not self.config.enable_profiling:
            return
        
        profiler = cProfile.Profile()
        profiler.enable()
        self.active_profiles[profile_name] = {
            'profiler': profiler,
            'start_time': time.time()
        }
    
    def stop_profiling(self, profile_name: str) -> Dict[str, Any]:
        """Stop profiling and return results"""
        if not self.config.enable_profiling or profile_name not in self.active_profiles:
            return {}
        
        profile_data = self.active_profiles.pop(profile_name)
        profiler = profile_data['profiler']
        profiler.disable()
        
        # Generate stats
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        # Save to file
        output_file = os.path.join(
            self.config.profile_output_dir,
            f"{profile_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prof"
        )
        stats.dump_stats(output_file)
        
        # Extract key metrics
        total_time = time.time() - profile_data['start_time']
        stats_dict = {}
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            stats_dict[func] = {
                'calls': nc,
                'total_time': tt,
                'cumulative_time': ct
            }
        
        result = {
            'total_time': total_time,
            'output_file': output_file,
            'stats': stats_dict,
            'top_functions': self._get_top_functions(stats)
        }
        
        self.profiles[profile_name] = result
        return result
    
    def _get_top_functions(self, stats: pstats.Stats, top_n: int = 10) -> List[Dict[str, Any]]:
        """Extract top functions by cumulative time"""
        top_functions = []
        stats.sort_stats('cumulative')
        
        # Get top functions (simplified)
        for i, (func, (cc, nc, tt, ct, callers)) in enumerate(stats.stats.items()):
            if i >= top_n:
                break
            
            top_functions.append({
                'function': str(func),
                'calls': nc,
                'total_time': tt,
                'cumulative_time': ct,
                'per_call': ct / nc if nc > 0 else 0
            })
        
        return top_functions


class MemoryOptimizer:
    """Memory usage optimization"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.memory_caches = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def optimize_memory_usage(self):
        """Perform memory optimization"""
        optimizations = []
        
        # Force garbage collection
        collected = gc.collect()
        optimizations.append(f"Garbage collection freed {collected} objects")
        
        # Clear unused caches
        if hasattr(self, 'memory_caches'):
            cache_size_before = len(self.memory_caches)
            self._clean_expired_cache()
            cache_size_after = len(self.memory_caches)
            optimizations.append(f"Cache cleaned: {cache_size_before} -> {cache_size_after} entries")
        
        # Get memory info
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        if memory_mb > self.config.target_memory_usage_mb:
            optimizations.append(f"Memory usage high: {memory_mb:.1f}MB > {self.config.target_memory_usage_mb}MB")
        
        return {
            'optimizations_applied': optimizations,
            'memory_usage_mb': memory_mb,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        }
    
    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result with TTL check"""
        if not self.config.enable_result_caching:
            return None
        
        if cache_key in self.memory_caches:
            entry = self.memory_caches[cache_key]
            if time.time() - entry['timestamp'] < self.config.cache_ttl_seconds:
                self.cache_hits += 1
                return entry['data']
            else:
                # Expired
                del self.memory_caches[cache_key]
        
        self.cache_misses += 1
        return None
    
    def cache_result(self, cache_key: str, data: Any):
        """Cache result with size limit"""
        if not self.config.enable_result_caching:
            return
        
        # Check cache size limit
        if len(self.memory_caches) >= self.config.cache_size:
            self._evict_oldest_cache_entry()
        
        self.memory_caches[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
    
    def _clean_expired_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.memory_caches.items()
            if current_time - entry['timestamp'] >= self.config.cache_ttl_seconds
        ]
        
        for key in expired_keys:
            del self.memory_caches[key]
    
    def _evict_oldest_cache_entry(self):
        """Remove oldest cache entry"""
        if self.memory_caches:
            oldest_key = min(
                self.memory_caches.keys(),
                key=lambda k: self.memory_caches[k]['timestamp']
            )
            del self.memory_caches[oldest_key]


class CPUOptimizer:
    """CPU utilization optimization"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.thread_pool = None
        self.process_pool = None
        
        if config.enable_parallel_processing:
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=config.max_thread_pool_size
            )
            self.process_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=config.max_workers
            )
    
    def optimize_cpu_intensive_task(self, func: Callable, *args, **kwargs) -> Any:
        """Optimize CPU-intensive task execution"""
        
        # Try numba optimization if available
        if NUMBA_AVAILABLE and self.config.level in [OptimizationLevel.ADVANCED, OptimizationLevel.EXTREME]:
            try:
                # Attempt to JIT compile the function
                jit_func = numba.jit(func, nopython=True)
                return jit_func(*args, **kwargs)
            except Exception:
                pass  # Fall back to normal execution
        
        # Use process pool for CPU-intensive tasks
        if self.process_pool and self.config.enable_parallel_processing:
            try:
                future = self.process_pool.submit(func, *args, **kwargs)
                return future.result()
            except Exception:
                pass  # Fall back to normal execution
        
        # Normal execution
        return func(*args, **kwargs)
    
    def optimize_io_task(self, func: Callable, *args, **kwargs) -> Any:
        """Optimize I/O bound task execution"""
        
        # Use thread pool for I/O tasks
        if self.thread_pool and self.config.enable_parallel_processing:
            try:
                future = self.thread_pool.submit(func, *args, **kwargs)
                return future.result()
            except Exception:
                pass  # Fall back to normal execution
        
        # Normal execution
        return func(*args, **kwargs)
    
    async def optimize_async_task(self, coro) -> Any:
        """Optimize asynchronous task execution"""
        if self.config.async_processing:
            return await coro
        else:
            # Convert async to sync if needed
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        return psutil.cpu_percent(interval=0.1)
    
    def cleanup(self):
        """Cleanup thread and process pools"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)


class ResponseTimeOptimizer:
    """Response time optimization"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.response_times = []
        self.memory_optimizer = MemoryOptimizer(config)
        self.cpu_optimizer = CPUOptimizer(config)
    
    def optimize_function_call(self, func: Callable, cache_key: str = None, 
                             is_cpu_intensive: bool = False, 
                             is_io_bound: bool = False, *args, **kwargs) -> Tuple[Any, float]:
        """Optimize function call with caching and parallel processing"""
        start_time = time.time()
        
        # Try cache first
        if cache_key:
            cached_result = self.memory_optimizer.get_cached_result(cache_key)
            if cached_result is not None:
                response_time = time.time() - start_time
                self.response_times.append(response_time * 1000)  # Convert to ms
                return cached_result, response_time
        
        # Execute function with appropriate optimization
        try:
            if is_cpu_intensive:
                result = self.cpu_optimizer.optimize_cpu_intensive_task(func, *args, **kwargs)
            elif is_io_bound:
                result = self.cpu_optimizer.optimize_io_task(func, *args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache result if cache key provided
            if cache_key:
                self.memory_optimizer.cache_result(cache_key, result)
            
            response_time = time.time() - start_time
            self.response_times.append(response_time * 1000)  # Convert to ms
            
            return result, response_time
        
        except Exception as e:
            response_time = time.time() - start_time
            self.response_times.append(response_time * 1000)
            raise e
    
    def get_response_time_stats(self) -> Dict[str, float]:
        """Get response time statistics"""
        if not self.response_times:
            return {}
        
        response_times = np.array(self.response_times[-1000:])  # Last 1000 calls
        
        return {
            'avg_ms': np.mean(response_times),
            'min_ms': np.min(response_times),
            'max_ms': np.max(response_times),
            'p95_ms': np.percentile(response_times, 95),
            'p99_ms': np.percentile(response_times, 99),
            'std_ms': np.std(response_times),
            'count': len(response_times)
        }


class PerformanceOptimizer:
    """Main performance optimization system"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        
        # Initialize optimizers
        self.profiler = PerformanceProfiler(self.config)
        self.memory_optimizer = MemoryOptimizer(self.config)
        self.cpu_optimizer = CPUOptimizer(self.config)
        self.response_time_optimizer = ResponseTimeOptimizer(self.config)
        
        # Monitoring
        self.metrics_history = []
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info(f"Performance Optimizer initialized with level: {self.config.level.value}")
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = self.collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 metrics
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Auto-optimize if thresholds exceeded
                self._auto_optimize_if_needed(metrics)
                
                time.sleep(self.config.monitoring_interval_seconds)
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        
        # System metrics
        process = psutil.Process()
        cpu_usage = psutil.cpu_percent(interval=0)
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = process.memory_percent()
        
        # Response time metrics
        response_stats = self.response_time_optimizer.get_response_time_stats()
        
        # Calculate throughput (operations per second)
        current_time = time.time()
        recent_metrics = [
            m for m in self.metrics_history 
            if (current_time - m.timestamp.timestamp()) < 60  # Last minute
        ]
        
        if recent_metrics:
            total_ops = sum(m.total_operations for m in recent_metrics)
            time_window = 60  # 1 minute
            ops_per_second = total_ops / time_window
        else:
            ops_per_second = 0
        
        # Calculate system health score
        health_score = self._calculate_system_health_score(
            response_stats.get('avg_ms', 100),
            cpu_usage,
            memory_mb
        )
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(response_stats, cpu_usage, memory_mb)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(bottlenecks, response_stats, cpu_usage, memory_mb)
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            avg_response_time_ms=response_stats.get('avg_ms', 0),
            min_response_time_ms=response_stats.get('min_ms', 0),
            max_response_time_ms=response_stats.get('max_ms', 0),
            p95_response_time_ms=response_stats.get('p95_ms', 0),
            p99_response_time_ms=response_stats.get('p99_ms', 0),
            cpu_usage_percent=cpu_usage,
            memory_usage_mb=memory_mb,
            memory_usage_percent=memory_percent,
            operations_per_second=ops_per_second,
            total_operations=len(self.response_time_optimizer.response_times),
            successful_operations=len(self.response_time_optimizer.response_times),  # Simplified
            failed_operations=0,  # Simplified
            accuracy_score=0.95,  # Placeholder
            success_rate=1.0,  # Simplified
            system_health_score=health_score,
            bottlenecks=bottlenecks,
            recommendations=recommendations
        )
    
    def _calculate_system_health_score(self, avg_response_ms: float, 
                                     cpu_usage: float, memory_mb: float) -> float:
        """Calculate overall system health score (0-100)"""
        
        # Response time score (0-40 points)
        if avg_response_ms <= self.config.target_response_time_ms:
            response_score = 40
        else:
            response_score = max(0, 40 - (avg_response_ms - self.config.target_response_time_ms))
        
        # CPU score (0-30 points)
        if cpu_usage <= self.config.target_cpu_usage_percent:
            cpu_score = 30
        else:
            cpu_score = max(0, 30 - (cpu_usage - self.config.target_cpu_usage_percent))
        
        # Memory score (0-30 points)
        if memory_mb <= self.config.target_memory_usage_mb:
            memory_score = 30
        else:
            memory_score = max(0, 30 - (memory_mb - self.config.target_memory_usage_mb) / 10)
        
        return min(100, response_score + cpu_score + memory_score)
    
    def _identify_bottlenecks(self, response_stats: Dict, cpu_usage: float, 
                            memory_mb: float) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # Response time bottlenecks
        avg_response_ms = response_stats.get('avg_ms', 0)
        if avg_response_ms > self.config.target_response_time_ms * 1.5:
            bottlenecks.append(f"High response time: {avg_response_ms:.1f}ms")
        
        p99_response_ms = response_stats.get('p99_ms', 0)
        if p99_response_ms > self.config.target_response_time_ms * 3:
            bottlenecks.append(f"High P99 response time: {p99_response_ms:.1f}ms")
        
        # CPU bottlenecks
        if cpu_usage > self.config.target_cpu_usage_percent * 1.2:
            bottlenecks.append(f"High CPU usage: {cpu_usage:.1f}%")
        
        # Memory bottlenecks
        if memory_mb > self.config.target_memory_usage_mb * 1.2:
            bottlenecks.append(f"High memory usage: {memory_mb:.1f}MB")
        
        return bottlenecks
    
    def _generate_recommendations(self, bottlenecks: List[str], response_stats: Dict,
                                cpu_usage: float, memory_mb: float) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Response time recommendations
        avg_response_ms = response_stats.get('avg_ms', 0)
        if avg_response_ms > self.config.target_response_time_ms:
            recommendations.append("Enable result caching to reduce response time")
            recommendations.append("Consider parallel processing for CPU-intensive operations")
        
        # CPU recommendations
        if cpu_usage > self.config.target_cpu_usage_percent:
            recommendations.append("Enable parallel processing to distribute CPU load")
            recommendations.append("Consider using Numba JIT compilation for hot functions")
        
        # Memory recommendations
        if memory_mb > self.config.target_memory_usage_mb:
            recommendations.append("Enable memory optimization and garbage collection")
            recommendations.append("Reduce cache size or TTL to free memory")
        
        # Cache recommendations
        cache_hit_rate = self.memory_optimizer.cache_hits / (
            self.memory_optimizer.cache_hits + self.memory_optimizer.cache_misses
        ) if (self.memory_optimizer.cache_hits + self.memory_optimizer.cache_misses) > 0 else 0
        
        if cache_hit_rate < 0.7:
            recommendations.append("Increase cache size or TTL to improve hit rate")
        
        return recommendations
    
    def _auto_optimize_if_needed(self, metrics: PerformanceMetrics):
        """Automatically apply optimizations if thresholds exceeded"""
        
        # Auto memory optimization
        if (metrics.memory_usage_mb > self.config.target_memory_usage_mb * 1.5 or
            metrics.system_health_score < 60):
            
            try:
                self.memory_optimizer.optimize_memory_usage()
                logger.info("Auto memory optimization applied")
            except Exception as e:
                logger.error(f"Auto memory optimization failed: {e}")
    
    def optimize_function(self, func: Callable, cache_key: str = None,
                         is_cpu_intensive: bool = False, is_io_bound: bool = False):
        """Decorator for function optimization"""
        def wrapper(*args, **kwargs):
            return self.response_time_optimizer.optimize_function_call(
                func, cache_key, is_cpu_intensive, is_io_bound, *args, **kwargs
            )[0]  # Return only result, not timing
        return wrapper
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        
        if not self.metrics_history:
            current_metrics = self.collect_metrics()
            self.metrics_history.append(current_metrics)
        
        latest_metrics = self.metrics_history[-1]
        
        # Historical analysis
        if len(self.metrics_history) > 1:
            response_times = [m.avg_response_time_ms for m in self.metrics_history[-100:]]
            cpu_usages = [m.cpu_usage_percent for m in self.metrics_history[-100:]]
            memory_usages = [m.memory_usage_mb for m in self.metrics_history[-100:]]
        else:
            response_times = [latest_metrics.avg_response_time_ms]
            cpu_usages = [latest_metrics.cpu_usage_percent]
            memory_usages = [latest_metrics.memory_usage_mb]
        
        return {
            'timestamp': datetime.now(),
            'optimization_level': self.config.level.value,
            'current_metrics': {
                'response_time_ms': latest_metrics.avg_response_time_ms,
                'cpu_usage_percent': latest_metrics.cpu_usage_percent,
                'memory_usage_mb': latest_metrics.memory_usage_mb,
                'system_health_score': latest_metrics.system_health_score,
                'operations_per_second': latest_metrics.operations_per_second
            },
            'targets': {
                'response_time_ms': self.config.target_response_time_ms,
                'cpu_usage_percent': self.config.target_cpu_usage_percent,
                'memory_usage_mb': self.config.target_memory_usage_mb,
                'throughput_ops': self.config.target_throughput_ops
            },
            'target_achievement': {
                'response_time': latest_metrics.avg_response_time_ms <= self.config.target_response_time_ms,
                'cpu_usage': latest_metrics.cpu_usage_percent <= self.config.target_cpu_usage_percent,
                'memory_usage': latest_metrics.memory_usage_mb <= self.config.target_memory_usage_mb,
                'overall_health': latest_metrics.system_health_score >= 80
            },
            'trends': {
                'response_time_trend': 'improving' if len(response_times) > 1 and response_times[-1] < response_times[0] else 'stable',
                'cpu_trend': 'improving' if len(cpu_usages) > 1 and cpu_usages[-1] < cpu_usages[0] else 'stable',
                'memory_trend': 'improving' if len(memory_usages) > 1 and memory_usages[-1] < memory_usages[0] else 'stable'
            },
            'bottlenecks': latest_metrics.bottlenecks,
            'recommendations': latest_metrics.recommendations,
            'cache_performance': {
                'hit_rate': self.memory_optimizer.cache_hits / (self.memory_optimizer.cache_hits + self.memory_optimizer.cache_misses) if (self.memory_optimizer.cache_hits + self.memory_optimizer.cache_misses) > 0 else 0,
                'total_hits': self.memory_optimizer.cache_hits,
                'total_misses': self.memory_optimizer.cache_misses,
                'cache_size': len(self.memory_optimizer.memory_caches)
            },
            'optimization_features': {
                'numba_available': NUMBA_AVAILABLE,
                'cython_available': CYTHON_AVAILABLE,
                'parallel_processing': self.config.enable_parallel_processing,
                'result_caching': self.config.enable_result_caching,
                'async_processing': self.config.async_processing
            }
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_monitoring()
        self.cpu_optimizer.cleanup()
        logger.info("Performance Optimizer cleaned up")


def create_performance_optimizer(level: str = "advanced", 
                               target_response_time_ms: float = 30.0) -> PerformanceOptimizer:
    """Factory function to create performance optimizer"""
    
    config = OptimizationConfig(
        level=OptimizationLevel(level),
        target_response_time_ms=target_response_time_ms,
        enable_parallel_processing=True,
        enable_memory_optimization=True,
        enable_cpu_optimization=True,
        enable_result_caching=True
    )
    
    return PerformanceOptimizer(config)


def benchmark_function(func: Callable, *args, iterations: int = 1000, **kwargs) -> Dict[str, float]:
    """Benchmark function performance"""
    
    times = []
    for _ in range(iterations):
        start_time = time.time()
        func(*args, **kwargs)
        times.append((time.time() - start_time) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    return {
        'iterations': iterations,
        'avg_ms': np.mean(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99),
        'std_ms': np.std(times),
        'ops_per_second': 1000 / np.mean(times) if np.mean(times) > 0 else 0
    }


if __name__ == "__main__":
    # Demo performance optimization
    print("🚀 Performance Optimizer Demo")
    
    optimizer = create_performance_optimizer()
    optimizer.start_monitoring()
    
    # Simulate some work
    time.sleep(2)
    
    report = optimizer.get_optimization_report()
    print(f"System Health Score: {report['current_metrics']['system_health_score']:.1f}")
    print(f"Response Time: {report['current_metrics']['response_time_ms']:.1f}ms")
    
    optimizer.cleanup()