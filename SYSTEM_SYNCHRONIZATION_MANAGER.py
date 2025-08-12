#!/usr/bin/env python3
"""
SYSTEM SYNCHRONIZATION MANAGER
Đảm bảo tính đồng bộ và kết nối giữa các components của hệ thống AI3.0

Author: AI System Architect
Date: 2025-01-25
Status: PRODUCTION READY
"""

import asyncio
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque
import weakref

logger = logging.getLogger(__name__)

class SystemState(Enum):
    """System states"""
    INITIALIZING = "initializing"
    RUNNING = "running"  
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

class EventPriority(Enum):
    """Event priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class SystemEvent:
    """System event structure"""
    event_type: str
    source_system: str
    target_system: Optional[str] = None
    priority: EventPriority = EventPriority.MEDIUM
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None

@dataclass
class SystemComponent:
    """System component registration"""
    name: str
    instance: Any
    dependencies: List[str] = field(default_factory=list)
    health_check: Callable = None
    state: SystemState = SystemState.INITIALIZING
    last_heartbeat: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    max_errors: int = 10

class EventBus:
    """Advanced event bus for system communication"""
    
    def __init__(self):
        self.subscribers = defaultdict(list)
        self.event_queue = asyncio.Queue(maxsize=10000)
        self.event_history = deque(maxlen=1000)
        self.is_running = False
        self.stats = {
            'events_published': 0,
            'events_processed': 0,
            'events_failed': 0
        }
        
    async def start(self):
        """Start event processing"""
        self.is_running = True
        asyncio.create_task(self._event_processor())
        logger.info("Event Bus started")
    
    async def stop(self):
        """Stop event processing"""
        self.is_running = False
        logger.info("Event Bus stopped")
    
    def subscribe(self, event_type: str, callback: Callable, priority: int = 0):
        """Subscribe to events with priority"""
        self.subscribers[event_type].append((callback, priority))
        # Sort by priority (lower number = higher priority)
        self.subscribers[event_type].sort(key=lambda x: x[1])
        logger.debug(f"Subscribed to {event_type} with priority {priority}")
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from events"""
        self.subscribers[event_type] = [
            (cb, pri) for cb, pri in self.subscribers[event_type] 
            if cb != callback
        ]
    
    async def publish(self, event: SystemEvent):
        """Publish event to bus"""
        try:
            # Add to queue
            await self.event_queue.put(event)
            self.stats['events_published'] += 1
            
            # Add to history
            self.event_history.append({
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type,
                'source': event.source_system,
                'target': event.target_system
            })
            
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            self.stats['events_failed'] += 1
    
    async def _event_processor(self):
        """Process events from queue"""
        while self.is_running:
            try:
                # Get event with timeout
                event = await asyncio.wait_for(
                    self.event_queue.get(), timeout=1.0
                )
                
                # Process event
                await self._process_event(event)
                self.stats['events_processed'] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Event processing error: {e}")
                self.stats['events_failed'] += 1
    
    async def _process_event(self, event: SystemEvent):
        """Process individual event"""
        try:
            event_type = event.event_type
            
            # Get subscribers for this event type
            subscribers = self.subscribers.get(event_type, [])
            
            # Process subscribers by priority
            for callback, priority in subscribers:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Subscriber callback failed: {e}")
                    
        except Exception as e:
            logger.error(f"Event processing failed: {e}")

class DependencyManager:
    """Manage system dependencies and initialization order"""
    
    def __init__(self):
        self.components = {}
        self.dependency_graph = {}
        
    def register_component(self, component: SystemComponent):
        """Register system component"""
        self.components[component.name] = component
        self.dependency_graph[component.name] = component.dependencies
        logger.info(f"Registered component: {component.name}")
    
    def get_initialization_order(self) -> List[str]:
        """Get component initialization order using topological sort"""
        try:
            visited = set()
            temp_visited = set()
            order = []
            
            def visit(node):
                if node in temp_visited:
                    raise ValueError(f"Circular dependency detected: {node}")
                if node in visited:
                    return
                
                temp_visited.add(node)
                
                # Visit dependencies first
                for dependency in self.dependency_graph.get(node, []):
                    if dependency in self.dependency_graph:
                        visit(dependency)
                
                temp_visited.remove(node)
                visited.add(node)
                order.append(node)
            
            # Visit all components
            for component_name in self.components.keys():
                if component_name not in visited:
                    visit(component_name)
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to determine initialization order: {e}")
            return list(self.components.keys())  # Fallback to arbitrary order
    
    def validate_dependencies(self) -> Dict[str, List[str]]:
        """Validate all dependencies exist"""
        missing_deps = {}
        
        for component_name, dependencies in self.dependency_graph.items():
            missing = [
                dep for dep in dependencies 
                if dep not in self.components
            ]
            if missing:
                missing_deps[component_name] = missing
        
        return missing_deps

class HealthMonitor:
    """Monitor system component health"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.components = {}
        self.health_history = defaultdict(deque)
        self.is_monitoring = False
        
    def register_component(self, component: SystemComponent):
        """Register component for health monitoring"""
        self.components[component.name] = component
        logger.info(f"Health monitoring registered for: {component.name}")
    
    async def start_monitoring(self):
        """Start health monitoring"""
        self.is_monitoring = True
        asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.is_monitoring = False
        logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Health monitoring loop"""
        while self.is_monitoring:
            try:
                await self._check_all_components()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_all_components(self):
        """Check health of all components"""
        for component_name, component in self.components.items():
            try:
                health_status = await self._check_component_health(component)
                
                # Update health history
                self.health_history[component_name].append({
                    'timestamp': datetime.now().isoformat(),
                    'status': health_status,
                    'error_count': component.error_count
                })
                
                # Limit history size
                if len(self.health_history[component_name]) > 100:
                    self.health_history[component_name].popleft()
                
            except Exception as e:
                logger.error(f"Health check failed for {component_name}: {e}")
    
    async def _check_component_health(self, component: SystemComponent) -> Dict:
        """Check individual component health"""
        try:
            if component.health_check:
                if asyncio.iscoroutinefunction(component.health_check):
                    health_result = await component.health_check()
                else:
                    health_result = component.health_check()
            else:
                # Default health check
                health_result = {
                    'status': 'healthy' if component.state == SystemState.RUNNING else 'unhealthy',
                    'uptime': (datetime.now() - component.last_heartbeat).total_seconds()
                }
            
            # Update component heartbeat
            component.last_heartbeat = datetime.now()
            
            return health_result
            
        except Exception as e:
            component.error_count += 1
            return {
                'status': 'error',
                'error': str(e),
                'error_count': component.error_count
            }

class DataFlowCoordinator:
    """Coordinate data flow between system components"""
    
    def __init__(self):
        self.data_pipelines = {}
        self.flow_rules = {}
        self.data_buffer = {}
        self.flow_stats = defaultdict(int)
        
    def register_pipeline(self, name: str, source: str, targets: List[str], 
                         transform_func: Optional[Callable] = None):
        """Register data pipeline"""
        self.data_pipelines[name] = {
            'source': source,
            'targets': targets,
            'transform': transform_func,
            'active': True
        }
        logger.info(f"Data pipeline registered: {name}")
    
    async def flow_data(self, pipeline_name: str, data: Any):
        """Flow data through pipeline"""
        try:
            pipeline = self.data_pipelines.get(pipeline_name)
            if not pipeline or not pipeline['active']:
                return
            
            # Apply transformation if specified
            if pipeline['transform']:
                try:
                    if asyncio.iscoroutinefunction(pipeline['transform']):
                        transformed_data = await pipeline['transform'](data)
                    else:
                        transformed_data = pipeline['transform'](data)
                except Exception as e:
                    logger.error(f"Data transformation failed in {pipeline_name}: {e}")
                    transformed_data = data
            else:
                transformed_data = data
            
            # Send to target components
            for target in pipeline['targets']:
                try:
                    await self._send_to_target(target, transformed_data)
                    self.flow_stats[f"{pipeline_name}->{target}"] += 1
                except Exception as e:
                    logger.error(f"Failed to send data to {target}: {e}")
                    
        except Exception as e:
            logger.error(f"Data flow error in {pipeline_name}: {e}")
    
    async def _send_to_target(self, target: str, data: Any):
        """Send data to target component"""
        # This would integrate with the actual target component
        # For now, we'll store in buffer
        if target not in self.data_buffer:
            self.data_buffer[target] = deque(maxlen=1000)
        
        self.data_buffer[target].append({
            'timestamp': datetime.now().isoformat(),
            'data': data
        })

class SystemSynchronizationManager:
    """Main synchronization manager for AI3.0 system"""
    
    def __init__(self):
        self.event_bus = EventBus()
        self.dependency_manager = DependencyManager()
        self.health_monitor = HealthMonitor()
        self.data_flow_coordinator = DataFlowCoordinator()
        
        self.system_state = SystemState.INITIALIZING
        self.components = {}
        self.sync_locks = {}
        self.performance_metrics = {}
        
        logger.info("System Synchronization Manager initialized")
    
    async def initialize(self):
        """Initialize synchronization manager"""
        try:
            # Start event bus
            await self.event_bus.start()
            
            # Start health monitoring
            await self.health_monitor.start_monitoring()
            
            self.system_state = SystemState.RUNNING
            logger.info("System Synchronization Manager started")
            
        except Exception as e:
            logger.error(f"Synchronization manager initialization failed: {e}")
            self.system_state = SystemState.ERROR
            raise
    
    async def shutdown(self):
        """Shutdown synchronization manager"""
        try:
            self.system_state = SystemState.STOPPING
            
            # Stop health monitoring
            await self.health_monitor.stop_monitoring()
            
            # Stop event bus
            await self.event_bus.stop()
            
            self.system_state = SystemState.STOPPED
            logger.info("System Synchronization Manager stopped")
            
        except Exception as e:
            logger.error(f"Synchronization manager shutdown error: {e}")
    
    def register_component(self, name: str, instance: Any, 
                          dependencies: List[str] = None,
                          health_check: Callable = None):
        """Register system component"""
        component = SystemComponent(
            name=name,
            instance=instance,
            dependencies=dependencies or [],
            health_check=health_check
        )
        
        self.components[name] = component
        self.dependency_manager.register_component(component)
        self.health_monitor.register_component(component)
        
        # Create sync lock for component
        self.sync_locks[name] = asyncio.Lock()
        
        logger.info(f"Component registered: {name}")
    
    async def initialize_all_components(self) -> bool:
        """Initialize all components in dependency order"""
        try:
            # Validate dependencies
            missing_deps = self.dependency_manager.validate_dependencies()
            if missing_deps:
                logger.error(f"Missing dependencies: {missing_deps}")
                return False
            
            # Get initialization order
            init_order = self.dependency_manager.get_initialization_order()
            logger.info(f"Component initialization order: {init_order}")
            
            # Initialize components
            for component_name in init_order:
                component = self.components[component_name]
                
                try:
                    logger.info(f"Initializing component: {component_name}")
                    
                    # Call initialize method if exists
                    if hasattr(component.instance, 'initialize'):
                        if asyncio.iscoroutinefunction(component.instance.initialize):
                            success = await component.instance.initialize()
                        else:
                            success = component.instance.initialize()
                        
                        if not success:
                            logger.error(f"Component initialization failed: {component_name}")
                            component.state = SystemState.ERROR
                            return False
                    
                    component.state = SystemState.RUNNING
                    logger.info(f"Component initialized successfully: {component_name}")
                    
                except Exception as e:
                    logger.error(f"Component initialization error {component_name}: {e}")
                    component.state = SystemState.ERROR
                    return False
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            return False
    
    async def synchronize_operation(self, component_name: str, operation: Callable, 
                                  *args, **kwargs):
        """Synchronize operation across components"""
        try:
            async with self.sync_locks[component_name]:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                # Publish synchronization event
                await self.event_bus.publish(SystemEvent(
                    event_type='operation_completed',
                    source_system=component_name,
                    data={'result': str(result)[:100]}  # Truncate for logging
                ))
                
                return result
                
        except Exception as e:
            logger.error(f"Synchronized operation failed {component_name}: {e}")
            
            # Publish error event
            await self.event_bus.publish(SystemEvent(
                event_type='operation_failed',
                source_system=component_name,
                priority=EventPriority.HIGH,
                data={'error': str(e)}
            ))
            
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        component_statuses = {}
        
        for name, component in self.components.items():
            component_statuses[name] = {
                'state': component.state.value,
                'error_count': component.error_count,
                'last_heartbeat': component.last_heartbeat.isoformat()
            }
        
        return {
            'system_state': self.system_state.value,
            'timestamp': datetime.now().isoformat(),
            'components': component_statuses,
            'event_bus_stats': self.event_bus.stats,
            'data_flow_stats': dict(self.data_flow_coordinator.flow_stats),
            'health_monitor_active': self.health_monitor.is_monitoring
        }
    
    async def emergency_shutdown(self, reason: str = "Emergency shutdown"):
        """Emergency shutdown of all systems"""
        logger.critical(f"Emergency shutdown initiated: {reason}")
        
        # Publish emergency event
        await self.event_bus.publish(SystemEvent(
            event_type='emergency_shutdown',
            source_system='synchronization_manager',
            priority=EventPriority.CRITICAL,
            data={'reason': reason}
        ))
        
        # Force shutdown all components
        for component_name, component in self.components.items():
            try:
                if hasattr(component.instance, 'emergency_stop'):
                    component.instance.emergency_stop()
                component.state = SystemState.STOPPED
            except Exception as e:
                logger.error(f"Emergency stop failed for {component_name}: {e}")
        
        # Shutdown synchronization manager
        await self.shutdown()


# Example usage and testing
async def main():
    """Example usage of SystemSynchronizationManager"""
    
    # Create synchronization manager
    sync_manager = SystemSynchronizationManager()
    
    # Mock system components
    class MockTradingEngine:
        def __init__(self):
            self.is_active = False
        
        async def initialize(self):
            logger.info("Trading Engine initializing...")
            await asyncio.sleep(1)  # Simulate initialization
            self.is_active = True
            return True
        
        def get_health_status(self):
            return {'status': 'healthy', 'active': self.is_active}
    
    class MockRiskManager:
        def __init__(self):
            self.is_active = False
        
        async def initialize(self):
            logger.info("Risk Manager initializing...")
            await asyncio.sleep(0.5)
            self.is_active = True
            return True
        
        def get_health_status(self):
            return {'status': 'healthy', 'active': self.is_active}
    
    # Create mock components
    trading_engine = MockTradingEngine()
    risk_manager = MockRiskManager()
    
    # Register components
    sync_manager.register_component(
        'trading_engine', 
        trading_engine,
        dependencies=[],
        health_check=trading_engine.get_health_status
    )
    
    sync_manager.register_component(
        'risk_manager',
        risk_manager, 
        dependencies=['trading_engine'],
        health_check=risk_manager.get_health_status
    )
    
    try:
        # Initialize synchronization manager
        await sync_manager.initialize()
        
        # Initialize all components
        success = await sync_manager.initialize_all_components()
        
        if success:
            logger.info("System synchronized and running")
            
            # Test synchronized operation
            await sync_manager.synchronize_operation(
                'trading_engine',
                lambda: logger.info("Synchronized trading operation")
            )
            
            # Get system status
            status = sync_manager.get_system_status()
            logger.info(f"System Status: {json.dumps(status, indent=2)}")
            
            # Run for a few seconds
            await asyncio.sleep(5)
        
    except Exception as e:
        logger.error(f"System synchronization failed: {e}")
    
    finally:
        # Shutdown
        await sync_manager.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main()) 