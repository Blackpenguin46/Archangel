"""
Plane Coordinator for Control and Data Plane Integration

This module provides the coordination layer between the control plane
(agent decision-making) and data plane (environment state management),
ensuring proper separation while enabling necessary communication.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from uuid import uuid4
import threading
import time

from .control_plane import (
    ControlPlaneOrchestrator, 
    DecisionEngine, 
    AgentDecision, 
    DecisionType,
    get_control_plane
)
from .data_plane import (
    DataPlaneOrchestrator,
    EnvironmentStateManager,
    SimulationExecutor,
    EnvironmentEntity,
    get_data_plane
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoordinationStatus(Enum):
    """Status of plane coordination"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"
    SHUTDOWN = "shutdown"


class ActionType(Enum):
    """Types of actions that cross plane boundaries"""
    ENVIRONMENT_QUERY = "environment_query"
    ENVIRONMENT_MODIFY = "environment_modify"
    STATE_NOTIFICATION = "state_notification"
    DECISION_EXECUTION = "decision_execution"


@dataclass
class PlaneMessage:
    """Message passed between control and data planes"""
    message_id: str
    source_plane: str
    target_plane: str
    action_type: ActionType
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            **asdict(self),
            'action_type': self.action_type.value,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ActionRequest:
    """Request for action execution across planes"""
    request_id: str
    agent_id: str
    action_type: ActionType
    parameters: Dict[str, Any]
    timeout: timedelta
    priority: int
    timestamp: datetime


@dataclass
class ActionResult:
    """Result of action execution"""
    request_id: str
    success: bool
    result_data: Optional[Dict[str, Any]]
    error_message: Optional[str]
    execution_time: float
    timestamp: datetime


class PlaneInterface(ABC):
    """Abstract interface for plane communication"""
    
    @abstractmethod
    async def send_message(self, message: PlaneMessage) -> bool:
        """Send a message to the other plane"""
        pass
    
    @abstractmethod
    async def receive_message(self, message: PlaneMessage) -> Optional[PlaneMessage]:
        """Receive and process a message from the other plane"""
        pass
    
    @abstractmethod
    def register_handler(self, action_type: ActionType, handler: Callable):
        """Register a handler for specific action types"""
        pass


class ControlPlaneInterface(PlaneInterface):
    """Interface for control plane communication"""
    
    def __init__(self, coordinator: 'PlaneCoordinator'):
        self.coordinator = coordinator
        self.handlers: Dict[ActionType, Callable] = {}
        self.pending_requests: Dict[str, ActionRequest] = {}
        self._lock = threading.Lock()
    
    async def send_message(self, message: PlaneMessage) -> bool:
        """Send message to data plane"""
        try:
            return await self.coordinator.route_message_to_data_plane(message)
        except Exception as e:
            logger.error(f"Failed to send message to data plane: {e}")
            return False
    
    async def receive_message(self, message: PlaneMessage) -> Optional[PlaneMessage]:
        """Receive and process message from data plane"""
        try:
            if message.action_type in self.handlers:
                handler = self.handlers[message.action_type]
                result = await handler(message)
                
                if result:
                    response = PlaneMessage(
                        message_id=str(uuid4()),
                        source_plane="control",
                        target_plane="data",
                        action_type=message.action_type,
                        payload=result,
                        timestamp=datetime.now(),
                        correlation_id=message.message_id
                    )
                    return response
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to process message from data plane: {e}")
            return None
    
    def register_handler(self, action_type: ActionType, handler: Callable):
        """Register handler for action type"""
        self.handlers[action_type] = handler
        logger.debug(f"Registered control plane handler for {action_type}")
    
    async def execute_environment_query(self, 
                                      agent_id: str,
                                      query_type: str,
                                      parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a query against the environment state"""
        # For simplicity in this implementation, directly call the data plane
        # In a production system, this would use proper async messaging
        try:
            from .data_plane import get_data_plane
            
            data_plane = await get_data_plane()
            state_manager = data_plane.get_state_manager()
            
            if query_type == "get_entity":
                entity_id = parameters.get('entity_id')
                entity = state_manager.get_entity(entity_id)
                return {
                    'success': True,
                    'entity': entity.to_dict() if entity else None
                }
            elif query_type == "query_entities":
                entity_type = parameters.get('entity_type')
                properties_filter = parameters.get('properties_filter')
                entities = state_manager.query_entities(entity_type, properties_filter)
                return {
                    'success': True,
                    'entities': [e.to_dict() for e in entities]
                }
            else:
                return {'success': False, 'error': f'Unknown query type: {query_type}'}
                
        except Exception as e:
            logger.error(f"Error executing environment query: {e}")
            return {'success': False, 'error': str(e)}
    
    async def execute_environment_modification(self,
                                            agent_id: str,
                                            modification_type: str,
                                            parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a modification to the environment state"""
        # For simplicity in this implementation, directly call the data plane
        # In a production system, this would use proper async messaging
        try:
            from .data_plane import get_data_plane
            
            data_plane = await get_data_plane()
            state_manager = data_plane.get_state_manager()
            
            if modification_type == "create_entity":
                entity_type = parameters.get('entity_type')
                properties = parameters.get('properties', {})
                position = parameters.get('position')
                
                entity_id = state_manager.create_entity(entity_type, properties, position)
                return {
                    'success': True,
                    'entity_id': entity_id
                }
            elif modification_type == "update_entity":
                entity_id = parameters.get('entity_id')
                properties = parameters.get('properties')
                position = parameters.get('position')
                
                success = state_manager.update_entity(entity_id, properties, position)
                return {
                    'success': success
                }
            elif modification_type == "delete_entity":
                entity_id = parameters.get('entity_id')
                success = state_manager.delete_entity(entity_id)
                return {
                    'success': success
                }
            else:
                return {'success': False, 'error': f'Unknown modification type: {modification_type}'}
                
        except Exception as e:
            logger.error(f"Error executing environment modification: {e}")
            return {'success': False, 'error': str(e)}


class DataPlaneInterface(PlaneInterface):
    """Interface for data plane communication"""
    
    def __init__(self, coordinator: 'PlaneCoordinator'):
        self.coordinator = coordinator
        self.handlers: Dict[ActionType, Callable] = {}
        self.state_subscribers: List[Callable] = []
        self._lock = threading.Lock()
    
    async def send_message(self, message: PlaneMessage) -> bool:
        """Send message to control plane"""
        try:
            return await self.coordinator.route_message_to_control_plane(message)
        except Exception as e:
            logger.error(f"Failed to send message to control plane: {e}")
            return False
    
    async def receive_message(self, message: PlaneMessage) -> Optional[PlaneMessage]:
        """Receive and process message from control plane"""
        try:
            if message.action_type in self.handlers:
                handler = self.handlers[message.action_type]
                result = await handler(message)
                
                if result:
                    response = PlaneMessage(
                        message_id=str(uuid4()),
                        source_plane="data",
                        target_plane="control",
                        action_type=message.action_type,
                        payload=result,
                        timestamp=datetime.now(),
                        correlation_id=message.message_id
                    )
                    return response
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to process message from control plane: {e}")
            return None
    
    def register_handler(self, action_type: ActionType, handler: Callable):
        """Register handler for action type"""
        self.handlers[action_type] = handler
        logger.debug(f"Registered data plane handler for {action_type}")
    
    def subscribe_to_state_changes(self, callback: Callable):
        """Subscribe to state change notifications"""
        with self._lock:
            self.state_subscribers.append(callback)
    
    async def notify_state_change(self, change_data: Dict[str, Any]):
        """Notify control plane of state changes"""
        message = PlaneMessage(
            message_id=str(uuid4()),
            source_plane="data",
            target_plane="control",
            action_type=ActionType.STATE_NOTIFICATION,
            payload=change_data,
            timestamp=datetime.now()
        )
        
        await self.send_message(message)
    
    async def handle_environment_query(self, message: PlaneMessage) -> Dict[str, Any]:
        """Handle environment query from control plane"""
        try:
            payload = message.payload
            
            # Extract parameters from the ActionRequest structure
            if 'parameters' in payload:
                parameters = payload['parameters']
                query_type = parameters.get('query_type')
            else:
                query_type = payload.get('query_type')
                parameters = payload
            
            if query_type:
                # Get data plane for actual query execution
                data_plane = await get_data_plane()
                state_manager = data_plane.get_state_manager()
                
                if query_type == "get_entity":
                    entity_id = parameters.get('entity_id')
                    entity = state_manager.get_entity(entity_id)
                    return {
                        'success': True,
                        'entity': entity.to_dict() if entity else None
                    }
                elif query_type == "query_entities":
                    entity_type = parameters.get('entity_type')
                    properties_filter = parameters.get('properties_filter')
                    entities = state_manager.query_entities(entity_type, properties_filter)
                    return {
                        'success': True,
                        'entities': [e.to_dict() for e in entities]
                    }
                else:
                    return {'success': False, 'error': f'Unknown query type: {query_type}'}
            
            return {'success': False, 'error': 'No query type specified'}
            
        except Exception as e:
            logger.error(f"Error handling environment query: {e}")
            return {'success': False, 'error': str(e)}
    
    async def handle_environment_modification(self, message: PlaneMessage) -> Dict[str, Any]:
        """Handle environment modification from control plane"""
        try:
            payload = message.payload
            
            # Extract parameters from the ActionRequest structure
            if 'parameters' in payload:
                parameters = payload['parameters']
                mod_type = parameters.get('modification_type')
            else:
                mod_type = payload.get('modification_type')
                parameters = payload
            
            if mod_type:
                # Get data plane for actual modification
                data_plane = await get_data_plane()
                state_manager = data_plane.get_state_manager()
                
                if mod_type == "create_entity":
                    entity_type = parameters.get('entity_type')
                    properties = parameters.get('properties', {})
                    position = parameters.get('position')
                    
                    entity_id = state_manager.create_entity(entity_type, properties, position)
                    return {
                        'success': True,
                        'entity_id': entity_id
                    }
                elif mod_type == "update_entity":
                    entity_id = parameters.get('entity_id')
                    properties = parameters.get('properties')
                    position = parameters.get('position')
                    
                    success = state_manager.update_entity(entity_id, properties, position)
                    return {
                        'success': success
                    }
                elif mod_type == "delete_entity":
                    entity_id = parameters.get('entity_id')
                    success = state_manager.delete_entity(entity_id)
                    return {
                        'success': success
                    }
                else:
                    return {'success': False, 'error': f'Unknown modification type: {mod_type}'}
            
            return {'success': False, 'error': 'No modification type specified'}
            
        except Exception as e:
            logger.error(f"Error handling environment modification: {e}")
            return {'success': False, 'error': str(e)}


class PlaneCoordinator:
    """
    Main coordinator between control and data planes
    """
    
    def __init__(self):
        self.status = CoordinationStatus.INITIALIZING
        self.control_plane_interface = ControlPlaneInterface(self)
        self.data_plane_interface = DataPlaneInterface(self)
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.metrics = {
            'messages_processed': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'average_response_time': 0.0
        }
        self._lock = threading.Lock()
        self._processing_task: Optional[asyncio.Task] = None
        self._response_times: List[float] = []
    
    async def initialize(self) -> bool:
        """Initialize the plane coordinator"""
        try:
            logger.info("Initializing plane coordinator")
            
            # Register default handlers
            self.data_plane_interface.register_handler(
                ActionType.ENVIRONMENT_QUERY,
                self.data_plane_interface.handle_environment_query
            )
            self.data_plane_interface.register_handler(
                ActionType.ENVIRONMENT_MODIFY,
                self.data_plane_interface.handle_environment_modification
            )
            
            # Start message processing
            self._processing_task = asyncio.create_task(self._process_messages())
            
            self.status = CoordinationStatus.ACTIVE
            logger.info("Plane coordinator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize plane coordinator: {e}")
            self.status = CoordinationStatus.FAILED
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the plane coordinator"""
        try:
            logger.info("Shutting down plane coordinator")
            
            # Cancel processing task
            if self._processing_task:
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except asyncio.CancelledError:
                    pass
            
            self.status = CoordinationStatus.SHUTDOWN
            logger.info("Plane coordinator shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to shutdown plane coordinator: {e}")
            return False
    
    def get_control_plane_interface(self) -> ControlPlaneInterface:
        """Get the control plane interface"""
        return self.control_plane_interface
    
    def get_data_plane_interface(self) -> DataPlaneInterface:
        """Get the data plane interface"""
        return self.data_plane_interface
    
    async def route_message_to_control_plane(self, message: PlaneMessage) -> bool:
        """Route message to control plane"""
        try:
            await self.message_queue.put(('control', message))
            return True
        except Exception as e:
            logger.error(f"Failed to route message to control plane: {e}")
            return False
    
    async def route_message_to_data_plane(self, message: PlaneMessage) -> bool:
        """Route message to data plane"""
        try:
            await self.message_queue.put(('data', message))
            return True
        except Exception as e:
            logger.error(f"Failed to route message to data plane: {e}")
            return False
    
    async def _process_messages(self):
        """Process messages between planes"""
        while self.status == CoordinationStatus.ACTIVE:
            try:
                # Get message from queue with timeout
                target_plane, message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                
                start_time = time.time()
                
                # Route message to appropriate plane
                response = None
                if target_plane == 'control':
                    response = await self.control_plane_interface.receive_message(message)
                elif target_plane == 'data':
                    response = await self.data_plane_interface.receive_message(message)
                
                # Update metrics
                response_time = time.time() - start_time
                with self._lock:
                    self.metrics['messages_processed'] += 1
                    if response:
                        self.metrics['successful_operations'] += 1
                    else:
                        self.metrics['failed_operations'] += 1
                    
                    self._response_times.append(response_time)
                    if len(self._response_times) > 1000:
                        self._response_times = self._response_times[-1000:]
                    
                    self.metrics['average_response_time'] = (
                        sum(self._response_times) / len(self._response_times)
                    )
                
                # Send response if generated
                if response:
                    if response.target_plane == 'control':
                        await self.route_message_to_control_plane(response)
                    elif response.target_plane == 'data':
                        await self.route_message_to_data_plane(response)
                
            except asyncio.TimeoutError:
                # No messages to process, continue
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await asyncio.sleep(0.1)  # Brief pause before retry
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get coordination metrics"""
        with self._lock:
            return {
                **self.metrics,
                'status': self.status.value,
                'queue_size': self.message_queue.qsize()
            }


# Global plane coordinator instance
_plane_coordinator: Optional[PlaneCoordinator] = None


async def get_plane_coordinator() -> PlaneCoordinator:
    """Get or create the global plane coordinator"""
    global _plane_coordinator
    
    if _plane_coordinator is None:
        _plane_coordinator = PlaneCoordinator()
        await _plane_coordinator.initialize()
    
    return _plane_coordinator


async def shutdown_plane_coordinator():
    """Shutdown the global plane coordinator"""
    global _plane_coordinator
    
    if _plane_coordinator is not None:
        await _plane_coordinator.shutdown()
        _plane_coordinator = None


class AgentPlaneAdapter:
    """
    Adapter that allows agents to interact with both planes through a unified interface
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._coordinator: Optional[PlaneCoordinator] = None
        self._control_interface: Optional[ControlPlaneInterface] = None
        self._data_interface: Optional[DataPlaneInterface] = None
    
    async def initialize(self):
        """Initialize the adapter"""
        self._coordinator = await get_plane_coordinator()
        self._control_interface = self._coordinator.get_control_plane_interface()
        self._data_interface = self._coordinator.get_data_plane_interface()
    
    async def query_environment(self, 
                              query_type: str,
                              **parameters) -> Optional[Dict[str, Any]]:
        """Query the environment state"""
        if not self._control_interface:
            await self.initialize()
        
        return await self._control_interface.execute_environment_query(
            self.agent_id, query_type, parameters
        )
    
    async def modify_environment(self,
                               modification_type: str,
                               **parameters) -> Optional[Dict[str, Any]]:
        """Modify the environment state"""
        if not self._control_interface:
            await self.initialize()
        
        return await self._control_interface.execute_environment_modification(
            self.agent_id, modification_type, parameters
        )
    
    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific entity from the environment"""
        return await self.query_environment("get_entity", entity_id=entity_id)
    
    async def find_entities(self, 
                          entity_type: Optional[str] = None,
                          **filters) -> List[Dict[str, Any]]:
        """Find entities matching criteria"""
        result = await self.query_environment(
            "query_entities",
            entity_type=entity_type,
            properties_filter=filters
        )
        
        if result and result.get('success'):
            return result.get('entities', [])
        return []
    
    async def create_entity(self,
                          entity_type: str,
                          properties: Dict[str, Any],
                          position: Optional[Dict[str, float]] = None) -> Optional[str]:
        """Create a new entity in the environment"""
        result = await self.modify_environment(
            "create_entity",
            entity_type=entity_type,
            properties=properties,
            position=position
        )
        
        if result and result.get('success'):
            return result.get('entity_id')
        return None
    
    async def update_entity(self,
                          entity_id: str,
                          properties: Optional[Dict[str, Any]] = None,
                          position: Optional[Dict[str, float]] = None) -> bool:
        """Update an existing entity"""
        result = await self.modify_environment(
            "update_entity",
            entity_id=entity_id,
            properties=properties,
            position=position
        )
        
        return result and result.get('success', False)
    
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity from the environment"""
        result = await self.modify_environment(
            "delete_entity",
            entity_id=entity_id
        )
        
        return result and result.get('success', False)