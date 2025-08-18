"""
Data Plane Implementation for Environment State Management

This module implements the data plane that manages environment state,
simulation execution, and data persistence separate from the control
plane that handles agent decision-making and coordination.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Union
from uuid import uuid4
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import sqlite3
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPlaneStatus(Enum):
    """Status of data plane components"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"
    SHUTDOWN = "shutdown"


class StateChangeType(Enum):
    """Types of state changes in the environment"""
    ENTITY_CREATED = "entity_created"
    ENTITY_UPDATED = "entity_updated"
    ENTITY_DELETED = "entity_deleted"
    RELATIONSHIP_CREATED = "relationship_created"
    RELATIONSHIP_DELETED = "relationship_deleted"
    PROPERTY_CHANGED = "property_changed"


@dataclass
class EnvironmentEntity:
    """Represents an entity in the simulation environment"""
    entity_id: str
    entity_type: str
    properties: Dict[str, Any]
    position: Optional[Dict[str, float]]
    relationships: Dict[str, List[str]]
    created_at: datetime
    updated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary"""
        return {
            **asdict(self),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class StateChange:
    """Represents a change in environment state"""
    change_id: str
    change_type: StateChangeType
    entity_id: str
    old_value: Optional[Any]
    new_value: Optional[Any]
    timestamp: datetime
    source: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state change to dictionary"""
        return {
            **asdict(self),
            'change_type': self.change_type.value,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class SimulationSnapshot:
    """Snapshot of simulation state at a point in time"""
    snapshot_id: str
    timestamp: datetime
    entities: Dict[str, EnvironmentEntity]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary"""
        return {
            'snapshot_id': self.snapshot_id,
            'timestamp': self.timestamp.isoformat(),
            'entities': {k: v.to_dict() for k, v in self.entities.items()},
            'metadata': self.metadata
        }


class DataPlaneInterface(ABC):
    """Abstract interface for data plane components"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the data plane component"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown the data plane component"""
        pass
    
    @abstractmethod
    def get_status(self) -> DataPlaneStatus:
        """Get current status of the component"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        pass


class EnvironmentStateManager(DataPlaneInterface):
    """
    Manages the current state of the simulation environment
    """
    
    def __init__(self, state_db_path: str = "environment_state.db"):
        self.state_db_path = state_db_path
        self.status = DataPlaneStatus.INITIALIZING
        self.entities: Dict[str, EnvironmentEntity] = {}
        self.state_history: List[StateChange] = []
        self.subscribers: Dict[str, List[callable]] = {}
        self.metrics = {
            'entities_count': 0,
            'state_changes': 0,
            'queries_per_second': 0.0,
            'average_query_time': 0.0
        }
        self._lock = threading.RLock()
        self._db_connection: Optional[sqlite3.Connection] = None
        self._query_times: List[float] = []
    
    async def initialize(self) -> bool:
        """Initialize the environment state manager"""
        try:
            logger.info("Initializing environment state manager")
            
            # Initialize database
            self._db_connection = sqlite3.connect(
                self.state_db_path, 
                check_same_thread=False
            )
            
            # Create tables
            await self._create_tables()
            
            # Load existing state
            await self._load_state_from_db()
            
            self.status = DataPlaneStatus.ACTIVE
            logger.info("Environment state manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize environment state manager: {e}")
            self.status = DataPlaneStatus.FAILED
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the environment state manager"""
        try:
            logger.info("Shutting down environment state manager")
            
            # Save current state
            await self._save_state_to_db()
            
            # Close database connection
            if self._db_connection:
                self._db_connection.close()
            
            self.status = DataPlaneStatus.SHUTDOWN
            logger.info("Environment state manager shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to shutdown environment state manager: {e}")
            return False
    
    def get_status(self) -> DataPlaneStatus:
        """Get current status"""
        return self.status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get state manager metrics"""
        with self._lock:
            # Calculate queries per second
            if self._query_times:
                recent_queries = [t for t in self._query_times if time.time() - t < 60]
                qps = len(recent_queries) / 60.0
            else:
                qps = 0.0
            
            return {
                **self.metrics,
                'entities_count': len(self.entities),
                'state_changes': len(self.state_history),
                'queries_per_second': qps
            }
    
    async def _create_tables(self):
        """Create database tables for state persistence"""
        cursor = self._db_connection.cursor()
        
        # Entities table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                properties TEXT NOT NULL,
                position TEXT,
                relationships TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # State changes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS state_changes (
                change_id TEXT PRIMARY KEY,
                change_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL
            )
        ''')
        
        self._db_connection.commit()
    
    async def _load_state_from_db(self):
        """Load existing state from database"""
        cursor = self._db_connection.cursor()
        
        # Load entities
        cursor.execute('SELECT * FROM entities')
        for row in cursor.fetchall():
            entity = EnvironmentEntity(
                entity_id=row[0],
                entity_type=row[1],
                properties=json.loads(row[2]),
                position=json.loads(row[3]) if row[3] else None,
                relationships=json.loads(row[4]),
                created_at=datetime.fromisoformat(row[5]),
                updated_at=datetime.fromisoformat(row[6])
            )
            self.entities[entity.entity_id] = entity
        
        logger.info(f"Loaded {len(self.entities)} entities from database")
    
    async def _save_state_to_db(self):
        """Save current state to database"""
        cursor = self._db_connection.cursor()
        
        # Clear existing data
        cursor.execute('DELETE FROM entities')
        cursor.execute('DELETE FROM state_changes')
        
        # Save entities
        for entity in self.entities.values():
            cursor.execute('''
                INSERT INTO entities VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                entity.entity_id,
                entity.entity_type,
                json.dumps(entity.properties),
                json.dumps(entity.position) if entity.position else None,
                json.dumps(entity.relationships),
                entity.created_at.isoformat(),
                entity.updated_at.isoformat()
            ))
        
        # Save state changes (last 1000)
        recent_changes = self.state_history[-1000:]
        for change in recent_changes:
            cursor.execute('''
                INSERT INTO state_changes VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                change.change_id,
                change.change_type.value,
                change.entity_id,
                json.dumps(change.old_value) if change.old_value else None,
                json.dumps(change.new_value) if change.new_value else None,
                change.timestamp.isoformat(),
                change.source
            ))
        
        self._db_connection.commit()
        logger.info(f"Saved {len(self.entities)} entities and {len(recent_changes)} state changes")
    
    def create_entity(self, 
                     entity_type: str,
                     properties: Dict[str, Any],
                     position: Optional[Dict[str, float]] = None,
                     entity_id: Optional[str] = None) -> str:
        """Create a new entity in the environment"""
        start_time = time.time()
        
        if entity_id is None:
            entity_id = str(uuid4())
        
        now = datetime.now()
        entity = EnvironmentEntity(
            entity_id=entity_id,
            entity_type=entity_type,
            properties=properties.copy(),
            position=position.copy() if position else None,
            relationships={},
            created_at=now,
            updated_at=now
        )
        
        with self._lock:
            self.entities[entity_id] = entity
            
            # Record state change
            change = StateChange(
                change_id=str(uuid4()),
                change_type=StateChangeType.ENTITY_CREATED,
                entity_id=entity_id,
                old_value=None,
                new_value=entity.to_dict(),
                timestamp=now,
                source="environment_state_manager"
            )
            self.state_history.append(change)
            
            # Update metrics
            self.metrics['entities_count'] = len(self.entities)
            self.metrics['state_changes'] = len(self.state_history)
            self._query_times.append(time.time())
        
        # Notify subscribers
        self._notify_subscribers('entity_created', entity)
        
        query_time = time.time() - start_time
        self._update_query_metrics(query_time)
        
        logger.debug(f"Created entity {entity_id} of type {entity_type}")
        return entity_id
    
    def get_entity(self, entity_id: str) -> Optional[EnvironmentEntity]:
        """Get an entity by ID"""
        start_time = time.time()
        
        with self._lock:
            entity = self.entities.get(entity_id)
            self._query_times.append(time.time())
        
        query_time = time.time() - start_time
        self._update_query_metrics(query_time)
        
        return entity
    
    def update_entity(self, 
                     entity_id: str,
                     properties: Optional[Dict[str, Any]] = None,
                     position: Optional[Dict[str, float]] = None) -> bool:
        """Update an existing entity"""
        start_time = time.time()
        
        with self._lock:
            if entity_id not in self.entities:
                return False
            
            entity = self.entities[entity_id]
            old_value = entity.to_dict()
            
            # Update properties
            if properties:
                entity.properties.update(properties)
            
            # Update position
            if position:
                entity.position = position.copy()
            
            entity.updated_at = datetime.now()
            
            # Record state change
            change = StateChange(
                change_id=str(uuid4()),
                change_type=StateChangeType.ENTITY_UPDATED,
                entity_id=entity_id,
                old_value=old_value,
                new_value=entity.to_dict(),
                timestamp=entity.updated_at,
                source="environment_state_manager"
            )
            self.state_history.append(change)
            
            self.metrics['state_changes'] = len(self.state_history)
            self._query_times.append(time.time())
        
        # Notify subscribers
        self._notify_subscribers('entity_updated', entity)
        
        query_time = time.time() - start_time
        self._update_query_metrics(query_time)
        
        logger.debug(f"Updated entity {entity_id}")
        return True
    
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity"""
        start_time = time.time()
        
        with self._lock:
            if entity_id not in self.entities:
                return False
            
            entity = self.entities[entity_id]
            old_value = entity.to_dict()
            
            # Remove entity
            del self.entities[entity_id]
            
            # Record state change
            change = StateChange(
                change_id=str(uuid4()),
                change_type=StateChangeType.ENTITY_DELETED,
                entity_id=entity_id,
                old_value=old_value,
                new_value=None,
                timestamp=datetime.now(),
                source="environment_state_manager"
            )
            self.state_history.append(change)
            
            # Update metrics
            self.metrics['entities_count'] = len(self.entities)
            self.metrics['state_changes'] = len(self.state_history)
            self._query_times.append(time.time())
        
        # Notify subscribers
        self._notify_subscribers('entity_deleted', {'entity_id': entity_id})
        
        query_time = time.time() - start_time
        self._update_query_metrics(query_time)
        
        logger.debug(f"Deleted entity {entity_id}")
        return True
    
    def query_entities(self, 
                      entity_type: Optional[str] = None,
                      properties_filter: Optional[Dict[str, Any]] = None,
                      position_filter: Optional[Dict[str, Any]] = None) -> List[EnvironmentEntity]:
        """Query entities based on filters"""
        start_time = time.time()
        
        results = []
        
        with self._lock:
            for entity in self.entities.values():
                # Type filter
                if entity_type and entity.entity_type != entity_type:
                    continue
                
                # Properties filter
                if properties_filter:
                    match = True
                    for key, value in properties_filter.items():
                        if key not in entity.properties or entity.properties[key] != value:
                            match = False
                            break
                    if not match:
                        continue
                
                # Position filter (simple range check)
                if position_filter and entity.position:
                    if 'x_min' in position_filter and entity.position.get('x', 0) < position_filter['x_min']:
                        continue
                    if 'x_max' in position_filter and entity.position.get('x', 0) > position_filter['x_max']:
                        continue
                    if 'y_min' in position_filter and entity.position.get('y', 0) < position_filter['y_min']:
                        continue
                    if 'y_max' in position_filter and entity.position.get('y', 0) > position_filter['y_max']:
                        continue
                
                results.append(entity)
            
            self._query_times.append(time.time())
        
        query_time = time.time() - start_time
        self._update_query_metrics(query_time)
        
        return results
    
    def create_relationship(self, 
                          entity1_id: str,
                          entity2_id: str,
                          relationship_type: str) -> bool:
        """Create a relationship between two entities"""
        with self._lock:
            if entity1_id not in self.entities or entity2_id not in self.entities:
                return False
            
            entity1 = self.entities[entity1_id]
            entity2 = self.entities[entity2_id]
            
            # Add relationship
            if relationship_type not in entity1.relationships:
                entity1.relationships[relationship_type] = []
            if entity2_id not in entity1.relationships[relationship_type]:
                entity1.relationships[relationship_type].append(entity2_id)
            
            # Add reverse relationship
            reverse_type = f"reverse_{relationship_type}"
            if reverse_type not in entity2.relationships:
                entity2.relationships[reverse_type] = []
            if entity1_id not in entity2.relationships[reverse_type]:
                entity2.relationships[reverse_type].append(entity1_id)
            
            # Update timestamps
            now = datetime.now()
            entity1.updated_at = now
            entity2.updated_at = now
            
            # Record state change
            change = StateChange(
                change_id=str(uuid4()),
                change_type=StateChangeType.RELATIONSHIP_CREATED,
                entity_id=entity1_id,
                old_value=None,
                new_value={
                    'target_entity': entity2_id,
                    'relationship_type': relationship_type
                },
                timestamp=now,
                source="environment_state_manager"
            )
            self.state_history.append(change)
            
            self.metrics['state_changes'] = len(self.state_history)
        
        logger.debug(f"Created relationship {relationship_type} between {entity1_id} and {entity2_id}")
        return True
    
    def subscribe_to_changes(self, event_type: str, callback: callable):
        """Subscribe to state change notifications"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    def _notify_subscribers(self, event_type: str, data: Any):
        """Notify subscribers of state changes"""
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in subscriber callback: {e}")
    
    def _update_query_metrics(self, query_time: float):
        """Update query performance metrics"""
        with self._lock:
            # Keep only recent query times (last 1000)
            if len(self._query_times) > 1000:
                self._query_times = self._query_times[-1000:]
            
            # Calculate average query time
            if self._query_times:
                self.metrics['average_query_time'] = sum(self._query_times) / len(self._query_times)
    
    def create_snapshot(self) -> SimulationSnapshot:
        """Create a snapshot of current simulation state"""
        with self._lock:
            snapshot = SimulationSnapshot(
                snapshot_id=str(uuid4()),
                timestamp=datetime.now(),
                entities=self.entities.copy(),
                metadata={
                    'total_entities': len(self.entities),
                    'total_state_changes': len(self.state_history)
                }
            )
        
        logger.info(f"Created simulation snapshot {snapshot.snapshot_id}")
        return snapshot


class SimulationExecutor(DataPlaneInterface):
    """
    Manages simulation execution and time progression
    """
    
    def __init__(self, state_manager: EnvironmentStateManager):
        self.state_manager = state_manager
        self.status = DataPlaneStatus.INITIALIZING
        self.simulation_time = 0.0
        self.time_scale = 1.0
        self.is_running = False
        self.execution_tasks: List[asyncio.Task] = []
        self.metrics = {
            'simulation_time': 0.0,
            'real_time_elapsed': 0.0,
            'time_scale': 1.0,
            'ticks_per_second': 0.0
        }
        self._lock = threading.Lock()
        self._start_real_time: Optional[float] = None
        self._tick_times: List[float] = []
    
    async def initialize(self) -> bool:
        """Initialize the simulation executor"""
        try:
            logger.info("Initializing simulation executor")
            self.status = DataPlaneStatus.ACTIVE
            return True
        except Exception as e:
            logger.error(f"Failed to initialize simulation executor: {e}")
            self.status = DataPlaneStatus.FAILED
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the simulation executor"""
        try:
            logger.info("Shutting down simulation executor")
            
            # Stop simulation
            await self.stop_simulation()
            
            # Cancel all tasks
            for task in self.execution_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            self.status = DataPlaneStatus.SHUTDOWN
            return True
        except Exception as e:
            logger.error(f"Failed to shutdown simulation executor: {e}")
            return False
    
    def get_status(self) -> DataPlaneStatus:
        """Get current status"""
        return self.status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get simulation executor metrics"""
        with self._lock:
            real_time_elapsed = 0.0
            if self._start_real_time:
                real_time_elapsed = time.time() - self._start_real_time
            
            # Calculate ticks per second
            recent_ticks = [t for t in self._tick_times if time.time() - t < 60]
            tps = len(recent_ticks) / 60.0
            
            return {
                **self.metrics,
                'simulation_time': self.simulation_time,
                'real_time_elapsed': real_time_elapsed,
                'time_scale': self.time_scale,
                'ticks_per_second': tps,
                'is_running': self.is_running
            }
    
    async def start_simulation(self, time_scale: float = 1.0):
        """Start the simulation"""
        with self._lock:
            if self.is_running:
                logger.warning("Simulation is already running")
                return
            
            self.time_scale = time_scale
            self.is_running = True
            self._start_real_time = time.time()
        
        # Start simulation loop
        task = asyncio.create_task(self._simulation_loop())
        self.execution_tasks.append(task)
        
        logger.info(f"Simulation started with time scale {time_scale}")
    
    async def stop_simulation(self):
        """Stop the simulation"""
        with self._lock:
            self.is_running = False
        
        logger.info("Simulation stopped")
    
    async def pause_simulation(self):
        """Pause the simulation"""
        with self._lock:
            self.is_running = False
        
        logger.info("Simulation paused")
    
    async def resume_simulation(self):
        """Resume the simulation"""
        with self._lock:
            self.is_running = True
            self._start_real_time = time.time()
        
        logger.info("Simulation resumed")
    
    def set_time_scale(self, time_scale: float):
        """Set the simulation time scale"""
        with self._lock:
            self.time_scale = time_scale
        
        logger.info(f"Time scale set to {time_scale}")
    
    async def _simulation_loop(self):
        """Main simulation execution loop"""
        tick_interval = 0.1  # 10 ticks per second
        
        while self.is_running:
            try:
                tick_start = time.time()
                
                # Advance simulation time
                with self._lock:
                    self.simulation_time += tick_interval * self.time_scale
                    self._tick_times.append(time.time())
                    
                    # Keep only recent tick times
                    if len(self._tick_times) > 600:  # 10 minutes at 10 TPS
                        self._tick_times = self._tick_times[-600:]
                
                # Process simulation tick
                await self._process_simulation_tick()
                
                # Sleep for remaining tick time
                tick_duration = time.time() - tick_start
                sleep_time = max(0, tick_interval - tick_duration)
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                await asyncio.sleep(1)  # Brief pause before retry
    
    async def _process_simulation_tick(self):
        """Process a single simulation tick"""
        # This is where simulation logic would be executed
        # For now, just update metrics
        pass


class DataPlaneOrchestrator:
    """
    Main orchestrator for the data plane
    """
    
    def __init__(self, state_db_path: str = "data_plane_state.db"):
        self.status = DataPlaneStatus.INITIALIZING
        self.state_manager = EnvironmentStateManager(state_db_path)
        self.simulation_executor = SimulationExecutor(self.state_manager)
        self.snapshots: Dict[str, SimulationSnapshot] = {}
        self._lock = threading.Lock()
    
    async def initialize(self) -> bool:
        """Initialize the data plane orchestrator"""
        try:
            logger.info("Initializing data plane orchestrator")
            
            # Initialize state manager
            if not await self.state_manager.initialize():
                raise Exception("Failed to initialize state manager")
            
            # Initialize simulation executor
            if not await self.simulation_executor.initialize():
                raise Exception("Failed to initialize simulation executor")
            
            self.status = DataPlaneStatus.ACTIVE
            logger.info("Data plane orchestrator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize data plane orchestrator: {e}")
            self.status = DataPlaneStatus.FAILED
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the data plane orchestrator"""
        try:
            logger.info("Shutting down data plane orchestrator")
            
            # Shutdown simulation executor
            await self.simulation_executor.shutdown()
            
            # Shutdown state manager
            await self.state_manager.shutdown()
            
            self.status = DataPlaneStatus.SHUTDOWN
            logger.info("Data plane orchestrator shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to shutdown data plane orchestrator: {e}")
            return False
    
    def get_state_manager(self) -> EnvironmentStateManager:
        """Get the environment state manager"""
        return self.state_manager
    
    def get_simulation_executor(self) -> SimulationExecutor:
        """Get the simulation executor"""
        return self.simulation_executor
    
    def create_snapshot(self) -> str:
        """Create a snapshot of current data plane state"""
        snapshot = self.state_manager.create_snapshot()
        
        with self._lock:
            self.snapshots[snapshot.snapshot_id] = snapshot
        
        return snapshot.snapshot_id
    
    def get_snapshot(self, snapshot_id: str) -> Optional[SimulationSnapshot]:
        """Get a specific snapshot"""
        with self._lock:
            return self.snapshots.get(snapshot_id)
    
    def list_snapshots(self) -> List[str]:
        """List all available snapshots"""
        with self._lock:
            return list(self.snapshots.keys())
    
    def get_overall_metrics(self) -> Dict[str, Any]:
        """Get overall data plane metrics"""
        state_metrics = self.state_manager.get_metrics()
        sim_metrics = self.simulation_executor.get_metrics()
        
        return {
            'status': self.status.value,
            'state_manager': state_metrics,
            'simulation_executor': sim_metrics,
            'snapshots_count': len(self.snapshots)
        }


# Global data plane instance
_data_plane_orchestrator: Optional[DataPlaneOrchestrator] = None


async def get_data_plane() -> DataPlaneOrchestrator:
    """Get or create the global data plane orchestrator"""
    global _data_plane_orchestrator
    
    if _data_plane_orchestrator is None:
        _data_plane_orchestrator = DataPlaneOrchestrator()
        await _data_plane_orchestrator.initialize()
    
    return _data_plane_orchestrator


async def shutdown_data_plane():
    """Shutdown the global data plane orchestrator"""
    global _data_plane_orchestrator
    
    if _data_plane_orchestrator is not None:
        await _data_plane_orchestrator.shutdown()
        _data_plane_orchestrator = None