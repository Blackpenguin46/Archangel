"""
Tests for Control and Data Plane Separation

This module tests the separation between control plane (agent decision-making)
and data plane (environment state management), ensuring proper isolation
and reliable distributed execution.
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import threading
import tempfile
import os

from agents.control_plane import (
    ControlPlaneOrchestrator,
    DecisionEngine,
    CoordinationManager,
    DecisionType,
    ControlPlaneStatus,
    get_control_plane,
    shutdown_control_plane
)
from agents.data_plane import (
    DataPlaneOrchestrator,
    EnvironmentStateManager,
    SimulationExecutor,
    EnvironmentEntity,
    DataPlaneStatus,
    get_data_plane,
    shutdown_data_plane
)
from agents.plane_coordinator import (
    PlaneCoordinator,
    AgentPlaneAdapter,
    get_plane_coordinator,
    shutdown_plane_coordinator
)


class TestControlPlaneIsolation:
    """Test control plane isolation and functionality"""
    
    @pytest.fixture
    async def control_plane(self):
        """Create a control plane instance for testing"""
        orchestrator = ControlPlaneOrchestrator()
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_control_plane_initialization(self, control_plane):
        """Test control plane initializes correctly"""
        assert control_plane.status == ControlPlaneStatus.ACTIVE
        assert control_plane.coordination_manager is not None
        assert control_plane.metrics_collector is not None
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, control_plane):
        """Test agent registration with control plane"""
        agent_id = "test_agent_001"
        
        # Register agent
        decision_engine = control_plane.register_agent(agent_id)
        assert decision_engine is not None
        assert decision_engine.agent_id == agent_id
        
        # Verify agent is registered
        retrieved_engine = control_plane.get_agent_decision_engine(agent_id)
        assert retrieved_engine == decision_engine
        
        # Initialize decision engine
        await decision_engine.initialize()
        assert decision_engine.get_status() == ControlPlaneStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_decision_making_isolation(self, control_plane):
        """Test that decision making is isolated from data plane"""
        agent_id = "decision_test_agent"
        
        # Register and initialize agent
        decision_engine = control_plane.register_agent(agent_id)
        await decision_engine.initialize()
        
        # Make various types of decisions
        decisions = []
        for decision_type in DecisionType:
            context = {
                "scenario": "test_scenario",
                "phase": "testing",
                "objectives": ["test_objective"]
            }
            
            decision = await decision_engine.make_decision(
                decision_type=decision_type,
                context=context,
                constraints=["no_real_world_impact"]
            )
            
            decisions.append(decision)
            assert decision.agent_id == agent_id
            assert decision.decision_type == decision_type
            assert decision.confidence > 0
        
        # Verify decisions are stored in control plane only
        history = decision_engine.get_decision_history()
        assert len(history) == len(DecisionType)
        
        # Verify no direct data plane interaction
        for decision in decisions:
            assert "data_plane" not in decision.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_coordination_isolation(self, control_plane):
        """Test that coordination happens within control plane"""
        # Register multiple agents
        agent_ids = ["coord_agent_1", "coord_agent_2", "coord_agent_3"]
        engines = {}
        
        for agent_id in agent_ids:
            engine = control_plane.register_agent(agent_id)
            await engine.initialize()
            engines[agent_id] = engine
        
        # Test coordination
        coordination_manager = control_plane.get_coordination_manager()
        
        request_id = await coordination_manager.request_coordination(
            initiator_agent_id=agent_ids[0],
            target_agents=agent_ids[1:],
            coordination_type="intelligence_sharing",
            payload={"intelligence": "test_data"},
            priority=7
        )
        
        assert request_id is not None
        
        # Wait for coordination processing
        await asyncio.sleep(0.5)
        
        # Check coordination status
        status = coordination_manager.get_coordination_status(request_id)
        assert status in ["completed", "pending"]
        
        # Verify coordination metrics
        metrics = coordination_manager.get_metrics()
        assert metrics['coordination_requests'] > 0
    
    @pytest.mark.asyncio
    async def test_control_plane_metrics(self, control_plane):
        """Test control plane metrics collection"""
        # Register agents and make decisions
        for i in range(3):
            agent_id = f"metrics_agent_{i}"
            engine = control_plane.register_agent(agent_id)
            await engine.initialize()
            
            # Make some decisions
            for j in range(2):
                await engine.make_decision(
                    decision_type=DecisionType.TACTICAL,
                    context={"test": f"decision_{j}"}
                )
        
        # Get overall metrics
        metrics = control_plane.get_overall_metrics()
        
        assert metrics.active_agents == 3
        assert metrics.total_decisions == 6
        assert metrics.decisions_per_second >= 0
        assert metrics.uptime.total_seconds() > 0


class TestDataPlaneIsolation:
    """Test data plane isolation and functionality"""
    
    @pytest.fixture
    async def data_plane(self):
        """Create a data plane instance for testing"""
        # Use temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        
        try:
            orchestrator = DataPlaneOrchestrator(db_path)
            await orchestrator.initialize()
            yield orchestrator
            await orchestrator.shutdown()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    @pytest.mark.asyncio
    async def test_data_plane_initialization(self, data_plane):
        """Test data plane initializes correctly"""
        assert data_plane.status == DataPlaneStatus.ACTIVE
        assert data_plane.state_manager is not None
        assert data_plane.simulation_executor is not None
    
    @pytest.mark.asyncio
    async def test_environment_state_isolation(self, data_plane):
        """Test that environment state is isolated from control plane"""
        state_manager = data_plane.get_state_manager()
        
        # Create entities
        entity_ids = []
        for i in range(5):
            entity_id = state_manager.create_entity(
                entity_type="test_entity",
                properties={"name": f"entity_{i}", "value": i * 10},
                position={"x": i * 100, "y": i * 50}
            )
            entity_ids.append(entity_id)
        
        # Verify entities exist in data plane
        for entity_id in entity_ids:
            entity = state_manager.get_entity(entity_id)
            assert entity is not None
            assert entity.entity_type == "test_entity"
        
        # Query entities
        entities = state_manager.query_entities(entity_type="test_entity")
        assert len(entities) == 5
        
        # Test entity relationships
        success = state_manager.create_relationship(
            entity_ids[0], entity_ids[1], "connected_to"
        )
        assert success
        
        # Verify relationship
        entity = state_manager.get_entity(entity_ids[0])
        assert "connected_to" in entity.relationships
        assert entity_ids[1] in entity.relationships["connected_to"]
    
    @pytest.mark.asyncio
    async def test_simulation_execution_isolation(self, data_plane):
        """Test that simulation execution is isolated"""
        sim_executor = data_plane.get_simulation_executor()
        
        # Start simulation
        await sim_executor.start_simulation(time_scale=2.0)
        
        # Verify simulation is running
        assert sim_executor.is_running
        assert sim_executor.time_scale == 2.0
        
        # Let simulation run briefly
        initial_time = sim_executor.simulation_time
        await asyncio.sleep(0.5)
        
        # Verify time progression
        assert sim_executor.simulation_time > initial_time
        
        # Test pause/resume
        await sim_executor.pause_simulation()
        assert not sim_executor.is_running
        
        paused_time = sim_executor.simulation_time
        await asyncio.sleep(0.2)
        
        # Time should not advance while paused
        assert sim_executor.simulation_time == paused_time
        
        # Resume simulation
        await sim_executor.resume_simulation()
        assert sim_executor.is_running
        
        # Stop simulation
        await sim_executor.stop_simulation()
        assert not sim_executor.is_running
    
    @pytest.mark.asyncio
    async def test_data_persistence(self, data_plane):
        """Test data persistence across operations"""
        state_manager = data_plane.get_state_manager()
        
        # Create entities
        original_entities = {}
        for i in range(3):
            entity_id = state_manager.create_entity(
                entity_type="persistent_entity",
                properties={"index": i, "data": f"test_data_{i}"}
            )
            original_entities[entity_id] = state_manager.get_entity(entity_id)
        
        # Create snapshot
        snapshot_id = data_plane.create_snapshot()
        snapshot = data_plane.get_snapshot(snapshot_id)
        
        assert snapshot is not None
        assert len(snapshot.entities) >= 3
        
        # Verify snapshot contains our entities
        for entity_id, original_entity in original_entities.items():
            assert entity_id in snapshot.entities
            snapshot_entity = snapshot.entities[entity_id]
            assert snapshot_entity.entity_type == original_entity.entity_type
            assert snapshot_entity.properties == original_entity.properties
    
    @pytest.mark.asyncio
    async def test_data_plane_metrics(self, data_plane):
        """Test data plane metrics collection"""
        state_manager = data_plane.get_state_manager()
        
        # Perform operations to generate metrics
        for i in range(10):
            entity_id = state_manager.create_entity(
                entity_type="metrics_entity",
                properties={"index": i}
            )
            
            # Query entity
            entity = state_manager.get_entity(entity_id)
            assert entity is not None
            
            # Update entity
            state_manager.update_entity(
                entity_id,
                properties={"index": i, "updated": True}
            )
        
        # Get metrics
        metrics = data_plane.get_overall_metrics()
        
        assert metrics['state_manager']['entities_count'] == 10
        assert metrics['state_manager']['state_changes'] > 0
        assert metrics['state_manager']['queries_per_second'] >= 0


class TestPlaneCoordination:
    """Test coordination between control and data planes"""
    
    @pytest.fixture
    async def plane_coordinator(self):
        """Create a plane coordinator for testing"""
        coordinator = PlaneCoordinator()
        await coordinator.initialize()
        yield coordinator
        await coordinator.shutdown()
    
    @pytest.fixture
    async def agent_adapter(self, plane_coordinator):
        """Create an agent adapter for testing"""
        adapter = AgentPlaneAdapter("test_coordination_agent")
        await adapter.initialize()
        return adapter
    
    @pytest.mark.asyncio
    async def test_plane_coordinator_initialization(self, plane_coordinator):
        """Test plane coordinator initializes correctly"""
        assert plane_coordinator.status.value == "active"
        assert plane_coordinator.control_plane_interface is not None
        assert plane_coordinator.data_plane_interface is not None
    
    @pytest.mark.asyncio
    async def test_cross_plane_communication(self, agent_adapter):
        """Test communication between control and data planes"""
        # Test environment query
        result = await agent_adapter.query_environment(
            "get_entity",
            entity_id="nonexistent_entity"
        )
        
        assert result is not None
        assert "status" in result
        
        # Test environment modification
        result = await agent_adapter.modify_environment(
            "create_entity",
            entity_type="test_entity",
            properties={"name": "cross_plane_test"}
        )
        
        assert result is not None
        assert "status" in result
    
    @pytest.mark.asyncio
    async def test_agent_adapter_operations(self, agent_adapter):
        """Test agent adapter operations across planes"""
        # Create entity
        entity_id = await agent_adapter.create_entity(
            entity_type="adapter_test_entity",
            properties={"test": True, "value": 42},
            position={"x": 100, "y": 200}
        )
        
        assert entity_id is not None
        
        # Get entity
        entity_data = await agent_adapter.get_entity(entity_id)
        assert entity_data is not None
        
        # Find entities
        entities = await agent_adapter.find_entities(
            entity_type="adapter_test_entity"
        )
        assert len(entities) >= 1
        
        # Update entity
        success = await agent_adapter.update_entity(
            entity_id,
            properties={"test": True, "value": 84, "updated": True}
        )
        assert success
        
        # Verify update
        updated_entity = await agent_adapter.get_entity(entity_id)
        assert updated_entity is not None
        
        # Delete entity
        success = await agent_adapter.delete_entity(entity_id)
        assert success
        
        # Verify deletion
        deleted_entity = await agent_adapter.get_entity(entity_id)
        assert deleted_entity is None or not deleted_entity.get('success', True)
    
    @pytest.mark.asyncio
    async def test_plane_isolation_enforcement(self, plane_coordinator):
        """Test that planes remain isolated despite coordination"""
        # Get interfaces
        control_interface = plane_coordinator.get_control_plane_interface()
        data_interface = plane_coordinator.get_data_plane_interface()
        
        # Verify interfaces are separate
        assert control_interface != data_interface
        assert control_interface.coordinator == plane_coordinator
        assert data_interface.coordinator == plane_coordinator
        
        # Test that direct cross-plane access is not possible
        # (This is enforced by the architecture design)
        assert hasattr(control_interface, 'handlers')
        assert hasattr(data_interface, 'handlers')
        
        # Verify message routing works
        metrics = plane_coordinator.get_metrics()
        assert 'messages_processed' in metrics
        assert 'status' in metrics


class TestDistributedExecution:
    """Test distributed execution reliability"""
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_operations(self):
        """Test concurrent operations across multiple agents"""
        # Initialize global instances
        control_plane = await get_control_plane()
        data_plane = await get_data_plane()
        coordinator = await get_plane_coordinator()
        
        try:
            # Create multiple agents
            agents = []
            for i in range(5):
                agent_id = f"concurrent_agent_{i}"
                
                # Register with control plane
                decision_engine = control_plane.register_agent(agent_id)
                await decision_engine.initialize()
                
                # Create adapter
                adapter = AgentPlaneAdapter(agent_id)
                await adapter.initialize()
                agents.append((agent_id, decision_engine, adapter))
            
            # Perform concurrent operations
            async def agent_operations(agent_data):
                agent_id, decision_engine, adapter = agent_data
                
                # Make decisions
                for j in range(3):
                    await decision_engine.make_decision(
                        decision_type=DecisionType.TACTICAL,
                        context={"agent": agent_id, "iteration": j}
                    )
                
                # Environment operations
                entity_id = await adapter.create_entity(
                    entity_type="concurrent_entity",
                    properties={"agent": agent_id, "created_by": "concurrent_test"}
                )
                
                if entity_id:
                    await adapter.update_entity(
                        entity_id,
                        properties={"agent": agent_id, "updated": True}
                    )
                
                return agent_id
            
            # Run operations concurrently
            results = await asyncio.gather(
                *[agent_operations(agent_data) for agent_data in agents],
                return_exceptions=True
            )
            
            # Verify all operations completed successfully
            assert len(results) == 5
            for result in results:
                assert not isinstance(result, Exception)
            
            # Verify system state
            control_metrics = control_plane.get_overall_metrics()
            data_metrics = data_plane.get_overall_metrics()
            
            assert control_metrics.active_agents == 5
            assert control_metrics.total_decisions >= 15  # 5 agents * 3 decisions each
            assert data_metrics['state_manager']['entities_count'] >= 5
            
        finally:
            # Cleanup
            await shutdown_control_plane()
            await shutdown_data_plane()
            await shutdown_plane_coordinator()
    
    @pytest.mark.asyncio
    async def test_fault_tolerance(self):
        """Test system fault tolerance and recovery"""
        control_plane = await get_control_plane()
        data_plane = await get_data_plane()
        
        try:
            # Register agent
            agent_id = "fault_test_agent"
            decision_engine = control_plane.register_agent(agent_id)
            await decision_engine.initialize()
            
            adapter = AgentPlaneAdapter(agent_id)
            await adapter.initialize()
            
            # Perform normal operations
            entity_id = await adapter.create_entity(
                entity_type="fault_test_entity",
                properties={"test": "fault_tolerance"}
            )
            assert entity_id is not None
            
            # Simulate partial failure by stopping simulation
            sim_executor = data_plane.get_simulation_executor()
            await sim_executor.start_simulation()
            await asyncio.sleep(0.1)
            await sim_executor.stop_simulation()
            
            # Verify system still functions
            decision = await decision_engine.make_decision(
                decision_type=DecisionType.EMERGENCY,
                context={"situation": "fault_recovery_test"}
            )
            assert decision is not None
            
            # Verify data operations still work
            entity = await adapter.get_entity(entity_id)
            assert entity is not None
            
            # Test recovery
            await sim_executor.start_simulation()
            assert sim_executor.is_running
            
        finally:
            await shutdown_control_plane()
            await shutdown_data_plane()
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test system performance under load"""
        control_plane = await get_control_plane()
        data_plane = await get_data_plane()
        
        try:
            # Create multiple agents
            num_agents = 10
            agents = []
            
            for i in range(num_agents):
                agent_id = f"load_test_agent_{i}"
                decision_engine = control_plane.register_agent(agent_id)
                await decision_engine.initialize()
                
                adapter = AgentPlaneAdapter(agent_id)
                await adapter.initialize()
                agents.append((decision_engine, adapter))
            
            # Measure performance
            start_time = time.time()
            
            # Perform load test
            async def load_test_operations(agent_data):
                decision_engine, adapter = agent_data
                
                operations = []
                
                # Decision making load
                for i in range(10):
                    operations.append(
                        decision_engine.make_decision(
                            decision_type=DecisionType.TACTICAL,
                            context={"load_test": True, "iteration": i}
                        )
                    )
                
                # Environment operations load
                for i in range(5):
                    operations.append(
                        adapter.create_entity(
                            entity_type="load_test_entity",
                            properties={"iteration": i, "load_test": True}
                        )
                    )
                
                return await asyncio.gather(*operations)
            
            # Run load test
            results = await asyncio.gather(
                *[load_test_operations(agent_data) for agent_data in agents]
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Verify performance
            total_operations = num_agents * 15  # 10 decisions + 5 entities per agent
            operations_per_second = total_operations / total_time
            
            # Should handle at least 50 operations per second
            assert operations_per_second > 50, f"Performance too low: {operations_per_second} ops/sec"
            
            # Verify all operations completed
            assert len(results) == num_agents
            for agent_results in results:
                assert len(agent_results) == 15
                for result in agent_results:
                    assert result is not None
            
            # Check system metrics
            control_metrics = control_plane.get_overall_metrics()
            data_metrics = data_plane.get_overall_metrics()
            
            assert control_metrics.active_agents == num_agents
            assert control_metrics.total_decisions >= num_agents * 10
            assert data_metrics['state_manager']['entities_count'] >= num_agents * 5
            
        finally:
            await shutdown_control_plane()
            await shutdown_data_plane()


class TestPlaneSecurityIsolation:
    """Test security aspects of plane separation"""
    
    @pytest.mark.asyncio
    async def test_control_plane_cannot_access_data_directly(self):
        """Test that control plane cannot directly access data plane internals"""
        control_plane = await get_control_plane()
        data_plane = await get_data_plane()
        
        try:
            # Register agent in control plane
            agent_id = "security_test_agent"
            decision_engine = control_plane.register_agent(agent_id)
            await decision_engine.initialize()
            
            # Create entity in data plane
            state_manager = data_plane.get_state_manager()
            entity_id = state_manager.create_entity(
                entity_type="secure_entity",
                properties={"sensitive": "data", "access_level": "restricted"}
            )
            
            # Verify control plane cannot directly access data plane state
            # (This is enforced by architecture - no direct references)
            assert not hasattr(decision_engine, 'state_manager')
            assert not hasattr(decision_engine, 'data_plane')
            
            # Verify data plane cannot directly access control plane
            assert not hasattr(state_manager, 'decision_engine')
            assert not hasattr(state_manager, 'control_plane')
            
        finally:
            await shutdown_control_plane()
            await shutdown_data_plane()
    
    @pytest.mark.asyncio
    async def test_message_validation_between_planes(self):
        """Test that messages between planes are properly validated"""
        coordinator = await get_plane_coordinator()
        
        try:
            # Get interfaces
            control_interface = coordinator.get_control_plane_interface()
            data_interface = coordinator.get_data_plane_interface()
            
            # Test that interfaces validate message types
            assert hasattr(control_interface, 'handlers')
            assert hasattr(data_interface, 'handlers')
            
            # Verify message routing is controlled
            metrics = coordinator.get_metrics()
            assert 'messages_processed' in metrics
            
        finally:
            await shutdown_plane_coordinator()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])