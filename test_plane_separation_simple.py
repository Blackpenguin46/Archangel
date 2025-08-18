"""
Simple test runner for plane separation implementation

This script provides a quick way to test the basic functionality
of the control and data plane separation without running the full test suite.
"""

import asyncio
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_basic_functionality():
    """Test basic functionality of plane separation"""
    logger.info("Starting basic plane separation test...")
    
    try:
        # Import modules
        from agents.control_plane import (
            ControlPlaneOrchestrator,
            DecisionType,
            get_control_plane,
            shutdown_control_plane
        )
        from agents.data_plane import (
            DataPlaneOrchestrator,
            get_data_plane,
            shutdown_data_plane
        )
        from agents.plane_coordinator import (
            PlaneCoordinator,
            AgentPlaneAdapter,
            get_plane_coordinator,
            shutdown_plane_coordinator
        )
        
        logger.info("✓ All modules imported successfully")
        
        # Test control plane
        logger.info("Testing control plane...")
        control_plane = await get_control_plane()
        
        # Register test agent
        agent_id = "test_agent"
        decision_engine = control_plane.register_agent(agent_id)
        await decision_engine.initialize()
        
        # Make a decision
        decision = await decision_engine.make_decision(
            decision_type=DecisionType.TACTICAL,
            context={"test": "basic_functionality"}
        )
        
        assert decision is not None
        assert decision.agent_id == agent_id
        logger.info("✓ Control plane basic functionality working")
        
        # Test data plane
        logger.info("Testing data plane...")
        data_plane = await get_data_plane()
        
        state_manager = data_plane.get_state_manager()
        
        # Create test entity
        entity_id = state_manager.create_entity(
            entity_type="test_entity",
            properties={"test": True, "created_at": datetime.now().isoformat()}
        )
        
        # Retrieve entity
        entity = state_manager.get_entity(entity_id)
        assert entity is not None
        assert entity.entity_type == "test_entity"
        logger.info("✓ Data plane basic functionality working")
        
        # Test plane coordination
        logger.info("Testing plane coordination...")
        coordinator = await get_plane_coordinator()
        
        # Create agent adapter
        adapter = AgentPlaneAdapter("coordination_test_agent")
        await adapter.initialize()
        
        # Test cross-plane operation
        new_entity_id = await adapter.create_entity(
            entity_type="coordination_test",
            properties={"coordinated": True}
        )
        
        assert new_entity_id is not None
        logger.info("✓ Plane coordination basic functionality working")
        
        # Test metrics
        logger.info("Testing metrics collection...")
        
        control_metrics = control_plane.get_overall_metrics()
        data_metrics = data_plane.get_overall_metrics()
        coord_metrics = coordinator.get_metrics()
        
        assert control_metrics.active_agents > 0
        assert data_metrics['state_manager']['entities_count'] > 0
        assert 'messages_processed' in coord_metrics
        
        logger.info("✓ Metrics collection working")
        
        # Cleanup
        logger.info("Cleaning up...")
        await shutdown_control_plane()
        await shutdown_data_plane()
        await shutdown_plane_coordinator()
        
        logger.info("✓ All tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_performance():
    """Test basic performance characteristics"""
    logger.info("Testing performance characteristics...")
    
    try:
        from agents.control_plane import get_control_plane, shutdown_control_plane, DecisionType
        from agents.data_plane import get_data_plane, shutdown_data_plane
        import time
        
        # Initialize
        control_plane = await get_control_plane()
        data_plane = await get_data_plane()
        
        # Performance test: decision making
        start_time = time.time()
        
        agent_id = "perf_test_agent"
        decision_engine = control_plane.register_agent(agent_id)
        await decision_engine.initialize()
        
        # Make multiple decisions
        for i in range(10):
            await decision_engine.make_decision(
                decision_type=DecisionType.TACTICAL,
                context={"iteration": i, "performance_test": True}
            )
        
        decision_time = time.time() - start_time
        logger.info(f"✓ 10 decisions completed in {decision_time:.3f}s ({10/decision_time:.1f} decisions/sec)")
        
        # Performance test: entity operations
        start_time = time.time()
        
        state_manager = data_plane.get_state_manager()
        
        # Create multiple entities
        for i in range(50):
            state_manager.create_entity(
                entity_type="perf_test_entity",
                properties={"iteration": i, "performance_test": True}
            )
        
        entity_time = time.time() - start_time
        logger.info(f"✓ 50 entities created in {entity_time:.3f}s ({50/entity_time:.1f} entities/sec)")
        
        # Cleanup
        await shutdown_control_plane()
        await shutdown_data_plane()
        
        logger.info("✓ Performance test completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Performance test failed: {e}")
        return False


async def test_isolation():
    """Test that planes are properly isolated"""
    logger.info("Testing plane isolation...")
    
    try:
        from agents.control_plane import get_control_plane, shutdown_control_plane
        from agents.data_plane import get_data_plane, shutdown_data_plane
        
        # Initialize planes
        control_plane = await get_control_plane()
        data_plane = await get_data_plane()
        
        # Register agent in control plane
        agent_id = "isolation_test_agent"
        decision_engine = control_plane.register_agent(agent_id)
        await decision_engine.initialize()
        
        # Create entity in data plane
        state_manager = data_plane.get_state_manager()
        entity_id = state_manager.create_entity(
            entity_type="isolation_test_entity",
            properties={"isolated": True}
        )
        
        # Verify isolation: control plane should not have direct access to data plane
        assert not hasattr(decision_engine, 'state_manager')
        assert not hasattr(decision_engine, 'data_plane')
        
        # Verify isolation: data plane should not have direct access to control plane
        assert not hasattr(state_manager, 'decision_engine')
        assert not hasattr(state_manager, 'control_plane')
        
        logger.info("✓ Plane isolation verified")
        
        # Cleanup
        await shutdown_control_plane()
        await shutdown_data_plane()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Isolation test failed: {e}")
        return False


async def run_all_tests():
    """Run all simple tests"""
    logger.info("Running plane separation simple tests...")
    logger.info("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Performance", test_performance),
        ("Isolation", test_isolation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST RESULTS SUMMARY:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed!")
        return False


def main():
    """Main function"""
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()