#!/usr/bin/env python3
"""
Integration test for scoring engine with coordinator
"""

import asyncio
import logging
from datetime import datetime, timedelta

from agents.coordinator import LangGraphCoordinator, Scenario, Phase
from agents.communication import MessageBus
from agents.base_agent import BaseAgent, AgentConfig, Team, Role

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_scoring_integration():
    """Test scoring engine integration with coordinator"""
    logger.info("🧪 Testing Scoring Engine Integration")
    
    # Initialize message bus
    message_bus = MessageBus()
    await message_bus.initialize()
    await message_bus.start_message_processing()
    
    # Initialize coordinator with scoring
    coordinator = LangGraphCoordinator(message_bus)
    await coordinator.initialize()
    
    try:
        # Create test agents
        red_config = AgentConfig(
            agent_id="red_test_1",
            team=Team.RED,
            role=Role.RECON,
            name="Red Recon Agent",
            description="Test red team agent"
        )
        
        blue_config = AgentConfig(
            agent_id="blue_test_1",
            team=Team.BLUE,
            role=Role.SOC_ANALYST,
            name="Blue SOC Agent",
            description="Test blue team agent"
        )
        
        # Create mock agents (simplified for testing)
        class MockRedAgent(BaseAgent):
            async def execute_action(self, action):
                from agents.base_agent import ActionResult
                return ActionResult(
                    action_id="test_action",
                    action_type="reconnaissance",
                    success=True,
                    outcome="Network scan completed",
                    data={"targets_found": 5},
                    duration=45.0,
                    errors=[],
                    side_effects=[],
                    confidence=0.8,
                    timestamp=datetime.now()
                )
        
        class MockBlueAgent(BaseAgent):
            async def execute_action(self, action):
                from agents.base_agent import ActionResult
                return ActionResult(
                    action_id="test_action",
                    action_type="detection",
                    success=True,
                    outcome="Threat detected",
                    data={"threat_type": "reconnaissance"},
                    duration=30.0,
                    errors=[],
                    side_effects=[],
                    confidence=0.9,
                    timestamp=datetime.now()
                )
        
        red_agent = MockRedAgent(red_config)
        blue_agent = MockBlueAgent(blue_config)
        
        await red_agent.initialize()
        await blue_agent.initialize()
        
        # Register agents with coordinator
        await coordinator.register_agent(red_agent, ["reconnaissance", "scanning"])
        await coordinator.register_agent(blue_agent, ["detection", "monitoring"])
        
        logger.info("✅ Agents registered successfully")
        
        # Test scoring integration
        logger.info("🎯 Testing scoring integration...")
        
        # Record Red team attack
        await coordinator.record_agent_action(
            agent_id="red_test_1",
            action_type="reconnaissance",
            success=True,
            context={
                "target": "corporate_network",
                "duration": 45.0,
                "stealth_score": 0.8
            }
        )
        
        # Record Blue team detection
        await coordinator.record_agent_action(
            agent_id="blue_test_1",
            action_type="detection",
            success=True,
            context={
                "detected_agent": "red_test_1",
                "detection_time": 60.0,
                "accuracy": 0.9
            }
        )
        
        # Record collaboration
        await coordinator.record_agent_action(
            agent_id="red_test_1",
            action_type="collaboration",
            success=True,
            context={
                "collaboration_type": "intelligence_sharing",
                "effectiveness": 0.8
            }
        )
        
        logger.info("✅ Actions recorded successfully")
        
        # Get current scores
        scores = await coordinator.get_current_scores()
        logger.info("📊 Current Scores:")
        for team, score_data in scores.items():
            logger.info(f"   {team.upper()}: {score_data['total_score']:.2f} points")
        
        # Get performance analysis
        analysis = await coordinator.get_performance_analysis()
        logger.info("📈 Performance Analysis:")
        logger.info(f"   Leading Team: {analysis['comparative_analysis']['leading_team']}")
        logger.info(f"   Red Success Rate: {analysis['performance_metrics']['red_team_success_rate']:.1%}")
        logger.info(f"   Blue Success Rate: {analysis['performance_metrics']['blue_team_success_rate']:.1%}")
        
        # Test coordination status with scoring
        status = coordinator.get_coordination_status()
        logger.info("🔧 Coordination Status:")
        logger.info(f"   Scoring Enabled: {status['scoring_enabled']}")
        logger.info(f"   Registered Agents: {status['registered_agents']}")
        logger.info(f"   Red Team Agents: {status['team_assignments']['red']}")
        logger.info(f"   Blue Team Agents: {status['team_assignments']['blue']}")
        
        # Verify scoring is working
        assert scores["red"]["total_score"] > 0, "Red team should have score > 0"
        assert scores["blue"]["total_score"] > 0, "Blue team should have score > 0"
        assert status["scoring_enabled"] is True, "Scoring should be enabled"
        
        logger.info("🎉 Scoring integration test passed!")
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        raise
    finally:
        # Cleanup
        await coordinator.shutdown()
        await message_bus.shutdown()

async def test_scenario_with_scoring():
    """Test a complete scenario with scoring"""
    logger.info("\n🧪 Testing Complete Scenario with Scoring")
    
    # Initialize components
    message_bus = MessageBus()
    await message_bus.initialize()
    await message_bus.start_message_processing()
    
    coordinator = LangGraphCoordinator(message_bus)
    await coordinator.initialize()
    
    try:
        # Create scenario
        scenario = Scenario(
            scenario_id="test_scenario_1",
            name="Red vs Blue Cyber Exercise",
            description="Test scenario with scoring",
            duration=timedelta(minutes=10),
            phases=[Phase.RECONNAISSANCE, Phase.EXPLOITATION, Phase.DEFENSE],
            objectives={
                Team.RED: ["Gain initial access", "Establish persistence"],
                Team.BLUE: ["Detect intrusion", "Contain threats"]
            },
            constraints=["No real-world impact"],
            success_criteria={
                Team.RED: ["Successful exploitation", "Stealth maintenance"],
                Team.BLUE: ["Rapid detection", "Effective containment"]
            },
            environment_config={}
        )
        
        # Start scenario
        workflow_id = await coordinator.start_scenario(scenario)
        logger.info(f"✅ Scenario started: {workflow_id}")
        
        # Simulate scenario activities with scoring
        activities = [
            # Red team reconnaissance
            ("red_agent_1", "reconnaissance", True, {
                "target": "web_server", "duration": 30.0, "stealth_score": 0.9
            }),
            # Blue team detection
            ("blue_agent_1", "detection", True, {
                "detected_agent": "red_agent_1", "detection_time": 45.0, "accuracy": 0.85
            }),
            # Red team exploitation
            ("red_agent_2", "exploit", True, {
                "target": "database", "duration": 120.0, "stealth_score": 0.6
            }),
            # Blue team containment
            ("blue_agent_2", "containment", True, {
                "threat_id": "sql_injection", "containment_time": 180.0, "effectiveness": 0.9
            })
        ]
        
        for agent_id, action_type, success, context in activities:
            await coordinator.record_agent_action(agent_id, action_type, success, context)
            logger.info(f"   📝 Recorded {action_type} by {agent_id}")
        
        # Get final analysis
        analysis = await coordinator.get_performance_analysis()
        
        logger.info("🏆 Scenario Results:")
        logger.info(f"   Winner: {analysis['comparative_analysis']['leading_team'].upper()}")
        logger.info(f"   Score Difference: {analysis['comparative_analysis']['score_difference']:.2f}")
        logger.info(f"   Total Metrics: {analysis['performance_metrics']['total_metrics_recorded']}")
        
        # Verify scenario scoring
        assert analysis["performance_metrics"]["total_metrics_recorded"] > 0
        logger.info("🎉 Scenario with scoring test passed!")
        
    except Exception as e:
        logger.error(f"❌ Scenario test failed: {e}")
        raise
    finally:
        await coordinator.shutdown()
        await message_bus.shutdown()

async def run_integration_tests():
    """Run all integration tests"""
    logger.info("🚀 Starting Scoring Engine Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        test_scoring_integration,
        test_scenario_with_scoring
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            logger.error(f"❌ Test {test.__name__} failed: {e}")
            failed += 1
    
    logger.info("\n" + "=" * 60)
    logger.info(f"🏆 Integration Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All integration tests passed!")
        return True
    else:
        logger.error("❌ Some integration tests failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_integration_tests())
    exit(0 if success else 1)