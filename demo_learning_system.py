#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Learning System Demo
Demonstrates adversarial self-play and learning capabilities
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

from agents.learning_system import (
    LearningSystem, LearningConfig, PerformanceMetrics, 
    create_learning_system
)
from agents.self_play_coordinator import (
    SelfPlayCoordinator, SelfPlayConfig, 
    create_self_play_coordinator
)
from agents.experience_replay import (
    ExperienceReplaySystem, ReplayConfig, ReplayMode, KnowledgeType,
    create_experience_replay_system
)
from agents.base_agent import Experience, Team, Role, ActionResult, ActionPlan
from memory.memory_manager import create_memory_manager
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockAgent:
    """Mock agent for demonstration purposes"""
    
    def __init__(self, agent_id: str, team: Team, role: Role, name: str):
        self.agent_id = agent_id
        self.team = team
        self.role = role
        self.name = name
        self.performance_history = []
    
    def __repr__(self):
        return f"<MockAgent(id={self.agent_id}, team={self.team.value}, role={self.role.value})>"

def create_mock_experience(agent_id: str, success: bool = True, confidence: float = 0.7) -> Experience:
    """Create a mock experience for demonstration"""
    return Experience(
        experience_id=str(uuid.uuid4()),
        agent_id=agent_id,
        timestamp=datetime.now(),
        context=None,  # Would be EnvironmentState in real implementation
        action_taken=None,  # Would be ActionPlan in real implementation
        reasoning=None,  # Would be ReasoningResult in real implementation
        outcome=None,  # Would be ActionResult in real implementation
        success=success,
        lessons_learned=[
            f"Lesson from {'successful' if success else 'failed'} action",
            f"Agent {agent_id} learned tactical insight"
        ] if success or confidence < 0.5 else [],
        mitre_attack_mapping=[
            "T1001",  # Data Obfuscation
            "T1055"   # Process Injection
        ] if success else ["T1078"],  # Valid Accounts
        confidence_score=confidence
    )

async def demo_learning_system():
    """Demonstrate the learning system functionality"""
    print("\n" + "="*60)
    print("ARCHANGEL LEARNING SYSTEM DEMO")
    print("="*60)
    
    try:
        # Initialize memory manager
        print("\n1. Initializing Memory Manager...")
        memory_manager = create_memory_manager()
        await memory_manager.initialize()
        print("✓ Memory manager initialized")
        
        # Initialize learning system
        print("\n2. Initializing Learning System...")
        learning_config = LearningConfig(
            learning_rate=0.01,
            self_play_episodes=20,  # Reduced for demo
            replay_buffer_size=1000,
            batch_size=16
        )
        
        learning_system = create_learning_system(learning_config, memory_manager)
        await learning_system.initialize()
        print("✓ Learning system initialized")
        
        # Add sample experiences to replay buffer
        print("\n3. Adding Sample Experiences...")
        agents = ["red_recon", "red_exploit", "blue_analyst", "blue_firewall"]
        
        for i in range(50):
            agent_id = agents[i % len(agents)]
            success = (i % 3) != 0  # 2/3 success rate
            confidence = 0.4 + (i % 6) * 0.1  # Varying confidence
            
            experience = create_mock_experience(agent_id, success, confidence)
            await learning_system.add_experience_to_replay_buffer(experience)
        
        print(f"✓ Added 50 experiences to replay buffer")
        print(f"  Buffer size: {learning_system.replay_buffer.size()}")
        print(f"  Current exploration rate: {learning_system.current_exploration_rate:.3f}")
        
        # Demonstrate learning batch sampling
        print("\n4. Sampling Learning Batches...")
        batch = await learning_system.get_learning_batch(10)
        print(f"✓ Sampled batch of {len(batch)} experiences")
        
        # Calculate agent performance
        print("\n5. Calculating Agent Performance...")
        for agent_id in agents:
            performance = await learning_system._calculate_agent_performance(agent_id)
            print(f"  {agent_id}:")
            print(f"    Success Rate: {performance.success_rate:.2f}")
            print(f"    Confidence: {performance.confidence_score:.2f}")
            print(f"    Improvement Rate: {performance.improvement_rate:.2f}")
        
        # Get learning statistics
        print("\n6. Learning System Statistics:")
        stats = await learning_system.get_learning_statistics()
        
        learning_stats = stats["learning_system"]
        print(f"  Learning Iterations: {learning_stats['learning_iteration']}")
        print(f"  Exploration Rate: {learning_stats['current_exploration_rate']:.3f}")
        print(f"  Replay Buffer Size: {learning_stats['replay_buffer_size']}")
        
        performance_stats = stats["performance_tracking"]
        print(f"  Agents Tracked: {performance_stats['agents_tracked']}")
        print(f"  Performance Records: {performance_stats['total_performance_records']}")
        
        strategy_stats = stats["strategy_evolution"]
        print(f"  Strategy Generation: {strategy_stats['generation']}")
        print(f"  Strategies Tracked: {strategy_stats['strategies_tracked']}")
        
        print("\n✓ Learning system demo completed successfully!")
        
        await learning_system.shutdown()
        await memory_manager.shutdown()
        
    except Exception as e:
        logger.error(f"Error in learning system demo: {e}")
        raise

async def demo_self_play_coordinator():
    """Demonstrate the self-play coordinator functionality"""
    print("\n" + "="*60)
    print("SELF-PLAY COORDINATOR DEMO")
    print("="*60)
    
    try:
        # Initialize learning system
        print("\n1. Initializing Learning System...")
        memory_manager = create_memory_manager()
        await memory_manager.initialize()
        
        learning_system = create_learning_system(memory_manager=memory_manager)
        await learning_system.initialize()
        print("✓ Learning system initialized")
        
        # Initialize self-play coordinator
        print("\n2. Initializing Self-Play Coordinator...")
        coordinator_config = SelfPlayConfig(
            default_session_duration=timedelta(seconds=10),  # Short for demo
            max_concurrent_sessions=2,
            auto_matchmaking=False  # Manual control for demo
        )
        
        coordinator = create_self_play_coordinator(coordinator_config, learning_system)
        await coordinator.initialize()
        print("✓ Self-play coordinator initialized")
        
        # Create mock agents
        print("\n3. Creating Mock Agents...")
        mock_agents = [
            MockAgent("red_recon_1", Team.RED, Role.RECON, "Red Recon Agent 1"),
            MockAgent("red_exploit_1", Team.RED, Role.EXPLOIT, "Red Exploit Agent 1"),
            MockAgent("blue_analyst_1", Team.BLUE, Role.SOC_ANALYST, "Blue SOC Analyst 1"),
            MockAgent("blue_firewall_1", Team.BLUE, Role.FIREWALL_CONFIG, "Blue Firewall Agent 1"),
        ]
        
        # Register agents
        for agent in mock_agents:
            await coordinator.register_agent(agent)
            print(f"  ✓ Registered {agent.name}")
        
        # Get coordinator status
        print("\n4. Coordinator Status:")
        status = await coordinator.get_coordinator_status()
        
        coord_info = status["coordinator"]
        print(f"  Registered Agents: {coord_info['registered_agents']}")
        print(f"  Available Agents: {coord_info['available_agents']}")
        print(f"  Active Sessions: {coord_info['active_sessions']}")
        print(f"  Auto Matchmaking: {coord_info['config']['auto_matchmaking']}")
        
        # Create manual matchup
        print("\n5. Creating Manual Matchup...")
        red_agents = [agent.agent_id for agent in mock_agents if agent.team == Team.RED]
        blue_agents = [agent.agent_id for agent in mock_agents if agent.team == Team.BLUE]
        
        session_id = await coordinator.create_manual_matchup(
            red_agents[:1],  # One red agent
            blue_agents[:1], # One blue agent
            duration=timedelta(seconds=5)
        )
        
        print(f"✓ Created session: {session_id}")
        print(f"  Red Team: {red_agents[:1]}")
        print(f"  Blue Team: {blue_agents[:1]}")
        
        # Wait for session to complete
        print("\n6. Waiting for session completion...")
        await asyncio.sleep(7)  # Wait for session to finish
        
        # Check final status
        final_status = await coordinator.get_coordinator_status()
        print(f"✓ Session completed")
        print(f"  Total Sessions: {final_status['statistics']['total_sessions']}")
        print(f"  Successful Sessions: {final_status['statistics']['successful_sessions']}")
        
        print("\n✓ Self-play coordinator demo completed successfully!")
        
        await coordinator.shutdown()
        await learning_system.shutdown()
        await memory_manager.shutdown()
        
    except Exception as e:
        logger.error(f"Error in self-play coordinator demo: {e}")
        raise

async def demo_experience_replay():
    """Demonstrate the experience replay system functionality"""
    print("\n" + "="*60)
    print("EXPERIENCE REPLAY SYSTEM DEMO")
    print("="*60)
    
    try:
        # Initialize memory manager
        print("\n1. Initializing Memory Manager...")
        memory_manager = create_memory_manager()
        await memory_manager.initialize()
        print("✓ Memory manager initialized")
        
        # Initialize experience replay system
        print("\n2. Initializing Experience Replay System...")
        replay_config = ReplayConfig(
            buffer_size=200,
            batch_size=8,
            replay_frequency=2,  # Fast for demo
            min_replay_size=10
        )
        
        replay_system = create_experience_replay_system(replay_config, memory_manager)
        await replay_system.initialize()
        print("✓ Experience replay system initialized")
        
        # Add diverse experiences
        print("\n3. Adding Diverse Experiences...")
        agents = ["red_recon", "red_exploit", "blue_analyst", "blue_firewall"]
        
        for i in range(30):
            agent_id = agents[i % len(agents)]
            
            # Create varied experiences
            success = (i % 4) != 3  # 3/4 success rate
            confidence = 0.3 + (i % 7) * 0.1  # 0.3 to 0.9
            
            experience = create_mock_experience(agent_id, success, confidence)
            
            # Add some with custom priority
            if i % 5 == 0:  # Every 5th experience gets high priority
                await replay_system.add_experience(experience, priority=0.9)
            else:
                await replay_system.add_experience(experience)
        
        print(f"✓ Added 30 experiences to replay buffer")
        print(f"  Buffer size: {replay_system.replay_buffer.size()}")
        
        # Demonstrate different sampling modes
        print("\n4. Demonstrating Sampling Modes...")
        
        # Prioritized sampling
        prioritized_batch = await replay_system.sample_replay_batch(
            batch_size=5,
            replay_mode=ReplayMode.PRIORITIZED
        )
        print(f"  ✓ Prioritized batch: {len(prioritized_batch.experiences)} experiences")
        
        # Random sampling
        random_batch = await replay_system.sample_replay_batch(
            batch_size=5,
            replay_mode=ReplayMode.RANDOM
        )
        print(f"  ✓ Random batch: {len(random_batch.experiences)} experiences")
        
        # Temporal sampling
        temporal_batch = await replay_system.sample_replay_batch(
            batch_size=5,
            replay_mode=ReplayMode.TEMPORAL
        )
        print(f"  ✓ Temporal batch: {len(temporal_batch.experiences)} experiences")
        
        # Strategic sampling
        strategic_batch = await replay_system.sample_replay_batch(
            batch_size=5,
            replay_mode=ReplayMode.STRATEGIC,
            learning_objectives=["improve_reconnaissance", "enhance_defense"]
        )
        print(f"  ✓ Strategic batch: {len(strategic_batch.experiences)} experiences")
        
        # Process a batch
        print("\n5. Processing Replay Batch...")
        results = await replay_system.process_replay_batch(prioritized_batch)
        
        if results['success']:
            print(f"✓ Batch processed successfully")
            print(f"  Learning Outcomes: {len(results['learning_outcomes'])}")
            print(f"  Knowledge Extracted: {len(results['knowledge_extracted'])}")
            print(f"  Pattern Discoveries: {len(results['pattern_discoveries'])}")
            print(f"  Processing Time: {results['processing_time']:.3f}s")
        
        # Demonstrate knowledge distillation
        print("\n6. Demonstrating Knowledge Distillation...")
        experiences_for_distillation = []
        for i in range(min(10, replay_system.replay_buffer.size())):
            experiences_for_distillation.append(replay_system.replay_buffer.buffer[i])
        
        knowledge = await replay_system.distill_tactical_knowledge(
            experiences_for_distillation,
            KnowledgeType.TACTICAL
        )
        
        if knowledge:
            print(f"✓ Knowledge distilled: {knowledge.knowledge_id}")
            print(f"  Knowledge Type: {knowledge.knowledge_type.value}")
            print(f"  Source Experiences: {len(knowledge.source_experiences)}")
            print(f"  Confidence Score: {knowledge.confidence_score:.3f}")
            print(f"  Pattern Count: {len(knowledge.distilled_patterns)}")
        else:
            print("  No significant patterns found for distillation")
        
        # Get replay statistics
        print("\n7. Experience Replay Statistics:")
        stats = await replay_system.get_replay_statistics()
        
        replay_stats = stats["replay_system"]
        print(f"  Buffer Size: {replay_stats['buffer_size']}/{replay_stats['max_buffer_size']}")
        print(f"  Distilled Knowledge: {replay_stats['distilled_knowledge_count']}")
        print(f"  Knowledge Patterns: {sum(replay_stats['knowledge_patterns'].values())}")
        
        system_stats = stats["statistics"]
        print(f"  Total Replays: {system_stats['total_replays']}")
        print(f"  Successful Replays: {system_stats['successful_replays']}")
        print(f"  Knowledge Extractions: {system_stats['knowledge_extractions']}")
        print(f"  Pattern Discoveries: {system_stats['pattern_discoveries']}")
        
        print("\n✓ Experience replay demo completed successfully!")
        
        await replay_system.shutdown()
        await memory_manager.shutdown()
        
    except Exception as e:
        logger.error(f"Error in experience replay demo: {e}")
        raise

async def demo_integrated_learning():
    """Demonstrate integrated learning system with all components"""
    print("\n" + "="*60)
    print("INTEGRATED LEARNING SYSTEM DEMO")
    print("="*60)
    
    try:
        # Initialize all systems
        print("\n1. Initializing Integrated Learning Systems...")
        
        memory_manager = create_memory_manager()
        await memory_manager.initialize()
        
        learning_system = create_learning_system(memory_manager=memory_manager)
        await learning_system.initialize()
        
        replay_system = create_experience_replay_system(memory_manager=memory_manager)
        await replay_system.initialize()
        
        coordinator = create_self_play_coordinator(
            SelfPlayConfig(
                default_session_duration=timedelta(seconds=8),
                auto_matchmaking=False
            ),
            learning_system
        )
        await coordinator.initialize()
        
        print("✓ All systems initialized")
        
        # Create and register agents
        print("\n2. Setting up Agent Environment...")
        agents = [
            MockAgent("red_recon", Team.RED, Role.RECON, "Red Recon Specialist"),
            MockAgent("red_exploit", Team.RED, Role.EXPLOIT, "Red Exploit Specialist"),
            MockAgent("blue_soc", Team.BLUE, Role.SOC_ANALYST, "Blue SOC Analyst"),
            MockAgent("blue_firewall", Team.BLUE, Role.FIREWALL_CONFIG, "Blue Firewall Admin"),
        ]
        
        for agent in agents:
            await coordinator.register_agent(agent)
        
        print(f"✓ Registered {len(agents)} agents")
        
        # Generate learning experiences
        print("\n3. Generating Learning Experiences...")
        for i in range(40):
            agent = agents[i % len(agents)]
            
            # Simulate varied performance
            success_rate = 0.6 if agent.team == Team.RED else 0.7
            success = (i % 10) < (success_rate * 10)
            confidence = 0.4 + (i % 6) * 0.1
            
            experience = create_mock_experience(agent.agent_id, success, confidence)
            
            # Add to both systems
            await learning_system.add_experience_to_replay_buffer(experience)
            await replay_system.add_experience(experience)
        
        print("✓ Generated 40 learning experiences")
        
        # Start self-play session
        print("\n4. Running Self-Play Session...")
        session_id = await coordinator.create_manual_matchup(
            ["red_recon", "red_exploit"],
            ["blue_soc", "blue_firewall"],
            duration=timedelta(seconds=6)
        )
        
        print(f"✓ Started self-play session: {session_id}")
        
        # Process replay batches while session runs
        print("\n5. Processing Experience Replay...")
        batch = await replay_system.sample_replay_batch(
            batch_size=8,
            replay_mode=ReplayMode.PRIORITIZED
        )
        
        results = await replay_system.process_replay_batch(batch)
        print(f"✓ Processed replay batch with {len(results.get('learning_outcomes', []))} outcomes")
        
        # Wait for session completion
        await asyncio.sleep(8)
        
        # Analyze integrated results
        print("\n6. Integrated Learning Analysis:")
        
        # Learning system stats
        learning_stats = await learning_system.get_learning_statistics()
        print(f"  Learning Iterations: {learning_stats['learning_system']['learning_iteration']}")
        print(f"  Agents Tracked: {learning_stats['performance_tracking']['agents_tracked']}")
        
        # Replay system stats
        replay_stats = await replay_system.get_replay_statistics()
        print(f"  Replay Buffer: {replay_stats['replay_system']['buffer_size']}")
        print(f"  Knowledge Distilled: {replay_stats['replay_system']['distilled_knowledge_count']}")
        
        # Coordinator stats
        coord_stats = await coordinator.get_coordinator_status()
        print(f"  Total Sessions: {coord_stats['statistics']['total_sessions']}")
        print(f"  Learning Outcomes: {coord_stats['statistics']['total_learning_outcomes']}")
        
        # Calculate overall improvement
        total_experiences = learning_stats['learning_system']['replay_buffer_size']
        total_knowledge = replay_stats['replay_system']['distilled_knowledge_count']
        total_sessions = coord_stats['statistics']['total_sessions']
        
        improvement_score = (total_knowledge * 0.4 + total_sessions * 0.6) / max(1, total_experiences / 10)
        
        print(f"\n7. Overall Learning Effectiveness:")
        print(f"  Total Experiences: {total_experiences}")
        print(f"  Knowledge Distilled: {total_knowledge}")
        print(f"  Self-Play Sessions: {total_sessions}")
        print(f"  Learning Effectiveness Score: {improvement_score:.2f}")
        
        if improvement_score > 0.5:
            print("  ✓ EXCELLENT - System is learning effectively!")
        elif improvement_score > 0.3:
            print("  ✓ GOOD - System shows solid learning progress")
        else:
            print("  ⚠ MODERATE - System learning could be improved")
        
        print("\n✓ Integrated learning demo completed successfully!")
        
        # Cleanup
        await coordinator.shutdown()
        await replay_system.shutdown()
        await learning_system.shutdown()
        await memory_manager.shutdown()
        
    except Exception as e:
        logger.error(f"Error in integrated learning demo: {e}")
        raise

async def main():
    """Run all learning system demonstrations"""
    print("ARCHANGEL AUTONOMOUS AI EVOLUTION")
    print("Learning System Comprehensive Demo")
    print("=" * 80)
    
    try:
        # Run individual component demos
        await demo_learning_system()
        await asyncio.sleep(1)
        
        await demo_self_play_coordinator()
        await asyncio.sleep(1)
        
        await demo_experience_replay()
        await asyncio.sleep(1)
        
        # Run integrated demo
        await demo_integrated_learning()
        
        print("\n" + "="*80)
        print("ALL LEARNING SYSTEM DEMOS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nKey Features Demonstrated:")
        print("✓ Adversarial Self-Play Learning")
        print("✓ Experience Replay and Knowledge Distillation")
        print("✓ Strategy Evolution and Performance Tracking")
        print("✓ Multi-Agent Coordination and Competition")
        print("✓ Prioritized Experience Sampling")
        print("✓ Tactical Knowledge Extraction")
        print("✓ Integrated Learning Pipeline")
        
        print("\nThe learning system is ready for autonomous agent training!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)