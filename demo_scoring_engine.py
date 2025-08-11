#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Dynamic Scoring Engine Demo
Demonstrates the real-time scoring and evaluation capabilities
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any

from agents.scoring_engine import (
    DynamicScoringEngine, ScoringConfig, ScoringWeight, ScoreCategory,
    MetricType, DEFAULT_SCORING_CONFIG
)
from agents.base_agent import Team

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ScoringEngineDemo:
    """Demo class for the Dynamic Scoring Engine"""
    
    def __init__(self):
        self.scoring_engine = None
        self.demo_agents = {
            Team.RED: ["red_recon", "red_exploit", "red_persist"],
            Team.BLUE: ["blue_soc", "blue_firewall", "blue_siem"]
        }
    
    async def initialize(self):
        """Initialize the demo"""
        logger.info("🚀 Initializing Dynamic Scoring Engine Demo")
        
        # Create custom configuration for demo
        demo_config = ScoringConfig(
            weights={
                ScoreCategory.ATTACK_SUCCESS: ScoringWeight(
                    category=ScoreCategory.ATTACK_SUCCESS,
                    weight=0.25,
                    max_score=100.0,
                    description="Red team attack success and effectiveness"
                ),
                ScoreCategory.DEFENSE_SUCCESS: ScoringWeight(
                    category=ScoreCategory.DEFENSE_SUCCESS,
                    weight=0.25,
                    max_score=100.0,
                    description="Blue team defense success and effectiveness"
                ),
                ScoreCategory.DETECTION_SPEED: ScoringWeight(
                    category=ScoreCategory.DETECTION_SPEED,
                    weight=0.20,
                    max_score=100.0,
                    description="Speed of threat detection"
                ),
                ScoreCategory.CONTAINMENT_TIME: ScoringWeight(
                    category=ScoreCategory.CONTAINMENT_TIME,
                    weight=0.15,
                    max_score=100.0,
                    description="Time to contain threats"
                ),
                ScoreCategory.STEALTH_MAINTENANCE: ScoringWeight(
                    category=ScoreCategory.STEALTH_MAINTENANCE,
                    weight=0.10,
                    max_score=100.0,
                    description="Stealth and evasion capabilities"
                ),
                ScoreCategory.COLLABORATION: ScoringWeight(
                    category=ScoreCategory.COLLABORATION,
                    weight=0.05,
                    max_score=100.0,
                    description="Team coordination effectiveness"
                )
            },
            evaluation_window=timedelta(minutes=2),
            trend_history_size=50,
            real_time_updates=True,
            fairness_adjustments=True
        )
        
        # Initialize scoring engine
        self.scoring_engine = DynamicScoringEngine(demo_config)
        await self.scoring_engine.initialize()
        
        logger.info("✅ Scoring engine initialized successfully")
    
    async def demonstrate_basic_scoring(self):
        """Demonstrate basic scoring functionality"""
        logger.info("\n📊 === BASIC SCORING DEMONSTRATION ===")
        
        # Red team activities
        logger.info("🔴 Red Team Activities:")
        
        # Reconnaissance phase
        await self.scoring_engine.record_attack_success(
            agent_id="red_recon",
            target="corporate_network",
            attack_type="network_scanning",
            success=True,
            duration=45.0,
            stealth_score=0.8
        )
        logger.info("   ✓ Network reconnaissance completed (45s, stealth: 0.8)")
        
        # Exploitation phase
        await self.scoring_engine.record_attack_success(
            agent_id="red_exploit",
            target="web_application",
            attack_type="sql_injection",
            success=True,
            duration=120.0,
            stealth_score=0.6
        )
        logger.info("   ✓ SQL injection attack successful (120s, stealth: 0.6)")
        
        # Failed persistence attempt
        await self.scoring_engine.record_attack_success(
            agent_id="red_persist",
            target="domain_controller",
            attack_type="privilege_escalation",
            success=False,
            duration=300.0,
            stealth_score=0.3
        )
        logger.info("   ✗ Privilege escalation failed (300s, stealth: 0.3)")
        
        # Blue team responses
        logger.info("\n🔵 Blue Team Responses:")
        
        # Detection events
        await self.scoring_engine.record_detection_event(
            agent_id="blue_soc",
            detected_agent="red_recon",
            detection_time=60.0,
            accuracy=0.9
        )
        logger.info("   ✓ Network scanning detected (60s, accuracy: 0.9)")
        
        await self.scoring_engine.record_detection_event(
            agent_id="blue_siem",
            detected_agent="red_exploit",
            detection_time=180.0,
            accuracy=0.85
        )
        logger.info("   ✓ SQL injection detected (180s, accuracy: 0.85)")
        
        # Containment actions
        await self.scoring_engine.record_containment_action(
            agent_id="blue_firewall",
            threat_id="sql_injection_threat",
            containment_time=240.0,
            effectiveness=0.9
        )
        logger.info("   ✓ SQL injection contained (240s, effectiveness: 0.9)")
        
        # Display current scores
        await self.display_current_scores()
    
    async def demonstrate_collaboration_scoring(self):
        """Demonstrate collaboration and team coordination scoring"""
        logger.info("\n🤝 === COLLABORATION SCORING DEMONSTRATION ===")
        
        # Red team collaboration
        await self.scoring_engine.record_collaboration_event(
            agent_id="red_recon",
            team=Team.RED,
            collaboration_type="intelligence_sharing",
            effectiveness=0.85
        )
        logger.info("🔴 Red team intelligence sharing (effectiveness: 0.85)")
        
        await self.scoring_engine.record_collaboration_event(
            agent_id="red_exploit",
            team=Team.RED,
            collaboration_type="coordinated_attack",
            effectiveness=0.75
        )
        logger.info("🔴 Red team coordinated attack (effectiveness: 0.75)")
        
        # Blue team collaboration
        await self.scoring_engine.record_collaboration_event(
            agent_id="blue_soc",
            team=Team.BLUE,
            collaboration_type="incident_coordination",
            effectiveness=0.9
        )
        logger.info("🔵 Blue team incident coordination (effectiveness: 0.9)")
        
        await self.scoring_engine.record_collaboration_event(
            agent_id="blue_firewall",
            team=Team.BLUE,
            collaboration_type="automated_response",
            effectiveness=0.8
        )
        logger.info("🔵 Blue team automated response (effectiveness: 0.8)")
        
        # Display updated scores
        await self.display_current_scores()
    
    async def demonstrate_learning_adaptation(self):
        """Demonstrate learning and adaptation scoring"""
        logger.info("\n🧠 === LEARNING & ADAPTATION DEMONSTRATION ===")
        
        # Red team learning from failures
        await self.scoring_engine.record_learning_adaptation(
            agent_id="red_persist",
            team=Team.RED,
            adaptation_score=0.7,
            context={
                "learning_type": "failure_analysis",
                "failed_technique": "privilege_escalation",
                "new_strategy": "lateral_movement",
                "confidence_improvement": 0.3
            }
        )
        logger.info("🔴 Red team learned from privilege escalation failure")
        
        # Blue team improving detection
        await self.scoring_engine.record_learning_adaptation(
            agent_id="blue_siem",
            team=Team.BLUE,
            adaptation_score=0.8,
            context={
                "learning_type": "pattern_recognition",
                "improved_detection": "sql_injection_variants",
                "accuracy_improvement": 0.15
            }
        )
        logger.info("🔵 Blue team improved SQL injection detection patterns")
        
        # Display updated scores
        await self.display_current_scores()
    
    async def demonstrate_performance_analysis(self):
        """Demonstrate comprehensive performance analysis"""
        logger.info("\n📈 === PERFORMANCE ANALYSIS DEMONSTRATION ===")
        
        # Get comprehensive analysis
        analysis = await self.scoring_engine.get_performance_analysis()
        
        # Display team scores
        logger.info("\n🏆 Team Scores:")
        for team_name, team_data in analysis["team_scores"].items():
            logger.info(f"   {team_name.upper()} Team:")
            logger.info(f"     Total Score: {team_data['total_score']:.2f}")
            logger.info(f"     Trend: {team_data['trend_direction']}")
            logger.info(f"     Metrics Count: {team_data['metrics_count']}")
            
            logger.info("     Category Breakdown:")
            for category, score in team_data["category_scores"].items():
                logger.info(f"       {category}: {score:.2f}")
        
        # Display comparative analysis
        logger.info("\n⚖️  Comparative Analysis:")
        comp = analysis["comparative_analysis"]
        logger.info(f"   Leading Team: {comp['leading_team'].upper()}")
        logger.info(f"   Score Difference: {comp['score_difference']:.2f}")
        logger.info(f"   Performance Balance: {comp['performance_balance']:.2f}")
        
        # Display performance metrics
        logger.info("\n📊 Performance Metrics:")
        metrics = analysis["performance_metrics"]
        logger.info(f"   Average Detection Time: {metrics['average_detection_time']:.1f}s")
        logger.info(f"   Average Containment Time: {metrics['average_containment_time']:.1f}s")
        logger.info(f"   Red Team Success Rate: {metrics['red_team_success_rate']:.1%}")
        logger.info(f"   Blue Team Success Rate: {metrics['blue_team_success_rate']:.1%}")
        
        # Display recommendations
        logger.info("\n💡 Recommendations:")
        for rec in analysis["recommendations"]:
            logger.info(f"   {rec['team'].upper()}: {rec['recommendation']}")
        
        return analysis
    
    async def demonstrate_real_time_updates(self):
        """Demonstrate real-time scoring updates"""
        logger.info("\n⚡ === REAL-TIME UPDATES DEMONSTRATION ===")
        
        logger.info("Simulating ongoing Red vs Blue activities...")
        
        # Simulate ongoing activities with real-time updates
        activities = [
            ("red_recon", "port_scanning", True, 30.0, 0.9),
            ("blue_soc", "anomaly_detection", 45.0, 0.8),
            ("red_exploit", "buffer_overflow", False, 180.0, 0.4),
            ("blue_firewall", "traffic_blocking", 60.0, 0.95),
            ("red_persist", "backdoor_installation", True, 240.0, 0.7),
            ("blue_siem", "correlation_analysis", 120.0, 0.85)
        ]
        
        for i, activity in enumerate(activities):
            if activity[0].startswith("red"):
                # Red team activity
                await self.scoring_engine.record_attack_success(
                    agent_id=activity[0],
                    target=f"target_{i}",
                    attack_type=activity[1],
                    success=activity[2],
                    duration=activity[3],
                    stealth_score=activity[4]
                )
                logger.info(f"🔴 {activity[0]}: {activity[1]} ({'✓' if activity[2] else '✗'})")
            else:
                # Blue team activity
                if "detection" in activity[1]:
                    await self.scoring_engine.record_detection_event(
                        agent_id=activity[0],
                        detected_agent=f"red_agent_{i}",
                        detection_time=activity[2],
                        accuracy=activity[3]
                    )
                    logger.info(f"🔵 {activity[0]}: {activity[1]} ({activity[2]:.1f}s)")
                else:
                    await self.scoring_engine.record_containment_action(
                        agent_id=activity[0],
                        threat_id=f"threat_{i}",
                        containment_time=activity[2],
                        effectiveness=activity[3]
                    )
                    logger.info(f"🔵 {activity[0]}: {activity[1]} ({activity[2]:.1f}s)")
            
            # Brief pause to simulate real-time
            await asyncio.sleep(0.5)
        
        # Display final scores
        await self.display_current_scores()
    
    async def demonstrate_metrics_export(self):
        """Demonstrate metrics export functionality"""
        logger.info("\n💾 === METRICS EXPORT DEMONSTRATION ===")
        
        # Export metrics
        exported_data = await self.scoring_engine.export_metrics("json")
        
        # Parse and display summary
        data = json.loads(exported_data)
        
        logger.info(f"📤 Exported {len(data['metrics_history'])} metrics")
        logger.info(f"   Export timestamp: {data['export_timestamp']}")
        logger.info(f"   Engine ID: {data['engine_id']}")
        
        # Save to file for inspection
        filename = f"scoring_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            f.write(exported_data)
        
        logger.info(f"   Saved to: {filename}")
        
        # Display sample metrics
        logger.info("\n📋 Sample Metrics:")
        for i, metric in enumerate(data['metrics_history'][:3]):
            logger.info(f"   {i+1}. {metric['team']} - {metric['category']}: {metric['value']:.2f}")
    
    async def display_current_scores(self):
        """Display current team scores"""
        scores = await self.scoring_engine.get_current_scores()
        
        logger.info("\n🏆 Current Scores:")
        for team, score in scores.items():
            logger.info(f"   {team.value.upper()} Team: {score.total_score:.2f} points")
            logger.info(f"     Metrics: {score.metrics_count} | Last Updated: {score.last_updated.strftime('%H:%M:%S')}")
    
    async def run_complete_demo(self):
        """Run the complete demonstration"""
        try:
            await self.initialize()
            
            logger.info("🎯 Starting Dynamic Scoring Engine Demonstration")
            logger.info("=" * 60)
            
            # Run demonstration phases
            await self.demonstrate_basic_scoring()
            await asyncio.sleep(1)
            
            await self.demonstrate_collaboration_scoring()
            await asyncio.sleep(1)
            
            await self.demonstrate_learning_adaptation()
            await asyncio.sleep(1)
            
            await self.demonstrate_real_time_updates()
            await asyncio.sleep(1)
            
            # Comprehensive analysis
            analysis = await self.demonstrate_performance_analysis()
            await asyncio.sleep(1)
            
            await self.demonstrate_metrics_export()
            
            logger.info("\n" + "=" * 60)
            logger.info("🎉 Dynamic Scoring Engine Demo Completed Successfully!")
            
            # Final summary
            red_score = analysis["team_scores"]["red"]["total_score"]
            blue_score = analysis["team_scores"]["blue"]["total_score"]
            winner = "RED" if red_score > blue_score else "BLUE"
            
            logger.info(f"\n🏆 Final Results:")
            logger.info(f"   Winner: {winner} Team")
            logger.info(f"   Red Team: {red_score:.2f} points")
            logger.info(f"   Blue Team: {blue_score:.2f} points")
            logger.info(f"   Total Metrics: {analysis['performance_metrics']['total_metrics_recorded']}")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
        finally:
            if self.scoring_engine:
                await self.scoring_engine.shutdown()
                logger.info("🔄 Scoring engine shutdown complete")

async def main():
    """Main demo function"""
    demo = ScoringEngineDemo()
    await demo.run_complete_demo()

if __name__ == "__main__":
    asyncio.run(main())