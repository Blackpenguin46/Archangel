#!/usr/bin/env python3
"""
Standalone test for scoring engine functionality
"""

import asyncio
import logging
from datetime import datetime, timedelta

from agents.scoring_engine import (
    DynamicScoringEngine, ScoringConfig, ScoringWeight, ScoreCategory,
    MetricType, DEFAULT_SCORING_CONFIG
)
from agents.base_agent import Team

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_scoring_accuracy():
    """Test scoring accuracy and fairness"""
    logger.info("🧪 Testing Scoring Accuracy and Fairness")
    
    engine = DynamicScoringEngine(DEFAULT_SCORING_CONFIG)
    await engine.initialize()
    
    try:
        # Scenario 1: Balanced competition
        logger.info("📊 Scenario 1: Balanced Red vs Blue Competition")
        
        # Red team activities
        await engine.record_attack_success("red_1", "server_1", "exploit", True, 60.0, 0.8)
        await engine.record_attack_success("red_2", "server_2", "exploit", False, 120.0, 0.4)
        await engine.record_collaboration_event("red_1", Team.RED, "intel_sharing", 0.7)
        
        # Blue team activities  
        await engine.record_detection_event("blue_1", "red_1", 45.0, 0.9)
        await engine.record_containment_action("blue_2", "threat_1", 90.0, 0.85)
        await engine.record_collaboration_event("blue_1", Team.BLUE, "incident_response", 0.8)
        
        analysis1 = await engine.get_performance_analysis()
        
        logger.info(f"   Red Score: {analysis1['team_scores']['red']['total_score']:.2f}")
        logger.info(f"   Blue Score: {analysis1['team_scores']['blue']['total_score']:.2f}")
        logger.info(f"   Balance: {analysis1['comparative_analysis']['performance_balance']:.2f}")
        
        # Scenario 2: Red team dominance
        logger.info("\n📊 Scenario 2: Red Team Dominance")
        
        # Multiple successful Red team attacks
        for i in range(5):
            await engine.record_attack_success(f"red_{i+3}", f"target_{i}", "exploit", True, 30.0, 0.9)
        
        # Limited Blue team response
        await engine.record_detection_event("blue_3", "red_3", 180.0, 0.6)
        
        analysis2 = await engine.get_performance_analysis()
        
        logger.info(f"   Red Score: {analysis2['team_scores']['red']['total_score']:.2f}")
        logger.info(f"   Blue Score: {analysis2['team_scores']['blue']['total_score']:.2f}")
        logger.info(f"   Balance: {analysis2['comparative_analysis']['performance_balance']:.2f}")
        logger.info(f"   Leading Team: {analysis2['comparative_analysis']['leading_team']}")
        
        # Verify scoring logic
        assert analysis1['comparative_analysis']['performance_balance'] < 0.5, "Balanced scenario should have good balance"
        assert analysis2['comparative_analysis']['leading_team'] == 'red', "Red team should be leading in scenario 2"
        
        logger.info("✅ Scoring accuracy test passed!")
        
    finally:
        await engine.shutdown()

async def test_performance_metrics():
    """Test performance metrics calculation"""
    logger.info("\n🧪 Testing Performance Metrics Calculation")
    
    engine = DynamicScoringEngine(DEFAULT_SCORING_CONFIG)
    await engine.initialize()
    
    try:
        # Record various activities with known metrics
        detection_times = [30.0, 45.0, 60.0, 90.0]
        containment_times = [120.0, 180.0, 240.0]
        
        for i, det_time in enumerate(detection_times):
            await engine.record_detection_event(f"blue_{i}", f"red_{i}", det_time, 0.8)
        
        for i, cont_time in enumerate(containment_times):
            await engine.record_containment_action(f"blue_{i+10}", f"threat_{i}", cont_time, 0.9)
        
        # Record success/failure patterns
        successes = [True, True, False, True, False, True]  # 4/6 = 66.7%
        for i, success in enumerate(successes):
            await engine.record_attack_success(f"red_{i+10}", f"target_{i}", "exploit", success, 60.0, 0.7)
        
        analysis = await engine.get_performance_analysis()
        metrics = analysis['performance_metrics']
        
        # Verify calculated metrics
        expected_avg_detection = sum(detection_times) / len(detection_times)  # 56.25
        expected_avg_containment = sum(containment_times) / len(containment_times)  # 180.0
        expected_red_success_rate = 4/6  # 0.667
        
        logger.info(f"   Average Detection Time: {metrics['average_detection_time']:.1f}s (expected: {expected_avg_detection:.1f}s)")
        logger.info(f"   Average Containment Time: {metrics['average_containment_time']:.1f}s (expected: {expected_avg_containment:.1f}s)")
        logger.info(f"   Red Success Rate: {metrics['red_team_success_rate']:.1%} (expected: {expected_red_success_rate:.1%})")
        
        # Verify accuracy (within small tolerance for floating point)
        assert abs(metrics['average_detection_time'] - expected_avg_detection) < 0.1
        assert abs(metrics['average_containment_time'] - expected_avg_containment) < 0.1
        assert abs(metrics['red_team_success_rate'] - expected_red_success_rate) < 0.01
        
        logger.info("✅ Performance metrics test passed!")
        
    finally:
        await engine.shutdown()

async def test_real_time_scoring():
    """Test real-time scoring updates"""
    logger.info("\n🧪 Testing Real-Time Scoring Updates")
    
    # Create config with faster updates for testing
    config = ScoringConfig(
        weights=DEFAULT_SCORING_CONFIG.weights,
        evaluation_window=timedelta(seconds=30),
        real_time_updates=True
    )
    
    engine = DynamicScoringEngine(config)
    await engine.initialize()
    
    try:
        # Record initial activity
        await engine.record_attack_success("red_rt", "target_rt", "exploit", True, 45.0, 0.8)
        
        # Get initial scores
        scores1 = await engine.get_current_scores()
        initial_red_score = scores1[Team.RED].total_score
        
        logger.info(f"   Initial Red Score: {initial_red_score:.2f}")
        
        # Add more activity
        await engine.record_attack_success("red_rt2", "target_rt2", "exploit", True, 30.0, 0.9)
        await engine.record_collaboration_event("red_rt", Team.RED, "coordination", 0.8)
        
        # Get updated scores
        scores2 = await engine.get_current_scores()
        updated_red_score = scores2[Team.RED].total_score
        
        logger.info(f"   Updated Red Score: {updated_red_score:.2f}")
        
        # Score should have increased
        assert updated_red_score > initial_red_score, "Score should increase with more successful activities"
        
        # Test Blue team response
        await engine.record_detection_event("blue_rt", "red_rt", 20.0, 0.95)
        await engine.record_containment_action("blue_rt2", "threat_rt", 60.0, 0.9)
        
        scores3 = await engine.get_current_scores()
        blue_score = scores3[Team.BLUE].total_score
        
        logger.info(f"   Blue Score after response: {blue_score:.2f}")
        
        assert blue_score > 0, "Blue team should have score after activities"
        
        logger.info("✅ Real-time scoring test passed!")
        
    finally:
        await engine.shutdown()

async def test_trend_analysis():
    """Test performance trend analysis"""
    logger.info("\n🧪 Testing Performance Trend Analysis")
    
    engine = DynamicScoringEngine(DEFAULT_SCORING_CONFIG)
    await engine.initialize()
    
    try:
        # Simulate improving Red team performance
        base_stealth = 0.3
        for i in range(10):
            stealth_score = base_stealth + (i * 0.05)  # Gradually improving
            await engine.record_attack_success(f"red_trend_{i}", f"target_{i}", "exploit", True, 60.0, stealth_score)
            await engine._update_team_scores()  # Force score update
        
        red_score = engine.team_scores[Team.RED]
        trend_direction = engine._calculate_trend_direction(red_score.performance_trend)
        
        logger.info(f"   Red Team Trend: {trend_direction}")
        logger.info(f"   Trend Data Points: {len(red_score.performance_trend)}")
        
        # Should show improving trend
        assert trend_direction == "improving", f"Expected improving trend, got {trend_direction}"
        
        # Simulate declining Blue team performance
        base_accuracy = 0.9
        for i in range(10):
            accuracy = base_accuracy - (i * 0.05)  # Gradually declining
            await engine.record_detection_event(f"blue_trend_{i}", f"red_trend_{i}", 60.0, max(0.1, accuracy))
            await engine._update_team_scores()
        
        blue_score = engine.team_scores[Team.BLUE]
        blue_trend = engine._calculate_trend_direction(blue_score.performance_trend)
        
        logger.info(f"   Blue Team Trend: {blue_trend}")
        
        # Should show declining trend
        assert blue_trend == "declining", f"Expected declining trend, got {blue_trend}"
        
        logger.info("✅ Trend analysis test passed!")
        
    finally:
        await engine.shutdown()

async def test_export_functionality():
    """Test metrics export functionality"""
    logger.info("\n🧪 Testing Metrics Export Functionality")
    
    engine = DynamicScoringEngine(DEFAULT_SCORING_CONFIG)
    await engine.initialize()
    
    try:
        # Generate some test data
        await engine.record_attack_success("red_export", "target_export", "exploit", True, 45.0, 0.8)
        await engine.record_detection_event("blue_export", "red_export", 30.0, 0.9)
        await engine.record_collaboration_event("red_export", Team.RED, "intel", 0.7)
        
        # Export metrics
        exported_json = await engine.export_metrics("json")
        
        # Verify export
        assert exported_json is not None
        assert len(exported_json) > 0
        assert "export_timestamp" in exported_json
        assert "metrics_history" in exported_json
        
        # Parse and verify structure
        import json
        data = json.loads(exported_json)
        
        assert len(data["metrics_history"]) >= 3  # At least our test metrics
        assert "team_scores" in data
        assert "red" in data["team_scores"]
        assert "blue" in data["team_scores"]
        
        logger.info(f"   Exported {len(data['metrics_history'])} metrics")
        logger.info(f"   Export size: {len(exported_json)} characters")
        
        logger.info("✅ Export functionality test passed!")
        
    finally:
        await engine.shutdown()

async def run_standalone_tests():
    """Run all standalone scoring tests"""
    logger.info("🚀 Starting Standalone Scoring Engine Tests")
    logger.info("=" * 60)
    
    tests = [
        test_scoring_accuracy,
        test_performance_metrics,
        test_real_time_scoring,
        test_trend_analysis,
        test_export_functionality
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            logger.error(f"❌ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    logger.info("\n" + "=" * 60)
    logger.info(f"🏆 Standalone Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All standalone tests passed!")
        return True
    else:
        logger.error("❌ Some standalone tests failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_standalone_tests())
    exit(0 if success else 1)