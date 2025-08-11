#!/usr/bin/env python3
"""
Simple test for the Dynamic Scoring Engine without pytest
"""

import asyncio
import sys
import traceback
from datetime import datetime, timedelta

from agents.scoring_engine import (
    DynamicScoringEngine, ScoringConfig, ScoringWeight, ScoreCategory,
    MetricType, DEFAULT_SCORING_CONFIG
)
from agents.base_agent import Team

async def test_basic_functionality():
    """Test basic scoring engine functionality"""
    print("🧪 Testing basic scoring engine functionality...")
    
    # Create engine
    engine = DynamicScoringEngine(DEFAULT_SCORING_CONFIG)
    
    try:
        # Initialize
        await engine.initialize()
        print("✅ Engine initialization successful")
        
        # Test metric recording
        metric_id = await engine.record_metric(
            agent_id="test_agent",
            team=Team.RED,
            category=ScoreCategory.ATTACK_SUCCESS,
            metric_type=MetricType.SCORE,
            value=0.8,
            context={"test": "data"}
        )
        
        assert metric_id is not None
        assert len(engine.metrics_history) == 1
        print("✅ Metric recording successful")
        
        # Test attack success recording
        attack_id = await engine.record_attack_success(
            agent_id="red_agent",
            target="test_target",
            attack_type="test_attack",
            success=True,
            duration=60.0,
            stealth_score=0.7
        )
        
        assert attack_id is not None
        assert len(engine.metrics_history) == 3  # attack + stealth metrics
        print("✅ Attack success recording successful")
        
        # Test detection event recording
        detection_id = await engine.record_detection_event(
            agent_id="blue_agent",
            detected_agent="red_agent",
            detection_time=30.0,
            accuracy=0.9
        )
        
        assert detection_id is not None
        assert len(engine.metrics_history) == 5  # defense + speed metrics
        print("✅ Detection event recording successful")
        
        # Test score calculation
        await engine._update_team_scores()
        
        red_score = engine.team_scores[Team.RED]
        blue_score = engine.team_scores[Team.BLUE]
        
        assert red_score.total_score > 0
        assert blue_score.total_score > 0
        print("✅ Score calculation successful")
        
        # Test performance analysis
        analysis = await engine.get_performance_analysis()
        
        assert "team_scores" in analysis
        assert "performance_metrics" in analysis
        assert "comparative_analysis" in analysis
        print("✅ Performance analysis successful")
        
        # Test metrics export
        exported = await engine.export_metrics("json")
        assert exported is not None
        assert len(exported) > 0
        print("✅ Metrics export successful")
        
        print("🎉 All basic functionality tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False
    finally:
        await engine.shutdown()
    
    return True

async def test_comprehensive_scenario():
    """Test a comprehensive Red vs Blue scenario"""
    print("\n🧪 Testing comprehensive Red vs Blue scenario...")
    
    engine = DynamicScoringEngine(DEFAULT_SCORING_CONFIG)
    
    try:
        await engine.initialize()
        
        # Simulate a complete attack scenario
        
        # Phase 1: Red team reconnaissance
        await engine.record_attack_success(
            "red_recon", "network", "scanning", True, 45.0, 0.9
        )
        
        # Phase 2: Blue team detection
        await engine.record_detection_event(
            "blue_soc", "red_recon", 60.0, 0.8
        )
        
        # Phase 3: Red team exploitation
        await engine.record_attack_success(
            "red_exploit", "web_server", "sql_injection", True, 120.0, 0.6
        )
        
        # Phase 4: Blue team containment
        await engine.record_containment_action(
            "blue_firewall", "sql_threat", 180.0, 0.9
        )
        
        # Phase 5: Team collaboration
        await engine.record_collaboration_event(
            "red_recon", Team.RED, "intel_sharing", 0.8
        )
        await engine.record_collaboration_event(
            "blue_soc", Team.BLUE, "incident_response", 0.85
        )
        
        # Phase 6: Learning adaptation
        await engine.record_learning_adaptation(
            "red_exploit", Team.RED, 0.7, {"learning": "evasion_techniques"}
        )
        await engine.record_learning_adaptation(
            "blue_soc", Team.BLUE, 0.8, {"learning": "pattern_recognition"}
        )
        
        # Get final analysis
        analysis = await engine.get_performance_analysis()
        
        # Verify results
        assert analysis["team_scores"]["red"]["total_score"] > 0
        assert analysis["team_scores"]["blue"]["total_score"] > 0
        assert analysis["performance_metrics"]["red_team_success_rate"] == 1.0
        assert analysis["performance_metrics"]["blue_team_success_rate"] == 1.0
        
        print("✅ Comprehensive scenario test passed!")
        
        # Display results
        print(f"   Red Team Score: {analysis['team_scores']['red']['total_score']:.2f}")
        print(f"   Blue Team Score: {analysis['team_scores']['blue']['total_score']:.2f}")
        print(f"   Leading Team: {analysis['comparative_analysis']['leading_team']}")
        
    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")
        traceback.print_exc()
        return False
    finally:
        await engine.shutdown()
    
    return True

async def test_fairness_adjustments():
    """Test fairness adjustment functionality"""
    print("\n🧪 Testing fairness adjustments...")
    
    # Create config with fairness enabled
    config = ScoringConfig(
        weights={
            ScoreCategory.ATTACK_SUCCESS: ScoringWeight(
                category=ScoreCategory.ATTACK_SUCCESS,
                weight=0.5,
                max_score=100.0
            ),
            ScoreCategory.DEFENSE_SUCCESS: ScoringWeight(
                category=ScoreCategory.DEFENSE_SUCCESS,
                weight=0.5,
                max_score=100.0
            )
        },
        fairness_adjustments=True,
        real_time_updates=False
    )
    
    engine = DynamicScoringEngine(config)
    
    try:
        await engine.initialize()
        
        # Create imbalanced scenario
        for i in range(10):
            await engine.record_metric(
                f"red_agent_{i}", Team.RED, ScoreCategory.ATTACK_SUCCESS,
                MetricType.SCORE, 0.9
            )
        
        for i in range(3):
            await engine.record_metric(
                f"blue_agent_{i}", Team.BLUE, ScoreCategory.DEFENSE_SUCCESS,
                MetricType.SCORE, 0.3
            )
        
        await engine._update_team_scores()
        
        red_score = engine.team_scores[Team.RED].total_score
        blue_score = engine.team_scores[Team.BLUE].total_score
        
        print(f"   With fairness - Red: {red_score:.2f}, Blue: {blue_score:.2f}")
        
        # Test should show Red ahead but not excessively
        assert red_score > blue_score
        print("✅ Fairness adjustments test passed!")
        
    except Exception as e:
        print(f"❌ Fairness test failed: {e}")
        traceback.print_exc()
        return False
    finally:
        await engine.shutdown()
    
    return True

async def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Dynamic Scoring Engine Tests")
    print("=" * 50)
    
    tests = [
        test_basic_functionality,
        test_comprehensive_scenario,
        test_fairness_adjustments
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"🏆 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed successfully!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)