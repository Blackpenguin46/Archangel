#!/usr/bin/env python3
"""
Test DeepSeek R1T2 Integration with Archangel Autonomous System
Comprehensive testing of advanced reasoning capabilities
"""

import asyncio
import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any

# Import DeepSeek integration
from core.deepseek_integration import (
    DeepSeekR1T2Agent,
    create_deepseek_agent,
    DeepSeekEnhancedAgent,
    create_enhanced_autonomous_agent
)

# Import autonomous agents
from core.autonomous_security_agents import (
    BlueTeamDefenderAgent,
    RedTeamAttackerAgent,
    AutonomousSecurityOrchestrator
)

async def test_deepseek_basic_reasoning():
    """Test basic DeepSeek R1T2 reasoning capabilities"""
    print("\n🧠 Testing DeepSeek R1T2 Basic Reasoning")
    print("=" * 50)
    
    try:
        # Create DeepSeek agent
        deepseek_agent = create_deepseek_agent()
        
        # Initialize agent
        if not await deepseek_agent.initialize():
            print("❌ DeepSeek initialization failed - likely model not available")
            return False
        
        print("✅ DeepSeek R1T2 model loaded successfully")
        
        # Test scenario
        test_scenario = """
        A security monitoring system has detected unusual network traffic patterns:
        - Multiple failed SSH login attempts from IP 192.168.1.100
        - Unusual data transfer volumes during off-hours (2-4 AM)
        - New processes spawned with elevated privileges
        - DNS queries to suspicious domains
        
        This activity started 3 days ago and is increasing in frequency.
        """
        
        context = {
            "environment": "corporate_network",
            "business_hours": "9AM-6PM",
            "critical_systems": ["database_server", "email_server", "file_server"],
            "security_tools": ["SIEM", "IDS", "endpoint_protection"]
        }
        
        print("🔍 Analyzing security scenario with DeepSeek reasoning...")
        
        # Perform autonomous reasoning
        result = await deepseek_agent.autonomous_security_reasoning(
            test_scenario,
            context,
            "threat_analysis"
        )
        
        print(f"\n📊 DeepSeek Analysis Results:")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Processing Time: {result.processing_time:.2f}s")
        print(f"Reasoning Steps: {len(result.reasoning_steps)}")
        
        print(f"\n🧠 Reasoning Chain:")
        for i, step in enumerate(result.reasoning_steps[:3], 1):
            print(f"{i}. {step[:100]}...")
        
        print(f"\n💡 Key Insights:")
        insights = result.content[:300] + "..." if len(result.content) > 300 else result.content
        print(insights)
        
        # Test threat analysis
        print(f"\n🎯 Testing Threat Analysis...")
        threat_data = {
            "source_ip": "192.168.1.100",
            "failed_logins": 47,
            "time_pattern": "off_hours",
            "privilege_escalation": True,
            "suspicious_domains": ["malware-c2.evil.com", "data-exfil.bad.net"]
        }
        
        threat_analysis = await deepseek_agent.analyze_threat_with_reasoning(threat_data)
        
        print(f"Threat Severity: {threat_analysis['threat_assessment']['severity']}")
        print(f"Classification: {threat_analysis['threat_assessment']['classification']}")
        print(f"Analysis Confidence: {threat_analysis['confidence']:.2f}")
        
        # Cleanup
        await deepseek_agent.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ DeepSeek test failed: {e}")
        return False

async def test_deepseek_strategy_generation():
    """Test DeepSeek strategy generation capabilities"""
    print("\n🎯 Testing DeepSeek Strategy Generation")
    print("=" * 50)
    
    try:
        deepseek_agent = create_deepseek_agent()
        
        if not await deepseek_agent.initialize():
            print("❌ DeepSeek initialization failed")
            return False
        
        # Test autonomous strategy generation
        threat_type = "Advanced Persistent Threat (APT)"
        current_defenses = [
            "Firewall with IPS",
            "Endpoint antivirus",
            "Email security gateway",
            "Basic SIEM monitoring"
        ]
        
        constraints = {
            "budget": "medium",
            "staff_size": "small_team",
            "compliance_requirements": ["SOX", "PCI-DSS"],
            "business_continuity": "critical"
        }
        
        print(f"🛡️ Generating strategy for {threat_type}...")
        
        strategy_result = await deepseek_agent.generate_autonomous_strategy(
            threat_type,
            current_defenses,
            constraints
        )
        
        print(f"\n📋 Generated Strategy:")
        strategy = strategy_result["strategy"]
        print(f"Immediate Actions: {len(strategy.get('immediate_actions', []))}")
        print(f"Medium-term Actions: {len(strategy.get('medium_term_actions', []))}")
        print(f"Long-term Actions: {len(strategy.get('long_term_actions', []))}")
        
        print(f"\n🔍 Sample Immediate Actions:")
        for action in strategy.get('immediate_actions', [])[:3]:
            print(f"• {action}")
        
        print(f"\nStrategy Confidence: {strategy_result['confidence']:.2f}")
        print(f"Reasoning Steps: {len(strategy_result['reasoning'])}")
        
        await deepseek_agent.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Strategy generation test failed: {e}")
        return False

async def test_deepseek_enhanced_agent():
    """Test DeepSeek enhanced autonomous agent"""
    print("\n🤖 Testing DeepSeek Enhanced Autonomous Agent")
    print("=" * 50)
    
    try:
        # Create enhanced agent
        enhanced_agent = await create_enhanced_autonomous_agent(
            "deepseek_enhanced_001",
            BlueTeamDefenderAgent
        )
        
        print("✅ DeepSeek enhanced agent created")
        
        # Test enhanced operation
        objective = "Detect and respond to potential data exfiltration attempt"
        context = {
            "alert_source": "network_monitoring",
            "indicators": [
                "large_data_transfer",
                "unusual_time_pattern", 
                "external_destination"
            ],
            "environment": "production"
        }
        
        print("🎯 Executing enhanced autonomous operation...")
        
        result = await enhanced_agent.enhanced_autonomous_operation(objective, context)
        
        print(f"\n📊 Enhanced Operation Results:")
        print(f"Status: {result['status']}")
        print(f"Operation ID: {result['operation_id']}")
        
        # DeepSeek planning insights
        planning = result.get('deepseek_planning', {})
        print(f"\n🧠 DeepSeek Planning:")
        print(f"Planning Confidence: {planning.get('confidence', 0):.2f}")
        print(f"Reasoning Steps: {len(planning.get('reasoning_steps', []))}")
        
        # DeepSeek analysis insights  
        analysis = result.get('deepseek_analysis', {})
        print(f"\n📊 DeepSeek Analysis:")
        print(f"Analysis Confidence: {analysis.get('confidence', 0):.2f}")
        print(f"Analysis Steps: {len(analysis.get('reasoning_steps', []))}")
        
        # Enhanced learning
        learning = result.get('enhanced_learning', {})
        print(f"\n🎓 Enhanced Learning:")
        print(f"Learning Quality: {learning.get('learning_quality', 'unknown')}")
        print(f"Overall Confidence: {learning.get('overall_confidence', 0):.2f}")
        
        # Cleanup
        await enhanced_agent.base_agent.cleanup()
        await enhanced_agent.deepseek_agent.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced agent test failed: {e}")
        return False

async def test_deepseek_continuous_reasoning():
    """Test DeepSeek continuous reasoning loop"""
    print("\n🔄 Testing DeepSeek Continuous Reasoning")
    print("=" * 50)
    
    try:
        deepseek_agent = create_deepseek_agent()
        
        if not await deepseek_agent.initialize():
            print("❌ DeepSeek initialization failed")
            return False
        
        # Create multiple test scenarios
        scenarios = [
            {
                "id": "scenario_1",
                "description": "Phishing email campaign targeting executives",
                "context": {"target": "executives", "vector": "email"},
                "type": "threat_analysis"
            },
            {
                "id": "scenario_2", 
                "description": "Suspicious lateral movement in network",
                "context": {"indicators": ["unusual_connections", "privilege_escalation"]},
                "type": "incident_response"
            },
            {
                "id": "scenario_3",
                "description": "Plan security improvements for cloud infrastructure",
                "context": {"platform": "AWS", "budget": "limited"},
                "type": "strategy_planning"
            }
        ]
        
        print(f"🔄 Processing {len(scenarios)} scenarios with continuous reasoning...")
        
        # Learning callback to track improvements
        learning_data = []
        
        async def learning_callback(scenario, result):
            learning_data.append({
                "scenario_id": scenario["id"],
                "confidence": result.confidence,
                "reasoning_steps": len(result.reasoning_steps),
                "processing_time": result.processing_time
            })
        
        # Run continuous reasoning
        results = await deepseek_agent.continuous_reasoning_loop(
            scenarios,
            learning_callback
        )
        
        print(f"\n📊 Continuous Reasoning Results:")
        print(f"Scenarios Processed: {len(results)}")
        
        # Analyze learning progression
        if learning_data:
            avg_confidence = sum(d["confidence"] for d in learning_data) / len(learning_data)
            avg_reasoning_steps = sum(d["reasoning_steps"] for d in learning_data) / len(learning_data)
            total_processing_time = sum(d["processing_time"] for d in learning_data)
            
            print(f"Average Confidence: {avg_confidence:.2f}")
            print(f"Average Reasoning Steps: {avg_reasoning_steps:.1f}")
            print(f"Total Processing Time: {total_processing_time:.2f}s")
        
        # Show performance metrics
        metrics = deepseek_agent.get_performance_metrics()
        print(f"\n⚡ Performance Metrics:")
        print(f"Total Inferences: {metrics['inference_count']}")
        print(f"Average Inference Time: {metrics['average_inference_time']:.2f}s")
        
        await deepseek_agent.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Continuous reasoning test failed: {e}")
        return False

async def test_deepseek_integration_with_orchestrator():
    """Test DeepSeek integration with full orchestrator"""
    print("\n🎼 Testing DeepSeek Integration with Orchestrator")
    print("=" * 50)
    
    try:
        # This would test integration with the full system
        # For now, just verify the integration points exist
        
        from core.autonomous_security_agents import DEEPSEEK_AVAILABLE
        
        print(f"DeepSeek Integration Available: {'✅' if DEEPSEEK_AVAILABLE else '❌'}")
        
        if DEEPSEEK_AVAILABLE:
            print("✅ DeepSeek integration successfully imported")
            
            # Test agent creation with DeepSeek
            agent = BlueTeamDefenderAgent("deepseek_test_001")
            success = await agent.initialize()
            
            if success:
                print("✅ Agent with DeepSeek integration initialized")
                
                # Check if DeepSeek agent was created
                if hasattr(agent, 'deepseek_agent') and agent.deepseek_agent:
                    print("✅ DeepSeek reasoning engine attached to agent")
                else:
                    print("⚠️ DeepSeek agent not attached (likely model not available)")
                
                await agent.cleanup()
            else:
                print("❌ Agent initialization failed")
        
        return DEEPSEEK_AVAILABLE
        
    except Exception as e:
        print(f"❌ Orchestrator integration test failed: {e}")
        return False

async def run_all_deepseek_tests():
    """Run comprehensive DeepSeek integration tests"""
    print("🧠 DeepSeek R1T2 Integration Test Suite")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    
    tests = [
        ("Basic Reasoning", test_deepseek_basic_reasoning),
        ("Strategy Generation", test_deepseek_strategy_generation),
        ("Enhanced Agent", test_deepseek_enhanced_agent),
        ("Continuous Reasoning", test_deepseek_continuous_reasoning),
        ("Orchestrator Integration", test_deepseek_integration_with_orchestrator)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = await test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}: {test_name}")
        except Exception as e:
            results[test_name] = False
            print(f"❌ ERROR in {test_name}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("🧠 DeepSeek R1T2 Test Summary")
    print(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    if passed == total:
        print("\n🎉 All DeepSeek integration tests passed!")
        return True
    else:
        print(f"\n⚠️ {total-passed} tests failed. Check model availability and dependencies.")
        return False

async def main():
    """Main test runner"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        success = await run_all_deepseek_tests()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))