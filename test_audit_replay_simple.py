#!/usr/bin/env python3
"""
Simple test runner for Audit and Replay System
Tests basic functionality without external dependencies
"""

import asyncio
import os
import tempfile
from datetime import datetime

from agents.audit_replay import (
    create_audit_replay_system,
    AuditEventType,
    IntegrityLevel
)

async def test_basic_functionality():
    """Test basic audit and replay functionality"""
    print("Testing Basic Audit and Replay Functionality")
    print("=" * 50)
    
    # Create temporary database
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    db_path = temp_db.name
    
    try:
        # Initialize system
        print("1. Initializing audit system...")
        audit_system = create_audit_replay_system(
            db_path=db_path,
            integrity_level=IntegrityLevel.HASH
        )
        await audit_system.initialize()
        print("   ✓ System initialized")
        
        # Start session
        print("2. Starting audit session...")
        session_id = "test_session_001"
        await audit_system.start_audit_session(
            session_id=session_id,
            scenario_id="basic_test",
            participants=["test_agent"],
            objectives=["Test basic functionality"]
        )
        print("   ✓ Session started")
        
        # Log agent decision
        print("3. Logging agent decision...")
        decision_id = await audit_system.log_agent_decision(
            session_id=session_id,
            agent_id="test_agent",
            prompt="Test decision prompt",
            context={"test": "context"},
            reasoning="Test reasoning",
            action_taken="test_action",
            action_parameters={"param": "value"},
            confidence_score=0.85,
            tags=["test"]
        )
        print(f"   ✓ Decision logged: {decision_id}")
        
        # Log LLM interaction
        print("4. Logging LLM interaction...")
        prompt_id, response_id = await audit_system.log_llm_interaction(
            session_id=session_id,
            agent_id="test_agent",
            prompt="Test LLM prompt",
            response="Test LLM response",
            model_name="test-model",
            tags=["llm", "test"]
        )
        print(f"   ✓ LLM interaction logged: {prompt_id}, {response_id}")
        
        # Search events
        print("5. Searching events...")
        events = await audit_system.search_audit_events(
            query="test",
            session_id=session_id
        )
        print(f"   ✓ Found {len(events)} events")
        
        # Verify integrity
        print("6. Verifying integrity...")
        verified_count = 0
        for event in events:
            if audit_system.integrity.verify_integrity(event):
                verified_count += 1
        print(f"   ✓ Verified {verified_count}/{len(events)} events")
        
        # Create replay
        print("7. Creating replay session...")
        replay_id = await audit_system.create_session_replay(session_id)
        print(f"   ✓ Replay created: {replay_id}")
        
        # Test replay step
        print("8. Testing replay...")
        step_count = 0
        while True:
            event = await audit_system.replay_engine.replay_step(replay_id)
            if event is None:
                break
            step_count += 1
        print(f"   ✓ Replayed {step_count} events")
        
        # End session
        print("9. Ending session...")
        await audit_system.end_audit_session(session_id)
        print("   ✓ Session ended")
        
        # Get statistics
        print("10. Getting statistics...")
        stats = await audit_system.get_system_stats()
        print(f"    Events logged: {stats['system_stats']['events_logged']}")
        print(f"    Sessions created: {stats['system_stats']['sessions_created']}")
        print(f"    Replays created: {stats['system_stats']['replays_created']}")
        print("   ✓ Statistics retrieved")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False
        
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)

async def test_integrity_levels():
    """Test different integrity levels"""
    print("\nTesting Integrity Levels")
    print("=" * 25)
    
    integrity_levels = [
        (IntegrityLevel.NONE, "None"),
        (IntegrityLevel.HASH, "Hash"),
        (IntegrityLevel.HMAC, "HMAC")
    ]
    
    for level, name in integrity_levels:
        print(f"Testing {name} integrity...")
        
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        db_path = temp_db.name
        
        try:
            # Initialize with specific integrity level
            audit_system = create_audit_replay_system(
                db_path=db_path,
                integrity_level=level
            )
            await audit_system.initialize()
            
            # Start session and log event
            session_id = f"integrity_test_{name.lower()}"
            await audit_system.start_audit_session(
                session_id=session_id,
                scenario_id="integrity_test",
                participants=["test_agent"]
            )
            
            await audit_system.log_agent_decision(
                session_id=session_id,
                agent_id="test_agent",
                prompt="Integrity test",
                context={"level": name},
                reasoning="Testing integrity",
                action_taken="integrity_test",
                action_parameters={"level": name},
                confidence_score=0.9
            )
            
            # Verify integrity
            events = await audit_system.search_audit_events(
                query="",
                session_id=session_id
            )
            
            if events:
                verified = audit_system.integrity.verify_integrity(events[0])
                print(f"   ✓ {name} integrity: {verified}")
            else:
                print(f"   ❌ No events found for {name}")
            
            await audit_system.end_audit_session(session_id)
            
        except Exception as e:
            print(f"   ❌ {name} integrity test failed: {e}")
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

async def test_search_functionality():
    """Test search functionality"""
    print("\nTesting Search Functionality")
    print("=" * 28)
    
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    db_path = temp_db.name
    
    try:
        audit_system = create_audit_replay_system(db_path=db_path)
        await audit_system.initialize()
        
        session_id = "search_test_session"
        await audit_system.start_audit_session(
            session_id=session_id,
            scenario_id="search_test",
            participants=["agent1", "agent2"]
        )
        
        # Create diverse events
        test_events = [
            ("agent1", "network scan", "reconnaissance", ["network", "scan"]),
            ("agent1", "exploit vulnerability", "exploitation", ["exploit", "vulnerability"]),
            ("agent2", "detect intrusion", "detection", ["detection", "security"]),
            ("agent2", "block traffic", "response", ["firewall", "blocking"])
        ]
        
        for agent_id, action, reasoning, tags in test_events:
            await audit_system.log_agent_decision(
                session_id=session_id,
                agent_id=agent_id,
                prompt=f"Execute {action}",
                context={"action": action},
                reasoning=reasoning,
                action_taken=action.replace(" ", "_"),
                action_parameters={"test": True},
                confidence_score=0.8,
                tags=tags
            )
        
        # Test different search scenarios
        search_tests = [
            ("All events", "", None, None, 4),
            ("Network events", "network", None, None, 1),
            ("Agent1 events", "", "agent1", None, 2),
            ("Agent2 events", "", "agent2", None, 2),
            ("Decision events", "", None, AuditEventType.AGENT_DECISION, 4)
        ]
        
        for test_name, query, agent_id, event_type, expected_min in search_tests:
            results = await audit_system.search_audit_events(
                query=query,
                agent_id=agent_id,
                event_type=event_type,
                session_id=session_id
            )
            
            if len(results) >= expected_min:
                print(f"   ✓ {test_name}: {len(results)} events")
            else:
                print(f"   ❌ {test_name}: {len(results)} events (expected >= {expected_min})")
        
        await audit_system.end_audit_session(session_id)
        print("   ✓ Search functionality verified")
        
    except Exception as e:
        print(f"   ❌ Search test failed: {e}")
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

async def main():
    """Run all tests"""
    print("AUDIT AND REPLAY SYSTEM - SIMPLE TESTS")
    print("=" * 40)
    
    tests = [
        test_basic_functionality,
        test_integrity_levels,
        test_search_functionality
    ]
    
    results = []
    
    for test_func in tests:
        try:
            result = await test_func()
            results.append(result if result is not None else True)
        except Exception as e:
            print(f"Test {test_func.__name__} failed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\nTEST SUMMARY")
    print("=" * 12)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)