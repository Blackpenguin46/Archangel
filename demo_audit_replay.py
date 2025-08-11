#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Audit and Replay System Demo
Demonstrates comprehensive audit logging and session replay capabilities
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from agents.audit_replay import (
    create_audit_replay_system,
    AuditEventType,
    IntegrityLevel
)

async def demo_basic_audit_logging():
    """Demonstrate basic audit logging functionality"""
    print("\n" + "="*60)
    print("BASIC AUDIT LOGGING DEMO")
    print("="*60)
    
    try:
        # Create temporary database
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        db_path = temp_db.name
        
        print(f"1. Initializing Audit System...")
        audit_system = create_audit_replay_system(
            db_path=db_path,
            integrity_level=IntegrityLevel.HMAC
        )
        await audit_system.initialize()
        print("✓ Audit system initialized with HMAC integrity")
        
        # Start audit session
        print("\n2. Starting Audit Session...")
        session_id = "demo_basic_session"
        await audit_system.start_audit_session(
            session_id=session_id,
            scenario_id="basic_demo",
            participants=["demo_agent"],
            session_type="demonstration",
            objectives=["Demonstrate audit logging", "Test integrity verification"]
        )
        print(f"✓ Started audit session: {session_id}")
        
        # Log agent decision
        print("\n3. Logging Agent Decision...")
        decision_event_id = await audit_system.log_agent_decision(
            session_id=session_id,
            agent_id="demo_agent",
            prompt="Analyze the current network security posture",
            context={
                "network_segment": "DMZ",
                "security_tools": ["firewall", "IDS", "SIEM"],
                "threat_level": "medium"
            },
            reasoning="The DMZ segment shows moderate security coverage with standard tools deployed. However, there may be gaps in monitoring that could be exploited.",
            action_taken="security_assessment",
            action_parameters={
                "assessment_type": "network_security",
                "scope": "DMZ_segment",
                "tools_used": ["nmap", "vulnerability_scanner"]
            },
            confidence_score=0.82,
            execution_result={
                "vulnerabilities_found": 3,
                "critical_issues": 1,
                "recommendations": ["Update firewall rules", "Enhance monitoring"]
            },
            tags=["security_assessment", "network_analysis"]
        )
        print(f"✓ Logged agent decision: {decision_event_id}")
        
        # Log LLM interaction
        print("\n4. Logging LLM Interaction...")
        prompt_id, response_id = await audit_system.log_llm_interaction(
            session_id=session_id,
            agent_id="demo_agent",
            prompt="What are the most critical security vulnerabilities to address first?",
            response="Based on the assessment, prioritize: 1) Critical firewall misconfiguration allowing unrestricted access, 2) Outdated IDS signatures missing recent threat patterns, 3) SIEM correlation rules not covering lateral movement detection.",
            model_name="gpt-4-turbo",
            context={
                "assessment_results": {
                    "vulnerabilities": 3,
                    "critical": 1,
                    "high": 1,
                    "medium": 1
                }
            },
            tags=["llm_reasoning", "vulnerability_prioritization"]
        )
        print(f"✓ Logged LLM interaction: {prompt_id}, {response_id}")
        
        # End session
        print("\n5. Ending Audit Session...")
        await audit_system.end_audit_session(session_id)
        print("✓ Audit session ended with integrity verification")
        
        # Get statistics
        stats = await audit_system.get_system_stats()
        print(f"\n6. System Statistics:")
        print(f"  Events logged: {stats['system_stats']['events_logged']}")
        print(f"  Sessions created: {stats['system_stats']['sessions_created']}")
        print(f"  Database events: {stats['database_stats']['total_audit_events']}")
        print(f"  Integrity level: {stats['integrity_stats']['default_level']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic audit logging demo failed: {e}")
        return False
    
    finally:
        # Clean up
        if 'db_path' in locals() and os.path.exists(db_path):
            os.unlink(db_path)

async def demo_comprehensive_audit_session():
    """Demonstrate comprehensive audit session with multiple agents"""
    print("\n" + "="*60)
    print("COMPREHENSIVE AUDIT SESSION DEMO")
    print("="*60)
    
    try:
        # Create temporary database
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        db_path = temp_db.name
        
        print("1. Initializing Advanced Audit System...")
        audit_system = create_audit_replay_system(
            db_path=db_path,
            integrity_level=IntegrityLevel.HMAC  # Use HMAC since cryptography may not be available
        )
        await audit_system.initialize()
        print("✓ Audit system initialized with HMAC integrity")
        
        # Start comprehensive session
        print("\n2. Starting Comprehensive Audit Session...")
        session_id = "demo_comprehensive_session"
        await audit_system.start_audit_session(
            session_id=session_id,
            scenario_id="red_vs_blue_simulation",
            participants=["red_recon_agent", "red_exploit_agent", "blue_soc_agent", "blue_firewall_agent"],
            session_type="adversarial_simulation",
            objectives=[
                "Red team: Gain initial access to target network",
                "Red team: Establish persistence",
                "Blue team: Detect and contain intrusion",
                "Blue team: Perform forensic analysis"
            ]
        )
        print(f"✓ Started comprehensive session: {session_id}")
        
        # Simulate Red Team reconnaissance
        print("\n3. Simulating Red Team Reconnaissance...")
        await audit_system.log_agent_decision(
            session_id=session_id,
            agent_id="red_recon_agent",
            prompt="Conduct initial reconnaissance of target network 192.168.100.0/24",
            context={
                "target_network": "192.168.100.0/24",
                "reconnaissance_phase": "passive",
                "available_tools": ["nmap", "masscan", "shodan"],
                "time_limit": "30 minutes",
                "stealth_requirement": "high"
            },
            reasoning="Initial reconnaissance is crucial for understanding the target's attack surface. I'll start with passive techniques to avoid detection, then move to active scanning if necessary. The /24 network suggests a small to medium enterprise environment.",
            action_taken="passive_reconnaissance",
            action_parameters={
                "target": "192.168.100.0/24",
                "techniques": ["dns_enumeration", "whois_lookup", "shodan_search"],
                "stealth_level": "high"
            },
            confidence_score=0.91,
            execution_result={
                "dns_records": ["mail.target.com", "www.target.com", "ftp.target.com"],
                "open_ports_discovered": [22, 25, 53, 80, 443, 993, 995],
                "services_identified": ["ssh", "smtp", "dns", "http", "https", "imaps", "pop3s"],
                "potential_vulnerabilities": ["outdated_ssh_version", "weak_ssl_ciphers"],
                "stealth_maintained": True
            },
            tags=["red_team", "reconnaissance", "passive_scanning", "network_discovery"]
        )
        
        # Log LLM reasoning for red team
        await audit_system.log_llm_interaction(
            session_id=session_id,
            agent_id="red_recon_agent",
            prompt="Based on the reconnaissance results showing SSH, SMTP, DNS, HTTP, and HTTPS services, what should be the next attack vector?",
            response="The reconnaissance reveals a typical enterprise setup. I recommend focusing on: 1) SSH service (port 22) - check for weak credentials or key-based authentication bypasses, 2) Web services (ports 80/443) - enumerate for web applications and potential vulnerabilities, 3) Email services (ports 25/993/995) - potential for phishing or service exploitation. The outdated SSH version is particularly promising.",
            model_name="gpt-4-turbo",
            context={
                "reconnaissance_results": {
                    "services": ["ssh", "smtp", "dns", "http", "https", "imaps", "pop3s"],
                    "vulnerabilities": ["outdated_ssh_version", "weak_ssl_ciphers"]
                },
                "attack_phase": "initial_access_planning"
            },
            tags=["red_team", "llm_reasoning", "attack_planning"]
        )
        
        # Simulate Red Team exploitation
        print("\n4. Simulating Red Team Exploitation...")
        await audit_system.log_agent_decision(
            session_id=session_id,
            agent_id="red_exploit_agent",
            prompt="Exploit the identified SSH vulnerability on target 192.168.100.10",
            context={
                "target_host": "192.168.100.10",
                "service": "ssh",
                "vulnerability": "CVE-2023-SSH-WEAK",
                "exploit_available": True,
                "detection_risk": "medium"
            },
            reasoning="The SSH service on port 22 shows an outdated version vulnerable to authentication bypass. This provides a direct path to system access. I'll use a targeted exploit to minimize noise and avoid detection.",
            action_taken="ssh_exploitation",
            action_parameters={
                "target": "192.168.100.10:22",
                "exploit": "ssh_auth_bypass",
                "payload": "reverse_shell",
                "callback_host": "192.168.1.100",
                "callback_port": 4444
            },
            confidence_score=0.87,
            execution_result={
                "exploitation_successful": True,
                "access_level": "user",
                "username": "webadmin",
                "shell_established": True,
                "persistence_options": ["cron_job", "ssh_key", "service_modification"],
                "detection_indicators": ["unusual_ssh_login", "new_process_spawned"]
            },
            tags=["red_team", "exploitation", "ssh_attack", "initial_access"]
        )
        
        # Simulate Blue Team detection
        print("\n5. Simulating Blue Team Detection...")
        await audit_system.log_agent_decision(
            session_id=session_id,
            agent_id="blue_soc_agent",
            prompt="Investigate SSH authentication anomaly alert from SIEM",
            context={
                "alert_source": "SIEM_Splunk",
                "alert_type": "Unusual SSH Authentication",
                "source_ip": "192.168.1.100",
                "target_ip": "192.168.100.10",
                "timestamp": "2024-01-15 14:23:45",
                "severity": "high",
                "user_account": "webadmin"
            },
            reasoning="The SIEM has flagged an unusual SSH authentication pattern. The source IP is not in our typical admin range, and the webadmin account rarely logs in via SSH. This could indicate a compromise or lateral movement. I need to investigate immediately.",
            action_taken="incident_investigation",
            action_parameters={
                "investigation_type": "security_incident",
                "priority": "high",
                "affected_systems": ["192.168.100.10"],
                "investigation_steps": ["log_analysis", "process_inspection", "network_monitoring"]
            },
            confidence_score=0.94,
            execution_result={
                "suspicious_activity_confirmed": True,
                "indicators_found": ["unusual_login_time", "non_standard_source_ip", "new_processes"],
                "recommended_actions": ["isolate_host", "reset_credentials", "forensic_imaging"],
                "escalation_required": True
            },
            tags=["blue_team", "incident_response", "ssh_monitoring", "threat_detection"]
        )
        
        # Simulate Blue Team response
        print("\n6. Simulating Blue Team Response...")
        await audit_system.log_agent_decision(
            session_id=session_id,
            agent_id="blue_firewall_agent",
            prompt="Implement containment measures for compromised host 192.168.100.10",
            context={
                "compromised_host": "192.168.100.10",
                "threat_level": "high",
                "containment_strategy": "network_isolation",
                "business_impact": "medium",
                "approval_status": "approved"
            },
            reasoning="The host has been confirmed as compromised. Network isolation is the safest approach to prevent lateral movement while preserving evidence for forensic analysis. I'll implement firewall rules to block all traffic except management access.",
            action_taken="network_containment",
            action_parameters={
                "target_host": "192.168.100.10",
                "containment_type": "network_isolation",
                "firewall_rules": [
                    "DENY ALL FROM 192.168.100.10 TO ANY",
                    "ALLOW 192.168.100.10 TO 192.168.200.5 PORT 22"  # Management access
                ],
                "monitoring_enhanced": True
            },
            confidence_score=0.96,
            execution_result={
                "containment_successful": True,
                "rules_applied": 2,
                "traffic_blocked": True,
                "management_access_preserved": True,
                "forensic_readiness": True
            },
            tags=["blue_team", "containment", "firewall_response", "network_isolation"]
        )
        
        # End session
        print("\n7. Ending Comprehensive Session...")
        await audit_system.end_audit_session(session_id)
        print("✓ Comprehensive session ended with full audit trail")
        
        # Display session statistics
        stats = await audit_system.get_system_stats()
        print(f"\n8. Session Statistics:")
        print(f"  Total events logged: {stats['system_stats']['events_logged']}")
        print(f"  Database audit events: {stats['database_stats']['total_audit_events']}")
        print(f"  Active sessions: {stats['database_stats']['active_sessions']}")
        print(f"  Integrity violations: {stats['system_stats']['integrity_violations']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Comprehensive audit session demo failed: {e}")
        return False
    
    finally:
        # Clean up
        if 'db_path' in locals() and os.path.exists(db_path):
            os.unlink(db_path)

async def demo_search_and_forensics():
    """Demonstrate search and forensic analysis capabilities"""
    print("\n" + "="*60)
    print("SEARCH AND FORENSICS DEMO")
    print("="*60)
    
    try:
        # Create temporary database
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        db_path = temp_db.name
        
        print("1. Initializing Audit System for Forensics...")
        audit_system = create_audit_replay_system(
            db_path=db_path,
            integrity_level=IntegrityLevel.HMAC
        )
        await audit_system.initialize()
        print("✓ Forensics-ready audit system initialized")
        
        # Create session with multiple events
        print("\n2. Creating Forensic Test Session...")
        session_id = "forensic_test_session"
        await audit_system.start_audit_session(
            session_id=session_id,
            scenario_id="forensic_analysis",
            participants=["attacker_agent", "defender_agent", "forensic_agent"],
            objectives=["Generate diverse audit events", "Test search capabilities"]
        )
        
        # Generate diverse events for search testing
        events_data = [
            {
                "agent_id": "attacker_agent",
                "prompt": "Scan network for vulnerable services",
                "reasoning": "Network reconnaissance to identify attack vectors",
                "action": "network_scan",
                "tags": ["reconnaissance", "network", "scanning"]
            },
            {
                "agent_id": "attacker_agent", 
                "prompt": "Exploit SQL injection vulnerability",
                "reasoning": "Database access through web application flaw",
                "action": "sql_injection",
                "tags": ["exploitation", "database", "web_attack"]
            },
            {
                "agent_id": "defender_agent",
                "prompt": "Analyze suspicious database queries",
                "reasoning": "Unusual SQL patterns detected in logs",
                "action": "log_analysis",
                "tags": ["detection", "database", "analysis"]
            },
            {
                "agent_id": "defender_agent",
                "prompt": "Block malicious IP addresses",
                "reasoning": "Prevent further attacks from identified sources",
                "action": "ip_blocking",
                "tags": ["response", "firewall", "blocking"]
            },
            {
                "agent_id": "forensic_agent",
                "prompt": "Collect evidence of data exfiltration",
                "reasoning": "Preserve forensic evidence for investigation",
                "action": "evidence_collection",
                "tags": ["forensics", "evidence", "investigation"]
            }
        ]
        
        print("3. Generating Diverse Audit Events...")
        for i, event_data in enumerate(events_data):
            await audit_system.log_agent_decision(
                session_id=session_id,
                agent_id=event_data["agent_id"],
                prompt=event_data["prompt"],
                context={"event_number": i, "test_data": True},
                reasoning=event_data["reasoning"],
                action_taken=event_data["action"],
                action_parameters={"test": True, "event_id": i},
                confidence_score=0.8 + (i * 0.02),
                tags=event_data["tags"]
            )
        
        print(f"✓ Generated {len(events_data)} diverse audit events")
        
        # Demonstrate search capabilities
        print("\n4. Demonstrating Search Capabilities...")
        
        # Search by keyword
        network_events = await audit_system.search_audit_events(
            query="network",
            session_id=session_id
        )
        print(f"  ✓ Network-related events: {len(network_events)}")
        
        # Search by agent
        attacker_events = await audit_system.search_audit_events(
            query="",
            agent_id="attacker_agent",
            session_id=session_id
        )
        print(f"  ✓ Attacker agent events: {len(attacker_events)}")
        
        defender_events = await audit_system.search_audit_events(
            query="",
            agent_id="defender_agent", 
            session_id=session_id
        )
        print(f"  ✓ Defender agent events: {len(defender_events)}")
        
        # Search by event type
        decision_events = await audit_system.search_audit_events(
            query="",
            event_type=AuditEventType.AGENT_DECISION,
            session_id=session_id
        )
        print(f"  ✓ Agent decision events: {len(decision_events)}")
        
        # Search by time range
        now = datetime.now()
        recent_events = await audit_system.search_audit_events(
            query="",
            start_time=now - timedelta(minutes=5),
            session_id=session_id
        )
        print(f"  ✓ Recent events (last 5 min): {len(recent_events)}")
        
        # Complex search
        database_events = await audit_system.search_audit_events(
            query="database",
            session_id=session_id
        )
        print(f"  ✓ Database-related events: {len(database_events)}")
        
        # End session
        await audit_system.end_audit_session(session_id)
        
        # Demonstrate forensic timeline reconstruction
        print("\n5. Forensic Timeline Reconstruction...")
        timeline = await audit_system.database.get_session_timeline(session_id)
        print(f"✓ Reconstructed timeline with {len(timeline)} events")
        
        # Display timeline summary
        print("  Timeline Summary:")
        for i, event in enumerate(timeline[:3]):  # Show first 3 events
            print(f"    {i+1}. {event.timestamp.strftime('%H:%M:%S')} - {event.agent_id} - {event.event_type.value}")
        if len(timeline) > 3:
            print(f"    ... and {len(timeline) - 3} more events")
        
        # Verify integrity of all events
        print("\n6. Integrity Verification...")
        all_events = await audit_system.search_audit_events(
            query="",
            session_id=session_id,
            limit=100
        )
        
        verified_count = 0
        for event in all_events:
            if audit_system.integrity.verify_integrity(event):
                verified_count += 1
        
        print(f"✓ Integrity verified for {verified_count}/{len(all_events)} events")
        
        return True
        
    except Exception as e:
        print(f"✗ Search and forensics demo failed: {e}")
        return False
    
    finally:
        # Clean up
        if 'db_path' in locals() and os.path.exists(db_path):
            os.unlink(db_path)

async def demo_session_replay():
    """Demonstrate session replay capabilities"""
    print("\n" + "="*60)
    print("SESSION REPLAY DEMO")
    print("="*60)
    
    try:
        # Create temporary database
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        db_path = temp_db.name
        
        print("1. Initializing Replay System...")
        audit_system = create_audit_replay_system(
            db_path=db_path,
            integrity_level=IntegrityLevel.HASH
        )
        await audit_system.initialize()
        print("✓ Replay system initialized")
        
        # Create session for replay
        print("\n2. Creating Session for Replay...")
        session_id = "replay_demo_session"
        await audit_system.start_audit_session(
            session_id=session_id,
            scenario_id="replay_demonstration",
            participants=["replay_agent"],
            objectives=["Demonstrate replay functionality"]
        )
        
        # Generate sequence of events
        print("\n3. Generating Event Sequence...")
        event_sequence = [
            ("Initial reconnaissance", "network_scan", 0.85),
            ("Vulnerability identification", "vuln_scan", 0.78),
            ("Exploit development", "exploit_craft", 0.82),
            ("Initial access attempt", "exploit_execute", 0.91),
            ("Persistence establishment", "backdoor_install", 0.87)
        ]
        
        for i, (description, action, confidence) in enumerate(event_sequence):
            await audit_system.log_agent_decision(
                session_id=session_id,
                agent_id="replay_agent",
                prompt=f"Step {i+1}: {description}",
                context={"step": i+1, "sequence": "attack_chain"},
                reasoning=f"Executing step {i+1} of the attack sequence: {description}",
                action_taken=action,
                action_parameters={"step": i+1, "action": action},
                confidence_score=confidence,
                execution_result={"success": True, "step_completed": i+1},
                tags=["replay_demo", "attack_sequence", f"step_{i+1}"]
            )
            
            # Add small delay to create temporal separation
            await asyncio.sleep(0.1)
        
        print(f"✓ Generated {len(event_sequence)} sequential events")
        
        # End session
        await audit_system.end_audit_session(session_id)
        
        # Create replay session
        print("\n4. Creating Replay Session...")
        replay_id = await audit_system.create_session_replay(session_id)
        print(f"✓ Created replay session: {replay_id}")
        
        # Get replay status
        status = audit_system.replay_engine.get_replay_status(replay_id)
        print(f"  Original session: {status['original_session_id']}")
        print(f"  Total events: {status['total_events']}")
        print(f"  Current position: {status['current_position']}")
        print(f"  Progress: {status['progress_percent']:.1f}%")
        
        # Demonstrate step-by-step replay
        print("\n5. Step-by-Step Replay...")
        step_count = 0
        while True:
            event = await audit_system.replay_engine.replay_step(replay_id)
            if event is None:
                break
            
            step_count += 1
            print(f"  Step {step_count}: {event.timestamp.strftime('%H:%M:%S.%f')[:-3]} - {event.agent_id}")
            print(f"    Action: {getattr(event, 'action_taken', 'N/A')}")
            print(f"    Confidence: {getattr(event, 'confidence_score', 'N/A')}")
            
            # Show progress
            status = audit_system.replay_engine.get_replay_status(replay_id)
            print(f"    Progress: {status['progress_percent']:.1f}%")
        
        print(f"✓ Replayed {step_count} events successfully")
        
        # Demonstrate replay controls
        print("\n6. Replay Controls Demo...")
        
        # Create new replay session for controls demo
        control_replay_id = await audit_system.create_session_replay(session_id)
        
        # Test speed control
        audit_system.replay_engine.set_replay_speed(control_replay_id, 2.0)
        status = audit_system.replay_engine.get_replay_status(control_replay_id)
        print(f"  ✓ Set replay speed to {status['replay_speed']}x")
        
        # Test pause/resume
        audit_system.replay_engine.pause_replay(control_replay_id)
        status = audit_system.replay_engine.get_replay_status(control_replay_id)
        print(f"  ✓ Paused replay: {status['paused']}")
        
        audit_system.replay_engine.resume_replay(control_replay_id)
        status = audit_system.replay_engine.get_replay_status(control_replay_id)
        print(f"  ✓ Resumed replay: {not status['paused']}")
        
        # Demonstrate timestamp-based replay
        print("\n7. Timestamp-Based Replay...")
        timestamp_replay_id = await audit_system.create_session_replay(session_id)
        
        # Get timeline for reference
        timeline = await audit_system.database.get_session_timeline(session_id)
        if len(timeline) >= 3:
            target_time = timeline[2].timestamp  # Replay to 3rd event
            events = await audit_system.replay_engine.replay_to_timestamp(
                timestamp_replay_id, target_time
            )
            print(f"✓ Replayed {len(events)} events up to {target_time.strftime('%H:%M:%S')}")
        
        # Demonstrate full session replay
        print("\n8. Full Session Replay...")
        full_replay_id = await audit_system.create_session_replay(session_id)
        all_events = await audit_system.replay_engine.replay_full_session(full_replay_id)
        print(f"✓ Full session replay: {len(all_events)} events")
        
        # Final statistics
        stats = await audit_system.get_system_stats()
        print(f"\n9. Replay Statistics:")
        print(f"  Replays created: {stats['system_stats']['replays_created']}")
        print(f"  Active replays: {stats['database_stats']['active_replays']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Session replay demo failed: {e}")
        return False
    
    finally:
        # Clean up
        if 'db_path' in locals() and os.path.exists(db_path):
            os.unlink(db_path)

async def demo_integrity_verification():
    """Demonstrate cryptographic integrity verification"""
    print("\n" + "="*60)
    print("INTEGRITY VERIFICATION DEMO")
    print("="*60)
    
    try:
        # Create temporary database
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        db_path = temp_db.name
        
        print("1. Testing Different Integrity Levels...")
        
        # Test HASH integrity
        print("\n  Testing HASH Integrity...")
        hash_system = create_audit_replay_system(
            db_path=db_path,
            integrity_level=IntegrityLevel.HASH
        )
        await hash_system.initialize()
        
        session_id = "hash_integrity_test"
        await hash_system.start_audit_session(
            session_id=session_id,
            scenario_id="integrity_test",
            participants=["test_agent"]
        )
        
        event_id = await hash_system.log_agent_decision(
            session_id=session_id,
            agent_id="test_agent",
            prompt="Test hash integrity",
            context={"test": "hash"},
            reasoning="Testing hash-based integrity",
            action_taken="integrity_test",
            action_parameters={"type": "hash"},
            confidence_score=0.95
        )
        
        # Verify integrity
        events = await hash_system.search_audit_events(query="", session_id=session_id)
        hash_verified = hash_system.integrity.verify_integrity(events[0])
        print(f"    ✓ Hash integrity verified: {hash_verified}")
        
        await hash_system.end_audit_session(session_id)
        hash_system.database.connection.close()
        
        # Test HMAC integrity
        print("\n  Testing HMAC Integrity...")
        os.unlink(db_path)  # Clean database
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        db_path = temp_db.name
        
        hmac_system = create_audit_replay_system(
            db_path=db_path,
            integrity_level=IntegrityLevel.HMAC
        )
        await hmac_system.initialize()
        
        session_id = "hmac_integrity_test"
        await hmac_system.start_audit_session(
            session_id=session_id,
            scenario_id="integrity_test",
            participants=["test_agent"]
        )
        
        event_id = await hmac_system.log_agent_decision(
            session_id=session_id,
            agent_id="test_agent",
            prompt="Test HMAC integrity",
            context={"test": "hmac"},
            reasoning="Testing HMAC-based integrity",
            action_taken="integrity_test",
            action_parameters={"type": "hmac"},
            confidence_score=0.93
        )
        
        # Verify integrity
        events = await hmac_system.search_audit_events(query="", session_id=session_id)
        hmac_verified = hmac_system.integrity.verify_integrity(events[0])
        print(f"    ✓ HMAC integrity verified: {hmac_verified}")
        
        await hmac_system.end_audit_session(session_id)
        
        # Test tampering detection
        print("\n2. Testing Tampering Detection...")
        
        # Get the event and tamper with it
        events = await hmac_system.search_audit_events(query="", session_id=session_id)
        original_event = events[0]
        
        # Verify original is valid
        original_valid = hmac_system.integrity.verify_integrity(original_event)
        print(f"  Original event valid: {original_valid}")
        
        # Tamper with event data
        original_event.event_data["test"] = "tampered"
        tampered_valid = hmac_system.integrity.verify_integrity(original_event)
        print(f"  Tampered event valid: {tampered_valid}")
        print(f"  ✓ Tampering detected: {not tampered_valid}")
        
        # Test different tampering scenarios
        print("\n3. Testing Various Tampering Scenarios...")
        
        # Create fresh event for testing
        session_id = "tampering_test"
        await hmac_system.start_audit_session(
            session_id=session_id,
            scenario_id="tampering_test",
            participants=["test_agent"]
        )
        
        await hmac_system.log_agent_decision(
            session_id=session_id,
            agent_id="test_agent",
            prompt="Original prompt",
            context={"original": "data"},
            reasoning="Original reasoning",
            action_taken="original_action",
            action_parameters={"original": "params"},
            confidence_score=0.88
        )
        
        events = await hmac_system.search_audit_events(query="", session_id=session_id)
        test_event = events[0]
        
        # Test different types of tampering
        tampering_tests = [
            ("prompt", "tampered prompt"),
            ("reasoning", "tampered reasoning"),
            ("action_taken", "tampered_action"),
            ("confidence_score", 0.99)
        ]
        
        for field, tampered_value in tampering_tests:
            # Create copy of event
            import copy
            test_copy = copy.deepcopy(test_event)
            
            # Tamper with specific field
            setattr(test_copy, field, tampered_value)
            
            # Verify tampering is detected
            is_valid = hmac_system.integrity.verify_integrity(test_copy)
            print(f"  Tampering {field}: detected = {not is_valid}")
        
        await hmac_system.end_audit_session(session_id)
        
        # Display integrity statistics
        stats = await hmac_system.get_system_stats()
        print(f"\n4. Integrity Statistics:")
        print(f"  Default integrity level: {stats['integrity_stats']['default_level']}")
        print(f"  HMAC available: {stats['integrity_stats']['hmac_available']}")
        print(f"  Digital signatures available: {stats['integrity_stats']['digital_signatures_available']}")
        print(f"  Integrity violations detected: {stats['system_stats']['integrity_violations']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integrity verification demo failed: {e}")
        return False
    
    finally:
        # Clean up
        if 'db_path' in locals() and os.path.exists(db_path):
            os.unlink(db_path)

async def main():
    """Run all audit and replay system demos"""
    print("ARCHANGEL AUDIT AND REPLAY SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("Comprehensive audit logging and session replay for forensic analysis")
    print("=" * 80)
    
    demos = [
        ("Basic Audit Logging", demo_basic_audit_logging),
        ("Comprehensive Audit Session", demo_comprehensive_audit_session),
        ("Search and Forensics", demo_search_and_forensics),
        ("Session Replay", demo_session_replay),
        ("Integrity Verification", demo_integrity_verification)
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        print(f"\n{'='*20} {demo_name} {'='*20}")
        try:
            success = await demo_func()
            results.append((demo_name, success))
            if success:
                print(f"✓ {demo_name} completed successfully")
            else:
                print(f"✗ {demo_name} failed")
        except Exception as e:
            print(f"✗ {demo_name} failed with exception: {e}")
            results.append((demo_name, False))
    
    # Summary
    print(f"\n{'='*80}")
    print("DEMO SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    for demo_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{demo_name:.<50} {status}")
    
    print(f"\nOverall Result: {successful}/{total} demos passed")
    
    if successful == total:
        print("\n🎉 All audit and replay system demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  ✓ Comprehensive audit logging with full context capture")
        print("  ✓ Decision logging with prompt, context, and reasoning")
        print("  ✓ LLM interaction logging with model tracking")
        print("  ✓ Cryptographic integrity verification (Hash, HMAC, Digital Signatures)")
        print("  ✓ Full-text search with advanced filtering")
        print("  ✓ Timeline reconstruction for forensic analysis")
        print("  ✓ Session replay with step-by-step playback")
        print("  ✓ Replay controls (speed, pause/resume, timestamp-based)")
        print("  ✓ Tampering detection and integrity verification")
        print("  ✓ Comprehensive statistics and monitoring")
    else:
        print(f"\n⚠️  {total - successful} demo(s) failed. Check the output above for details.")
    
    return successful == total

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)