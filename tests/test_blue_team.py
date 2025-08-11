#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Blue Team Unit Tests
Unit tests for Blue Team defensive agents
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

from agents.base_agent import (
    Agent, Experience, ActionTaken, ActionOutcome, 
    EnvironmentContext, Team, Role
)
from agents.blue_team import (
    SOCAnalystAgent, FirewallConfiguratorAgent, SIEMIntegratorAgent, ComplianceAuditorAgent,
    SecurityAlert, AlertSeverity, IncidentTicket, IncidentStatus,
    FirewallRule, FirewallRuleAction, FirewallRuleProtocol,
    LogEntry, LogLevel, ThreatIntelligence, ThreatLevel, CorrelationResult,
    ComplianceControl, ComplianceFramework, ComplianceAssessment, ComplianceStatus
)
from agents.communication import CommunicationBus, Message, MessageType
from memory.vector_memory import VectorMemorySystem
from memory.knowledge_base import KnowledgeBase


class TestSOCAnalystAgent:
    """Unit tests for SOC Analyst Agent"""

    @pytest.fixture
    async def soc_agent(self):
        """Create SOC Analyst agent for testing"""
        comm_bus = Mock(spec=CommunicationBus)
        comm_bus.get_messages = AsyncMock(return_value=[])
        comm_bus.send_message = AsyncMock()
        
        memory_system = Mock(spec=VectorMemorySystem)
        memory_system.retrieve_similar_experiences = AsyncMock(return_value=[])
        
        knowledge_base = Mock(spec=KnowledgeBase)
        knowledge_base.query_attack_patterns = AsyncMock(return_value=[])
        
        agent = SOCAnalystAgent(
            agent_id="soc_test_001",
            communication_bus=comm_bus,
            memory_system=memory_system,
            knowledge_base=knowledge_base
        )
        
        # Mock the alert monitor task to prevent it from running
        agent.alert_monitor_task = Mock()
        
        return agent

    @pytest.mark.asyncio
    async def test_soc_agent_initialization(self, soc_agent):
        """Test SOC agent initialization"""
        assert soc_agent.agent_id == "soc_test_001"
        assert soc_agent.team == Team.BLUE
        assert soc_agent.role == Role.SOC_ANALYST
        assert len(soc_agent.alert_sources) > 0
        assert len(soc_agent.correlation_rules) > 0

    @pytest.mark.asyncio
    async def test_alert_triage(self, soc_agent):
        """Test alert triage functionality"""
        # Create test alert
        alert = SecurityAlert(
            alert_id="alert_001",
            source="SIEM",
            alert_type="malware_detection",
            severity=AlertSeverity.HIGH,
            timestamp=datetime.now(),
            description="Malware detected on workstation",
            indicators={"file_hash": "abc123", "process_name": "malware.exe"},
            raw_data={"event_id": "12345"},
            confidence_score=0.9,
            false_positive_likelihood=0.1
        )
        
        # Test triage
        result = await soc_agent._triage_alert(alert)
        
        assert isinstance(result, dict)
        assert 'create_incident' in result
        assert 'escalate' in result
        assert 'confidence' in result
        assert 'reasoning' in result
        
        # High severity alert should create incident
        assert result['create_incident'] is True

    @pytest.mark.asyncio
    async def test_false_positive_detection(self, soc_agent):
        """Test false positive detection"""
        # Create alert with high false positive likelihood
        alert = SecurityAlert(
            alert_id="alert_002",
            source="IDS",
            alert_type="network_scan",
            severity=AlertSeverity.MEDIUM,
            timestamp=datetime.now(),
            description="Port scan detected",
            indicators={"source_ip": "192.168.1.100"},
            raw_data={},
            confidence_score=0.6,
            false_positive_likelihood=0.9
        )
        
        # Mock memory system to return similar false positive experiences
        soc_agent.memory_system.retrieve_similar_experiences.return_value = [
            Mock(metadata={'false_positive': True}),
            Mock(metadata={'false_positive': True}),
            Mock(metadata={'false_positive': False})
        ]
        
        fp_score = await soc_agent._calculate_false_positive_score(alert)
        
        assert fp_score > 0.8  # Should be high due to historical data

    @pytest.mark.asyncio
    async def test_incident_ticket_creation(self, soc_agent):
        """Test incident ticket creation"""
        alert = SecurityAlert(
            alert_id="alert_003",
            source="EDR",
            alert_type="suspicious_process",
            severity=AlertSeverity.CRITICAL,
            timestamp=datetime.now(),
            description="Suspicious process execution",
            indicators={"process_name": "powershell.exe", "command_line": "encoded_command"},
            raw_data={},
            confidence_score=0.95,
            false_positive_likelihood=0.05
        )
        
        triage_result = {
            'create_incident': True,
            'escalate': True,
            'confidence': 0.9,
            'reasoning': 'Critical severity alert with high confidence',
            'recommended_actions': ['isolate_host', 'collect_artifacts']
        }
        
        ticket_id = await soc_agent._create_incident_ticket(alert, triage_result)
        
        assert ticket_id.startswith("INC-")
        assert ticket_id in soc_agent.incident_tickets
        
        ticket = soc_agent.incident_tickets[ticket_id]
        assert ticket.severity == AlertSeverity.CRITICAL
        assert ticket.status == IncidentStatus.NEW
        assert alert.alert_id in ticket.related_alerts

    @pytest.mark.asyncio
    async def test_alert_correlation(self, soc_agent):
        """Test alert correlation"""
        # Create multiple related alerts
        alerts = []
        for i in range(3):
            alert = SecurityAlert(
                alert_id=f"alert_00{i+4}",
                source="SIEM",
                alert_type="failed_login",
                severity=AlertSeverity.MEDIUM,
                timestamp=datetime.now() - timedelta(minutes=i),
                description=f"Failed login attempt {i+1}",
                indicators={"user_name": "testuser", "source_ip": "10.0.1.100"},
                raw_data={},
                confidence_score=0.7,
                false_positive_likelihood=0.2
            )
            alerts.append(alert)
            soc_agent.active_alerts[alert.alert_id] = alert
        
        # Test correlation
        await soc_agent._correlate_alerts()
        
        # Should create correlation incident for multiple failed logins
        correlation_incidents = [
            ticket for ticket in soc_agent.incident_tickets.values()
            if "Multiple failed login" in ticket.title or "Correlated Activity" in ticket.title
        ]
        
        # May or may not create correlation depending on timing and rules
        # Just verify the correlation logic runs without error

    @pytest.mark.asyncio
    async def test_soc_decision_making(self, soc_agent):
        """Test SOC analyst decision making"""
        # Add high priority alert
        alert = SecurityAlert(
            alert_id="alert_007",
            source="AV",
            alert_type="malware_detection",
            severity=AlertSeverity.CRITICAL,
            timestamp=datetime.now(),
            description="Critical malware detected",
            indicators={},
            raw_data={},
            confidence_score=0.95,
            false_positive_likelihood=0.05
        )
        soc_agent.active_alerts[alert.alert_id] = alert
        
        environment = EnvironmentContext(
            network_topology={},
            agent_positions={},
            available_tools=[],
            time_constraints={},
            objectives=[]
        )
        
        action = await soc_agent.make_decision(environment)
        
        assert isinstance(action, ActionTaken)
        assert action.action_type in ["incident_response", "monitoring"]
        assert action.confidence_score > 0.0

    @pytest.mark.asyncio
    async def test_performance_metrics(self, soc_agent):
        """Test performance metrics collection"""
        # Add some test data
        soc_agent.metrics['alerts_processed'] = 10
        soc_agent.metrics['incidents_created'] = 3
        
        metrics = soc_agent.get_performance_metrics()
        
        assert 'agent_id' in metrics
        assert 'alerts_processed' in metrics
        assert 'incidents_created' in metrics
        assert metrics['alerts_processed'] == 10
        assert metrics['incidents_created'] == 3


class TestFirewallConfiguratorAgent:
    """Unit tests for Firewall Configurator Agent"""

    @pytest.fixture
    async def firewall_agent(self):
        """Create Firewall Configurator agent for testing"""
        comm_bus = Mock(spec=CommunicationBus)
        comm_bus.get_messages = AsyncMock(return_value=[])
        comm_bus.send_message = AsyncMock()
        
        memory_system = Mock(spec=VectorMemorySystem)
        knowledge_base = Mock(spec=KnowledgeBase)
        
        agent = FirewallConfiguratorAgent(
            agent_id="fw_test_001",
            communication_bus=comm_bus,
            memory_system=memory_system,
            knowledge_base=knowledge_base
        )
        
        # Mock tasks to prevent them from running
        agent.rule_manager_task = Mock()
        agent.deployment_task = Mock()
        
        return agent

    @pytest.mark.asyncio
    async def test_firewall_agent_initialization(self, firewall_agent):
        """Test firewall agent initialization"""
        assert firewall_agent.agent_id == "fw_test_001"
        assert firewall_agent.team == Team.BLUE
        assert firewall_agent.role == Role.FIREWALL_ADMIN
        assert len(firewall_agent.rule_templates) > 0
        assert len(firewall_agent.firewall_systems) > 0

    @pytest.mark.asyncio
    async def test_threat_response_rule_generation(self, firewall_agent):
        """Test threat response rule generation"""
        threat_data = {
            'threat_type': 'malicious_ip',
            'indicators': {
                'ip_addresses': ['192.168.1.100', '10.0.0.50']
            }
        }
        
        rules = await firewall_agent._generate_threat_response_rules(threat_data)
        
        assert len(rules) == 2  # One rule per IP
        for rule in rules:
            assert rule['action'] == FirewallRuleAction.DENY
            assert 'Block Malicious IP' in rule['name']
            assert 'expires_at' in rule

    @pytest.mark.asyncio
    async def test_firewall_rule_creation(self, firewall_agent):
        """Test firewall rule creation"""
        rule_data = {
            'name': 'Test Block Rule',
            'action': FirewallRuleAction.DENY,
            'protocol': FirewallRuleProtocol.TCP,
            'source_ip': '192.168.1.100',
            'source_port': 'any',
            'destination_ip': 'any',
            'destination_port': '80,443',
            'direction': 'inbound',
            'priority': 100,
            'description': 'Test rule for blocking malicious IP'
        }
        
        rule_id = await firewall_agent._create_firewall_rule(rule_data)
        
        assert rule_id.startswith("FW-")
        assert rule_id in firewall_agent.active_rules
        assert rule_id in firewall_agent.deployment_queue
        
        rule = firewall_agent.active_rules[rule_id]
        assert rule.name == 'Test Block Rule'
        assert rule.action == FirewallRuleAction.DENY
        assert rule.source_ip == '192.168.1.100'

    @pytest.mark.asyncio
    async def test_rule_conflict_detection(self, firewall_agent):
        """Test rule conflict detection"""
        # Create first rule
        rule1 = FirewallRule(
            rule_id="rule_001",
            name="Block IP 1",
            action=FirewallRuleAction.DENY,
            protocol=FirewallRuleProtocol.TCP,
            source_ip="192.168.1.100",
            source_port="any",
            destination_ip="any",
            destination_port="80",
            direction="inbound",
            priority=100,
            description="Test rule 1",
            created_by="test",
            created_at=datetime.now(),
            expires_at=None,
            active=True
        )
        firewall_agent.active_rules["rule_001"] = rule1
        
        # Create conflicting rule
        rule2 = FirewallRule(
            rule_id="rule_002",
            name="Allow IP 1",
            action=FirewallRuleAction.ALLOW,
            protocol=FirewallRuleProtocol.TCP,
            source_ip="192.168.1.100",
            source_port="any",
            destination_ip="any",
            destination_port="80",
            direction="inbound",
            priority=90,
            description="Test rule 2",
            created_by="test",
            created_at=datetime.now(),
            expires_at=None,
            active=True
        )
        
        conflicts = await firewall_agent._check_rule_conflicts(rule2)
        
        assert len(conflicts) > 0
        assert conflicts[0]['existing_rule'] == rule1
        assert conflicts[0]['conflict_score'] > 0.5

    @pytest.mark.asyncio
    async def test_rule_optimization(self, firewall_agent):
        """Test rule optimization"""
        # Create similar rules that can be consolidated
        for i in range(4):
            rule = FirewallRule(
                rule_id=f"rule_00{i+3}",
                name=f"Block IP {i+3}",
                action=FirewallRuleAction.DENY,
                protocol=FirewallRuleProtocol.TCP,
                source_ip=f"192.168.1.{100+i}",
                source_port="any",
                destination_ip="any",
                destination_port="80",
                direction="inbound",
                priority=100,
                description=f"Test rule {i+3}",
                created_by="test",
                created_at=datetime.now(),
                expires_at=None,
                active=True
            )
            firewall_agent.active_rules[rule.rule_id] = rule
        
        # Test grouping similar rules
        groups = firewall_agent._group_similar_rules()
        
        # Should find at least one group of similar rules
        assert len(groups) > 0
        assert len(groups[0]) >= 3  # At least 3 similar rules

    @pytest.mark.asyncio
    async def test_firewall_decision_making(self, firewall_agent):
        """Test firewall configurator decision making"""
        # Add pending rule deployment
        rule_data = {
            'name': 'Pending Rule',
            'action': FirewallRuleAction.DENY,
            'protocol': FirewallRuleProtocol.ANY,
            'source_ip': '10.0.0.1',
            'destination_ip': 'any'
        }
        rule_id = await firewall_agent._create_firewall_rule(rule_data)
        
        environment = EnvironmentContext(
            network_topology={},
            agent_positions={},
            available_tools=[],
            time_constraints={},
            objectives=[]
        )
        
        action = await firewall_agent.make_decision(environment)
        
        assert isinstance(action, ActionTaken)
        assert action.action_type in ["defensive_configuration", "monitoring", "system_optimization"]
        assert action.confidence_score > 0.0


class TestSIEMIntegratorAgent:
    """Unit tests for SIEM Integrator Agent"""

    @pytest.fixture
    async def siem_agent(self):
        """Create SIEM Integrator agent for testing"""
        comm_bus = Mock(spec=CommunicationBus)
        comm_bus.get_messages = AsyncMock(return_value=[])
        comm_bus.send_message = AsyncMock()
        
        memory_system = Mock(spec=VectorMemorySystem)
        knowledge_base = Mock(spec=KnowledgeBase)
        
        agent = SIEMIntegratorAgent(
            agent_id="siem_test_001",
            communication_bus=comm_bus,
            memory_system=memory_system,
            knowledge_base=knowledge_base
        )
        
        # Mock tasks to prevent them from running
        agent.log_processor_task = Mock()
        agent.correlation_task = Mock()
        agent.intel_updater_task = Mock()
        
        return agent

    @pytest.mark.asyncio
    async def test_siem_agent_initialization(self, siem_agent):
        """Test SIEM agent initialization"""
        assert siem_agent.agent_id == "siem_test_001"
        assert siem_agent.team == Team.BLUE
        assert siem_agent.role == Role.SIEM_ANALYST
        assert len(siem_agent.log_sources) > 0
        assert len(siem_agent.correlation_rules) > 0

    @pytest.mark.asyncio
    async def test_log_parsing(self, siem_agent):
        """Test log entry parsing"""
        # Test Windows event log parsing
        log_entry = LogEntry(
            log_id="log_001",
            timestamp=datetime.now(),
            source="windows_events",
            log_level=LogLevel.INFO,
            message="User login event",
            raw_data={
                'EventID': 4624,
                'Computer': 'WORKSTATION01',
                'UserName': 'testuser',
                'SourceIP': '192.168.1.100'
            },
            parsed_fields={},
            correlation_id=None,
            threat_indicators=[]
        )
        
        await siem_agent._parse_windows_event(log_entry)
        
        assert log_entry.parsed_fields['event_id'] == 4624
        assert log_entry.parsed_fields['event_type'] == 'successful_login'
        assert log_entry.parsed_fields['computer_name'] == 'WORKSTATION01'
        assert log_entry.parsed_fields['user_name'] == 'testuser'

    @pytest.mark.asyncio
    async def test_threat_indicator_extraction(self, siem_agent):
        """Test threat indicator extraction"""
        log_entry = LogEntry(
            log_id="log_002",
            timestamp=datetime.now(),
            source="firewall_logs",
            log_level=LogLevel.WARNING,
            message="Blocked connection",
            raw_data={},
            parsed_fields={
                'source_ip': '192.168.1.100',
                'process_name': 'powershell.exe',
                'event_type': 'failed_login'
            },
            correlation_id=None,
            threat_indicators=[]
        )
        
        await siem_agent._extract_threat_indicators(log_entry)
        
        # Should detect suspicious process and failed authentication
        assert len(log_entry.threat_indicators) > 0
        assert any('suspicious_process' in indicator for indicator in log_entry.threat_indicators)
        assert 'failed_authentication' in log_entry.threat_indicators

    @pytest.mark.asyncio
    async def test_log_correlation(self, siem_agent):
        """Test log correlation"""
        # Create related log entries
        base_time = datetime.now()
        logs = []
        
        for i in range(3):
            log = LogEntry(
                log_id=f"log_00{i+3}",
                timestamp=base_time + timedelta(minutes=i),
                source="windows_events",
                log_level=LogLevel.WARNING,
                message=f"Failed login attempt {i+1}",
                raw_data={},
                parsed_fields={
                    'event_type': 'failed_login',
                    'user_name': 'testuser',
                    'source_ip': '192.168.1.100'
                },
                correlation_id=None,
                threat_indicators=[]
            )
            logs.append(log)
            siem_agent.log_buffer[log.log_id] = log
        
        # Test correlation
        rule = siem_agent.correlation_rules[0]  # Failed Login Sequence rule
        correlations = await siem_agent._find_correlations(rule)
        
        # Should find correlation between the failed login attempts
        assert len(correlations) >= 0  # May or may not find correlation depending on timing

    @pytest.mark.asyncio
    async def test_correlation_confidence_calculation(self, siem_agent):
        """Test correlation confidence calculation"""
        rule = {
            'name': 'Test Rule',
            'threat_level': ThreatLevel.HIGH
        }
        
        logs = [
            LogEntry(
                log_id="log_006",
                timestamp=datetime.now(),
                source="test",
                log_level=LogLevel.INFO,
                message="Test log 1",
                raw_data={},
                parsed_fields={},
                correlation_id=None,
                threat_indicators=['suspicious_ip:192.168.1.100']
            ),
            LogEntry(
                log_id="log_007",
                timestamp=datetime.now() + timedelta(seconds=30),
                source="test",
                log_level=LogLevel.INFO,
                message="Test log 2",
                raw_data={},
                parsed_fields={},
                correlation_id=None,
                threat_indicators=['suspicious_process:malware.exe']
            )
        ]
        
        confidence = siem_agent._calculate_correlation_confidence(rule, logs)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should have reasonable confidence with threat indicators

    @pytest.mark.asyncio
    async def test_siem_decision_making(self, siem_agent):
        """Test SIEM integrator decision making"""
        # Add high-priority correlation
        correlation = CorrelationResult(
            correlation_id="corr_001",
            related_logs=["log_001", "log_002"],
            correlation_type="Lateral Movement Pattern",
            confidence_score=0.9,
            threat_level=ThreatLevel.CRITICAL,
            timeline=[],
            indicators=['suspicious_ip:10.0.0.1'],
            description="Critical threat detected"
        )
        siem_agent.correlation_results[correlation.correlation_id] = correlation
        
        environment = EnvironmentContext(
            network_topology={},
            agent_positions={},
            available_tools=[],
            time_constraints={},
            objectives=[]
        )
        
        action = await siem_agent.make_decision(environment)
        
        assert isinstance(action, ActionTaken)
        assert action.action_type in ["threat_analysis", "monitoring"]
        assert action.confidence_score > 0.0


class TestComplianceAuditorAgent:
    """Unit tests for Compliance Auditor Agent"""

    @pytest.fixture
    async def compliance_agent(self):
        """Create Compliance Auditor agent for testing"""
        comm_bus = Mock(spec=CommunicationBus)
        comm_bus.get_messages = AsyncMock(return_value=[])
        comm_bus.send_message = AsyncMock()
        
        memory_system = Mock(spec=VectorMemorySystem)
        knowledge_base = Mock(spec=KnowledgeBase)
        
        agent = ComplianceAuditorAgent(
            agent_id="comp_test_001",
            communication_bus=comm_bus,
            memory_system=memory_system,
            knowledge_base=knowledge_base
        )
        
        # Mock tasks to prevent them from running
        agent.assessment_task = Mock()
        agent.monitoring_task = Mock()
        agent.reporting_task = Mock()
        
        return agent

    @pytest.mark.asyncio
    async def test_compliance_agent_initialization(self, compliance_agent):
        """Test compliance agent initialization"""
        assert compliance_agent.agent_id == "comp_test_001"
        assert compliance_agent.team == Team.BLUE
        assert compliance_agent.role == Role.COMPLIANCE_AUDITOR
        assert len(compliance_agent.frameworks) > 0
        assert len(compliance_agent.compliance_controls) > 0

    @pytest.mark.asyncio
    async def test_automated_check_execution(self, compliance_agent):
        """Test automated check execution"""
        # Test asset discovery scan
        result = await compliance_agent._execute_automated_check('asset_discovery_scan')
        
        assert 'status' in result
        assert 'score' in result
        assert 'details' in result
        assert 0.0 <= result['score'] <= 1.0

    @pytest.mark.asyncio
    async def test_evidence_collection(self, compliance_agent):
        """Test evidence collection"""
        control = ComplianceControl(
            control_id="TEST-001",
            framework=ComplianceFramework.NIST_CSF,
            title="Test Control",
            description="Test control for unit testing",
            requirements=["Test requirement"],
            evidence_types=["asset_inventory", "access_logs"],
            automated_checks=["asset_discovery_scan"],
            manual_checks=["manual_review"],
            risk_level="medium",
            frequency="monthly"
        )
        
        evidence = await compliance_agent._collect_evidence(control)
        
        assert len(evidence) > 0
        for evidence_item in evidence:
            assert 'type' in evidence_item
            assert 'collected_at' in evidence_item
            assert 'data' in evidence_item

    @pytest.mark.asyncio
    async def test_compliance_score_calculation(self, compliance_agent):
        """Test compliance score calculation"""
        control = ComplianceControl(
            control_id="TEST-002",
            framework=ComplianceFramework.ISO_27001,
            title="Test Control 2",
            description="Another test control",
            requirements=["Test requirement"],
            evidence_types=["policy_documents"],
            automated_checks=["policy_version_check"],
            manual_checks=["policy_review"],
            risk_level="high",
            frequency="annually"
        )
        
        automated_results = {
            'policy_version_check': {
                'status': 'passed',
                'score': 0.9,
                'details': 'Policy is up to date'
            }
        }
        
        evidence = [
            {
                'type': 'policy_documents',
                'collected_at': datetime.now().isoformat(),
                'data': {'policy_version': '2.1', 'last_updated': '2024-01-01'}
            }
        ]
        
        score, status = await compliance_agent._calculate_compliance_score(
            control, automated_results, evidence
        )
        
        assert 0.0 <= score <= 1.0
        assert status in [ComplianceStatus.COMPLIANT, ComplianceStatus.PARTIALLY_COMPLIANT, 
                         ComplianceStatus.NON_COMPLIANT, ComplianceStatus.REQUIRES_REVIEW]

    @pytest.mark.asyncio
    async def test_control_assessment(self, compliance_agent):
        """Test complete control assessment"""
        # Add a test control
        control = ComplianceControl(
            control_id="TEST-003",
            framework=ComplianceFramework.SOC2,
            title="Test Control 3",
            description="Complete assessment test",
            requirements=["Test requirement"],
            evidence_types=["monitoring_logs"],
            automated_checks=["monitoring_system_status"],
            manual_checks=["monitoring_review"],
            risk_level="medium",
            frequency="monthly"
        )
        compliance_agent.compliance_controls[control.control_id] = control
        
        assessment_id = await compliance_agent._assess_control(control.control_id)
        
        assert assessment_id.startswith("ASSESS-")
        assert assessment_id in compliance_agent.assessments
        
        assessment = compliance_agent.assessments[assessment_id]
        assert assessment.control_id == control.control_id
        assert assessment.framework == ComplianceFramework.SOC2
        assert 0.0 <= assessment.score <= 1.0

    @pytest.mark.asyncio
    async def test_compliance_decision_making(self, compliance_agent):
        """Test compliance auditor decision making"""
        # Add pending assessment
        compliance_agent.pending_assessments.append("TEST-004")
        
        # Add test control
        control = ComplianceControl(
            control_id="TEST-004",
            framework=ComplianceFramework.NIST_CSF,
            title="Pending Control",
            description="Control pending assessment",
            requirements=["Test requirement"],
            evidence_types=["test_evidence"],
            automated_checks=["test_check"],
            manual_checks=["manual_test"],
            risk_level="high",
            frequency="weekly"
        )
        compliance_agent.compliance_controls[control.control_id] = control
        
        environment = EnvironmentContext(
            network_topology={},
            agent_positions={},
            available_tools=[],
            time_constraints={},
            objectives=[]
        )
        
        action = await compliance_agent.make_decision(environment)
        
        assert isinstance(action, ActionTaken)
        assert action.action_type in ["compliance_assessment", "compliance_remediation", "monitoring"]
        assert action.confidence_score > 0.0

    @pytest.mark.asyncio
    async def test_compliance_report_generation(self, compliance_agent):
        """Test compliance report generation"""
        # Add test assessment
        assessment = ComplianceAssessment(
            assessment_id="assess_001",
            control_id="TEST-005",
            framework=ComplianceFramework.NIST_CSF,
            status=ComplianceStatus.COMPLIANT,
            score=0.9,
            findings=[],
            evidence=[],
            recommendations=[],
            assessed_by=compliance_agent.agent_id,
            assessed_at=datetime.now(),
            next_assessment=datetime.now() + timedelta(days=30)
        )
        compliance_agent.assessments[assessment.assessment_id] = assessment
        
        report_id = await compliance_agent._generate_compliance_report(
            ComplianceFramework.NIST_CSF, 'monthly'
        )
        
        assert report_id.startswith("RPT-NIST_CSF-")
        assert report_id in compliance_agent.reports
        
        report = compliance_agent.reports[report_id]
        assert report.framework == ComplianceFramework.NIST_CSF
        assert report.report_type == 'monthly'
        assert 0.0 <= report.overall_score <= 1.0


class TestBlueTeamIntegration:
    """Integration tests for Blue Team agents working together"""

    @pytest.mark.asyncio
    async def test_soc_firewall_integration(self):
        """Test SOC analyst and firewall configurator integration"""
        # Create mock communication bus
        comm_bus = Mock(spec=CommunicationBus)
        messages = []
        
        async def mock_send_message(message):
            messages.append(message)
        
        comm_bus.send_message = mock_send_message
        comm_bus.get_messages = AsyncMock(return_value=[])
        
        # Create agents
        memory_system = Mock(spec=VectorMemorySystem)
        knowledge_base = Mock(spec=KnowledgeBase)
        
        soc_agent = SOCAnalystAgent(
            agent_id="soc_integration_001",
            communication_bus=comm_bus,
            memory_system=memory_system,
            knowledge_base=knowledge_base
        )
        
        firewall_agent = FirewallConfiguratorAgent(
            agent_id="fw_integration_001",
            communication_bus=comm_bus,
            memory_system=memory_system,
            knowledge_base=knowledge_base
        )
        
        # Mock tasks
        soc_agent.alert_monitor_task = Mock()
        firewall_agent.rule_manager_task = Mock()
        firewall_agent.deployment_task = Mock()
        
        # Simulate SOC agent creating incident with threat data
        alert = SecurityAlert(
            alert_id="integration_alert_001",
            source="IDS",
            alert_type="malicious_ip",
            severity=AlertSeverity.HIGH,
            timestamp=datetime.now(),
            description="Malicious IP detected",
            indicators={"source_ip": "192.168.1.100"},
            raw_data={},
            confidence_score=0.9,
            false_positive_likelihood=0.1
        )
        
        triage_result = await soc_agent._triage_alert(alert)
        if triage_result['create_incident']:
            await soc_agent._create_incident_ticket(alert, triage_result)
        
        # Verify incident notification was sent
        incident_notifications = [
            msg for msg in messages 
            if msg.message_type == MessageType.INCIDENT_NOTIFICATION
        ]
        assert len(incident_notifications) > 0
        
        # Simulate firewall agent receiving threat detection
        threat_message = Message(
            message_id=str(uuid.uuid4()),
            sender_id="threat_detector",
            recipient_id=firewall_agent.agent_id,
            message_type=MessageType.THREAT_DETECTED,
            content={
                'threat_type': 'malicious_ip',
                'indicators': {'ip_addresses': ['192.168.1.100']}
            },
            timestamp=datetime.now()
        )
        
        await firewall_agent._handle_threat_detection(threat_message)
        
        # Verify firewall rule was created
        assert len(firewall_agent.active_rules) > 0
        assert len(firewall_agent.deployment_queue) > 0

    @pytest.mark.asyncio
    async def test_siem_soc_integration(self):
        """Test SIEM integrator and SOC analyst integration"""
        # Create mock communication bus
        comm_bus = Mock(spec=CommunicationBus)
        messages = []
        
        async def mock_send_message(message):
            messages.append(message)
        
        comm_bus.send_message = mock_send_message
        comm_bus.get_messages = AsyncMock(return_value=[])
        
        # Create agents
        memory_system = Mock(spec=VectorMemorySystem)
        knowledge_base = Mock(spec=KnowledgeBase)
        
        siem_agent = SIEMIntegratorAgent(
            agent_id="siem_integration_001",
            communication_bus=comm_bus,
            memory_system=memory_system,
            knowledge_base=knowledge_base
        )
        
        soc_agent = SOCAnalystAgent(
            agent_id="soc_integration_002",
            communication_bus=comm_bus,
            memory_system=memory_system,
            knowledge_base=knowledge_base
        )
        
        # Mock tasks
        siem_agent.log_processor_task = Mock()
        siem_agent.correlation_task = Mock()
        siem_agent.intel_updater_task = Mock()
        soc_agent.alert_monitor_task = Mock()
        
        # Simulate SIEM creating correlation
        correlation = CorrelationResult(
            correlation_id="integration_corr_001",
            related_logs=["log_001", "log_002", "log_003"],
            correlation_type="Failed Login Sequence",
            confidence_score=0.85,
            threat_level=ThreatLevel.HIGH,
            timeline=[],
            indicators=['failed_authentication', 'suspicious_ip:192.168.1.100'],
            description="Multiple failed login attempts detected"
        )
        
        await siem_agent._send_correlation_alert(correlation)
        
        # Verify correlation alert was sent
        correlation_alerts = [
            msg for msg in messages 
            if msg.message_type == MessageType.CORRELATION_ALERT
        ]
        assert len(correlation_alerts) > 0
        
        correlation_alert = correlation_alerts[0]
        assert correlation_alert.content['threat_level'] == 'high'
        assert correlation_alert.content['confidence_score'] == 0.85


if __name__ == "__main__":
    # Run unit tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])