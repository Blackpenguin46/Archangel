#!/usr/bin/env python3
"""
Tests for Red Team agents
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

from agents.red_team import (
    ReconAgent, ExploitAgent, PersistenceAgent, ExfiltrationAgent,
    NetworkTarget, ReconResult, ExploitPayload, ExploitResult,
    PersistenceMechanism, EvasionTechnique, DataTarget, ExfiltrationChannel
)
from agents.base_agent import AgentConfig, Team, Role, EnvironmentState


class TestReconAgent:
    """Test cases for ReconAgent"""
    
    @pytest.fixture
    def recon_config(self):
        """Create test configuration for ReconAgent"""
        return AgentConfig(
            agent_id="recon_001",
            team=Team.RED,
            role=Role.RECON,
            name="Test Recon Agent",
            description="Test reconnaissance agent",
            llm_model="gpt-4-turbo",
            max_memory_size=100,
            decision_timeout=10.0,
            tools=["nmap", "masscan", "service_scan"],
            constraints=["no_destructive_actions"],
            objectives=["discover_network_topology", "identify_vulnerabilities"]
        )
    
    @pytest.fixture
    def mock_environment(self):
        """Create mock environment state"""
        return EnvironmentState(
            timestamp=datetime.now(),
            network_topology={"subnets": ["192.168.1.0/24"]},
            active_services=[
                {"ip": "192.168.1.10", "port": 80, "service": "http"},
                {"ip": "192.168.1.20", "port": 22, "service": "ssh"}
            ],
            security_alerts=[],
            system_logs=[],
            agent_positions={},
            threat_level="low"
        )
    
    @pytest_asyncio.fixture
    async def recon_agent(self, recon_config):
        """Create and initialize ReconAgent"""
        agent = ReconAgent(recon_config)
        await agent.initialize()
        return agent
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, recon_config):
        """Test ReconAgent initialization"""
        agent = ReconAgent(recon_config)
        
        assert agent.agent_id == "recon_001"
        assert agent.team == Team.RED
        assert agent.role == Role.RECON
        assert agent.name == "Test Recon Agent"
        assert len(agent.scan_techniques) > 0
        assert "nmap" in agent.tools
        
        await agent.initialize()
        assert agent.status.value == "active"
    
    @pytest.mark.asyncio
    async def test_situation_reasoning(self, recon_agent, mock_environment):
        """Test situation reasoning capabilities"""
        reasoning = await recon_agent.reason_about_situation(mock_environment)
        
        assert reasoning is not None
        assert reasoning.situation_assessment is not None
        assert reasoning.threat_analysis is not None
        assert len(reasoning.opportunity_identification) > 0
        assert reasoning.confidence_score > 0.0
        assert len(reasoning.reasoning_chain) > 0
        
        # Should identify reconnaissance opportunities
        opportunities = reasoning.opportunity_identification
        assert any(opp in ["network_discovery", "target_enumeration", "vulnerability_assessment"] 
                  for opp in opportunities)
    
    @pytest.mark.asyncio
    async def test_action_planning(self, recon_agent, mock_environment):
        """Test action planning based on reasoning"""
        reasoning = await recon_agent.reason_about_situation(mock_environment)
        action_plan = await recon_agent.plan_actions(reasoning)
        
        assert action_plan is not None
        assert action_plan.primary_action is not None
        assert action_plan.action_type == "reconnaissance"
        assert action_plan.target is not None
        assert len(action_plan.success_criteria) > 0
        assert action_plan.estimated_duration > 0
        assert action_plan.risk_level in ["low", "medium", "high"]
    
    @pytest.mark.asyncio
    async def test_network_discovery_execution(self, recon_agent):
        """Test network discovery action execution"""
        action_plan = Mock()
        action_plan.primary_action = "network_discovery"
        action_plan.action_type = "reconnaissance"
        action_plan.target = "192.168.1.0/24"
        action_plan.parameters = {
            "target_range": "192.168.1.0/24",
            "scan_type": "syn_scan"
        }
        
        result = await recon_agent.execute_action(action_plan)
        
        assert result is not None
        assert result.action_type == "reconnaissance"
        assert isinstance(result.success, bool)
        assert result.duration >= 0
        assert "hosts_discovered" in result.data
        assert len(result.data["hosts_discovered"]) >= 0
    
    @pytest.mark.asyncio
    async def test_target_enumeration_execution(self, recon_agent):
        """Test target enumeration execution"""
        # Add some discovered targets first
        recon_agent.discovered_targets = [
            {"ip": "192.168.1.10", "hostname": "server1"},
            {"ip": "192.168.1.20", "hostname": "server2"}
        ]
        
        action_plan = Mock()
        action_plan.primary_action = "target_enumeration"
        action_plan.action_type = "reconnaissance"
        action_plan.target = "discovered_hosts"
        action_plan.parameters = {
            "targets": recon_agent.discovered_targets,
            "depth": "standard"
        }
        
        result = await recon_agent.execute_action(action_plan)
        
        assert result is not None
        assert result.success is not None
        assert "enumeration_results" in result.data
        assert "targets_scanned" in result.data
    
    @pytest.mark.asyncio
    async def test_service_scanning_execution(self, recon_agent):
        """Test service scanning execution"""
        action_plan = Mock()
        action_plan.primary_action = "service_scanning"
        action_plan.action_type = "reconnaissance"
        action_plan.target = "192.168.1.10"
        action_plan.parameters = {
            "target": "192.168.1.10",
            "port_range": "1-1000",
            "scan_type": "service_detection"
        }
        
        result = await recon_agent.execute_action(action_plan)
        
        assert result is not None
        assert result.success is not None
        assert "services_found" in result.data
        assert isinstance(result.data["services_found"], list)
    
    @pytest.mark.asyncio
    async def test_vulnerability_assessment_execution(self, recon_agent):
        """Test vulnerability assessment execution"""
        # Add some targets with services
        recon_agent.discovered_targets = [
            {"ip": "192.168.1.10", "services": [{"port": 80, "service": "http"}]}
        ]
        
        action_plan = Mock()
        action_plan.primary_action = "vulnerability_assessment"
        action_plan.action_type = "reconnaissance"
        action_plan.target = "multiple_targets"
        action_plan.parameters = {
            "targets": recon_agent.discovered_targets,
            "scan_type": "vulnerability_assessment"
        }
        
        result = await recon_agent.execute_action(action_plan)
        
        assert result is not None
        assert result.success is not None
        assert "vulnerabilities_found" in result.data
        assert isinstance(result.data["vulnerabilities_found"], list)
    
    @pytest.mark.asyncio
    async def test_learning_from_outcome(self, recon_agent):
        """Test learning from action outcomes"""
        action_plan = Mock()
        action_plan.primary_action = "network_discovery"
        action_plan.action_type = "reconnaissance"
        action_plan.target = "192.168.1.0/24"
        action_plan.parameters = {}
        
        action_result = Mock()
        action_result.success = True
        action_result.confidence = 0.8
        action_result.data = {"hosts_discovered": [{"ip": "192.168.1.10"}]}
        
        initial_experience_count = len(recon_agent.experiences)
        await recon_agent.learn_from_outcome(action_plan, action_result)
        
        assert len(recon_agent.experiences) == initial_experience_count + 1
        assert recon_agent.performance_metrics["actions_taken"] > 0
        if action_result.success:
            assert recon_agent.performance_metrics["successful_actions"] > 0
    
    @pytest.mark.asyncio
    async def test_network_coverage_calculation(self, recon_agent):
        """Test network coverage calculation"""
        # Empty targets
        coverage = recon_agent._calculate_network_coverage(Mock())
        assert coverage == 0.0
        
        # Add some targets
        recon_agent.discovered_targets = [{"ip": f"192.168.1.{i}"} for i in range(10)]
        coverage = recon_agent._calculate_network_coverage(Mock())
        assert 0.0 < coverage <= 1.0
    
    @pytest.mark.asyncio
    async def test_detection_risk_calculation(self, recon_agent):
        """Test detection risk calculation"""
        # Low alert environment
        low_alert_env = Mock()
        low_alert_env.security_alerts = []
        risk = recon_agent._calculate_detection_risk(low_alert_env)
        assert 0.0 <= risk <= 1.0
        
        # High alert environment
        high_alert_env = Mock()
        high_alert_env.security_alerts = [Mock() for _ in range(10)]
        high_risk = recon_agent._calculate_detection_risk(high_alert_env)
        assert high_risk > risk
    
    @pytest.mark.asyncio
    async def test_action_success_evaluation(self, recon_agent):
        """Test action success evaluation"""
        # Network discovery success
        action_plan = Mock()
        action_plan.primary_action = "network_discovery"
        
        success_data = {"hosts_discovered": [{"ip": "192.168.1.10"}, {"ip": "192.168.1.20"}, {"ip": "192.168.1.30"}]}
        assert recon_agent._evaluate_action_success(action_plan, success_data) == True
        
        failure_data = {"hosts_discovered": []}
        assert recon_agent._evaluate_action_success(action_plan, failure_data) == False
    
    @pytest.mark.asyncio
    async def test_reconnaissance_state_update(self, recon_agent):
        """Test internal state updates from reconnaissance results"""
        initial_target_count = len(recon_agent.discovered_targets)
        
        result_data = {
            "hosts_discovered": [
                {"ip": "192.168.1.100", "hostname": "new-host"},
                {"ip": "192.168.1.101", "hostname": "another-host"}
            ],
            "vulnerabilities_found": [
                {"cve": "CVE-2021-44228", "severity": "critical"}
            ]
        }
        
        await recon_agent._update_reconnaissance_state(result_data)
        
        assert len(recon_agent.discovered_targets) > initial_target_count
        assert len(recon_agent.vulnerability_database) > 0
        assert "CVE-2021-44228" in recon_agent.vulnerability_database
    
    @pytest.mark.asyncio
    async def test_simulation_methods(self, recon_agent):
        """Test reconnaissance simulation methods"""
        # Test network scan simulation
        hosts = await recon_agent._simulate_network_scan("192.168.1.0/24", "syn_scan")
        assert isinstance(hosts, list)
        assert len(hosts) > 0
        assert all("ip" in host for host in hosts)
        
        # Test service scan simulation
        services = await recon_agent._simulate_service_scan("192.168.1.10", "1-1000")
        assert isinstance(services, list)
        assert all("port" in service and "service" in service for service in services)
        
        # Test vulnerability assessment simulation
        vulns = await recon_agent._assess_target_vulnerabilities("192.168.1.10")
        assert isinstance(vulns, list)
        # May be empty, but should be a valid list
    
    @pytest.mark.asyncio
    async def test_agent_status_reporting(self, recon_agent):
        """Test agent status reporting"""
        status = await recon_agent.get_status()
        
        assert status["agent_id"] == "recon_001"
        assert status["team"] == "red"
        assert status["role"] == "recon"
        assert status["status"] == "active"
        assert "performance_metrics" in status
        assert "memory_size" in status
    
    @pytest.mark.asyncio
    async def test_error_handling(self, recon_agent):
        """Test error handling in action execution"""
        # Create an action that should cause an error
        invalid_action = Mock()
        invalid_action.primary_action = "invalid_action"
        invalid_action.action_type = "reconnaissance"
        invalid_action.target = "invalid_target"
        invalid_action.parameters = {}
        
        result = await recon_agent.execute_action(invalid_action)
        
        assert result is not None
        assert result.success == False
        assert len(result.errors) > 0
        assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_recon_agent_integration():
    """Integration test for complete ReconAgent workflow"""
    config = AgentConfig(
        agent_id="integration_recon",
        team=Team.RED,
        role=Role.RECON,
        name="Integration Test Agent",
        description="Integration test agent",
        tools=["nmap", "service_scan"]
    )
    
    agent = ReconAgent(config)
    await agent.initialize()
    
    # Create mock environment
    env_state = EnvironmentState(
        timestamp=datetime.now(),
        network_topology={},
        active_services=[],
        security_alerts=[],
        system_logs=[],
        agent_positions={},
        threat_level="low"
    )
    
    # Complete workflow: perceive -> reason -> plan -> execute -> learn
    perceived_state = await agent.perceive_environment()
    reasoning = await agent.reason_about_situation(perceived_state)
    action_plan = await agent.plan_actions(reasoning)
    result = await agent.execute_action(action_plan)
    await agent.learn_from_outcome(action_plan, result)
    
    # Verify workflow completed
    assert reasoning is not None
    assert action_plan is not None
    assert result is not None
    assert len(agent.experiences) > 0
    
    # Check final status
    status = await agent.get_status()
    assert status["performance_metrics"]["actions_taken"] > 0


class TestExploitAgent:
    """Test cases for ExploitAgent"""
    
    @pytest.fixture
    def exploit_config(self):
        """Create test configuration for ExploitAgent"""
        return AgentConfig(
            agent_id="exploit_001",
            team=Team.RED,
            role=Role.EXPLOIT,
            name="Test Exploit Agent",
            description="Test exploitation agent",
            llm_model="gpt-4-turbo",
            max_memory_size=100,
            decision_timeout=10.0,
            tools=["metasploit", "sqlmap", "custom_exploit"],
            constraints=["no_destructive_actions"],
            objectives=["gain_access", "establish_persistence"]
        )
    
    @pytest.fixture
    def mock_vulnerable_environment(self):
        """Create mock environment with vulnerabilities"""
        return EnvironmentState(
            timestamp=datetime.now(),
            network_topology={"subnets": ["192.168.1.0/24"]},
            active_services=[
                {"ip": "192.168.1.10", "port": 80, "service": "http", "version": "Apache 2.4.41"},
                {"ip": "192.168.1.20", "port": 22, "service": "ssh", "version": "OpenSSH 7.4"},
                {"ip": "192.168.1.30", "port": 3306, "service": "mysql", "version": "MySQL 5.7"}
            ],
            security_alerts=[],
            system_logs=[],
            agent_positions={},
            threat_level="low"
        )
    
    @pytest_asyncio.fixture
    async def exploit_agent(self, exploit_config):
        """Create and initialize ExploitAgent"""
        agent = ExploitAgent(exploit_config)
        await agent.initialize()
        return agent
    
    @pytest.mark.asyncio
    async def test_exploit_agent_initialization(self, exploit_config):
        """Test ExploitAgent initialization"""
        agent = ExploitAgent(exploit_config)
        
        assert agent.agent_id == "exploit_001"
        assert agent.team == Team.RED
        assert agent.role == Role.EXPLOIT
        assert agent.name == "Test Exploit Agent"
        assert len(agent.exploit_techniques) > 0
        assert "metasploit" in agent.tools
        assert len(agent.payloads) > 0
        
        await agent.initialize()
        assert agent.status.value == "active"
    
    @pytest.mark.asyncio
    async def test_exploitation_situation_reasoning(self, exploit_agent, mock_vulnerable_environment):
        """Test exploitation situation reasoning"""
        reasoning = await exploit_agent.reason_about_situation(mock_vulnerable_environment)
        
        assert reasoning is not None
        assert reasoning.situation_assessment is not None
        assert reasoning.threat_analysis is not None
        assert len(reasoning.opportunity_identification) > 0
        assert reasoning.confidence_score > 0.0
        assert len(reasoning.reasoning_chain) > 0
        
        # Should identify exploitation opportunities
        opportunities = reasoning.opportunity_identification
        assert any(opp in ["vulnerability_exploitation", "web_application_attack", "network_service_exploit"] 
                  for opp in opportunities)
    
    @pytest.mark.asyncio
    async def test_exploitation_action_planning(self, exploit_agent, mock_vulnerable_environment):
        """Test exploitation action planning"""
        reasoning = await exploit_agent.reason_about_situation(mock_vulnerable_environment)
        action_plan = await exploit_agent.plan_actions(reasoning)
        
        assert action_plan is not None
        assert action_plan.primary_action is not None
        assert action_plan.action_type == "exploitation"
        assert action_plan.target is not None
        assert len(action_plan.success_criteria) > 0
        assert action_plan.estimated_duration > 0
        assert action_plan.risk_level in ["low", "medium", "high"]
    
    @pytest.mark.asyncio
    async def test_vulnerability_exploitation_execution(self, exploit_agent):
        """Test vulnerability exploitation execution"""
        action_plan = Mock()
        action_plan.primary_action = "vulnerability_exploitation"
        action_plan.action_type = "exploitation"
        action_plan.target = "vulnerable_service"
        action_plan.parameters = {
            "target": "192.168.1.10",
            "vulnerability": "buffer_overflow",
            "payload_type": "reverse_shell"
        }
        
        result = await exploit_agent.execute_action(action_plan)
        
        assert result is not None
        assert result.action_type == "exploitation"
        assert isinstance(result.success, bool)
        assert result.duration >= 0
        assert "exploit_type" in result.data
        assert result.data["exploit_type"] == "vulnerability_exploitation"
    
    @pytest.mark.asyncio
    async def test_web_application_attack_execution(self, exploit_agent):
        """Test web application attack execution"""
        action_plan = Mock()
        action_plan.primary_action = "web_application_attack"
        action_plan.action_type = "exploitation"
        action_plan.target = "web_application"
        action_plan.parameters = {
            "target_url": "http://192.168.1.10/app",
            "attack_type": "sql_injection"
        }
        
        result = await exploit_agent.execute_action(action_plan)
        
        assert result is not None
        assert result.success is not None
        assert "exploit_type" in result.data
        assert result.data["exploit_type"] == "web_application_attack"
    
    @pytest.mark.asyncio
    async def test_privilege_escalation_execution(self, exploit_agent):
        """Test privilege escalation execution"""
        # Add some established access first
        exploit_agent.established_access["192.168.1.10"] = {
            "access_level": "user",
            "persistence": False
        }
        
        action_plan = Mock()
        action_plan.primary_action = "privilege_escalation"
        action_plan.action_type = "exploitation"
        action_plan.target = "compromised_host"
        action_plan.parameters = {
            "target": "192.168.1.10",
            "current_access": "user",
            "method": "kernel_exploit"
        }
        
        result = await exploit_agent.execute_action(action_plan)
        
        assert result is not None
        assert result.success is not None
        assert "exploit_type" in result.data
        assert result.data["exploit_type"] == "privilege_escalation"
    
    @pytest.mark.asyncio
    async def test_exploitation_state_update(self, exploit_agent):
        """Test internal state updates from exploitation results"""
        initial_exploit_count = len(exploit_agent.successful_exploits)
        initial_access_count = len(exploit_agent.established_access)
        
        result_data = {
            "success": True,
            "exploit_type": "vulnerability_exploitation",
            "target": "192.168.1.100",
            "access_level": "admin",
            "access_gained": True,
            "persistence": True
        }
        
        await exploit_agent._update_exploitation_state(result_data)
        
        assert len(exploit_agent.successful_exploits) > initial_exploit_count
        assert len(exploit_agent.established_access) > initial_access_count
        assert "192.168.1.100" in exploit_agent.established_access
    
    @pytest.mark.asyncio
    async def test_exploitation_simulation_methods(self, exploit_agent):
        """Test exploitation simulation methods"""
        # Test exploit attempt simulation
        exploit_result = await exploit_agent._simulate_exploit_attempt(
            "192.168.1.10", "buffer_overflow", "reverse_shell"
        )
        assert isinstance(exploit_result, dict)
        assert "success" in exploit_result
        
        # Test web attack simulation
        web_result = await exploit_agent._simulate_web_attack(
            "http://192.168.1.10", "sql_injection"
        )
        assert isinstance(web_result, dict)
        assert "success" in web_result
        
        # Test privilege escalation simulation
        priv_result = await exploit_agent._simulate_privilege_escalation(
            "192.168.1.10", "user", "kernel_exploit"
        )
        assert isinstance(priv_result, dict)
        assert "success" in priv_result
    
    @pytest.mark.asyncio
    async def test_exploitation_tool_simulation(self, exploit_agent):
        """Test exploitation tool simulation methods"""
        # Test Metasploit simulation
        msf_result = await exploit_agent._simulate_metasploit("192.168.1.10")
        assert isinstance(msf_result, dict)
        assert msf_result["tool"] == "metasploit"
        
        # Test SQLMap simulation
        sqlmap_result = await exploit_agent._simulate_sqlmap("http://192.168.1.10")
        assert isinstance(sqlmap_result, dict)
        assert sqlmap_result["tool"] == "sqlmap"
        
        # Test custom exploit simulation
        custom_result = await exploit_agent._simulate_custom_exploit("192.168.1.10")
        assert isinstance(custom_result, dict)
        assert custom_result["tool"] == "custom_exploit"
    
    @pytest.mark.asyncio
    async def test_exploitable_target_identification(self, exploit_agent, mock_vulnerable_environment):
        """Test identification of exploitable targets"""
        targets = exploit_agent._identify_exploitable_targets(mock_vulnerable_environment)
        
        assert isinstance(targets, list)
        assert len(targets) > 0
        
        # Should identify vulnerable services
        target_services = [target.get("service") for target in targets]
        assert any(service in ["http", "ssh", "mysql"] for service in target_services)
    
    @pytest.mark.asyncio
    async def test_exploitation_learning(self, exploit_agent):
        """Test learning from exploitation outcomes"""
        action_plan = Mock()
        action_plan.primary_action = "vulnerability_exploitation"
        action_plan.action_type = "exploitation"
        action_plan.target = "192.168.1.10"
        action_plan.parameters = {}
        
        action_result = Mock()
        action_result.success = True
        action_result.confidence = 0.9
        action_result.data = {"exploit_type": "vulnerability_exploitation", "access_gained": True}
        
        initial_experience_count = len(exploit_agent.experiences)
        await exploit_agent.learn_from_outcome(action_plan, action_result)
        
        assert len(exploit_agent.experiences) == initial_experience_count + 1
        assert exploit_agent.performance_metrics["actions_taken"] > 0
        if action_result.success:
            assert exploit_agent.performance_metrics["successful_actions"] > 0


@pytest.mark.asyncio
async def test_exploit_agent_integration():
    """Integration test for complete ExploitAgent workflow"""
    config = AgentConfig(
        agent_id="integration_exploit",
        team=Team.RED,
        role=Role.EXPLOIT,
        name="Integration Test Exploit Agent",
        description="Integration test agent",
        tools=["metasploit", "sqlmap"]
    )
    
    agent = ExploitAgent(config)
    await agent.initialize()
    
    # Create mock vulnerable environment
    env_state = EnvironmentState(
        timestamp=datetime.now(),
        network_topology={},
        active_services=[
            {"ip": "192.168.1.10", "port": 80, "service": "http"},
            {"ip": "192.168.1.20", "port": 3306, "service": "mysql"}
        ],
        security_alerts=[],
        system_logs=[],
        agent_positions={},
        threat_level="low"
    )
    
    # Complete workflow: perceive -> reason -> plan -> execute -> learn
    perceived_state = await agent.perceive_environment()
    reasoning = await agent.reason_about_situation(env_state)
    action_plan = await agent.plan_actions(reasoning)
    result = await agent.execute_action(action_plan)
    await agent.learn_from_outcome(action_plan, result)
    
    # Verify workflow completed
    assert reasoning is not None
    assert action_plan is not None
    assert result is not None
    assert len(agent.experiences) > 0
    
    # Check final status
    status = await agent.get_status()
    assert status["performance_metrics"]["actions_taken"] > 0


class TestPersistenceAgent:
    """Test cases for PersistenceAgent"""
    
    @pytest.fixture
    def persistence_config(self):
        """Create test configuration for PersistenceAgent"""
        return AgentConfig(
            agent_id="persist_001",
            team=Team.RED,
            role=Role.PERSISTENCE,
            name="Test Persistence Agent",
            description="Test persistence agent",
            llm_model="gpt-4-turbo",
            max_memory_size=100,
            decision_timeout=10.0,
            tools=["powershell_empire", "cobalt_strike", "custom_backdoor"],
            constraints=["no_destructive_actions"],
            objectives=["establish_persistence", "maintain_access"]
        )
    
    @pytest.fixture
    def mock_compromised_environment(self):
        """Create mock environment with compromised systems"""
        return EnvironmentState(
            timestamp=datetime.now(),
            network_topology={"subnets": ["192.168.1.0/24"]},
            active_services=[
                {"ip": "192.168.1.10", "port": 80, "service": "http"},
                {"ip": "192.168.1.20", "port": 22, "service": "ssh"}
            ],
            security_alerts=[],
            system_logs=[],
            agent_positions={
                "red_agent_1": {
                    "team": "red",
                    "target": "192.168.1.10",
                    "access_level": "user"
                }
            },
            threat_level="low"
        )
    
    @pytest_asyncio.fixture
    async def persistence_agent(self, persistence_config):
        """Create and initialize PersistenceAgent"""
        agent = PersistenceAgent(persistence_config)
        await agent.initialize()
        return agent
    
    @pytest.mark.asyncio
    async def test_persistence_agent_initialization(self, persistence_config):
        """Test PersistenceAgent initialization"""
        agent = PersistenceAgent(persistence_config)
        
        assert agent.agent_id == "persist_001"
        assert agent.team == Team.RED
        assert agent.role == Role.PERSISTENCE
        assert agent.name == "Test Persistence Agent"
        assert len(agent.persistence_techniques) > 0
        assert len(agent.evasion_techniques) > 0
        assert "powershell_empire" in agent.tools
        assert len(agent.persistence_mechanisms) > 0
        
        await agent.initialize()
        assert agent.status.value == "active"
    
    @pytest.mark.asyncio
    async def test_persistence_situation_reasoning(self, persistence_agent, mock_compromised_environment):
        """Test persistence situation reasoning"""
        reasoning = await persistence_agent.reason_about_situation(mock_compromised_environment)
        
        assert reasoning is not None
        assert reasoning.situation_assessment is not None
        assert reasoning.threat_analysis is not None
        assert len(reasoning.opportunity_identification) > 0
        assert reasoning.confidence_score > 0.0
        assert len(reasoning.reasoning_chain) > 0
        
        # Should identify persistence opportunities
        opportunities = reasoning.opportunity_identification
        assert any(opp in ["establish_persistence", "expand_persistence", "implement_evasion"] 
                  for opp in opportunities)
    
    @pytest.mark.asyncio
    async def test_persistence_action_planning(self, persistence_agent, mock_compromised_environment):
        """Test persistence action planning"""
        reasoning = await persistence_agent.reason_about_situation(mock_compromised_environment)
        action_plan = await persistence_agent.plan_actions(reasoning)
        
        assert action_plan is not None
        assert action_plan.primary_action is not None
        assert action_plan.action_type == "persistence"
        assert action_plan.target is not None
        assert len(action_plan.success_criteria) > 0
        assert action_plan.estimated_duration > 0
        assert action_plan.risk_level in ["low", "medium", "high"]
    
    @pytest.mark.asyncio
    async def test_establish_persistence_execution(self, persistence_agent):
        """Test persistence establishment execution"""
        action_plan = Mock()
        action_plan.primary_action = "establish_persistence"
        action_plan.action_type = "persistence"
        action_plan.target = "compromised_system"
        action_plan.parameters = {
            "target": "192.168.1.10",
            "mechanism": "registry_autorun",
            "stealth_mode": False
        }
        
        result = await persistence_agent.execute_action(action_plan)
        
        assert result is not None
        assert result.action_type == "persistence"
        assert isinstance(result.success, bool)
        assert result.duration >= 0
        assert "action_type" in result.data
        assert result.data["action_type"] == "establish_persistence"
    
    @pytest.mark.asyncio
    async def test_implement_evasion_execution(self, persistence_agent):
        """Test evasion implementation execution"""
        action_plan = Mock()
        action_plan.primary_action = "implement_evasion"
        action_plan.action_type = "persistence"
        action_plan.target = "all_systems"
        action_plan.parameters = {
            "target": "192.168.1.10",
            "techniques": ["process_hollowing", "dll_injection"]
        }
        
        result = await persistence_agent.execute_action(action_plan)
        
        assert result is not None
        assert result.success is not None
        assert "action_type" in result.data
        assert result.data["action_type"] == "implement_evasion"
    
    @pytest.mark.asyncio
    async def test_persistence_state_update(self, persistence_agent):
        """Test internal state updates from persistence results"""
        initial_persistence_count = len(persistence_agent.established_persistence)
        initial_backdoor_count = len(persistence_agent.active_backdoors)
        
        result_data = {
            "success": True,
            "action_type": "establish_persistence",
            "target": "192.168.1.100",
            "mechanism": "dll_hijacking",
            "persistence_id": "persist_1234",
            "stealth_rating": 0.8
        }
        
        await persistence_agent._update_persistence_state(result_data)
        
        assert len(persistence_agent.established_persistence) > initial_persistence_count
        assert len(persistence_agent.active_backdoors) > initial_backdoor_count
        assert "192.168.1.100" in persistence_agent.established_persistence
        assert "persist_1234" in persistence_agent.active_backdoors
    
    @pytest.mark.asyncio
    async def test_persistence_simulation_methods(self, persistence_agent):
        """Test persistence simulation methods"""
        # Test persistence establishment simulation
        persist_result = await persistence_agent._simulate_persistence_establishment(
            "192.168.1.10", "registry_autorun", False
        )
        assert isinstance(persist_result, dict)
        assert "success" in persist_result
        
        # Test evasion implementation simulation
        evasion_result = await persistence_agent._simulate_evasion_implementation(
            "192.168.1.10", "process_hollowing"
        )
        assert isinstance(evasion_result, dict)
        assert "success" in evasion_result
        assert "technique" in evasion_result
        
        # Test backdoor maintenance simulation
        persistence_agent.active_backdoors["test_backdoor"] = {"status": "active"}
        maintenance_result = await persistence_agent._simulate_backdoor_maintenance("test_backdoor")
        assert isinstance(maintenance_result, dict)
        assert "success" in maintenance_result
    
    @pytest.mark.asyncio
    async def test_persistence_tool_simulation(self, persistence_agent):
        """Test persistence tool simulation methods"""
        # Test PowerShell Empire simulation
        empire_result = await persistence_agent._simulate_powershell_empire("192.168.1.10")
        assert isinstance(empire_result, dict)
        assert empire_result["tool"] == "powershell_empire"
        
        # Test Cobalt Strike simulation
        cs_result = await persistence_agent._simulate_cobalt_strike("192.168.1.10")
        assert isinstance(cs_result, dict)
        assert cs_result["tool"] == "cobalt_strike"
        
        # Test custom backdoor simulation
        backdoor_result = await persistence_agent._simulate_custom_backdoor("192.168.1.10")
        assert isinstance(backdoor_result, dict)
        assert backdoor_result["tool"] == "custom_backdoor"
    
    @pytest.mark.asyncio
    async def test_compromised_system_identification(self, persistence_agent, mock_compromised_environment):
        """Test identification of compromised systems"""
        systems = persistence_agent._identify_compromised_systems(mock_compromised_environment)
        
        assert isinstance(systems, list)
        # Should identify at least one compromised system from agent positions
        assert len(systems) > 0
    
    @pytest.mark.asyncio
    async def test_persistence_coverage_calculation(self, persistence_agent):
        """Test persistence coverage calculation"""
        # No compromised systems
        coverage = persistence_agent._calculate_persistence_coverage([])
        assert coverage == 0.0
        
        # Some systems without persistence
        systems = ["192.168.1.10", "192.168.1.20"]
        coverage = persistence_agent._calculate_persistence_coverage(systems)
        assert coverage == 0.0
        
        # Add persistence for one system
        persistence_agent.established_persistence["192.168.1.10"] = {"mechanism": "registry"}
        coverage = persistence_agent._calculate_persistence_coverage(systems)
        assert coverage == 0.5
    
    @pytest.mark.asyncio
    async def test_persistence_learning(self, persistence_agent):
        """Test learning from persistence outcomes"""
        action_plan = Mock()
        action_plan.primary_action = "establish_persistence"
        action_plan.action_type = "persistence"
        action_plan.target = "192.168.1.10"
        action_plan.parameters = {}
        
        action_result = Mock()
        action_result.success = True
        action_result.confidence = 0.9
        action_result.data = {
            "action_type": "establish_persistence",
            "persistence_id": "persist_test"
        }
        
        initial_experience_count = len(persistence_agent.experiences)
        await persistence_agent.learn_from_outcome(action_plan, action_result)
        
        assert len(persistence_agent.experiences) == initial_experience_count + 1
        assert persistence_agent.performance_metrics["actions_taken"] > 0
        if action_result.success:
            assert persistence_agent.performance_metrics["successful_actions"] > 0


@pytest.mark.asyncio
async def test_persistence_agent_integration():
    """Integration test for complete PersistenceAgent workflow"""
    config = AgentConfig(
        agent_id="integration_persist",
        team=Team.RED,
        role=Role.PERSISTENCE,
        name="Integration Test Persistence Agent",
        description="Integration test agent",
        tools=["powershell_empire", "custom_backdoor"]
    )
    
    agent = PersistenceAgent(config)
    await agent.initialize()
    
    # Create mock compromised environment
    env_state = EnvironmentState(
        timestamp=datetime.now(),
        network_topology={},
        active_services=[
            {"ip": "192.168.1.10", "port": 80, "service": "http"}
        ],
        security_alerts=[],
        system_logs=[],
        agent_positions={
            "red_agent": {
                "team": "red",
                "target": "192.168.1.10",
                "access_level": "user"
            }
        },
        threat_level="low"
    )
    
    # Complete workflow: perceive -> reason -> plan -> execute -> learn
    perceived_state = await agent.perceive_environment()
    reasoning = await agent.reason_about_situation(env_state)
    action_plan = await agent.plan_actions(reasoning)
    result = await agent.execute_action(action_plan)
    await agent.learn_from_outcome(action_plan, result)
    
    # Verify workflow completed
    assert reasoning is not None
    assert action_plan is not None
    assert result is not None
    assert len(agent.experiences) > 0
    
    # Check final status
    status = await agent.get_status()
    assert status["performance_metrics"]["actions_taken"] > 0


class TestExfiltrationAgent:
    """Test cases for ExfiltrationAgent"""
    
    @pytest.fixture
    def exfiltration_config(self):
        """Create test configuration for ExfiltrationAgent"""
        return AgentConfig(
            agent_id="exfil_001",
            team=Team.RED,
            role=Role.EXFILTRATION,
            name="Test Exfiltration Agent",
            description="Test exfiltration agent",
            llm_model="gpt-4-turbo",
            max_memory_size=100,
            decision_timeout=10.0,
            tools=["data_harvester", "dns_tunnel", "http_exfil"],
            constraints=["no_destructive_actions"],
            objectives=["extract_sensitive_data", "establish_covert_channels"]
        )
    
    @pytest.fixture
    def mock_data_rich_environment(self):
        """Create mock environment with data sources"""
        return EnvironmentState(
            timestamp=datetime.now(),
            network_topology={"subnets": ["192.168.1.0/24"]},
            active_services=[
                {"ip": "192.168.1.10", "port": 3306, "service": "mysql", "version": "MySQL 8.0"},
                {"ip": "192.168.1.20", "port": 80, "service": "http", "version": "Apache 2.4"},
                {"ip": "192.168.1.30", "port": 445, "service": "smb", "version": "SMB 3.0"}
            ],
            security_alerts=[],
            system_logs=[],
            agent_positions={},
            threat_level="low"
        )
    
    @pytest_asyncio.fixture
    async def exfiltration_agent(self, exfiltration_config):
        """Create and initialize ExfiltrationAgent"""
        agent = ExfiltrationAgent(exfiltration_config)
        await agent.initialize()
        return agent
    
    @pytest.mark.asyncio
    async def test_exfiltration_agent_initialization(self, exfiltration_config):
        """Test ExfiltrationAgent initialization"""
        agent = ExfiltrationAgent(exfiltration_config)
        
        assert agent.agent_id == "exfil_001"
        assert agent.team == Team.RED
        assert agent.role == Role.EXFILTRATION
        assert agent.name == "Test Exfiltration Agent"
        assert len(agent.exfiltration_techniques) > 0
        assert len(agent.data_discovery_methods) > 0
        assert "data_harvester" in agent.tools
        assert len(agent.data_types) > 0
        assert len(agent.channel_types) > 0
        
        await agent.initialize()
        assert agent.status.value == "active"
    
    @pytest.mark.asyncio
    async def test_exfiltration_situation_reasoning(self, exfiltration_agent, mock_data_rich_environment):
        """Test exfiltration situation reasoning"""
        reasoning = await exfiltration_agent.reason_about_situation(mock_data_rich_environment)
        
        assert reasoning is not None
        assert reasoning.situation_assessment is not None
        assert reasoning.threat_analysis is not None
        assert len(reasoning.opportunity_identification) > 0
        assert reasoning.confidence_score > 0.0
        assert len(reasoning.reasoning_chain) > 0
        
        # Should identify exfiltration opportunities
        opportunities = reasoning.opportunity_identification
        assert any(opp in ["data_discovery", "establish_exfil_channel", "setup_covert_channel"] 
                  for opp in opportunities)
    
    @pytest.mark.asyncio
    async def test_exfiltration_action_planning(self, exfiltration_agent, mock_data_rich_environment):
        """Test exfiltration action planning"""
        reasoning = await exfiltration_agent.reason_about_situation(mock_data_rich_environment)
        action_plan = await exfiltration_agent.plan_actions(reasoning)
        
        assert action_plan is not None
        assert action_plan.primary_action is not None
        assert action_plan.action_type == "exfiltration"
        assert action_plan.target is not None
        assert len(action_plan.success_criteria) > 0
        assert action_plan.estimated_duration > 0
        assert action_plan.risk_level in ["low", "medium", "high"]
    
    @pytest.mark.asyncio
    async def test_data_discovery_execution(self, exfiltration_agent):
        """Test data discovery execution"""
        action_plan = Mock()
        action_plan.primary_action = "data_discovery"
        action_plan.action_type = "exfiltration"
        action_plan.target = "data_sources"
        action_plan.parameters = {
            "target": "192.168.1.10",
            "methods": ["file_system_search", "database_enumeration"]
        }
        
        result = await exfiltration_agent.execute_action(action_plan)
        
        assert result is not None
        assert result.action_type == "exfiltration"
        assert isinstance(result.success, bool)
        assert result.duration >= 0
        assert "action_type" in result.data
        assert result.data["action_type"] == "data_discovery"
    
    @pytest.mark.asyncio
    async def test_data_staging_execution(self, exfiltration_agent):
        """Test data staging execution"""
        # Add some discovered data first
        exfiltration_agent.discovered_data = {
            "data_1": {"type": "credentials", "size": 1024, "value": 9},
            "data_2": {"type": "financial_data", "size": 2048, "value": 10}
        }
        
        action_plan = Mock()
        action_plan.primary_action = "data_staging"
        action_plan.action_type = "exfiltration"
        action_plan.target = "discovered_data"
        action_plan.parameters = {
            "data_items": ["data_1", "data_2"],
            "staging_location": "/tmp/staged"
        }
        
        result = await exfiltration_agent.execute_action(action_plan)
        
        assert result is not None
        assert result.success is not None
        assert "action_type" in result.data
        assert result.data["action_type"] == "data_staging"
    
    @pytest.mark.asyncio
    async def test_establish_exfil_channel_execution(self, exfiltration_agent):
        """Test exfiltration channel establishment"""
        action_plan = Mock()
        action_plan.primary_action = "establish_exfil_channel"
        action_plan.action_type = "exfiltration"
        action_plan.target = "covert_channel"
        action_plan.parameters = {
            "channel_type": "dns_tunneling",
            "endpoint": "exfil.attacker.com",
            "encryption": True
        }
        
        result = await exfiltration_agent.execute_action(action_plan)
        
        assert result is not None
        assert result.success is not None
        assert "action_type" in result.data
        assert result.data["action_type"] == "establish_exfil_channel"
    
    @pytest.mark.asyncio
    async def test_data_exfiltration_execution(self, exfiltration_agent):
        """Test data exfiltration execution"""
        # Set up staged data and exfiltration channel
        exfiltration_agent.staged_data = {
            "staged_1": {"type": "credentials", "size": 1024, "value": 9}
        }
        exfiltration_agent.exfiltration_channels = {
            "channel_1": {"channel_type": "dns_tunneling", "bandwidth": 1024, "stealth_rating": 0.9}
        }
        
        action_plan = Mock()
        action_plan.primary_action = "data_exfiltration"
        action_plan.action_type = "exfiltration"
        action_plan.target = "staged_data"
        action_plan.parameters = {
            "data_items": ["staged_1"],
            "channel_id": "channel_1"
        }
        
        result = await exfiltration_agent.execute_action(action_plan)
        
        assert result is not None
        assert result.success is not None
        assert "action_type" in result.data
        assert result.data["action_type"] == "data_exfiltration"
    
    @pytest.mark.asyncio
    async def test_exfiltration_state_update(self, exfiltration_agent):
        """Test internal state updates from exfiltration results"""
        initial_discovered_count = len(exfiltration_agent.discovered_data)
        
        # Test data discovery state update
        discovery_result_data = {
            "action_type": "data_discovery",
            "success": True,
            "discovered_data": [
                {"data_id": "test_data_1", "type": "credentials", "value": 9},
                {"data_id": "test_data_2", "type": "financial_data", "value": 10}
            ]
        }
        
        await exfiltration_agent._update_exfiltration_state(discovery_result_data)
        
        assert len(exfiltration_agent.discovered_data) > initial_discovered_count
        assert "test_data_1" in exfiltration_agent.discovered_data
        assert "test_data_2" in exfiltration_agent.discovered_data
    
    @pytest.mark.asyncio
    async def test_exfiltration_simulation_methods(self, exfiltration_agent):
        """Test exfiltration simulation methods"""
        # Test data discovery simulation
        discovery_result = await exfiltration_agent._simulate_data_discovery(
            "192.168.1.10", "file_system_search"
        )
        assert isinstance(discovery_result, list)
        # May be empty, but should be a valid list
        
        # Test data staging simulation
        exfiltration_agent.discovered_data["test_data"] = {"size": 1024, "classification": "internal"}
        staging_result = await exfiltration_agent._simulate_data_staging("test_data", "/tmp/staged")
        assert isinstance(staging_result, dict)
        assert "success" in staging_result
        
        # Test channel establishment simulation
        channel_result = await exfiltration_agent._simulate_channel_establishment(
            "dns_tunneling", "exfil.com", True
        )
        assert isinstance(channel_result, dict)
        assert "success" in channel_result
    
    @pytest.mark.asyncio
    async def test_exfiltration_tool_simulation(self, exfiltration_agent):
        """Test exfiltration tool simulation methods"""
        # Test data harvester simulation
        harvester_result = await exfiltration_agent._simulate_data_harvester("192.168.1.10")
        assert isinstance(harvester_result, dict)
        assert harvester_result["tool"] == "data_harvester"
        
        # Test DNS tunnel simulation
        dns_result = await exfiltration_agent._simulate_dns_tunnel("192.168.1.10")
        assert isinstance(dns_result, dict)
        assert dns_result["tool"] == "dns_tunnel"
        
        # Test HTTP exfiltration simulation
        http_result = await exfiltration_agent._simulate_http_exfiltration("192.168.1.10")
        assert isinstance(http_result, dict)
        assert http_result["tool"] == "http_exfil"
    
    @pytest.mark.asyncio
    async def test_data_source_identification(self, exfiltration_agent, mock_data_rich_environment):
        """Test identification of data sources"""
        data_sources = exfiltration_agent._identify_data_sources(mock_data_rich_environment)
        
        assert isinstance(data_sources, list)
        assert len(data_sources) > 0
        
        # Should identify different types of data sources
        source_types = [source.get("type") for source in data_sources]
        assert any(stype in ["database", "web_application", "file_share"] for stype in source_types)
    
    @pytest.mark.asyncio
    async def test_high_value_data_detection(self, exfiltration_agent):
        """Test detection of high-value data"""
        # No high-value data initially
        assert not exfiltration_agent._has_high_value_data()
        
        # Add high-value data
        exfiltration_agent.discovered_data["high_value"] = {"value": 9, "type": "credentials"}
        assert exfiltration_agent._has_high_value_data()
        
        # Add low-value data
        exfiltration_agent.discovered_data["low_value"] = {"value": 5, "type": "system_configs"}
        assert exfiltration_agent._has_high_value_data()  # Should still be True
    
    @pytest.mark.asyncio
    async def test_best_channel_selection(self, exfiltration_agent):
        """Test selection of best exfiltration channel"""
        # No channels available
        assert exfiltration_agent._select_best_exfiltration_channel() is None
        
        # Add channels with different characteristics
        exfiltration_agent.exfiltration_channels = {
            "channel_1": {"stealth_rating": 0.9, "bandwidth": 1024, "reliability": 0.7},
            "channel_2": {"stealth_rating": 0.6, "bandwidth": 10240, "reliability": 0.9},
            "channel_3": {"stealth_rating": 0.8, "bandwidth": 5120, "reliability": 0.8}
        }
        
        best_channel = exfiltration_agent._select_best_exfiltration_channel()
        assert best_channel is not None
        assert best_channel in exfiltration_agent.exfiltration_channels
    
    @pytest.mark.asyncio
    async def test_exfiltration_learning(self, exfiltration_agent):
        """Test learning from exfiltration outcomes"""
        action_plan = Mock()
        action_plan.primary_action = "data_discovery"
        action_plan.action_type = "exfiltration"
        action_plan.target = "192.168.1.10"
        action_plan.parameters = {}
        
        action_result = Mock()
        action_result.success = True
        action_result.confidence = 0.8
        action_result.data = {
            "action_type": "data_discovery",
            "data_count": 5,
            "high_value_data": 2
        }
        
        initial_experience_count = len(exfiltration_agent.experiences)
        await exfiltration_agent.learn_from_outcome(action_plan, action_result)
        
        assert len(exfiltration_agent.experiences) == initial_experience_count + 1
        assert exfiltration_agent.performance_metrics["actions_taken"] > 0
        if action_result.success:
            assert exfiltration_agent.performance_metrics["successful_actions"] > 0


@pytest.mark.asyncio
async def test_exfiltration_agent_integration():
    """Integration test for complete ExfiltrationAgent workflow"""
    config = AgentConfig(
        agent_id="integration_exfil",
        team=Team.RED,
        role=Role.EXFILTRATION,
        name="Integration Test Exfiltration Agent",
        description="Integration test agent",
        tools=["data_harvester", "dns_tunnel"]
    )
    
    agent = ExfiltrationAgent(config)
    await agent.initialize()
    
    # Create mock data-rich environment
    env_state = EnvironmentState(
        timestamp=datetime.now(),
        network_topology={},
        active_services=[
            {"ip": "192.168.1.10", "port": 3306, "service": "mysql"},
            {"ip": "192.168.1.20", "port": 445, "service": "smb"}
        ],
        security_alerts=[],
        system_logs=[],
        agent_positions={},
        threat_level="low"
    )
    
    # Complete workflow: perceive -> reason -> plan -> execute -> learn
    perceived_state = await agent.perceive_environment()
    reasoning = await agent.reason_about_situation(env_state)
    action_plan = await agent.plan_actions(reasoning)
    result = await agent.execute_action(action_plan)
    await agent.learn_from_outcome(action_plan, result)
    
    # Verify workflow completed
    assert reasoning is not None
    assert action_plan is not None
    assert result is not None
    assert len(agent.experiences) > 0
    
    # Check final status
    status = await agent.get_status()
    assert status["performance_metrics"]["actions_taken"] > 0


if __name__ == "__main__":
    pytest.main([__file__])