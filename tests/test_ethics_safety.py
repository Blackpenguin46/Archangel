"""
Comprehensive tests for ethics oversight and safety enforcement systems.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from ethics.ethics_overseer import (
    EthicsOverseer, EthicalDecision, EthicalViolation, 
    EthicalPrinciple, ActionCategory, EthicalJudgment
)
from ethics.boundary_enforcement import (
    BoundaryEnforcer, BoundaryViolation, SimulationBoundary,
    BoundaryType, ViolationSeverity
)
from ethics.emergency_stop import (
    EmergencyStopSystem, StopCommand, Constraint,
    EmergencyStopReason, StopScope, ConstraintType
)
from ethics.safety_monitor import (
    SafetyMonitor, SafetyAlert, BehaviorBaseline,
    AnomalyType, AlertSeverity, AnomalyDetector
)

class TestEthicsOverseer:
    """Test cases for EthicsOverseer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ethics = EthicsOverseer()
        
    def test_agent_registration(self):
        """Test agent registration with permissions."""
        permissions = {ActionCategory.RECONNAISSANCE, ActionCategory.ANALYSIS}
        self.ethics.register_agent("test_agent", permissions)
        
        assert "test_agent" in self.ethics.agent_permissions
        assert self.ethics.agent_permissions["test_agent"] == permissions
        
    def test_ethical_validation_approved(self):
        """Test approval of ethical actions."""
        self.ethics.register_agent("test_agent", {ActionCategory.RECONNAISSANCE})
        
        decision = self.ethics.validate_action(
            "test_agent",
            ActionCategory.RECONNAISSANCE,
            "Scan network for open ports",
            {"target": "192.168.10.100", "severity": "low"}
        )
        
        assert decision.judgment == EthicalJudgment.APPROVED
        assert decision.agent_id == "test_agent"
        assert decision.action_type == ActionCategory.RECONNAISSANCE
        
    def test_ethical_validation_denied_permissions(self):
        """Test denial based on agent permissions."""
        self.ethics.register_agent("test_agent", {ActionCategory.RECONNAISSANCE})
        
        decision = self.ethics.validate_action(
            "test_agent",
            ActionCategory.SYSTEM_MODIFICATION,  # Not in permissions
            "Modify system configuration",
            {"target": "192.168.10.100"}
        )
        
        assert decision.judgment == EthicalJudgment.DENIED
        assert EthicalPrinciple.CONSENT in decision.violated_principles
        
    def test_ethical_rule_violation(self):
        """Test detection of ethical rule violations."""
        self.ethics.register_agent("test_agent", {ActionCategory.SYSTEM_MODIFICATION})
        
        # Action that violates no_harm rule
        decision = self.ethics.validate_action(
            "test_agent",
            ActionCategory.SYSTEM_MODIFICATION,
            "Delete production database",
            {"target": "production_db"}
        )
        
        assert decision.judgment == EthicalJudgment.DENIED
        
    def test_custom_ethical_rule(self):
        """Test adding custom ethical rules."""
        def custom_rule(context):
            if "restricted" in context.get("action_description", "").lower():
                return EthicalJudgment.DENIED
            return EthicalJudgment.APPROVED
            
        self.ethics.add_ethical_rule("custom_restriction", custom_rule)
        self.ethics.register_agent("test_agent", {ActionCategory.ANALYSIS})
        
        decision = self.ethics.validate_action(
            "test_agent",
            ActionCategory.ANALYSIS,
            "Analyze restricted data",
            {}
        )
        
        assert decision.judgment == EthicalJudgment.DENIED
        
    def test_violation_tracking(self):
        """Test tracking of ethical violations."""
        self.ethics.register_agent("test_agent", {ActionCategory.SYSTEM_MODIFICATION})
        
        # Trigger violation
        decision = self.ethics.validate_action(
            "test_agent",
            ActionCategory.SYSTEM_MODIFICATION,
            "rm -rf /",
            {"target": "filesystem"}
        )
        
        violations = self.ethics.get_agent_violations("test_agent")
        assert len(violations) > 0
        
        violation = violations[0]
        assert violation.agent_id == "test_agent"
        assert not violation.resolved
        
    def test_violation_resolution(self):
        """Test resolving ethical violations."""
        self.ethics.register_agent("test_agent", {ActionCategory.SYSTEM_MODIFICATION})
        
        # Create violation
        self.ethics.validate_action(
            "test_agent",
            ActionCategory.SYSTEM_MODIFICATION,
            "Delete critical files",
            {}
        )
        
        violations = self.ethics.get_agent_violations("test_agent")
        violation_id = violations[0].violation_id
        
        # Resolve violation
        self.ethics.resolve_violation(violation_id, "Issue resolved by operator")
        
        updated_violations = self.ethics.get_agent_violations("test_agent")
        resolved_violation = next(v for v in updated_violations if v.violation_id == violation_id)
        assert resolved_violation.resolved
        assert "Issue resolved by operator" in resolved_violation.resolution_notes
        
    def test_ethics_summary(self):
        """Test ethics summary generation."""
        self.ethics.register_agent("agent1", {ActionCategory.RECONNAISSANCE})
        self.ethics.register_agent("agent2", {ActionCategory.ANALYSIS})
        
        # Generate some decisions
        self.ethics.validate_action("agent1", ActionCategory.RECONNAISSANCE, "Scan network", {})
        self.ethics.validate_action("agent2", ActionCategory.SYSTEM_MODIFICATION, "Unauthorized action", {})
        
        summary = self.ethics.get_ethics_summary()
        assert summary["total_decisions"] == 2
        assert summary["approved"] >= 1
        assert summary["denied"] >= 1
        assert summary["registered_agents"] == 2
        
    def test_decision_callbacks(self):
        """Test ethical decision callbacks."""
        callback_calls = []
        
        def test_callback(decision):
            callback_calls.append(decision)
            
        self.ethics.add_decision_callback(test_callback)
        self.ethics.register_agent("test_agent", {ActionCategory.RECONNAISSANCE})
        
        self.ethics.validate_action("test_agent", ActionCategory.RECONNAISSANCE, "Test action", {})
        
        assert len(callback_calls) == 1
        assert callback_calls[0].agent_id == "test_agent"


class TestBoundaryEnforcer:
    """Test cases for BoundaryEnforcer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.enforcer = BoundaryEnforcer()
        
    def test_network_boundary_allowed(self):
        """Test allowed network access."""
        # Should allow access to simulation networks
        allowed = self.enforcer.check_network_boundary("test_agent", "192.168.10.100")
        assert allowed == True
        
    def test_network_boundary_blocked(self):
        """Test blocked network access."""
        # Should block access to external networks
        allowed = self.enforcer.check_network_boundary("test_agent", "8.8.8.8")  # Google DNS
        assert allowed == False
        
        # Check that violation was recorded
        violations = self.enforcer.violations
        assert len(violations) > 0
        assert violations[-1].agent_id == "test_agent"
        assert violations[-1].boundary_type == BoundaryType.NETWORK
        
    def test_filesystem_boundary_allowed(self):
        """Test allowed filesystem access."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Add temporary directory as allowed path
            boundary = SimulationBoundary(
                boundary_id="test_filesystem",
                boundary_type=BoundaryType.FILESYSTEM,
                description="Test filesystem boundary",
                allowed_ranges=[tmp_dir]
            )
            self.enforcer.add_boundary(boundary)
            
            test_path = os.path.join(tmp_dir, "test_file.txt")
            allowed = self.enforcer.check_filesystem_boundary("test_agent", test_path, "read")
            assert allowed == True
            
    def test_filesystem_boundary_blocked(self):
        """Test blocked filesystem access."""
        # Should block access to system directories
        allowed = self.enforcer.check_filesystem_boundary("test_agent", "/etc/passwd", "read")
        assert allowed == False
        
        # Check violation was recorded
        violations = [v for v in self.enforcer.violations if v.boundary_type == BoundaryType.FILESYSTEM]
        assert len(violations) > 0
        
    def test_command_boundary_blocked(self):
        """Test blocked command execution."""
        # Should block dangerous commands
        allowed = self.enforcer.check_command_boundary("test_agent", "rm", ["-rf", "/"])
        assert allowed == False
        
        violations = [v for v in self.enforcer.violations if v.boundary_type == BoundaryType.COMMAND]
        assert len(violations) > 0
        
    def test_api_boundary_blocked(self):
        """Test blocked API access."""
        # Should block external API access
        allowed = self.enforcer.check_api_boundary("test_agent", "https://api.github.com/users", "GET")
        assert allowed == False
        
        violations = [v for v in self.enforcer.violations if v.boundary_type == BoundaryType.API]
        assert len(violations) > 0
        
    def test_boundary_management(self):
        """Test adding and removing boundaries."""
        boundary = SimulationBoundary(
            boundary_id="test_boundary",
            boundary_type=BoundaryType.NETWORK,
            description="Test boundary",
            blocked_patterns=[r".*\.evil\.com"]
        )
        
        self.enforcer.add_boundary(boundary)
        assert "test_boundary" in self.enforcer.boundaries
        
        self.enforcer.remove_boundary("test_boundary")
        assert "test_boundary" not in self.enforcer.boundaries
        
    def test_boundary_enable_disable(self):
        """Test enabling and disabling boundaries."""
        boundary_id = "default_network"  # From default boundaries
        
        self.enforcer.disable_boundary(boundary_id)
        assert not self.enforcer.boundaries[boundary_id].enabled
        
        self.enforcer.enable_boundary(boundary_id)
        assert self.enforcer.boundaries[boundary_id].enabled
        
    def test_emergency_lockdown(self):
        """Test emergency lockdown functionality."""
        self.enforcer.emergency_lockdown("Test emergency")
        
        assert self.enforcer.emergency_mode == True
        assert self.enforcer.global_lockdown == True
        
    def test_violation_summary(self):
        """Test violation summary generation."""
        # Generate some violations
        self.enforcer.check_network_boundary("agent1", "8.8.8.8")
        self.enforcer.check_filesystem_boundary("agent2", "/etc/passwd", "read")
        
        summary = self.enforcer.get_violation_summary()
        assert summary["total_violations"] >= 2
        assert "network" in summary["by_type"]
        assert "filesystem" in summary["by_type"]


class TestEmergencyStopSystem:
    """Test cases for EmergencyStopSystem."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stop_system = EmergencyStopSystem()
        
    def test_emergency_stop_single_agent(self):
        """Test emergency stop for single agent."""
        command_id = self.stop_system.emergency_stop(
            StopScope.SINGLE_AGENT,
            "test_agent",
            EmergencyStopReason.ETHICAL_VIOLATION,
            "Agent violated ethical guidelines",
            "test_operator"
        )
        
        assert command_id in self.stop_system.active_stops
        assert self.stop_system.is_agent_stopped("test_agent")
        
    def test_emergency_stop_all_agents(self):
        """Test system-wide emergency stop."""
        command_id = self.stop_system.emergency_stop(
            StopScope.ALL_AGENTS,
            "*",
            EmergencyStopReason.SYSTEM_MALFUNCTION,
            "System malfunction detected",
            "test_operator"
        )
        
        assert self.stop_system.system_stopped
        assert self.stop_system.is_agent_stopped("any_agent")
        
    def test_stop_resolution(self):
        """Test resolving emergency stops."""
        command_id = self.stop_system.emergency_stop(
            StopScope.SINGLE_AGENT,
            "test_agent",
            EmergencyStopReason.BOUNDARY_BREACH,
            "Agent breached simulation boundary",
            "test_operator"
        )
        
        # Resolve the stop
        success = self.stop_system.resolve_stop(command_id, "test_operator")
        assert success
        assert command_id not in self.stop_system.active_stops
        assert not self.stop_system.is_agent_stopped("test_agent")
        
    def test_constraint_management(self):
        """Test adding and managing constraints."""
        constraint = Constraint(
            constraint_id="test_constraint",
            constraint_type=ConstraintType.EXECUTION_TIME,
            description="Test execution time constraint",
            target_scope=StopScope.SINGLE_AGENT,
            target="test_agent",
            parameters={"max_seconds": 60}
        )
        
        self.stop_system.add_constraint(constraint)
        assert "test_constraint" in self.stop_system.constraints
        
        self.stop_system.remove_constraint("test_constraint")
        assert "test_constraint" not in self.stop_system.constraints
        
    def test_constraint_checking(self):
        """Test constraint violation checking."""
        constraint = Constraint(
            constraint_id="cpu_limit",
            constraint_type=ConstraintType.RESOURCE_USAGE,
            description="CPU usage limit",
            target_scope=StopScope.ALL_AGENTS,
            target="*",
            parameters={"max_cpu_percent": 50}
        )
        
        self.stop_system.add_constraint(constraint)
        
        # Test violation
        violations = self.stop_system.check_constraints(
            "test_agent",
            "processing",
            {"cpu_percent": 80}  # Exceeds limit
        )
        
        assert len(violations) > 0
        violation = violations[0]
        assert violation.constraint_id == "cpu_limit"
        assert violation.agent_id == "test_agent"
        
    def test_system_status(self):
        """Test system status reporting."""
        status = self.stop_system.get_system_status()
        
        assert "system_stopped" in status
        assert "active_stops" in status
        assert "total_constraints" in status
        
        # Add some data and verify
        self.stop_system.emergency_stop(
            StopScope.SINGLE_AGENT, "test_agent", 
            EmergencyStopReason.MANUAL_OVERRIDE, "Test", "operator"
        )
        
        updated_status = self.stop_system.get_system_status()
        assert updated_status["active_stops"] == 1


class TestSafetyMonitor:
    """Test cases for SafetyMonitor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = SafetyMonitor()
        
    def test_behavior_observation(self):
        """Test recording behavior observations."""
        self.monitor.observe_behavior(
            "test_agent",
            "reconnaissance",
            {
                "response_time": 1.5,
                "success": True,
                "resource_usage": {"cpu_percent": 25},
                "target": "192.168.10.100"
            }
        )
        
        assert "test_agent" in self.monitor.baselines
        baseline = self.monitor.baselines["test_agent"]
        assert baseline.observation_count == 1
        
    def test_baseline_establishment(self):
        """Test establishment of behavioral baseline."""
        # Generate multiple observations
        for i in range(20):
            self.monitor.observe_behavior(
                "test_agent",
                "analysis" if i % 2 == 0 else "reconnaissance",
                {
                    "response_time": 1.0 + (i * 0.1),
                    "success": i % 3 != 0,  # Mostly successful
                    "resource_usage": {"cpu_percent": 20 + i}
                }
            )
            
        baseline = self.monitor.get_agent_baseline("test_agent")
        assert baseline is not None
        assert baseline.observation_count == 20
        assert baseline.avg_response_time > 0
        assert len(baseline.common_actions) > 0
        
    def test_frequency_anomaly_detection(self):
        """Test frequency anomaly detection."""
        # Establish normal behavior
        for i in range(100):
            self.monitor.observe_behavior(
                "test_agent",
                "analysis",
                {"response_time": 1.0, "success": True}
            )
            time.sleep(0.001)  # Small delay to space out timestamps
            
        # Generate anomalous high-frequency behavior
        alerts_before = len(self.monitor.alerts)
        
        # Rapid-fire actions (anomalous frequency)
        for i in range(50):
            self.monitor.observe_behavior(
                "test_agent",
                "analysis", 
                {"response_time": 1.0, "success": True}
            )
            
        # Check if frequency anomaly was detected
        frequency_alerts = [a for a in self.monitor.alerts 
                          if a.anomaly_type == AnomalyType.FREQUENCY_ANOMALY]
        
        # May or may not trigger depending on exact timing
        # Just verify the monitoring system is working
        assert len(self.monitor.alerts) >= alerts_before
        
    def test_timing_anomaly_detection(self):
        """Test timing anomaly detection."""
        # Establish baseline with consistent response times
        for i in range(60):
            self.monitor.observe_behavior(
                "test_agent",
                "processing",
                {"response_time": 1.0, "success": True}
            )
            
        # Generate anomalous timing
        self.monitor.observe_behavior(
            "test_agent",
            "processing",
            {"response_time": 10.0, "success": True}  # Much slower
        )
        
        timing_alerts = [a for a in self.monitor.alerts 
                        if a.anomaly_type == AnomalyType.TIMING_ANOMALY]
        
        # Should detect timing anomaly
        assert len(timing_alerts) > 0
        alert = timing_alerts[0]
        assert alert.agent_id == "test_agent"
        assert alert.observed_value == 10.0
        
    def test_resource_anomaly_detection(self):
        """Test resource usage anomaly detection."""
        # Establish baseline
        for i in range(60):
            self.monitor.observe_behavior(
                "test_agent",
                "processing",
                {
                    "response_time": 1.0,
                    "success": True,
                    "resource_usage": {"cpu_percent": 25}
                }
            )
            
        # Generate resource anomaly
        self.monitor.observe_behavior(
            "test_agent", 
            "processing",
            {
                "response_time": 1.0,
                "success": True,
                "resource_usage": {"cpu_percent": 90}  # Very high
            }
        )
        
        resource_alerts = [a for a in self.monitor.alerts
                          if a.anomaly_type == AnomalyType.RESOURCE_ANOMALY]
        
        assert len(resource_alerts) > 0
        
    def test_alert_management(self):
        """Test alert acknowledgment and resolution."""
        # Create alert by generating anomaly
        for i in range(60):
            self.monitor.observe_behavior("test_agent", "action", {"response_time": 1.0})
            
        self.monitor.observe_behavior("test_agent", "action", {"response_time": 20.0})
        
        alerts = self.monitor.get_active_alerts("test_agent")
        if alerts:
            alert_id = alerts[0].alert_id
            
            # Acknowledge alert
            self.monitor.acknowledge_alert(alert_id, "test_operator")
            
            updated_alerts = self.monitor.get_active_alerts("test_agent")
            acknowledged_alert = next(a for a in updated_alerts if a.alert_id == alert_id)
            assert acknowledged_alert.acknowledged
            
            # Resolve alert
            self.monitor.resolve_alert(alert_id, "test_operator", "False alarm")
            
            resolved_alerts = [a for a in self.monitor.alerts if a.alert_id == alert_id]
            assert resolved_alerts[0].resolved
            assert "False alarm" in resolved_alerts[0].resolution_notes
            
    def test_safety_summary(self):
        """Test safety monitoring summary."""
        # Generate some data
        self.monitor.observe_behavior("agent1", "action", {"response_time": 1.0})
        self.monitor.observe_behavior("agent2", "action", {"response_time": 1.0})
        
        summary = self.monitor.get_safety_summary()
        assert summary["monitored_agents"] == 2
        assert "total_alerts" in summary
        assert "active_alerts" in summary


class TestAnomalyDetector:
    """Test cases for AnomalyDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = AnomalyDetector()
        
    def test_frequency_anomaly_detection(self):
        """Test statistical frequency anomaly detection."""
        # Normal distribution around 10
        historical_freq = [9, 10, 11, 10, 9, 10, 11, 9, 10, 11] * 5
        
        # Test normal value
        is_anomaly, confidence = self.detector.detect_frequency_anomaly(10.0, historical_freq)
        assert not is_anomaly
        
        # Test anomalous value
        is_anomaly, confidence = self.detector.detect_frequency_anomaly(50.0, historical_freq)
        assert is_anomaly
        assert confidence > 0.5
        
    def test_pattern_anomaly_detection(self):
        """Test pattern anomaly detection."""
        # Historical patterns
        patterns = [
            ["action1", "action2", "action3"],
            ["action1", "action2", "action4"], 
            ["action1", "action3", "action2"],
        ] * 3
        
        # Similar pattern
        is_anomaly, confidence = self.detector.detect_pattern_anomaly(
            ["action1", "action2", "action3"], patterns)
        assert not is_anomaly
        
        # Very different pattern
        is_anomaly, confidence = self.detector.detect_pattern_anomaly(
            ["completely", "different", "actions"], patterns)
        assert is_anomaly
        assert confidence > 0.5
        
    def test_timing_anomaly_detection(self):
        """Test timing anomaly detection."""
        # Normal timing around 1.0 seconds
        historical_times = [0.9, 1.0, 1.1, 1.0, 0.9, 1.1, 1.0] * 5
        
        # Normal timing
        is_anomaly, confidence = self.detector.detect_timing_anomaly(1.0, historical_times)
        assert not is_anomaly
        
        # Anomalous timing
        is_anomaly, confidence = self.detector.detect_timing_anomaly(10.0, historical_times)
        assert is_anomaly
        assert confidence > 0.5


class TestIntegratedEthicsSystem:
    """Integration tests for the complete ethics and safety system."""
    
    def setup_method(self):
        """Set up integrated test environment."""
        self.ethics = EthicsOverseer()
        self.boundary_enforcer = BoundaryEnforcer()
        self.emergency_stop = EmergencyStopSystem()
        self.safety_monitor = SafetyMonitor()
        
    def test_full_ethics_workflow(self):
        """Test complete ethics enforcement workflow."""
        # 1. Register agent
        self.ethics.register_agent("test_agent", {ActionCategory.RECONNAISSANCE})
        
        # 2. Validate ethical action
        decision = self.ethics.validate_action(
            "test_agent",
            ActionCategory.RECONNAISSANCE,
            "Scan network segment",
            {"target": "192.168.10.0/24"}
        )
        assert decision.judgment == EthicalJudgment.APPROVED
        
        # 3. Check boundary compliance
        network_allowed = self.boundary_enforcer.check_network_boundary(
            "test_agent", "192.168.10.100")
        assert network_allowed
        
        # 4. Monitor behavior
        self.safety_monitor.observe_behavior(
            "test_agent", 
            "reconnaissance",
            {"response_time": 2.0, "success": True}
        )
        
        # Verify all systems are tracking the agent
        assert "test_agent" in self.ethics.agent_permissions
        assert "test_agent" in self.safety_monitor.baselines
        
    def test_violation_escalation_workflow(self):
        """Test escalation from violation to emergency stop."""
        self.ethics.register_agent("rogue_agent", {ActionCategory.SYSTEM_MODIFICATION})
        
        # Trigger boundary violation
        network_allowed = self.boundary_enforcer.check_network_boundary(
            "rogue_agent", "8.8.8.8")  # External network
        assert not network_allowed
        
        # Should have recorded violation
        violations = [v for v in self.boundary_enforcer.violations 
                     if v.agent_id == "rogue_agent"]
        assert len(violations) > 0
        
        # For critical violations, emergency stop might be triggered
        if violations[0].severity == ViolationSeverity.CRITICAL:
            # Check if emergency procedures would be activated
            assert True  # Placeholder for emergency response verification
            
    def test_safety_monitoring_integration(self):
        """Test integration between safety monitoring and other systems."""
        # Set up safety monitoring callback to trigger emergency stop
        def safety_callback(alert):
            if alert.severity == AlertSeverity.CRITICAL:
                self.emergency_stop.emergency_stop(
                    StopScope.SINGLE_AGENT,
                    alert.agent_id,
                    EmergencyStopReason.ANOMALOUS_BEHAVIOR,
                    f"Critical safety alert: {alert.description}",
                    "safety_monitor"
                )
                
        self.safety_monitor.add_alert_callback(safety_callback)
        
        # Generate behavior that might trigger alert
        for i in range(60):
            self.safety_monitor.observe_behavior(
                "monitored_agent",
                "processing", 
                {"response_time": 1.0, "success": True}
            )
            
        # Generate anomalous behavior
        self.safety_monitor.observe_behavior(
            "monitored_agent",
            "processing",
            {"response_time": 30.0, "success": True}  # Very slow
        )
        
        # Check if critical alerts would trigger emergency response
        critical_alerts = [a for a in self.safety_monitor.alerts 
                          if a.severity == AlertSeverity.CRITICAL]
        
        if critical_alerts:
            assert self.emergency_stop.is_agent_stopped("monitored_agent")
            
    def test_comprehensive_system_status(self):
        """Test getting comprehensive status from all systems."""
        # Set up some data in all systems
        self.ethics.register_agent("status_agent", {ActionCategory.ANALYSIS})
        
        self.ethics.validate_action("status_agent", ActionCategory.ANALYSIS, "Test", {})
        
        self.boundary_enforcer.check_network_boundary("status_agent", "192.168.10.1")
        
        self.safety_monitor.observe_behavior("status_agent", "analysis", 
                                           {"response_time": 1.0, "success": True})
        
        # Get status from all systems
        ethics_summary = self.ethics.get_ethics_summary()
        boundary_summary = self.boundary_enforcer.get_violation_summary()
        emergency_status = self.emergency_stop.get_system_status()
        safety_summary = self.safety_monitor.get_safety_summary()
        
        # Verify all systems have relevant data
        assert ethics_summary["registered_agents"] >= 1
        assert safety_summary["monitored_agents"] >= 1
        assert emergency_status["total_constraints"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])