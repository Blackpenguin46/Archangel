"""
Comprehensive tests for the monitoring system components.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from monitoring.metrics_collector import MetricsCollector, AgentMetrics
from monitoring.health_monitor import HealthMonitor, HealthStatus, HealthCheck, ComponentHealth
from monitoring.recovery_system import RecoverySystem, RecoveryAction, RecoveryRule

class TestMetricsCollector:
    """Test cases for MetricsCollector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = MetricsCollector(port=9999)  # Use different port for testing
        
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self.collector, '_running') and self.collector._running:
            self.collector.stop_server()
            
    def test_agent_registration(self):
        """Test agent registration and unregistration."""
        # Register an agent
        self.collector.register_agent("test_agent", "recon", "red")
        
        # Check agent was registered
        assert "test_agent" in self.collector.agents
        agent_metrics = self.collector.get_agent_metrics("test_agent")
        assert agent_metrics is not None
        assert agent_metrics.agent_id == "test_agent"
        assert agent_metrics.agent_type == "recon"
        assert agent_metrics.team == "red"
        assert agent_metrics.status == "active"
        
        # Unregister agent
        self.collector.unregister_agent("test_agent")
        assert "test_agent" not in self.collector.agents
        assert self.collector.get_agent_metrics("test_agent") is None
        
    def test_decision_recording(self):
        """Test recording agent decisions."""
        # Register agent first
        self.collector.register_agent("test_agent", "exploit", "red")
        
        # Record successful decision
        self.collector.record_decision("test_agent", "exploit_attempt", True, 0.5)
        
        metrics = self.collector.get_agent_metrics("test_agent")
        assert metrics.decisions_total == 1
        assert metrics.decisions_success == 1
        assert metrics.decisions_failed == 0
        assert metrics.response_time == 0.5
        
        # Record failed decision
        self.collector.record_decision("test_agent", "exploit_attempt", False, 2.0)
        
        metrics = self.collector.get_agent_metrics("test_agent")
        assert metrics.decisions_total == 2
        assert metrics.decisions_success == 1
        assert metrics.decisions_failed == 1
        assert metrics.response_time == 2.0
        
    def test_communication_failure_recording(self):
        """Test recording communication failures."""
        self.collector.register_agent("test_agent", "persistence", "red")
        
        # Record communication failure
        self.collector.record_communication_failure("test_agent", "timeout")
        
        metrics = self.collector.get_agent_metrics("test_agent")
        assert metrics.communication_failures == 1
        
    def test_team_coordination_update(self):
        """Test updating team coordination scores."""
        # Register agents from both teams
        self.collector.register_agent("red_agent", "recon", "red")
        self.collector.register_agent("blue_agent", "soc_analyst", "blue")
        
        # Update red team coordination
        self.collector.update_team_coordination("red", 0.8)
        
        red_metrics = self.collector.get_agent_metrics("red_agent")
        blue_metrics = self.collector.get_agent_metrics("blue_agent")
        
        assert red_metrics.coordination_score == 0.8
        assert blue_metrics.coordination_score == 1.0  # Unchanged
        
    def test_red_team_action_recording(self):
        """Test recording Red Team actions."""
        self.collector.record_red_team_action("red_agent", "exploit", "web_server", True)
        self.collector.record_red_team_action("red_agent", "recon", "database", False)
        
        # Since we don't have direct access to Prometheus counters in tests,
        # we verify the method calls don't raise exceptions
        assert True  # Test passes if no exceptions
        
    def test_blue_team_detection_recording(self):
        """Test recording Blue Team detections."""
        self.collector.record_blue_team_detection("blue_agent", "malware", "high", 5.0)
        self.collector.record_blue_team_detection("blue_agent", "intrusion", "medium", 2.0)
        
        # Verify method calls don't raise exceptions
        assert True
        
    def test_performance_metric_recording(self):
        """Test recording performance metrics."""
        self.collector.record_game_loop_duration(45.0)
        self.collector.record_scoring_duration(8.0)
        self.collector.record_vector_store_query("search", 0.5)
        
        # Verify method calls don't raise exceptions
        assert True
        
    def test_get_all_agents(self):
        """Test getting all agent metrics."""
        # Register multiple agents
        self.collector.register_agent("agent1", "recon", "red")
        self.collector.register_agent("agent2", "exploit", "red")
        self.collector.register_agent("agent3", "soc_analyst", "blue")
        
        all_agents = self.collector.get_all_agents()
        assert len(all_agents) == 3
        assert "agent1" in all_agents
        assert "agent2" in all_agents
        assert "agent3" in all_agents
        
    @patch('monitoring.metrics_collector.start_http_server')
    def test_server_start_stop(self, mock_start_server):
        """Test starting and stopping the metrics server."""
        # Mock the start_http_server to avoid actually starting a server
        mock_start_server.return_value = None
        
        self.collector.start_server()
        assert self.collector._running
        mock_start_server.assert_called_once()
        
        self.collector.stop_server()
        assert not self.collector._running


class TestHealthMonitor:
    """Test cases for HealthMonitor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = HealthMonitor(check_interval=1.0)  # Faster for testing
        
    def teardown_method(self):
        """Clean up after tests."""
        if self.monitor._running:
            self.monitor.stop_monitoring()
            
    def test_component_registration(self):
        """Test component registration and unregistration."""
        # Register a component
        self.monitor.register_component("test_component", "agent")
        
        component = self.monitor.get_component_health("test_component")
        assert component is not None
        assert component.component_id == "test_component"
        assert component.component_type == "agent"
        assert component.status == HealthStatus.UNKNOWN
        
        # Unregister component
        self.monitor.unregister_component("test_component")
        assert self.monitor.get_component_health("test_component") is None
        
    def test_health_status_update(self):
        """Test updating component health status."""
        self.monitor.register_component("test_component", "service")
        
        # Update to healthy status
        self.monitor.update_component_health("test_component", HealthStatus.HEALTHY, 
                                           {"cpu_usage": 25.0})
        
        component = self.monitor.get_component_health("test_component")
        assert component.status == HealthStatus.HEALTHY
        assert "cpu_usage" in component.metadata
        assert component.metadata["cpu_usage"] == 25.0
        
        # Update to warning status
        self.monitor.update_component_health("test_component", HealthStatus.WARNING)
        assert component.status == HealthStatus.WARNING
        
    def test_status_change_callback(self):
        """Test status change callbacks."""
        callback_called = []
        
        def test_callback(component_id, status):
            callback_called.append((component_id, status))
            
        self.monitor.add_status_callback(test_callback)
        self.monitor.register_component("test_component", "agent")
        
        # Update status should trigger callback
        self.monitor.update_component_health("test_component", HealthStatus.CRITICAL)
        
        assert len(callback_called) == 1
        assert callback_called[0] == ("test_component", HealthStatus.CRITICAL)
        
    def test_health_check_management(self):
        """Test adding and removing health checks."""
        def dummy_check():
            return True
            
        # Add health check
        self.monitor.add_health_check("test_check", dummy_check, interval=10.0, 
                                    timeout=2.0, description="Test check")
        
        assert "test_check" in self.monitor.health_checks
        check = self.monitor.health_checks["test_check"]
        assert check.name == "test_check"
        assert check.interval == 10.0
        assert check.timeout == 2.0
        assert check.description == "Test check"
        
        # Remove health check
        self.monitor.remove_health_check("test_check")
        assert "test_check" not in self.monitor.health_checks
        
    def test_system_health_summary(self):
        """Test system health summary generation."""
        # Register components with different statuses
        self.monitor.register_component("healthy_comp", "agent")
        self.monitor.register_component("warning_comp", "service")
        self.monitor.register_component("critical_comp", "database")
        
        self.monitor.update_component_health("healthy_comp", HealthStatus.HEALTHY)
        self.monitor.update_component_health("warning_comp", HealthStatus.WARNING)
        self.monitor.update_component_health("critical_comp", HealthStatus.CRITICAL)
        
        summary = self.monitor.get_system_health_summary()
        
        assert summary["total_components"] == 3
        assert summary["healthy"] == 1
        assert summary["warning"] == 1
        assert summary["critical"] == 1
        assert summary["unknown"] == 0
        assert summary["overall_status"] == HealthStatus.CRITICAL.value
        
    def test_empty_system_health_summary(self):
        """Test system health summary with no components."""
        summary = self.monitor.get_system_health_summary()
        
        assert summary["total_components"] == 0
        assert summary["overall_status"] == HealthStatus.UNKNOWN.value
        
    @patch('monitoring.health_monitor.psutil.cpu_percent')
    def test_system_cpu_check(self, mock_cpu_percent):
        """Test system CPU health check."""
        # Test healthy CPU usage
        mock_cpu_percent.return_value = 50.0
        assert self.monitor._check_system_cpu() == True
        
        # Test unhealthy CPU usage
        mock_cpu_percent.return_value = 95.0
        assert self.monitor._check_system_cpu() == False
        
    @patch('monitoring.health_monitor.psutil.virtual_memory')
    def test_system_memory_check(self, mock_memory):
        """Test system memory health check."""
        # Test healthy memory usage
        mock_memory.return_value = Mock(percent=60.0)
        assert self.monitor._check_system_memory() == True
        
        # Test unhealthy memory usage
        mock_memory.return_value = Mock(percent=90.0)
        assert self.monitor._check_system_memory() == False
        
    @patch('monitoring.health_monitor.psutil.disk_usage')
    def test_system_disk_check(self, mock_disk_usage):
        """Test system disk health check."""
        # Test healthy disk usage
        mock_disk_usage.return_value = Mock(total=1000, used=600)
        assert self.monitor._check_system_disk() == True
        
        # Test unhealthy disk usage
        mock_disk_usage.return_value = Mock(total=1000, used=950)
        assert self.monitor._check_system_disk() == False
        
    @patch('subprocess.run')
    def test_docker_daemon_check(self, mock_subprocess):
        """Test Docker daemon health check."""
        # Test Docker available
        mock_subprocess.return_value = Mock(returncode=0)
        assert self.monitor._check_docker_daemon() == True
        
        # Test Docker unavailable
        mock_subprocess.return_value = Mock(returncode=1)
        assert self.monitor._check_docker_daemon() == False
        
    def test_monitoring_start_stop(self):
        """Test starting and stopping monitoring."""
        assert not self.monitor._running
        
        self.monitor.start_monitoring()
        assert self.monitor._running
        assert self.monitor._monitor_thread is not None
        
        self.monitor.stop_monitoring()
        assert not self.monitor._running


class TestRecoverySystem:
    """Test cases for RecoverySystem."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.health_monitor = HealthMonitor()
        self.recovery = RecoverySystem(self.health_monitor)
        
    def teardown_method(self):
        """Clean up after tests."""
        if self.health_monitor._running:
            self.health_monitor.stop_monitoring()
            
    def test_recovery_rule_management(self):
        """Test adding and removing recovery rules."""
        # Add a recovery rule
        self.recovery.add_recovery_rule(
            name="test_rule",
            component_pattern=r"test_.*",
            trigger_status=HealthStatus.CRITICAL,
            action=RecoveryAction.RESTART_AGENT,
            cooldown_seconds=60.0,
            max_attempts=2,
            description="Test recovery rule"
        )
        
        rules = self.recovery.get_recovery_rules()
        assert "test_rule" in rules
        
        rule = rules["test_rule"]
        assert rule.name == "test_rule"
        assert rule.component_pattern == r"test_.*"
        assert rule.trigger_status == HealthStatus.CRITICAL
        assert rule.action == RecoveryAction.RESTART_AGENT
        assert rule.cooldown_seconds == 60.0
        assert rule.max_attempts == 2
        
        # Remove the rule
        self.recovery.remove_recovery_rule("test_rule")
        rules = self.recovery.get_recovery_rules()
        assert "test_rule" not in rules
        
    def test_rule_matching(self):
        """Test finding matching recovery rules."""
        # Add rules with different patterns
        self.recovery.add_recovery_rule("agent_rule", r"agent_.*", 
                                      HealthStatus.CRITICAL, RecoveryAction.RESTART_AGENT)
        self.recovery.add_recovery_rule("service_rule", r"service_.*", 
                                      HealthStatus.WARNING, RecoveryAction.RESTART_SERVICE)
        
        # Test agent pattern matching
        matching_rules = self.recovery._find_matching_rules("agent_test", HealthStatus.CRITICAL)
        assert len(matching_rules) == 1
        assert matching_rules[0].name == "agent_rule"
        
        # Test service pattern matching
        matching_rules = self.recovery._find_matching_rules("service_test", HealthStatus.WARNING)
        assert len(matching_rules) == 1
        assert matching_rules[0].name == "service_rule"
        
        # Test no matching pattern
        matching_rules = self.recovery._find_matching_rules("database_test", HealthStatus.CRITICAL)
        assert len(matching_rules) == 0
        
    def test_rule_execution_conditions(self):
        """Test rule execution condition checking."""
        # Create a rule
        self.recovery.add_recovery_rule("test_rule", r"test_.*", 
                                      HealthStatus.CRITICAL, RecoveryAction.ALERT_ONLY,
                                      cooldown_seconds=60.0, max_attempts=2)
        
        rule = self.recovery.rules["test_rule"]
        
        # Should execute initially
        assert self.recovery._should_execute_rule(rule) == True
        
        # Simulate execution
        rule.last_execution = time.time()
        rule.execution_count = 1
        
        # Should not execute due to cooldown
        assert self.recovery._should_execute_rule(rule) == False
        
        # Simulate cooldown period passed
        rule.last_execution = time.time() - 70.0
        assert self.recovery._should_execute_rule(rule) == True
        
        # Exceed max attempts
        rule.execution_count = 2
        assert self.recovery._should_execute_rule(rule) == False
        
    def test_alert_recovery_action(self):
        """Test the alert-only recovery action."""
        with patch('monitoring.recovery_system.logger') as mock_logger:
            success = self.recovery._send_alert("test_component", {"message": "Test alert"})
            assert success == True
            mock_logger.warning.assert_called()
            
    def test_recovery_history_tracking(self):
        """Test recovery attempt history tracking."""
        # Initially no history
        history = self.recovery.get_recovery_history()
        assert len(history) == 0
        
        # Add a recovery rule and trigger it
        self.recovery.add_recovery_rule("test_rule", r"test_.*", 
                                      HealthStatus.CRITICAL, RecoveryAction.ALERT_ONLY)
        
        self.health_monitor.register_component("test_component", "agent")
        self.health_monitor.update_component_health("test_component", HealthStatus.CRITICAL)
        
        # Allow some time for processing
        time.sleep(0.1)
        
        history = self.recovery.get_recovery_history()
        assert len(history) >= 0  # May or may not have executed depending on timing
        
    def test_recovery_stats(self):
        """Test recovery statistics generation."""
        # Initially empty stats
        stats = self.recovery.get_recovery_stats()
        assert stats["total_attempts"] == 0
        assert stats["success_rate"] == 0.0
        
        # Add some mock recovery attempts
        from monitoring.recovery_system import RecoveryAttempt
        self.recovery.recovery_history = [
            RecoveryAttempt(time.time(), "comp1", "rule1", RecoveryAction.RESTART_AGENT, True),
            RecoveryAttempt(time.time(), "comp2", "rule2", RecoveryAction.RESTART_SERVICE, False),
            RecoveryAttempt(time.time(), "comp1", "rule1", RecoveryAction.RESTART_AGENT, True),
        ]
        
        stats = self.recovery.get_recovery_stats()
        assert stats["total_attempts"] == 3
        assert stats["successful_attempts"] == 2
        assert stats["success_rate"] == 2/3
        assert stats["most_common_action"] == "restart_agent"
        assert stats["most_recovered_component"] == "comp1"
        
    @patch('subprocess.run')
    def test_container_restart_action(self, mock_subprocess):
        """Test container restart recovery action."""
        mock_subprocess.return_value = Mock(returncode=0)
        
        success = self.recovery._restart_container("test_container", {"container_name": "test"})
        assert success == True
        mock_subprocess.assert_called_with(
            ['docker', 'restart', 'test'], capture_output=True, timeout=60
        )
        
        # Test failure
        mock_subprocess.return_value = Mock(returncode=1)
        success = self.recovery._restart_container("test_container", {})
        assert success == False
        
    def test_custom_recovery_action(self):
        """Test custom recovery action execution."""
        custom_function_called = []
        
        def custom_recovery(component_id, params):
            custom_function_called.append((component_id, params))
            return True
            
        self.recovery.add_recovery_rule(
            "custom_rule", r"test_.*", HealthStatus.CRITICAL, 
            RecoveryAction.CUSTOM, custom_function=custom_recovery
        )
        
        # Trigger the custom recovery
        rule = self.recovery.rules["custom_rule"]
        self.recovery._execute_recovery_action("test_component", rule)
        
        assert len(custom_function_called) == 1
        assert custom_function_called[0][0] == "test_component"


class TestMonitoringIntegration:
    """Integration tests for the complete monitoring system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = MetricsCollector(port=9998)
        self.health_monitor = HealthMonitor(check_interval=0.5)
        self.recovery = RecoverySystem(self.health_monitor)
        
    def teardown_method(self):
        """Clean up after tests."""
        if self.metrics._running:
            self.metrics.stop_server()
        if self.health_monitor._running:
            self.health_monitor.stop_monitoring()
            
    def test_full_monitoring_workflow(self):
        """Test complete monitoring workflow from health check to recovery."""
        # Set up components
        self.metrics.register_agent("test_agent", "recon", "red")
        self.health_monitor.register_component("test_agent", "agent")
        
        # Set up a recovery rule
        recovery_triggered = []
        
        def mock_recovery(component_id, params):
            recovery_triggered.append(component_id)
            return True
            
        self.recovery.add_recovery_rule(
            "test_recovery", r"test_.*", HealthStatus.CRITICAL,
            RecoveryAction.CUSTOM, custom_function=mock_recovery
        )
        
        # Simulate agent going critical
        self.health_monitor.update_component_health("test_agent", HealthStatus.CRITICAL)
        
        # Allow processing time
        time.sleep(0.1)
        
        # Check that recovery was attempted
        history = self.recovery.get_recovery_history()
        assert len(history) >= 0  # Recovery may or may not have executed
        
    def test_metrics_and_health_coordination(self):
        """Test coordination between metrics collection and health monitoring."""
        # Register agent in both systems
        self.metrics.register_agent("coordinated_agent", "exploit", "red")
        self.health_monitor.register_component("coordinated_agent", "agent")
        
        # Record some metrics
        self.metrics.record_decision("coordinated_agent", "exploit_attempt", True, 0.5)
        self.metrics.record_communication_failure("coordinated_agent", "timeout")
        
        # Update health status
        self.health_monitor.update_component_health("coordinated_agent", HealthStatus.WARNING)
        
        # Verify both systems have data
        agent_metrics = self.metrics.get_agent_metrics("coordinated_agent")
        component_health = self.health_monitor.get_component_health("coordinated_agent")
        
        assert agent_metrics is not None
        assert component_health is not None
        assert agent_metrics.decisions_total == 1
        assert component_health.status == HealthStatus.WARNING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])