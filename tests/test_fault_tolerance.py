"""
Comprehensive tests for the fault tolerance system.
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import random

from fault_tolerance.heartbeat_monitor import HeartbeatMonitor, HeartbeatConfig, AgentState
from fault_tolerance.circuit_breaker import CircuitBreaker, CircuitState, CircuitConfig, CircuitBreakerOpenError
from fault_tolerance.retry_manager import RetryManager, RetryPolicy, BackoffStrategy
from fault_tolerance.graceful_degradation import GracefulDegradation, DegradationLevel, ComponentCriticality
from fault_tolerance.recovery_strategies import RecoveryStrategyManager, RecoveryContext, RestartStrategy

class TestHeartbeatMonitor:
    """Test cases for HeartbeatMonitor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        config = HeartbeatConfig(
            interval=0.1,  # Fast for testing
            timeout=0.5,
            missed_threshold=2,
            recovery_threshold=1
        )
        self.monitor = HeartbeatMonitor(config)
        
    def teardown_method(self):
        """Clean up after tests."""
        if self.monitor._running:
            self.monitor.stop_monitoring()
            
    def test_agent_registration(self):
        """Test agent registration and unregistration."""
        # Register agent
        self.monitor.register_agent("test_agent", {"type": "recon", "team": "red"})
        
        # Check agent was registered
        assert "test_agent" in self.monitor.agents
        heartbeat = self.monitor.get_agent_heartbeat("test_agent")
        assert heartbeat is not None
        assert heartbeat.agent_id == "test_agent"
        assert heartbeat.state == AgentState.UNKNOWN
        
        # Unregister agent
        self.monitor.unregister_agent("test_agent")
        assert "test_agent" not in self.monitor.agents
        assert self.monitor.get_agent_heartbeat("test_agent") is None
        
    def test_heartbeat_recording(self):
        """Test recording heartbeats from agents."""
        self.monitor.register_agent("test_agent")
        
        # Record heartbeat
        self.monitor.record_heartbeat("test_agent", response_time=0.5, metadata={"load": 0.3})
        
        heartbeat = self.monitor.get_agent_heartbeat("test_agent")
        assert heartbeat.last_response_time == 0.5
        assert heartbeat.metadata["load"] == 0.3
        assert heartbeat.missed_count == 0
        assert heartbeat.state == AgentState.HEALTHY
        
    def test_heartbeat_timeout_detection(self):
        """Test detection of heartbeat timeouts."""
        self.monitor.register_agent("test_agent")
        self.monitor.start_monitoring()
        
        # Wait longer than timeout period
        time.sleep(0.6)
        
        heartbeat = self.monitor.get_agent_heartbeat("test_agent")
        assert heartbeat.missed_count > 0
        assert heartbeat.state in [AgentState.DEGRADED, AgentState.CRITICAL, AgentState.FAILED]
        
    def test_state_change_callbacks(self):
        """Test state change callback notifications."""
        callback_calls = []
        
        def test_callback(agent_id, old_state, new_state):
            callback_calls.append((agent_id, old_state, new_state))
            
        self.monitor.add_state_change_callback(test_callback)
        self.monitor.register_agent("test_agent")
        
        # Record heartbeat to trigger state change
        self.monitor.record_heartbeat("test_agent", response_time=0.2)
        
        assert len(callback_calls) > 0
        assert callback_calls[0][0] == "test_agent"
        
    def test_performance_degradation_detection(self):
        """Test detection of performance degradation."""
        config = HeartbeatConfig(degraded_threshold=1.0)
        self.monitor = HeartbeatMonitor(config)
        self.monitor.register_agent("test_agent")
        
        # Record slow response time
        self.monitor.record_heartbeat("test_agent", response_time=2.0)
        
        heartbeat = self.monitor.get_agent_heartbeat("test_agent")
        assert heartbeat.state == AgentState.DEGRADED
        
    def test_monitoring_summary(self):
        """Test monitoring summary generation."""
        # Register multiple agents
        self.monitor.register_agent("agent1")
        self.monitor.register_agent("agent2")
        self.monitor.register_agent("agent3")
        
        # Record some heartbeats
        self.monitor.record_heartbeat("agent1", response_time=0.5)
        self.monitor.record_heartbeat("agent2", response_time=0.3)
        # agent3 has no heartbeats (unknown state)
        
        summary = self.monitor.get_monitoring_summary()
        assert summary["total_agents"] == 3
        assert summary["healthy"] >= 1
        assert summary["unknown"] >= 1
        
    def test_failed_agent_detection(self):
        """Test getting failed agents."""
        self.monitor.register_agent("healthy_agent")
        self.monitor.register_agent("failed_agent")
        
        # Mark one as failed
        self.monitor.agents["failed_agent"].state = AgentState.FAILED
        
        failed_agents = self.monitor.get_failed_agents()
        assert "failed_agent" in failed_agents
        assert "healthy_agent" not in failed_agents


class TestCircuitBreaker:
    """Test cases for CircuitBreaker."""
    
    def setup_method(self):
        """Set up test fixtures."""
        config = CircuitConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout=0.1  # Short timeout for testing
        )
        self.circuit = CircuitBreaker("test_circuit", config)
        
    def test_initial_state(self):
        """Test initial circuit breaker state."""
        assert self.circuit.get_state() == CircuitState.CLOSED
        stats = self.circuit.get_stats()
        assert stats.total_requests == 0
        
    def test_successful_calls(self):
        """Test successful function calls through circuit breaker."""
        @self.circuit
        def test_function():
            return "success"
            
        result = test_function()
        assert result == "success"
        
        stats = self.circuit.get_stats()
        assert stats.total_requests == 1
        assert stats.successful_requests == 1
        
    def test_failed_calls(self):
        """Test failed function calls through circuit breaker."""
        @self.circuit
        def failing_function():
            raise ValueError("Test error")
            
        # Call function multiple times to exceed failure threshold
        for i in range(3):
            with pytest.raises(ValueError):
                failing_function()
                
        # Circuit should still be closed (exactly at threshold)
        assert self.circuit.get_state() == CircuitState.CLOSED
        
        # One more failure should open the circuit
        with pytest.raises(ValueError):
            failing_function()
            
        assert self.circuit.get_state() == CircuitState.OPEN
        
    def test_open_circuit_rejection(self):
        """Test that open circuit rejects calls."""
        # Force circuit open
        self.circuit.force_open()
        
        def test_function():
            return "should not execute"
            
        # Calls should be rejected
        with pytest.raises(CircuitBreakerOpenError):
            self.circuit.call(test_function)
            
    def test_half_open_transition(self):
        """Test transition from open to half-open state."""
        # Force circuit open
        self.circuit.force_open()
        assert self.circuit.get_state() == CircuitState.OPEN
        
        # Wait for timeout
        time.sleep(0.2)
        
        def test_function():
            return "success"
            
        # First call after timeout should work (transition to half-open)
        result = self.circuit.call(test_function)
        assert result == "success"
        
    def test_recovery_from_half_open(self):
        """Test recovery from half-open to closed state."""
        # Get circuit to half-open state
        self.circuit.force_open()
        time.sleep(0.2)
        
        @self.circuit
        def successful_function():
            return "success"
            
        # Make successful calls to reach success threshold
        for i in range(2):
            result = successful_function()
            assert result == "success"
            
        assert self.circuit.get_state() == CircuitState.CLOSED
        
    @pytest.mark.asyncio
    async def test_async_function_support(self):
        """Test circuit breaker with async functions."""
        @self.circuit
        async def async_function():
            await asyncio.sleep(0.01)
            return "async success"
            
        result = await async_function()
        assert result == "async success"
        
    def test_circuit_breaker_reset(self):
        """Test resetting circuit breaker state."""
        # Generate some failures
        @self.circuit
        def failing_function():
            raise ValueError("Test error")
            
        for i in range(5):
            try:
                failing_function()
            except:
                pass
                
        # Reset circuit
        self.circuit.reset()
        assert self.circuit.get_state() == CircuitState.CLOSED
        
        stats = self.circuit.get_stats()
        assert stats.total_requests == 0


class TestRetryManager:
    """Test cases for RetryManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        policy = RetryPolicy(
            max_attempts=3,
            base_delay=0.01,  # Fast for testing
            backoff_strategy=BackoffStrategy.EXPONENTIAL
        )
        self.retry_manager = RetryManager(policy)
        
    def test_successful_execution(self):
        """Test successful function execution without retries."""
        def successful_function():
            return "success"
            
        result = self.retry_manager.retry(successful_function)
        assert result == "success"
        
        stats = self.retry_manager.get_stats()
        assert stats.successful_attempts == 1
        assert stats.total_retries == 0
        
    def test_retry_on_failure(self):
        """Test retry logic on function failures."""
        call_count = 0
        
        def sometimes_failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"
            
        result = self.retry_manager.retry(sometimes_failing_function)
        assert result == "success"
        assert call_count == 3
        
        stats = self.retry_manager.get_stats()
        assert stats.successful_attempts == 1
        assert stats.total_retries == 2
        
    def test_max_attempts_exceeded(self):
        """Test behavior when max attempts are exceeded."""
        def always_failing_function():
            raise ConnectionError("Always fails")
            
        with pytest.raises(ConnectionError):
            self.retry_manager.retry(always_failing_function)
            
        stats = self.retry_manager.get_stats()
        assert stats.failed_attempts == 1
        assert stats.total_retries == 2  # 3 attempts = 2 retries
        
    def test_non_retryable_exception(self):
        """Test that non-retryable exceptions are not retried."""
        policy = RetryPolicy(
            max_attempts=3,
            retry_on_exceptions=(ConnectionError,),
            stop_on_exceptions=(ValueError,)
        )
        retry_manager = RetryManager(policy)
        
        def function_with_value_error():
            raise ValueError("Non-retryable error")
            
        with pytest.raises(ValueError):
            retry_manager.retry(function_with_value_error)
            
        stats = retry_manager.get_stats()
        assert stats.total_retries == 0  # No retries attempted
        
    def test_backoff_strategies(self):
        """Test different backoff strategies."""
        strategies = [
            BackoffStrategy.FIXED,
            BackoffStrategy.LINEAR,
            BackoffStrategy.EXPONENTIAL,
            BackoffStrategy.EXPONENTIAL_JITTER
        ]
        
        for strategy in strategies:
            policy = RetryPolicy(
                max_attempts=3,
                base_delay=0.01,
                backoff_strategy=strategy
            )
            retry_manager = RetryManager(policy)
            
            # Test delay calculation
            delay1 = retry_manager._calculate_delay(1)
            delay2 = retry_manager._calculate_delay(2)
            
            assert delay1 > 0
            assert delay2 > 0
            
            if strategy == BackoffStrategy.FIXED:
                assert delay1 == delay2
            elif strategy == BackoffStrategy.LINEAR:
                assert delay2 > delay1
            elif strategy in [BackoffStrategy.EXPONENTIAL, BackoffStrategy.EXPONENTIAL_JITTER]:
                assert delay2 > delay1
                
    @pytest.mark.asyncio
    async def test_async_retry(self):
        """Test retry logic with async functions."""
        call_count = 0
        
        async def async_failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Network error")
            return "async success"
            
        result = await self.retry_manager.retry_async(async_failing_function)
        assert result == "async success"
        assert call_count == 2
        
    def test_decorator_usage(self):
        """Test using retry manager as decorator."""
        @self.retry_manager
        def decorated_function():
            return "decorated success"
            
        result = decorated_function()
        assert result == "decorated success"


class TestGracefulDegradation:
    """Test cases for GracefulDegradation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.degradation = GracefulDegradation()
        
    def test_component_registration(self):
        """Test component registration."""
        self.degradation.register_component(
            "test_component",
            ComponentCriticality.CRITICAL,
            {"type": "database"}
        )
        
        assert "test_component" in self.degradation.components
        component = self.degradation.components["test_component"]
        assert component.criticality == ComponentCriticality.CRITICAL
        assert component.is_healthy == True
        
    def test_feature_registration(self):
        """Test feature registration."""
        self.degradation.register_component("database", ComponentCriticality.CRITICAL)
        self.degradation.register_feature(
            "analytics",
            required_components={"database"},
            resource_cost=2.0,
            user_impact=0.3
        )
        
        assert "analytics" in self.degradation.features
        feature = self.degradation.features["analytics"]
        assert feature.required_components == {"database"}
        assert feature.resource_cost == 2.0
        
    def test_component_health_update(self):
        """Test updating component health status."""
        self.degradation.register_component("test_component", ComponentCriticality.CRITICAL)
        
        # Update to unhealthy
        self.degradation.update_component_health("test_component", False)
        
        component = self.degradation.components["test_component"]
        assert not component.is_healthy
        assert component.failure_count == 1
        
        # Update back to healthy
        self.degradation.update_component_health("test_component", True)
        assert component.is_healthy
        assert component.failure_count == 0
        
    def test_feature_degradation(self):
        """Test feature degradation based on component health."""
        # Set up components and features
        self.degradation.register_component("database", ComponentCriticality.CRITICAL)
        self.degradation.register_component("cache", ComponentCriticality.IMPORTANT)
        
        self.degradation.register_feature("analytics", {"database"})
        self.degradation.register_feature("caching", {"cache"})
        
        # Mark database as unhealthy
        self.degradation.update_component_health("database", False)
        
        # Analytics feature should be disabled
        assert not self.degradation.is_feature_enabled("analytics")
        assert self.degradation.is_feature_enabled("caching")  # Should still work
        
    def test_degradation_status(self):
        """Test getting degradation status."""
        self.degradation.register_component("comp1", ComponentCriticality.CRITICAL)
        self.degradation.register_component("comp2", ComponentCriticality.IMPORTANT)
        self.degradation.register_feature("feature1", {"comp1"})
        
        status = self.degradation.get_degradation_status()
        assert status["total_components"] == 2
        assert status["total_features"] == 1
        assert status["current_level"] == DegradationLevel.FULL.value
        
    def test_impact_assessment(self):
        """Test impact assessment calculation."""
        self.degradation.register_component("comp1", ComponentCriticality.CRITICAL)
        self.degradation.register_feature("feature1", {"comp1"}, user_impact=0.5, resource_cost=2.0)
        
        # Mark component as failed
        self.degradation.update_component_health("comp1", False)
        
        impact = self.degradation.get_impact_assessment()
        assert impact["user_impact_percentage"] > 0
        assert impact["resource_savings_percentage"] > 0
        
    def test_degradation_callbacks(self):
        """Test degradation event callbacks."""
        callback_calls = []
        
        def degradation_callback(level, disabled_features):
            callback_calls.append((level, disabled_features))
            
        self.degradation.add_degradation_callback(degradation_callback)
        
        # Set up components and trigger degradation
        self.degradation.register_component("critical_comp", ComponentCriticality.CRITICAL)
        self.degradation.register_feature("important_feature", {"critical_comp"})
        
        self.degradation.update_component_health("critical_comp", False)
        
        # Should have triggered callback
        assert len(callback_calls) > 0
        
    def test_recovery_detection(self):
        """Test automatic recovery detection."""
        self.degradation.register_component("comp1", ComponentCriticality.CRITICAL)
        self.degradation.register_feature("feature1", {"comp1"})
        
        # Cause degradation
        self.degradation.update_component_health("comp1", False)
        assert not self.degradation.is_feature_enabled("feature1")
        
        # Recovery
        self.degradation.update_component_health("comp1", True)
        assert self.degradation.is_feature_enabled("feature1")
        assert self.degradation.current_level == DegradationLevel.FULL


class TestRecoveryStrategies:
    """Test cases for RecoveryStrategies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.recovery_manager = RecoveryStrategyManager()
        
    def test_strategy_registration(self):
        """Test adding and removing recovery strategies."""
        initial_count = len(self.recovery_manager.strategies)
        
        restart_strategy = RestartStrategy(max_restart_attempts=2)
        self.recovery_manager.add_strategy(restart_strategy)
        
        assert len(self.recovery_manager.strategies) == initial_count + 1
        
        self.recovery_manager.remove_strategy("restart")
        assert len(self.recovery_manager.strategies) == initial_count
        
    @pytest.mark.asyncio
    async def test_component_recovery(self):
        """Test component recovery process."""
        # Create recovery context
        context = RecoveryContext(
            component_id="test_component",
            component_type="agent",
            failure_type="heartbeat_timeout",
            failure_count=1,
            last_failure_time=time.time()
        )
        
        # Attempt recovery
        success = await self.recovery_manager.recover_component(context)
        
        # Should have attempted some recovery strategy
        assert isinstance(success, bool)
        
    def test_strategy_statistics(self):
        """Test getting strategy statistics."""
        stats = self.recovery_manager.get_strategy_statistics()
        
        # Should have default strategies
        assert len(stats) > 0
        
        for strategy_name, strategy_stats in stats.items():
            assert "success_count" in strategy_stats
            assert "failure_count" in strategy_stats
            assert "success_rate" in strategy_stats
            
    @pytest.mark.asyncio
    async def test_restart_strategy(self):
        """Test restart recovery strategy."""
        restart_strategy = RestartStrategy()
        
        context = RecoveryContext(
            component_id="test_agent",
            component_type="agent", 
            failure_type="heartbeat_timeout",
            failure_count=1,
            last_failure_time=time.time()
        )
        
        # Check if strategy can handle this failure
        can_recover = await restart_strategy.can_recover(context)
        assert can_recover == True
        
        # Execute recovery (this is simulated)
        success = await restart_strategy.execute_recovery(context)
        assert isinstance(success, bool)
        
    def test_recovery_history_tracking(self):
        """Test recovery history tracking."""
        initial_history_length = len(self.recovery_manager.get_recovery_history())
        
        # The history will be populated during recovery attempts
        # For now, just verify the method works
        history = self.recovery_manager.get_recovery_history(limit=10)
        assert isinstance(history, list)
        assert len(history) <= 10


class TestIntegratedFaultTolerance:
    """Integration tests for the complete fault tolerance system."""
    
    def setup_method(self):
        """Set up integrated test environment."""
        self.heartbeat_monitor = HeartbeatMonitor()
        self.circuit_breaker = CircuitBreaker("integration_test")
        self.retry_manager = RetryManager()
        self.degradation = GracefulDegradation()
        self.recovery_manager = RecoveryStrategyManager()
        
    def teardown_method(self):
        """Clean up test environment."""
        if self.heartbeat_monitor._running:
            self.heartbeat_monitor.stop_monitoring()
            
    def test_full_fault_tolerance_workflow(self):
        """Test complete fault tolerance workflow."""
        # 1. Register component for monitoring
        self.heartbeat_monitor.register_agent("test_agent")
        self.degradation.register_component("test_agent", ComponentCriticality.CRITICAL)
        
        # 2. Simulate normal operation
        self.heartbeat_monitor.record_heartbeat("test_agent", response_time=0.1)
        assert self.heartbeat_monitor.get_agent_state("test_agent") == AgentState.HEALTHY
        
        # 3. Simulate failure
        self.degradation.update_component_health("test_agent", False)
        
        # 4. Verify degradation was triggered
        status = self.degradation.get_degradation_status()
        assert len(status["unhealthy_components"]) == 1
        
    def test_circuit_breaker_with_retry(self):
        """Test circuit breaker integration with retry logic."""
        failure_count = 0
        
        @self.circuit_breaker
        @self.retry_manager
        def flaky_function():
            nonlocal failure_count
            failure_count += 1
            if failure_count < 3:
                raise ConnectionError("Network issue")
            return "success"
            
        # Should succeed after retries
        result = flaky_function()
        assert result == "success"
        
        # Circuit should still be closed
        assert self.circuit_breaker.get_state() == CircuitState.CLOSED
        
    def test_heartbeat_monitoring_with_recovery(self):
        """Test heartbeat monitoring triggering recovery."""
        recovery_triggered = []
        
        async def mock_recovery(context):
            recovery_triggered.append(context.component_id)
            return True
            
        # Set up monitoring and recovery
        self.heartbeat_monitor.register_agent("recovery_test_agent")
        
        # Add failure callback to trigger recovery
        def failure_callback(agent_id, failure_info):
            # In real implementation, this would trigger recovery
            recovery_triggered.append(agent_id)
            
        self.heartbeat_monitor.add_failure_callback(failure_callback)
        
        # Start monitoring
        self.heartbeat_monitor.start_monitoring()
        
        # Wait for heartbeat timeout
        time.sleep(0.2)
        
        # Should have triggered failure detection
        agent_state = self.heartbeat_monitor.get_agent_state("recovery_test_agent")
        # State might be degraded, critical, or failed depending on timing
        assert agent_state in [AgentState.DEGRADED, AgentState.CRITICAL, AgentState.FAILED]
        
    def test_performance_under_load(self):
        """Test system performance with multiple failing components."""
        # Register multiple agents
        for i in range(10):
            agent_id = f"load_test_agent_{i}"
            self.heartbeat_monitor.register_agent(agent_id)
            self.degradation.register_component(agent_id, ComponentCriticality.IMPORTANT)
            
        # Simulate mixed health states
        healthy_agents = 7
        for i in range(10):
            agent_id = f"load_test_agent_{i}"
            if i < healthy_agents:
                self.heartbeat_monitor.record_heartbeat(agent_id, response_time=0.1)
                self.degradation.update_component_health(agent_id, True)
            else:
                self.degradation.update_component_health(agent_id, False)
                
        # Check system can handle the load
        summary = self.heartbeat_monitor.get_monitoring_summary()
        assert summary["total_agents"] == 10
        assert summary["healthy"] == healthy_agents
        
        degradation_status = self.degradation.get_degradation_status()
        assert degradation_status["unhealthy_components"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])