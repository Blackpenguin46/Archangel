"""
Circuit breaker pattern implementation for fault tolerance.
"""

import time
import threading
import logging
import asyncio
from typing import Dict, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests flow through
    OPEN = "open"          # Failure state, requests are rejected
    HALF_OPEN = "half_open"  # Testing state, limited requests allowed

@dataclass
class CircuitConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Number of failures before opening
    success_threshold: int = 2  # Number of successes to close from half-open
    timeout: float = 60.0  # Timeout before transitioning to half-open (seconds)
    recovery_timeout: float = 10.0  # Timeout in half-open state
    expected_exception_types: tuple = (Exception,)  # Exceptions that count as failures

@dataclass
class CircuitStats:
    """Statistics for circuit breaker."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    state_transitions: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0

class CircuitBreaker:
    """
    Circuit breaker implementation with automatic failure detection and recovery.
    
    The circuit breaker prevents cascading failures by:
    - Monitoring request success/failure rates
    - Opening the circuit when failure threshold is reached
    - Rejecting requests quickly when circuit is open
    - Testing recovery with limited requests in half-open state
    - Automatically closing when service recovers
    """
    
    def __init__(self, name: str, config: Optional[CircuitConfig] = None):
        """Initialize circuit breaker.
        
        Args:
            name: Unique name for this circuit breaker
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._last_attempt_time = 0.0
        self._lock = threading.RLock()
        
        # Callbacks for state changes
        self._state_change_callbacks: list[Callable[[str, CircuitState, CircuitState], None]] = []
        
    def __call__(self, func: Callable):
        """Decorator to wrap functions with circuit breaker.
        
        Args:
            func: Function to wrap
            
        Returns:
            Wrapped function with circuit breaker protection
        """
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                return await self.call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                return self.call(func, *args, **kwargs)
            return sync_wrapper
            
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original function exceptions
        """
        if not self._can_execute():
            self._record_rejection()
            raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is open")
            
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            self._record_success(time.time() - start_time)
            return result
            
        except self.config.expected_exception_types as e:
            self._record_failure(time.time() - start_time)
            raise
            
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original function exceptions
        """
        if not self._can_execute():
            self._record_rejection()
            raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is open")
            
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            self._record_success(time.time() - start_time)
            return result
            
        except self.config.expected_exception_types as e:
            self._record_failure(time.time() - start_time)
            raise
            
    def _can_execute(self) -> bool:
        """Check if request can be executed based on circuit state.
        
        Returns:
            True if request can be executed
        """
        with self._lock:
            current_time = time.time()
            self._last_attempt_time = current_time
            
            if self.state == CircuitState.CLOSED:
                return True
                
            elif self.state == CircuitState.OPEN:
                # Check if timeout has elapsed to move to half-open
                if current_time - self._last_failure_time >= self.config.timeout:
                    self._transition_to_half_open()
                    return True
                return False
                
            elif self.state == CircuitState.HALF_OPEN:
                # Check if recovery timeout has elapsed
                if current_time - self._last_failure_time >= self.config.recovery_timeout:
                    self._transition_to_open()
                    return False
                return True
                
        return False
        
    def _record_success(self, execution_time: float):
        """Record successful execution.
        
        Args:
            execution_time: Time taken for execution
        """
        with self._lock:
            current_time = time.time()
            
            self.stats.total_requests += 1
            self.stats.successful_requests += 1
            self.stats.last_success_time = current_time
            
            if self.state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to_closed()
                    
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
                
    def _record_failure(self, execution_time: float):
        """Record failed execution.
        
        Args:
            execution_time: Time taken for execution
        """
        with self._lock:
            current_time = time.time()
            
            self.stats.total_requests += 1
            self.stats.failed_requests += 1
            self.stats.last_failure_time = current_time
            self._last_failure_time = current_time
            
            if self.state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to_open()
                    
            elif self.state == CircuitState.HALF_OPEN:
                # Go back to open on any failure in half-open state
                self._transition_to_open()
                
    def _record_rejection(self):
        """Record rejected request."""
        with self._lock:
            self.stats.rejected_requests += 1
            
    def _transition_to_open(self):
        """Transition circuit breaker to open state."""
        old_state = self.state
        self.state = CircuitState.OPEN
        self._failure_count = 0
        self._success_count = 0
        self.stats.state_transitions += 1
        
        logger.warning(f"Circuit breaker '{self.name}' opened")
        self._notify_state_change(old_state, self.state)
        
    def _transition_to_half_open(self):
        """Transition circuit breaker to half-open state."""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self._success_count = 0
        self.stats.state_transitions += 1
        
        logger.info(f"Circuit breaker '{self.name}' half-opened")
        self._notify_state_change(old_state, self.state)
        
    def _transition_to_closed(self):
        """Transition circuit breaker to closed state."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self.stats.state_transitions += 1
        
        logger.info(f"Circuit breaker '{self.name}' closed")
        self._notify_state_change(old_state, self.state)
        
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state.
        
        Returns:
            Current CircuitState
        """
        return self.state
        
    def get_stats(self) -> CircuitStats:
        """Get circuit breaker statistics.
        
        Returns:
            CircuitStats object
        """
        with self._lock:
            return CircuitStats(
                total_requests=self.stats.total_requests,
                successful_requests=self.stats.successful_requests,
                failed_requests=self.stats.failed_requests,
                rejected_requests=self.stats.rejected_requests,
                state_transitions=self.stats.state_transitions,
                last_failure_time=self.stats.last_failure_time,
                last_success_time=self.stats.last_success_time
            )
            
    def get_success_rate(self) -> float:
        """Get success rate for executed requests.
        
        Returns:
            Success rate as a float between 0 and 1
        """
        with self._lock:
            executed_requests = self.stats.total_requests
            if executed_requests == 0:
                return 0.0
            return self.stats.successful_requests / executed_requests
            
    def force_open(self):
        """Force circuit breaker to open state."""
        with self._lock:
            if self.state != CircuitState.OPEN:
                self._transition_to_open()
                logger.warning(f"Circuit breaker '{self.name}' force opened")
                
    def force_closed(self):
        """Force circuit breaker to closed state."""
        with self._lock:
            if self.state != CircuitState.CLOSED:
                self._transition_to_closed()
                logger.info(f"Circuit breaker '{self.name}' force closed")
                
    def reset(self):
        """Reset circuit breaker to initial state."""
        with self._lock:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self.stats = CircuitStats()
            
            logger.info(f"Circuit breaker '{self.name}' reset")
            if old_state != self.state:
                self._notify_state_change(old_state, self.state)
                
    def add_state_change_callback(self, callback: Callable[[str, CircuitState, CircuitState], None]):
        """Add callback for state changes.
        
        Args:
            callback: Function to call when state changes
        """
        self._state_change_callbacks.append(callback)
        
    def _notify_state_change(self, old_state: CircuitState, new_state: CircuitState):
        """Notify callbacks of state change.
        
        Args:
            old_state: Previous state
            new_state: New state
        """
        for callback in self._state_change_callbacks:
            try:
                callback(self.name, old_state, new_state)
            except Exception as e:
                logger.error(f"Error in circuit breaker state change callback: {e}")

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass

class CircuitBreakerManager:
    """Manager for multiple circuit breakers."""
    
    def __init__(self):
        """Initialize circuit breaker manager."""
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
        
    def get_circuit_breaker(self, name: str, config: Optional[CircuitConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker.
        
        Args:
            name: Name of the circuit breaker
            config: Optional configuration for new circuit breakers
            
        Returns:
            CircuitBreaker instance
        """
        with self._lock:
            if name not in self.circuit_breakers:
                self.circuit_breakers[name] = CircuitBreaker(name, config)
                logger.info(f"Created circuit breaker: {name}")
                
            return self.circuit_breakers[name]
            
    def remove_circuit_breaker(self, name: str):
        """Remove a circuit breaker.
        
        Args:
            name: Name of the circuit breaker to remove
        """
        with self._lock:
            if name in self.circuit_breakers:
                del self.circuit_breakers[name]
                logger.info(f"Removed circuit breaker: {name}")
                
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers.
        
        Returns:
            Dictionary mapping circuit breaker names to statistics
        """
        with self._lock:
            stats = {}
            for name, cb in self.circuit_breakers.items():
                cb_stats = cb.get_stats()
                stats[name] = {
                    "state": cb.get_state().value,
                    "total_requests": cb_stats.total_requests,
                    "successful_requests": cb_stats.successful_requests,
                    "failed_requests": cb_stats.failed_requests,
                    "rejected_requests": cb_stats.rejected_requests,
                    "success_rate": cb.get_success_rate(),
                    "state_transitions": cb_stats.state_transitions,
                    "last_failure_time": cb_stats.last_failure_time,
                    "last_success_time": cb_stats.last_success_time
                }
            return stats
            
    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for cb in self.circuit_breakers.values():
                cb.reset()
            logger.info("Reset all circuit breakers")

# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()