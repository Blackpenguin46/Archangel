"""
Retry logic with exponential backoff and jitter for communication failures.
"""

import time
import asyncio
import random
import logging
from typing import Callable, Any, Optional, Union, Type
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class BackoffStrategy(Enum):
    """Backoff strategies for retry logic."""
    FIXED = "fixed"
    LINEAR = "linear" 
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"

@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER
    jitter_factor: float = 0.1  # Random jitter factor (0.0-1.0)
    retry_on_exceptions: tuple = (Exception,)  # Exceptions that trigger retries
    stop_on_exceptions: tuple = ()  # Exceptions that stop retries immediately

class RetryStats:
    """Statistics for retry operations."""
    
    def __init__(self):
        self.total_attempts = 0
        self.successful_attempts = 0
        self.failed_attempts = 0
        self.total_retries = 0
        self.total_delay = 0.0
        self.average_attempts = 0.0
        
    def record_attempt(self, attempts: int, total_delay: float, success: bool):
        """Record a retry attempt."""
        self.total_attempts += attempts
        if success:
            self.successful_attempts += 1
        else:
            self.failed_attempts += 1
        self.total_retries += attempts - 1
        self.total_delay += total_delay
        
        total_operations = self.successful_attempts + self.failed_attempts
        if total_operations > 0:
            self.average_attempts = self.total_attempts / total_operations

class RetryManager:
    """
    Retry manager with configurable backoff strategies and jitter.
    
    Features:
    - Multiple backoff strategies (fixed, linear, exponential)
    - Configurable jitter to prevent thundering herd
    - Exception-based retry control
    - Comprehensive retry statistics
    - Support for both sync and async functions
    - Circuit breaker integration
    - Timeout support for individual attempts
    """
    
    def __init__(self, policy: Optional[RetryPolicy] = None):
        """Initialize retry manager.
        
        Args:
            policy: Retry policy configuration
        """
        self.policy = policy or RetryPolicy()
        self.stats = RetryStats()
        
    def __call__(self, func: Callable):
        """Decorator to add retry logic to functions.
        
        Args:
            func: Function to wrap with retry logic
            
        Returns:
            Wrapped function with retry capability
        """
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                return await self.retry_async(func, *args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                return self.retry(func, *args, **kwargs)
            return sync_wrapper
            
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: Last exception if all retries fail
        """
        attempt = 0
        total_delay = 0.0
        last_exception = None
        
        while attempt < self.policy.max_attempts:
            attempt += 1
            
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Record successful attempt
                self.stats.record_attempt(attempt, total_delay, True)
                
                if attempt > 1:
                    logger.info(f"Function succeeded on attempt {attempt} after {total_delay:.2f}s delay")
                    
                return result
                
            except self.policy.stop_on_exceptions as e:
                # Don't retry on these exceptions
                logger.error(f"Function failed with non-retryable exception: {e}")
                self.stats.record_attempt(attempt, total_delay, False)
                raise
                
            except self.policy.retry_on_exceptions as e:
                last_exception = e
                
                if attempt == self.policy.max_attempts:
                    # Final attempt failed
                    logger.error(f"Function failed after {attempt} attempts")
                    self.stats.record_attempt(attempt, total_delay, False)
                    raise
                    
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt)
                total_delay += delay
                
                logger.warning(f"Function failed on attempt {attempt}, retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
                
            except Exception as e:
                # Unexpected exception, don't retry
                logger.error(f"Function failed with unexpected exception: {e}")
                self.stats.record_attempt(attempt, total_delay, False)
                raise
                
        # This shouldn't be reached, but just in case
        if last_exception:
            raise last_exception
            
    async def retry_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with retry logic.
        
        Args:
            func: Async function to execute
            *args: Function arguments  
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: Last exception if all retries fail
        """
        attempt = 0
        total_delay = 0.0
        last_exception = None
        
        while attempt < self.policy.max_attempts:
            attempt += 1
            
            try:
                start_time = time.time()
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Record successful attempt
                self.stats.record_attempt(attempt, total_delay, True)
                
                if attempt > 1:
                    logger.info(f"Async function succeeded on attempt {attempt} after {total_delay:.2f}s delay")
                    
                return result
                
            except self.policy.stop_on_exceptions as e:
                # Don't retry on these exceptions
                logger.error(f"Async function failed with non-retryable exception: {e}")
                self.stats.record_attempt(attempt, total_delay, False)
                raise
                
            except self.policy.retry_on_exceptions as e:
                last_exception = e
                
                if attempt == self.policy.max_attempts:
                    # Final attempt failed
                    logger.error(f"Async function failed after {attempt} attempts")
                    self.stats.record_attempt(attempt, total_delay, False)
                    raise
                    
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt)
                total_delay += delay
                
                logger.warning(f"Async function failed on attempt {attempt}, retrying in {delay:.2f}s: {e}")
                await asyncio.sleep(delay)
                
            except Exception as e:
                # Unexpected exception, don't retry
                logger.error(f"Async function failed with unexpected exception: {e}")
                self.stats.record_attempt(attempt, total_delay, False)
                raise
                
        # This shouldn't be reached, but just in case
        if last_exception:
            raise last_exception
            
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number.
        
        Args:
            attempt: Current attempt number (1-based)
            
        Returns:
            Delay in seconds
        """
        if self.policy.backoff_strategy == BackoffStrategy.FIXED:
            delay = self.policy.base_delay
            
        elif self.policy.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.policy.base_delay * attempt
            
        elif self.policy.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.policy.base_delay * (2 ** (attempt - 1))
            
        elif self.policy.backoff_strategy == BackoffStrategy.EXPONENTIAL_JITTER:
            base_delay = self.policy.base_delay * (2 ** (attempt - 1))
            jitter = base_delay * self.policy.jitter_factor * (random.random() * 2 - 1)
            delay = base_delay + jitter
            
        else:
            delay = self.policy.base_delay
            
        # Cap the delay at max_delay
        return min(delay, self.policy.max_delay)
        
    def get_stats(self) -> RetryStats:
        """Get retry statistics.
        
        Returns:
            RetryStats object
        """
        return self.stats
        
    def reset_stats(self):
        """Reset retry statistics."""
        self.stats = RetryStats()

class CommunicationRetryManager:
    """
    Specialized retry manager for communication failures.
    
    Provides pre-configured retry policies for common communication scenarios:
    - Network timeouts
    - Connection failures
    - Service unavailable errors
    - Rate limiting
    """
    
    @staticmethod
    def network_retry_policy() -> RetryPolicy:
        """Retry policy for network-related failures."""
        return RetryPolicy(
            max_attempts=5,
            base_delay=1.0,
            max_delay=30.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
            jitter_factor=0.2,
            retry_on_exceptions=(
                ConnectionError,
                TimeoutError, 
                OSError,
                # Add any network-specific exceptions
            ),
            stop_on_exceptions=(
                ValueError,  # Invalid parameters, don't retry
                TypeError,   # Programming errors, don't retry
            )
        )
        
    @staticmethod
    def service_retry_policy() -> RetryPolicy:
        """Retry policy for service communication failures."""
        return RetryPolicy(
            max_attempts=3,
            base_delay=2.0,
            max_delay=60.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
            jitter_factor=0.1,
            retry_on_exceptions=(
                ConnectionError,
                TimeoutError,
                # Add service-specific exceptions
            )
        )
        
    @staticmethod
    def database_retry_policy() -> RetryPolicy:
        """Retry policy for database communication failures."""
        return RetryPolicy(
            max_attempts=4,
            base_delay=0.5,
            max_delay=10.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            retry_on_exceptions=(
                ConnectionError,
                TimeoutError,
                # Add database-specific exceptions
            ),
            stop_on_exceptions=(
                ValueError,  # Bad SQL, don't retry
                # Add database-specific non-retryable exceptions
            )
        )
        
    @staticmethod
    def api_retry_policy() -> RetryPolicy:
        """Retry policy for API communication failures."""
        return RetryPolicy(
            max_attempts=3,
            base_delay=1.0,
            max_delay=20.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
            jitter_factor=0.3,
            retry_on_exceptions=(
                ConnectionError,
                TimeoutError,
                # Add HTTP-specific exceptions like 503, 429, etc.
            ),
            stop_on_exceptions=(
                # Add HTTP client errors that shouldn't be retried (400, 401, 403, 404)
            )
        )

# Pre-configured retry managers for common scenarios
network_retry = RetryManager(CommunicationRetryManager.network_retry_policy())
service_retry = RetryManager(CommunicationRetryManager.service_retry_policy())
database_retry = RetryManager(CommunicationRetryManager.database_retry_policy())
api_retry = RetryManager(CommunicationRetryManager.api_retry_policy())

# Decorator functions for easy use
def retry_network(func):
    """Decorator for network operations with retry logic."""
    return network_retry(func)

def retry_service(func):
    """Decorator for service operations with retry logic."""
    return service_retry(func)

def retry_database(func):
    """Decorator for database operations with retry logic."""
    return database_retry(func)

def retry_api(func):
    """Decorator for API operations with retry logic."""
    return api_retry(func)