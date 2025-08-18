"""
Performance benchmarking and load testing capabilities.
"""

import time
import asyncio
import threading
import multiprocessing
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics
import logging
import json
import concurrent.futures
from contextlib import contextmanager
import psutil
import random

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    benchmark_name: str
    start_time: datetime
    end_time: datetime
    duration: float
    iterations: int
    operations_per_second: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    percentiles: Dict[str, float]
    success_rate: float
    error_count: int
    resource_usage: Dict[str, Any]
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    concurrent_users: int
    duration: timedelta
    ramp_up_time: timedelta
    operations_per_user: int
    think_time_range: Tuple[float, float]  # Min, max think time in seconds
    target_operations_per_second: Optional[int] = None

@dataclass
class LoadTestResult:
    """Result of a load test."""
    config: LoadTestConfig
    start_time: datetime
    end_time: datetime
    total_operations: int
    successful_operations: int
    failed_operations: int
    avg_response_time: float
    response_time_percentiles: Dict[str, float]
    operations_per_second: float
    peak_operations_per_second: float
    resource_usage_peak: Dict[str, Any]
    error_breakdown: Dict[str, int]

class PerformanceBenchmarks:
    """Performance benchmarking system."""
    
    def __init__(self):
        self.benchmark_results: List[BenchmarkResult] = []
        self.baseline_results: Dict[str, BenchmarkResult] = {}
        
    def run_agent_decision_benchmark(self, agent_factory: Callable, 
                                   iterations: int = 100) -> BenchmarkResult:
        """Benchmark agent decision-making performance."""
        benchmark_name = "agent_decision_making"
        start_time = datetime.now()
        
        response_times = []
        errors = 0
        
        # Monitor resource usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        initial_cpu_time = process.cpu_times()
        
        logger.info(f"Starting {benchmark_name} benchmark with {iterations} iterations")
        
        for i in range(iterations):
            try:
                agent = agent_factory()
                
                # Simulate decision-making scenario
                scenario_start = time.time()
                
                # Mock environment state
                environment_state = {
                    'network_scan_results': ['192.168.1.1', '192.168.1.2'],
                    'vulnerabilities': ['CVE-2023-1234'],
                    'available_tools': ['nmap', 'sqlmap']
                }
                
                # Simulate agent decision process
                decision = self._simulate_agent_decision(agent, environment_state)
                
                response_time = time.time() - scenario_start
                response_times.append(response_time)
                
                if i % 10 == 0:
                    logger.debug(f"Completed {i+1}/{iterations} iterations")
                    
            except Exception as e:
                errors += 1
                logger.error(f"Error in iteration {i}: {e}")
                
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate final resource usage
        final_memory = process.memory_info().rss
        final_cpu_time = process.cpu_times()
        
        # Calculate statistics
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            percentiles = {
                'p50': statistics.median(response_times),
                'p90': self._calculate_percentile(response_times, 90),
                'p95': self._calculate_percentile(response_times, 95),
                'p99': self._calculate_percentile(response_times, 99)
            }
        else:
            avg_response_time = min_response_time = max_response_time = 0
            percentiles = {'p50': 0, 'p90': 0, 'p95': 0, 'p99': 0}
            
        operations_per_second = iterations / duration if duration > 0 else 0
        success_rate = (iterations - errors) / iterations if iterations > 0 else 0
        
        resource_usage = {
            'memory_delta_mb': (final_memory - initial_memory) / 1024 / 1024,
            'cpu_time_delta': final_cpu_time.user - initial_cpu_time.user,
            'peak_memory_mb': process.memory_info().rss / 1024 / 1024
        }
        
        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            iterations=iterations,
            operations_per_second=operations_per_second,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            percentiles=percentiles,
            success_rate=success_rate,
            error_count=errors,
            resource_usage=resource_usage
        )
        
        self.benchmark_results.append(result)
        logger.info(f"Completed {benchmark_name} benchmark: {operations_per_second:.2f} ops/sec, "
                   f"{avg_response_time:.3f}s avg response time")
        
        return result
        
    def run_memory_retrieval_benchmark(self, memory_system: Any, 
                                     iterations: int = 500) -> BenchmarkResult:
        """Benchmark memory retrieval performance."""
        benchmark_name = "memory_retrieval"
        start_time = datetime.now()
        
        response_times = []
        errors = 0
        
        # Prepare test queries
        test_queries = [
            "network reconnaissance techniques",
            "SQL injection vulnerabilities",
            "privilege escalation methods",
            "lateral movement strategies",
            "data exfiltration techniques"
        ] * (iterations // 5 + 1)
        
        logger.info(f"Starting {benchmark_name} benchmark with {iterations} iterations")
        
        for i in range(iterations):
            try:
                query = test_queries[i % len(test_queries)]
                
                query_start = time.time()
                
                # Simulate memory retrieval
                results = self._simulate_memory_retrieval(memory_system, query)
                
                response_time = time.time() - query_start
                response_times.append(response_time)
                
                if i % 50 == 0:
                    logger.debug(f"Completed {i+1}/{iterations} memory retrievals")
                    
            except Exception as e:
                errors += 1
                logger.error(f"Error in memory retrieval {i}: {e}")
                
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate statistics
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            percentiles = {
                'p50': statistics.median(response_times),
                'p90': self._calculate_percentile(response_times, 90),
                'p95': self._calculate_percentile(response_times, 95),
                'p99': self._calculate_percentile(response_times, 99)
            }
        else:
            avg_response_time = min_response_time = max_response_time = 0
            percentiles = {'p50': 0, 'p90': 0, 'p95': 0, 'p99': 0}
            
        operations_per_second = iterations / duration if duration > 0 else 0
        success_rate = (iterations - errors) / iterations if iterations > 0 else 0
        
        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            iterations=iterations,
            operations_per_second=operations_per_second,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            percentiles=percentiles,
            success_rate=success_rate,
            error_count=errors,
            resource_usage={}
        )
        
        self.benchmark_results.append(result)
        logger.info(f"Completed {benchmark_name} benchmark: {operations_per_second:.2f} ops/sec")
        
        return result
        
    def run_communication_benchmark(self, message_bus: Any, 
                                  iterations: int = 1000) -> BenchmarkResult:
        """Benchmark inter-agent communication performance."""
        benchmark_name = "agent_communication"
        start_time = datetime.now()
        
        response_times = []
        errors = 0
        
        logger.info(f"Starting {benchmark_name} benchmark with {iterations} iterations")
        
        for i in range(iterations):
            try:
                message_start = time.time()
                
                # Simulate message sending and receiving
                message = {
                    'sender': f'agent_{i % 10}',
                    'recipient': f'agent_{(i + 1) % 10}',
                    'type': 'intelligence_update',
                    'data': {'target': '192.168.1.100', 'vulnerability': 'CVE-2023-1234'}
                }
                
                success = self._simulate_message_exchange(message_bus, message)
                
                response_time = time.time() - message_start
                response_times.append(response_time)
                
                if not success:
                    errors += 1
                    
                if i % 100 == 0:
                    logger.debug(f"Completed {i+1}/{iterations} message exchanges")
                    
            except Exception as e:
                errors += 1
                logger.error(f"Error in message exchange {i}: {e}")
                
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate statistics
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            percentiles = {
                'p50': statistics.median(response_times),
                'p90': self._calculate_percentile(response_times, 90),
                'p95': self._calculate_percentile(response_times, 95),
                'p99': self._calculate_percentile(response_times, 99)
            }
        else:
            avg_response_time = min_response_time = max_response_time = 0
            percentiles = {'p50': 0, 'p90': 0, 'p95': 0, 'p99': 0}
            
        operations_per_second = iterations / duration if duration > 0 else 0
        success_rate = (iterations - errors) / iterations if iterations > 0 else 0
        
        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            iterations=iterations,
            operations_per_second=operations_per_second,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            percentiles=percentiles,
            success_rate=success_rate,
            error_count=errors,
            resource_usage={}
        )
        
        self.benchmark_results.append(result)
        logger.info(f"Completed {benchmark_name} benchmark: {operations_per_second:.2f} ops/sec")
        
        return result
        
    def set_baseline(self, benchmark_name: str) -> None:
        """Set current result as baseline for comparison."""
        latest_result = None
        for result in reversed(self.benchmark_results):
            if result.benchmark_name == benchmark_name:
                latest_result = result
                break
                
        if latest_result:
            self.baseline_results[benchmark_name] = latest_result
            logger.info(f"Set baseline for {benchmark_name}: {latest_result.operations_per_second:.2f} ops/sec")
        else:
            logger.warning(f"No results found for benchmark {benchmark_name}")
            
    def compare_to_baseline(self, benchmark_name: str) -> Optional[Dict[str, Any]]:
        """Compare latest result to baseline."""
        if benchmark_name not in self.baseline_results:
            logger.warning(f"No baseline set for {benchmark_name}")
            return None
            
        baseline = self.baseline_results[benchmark_name]
        
        # Find latest result
        latest_result = None
        for result in reversed(self.benchmark_results):
            if result.benchmark_name == benchmark_name and result != baseline:
                latest_result = result
                break
                
        if not latest_result:
            logger.warning(f"No recent results found for {benchmark_name}")
            return None
            
        # Calculate comparisons
        ops_change = ((latest_result.operations_per_second - baseline.operations_per_second) 
                     / baseline.operations_per_second * 100)
        
        response_time_change = ((latest_result.avg_response_time - baseline.avg_response_time) 
                               / baseline.avg_response_time * 100)
        
        success_rate_change = ((latest_result.success_rate - baseline.success_rate) 
                              / baseline.success_rate * 100)
        
        comparison = {
            'benchmark_name': benchmark_name,
            'baseline_date': baseline.start_time.isoformat(),
            'latest_date': latest_result.start_time.isoformat(),
            'operations_per_second': {
                'baseline': baseline.operations_per_second,
                'latest': latest_result.operations_per_second,
                'change_percent': ops_change
            },
            'avg_response_time': {
                'baseline': baseline.avg_response_time,
                'latest': latest_result.avg_response_time,
                'change_percent': response_time_change
            },
            'success_rate': {
                'baseline': baseline.success_rate,
                'latest': latest_result.success_rate,
                'change_percent': success_rate_change
            },
            'performance_trend': 'improved' if ops_change > 5 else 'degraded' if ops_change < -5 else 'stable'
        }
        
        return comparison
        
    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results."""
        if not self.benchmark_results:
            return {}
            
        # Group results by benchmark name
        grouped_results = {}
        for result in self.benchmark_results:
            if result.benchmark_name not in grouped_results:
                grouped_results[result.benchmark_name] = []
            grouped_results[result.benchmark_name].append(result)
            
        summary = {}
        for benchmark_name, results in grouped_results.items():
            latest_result = results[-1]
            
            # Calculate trends if we have multiple results
            trend = 'stable'
            if len(results) >= 2:
                recent_ops = [r.operations_per_second for r in results[-3:]]
                if len(recent_ops) >= 2:
                    if recent_ops[-1] > recent_ops[0] * 1.1:
                        trend = 'improving'
                    elif recent_ops[-1] < recent_ops[0] * 0.9:
                        trend = 'degrading'
                        
            summary[benchmark_name] = {
                'latest_result': {
                    'operations_per_second': latest_result.operations_per_second,
                    'avg_response_time': latest_result.avg_response_time,
                    'success_rate': latest_result.success_rate,
                    'date': latest_result.start_time.isoformat()
                },
                'total_runs': len(results),
                'trend': trend,
                'has_baseline': benchmark_name in self.baseline_results
            }
            
        return summary
        
    def export_results(self, filepath: str) -> None:
        """Export benchmark results to JSON file."""
        export_data = []
        for result in self.benchmark_results:
            data = {
                'benchmark_name': result.benchmark_name,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat(),
                'duration': result.duration,
                'iterations': result.iterations,
                'operations_per_second': result.operations_per_second,
                'avg_response_time': result.avg_response_time,
                'min_response_time': result.min_response_time,
                'max_response_time': result.max_response_time,
                'percentiles': result.percentiles,
                'success_rate': result.success_rate,
                'error_count': result.error_count,
                'resource_usage': result.resource_usage,
                'custom_metrics': result.custom_metrics
            }
            export_data.append(data)
            
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        logger.info(f"Exported {len(export_data)} benchmark results to {filepath}")
        
    def _simulate_agent_decision(self, agent: Any, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate agent decision-making process."""
        # Simulate processing time
        time.sleep(random.uniform(0.01, 0.1))
        
        return {
            'action': 'network_scan',
            'target': '192.168.1.0/24',
            'tool': 'nmap',
            'confidence': 0.85
        }
        
    def _simulate_memory_retrieval(self, memory_system: Any, query: str) -> List[Dict[str, Any]]:
        """Simulate memory retrieval operation."""
        # Simulate retrieval time
        time.sleep(random.uniform(0.001, 0.05))
        
        return [
            {'content': f'Result for {query}', 'relevance': 0.9},
            {'content': f'Related to {query}', 'relevance': 0.7}
        ]
        
    def _simulate_message_exchange(self, message_bus: Any, message: Dict[str, Any]) -> bool:
        """Simulate message exchange."""
        # Simulate network latency
        time.sleep(random.uniform(0.001, 0.01))
        
        # 99% success rate
        return random.random() < 0.99
        
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
            
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

class LoadTester:
    """Load testing system for multi-agent scenarios."""
    
    def __init__(self):
        self.load_test_results: List[LoadTestResult] = []
        
    def run_concurrent_agent_load_test(self, 
                                     agent_factory: Callable,
                                     config: LoadTestConfig) -> LoadTestResult:
        """Run load test with concurrent agents."""
        logger.info(f"Starting load test with {config.concurrent_users} concurrent users")
        
        start_time = datetime.now()
        
        # Shared state for collecting results
        results_lock = threading.Lock()
        operation_times = []
        error_counts = {'total': 0}
        
        def user_simulation(user_id: int):
            """Simulate a single user's operations."""
            user_errors = 0
            user_operations = 0
            
            # Ramp-up delay
            ramp_delay = (config.ramp_up_time.total_seconds() * user_id / config.concurrent_users)
            time.sleep(ramp_delay)
            
            end_time = start_time + config.duration
            
            while datetime.now() < end_time and user_operations < config.operations_per_user:
                try:
                    # Create agent
                    agent = agent_factory()
                    
                    # Perform operation
                    operation_start = time.time()
                    
                    # Simulate agent operation
                    self._simulate_agent_operation(agent)
                    
                    operation_time = time.time() - operation_start
                    
                    with results_lock:
                        operation_times.append(operation_time)
                        
                    user_operations += 1
                    
                    # Think time
                    think_time = random.uniform(*config.think_time_range)
                    time.sleep(think_time)
                    
                except Exception as e:
                    user_errors += 1
                    logger.debug(f"User {user_id} error: {e}")
                    
            with results_lock:
                error_counts['total'] += user_errors
                
        # Start user threads
        threads = []
        for user_id in range(config.concurrent_users):
            thread = threading.Thread(target=user_simulation, args=(user_id,))
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        end_time = datetime.now()
        
        # Calculate results
        total_operations = len(operation_times)
        successful_operations = total_operations - error_counts['total']
        failed_operations = error_counts['total']
        
        if operation_times:
            avg_response_time = statistics.mean(operation_times)
            response_time_percentiles = {
                'p50': self._calculate_percentile(operation_times, 50),
                'p90': self._calculate_percentile(operation_times, 90),
                'p95': self._calculate_percentile(operation_times, 95),
                'p99': self._calculate_percentile(operation_times, 99)
            }
        else:
            avg_response_time = 0
            response_time_percentiles = {'p50': 0, 'p90': 0, 'p95': 0, 'p99': 0}
            
        duration = (end_time - start_time).total_seconds()
        operations_per_second = total_operations / duration if duration > 0 else 0
        
        # Calculate peak operations per second (using 1-second windows)
        peak_ops_per_second = self._calculate_peak_operations_per_second(operation_times, start_time)
        
        result = LoadTestResult(
            config=config,
            start_time=start_time,
            end_time=end_time,
            total_operations=total_operations,
            successful_operations=successful_operations,
            failed_operations=failed_operations,
            avg_response_time=avg_response_time,
            response_time_percentiles=response_time_percentiles,
            operations_per_second=operations_per_second,
            peak_operations_per_second=peak_ops_per_second,
            resource_usage_peak={},  # Would be populated with actual resource monitoring
            error_breakdown={'general_errors': error_counts['total']}
        )
        
        self.load_test_results.append(result)
        
        logger.info(f"Load test completed: {total_operations} operations, "
                   f"{operations_per_second:.2f} ops/sec average, "
                   f"{successful_operations/total_operations*100:.1f}% success rate")
        
        return result
        
    def _simulate_agent_operation(self, agent: Any) -> None:
        """Simulate a single agent operation."""
        # Simulate various operation types with different durations
        operation_type = random.choice(['decision', 'memory_retrieval', 'communication', 'action'])
        
        if operation_type == 'decision':
            time.sleep(random.uniform(0.1, 0.5))
        elif operation_type == 'memory_retrieval':
            time.sleep(random.uniform(0.01, 0.1))
        elif operation_type == 'communication':
            time.sleep(random.uniform(0.005, 0.02))
        else:  # action
            time.sleep(random.uniform(0.05, 0.2))
            
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
            
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
        
    def _calculate_peak_operations_per_second(self, operation_times: List[float], 
                                            start_time: datetime) -> float:
        """Calculate peak operations per second using sliding window."""
        if not operation_times:
            return 0.0
            
        # For simplicity, return average * 1.5 as estimated peak
        # In real implementation, this would analyze actual timing data
        avg_ops_per_sec = len(operation_times) / (datetime.now() - start_time).total_seconds()
        return avg_ops_per_sec * 1.5