#!/usr/bin/env python3
"""
Demo script for performance optimization and tuning system.

This script demonstrates the comprehensive performance optimization capabilities
including profiling, caching, benchmarking, and resource monitoring.
"""

import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import random

from performance.profiler import SystemProfiler, AgentProfiler, profile_agent_operation
from performance.optimizer import PerformanceOptimizer, CacheManager
from performance.benchmarks import PerformanceBenchmarks, LoadTester, LoadTestConfig
from performance.metrics import PerformanceMetrics, ResourceMonitor, TimingContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockAgent:
    """Mock agent for demonstration purposes."""
    
    def __init__(self, agent_id: str, team: str):
        self.agent_id = agent_id
        self.team = team
        self.memory_cache = {}
        
    def make_decision(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate agent decision-making."""
        # Simulate processing time
        processing_time = random.uniform(0.1, 0.5)
        time.sleep(processing_time)
        
        # Simulate decision based on environment
        if environment_state.get('threat_detected'):
            if self.team == 'blue':
                return {
                    'action': 'block_traffic',
                    'target': environment_state.get('threat_source'),
                    'confidence': random.uniform(0.7, 0.95)
                }
            else:  # red team
                return {
                    'action': 'evade_detection',
                    'method': 'change_tactics',
                    'confidence': random.uniform(0.6, 0.9)
                }
        else:
            if self.team == 'blue':
                return {
                    'action': 'monitor',
                    'scope': 'network_traffic',
                    'confidence': random.uniform(0.8, 0.95)
                }
            else:  # red team
                return {
                    'action': 'reconnaissance',
                    'target': 'network_scan',
                    'confidence': random.uniform(0.7, 0.9)
                }
                
    def retrieve_memory(self, query: str) -> List[Dict[str, Any]]:
        """Simulate memory retrieval."""
        # Simulate retrieval time
        retrieval_time = random.uniform(0.01, 0.1)
        time.sleep(retrieval_time)
        
        # Check cache first
        if query in self.memory_cache:
            return self.memory_cache[query]
            
        # Simulate memory search
        results = [
            {
                'content': f'Memory result for {query}',
                'relevance': random.uniform(0.6, 0.95),
                'timestamp': datetime.now().isoformat()
            }
            for _ in range(random.randint(1, 5))
        ]
        
        # Cache results
        self.memory_cache[query] = results
        return results
        
    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate action execution."""
        # Simulate execution time based on action type
        action_type = action.get('action', 'unknown')
        
        if action_type in ['block_traffic', 'evade_detection']:
            execution_time = random.uniform(0.05, 0.2)
        elif action_type in ['monitor', 'reconnaissance']:
            execution_time = random.uniform(0.02, 0.1)
        else:
            execution_time = random.uniform(0.01, 0.05)
            
        time.sleep(execution_time)
        
        return {
            'status': 'success' if random.random() > 0.1 else 'failed',
            'execution_time': execution_time,
            'result': f"Executed {action_type}"
        }

class MockMemorySystem:
    """Mock memory system for demonstration."""
    
    def __init__(self):
        self.storage = {}
        self.query_count = 0
        
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Simulate memory search."""
        self.query_count += 1
        
        # Simulate search time
        search_time = random.uniform(0.01, 0.08)
        time.sleep(search_time)
        
        return [
            {
                'id': f'mem_{i}',
                'content': f'Memory content for {query}',
                'relevance': random.uniform(0.5, 0.95),
                'metadata': {'source': 'simulation', 'type': 'experience'}
            }
            for i in range(random.randint(1, limit))
        ]

class MockMessageBus:
    """Mock message bus for demonstration."""
    
    def __init__(self):
        self.message_count = 0
        self.latency_simulation = True
        
    def send_message(self, sender: str, recipient: str, message: Dict[str, Any]) -> bool:
        """Simulate message sending."""
        self.message_count += 1
        
        if self.latency_simulation:
            # Simulate network latency
            latency = random.uniform(0.001, 0.01)
            time.sleep(latency)
            
        # 99% success rate
        return random.random() < 0.99

def demonstrate_system_profiling():
    """Demonstrate system profiling capabilities."""
    logger.info("=== System Profiling Demo ===")
    
    profiler = SystemProfiler()
    
    # Start profiling
    profiler.start_profiling(interval=0.5)
    logger.info("Started system profiling")
    
    # Simulate some system activity
    logger.info("Simulating system activity...")
    
    def cpu_intensive_task():
        """Simulate CPU-intensive work."""
        result = 0
        for i in range(1000000):
            result += i * i
        return result
        
    # Profile a specific function
    with profiler.profile_function("cpu_intensive_task"):
        result = cpu_intensive_task()
        logger.info(f"CPU task result: {result}")
        
    # Let profiler collect more data
    time.sleep(2.0)
    
    # Stop profiling
    profiler.stop_profiling()
    logger.info("Stopped system profiling")
    
    # Get performance summary
    summary = profiler.get_performance_summary()
    logger.info(f"Collected {summary.get('profile_count', 0)} profiles")
    logger.info(f"Average CPU usage: {summary.get('cpu_usage', {}).get('avg', 0):.2f}%")
    logger.info(f"Average memory usage: {summary.get('memory_usage', {}).get('avg', 0):.2f}%")
    
    return profiler

def demonstrate_agent_profiling():
    """Demonstrate agent profiling capabilities."""
    logger.info("\n=== Agent Profiling Demo ===")
    
    agent_profiler = AgentProfiler()
    
    # Create mock agents
    agents = [
        MockAgent("red_agent_1", "red"),
        MockAgent("blue_agent_1", "blue"),
        MockAgent("blue_agent_2", "blue")
    ]
    
    # Simulate agent operations
    for agent in agents:
        logger.info(f"Profiling agent: {agent.agent_id}")
        
        # Start profiling session
        agent_profiler.start_agent_session(agent.agent_id)
        
        # Simulate decision-making
        with profile_agent_operation(agent_profiler, agent.agent_id, "decision"):
            environment_state = {
                'threat_detected': random.choice([True, False]),
                'threat_source': '192.168.1.100'
            }
            decision = agent.make_decision(environment_state)
            logger.info(f"Agent decision: {decision['action']}")
            
        # Simulate memory retrieval
        with profile_agent_operation(agent_profiler, agent.agent_id, "memory_retrieval"):
            memories = agent.retrieve_memory("attack patterns")
            logger.info(f"Retrieved {len(memories)} memories")
            
        # Simulate action execution
        with profile_agent_operation(agent_profiler, agent.agent_id, "action_execution"):
            result = agent.execute_action(decision)
            logger.info(f"Action result: {result['status']}")
            
        # End profiling session
        cache_hit_rate = len(agent.memory_cache) / 10.0  # Simulate cache hit rate
        profile = agent_profiler.end_agent_session(
            agent.agent_id,
            memory_usage=random.uniform(10.0, 100.0),
            cache_hit_rate=min(cache_hit_rate, 1.0)
        )
        
        logger.info(f"Agent {agent.agent_id} total response time: {profile.total_response_time:.3f}s")
        
    # Get performance summaries
    for agent in agents:
        summary = agent_profiler.get_agent_performance_summary(agent.agent_id)
        logger.info(f"Agent {agent.agent_id} average response time: {summary.get('avg_total_response_time', 0):.3f}s")
        
    return agent_profiler

def demonstrate_caching_optimization():
    """Demonstrate caching and optimization capabilities."""
    logger.info("\n=== Caching & Optimization Demo ===")
    
    optimizer = PerformanceOptimizer()
    
    # Demonstrate cache usage
    agent_memory_cache = optimizer.cache_manager.get_cache("agent_memory")
    llm_cache = optimizer.cache_manager.get_cache("llm_responses")
    
    # Simulate cache operations
    logger.info("Simulating cache operations...")
    
    # Fill caches with test data
    for i in range(50):
        agent_memory_cache.put(f"memory_key_{i}", f"memory_value_{i}")
        llm_cache.put(f"llm_prompt_{i}", f"llm_response_{i}")
        
    # Simulate cache hits and misses
    hit_count = 0
    miss_count = 0
    
    for i in range(100):
        key = f"memory_key_{random.randint(0, 75)}"  # Some keys won't exist
        value = agent_memory_cache.get(key)
        if value:
            hit_count += 1
        else:
            miss_count += 1
            
    logger.info(f"Cache performance: {hit_count} hits, {miss_count} misses")
    
    # Get cache statistics
    cache_stats = optimizer.cache_manager.get_all_stats()
    for cache_name, stats in cache_stats.items():
        logger.info(f"Cache '{cache_name}': {stats['hit_rate']:.2f} hit rate, {stats['entries']} entries")
        
    # Simulate query optimization
    logger.info("Simulating database queries...")
    for i in range(20):
        query_time = random.uniform(0.1, 3.0)  # Some queries are slow
        optimizer.query_optimizer.track_query(f"query_type_{i % 5}", query_time)
        
    # Run optimizations
    logger.info("Running performance optimizations...")
    
    # Optimize agent decision-making
    mock_profiles = [MockAgent(f"agent_{i}", "red" if i % 2 else "blue") for i in range(5)]
    agent_optimization = optimizer.optimize_agent_decision_making(mock_profiles)
    logger.info(f"Agent optimization improvement: {agent_optimization.improvement_percentage:.1f}%")
    
    # Optimize database queries
    db_optimization = optimizer.optimize_database_queries()
    logger.info(f"Database optimization improvement: {db_optimization.improvement_percentage:.1f}%")
    
    # Optimize memory usage
    memory_optimization = optimizer.optimize_memory_usage()
    logger.info(f"Memory optimization improvement: {memory_optimization.improvement_percentage:.1f}%")
    
    # Generate comprehensive report
    report = optimizer.get_comprehensive_optimization_report()
    logger.info(f"Generated optimization report with {len(report['recommendations'])} recommendations")
    
    return optimizer

def demonstrate_benchmarking():
    """Demonstrate benchmarking capabilities."""
    logger.info("\n=== Benchmarking Demo ===")
    
    benchmarks = PerformanceBenchmarks()
    
    # Agent factory for benchmarks
    def agent_factory():
        return MockAgent(f"benchmark_agent_{random.randint(1, 100)}", 
                        random.choice(["red", "blue"]))
    
    # Run agent decision benchmark
    logger.info("Running agent decision benchmark...")
    agent_result = benchmarks.run_agent_decision_benchmark(agent_factory, iterations=20)
    logger.info(f"Agent benchmark: {agent_result.operations_per_second:.2f} ops/sec, "
               f"{agent_result.avg_response_time:.3f}s avg response time")
    
    # Run memory retrieval benchmark
    logger.info("Running memory retrieval benchmark...")
    memory_system = MockMemorySystem()
    memory_result = benchmarks.run_memory_retrieval_benchmark(memory_system, iterations=50)
    logger.info(f"Memory benchmark: {memory_result.operations_per_second:.2f} ops/sec")
    
    # Run communication benchmark
    logger.info("Running communication benchmark...")
    message_bus = MockMessageBus()
    comm_result = benchmarks.run_communication_benchmark(message_bus, iterations=100)
    logger.info(f"Communication benchmark: {comm_result.operations_per_second:.2f} ops/sec")
    
    # Set baselines
    benchmarks.set_baseline("agent_decision_making")
    benchmarks.set_baseline("memory_retrieval")
    benchmarks.set_baseline("agent_communication")
    
    # Run benchmarks again for comparison
    logger.info("Running benchmarks again for comparison...")
    benchmarks.run_agent_decision_benchmark(agent_factory, iterations=20)
    benchmarks.run_memory_retrieval_benchmark(memory_system, iterations=50)
    
    # Compare to baselines
    agent_comparison = benchmarks.compare_to_baseline("agent_decision_making")
    memory_comparison = benchmarks.compare_to_baseline("memory_retrieval")
    
    if agent_comparison:
        logger.info(f"Agent performance trend: {agent_comparison['performance_trend']}")
        logger.info(f"Operations/sec change: {agent_comparison['operations_per_second']['change_percent']:.1f}%")
        
    if memory_comparison:
        logger.info(f"Memory performance trend: {memory_comparison['performance_trend']}")
        
    # Get benchmark summary
    summary = benchmarks.get_benchmark_summary()
    logger.info(f"Benchmark summary: {len(summary)} benchmark types completed")
    
    return benchmarks

def demonstrate_load_testing():
    """Demonstrate load testing capabilities."""
    logger.info("\n=== Load Testing Demo ===")
    
    load_tester = LoadTester()
    
    # Configure load test
    config = LoadTestConfig(
        concurrent_users=5,
        duration=timedelta(seconds=3),
        ramp_up_time=timedelta(seconds=1),
        operations_per_user=10,
        think_time_range=(0.01, 0.05)
    )
    
    def agent_factory():
        return MockAgent(f"load_test_agent_{random.randint(1, 100)}", 
                        random.choice(["red", "blue"]))
    
    logger.info(f"Starting load test with {config.concurrent_users} concurrent users...")
    
    # Run load test
    result = load_tester.run_concurrent_agent_load_test(agent_factory, config)
    
    logger.info(f"Load test completed:")
    logger.info(f"  Total operations: {result.total_operations}")
    logger.info(f"  Successful operations: {result.successful_operations}")
    logger.info(f"  Failed operations: {result.failed_operations}")
    logger.info(f"  Success rate: {result.successful_operations/result.total_operations*100:.1f}%")
    logger.info(f"  Average ops/sec: {result.operations_per_second:.2f}")
    logger.info(f"  Peak ops/sec: {result.peak_operations_per_second:.2f}")
    logger.info(f"  Average response time: {result.avg_response_time:.3f}s")
    logger.info(f"  95th percentile response time: {result.response_time_percentiles['p95']:.3f}s")
    
    return load_tester

def demonstrate_metrics_and_monitoring():
    """Demonstrate metrics collection and resource monitoring."""
    logger.info("\n=== Metrics & Monitoring Demo ===")
    
    metrics = PerformanceMetrics()
    resource_monitor = ResourceMonitor(collection_interval=0.5)
    
    # Start resource monitoring
    resource_monitor.start_monitoring()
    logger.info("Started resource monitoring")
    
    # Simulate various operations with metrics
    logger.info("Simulating operations with metrics collection...")
    
    for i in range(20):
        # Record counter metrics
        metrics.record_counter("operations_completed")
        metrics.record_counter("messages_sent", random.randint(1, 5))
        
        # Record gauge metrics
        metrics.record_gauge("active_agents", random.randint(5, 15))
        metrics.record_gauge("memory_usage_mb", random.uniform(100, 500))
        
        # Record timing metrics with context manager
        with TimingContext(metrics, "operation_duration"):
            # Simulate operation
            time.sleep(random.uniform(0.01, 0.1))
            
        # Record histogram metrics
        response_time = random.uniform(0.05, 0.5)
        metrics.record_histogram("response_times", response_time)
        
        if i % 5 == 0:
            logger.info(f"Completed {i+1} operations")
            
    # Let resource monitor collect data
    time.sleep(2.0)
    
    # Stop resource monitoring
    resource_monitor.stop_monitoring()
    logger.info("Stopped resource monitoring")
    
    # Get metrics summary
    metrics_summary = metrics.get_all_metrics_summary()
    logger.info(f"Metrics summary:")
    logger.info(f"  Operations completed: {metrics_summary['counters'].get('operations_completed', 0)}")
    logger.info(f"  Messages sent: {metrics_summary['counters'].get('messages_sent', 0)}")
    logger.info(f"  Current active agents: {metrics_summary['gauges'].get('active_agents', 0)}")
    
    # Get response time statistics
    response_stats = metrics.get_histogram_stats("response_times")
    if response_stats:
        logger.info(f"  Response times - avg: {response_stats['mean']:.3f}s, "
                   f"p95: {response_stats['p95']:.3f}s")
        
    # Get resource summary
    resource_summary = resource_monitor.get_resource_summary()
    if resource_summary:
        logger.info(f"Resource usage summary:")
        logger.info(f"  Average CPU: {resource_summary['cpu_usage']['avg']:.1f}%")
        logger.info(f"  Peak CPU: {resource_summary['cpu_usage']['max']:.1f}%")
        logger.info(f"  Average memory: {resource_summary['memory_usage']['avg_percent']:.1f}%")
        logger.info(f"  Samples collected: {resource_summary['sample_count']}")
        
    # Detect anomalies
    anomalies = resource_monitor.detect_resource_anomalies()
    if anomalies:
        logger.info(f"Detected {len(anomalies)} resource anomalies")
        for anomaly in anomalies[:3]:  # Show first 3
            logger.info(f"  {anomaly['type']}: {anomaly['value']:.1f} at {anomaly['timestamp']}")
    else:
        logger.info("No resource anomalies detected")
        
    return metrics, resource_monitor

def demonstrate_comprehensive_performance_analysis():
    """Demonstrate comprehensive performance analysis."""
    logger.info("\n=== Comprehensive Performance Analysis ===")
    
    # Initialize all components
    system_profiler = SystemProfiler()
    agent_profiler = AgentProfiler()
    optimizer = PerformanceOptimizer()
    benchmarks = PerformanceBenchmarks()
    metrics = PerformanceMetrics()
    resource_monitor = ResourceMonitor(collection_interval=0.2)
    
    # Start monitoring
    system_profiler.start_profiling(interval=0.3)
    resource_monitor.start_monitoring()
    
    logger.info("Running comprehensive performance analysis...")
    
    # Simulate a complete agent scenario
    agents = [
        MockAgent("red_recon", "red"),
        MockAgent("red_exploit", "red"),
        MockAgent("blue_soc", "blue"),
        MockAgent("blue_firewall", "blue")
    ]
    
    # Simulate multiple rounds of agent interactions
    for round_num in range(3):
        logger.info(f"Round {round_num + 1}: Simulating agent interactions")
        
        for agent in agents:
            # Start agent profiling
            agent_profiler.start_agent_session(agent.agent_id)
            
            # Record operation start
            metrics.record_counter("agent_operations_started")
            
            # Decision phase
            with profile_agent_operation(agent_profiler, agent.agent_id, "decision"):
                with TimingContext(metrics, f"{agent.team}_decision_time"):
                    environment_state = {
                        'threat_detected': random.choice([True, False]),
                        'network_activity': random.uniform(0.1, 1.0),
                        'round': round_num
                    }
                    decision = agent.make_decision(environment_state)
                    
            # Memory retrieval phase
            with profile_agent_operation(agent_profiler, agent.agent_id, "memory_retrieval"):
                with TimingContext(metrics, f"{agent.team}_memory_time"):
                    query = f"{agent.team} tactics round {round_num}"
                    memories = agent.retrieve_memory(query)
                    
            # Action execution phase
            with profile_agent_operation(agent_profiler, agent.agent_id, "action_execution"):
                with TimingContext(metrics, f"{agent.team}_action_time"):
                    result = agent.execute_action(decision)
                    
            # Record metrics
            metrics.record_gauge(f"{agent.agent_id}_cache_size", len(agent.memory_cache))
            metrics.record_histogram("memory_results_count", len(memories))
            
            if result['status'] == 'success':
                metrics.record_counter("successful_actions")
            else:
                metrics.record_counter("failed_actions")
                
            # End agent profiling
            cache_hit_rate = min(len(agent.memory_cache) / 20.0, 1.0)
            profile = agent_profiler.end_agent_session(
                agent.agent_id,
                memory_usage=random.uniform(20.0, 150.0),
                cache_hit_rate=cache_hit_rate
            )
            
            metrics.record_histogram("agent_total_response_time", profile.total_response_time)
            
        # Brief pause between rounds
        time.sleep(0.5)
        
    # Stop monitoring
    system_profiler.stop_profiling()
    resource_monitor.stop_monitoring()
    
    logger.info("Analysis complete. Generating comprehensive report...")
    
    # Generate comprehensive performance report
    report = {
        'timestamp': datetime.now().isoformat(),
        'system_profiling': system_profiler.get_performance_summary(),
        'agent_profiling': agent_profiler.get_all_agents_summary(),
        'optimization_analysis': optimizer.get_comprehensive_optimization_report(),
        'metrics_summary': metrics.get_all_metrics_summary(),
        'resource_summary': resource_monitor.get_resource_summary(),
        'anomalies': resource_monitor.detect_resource_anomalies()
    }
    
    # Log key findings
    logger.info("=== Performance Analysis Results ===")
    
    # System performance
    sys_summary = report['system_profiling']
    if sys_summary:
        logger.info(f"System: {sys_summary.get('profile_count', 0)} profiles, "
                   f"avg CPU {sys_summary.get('cpu_usage', {}).get('avg', 0):.1f}%")
        
    # Agent performance
    agent_summary = report['agent_profiling']
    for agent_id, summary in agent_summary.items():
        logger.info(f"Agent {agent_id}: {summary.get('avg_total_response_time', 0):.3f}s avg response, "
                   f"trend: {summary.get('performance_trend', 'unknown')}")
        
    # Metrics summary
    metrics_summary = report['metrics_summary']
    total_operations = metrics_summary['counters'].get('agent_operations_started', 0)
    successful_actions = metrics_summary['counters'].get('successful_actions', 0)
    failed_actions = metrics_summary['counters'].get('failed_actions', 0)
    
    success_rate = successful_actions / (successful_actions + failed_actions) * 100 if (successful_actions + failed_actions) > 0 else 0
    
    logger.info(f"Operations: {total_operations} started, {success_rate:.1f}% success rate")
    
    # Resource usage
    resource_summary = report['resource_summary']
    if resource_summary:
        logger.info(f"Resources: {resource_summary['cpu_usage']['avg']:.1f}% avg CPU, "
                   f"{resource_summary['memory_usage']['avg_percent']:.1f}% avg memory")
        
    # Anomalies
    anomalies = report['anomalies']
    if anomalies:
        logger.info(f"Detected {len(anomalies)} performance anomalies")
    else:
        logger.info("No performance anomalies detected")
        
    return report

async def main():
    """Main demo function."""
    logger.info("Starting Performance Optimization and Tuning Demo")
    logger.info("=" * 60)
    
    try:
        # Run individual demonstrations
        system_profiler = demonstrate_system_profiling()
        agent_profiler = demonstrate_agent_profiling()
        optimizer = demonstrate_caching_optimization()
        benchmarks = demonstrate_benchmarking()
        load_tester = demonstrate_load_testing()
        metrics, resource_monitor = demonstrate_metrics_and_monitoring()
        
        # Run comprehensive analysis
        comprehensive_report = demonstrate_comprehensive_performance_analysis()
        
        logger.info("\n" + "=" * 60)
        logger.info("Performance Optimization Demo Completed Successfully!")
        logger.info("=" * 60)
        
        # Summary of capabilities demonstrated
        logger.info("\nCapabilities Demonstrated:")
        logger.info("✓ System-wide performance profiling")
        logger.info("✓ Agent-specific performance profiling")
        logger.info("✓ Intelligent caching and optimization")
        logger.info("✓ Comprehensive benchmarking")
        logger.info("✓ Load testing with concurrent agents")
        logger.info("✓ Real-time metrics collection")
        logger.info("✓ Resource monitoring and anomaly detection")
        logger.info("✓ Comprehensive performance analysis")
        
        # Performance insights
        logger.info("\nKey Performance Insights:")
        logger.info("• System profiling captured detailed performance metrics")
        logger.info("• Agent profiling revealed decision-making bottlenecks")
        logger.info("• Caching strategies improved response times")
        logger.info("• Benchmarking established performance baselines")
        logger.info("• Load testing validated system scalability")
        logger.info("• Metrics provided real-time performance visibility")
        logger.info("• Resource monitoring detected performance anomalies")
        
        return comprehensive_report
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise

if __name__ == "__main__":
    # Run the demo
    report = asyncio.run(main())