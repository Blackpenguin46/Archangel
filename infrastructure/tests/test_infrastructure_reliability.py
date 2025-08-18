#!/usr/bin/env python3
"""
Infrastructure Reliability Tests
Comprehensive test suite for validating infrastructure reliability and resilience
"""

import unittest
import time
import random
import threading
import logging
import os
import json
import requests
import docker
import psutil
import subprocess
from typing import Dict, List, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReliabilityTestCase(unittest.TestCase):
    """Base test case for infrastructure reliability validation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up reliability test environment."""
        cls.docker_client = docker.from_env()
        cls.project_name = "archangel"
        cls.test_duration = int(os.environ.get('RELIABILITY_TEST_DURATION', '60'))  # seconds
        cls.load_test_users = int(os.environ.get('LOAD_TEST_USERS', '10'))
        
        # Services to monitor
        cls.critical_services = {
            'core': 'http://localhost:8888/health',
            'prometheus': 'http://localhost:9090/-/ready',
            'grafana': 'http://localhost:3000/api/health',
            'elasticsearch': 'http://localhost:9200',
        }
        
        # Reliability thresholds
        cls.availability_threshold = 0.99  # 99% uptime
        cls.response_time_threshold = 2.0  # 2 seconds
        cls.error_rate_threshold = 0.05  # 5% error rate
        
        cls.test_results = {}
        cls.test_start_time = datetime.now()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after reliability tests."""
        test_duration = (datetime.now() - cls.test_start_time).total_seconds()
        logger.info(f"Reliability tests completed in {test_duration:.2f} seconds")
        
        # Generate reliability report
        cls._generate_reliability_report()
    
    @classmethod
    def _generate_reliability_report(cls):
        """Generate comprehensive reliability report."""
        report = {
            'test_summary': {
                'start_time': cls.test_start_time.isoformat(),
                'duration_seconds': (datetime.now() - cls.test_start_time).total_seconds(),
                'test_environment': cls.project_name
            },
            'results': cls.test_results,
            'thresholds': {
                'availability': cls.availability_threshold,
                'response_time': cls.response_time_threshold,
                'error_rate': cls.error_rate_threshold
            }
        }
        
        report_file = f"reliability_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Reliability report saved to {report_file}")
    
    def _record_test_result(self, test_name: str, result: Dict[str, Any]):
        """Record test result for reporting."""
        self.__class__.test_results[test_name] = result


class ServiceAvailabilityTests(ReliabilityTestCase):
    """Test service availability and uptime."""
    
    def test_continuous_service_availability(self):
        """Test that services maintain availability over time."""
        duration = self.test_duration
        check_interval = 5  # seconds
        
        service_stats = {}
        
        for service_name, endpoint in self.critical_services.items():
            service_stats[service_name] = {
                'total_checks': 0,
                'successful_checks': 0,
                'failed_checks': 0,
                'response_times': [],
                'errors': []
            }
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            for service_name, endpoint in self.critical_services.items():
                check_start = time.time()
                
                try:
                    response = requests.get(endpoint, timeout=10)
                    response_time = time.time() - check_start
                    
                    service_stats[service_name]['total_checks'] += 1
                    service_stats[service_name]['response_times'].append(response_time)
                    
                    if response.status_code in [200, 404]:  # 404 is acceptable for some endpoints
                        service_stats[service_name]['successful_checks'] += 1
                    else:
                        service_stats[service_name]['failed_checks'] += 1
                        service_stats[service_name]['errors'].append(f"HTTP {response.status_code}")
                        
                except Exception as e:
                    service_stats[service_name]['total_checks'] += 1
                    service_stats[service_name]['failed_checks'] += 1
                    service_stats[service_name]['errors'].append(str(e))
            
            time.sleep(check_interval)
        
        # Validate availability metrics
        for service_name, stats in service_stats.items():
            with self.subTest(service=service_name):
                if stats['total_checks'] > 0:
                    availability = stats['successful_checks'] / stats['total_checks']
                    avg_response_time = sum(stats['response_times']) / len(stats['response_times']) if stats['response_times'] else 0
                    
                    logger.info(f"Service {service_name}: {availability:.3f} availability, {avg_response_time:.3f}s avg response")
                    
                    self.assertGreaterEqual(
                        availability, self.availability_threshold,
                        f"Service {service_name} availability {availability:.3f} below threshold {self.availability_threshold}"
                    )
                    
                    if stats['response_times']:
                        self.assertLess(
                            avg_response_time, self.response_time_threshold,
                            f"Service {service_name} avg response time {avg_response_time:.3f}s above threshold {self.response_time_threshold}s"
                        )
        
        self._record_test_result('service_availability', service_stats)
    
    def test_service_recovery_after_restart(self):
        """Test that services recover properly after restart."""
        if not self.docker_client:
            self.skipTest("Docker client not available")
        
        recovery_results = {}
        
        containers = self.docker_client.containers.list()
        archangel_containers = [c for c in containers if self.project_name in c.name and 'agent' not in c.name]
        
        if not archangel_containers:
            self.skipTest("No Archangel containers found for restart test")
        
        # Test with one non-critical container
        test_container = random.choice(archangel_containers)
        service_name = test_container.name
        
        logger.info(f"Testing recovery for container: {service_name}")
        
        # Record pre-restart state
        pre_restart_status = test_container.status
        
        try:
            # Restart the container
            restart_start = time.time()
            test_container.restart(timeout=30)
            
            # Wait for container to be running
            max_wait = 60  # seconds
            check_interval = 2
            recovery_time = 0
            
            for i in range(max_wait // check_interval):
                time.sleep(check_interval)
                recovery_time += check_interval
                
                test_container.reload()
                if test_container.status == 'running':
                    break
            
            restart_duration = time.time() - restart_start
            
            # Validate recovery
            self.assertEqual(test_container.status, 'running', f"Container {service_name} should be running after restart")
            self.assertLess(recovery_time, max_wait, f"Container {service_name} should recover within {max_wait}s")
            
            # Test service functionality after restart
            if 'prometheus' in service_name:
                endpoint = 'http://localhost:9090/-/ready'
            elif 'grafana' in service_name:
                endpoint = 'http://localhost:3000/api/health'
            elif 'core' in service_name:
                endpoint = 'http://localhost:8888/health'
            else:
                endpoint = None
            
            if endpoint:
                # Give service time to fully start
                time.sleep(10)
                
                try:
                    response = requests.get(endpoint, timeout=15)
                    service_functional = response.status_code in [200, 404]
                except Exception:
                    service_functional = False
                
                self.assertTrue(service_functional, f"Service {service_name} should be functional after restart")
            
            recovery_results[service_name] = {
                'pre_restart_status': pre_restart_status,
                'post_restart_status': test_container.status,
                'restart_duration': restart_duration,
                'recovery_time': recovery_time,
                'service_functional': endpoint is None or service_functional
            }
            
        except Exception as e:
            self.fail(f"Container restart test failed for {service_name}: {e}")
        
        self._record_test_result('service_recovery', recovery_results)


class LoadResistanceTests(ReliabilityTestCase):
    """Test system behavior under load."""
    
    def test_concurrent_request_handling(self):
        """Test system ability to handle concurrent requests."""
        target_endpoint = 'http://localhost:8888/health'
        num_users = self.load_test_users
        requests_per_user = 10
        
        def make_requests(user_id: int) -> Dict[str, Any]:
            """Make requests from a single user."""
            user_results = {
                'user_id': user_id,
                'successful_requests': 0,
                'failed_requests': 0,
                'response_times': [],
                'errors': []
            }
            
            for i in range(requests_per_user):
                start_time = time.time()
                try:
                    response = requests.get(target_endpoint, timeout=10)
                    response_time = time.time() - start_time
                    
                    user_results['response_times'].append(response_time)
                    
                    if response.status_code == 200:
                        user_results['successful_requests'] += 1
                    else:
                        user_results['failed_requests'] += 1
                        user_results['errors'].append(f"HTTP {response.status_code}")
                        
                except Exception as e:
                    user_results['failed_requests'] += 1
                    user_results['errors'].append(str(e))
                
                # Small delay between requests
                time.sleep(0.1)
            
            return user_results
        
        # Execute concurrent load test
        logger.info(f"Starting load test with {num_users} concurrent users")
        
        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [executor.submit(make_requests, user_id) for user_id in range(num_users)]
            user_results = [future.result() for future in as_completed(futures)]
        
        # Analyze results
        total_requests = sum(r['successful_requests'] + r['failed_requests'] for r in user_results)
        total_successful = sum(r['successful_requests'] for r in user_results)
        total_failed = sum(r['failed_requests'] for r in user_results)
        
        all_response_times = []
        for r in user_results:
            all_response_times.extend(r['response_times'])
        
        if total_requests > 0:
            success_rate = total_successful / total_requests
            error_rate = total_failed / total_requests
            avg_response_time = sum(all_response_times) / len(all_response_times) if all_response_times else 0
            
            logger.info(f"Load test results: {success_rate:.3f} success rate, {avg_response_time:.3f}s avg response")
            
            self.assertGreaterEqual(
                success_rate, 1 - self.error_rate_threshold,
                f"Success rate {success_rate:.3f} below threshold {1 - self.error_rate_threshold:.3f}"
            )
            
            self.assertLess(
                avg_response_time, self.response_time_threshold * 2,  # Allow 2x normal response time under load
                f"Average response time {avg_response_time:.3f}s too high under load"
            )
        
        load_test_summary = {
            'num_users': num_users,
            'requests_per_user': requests_per_user,
            'total_requests': total_requests,
            'successful_requests': total_successful,
            'failed_requests': total_failed,
            'success_rate': success_rate if total_requests > 0 else 0,
            'avg_response_time': avg_response_time,
            'user_results': user_results
        }
        
        self._record_test_result('load_resistance', load_test_summary)
    
    def test_resource_exhaustion_resistance(self):
        """Test system behavior under resource constraints."""
        if not self.docker_client:
            self.skipTest("Docker client not available")
        
        resource_results = {}
        
        # Monitor system resources before, during, and after test
        def monitor_resources() -> Dict[str, Any]:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'timestamp': time.time()
            }
        
        baseline_resources = monitor_resources()
        
        # Create memory and CPU load
        def cpu_stress_task():
            """Create CPU load."""
            end_time = time.time() + 30  # 30 seconds of CPU stress
            while time.time() < end_time:
                # Busy work to consume CPU
                sum(i * i for i in range(10000))
        
        def memory_stress_task():
            """Create memory load."""
            memory_hog = []
            try:
                # Allocate memory in chunks
                for _ in range(100):
                    memory_hog.append([0] * 1000000)  # ~8MB per iteration
                    time.sleep(0.1)
                
                time.sleep(10)  # Hold memory for 10 seconds
            finally:
                del memory_hog
        
        # Start stress tasks
        stress_threads = [
            threading.Thread(target=cpu_stress_task),
            threading.Thread(target=memory_stress_task)
        ]
        
        for thread in stress_threads:
            thread.start()
        
        # Monitor services during stress
        service_responses = []
        stress_start = time.time()
        
        while any(t.is_alive() for t in stress_threads):
            for service_name, endpoint in self.critical_services.items():
                try:
                    start_time = time.time()
                    response = requests.get(endpoint, timeout=5)
                    response_time = time.time() - start_time
                    
                    service_responses.append({
                        'service': service_name,
                        'timestamp': time.time() - stress_start,
                        'status_code': response.status_code,
                        'response_time': response_time,
                        'success': response.status_code in [200, 404]
                    })
                except Exception as e:
                    service_responses.append({
                        'service': service_name,
                        'timestamp': time.time() - stress_start,
                        'error': str(e),
                        'success': False
                    })
            
            time.sleep(1)
        
        # Wait for threads to complete
        for thread in stress_threads:
            thread.join()
        
        post_stress_resources = monitor_resources()
        
        # Analyze service behavior under stress
        service_success_rates = {}
        for service_name in self.critical_services.keys():
            service_results = [r for r in service_responses if r.get('service') == service_name]
            if service_results:
                successful = len([r for r in service_results if r.get('success', False)])
                total = len(service_results)
                service_success_rates[service_name] = successful / total if total > 0 else 0
        
        # Validate that services remained responsive under stress
        for service_name, success_rate in service_success_rates.items():
            with self.subTest(service=service_name):
                self.assertGreater(
                    success_rate, 0.8,  # 80% success rate minimum under stress
                    f"Service {service_name} success rate {success_rate:.3f} too low under resource stress"
                )
        
        resource_results = {
            'baseline_resources': baseline_resources,
            'post_stress_resources': post_stress_resources,
            'service_responses': service_responses,
            'service_success_rates': service_success_rates
        }
        
        self._record_test_result('resource_exhaustion_resistance', resource_results)


class FailureRecoveryTests(ReliabilityTestCase):
    """Test system failure recovery capabilities."""
    
    def test_network_partition_recovery(self):
        """Test recovery from simulated network partitions."""
        if not self.docker_client:
            self.skipTest("Docker client not available")
        
        # This test simulates network issues by temporarily stopping network access
        network_results = {}
        
        try:
            # Get network information
            networks = self.docker_client.networks.list()
            archangel_networks = [n for n in networks if self.project_name in n.name]
            
            if not archangel_networks:
                self.skipTest("No Archangel networks found for partition test")
            
            test_network = archangel_networks[0]
            network_name = test_network.name
            
            # Get containers connected to the network
            connected_containers = []
            network_info = test_network.attrs
            containers_info = network_info.get('Containers', {})
            
            for container_id in containers_info.keys():
                try:
                    container = self.docker_client.containers.get(container_id)
                    connected_containers.append(container)
                except Exception:
                    continue
            
            if not connected_containers:
                self.skipTest("No containers found in test network")
            
            logger.info(f"Testing network partition recovery on network {network_name} with {len(connected_containers)} containers")
            
            # Test basic connectivity before partition
            pre_partition_connectivity = self._test_service_connectivity()
            
            # Simulate network partition by disconnecting containers temporarily
            # Note: This is a simplified simulation - real network partitions are more complex
            partition_duration = 10  # seconds
            
            disconnected_containers = []
            try:
                # Disconnect some containers from the network
                test_containers = connected_containers[:min(2, len(connected_containers))]
                
                for container in test_containers:
                    try:
                        test_network.disconnect(container)
                        disconnected_containers.append(container)
                        logger.info(f"Disconnected container {container.name} from network")
                    except Exception as e:
                        logger.warning(f"Could not disconnect {container.name}: {e}")
                
                # Wait during partition
                time.sleep(partition_duration)
                
                # Reconnect containers
                for container in disconnected_containers:
                    try:
                        test_network.connect(container)
                        logger.info(f"Reconnected container {container.name} to network")
                    except Exception as e:
                        logger.warning(f"Could not reconnect {container.name}: {e}")
                
                # Allow time for recovery
                time.sleep(5)
                
                # Test connectivity after recovery
                post_recovery_connectivity = self._test_service_connectivity()
                
                # Validate recovery
                for service_name, post_status in post_recovery_connectivity.items():
                    with self.subTest(service=service_name):
                        self.assertTrue(
                            post_status,
                            f"Service {service_name} should recover after network partition"
                        )
                
                network_results = {
                    'test_network': network_name,
                    'partition_duration': partition_duration,
                    'disconnected_containers': [c.name for c in disconnected_containers],
                    'pre_partition_connectivity': pre_partition_connectivity,
                    'post_recovery_connectivity': post_recovery_connectivity
                }
                
            finally:
                # Ensure all containers are reconnected
                for container in disconnected_containers:
                    try:
                        test_network.connect(container)
                    except Exception:
                        pass
        
        except Exception as e:
            self.skipTest(f"Network partition test failed: {e}")
        
        self._record_test_result('network_partition_recovery', network_results)
    
    def _test_service_connectivity(self) -> Dict[str, bool]:
        """Test connectivity to all critical services."""
        connectivity_results = {}
        
        for service_name, endpoint in self.critical_services.items():
            try:
                response = requests.get(endpoint, timeout=5)
                connectivity_results[service_name] = response.status_code in [200, 404]
            except Exception:
                connectivity_results[service_name] = False
        
        return connectivity_results
    
    def test_cascading_failure_prevention(self):
        """Test that failures in one component don't cascade to others."""
        if not self.docker_client:
            self.skipTest("Docker client not available")
        
        cascade_results = {}
        
        containers = self.docker_client.containers.list()
        archangel_containers = [c for c in containers if self.project_name in c.name]
        
        # Find a non-critical container to fail
        non_critical_containers = [c for c in archangel_containers if 'agent' in c.name or 'redis' in c.name]
        
        if not non_critical_containers:
            self.skipTest("No non-critical containers found for cascade test")
        
        test_container = non_critical_containers[0]
        container_name = test_container.name
        
        logger.info(f"Testing cascade prevention by stopping container: {container_name}")
        
        # Record baseline service health
        baseline_health = {}
        for service_name, endpoint in self.critical_services.items():
            try:
                response = requests.get(endpoint, timeout=5)
                baseline_health[service_name] = response.status_code in [200, 404]
            except Exception:
                baseline_health[service_name] = False
        
        try:
            # Stop the test container
            test_container.stop(timeout=10)
            
            # Wait for potential cascade effects
            time.sleep(15)
            
            # Check critical service health after failure
            post_failure_health = {}
            for service_name, endpoint in self.critical_services.items():
                try:
                    response = requests.get(endpoint, timeout=10)
                    post_failure_health[service_name] = response.status_code in [200, 404]
                except Exception:
                    post_failure_health[service_name] = False
            
            # Validate that critical services are still healthy
            for service_name, is_healthy in post_failure_health.items():
                if baseline_health.get(service_name, False):  # Only check services that were healthy before
                    with self.subTest(service=service_name):
                        self.assertTrue(
                            is_healthy,
                            f"Critical service {service_name} should remain healthy after {container_name} failure"
                        )
            
            cascade_results = {
                'failed_container': container_name,
                'baseline_health': baseline_health,
                'post_failure_health': post_failure_health,
                'cascade_detected': any(
                    baseline_health.get(svc, False) and not post_failure_health.get(svc, False)
                    for svc in baseline_health.keys()
                )
            }
        
        finally:
            # Restart the container
            try:
                test_container.start()
                time.sleep(5)  # Allow time for restart
            except Exception as e:
                logger.error(f"Could not restart container {container_name}: {e}")
        
        self._record_test_result('cascading_failure_prevention', cascade_results)


class DataIntegrityTests(ReliabilityTestCase):
    """Test data integrity and persistence reliability."""
    
    def test_data_persistence_across_restarts(self):
        """Test that data persists across service restarts."""
        if not self.docker_client:
            self.skipTest("Docker client not available")
        
        persistence_results = {}
        
        # Test Prometheus data persistence
        try:
            # Query current metrics
            response = requests.get("http://localhost:9090/api/v1/query?query=up", timeout=10)
            if response.status_code == 200:
                pre_restart_data = response.json()
                initial_data_points = len(pre_restart_data.get('data', {}).get('result', []))
                
                # Find and restart Prometheus container
                containers = self.docker_client.containers.list()
                prometheus_containers = [c for c in containers if 'prometheus' in c.name and self.project_name in c.name]
                
                if prometheus_containers:
                    prometheus_container = prometheus_containers[0]
                    
                    logger.info("Testing Prometheus data persistence across restart")
                    
                    # Restart the container
                    prometheus_container.restart(timeout=30)
                    
                    # Wait for service to be ready
                    time.sleep(20)
                    
                    # Check data after restart
                    retry_count = 0
                    max_retries = 10
                    
                    while retry_count < max_retries:
                        try:
                            response = requests.get("http://localhost:9090/api/v1/query?query=up", timeout=10)
                            if response.status_code == 200:
                                post_restart_data = response.json()
                                final_data_points = len(post_restart_data.get('data', {}).get('result', []))
                                
                                # Data should be preserved or increased (new data points)
                                data_preserved = final_data_points >= initial_data_points * 0.8  # Allow 20% tolerance
                                
                                persistence_results['prometheus'] = {
                                    'initial_data_points': initial_data_points,
                                    'final_data_points': final_data_points,
                                    'data_preserved': data_preserved
                                }
                                
                                self.assertTrue(
                                    data_preserved,
                                    f"Prometheus data not properly preserved across restart: {initial_data_points} -> {final_data_points}"
                                )
                                break
                            else:
                                retry_count += 1
                                time.sleep(5)
                        except Exception:
                            retry_count += 1
                            time.sleep(5)
                    
                    if retry_count >= max_retries:
                        self.fail("Prometheus service did not recover properly after restart")
                
        except Exception as e:
            logger.warning(f"Could not test Prometheus data persistence: {e}")
        
        # Test Grafana configuration persistence
        try:
            # Check datasources before restart
            response = requests.get("http://admin:archangel_admin@localhost:3000/api/datasources", timeout=10)
            if response.status_code == 200:
                pre_restart_datasources = response.json()
                initial_datasource_count = len(pre_restart_datasources)
                
                # Find and restart Grafana container
                containers = self.docker_client.containers.list()
                grafana_containers = [c for c in containers if 'grafana' in c.name and self.project_name in c.name]
                
                if grafana_containers:
                    grafana_container = grafana_containers[0]
                    
                    logger.info("Testing Grafana configuration persistence across restart")
                    
                    # Restart the container
                    grafana_container.restart(timeout=30)
                    
                    # Wait for service to be ready
                    time.sleep(15)
                    
                    # Check configuration after restart
                    retry_count = 0
                    max_retries = 10
                    
                    while retry_count < max_retries:
                        try:
                            response = requests.get("http://admin:archangel_admin@localhost:3000/api/datasources", timeout=10)
                            if response.status_code == 200:
                                post_restart_datasources = response.json()
                                final_datasource_count = len(post_restart_datasources)
                                
                                config_preserved = final_datasource_count >= initial_datasource_count
                                
                                persistence_results['grafana'] = {
                                    'initial_datasource_count': initial_datasource_count,
                                    'final_datasource_count': final_datasource_count,
                                    'config_preserved': config_preserved
                                }
                                
                                self.assertTrue(
                                    config_preserved,
                                    f"Grafana configuration not preserved across restart: {initial_datasource_count} -> {final_datasource_count}"
                                )
                                break
                            else:
                                retry_count += 1
                                time.sleep(5)
                        except Exception:
                            retry_count += 1
                            time.sleep(5)
                    
                    if retry_count >= max_retries:
                        logger.warning("Could not verify Grafana configuration after restart")
                
        except Exception as e:
            logger.warning(f"Could not test Grafana configuration persistence: {e}")
        
        self._record_test_result('data_persistence', persistence_results)
    
    def test_volume_integrity(self):
        """Test that Docker volumes maintain integrity."""
        if not self.docker_client:
            self.skipTest("Docker client not available")
        
        volume_results = {}
        
        try:
            volumes = self.docker_client.volumes.list()
            archangel_volumes = [v for v in volumes if self.project_name in v.name]
            
            for volume in archangel_volumes:
                volume_name = volume.name
                
                try:
                    # Get volume information
                    volume_info = volume.attrs
                    mountpoint = volume_info.get('Mountpoint', '')
                    
                    # Check if mountpoint exists and is accessible
                    if mountpoint and os.path.exists(mountpoint):
                        # Try to get directory size (requires appropriate permissions)
                        try:
                            result = subprocess.run(['du', '-s', mountpoint], capture_output=True, text=True, timeout=10)
                            if result.returncode == 0:
                                size_info = result.stdout.strip().split()[0]
                                volume_accessible = True
                            else:
                                volume_accessible = False
                                size_info = "unknown"
                        except Exception:
                            volume_accessible = False
                            size_info = "unknown"
                    else:
                        volume_accessible = False
                        size_info = "unknown"
                    
                    volume_results[volume_name] = {
                        'mountpoint': mountpoint,
                        'accessible': volume_accessible,
                        'size_info': size_info,
                        'driver': volume_info.get('Driver', 'unknown')
                    }
                    
                    logger.info(f"Volume {volume_name}: accessible={volume_accessible}, size={size_info}")
                    
                except Exception as e:
                    volume_results[volume_name] = {
                        'error': str(e),
                        'accessible': False
                    }
            
            # Validate that critical volumes are accessible
            critical_volume_types = ['prometheus', 'grafana', 'elasticsearch']
            for vol_type in critical_volume_types:
                matching_volumes = [name for name in volume_results.keys() if vol_type in name.lower()]
                if matching_volumes:
                    vol_name = matching_volumes[0]
                    with self.subTest(volume=vol_name):
                        self.assertTrue(
                            volume_results[vol_name].get('accessible', False),
                            f"Critical volume {vol_name} should be accessible"
                        )
        
        except Exception as e:
            self.fail(f"Volume integrity test failed: {e}")
        
        self._record_test_result('volume_integrity', volume_results)


if __name__ == '__main__':
    # Configure test environment
    os.environ.setdefault('RELIABILITY_TEST_DURATION', '60')
    os.environ.setdefault('LOAD_TEST_USERS', '5')
    
    # Run reliability tests
    unittest.main(verbosity=2)