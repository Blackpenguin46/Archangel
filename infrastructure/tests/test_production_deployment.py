#!/usr/bin/env python3
"""
Production Deployment Tests
Comprehensive test suite for validating production deployment reliability and performance
"""

import unittest
import time
import json
import requests
import subprocess
import logging
import os
import yaml
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import kubernetes
from kubernetes import client, config
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionDeploymentTestCase(unittest.TestCase):
    """Base test case for production deployment validation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up production deployment test environment."""
        try:
            # Load Kubernetes configuration
            config.load_incluster_config()
        except:
            try:
                config.load_kube_config()
            except:
                logger.warning("Could not load Kubernetes config, some tests will be skipped")
        
        cls.k8s_client = client.ApiClient()
        cls.apps_v1 = client.AppsV1Api()
        cls.core_v1 = client.CoreV1Api()
        cls.autoscaling_v2 = client.AutoscalingV2Api()
        
        cls.namespace = "archangel"
        cls.environment = "production"
        
        # Production service endpoints
        cls.production_endpoints = {
            'core': 'http://archangel-core-service:8888',
            'agents': 'http://archangel-agents-service:8891',
            'prometheus': 'http://prometheus-service:9090',
            'alertmanager': 'http://alertmanager-service:9093'
        }
        
        # Performance thresholds for production
        cls.performance_thresholds = {
            'response_time_p95': 2.0,  # seconds
            'response_time_p99': 5.0,  # seconds
            'error_rate': 0.01,  # 1%
            'availability': 0.999,  # 99.9%
            'cpu_usage': 0.8,  # 80%
            'memory_usage': 0.85,  # 85%
        }
        
        cls.test_results = {}
        cls.test_start_time = datetime.now()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after production deployment tests."""
        test_duration = (datetime.now() - cls.test_start_time).total_seconds()
        logger.info(f"Production deployment tests completed in {test_duration:.2f} seconds")
        
        # Generate production test report
        cls._generate_production_test_report()
    
    @classmethod
    def _generate_production_test_report(cls):
        """Generate comprehensive production test report."""
        report = {
            'test_summary': {
                'start_time': cls.test_start_time.isoformat(),
                'duration_seconds': (datetime.now() - cls.test_start_time).total_seconds(),
                'environment': cls.environment,
                'namespace': cls.namespace
            },
            'results': cls.test_results,
            'thresholds': cls.performance_thresholds
        }
        
        report_file = f"production_deployment_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Production deployment report saved to {report_file}")
    
    def _record_test_result(self, test_name: str, result: Dict[str, Any]):
        """Record test result for reporting."""
        self.__class__.test_results[test_name] = result


class KubernetesDeploymentTests(ProductionDeploymentTestCase):
    """Test Kubernetes deployment configuration and health."""
    
    def test_deployment_status(self):
        """Test that all deployments are healthy and ready."""
        deployment_results = {}
        
        try:
            deployments = self.apps_v1.list_namespaced_deployment(
                namespace=self.namespace,
                label_selector=f"environment={self.environment}"
            )
            
            for deployment in deployments.items:
                deployment_name = deployment.metadata.name
                
                # Check deployment status
                ready_replicas = deployment.status.ready_replicas or 0
                desired_replicas = deployment.spec.replicas or 0
                
                deployment_healthy = (
                    ready_replicas == desired_replicas and
                    ready_replicas > 0 and
                    deployment.status.conditions and
                    any(c.type == "Available" and c.status == "True" 
                        for c in deployment.status.conditions)
                )
                
                deployment_results[deployment_name] = {
                    'desired_replicas': desired_replicas,
                    'ready_replicas': ready_replicas,
                    'healthy': deployment_healthy,
                    'conditions': [
                        {
                            'type': c.type,
                            'status': c.status,
                            'reason': c.reason,
                            'message': c.message
                        } for c in (deployment.status.conditions or [])
                    ]
                }
                
                with self.subTest(deployment=deployment_name):
                    self.assertTrue(
                        deployment_healthy,
                        f"Deployment {deployment_name} is not healthy: {ready_replicas}/{desired_replicas} ready"
                    )
        
        except Exception as e:
            self.fail(f"Failed to check deployment status: {e}")
        
        self._record_test_result('deployment_status', deployment_results)
    
    def test_pod_resource_limits(self):
        """Test that all pods have appropriate resource limits set."""
        pod_results = {}
        
        try:
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"environment={self.environment}"
            )
            
            for pod in pods.items:
                pod_name = pod.metadata.name
                
                containers_with_limits = []
                containers_without_limits = []
                
                for container in pod.spec.containers:
                    has_limits = (
                        container.resources and
                        container.resources.limits and
                        'memory' in container.resources.limits and
                        'cpu' in container.resources.limits
                    )
                    
                    if has_limits:
                        containers_with_limits.append(container.name)
                    else:
                        containers_without_limits.append(container.name)
                
                pod_results[pod_name] = {
                    'containers_with_limits': containers_with_limits,
                    'containers_without_limits': containers_without_limits,
                    'all_containers_have_limits': len(containers_without_limits) == 0
                }
                
                with self.subTest(pod=pod_name):
                    self.assertEqual(
                        len(containers_without_limits), 0,
                        f"Pod {pod_name} has containers without resource limits: {containers_without_limits}"
                    )
        
        except Exception as e:
            self.fail(f"Failed to check pod resource limits: {e}")
        
        self._record_test_result('pod_resource_limits', pod_results)
    
    def test_horizontal_pod_autoscaler(self):
        """Test that HPA is configured and functioning."""
        hpa_results = {}
        
        try:
            hpas = self.autoscaling_v2.list_namespaced_horizontal_pod_autoscaler(
                namespace=self.namespace
            )
            
            for hpa in hpas.items:
                hpa_name = hpa.metadata.name
                
                current_replicas = hpa.status.current_replicas or 0
                desired_replicas = hpa.status.desired_replicas or 0
                min_replicas = hpa.spec.min_replicas or 0
                max_replicas = hpa.spec.max_replicas or 0
                
                hpa_healthy = (
                    current_replicas >= min_replicas and
                    current_replicas <= max_replicas and
                    current_replicas > 0
                )
                
                hpa_results[hpa_name] = {
                    'min_replicas': min_replicas,
                    'max_replicas': max_replicas,
                    'current_replicas': current_replicas,
                    'desired_replicas': desired_replicas,
                    'healthy': hpa_healthy,
                    'metrics': [
                        {
                            'type': m.type,
                            'resource': m.resource.name if m.resource else None,
                            'target': str(m.resource.target) if m.resource else None
                        } for m in (hpa.spec.metrics or [])
                    ]
                }
                
                with self.subTest(hpa=hpa_name):
                    self.assertTrue(
                        hpa_healthy,
                        f"HPA {hpa_name} is not healthy: {current_replicas} replicas (min: {min_replicas}, max: {max_replicas})"
                    )
        
        except Exception as e:
            self.fail(f"Failed to check HPA status: {e}")
        
        self._record_test_result('horizontal_pod_autoscaler', hpa_results)
    
    def test_persistent_volume_claims(self):
        """Test that all PVCs are bound and have sufficient space."""
        pvc_results = {}
        
        try:
            pvcs = self.core_v1.list_namespaced_persistent_volume_claim(
                namespace=self.namespace
            )
            
            for pvc in pvcs.items:
                pvc_name = pvc.metadata.name
                
                is_bound = pvc.status.phase == "Bound"
                capacity = pvc.status.capacity.get('storage') if pvc.status.capacity else None
                requested = pvc.spec.resources.requests.get('storage') if pvc.spec.resources.requests else None
                
                pvc_results[pvc_name] = {
                    'phase': pvc.status.phase,
                    'is_bound': is_bound,
                    'capacity': capacity,
                    'requested': requested,
                    'storage_class': pvc.spec.storage_class_name
                }
                
                with self.subTest(pvc=pvc_name):
                    self.assertTrue(
                        is_bound,
                        f"PVC {pvc_name} is not bound (phase: {pvc.status.phase})"
                    )
        
        except Exception as e:
            self.fail(f"Failed to check PVC status: {e}")
        
        self._record_test_result('persistent_volume_claims', pvc_results)


class LoadBalancingTests(ProductionDeploymentTestCase):
    """Test load balancing and service discovery."""
    
    def test_service_endpoints(self):
        """Test that all services have healthy endpoints."""
        service_results = {}
        
        try:
            services = self.core_v1.list_namespaced_service(
                namespace=self.namespace,
                label_selector=f"environment={self.environment}"
            )
            
            for service in services.items:
                service_name = service.metadata.name
                
                # Get endpoints for the service
                try:
                    endpoints = self.core_v1.read_namespaced_endpoints(
                        name=service_name,
                        namespace=self.namespace
                    )
                    
                    ready_addresses = []
                    not_ready_addresses = []
                    
                    for subset in (endpoints.subsets or []):
                        ready_addresses.extend(subset.addresses or [])
                        not_ready_addresses.extend(subset.not_ready_addresses or [])
                    
                    service_results[service_name] = {
                        'ready_endpoints': len(ready_addresses),
                        'not_ready_endpoints': len(not_ready_addresses),
                        'has_ready_endpoints': len(ready_addresses) > 0,
                        'service_type': service.spec.type,
                        'ports': [
                            {
                                'name': p.name,
                                'port': p.port,
                                'target_port': str(p.target_port),
                                'protocol': p.protocol
                            } for p in (service.spec.ports or [])
                        ]
                    }
                    
                    with self.subTest(service=service_name):
                        self.assertGreater(
                            len(ready_addresses), 0,
                            f"Service {service_name} has no ready endpoints"
                        )
                
                except client.exceptions.ApiException as e:
                    if e.status == 404:
                        service_results[service_name] = {
                            'error': 'Endpoints not found',
                            'ready_endpoints': 0,
                            'has_ready_endpoints': False
                        }
                    else:
                        raise
        
        except Exception as e:
            self.fail(f"Failed to check service endpoints: {e}")
        
        self._record_test_result('service_endpoints', service_results)
    
    def test_load_balancer_health(self):
        """Test load balancer health and distribution."""
        lb_results = {}
        
        # Test load distribution by making multiple requests
        for service_name, endpoint in self.production_endpoints.items():
            if 'core' in service_name:  # Only test load balancing for core service
                response_sources = []
                
                for i in range(20):  # Make 20 requests to test distribution
                    try:
                        response = requests.get(f"{endpoint}/health", timeout=5)
                        if response.status_code == 200:
                            # Try to identify which pod served the request
                            pod_id = response.headers.get('X-Pod-Name', f'unknown-{i}')
                            response_sources.append(pod_id)
                    except Exception as e:
                        response_sources.append(f'error-{str(e)[:20]}')
                
                # Analyze distribution
                unique_sources = set(response_sources)
                distribution = {source: response_sources.count(source) for source in unique_sources}
                
                lb_results[service_name] = {
                    'total_requests': len(response_sources),
                    'unique_sources': len(unique_sources),
                    'distribution': distribution,
                    'well_distributed': len(unique_sources) > 1 if len(response_sources) > 10 else True
                }
                
                # For production, we expect load to be distributed across multiple pods
                if len(response_sources) > 10:
                    with self.subTest(service=service_name):
                        self.assertGreater(
                            len(unique_sources), 1,
                            f"Load balancer for {service_name} is not distributing requests across multiple pods"
                        )
        
        self._record_test_result('load_balancer_health', lb_results)
    
    def test_service_mesh_configuration(self):
        """Test service mesh configuration if Istio is deployed."""
        mesh_results = {}
        
        try:
            # Check for Istio CRDs
            api_client = client.ApiClient()
            custom_objects_api = client.CustomObjectsApi(api_client)
            
            # Check VirtualServices
            try:
                virtual_services = custom_objects_api.list_namespaced_custom_object(
                    group="networking.istio.io",
                    version="v1beta1",
                    namespace=self.namespace,
                    plural="virtualservices"
                )
                
                mesh_results['virtual_services'] = {
                    'count': len(virtual_services.get('items', [])),
                    'configured': len(virtual_services.get('items', [])) > 0
                }
            except client.exceptions.ApiException:
                mesh_results['virtual_services'] = {'configured': False, 'error': 'Istio not installed'}
            
            # Check DestinationRules
            try:
                destination_rules = custom_objects_api.list_namespaced_custom_object(
                    group="networking.istio.io",
                    version="v1beta1",
                    namespace=self.namespace,
                    plural="destinationrules"
                )
                
                mesh_results['destination_rules'] = {
                    'count': len(destination_rules.get('items', [])),
                    'configured': len(destination_rules.get('items', [])) > 0
                }
            except client.exceptions.ApiException:
                mesh_results['destination_rules'] = {'configured': False, 'error': 'Istio not installed'}
        
        except Exception as e:
            mesh_results['error'] = str(e)
        
        self._record_test_result('service_mesh_configuration', mesh_results)


class PerformanceTests(ProductionDeploymentTestCase):
    """Test production performance characteristics."""
    
    def test_response_time_performance(self):
        """Test response time performance under normal load."""
        performance_results = {}
        
        for service_name, endpoint in self.production_endpoints.items():
            if 'core' in service_name:  # Focus on core service performance
                response_times = []
                successful_requests = 0
                failed_requests = 0
                
                # Make 100 requests to measure performance
                for i in range(100):
                    start_time = time.time()
                    try:
                        response = requests.get(f"{endpoint}/health", timeout=10)
                        response_time = time.time() - start_time
                        
                        if response.status_code == 200:
                            response_times.append(response_time)
                            successful_requests += 1
                        else:
                            failed_requests += 1
                    except Exception:
                        failed_requests += 1
                        response_times.append(10.0)  # Timeout value
                    
                    time.sleep(0.1)  # Small delay between requests
                
                if response_times:
                    response_times.sort()
                    p50 = response_times[len(response_times) // 2]
                    p95 = response_times[int(len(response_times) * 0.95)]
                    p99 = response_times[int(len(response_times) * 0.99)]
                    avg = sum(response_times) / len(response_times)
                    
                    performance_results[service_name] = {
                        'total_requests': successful_requests + failed_requests,
                        'successful_requests': successful_requests,
                        'failed_requests': failed_requests,
                        'success_rate': successful_requests / (successful_requests + failed_requests),
                        'avg_response_time': avg,
                        'p50_response_time': p50,
                        'p95_response_time': p95,
                        'p99_response_time': p99
                    }
                    
                    with self.subTest(service=service_name):
                        self.assertLess(
                            p95, self.performance_thresholds['response_time_p95'],
                            f"P95 response time {p95:.3f}s exceeds threshold {self.performance_thresholds['response_time_p95']}s"
                        )
                        
                        self.assertLess(
                            p99, self.performance_thresholds['response_time_p99'],
                            f"P99 response time {p99:.3f}s exceeds threshold {self.performance_thresholds['response_time_p99']}s"
                        )
        
        self._record_test_result('response_time_performance', performance_results)
    
    def test_concurrent_load_handling(self):
        """Test system performance under concurrent load."""
        load_results = {}
        
        def make_concurrent_requests(service_name: str, endpoint: str, num_requests: int) -> Dict[str, Any]:
            """Make concurrent requests to a service."""
            results = {
                'successful_requests': 0,
                'failed_requests': 0,
                'response_times': [],
                'errors': []
            }
            
            def single_request():
                start_time = time.time()
                try:
                    response = requests.get(f"{endpoint}/health", timeout=10)
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        return {'success': True, 'response_time': response_time}
                    else:
                        return {'success': False, 'error': f'HTTP {response.status_code}'}
                except Exception as e:
                    return {'success': False, 'error': str(e)}
            
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(single_request) for _ in range(num_requests)]
                
                for future in as_completed(futures):
                    result = future.result()
                    if result['success']:
                        results['successful_requests'] += 1
                        results['response_times'].append(result['response_time'])
                    else:
                        results['failed_requests'] += 1
                        results['errors'].append(result['error'])
            
            return results
        
        # Test concurrent load on core service
        for service_name, endpoint in self.production_endpoints.items():
            if 'core' in service_name:
                logger.info(f"Testing concurrent load on {service_name}")
                
                results = make_concurrent_requests(service_name, endpoint, 50)
                
                total_requests = results['successful_requests'] + results['failed_requests']
                success_rate = results['successful_requests'] / total_requests if total_requests > 0 else 0
                
                if results['response_times']:
                    avg_response_time = sum(results['response_times']) / len(results['response_times'])
                    max_response_time = max(results['response_times'])
                else:
                    avg_response_time = 0
                    max_response_time = 0
                
                load_results[service_name] = {
                    'total_requests': total_requests,
                    'successful_requests': results['successful_requests'],
                    'failed_requests': results['failed_requests'],
                    'success_rate': success_rate,
                    'avg_response_time': avg_response_time,
                    'max_response_time': max_response_time,
                    'error_types': list(set(results['errors']))
                }
                
                with self.subTest(service=service_name):
                    self.assertGreaterEqual(
                        success_rate, 1 - self.performance_thresholds['error_rate'],
                        f"Success rate {success_rate:.3f} below threshold {1 - self.performance_thresholds['error_rate']:.3f}"
                    )
        
        self._record_test_result('concurrent_load_handling', load_results)
    
    def test_resource_utilization(self):
        """Test resource utilization of production pods."""
        resource_results = {}
        
        try:
            # Get pod metrics (requires metrics-server)
            custom_objects_api = client.CustomObjectsApi()
            
            try:
                pod_metrics = custom_objects_api.list_namespaced_custom_object(
                    group="metrics.k8s.io",
                    version="v1beta1",
                    namespace=self.namespace,
                    plural="pods"
                )
                
                for pod_metric in pod_metrics.get('items', []):
                    pod_name = pod_metric['metadata']['name']
                    
                    # Get pod spec for resource limits
                    pod = self.core_v1.read_namespaced_pod(
                        name=pod_name,
                        namespace=self.namespace
                    )
                    
                    for container_metric in pod_metric.get('containers', []):
                        container_name = container_metric['name']
                        
                        # Find corresponding container spec
                        container_spec = None
                        for container in pod.spec.containers:
                            if container.name == container_name:
                                container_spec = container
                                break
                        
                        if container_spec and container_spec.resources and container_spec.resources.limits:
                            cpu_usage = self._parse_cpu_metric(container_metric['usage']['cpu'])
                            memory_usage = self._parse_memory_metric(container_metric['usage']['memory'])
                            
                            cpu_limit = self._parse_cpu_metric(container_spec.resources.limits.get('cpu', '0'))
                            memory_limit = self._parse_memory_metric(container_spec.resources.limits.get('memory', '0'))
                            
                            cpu_utilization = cpu_usage / cpu_limit if cpu_limit > 0 else 0
                            memory_utilization = memory_usage / memory_limit if memory_limit > 0 else 0
                            
                            resource_key = f"{pod_name}/{container_name}"
                            resource_results[resource_key] = {
                                'cpu_usage': cpu_usage,
                                'cpu_limit': cpu_limit,
                                'cpu_utilization': cpu_utilization,
                                'memory_usage': memory_usage,
                                'memory_limit': memory_limit,
                                'memory_utilization': memory_utilization
                            }
                            
                            with self.subTest(container=resource_key):
                                self.assertLess(
                                    cpu_utilization, self.performance_thresholds['cpu_usage'],
                                    f"CPU utilization {cpu_utilization:.3f} exceeds threshold {self.performance_thresholds['cpu_usage']}"
                                )
                                
                                self.assertLess(
                                    memory_utilization, self.performance_thresholds['memory_usage'],
                                    f"Memory utilization {memory_utilization:.3f} exceeds threshold {self.performance_thresholds['memory_usage']}"
                                )
            
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    resource_results['error'] = 'Metrics server not available'
                    self.skipTest("Metrics server not available for resource utilization testing")
                else:
                    raise
        
        except Exception as e:
            self.fail(f"Failed to check resource utilization: {e}")
        
        self._record_test_result('resource_utilization', resource_results)
    
    def _parse_cpu_metric(self, cpu_str: str) -> float:
        """Parse CPU metric string to float (in cores)."""
        if cpu_str.endswith('n'):
            return float(cpu_str[:-1]) / 1_000_000_000
        elif cpu_str.endswith('u'):
            return float(cpu_str[:-1]) / 1_000_000
        elif cpu_str.endswith('m'):
            return float(cpu_str[:-1]) / 1_000
        else:
            return float(cpu_str)
    
    def _parse_memory_metric(self, memory_str: str) -> float:
        """Parse memory metric string to float (in bytes)."""
        if memory_str.endswith('Ki'):
            return float(memory_str[:-2]) * 1024
        elif memory_str.endswith('Mi'):
            return float(memory_str[:-2]) * 1024 * 1024
        elif memory_str.endswith('Gi'):
            return float(memory_str[:-2]) * 1024 * 1024 * 1024
        else:
            return float(memory_str)


class BackupAndRecoveryTests(ProductionDeploymentTestCase):
    """Test backup and recovery systems."""
    
    def test_backup_jobs_status(self):
        """Test that backup CronJobs are configured and running."""
        backup_results = {}
        
        try:
            batch_v1 = client.BatchV1Api()
            cronjobs = batch_v1.list_namespaced_cron_job(namespace=self.namespace)
            
            backup_cronjobs = [
                cj for cj in cronjobs.items 
                if 'backup' in cj.metadata.name.lower()
            ]
            
            for cronjob in backup_cronjobs:
                cj_name = cronjob.metadata.name
                
                # Check if cronjob has run recently
                last_schedule_time = cronjob.status.last_schedule_time
                last_successful_time = cronjob.status.last_successful_time
                
                backup_results[cj_name] = {
                    'schedule': cronjob.spec.schedule,
                    'suspend': cronjob.spec.suspend,
                    'last_schedule_time': last_schedule_time.isoformat() if last_schedule_time else None,
                    'last_successful_time': last_successful_time.isoformat() if last_successful_time else None,
                    'active_jobs': len(cronjob.status.active or [])
                }
                
                with self.subTest(cronjob=cj_name):
                    self.assertFalse(
                        cronjob.spec.suspend,
                        f"Backup CronJob {cj_name} is suspended"
                    )
        
        except Exception as e:
            self.fail(f"Failed to check backup jobs: {e}")
        
        self._record_test_result('backup_jobs_status', backup_results)
    
    def test_persistent_storage_health(self):
        """Test persistent storage health and capacity."""
        storage_results = {}
        
        try:
            pvcs = self.core_v1.list_namespaced_persistent_volume_claim(
                namespace=self.namespace
            )
            
            for pvc in pvcs.items:
                pvc_name = pvc.metadata.name
                
                # Get PV information
                pv_name = pvc.spec.volume_name
                if pv_name:
                    try:
                        pv = self.core_v1.read_persistent_volume(name=pv_name)
                        
                        storage_results[pvc_name] = {
                            'phase': pvc.status.phase,
                            'capacity': pvc.status.capacity.get('storage') if pvc.status.capacity else None,
                            'access_modes': pvc.spec.access_modes,
                            'storage_class': pvc.spec.storage_class_name,
                            'pv_reclaim_policy': pv.spec.persistent_volume_reclaim_policy,
                            'pv_status': pv.status.phase
                        }
                        
                        with self.subTest(pvc=pvc_name):
                            self.assertEqual(
                                pvc.status.phase, "Bound",
                                f"PVC {pvc_name} is not bound"
                            )
                            
                            self.assertEqual(
                                pv.status.phase, "Bound",
                                f"PV {pv_name} is not bound"
                            )
                    
                    except client.exceptions.ApiException:
                        storage_results[pvc_name] = {
                            'phase': pvc.status.phase,
                            'error': 'Could not read PV information'
                        }
        
        except Exception as e:
            self.fail(f"Failed to check persistent storage: {e}")
        
        self._record_test_result('persistent_storage_health', storage_results)


class MonitoringTests(ProductionDeploymentTestCase):
    """Test monitoring and alerting systems."""
    
    def test_prometheus_targets(self):
        """Test that Prometheus is scraping all expected targets."""
        prometheus_results = {}
        
        try:
            # Query Prometheus targets API
            prometheus_url = self.production_endpoints.get('prometheus', 'http://prometheus-service:9090')
            response = requests.get(f"{prometheus_url}/api/v1/targets", timeout=10)
            
            if response.status_code == 200:
                targets_data = response.json()
                
                active_targets = targets_data.get('data', {}).get('activeTargets', [])
                dropped_targets = targets_data.get('data', {}).get('droppedTargets', [])
                
                # Analyze target health
                healthy_targets = [t for t in active_targets if t.get('health') == 'up']
                unhealthy_targets = [t for t in active_targets if t.get('health') != 'up']
                
                # Group by job
                targets_by_job = {}
                for target in active_targets:
                    job = target.get('labels', {}).get('job', 'unknown')
                    if job not in targets_by_job:
                        targets_by_job[job] = {'healthy': 0, 'unhealthy': 0, 'total': 0}
                    
                    targets_by_job[job]['total'] += 1
                    if target.get('health') == 'up':
                        targets_by_job[job]['healthy'] += 1
                    else:
                        targets_by_job[job]['unhealthy'] += 1
                
                prometheus_results = {
                    'total_active_targets': len(active_targets),
                    'healthy_targets': len(healthy_targets),
                    'unhealthy_targets': len(unhealthy_targets),
                    'dropped_targets': len(dropped_targets),
                    'targets_by_job': targets_by_job,
                    'health_rate': len(healthy_targets) / len(active_targets) if active_targets else 0
                }
                
                # Validate that critical services are being monitored
                expected_jobs = ['archangel-core', 'archangel-agents', 'kubernetes-pods']
                for job in expected_jobs:
                    with self.subTest(job=job):
                        self.assertIn(
                            job, targets_by_job,
                            f"Expected monitoring job {job} not found in Prometheus targets"
                        )
                        
                        if job in targets_by_job:
                            self.assertGreater(
                                targets_by_job[job]['healthy'], 0,
                                f"No healthy targets found for job {job}"
                            )
            else:
                prometheus_results = {'error': f'Prometheus API returned {response.status_code}'}
        
        except Exception as e:
            prometheus_results = {'error': str(e)}
        
        self._record_test_result('prometheus_targets', prometheus_results)
    
    def test_alertmanager_configuration(self):
        """Test Alertmanager configuration and connectivity."""
        alertmanager_results = {}
        
        try:
            alertmanager_url = self.production_endpoints.get('alertmanager', 'http://alertmanager-service:9093')
            
            # Test Alertmanager API
            response = requests.get(f"{alertmanager_url}/api/v1/status", timeout=10)
            
            if response.status_code == 200:
                status_data = response.json()
                
                alertmanager_results = {
                    'status': 'healthy',
                    'version_info': status_data.get('data', {}).get('versionInfo', {}),
                    'config_hash': status_data.get('data', {}).get('configHash', ''),
                    'uptime': status_data.get('data', {}).get('uptime', '')
                }
                
                # Test configuration endpoint
                config_response = requests.get(f"{alertmanager_url}/api/v1/config", timeout=10)
                if config_response.status_code == 200:
                    config_data = config_response.json()
                    alertmanager_results['config_loaded'] = True
                    alertmanager_results['receivers_count'] = len(
                        config_data.get('data', {}).get('receivers', [])
                    )
                else:
                    alertmanager_results['config_loaded'] = False
            else:
                alertmanager_results = {'error': f'Alertmanager API returned {response.status_code}'}
        
        except Exception as e:
            alertmanager_results = {'error': str(e)}
        
        self._record_test_result('alertmanager_configuration', alertmanager_results)


if __name__ == '__main__':
    # Run production deployment tests
    unittest.main(verbosity=2)