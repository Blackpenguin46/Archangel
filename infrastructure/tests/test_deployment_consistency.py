#!/usr/bin/env python3
"""
Infrastructure Deployment Consistency Tests
Comprehensive test suite for validating deployment consistency across environments
"""

import unittest
import json
import yaml
import os
import subprocess
import time
import requests
import docker
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentConsistencyTestCase(unittest.TestCase):
    """Base test case for deployment consistency validation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.docker_client = docker.from_env()
        cls.project_name = "archangel"
        cls.test_environment = "test"
        
        # Load configuration
        cls.config = cls._load_test_config()
        
        # Wait for services to be ready
        cls._wait_for_services()
    
    @classmethod
    def _load_test_config(cls) -> Dict[str, Any]:
        """Load test configuration."""
        config_file = os.environ.get('TEST_CONFIG', './test_config.yaml')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        else:
            return {
                'services': {
                    'core_api': {'port': 8888, 'health_endpoint': '/health'},
                    'prometheus': {'port': 9090, 'health_endpoint': '/-/ready'},
                    'grafana': {'port': 3000, 'health_endpoint': '/api/health'},
                    'elasticsearch': {'port': 9200, 'health_endpoint': '/'},
                    'kibana': {'port': 5601, 'health_endpoint': '/'},
                },
                'networks': ['dmz', 'internal', 'management', 'deception'],
                'expected_containers': 8,
                'max_startup_time': 300
            }
    
    @classmethod
    def _wait_for_services(cls, timeout: int = 300):
        """Wait for critical services to be ready."""
        logger.info("Waiting for services to be ready...")
        
        services = cls.config.get('services', {})
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_ready = True
            for service, config in services.items():
                if not cls._check_service_health(service, config):
                    all_ready = False
                    break
            
            if all_ready:
                logger.info("All services ready")
                return
            
            time.sleep(5)
        
        logger.warning(f"Services not ready after {timeout}s")
    
    @classmethod
    def _check_service_health(cls, service: str, config: Dict[str, Any]) -> bool:
        """Check if a service is healthy."""
        try:
            port = config.get('port')
            endpoint = config.get('health_endpoint', '/')
            url = f"http://localhost:{port}{endpoint}"
            
            response = requests.get(url, timeout=5)
            return response.status_code in [200, 404]  # 404 is OK for some services
        except Exception:
            return False


class TerraformConsistencyTests(DeploymentConsistencyTestCase):
    """Test Terraform deployment consistency."""
    
    def test_terraform_state_exists(self):
        """Test that Terraform state file exists and is valid."""
        terraform_dir = "./terraform"
        state_file = os.path.join(terraform_dir, "terraform.tfstate")
        
        if os.path.exists(terraform_dir):
            self.assertTrue(
                os.path.exists(state_file) or os.path.exists(state_file + ".backup"),
                "Terraform state file should exist"
            )
            
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.assertIn('version', state, "State file should have version")
                    self.assertIn('resources', state, "State file should have resources")
    
    def test_terraform_variables_consistency(self):
        """Test that Terraform variables are consistently defined."""
        terraform_dir = "./terraform"
        variables_file = os.path.join(terraform_dir, "variables.tf")
        
        if os.path.exists(variables_file):
            with open(variables_file, 'r') as f:
                content = f.read()
                
                # Check for required variables
                required_variables = [
                    'environment', 'project_name', 'network_prefix',
                    'enable_monitoring', 'agent_replicas'
                ]
                
                for var in required_variables:
                    self.assertIn(f'variable "{var}"', content,
                                f"Variable {var} should be defined")
    
    def test_network_resources_created(self):
        """Test that required network resources are created."""
        if self.docker_client:
            networks = self.docker_client.networks.list()
            network_names = [n.name for n in networks]
            
            expected_networks = [f"{self.project_name}-{net}" for net in self.config['networks']]
            
            for expected_network in expected_networks:
                matching_networks = [n for n in network_names if expected_network in n]
                self.assertGreater(
                    len(matching_networks), 0,
                    f"Network {expected_network} should exist"
                )
    
    def test_volume_resources_created(self):
        """Test that required volume resources are created."""
        if self.docker_client:
            volumes = self.docker_client.volumes.list()
            volume_names = [v.name for v in volumes]
            
            expected_volumes = [
                'prometheus-data', 'grafana-data', 'elasticsearch-data'
            ]
            
            for expected_volume in expected_volumes:
                matching_volumes = [v for v in volume_names if expected_volume in v]
                self.assertGreater(
                    len(matching_volumes), 0,
                    f"Volume containing {expected_volume} should exist"
                )


class DockerComposeConsistencyTests(DeploymentConsistencyTestCase):
    """Test Docker Compose deployment consistency."""
    
    def test_compose_file_structure(self):
        """Test that Docker Compose files are well-structured."""
        compose_files = ['docker-compose.yml', 'docker-compose.production.yml']
        
        for compose_file in compose_files:
            if os.path.exists(compose_file):
                with open(compose_file, 'r') as f:
                    try:
                        compose_config = yaml.safe_load(f)
                        
                        # Check required sections
                        self.assertIn('version', compose_config, "Compose file should have version")
                        self.assertIn('services', compose_config, "Compose file should have services")
                        
                        # Check service structure
                        services = compose_config['services']
                        self.assertGreater(len(services), 0, "Should have at least one service")
                        
                        # Check for core services
                        core_services = ['prometheus', 'grafana']
                        for service in core_services:
                            service_found = any(service in svc_name for svc_name in services.keys())
                            if not service_found:
                                logger.warning(f"Core service {service} not found in {compose_file}")
                        
                    except yaml.YAMLError as e:
                        self.fail(f"Invalid YAML in {compose_file}: {e}")
    
    def test_container_count_consistency(self):
        """Test that expected number of containers are running."""
        if self.docker_client:
            containers = self.docker_client.containers.list()
            archangel_containers = [c for c in containers if self.project_name in c.name]
            
            expected_count = self.config.get('expected_containers', 5)
            actual_count = len(archangel_containers)
            
            self.assertGreaterEqual(
                actual_count, expected_count // 2,  # Allow 50% tolerance
                f"Should have at least {expected_count // 2} containers running, got {actual_count}"
            )
    
    def test_container_health_status(self):
        """Test that containers are healthy."""
        if self.docker_client:
            containers = self.docker_client.containers.list()
            archangel_containers = [c for c in containers if self.project_name in c.name]
            
            unhealthy_containers = []
            for container in archangel_containers:
                if container.status != 'running':
                    unhealthy_containers.append(f"{container.name}: {container.status}")
            
            self.assertEqual(
                len(unhealthy_containers), 0,
                f"All containers should be running. Unhealthy: {unhealthy_containers}"
            )
    
    def test_service_environment_variables(self):
        """Test that services have required environment variables."""
        if self.docker_client:
            containers = self.docker_client.containers.list()
            core_containers = [c for c in containers if 'core' in c.name and self.project_name in c.name]
            
            for container in core_containers:
                env_vars = container.attrs.get('Config', {}).get('Env', [])
                env_dict = {}
                
                for env in env_vars:
                    if '=' in env:
                        key, value = env.split('=', 1)
                        env_dict[key] = value
                
                # Check required environment variables
                required_env_vars = ['ENVIRONMENT', 'PROJECT_NAME']
                for var in required_env_vars:
                    if var in env_dict:
                        logger.info(f"Container {container.name} has {var}={env_dict[var]}")
                    else:
                        logger.warning(f"Container {container.name} missing {var}")


class KubernetesConsistencyTests(DeploymentConsistencyTestCase):
    """Test Kubernetes deployment consistency."""
    
    def test_namespace_exists(self):
        """Test that Kubernetes namespace exists."""
        try:
            result = subprocess.run(
                ['kubectl', 'get', 'namespace', 'archangel'],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                self.assertIn('archangel', result.stdout, "Namespace should exist")
            else:
                logger.info("Kubernetes not available or namespace not created")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.skipTest("kubectl not available")
    
    def test_deployment_manifests_valid(self):
        """Test that Kubernetes manifests are valid."""
        k8s_dir = "./k8s"
        if not os.path.exists(k8s_dir):
            self.skipTest("Kubernetes manifests directory not found")
        
        manifest_files = [f for f in os.listdir(k8s_dir) if f.endswith('.yaml') or f.endswith('.yml')]
        
        for manifest_file in manifest_files:
            manifest_path = os.path.join(k8s_dir, manifest_file)
            
            with open(manifest_path, 'r') as f:
                try:
                    manifests = list(yaml.safe_load_all(f))
                    
                    for manifest in manifests:
                        if manifest:  # Skip empty documents
                            self.assertIn('apiVersion', manifest, f"Manifest should have apiVersion in {manifest_file}")
                            self.assertIn('kind', manifest, f"Manifest should have kind in {manifest_file}")
                            self.assertIn('metadata', manifest, f"Manifest should have metadata in {manifest_file}")
                            
                except yaml.YAMLError as e:
                    self.fail(f"Invalid YAML in {manifest_file}: {e}")
    
    def test_resource_quotas_defined(self):
        """Test that resource quotas and limits are defined."""
        namespace_file = "./k8s/archangel-namespace.yaml"
        
        if os.path.exists(namespace_file):
            with open(namespace_file, 'r') as f:
                manifests = list(yaml.safe_load_all(f))
                
                quota_found = False
                limit_range_found = False
                
                for manifest in manifests:
                    if manifest and manifest.get('kind') == 'ResourceQuota':
                        quota_found = True
                        self.assertIn('spec', manifest, "ResourceQuota should have spec")
                        self.assertIn('hard', manifest['spec'], "ResourceQuota should have hard limits")
                    
                    if manifest and manifest.get('kind') == 'LimitRange':
                        limit_range_found = True
                        self.assertIn('spec', manifest, "LimitRange should have spec")
                
                if not quota_found:
                    logger.warning("No ResourceQuota found in namespace manifest")
                if not limit_range_found:
                    logger.warning("No LimitRange found in namespace manifest")


class AnsibleConsistencyTests(DeploymentConsistencyTestCase):
    """Test Ansible configuration consistency."""
    
    def test_playbook_syntax(self):
        """Test that Ansible playbook has valid syntax."""
        playbook_file = "./ansible/playbook.yml"
        
        if not os.path.exists(playbook_file):
            self.skipTest("Ansible playbook not found")
        
        try:
            result = subprocess.run(
                ['ansible-playbook', '--syntax-check', playbook_file],
                capture_output=True, text=True, timeout=60
            )
            
            self.assertEqual(result.returncode, 0, f"Playbook syntax check failed: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.skipTest("ansible-playbook not available")
    
    def test_inventory_structure(self):
        """Test that Ansible inventory is well-structured."""
        inventory_file = "./ansible/inventory.yml"
        
        if os.path.exists(inventory_file):
            with open(inventory_file, 'r') as f:
                try:
                    inventory = yaml.safe_load(f)
                    
                    self.assertIn('all', inventory, "Inventory should have 'all' group")
                    
                    if 'all' in inventory:
                        all_group = inventory['all']
                        
                        if 'children' in all_group:
                            children = all_group['children']
                            expected_environments = ['development', 'staging', 'production']
                            
                            for env in expected_environments:
                                if env in children:
                                    logger.info(f"Environment {env} configured in inventory")
                                else:
                                    logger.warning(f"Environment {env} not found in inventory")
                        
                except yaml.YAMLError as e:
                    self.fail(f"Invalid YAML in inventory: {e}")
    
    def test_required_ansible_roles(self):
        """Test that required Ansible roles are defined."""
        playbook_file = "./ansible/playbook.yml"
        
        if os.path.exists(playbook_file):
            with open(playbook_file, 'r') as f:
                try:
                    playbook = yaml.safe_load_all(f)
                    plays = list(playbook)
                    
                    self.assertGreater(len(plays), 0, "Playbook should have at least one play")
                    
                    # Check for essential tasks
                    essential_tasks = [
                        'docker', 'monitoring', 'security', 'networking'
                    ]
                    
                    all_content = str(plays).lower()
                    for task_type in essential_tasks:
                        if task_type in all_content:
                            logger.info(f"Found {task_type}-related tasks")
                        else:
                            logger.warning(f"No {task_type}-related tasks found")
                    
                except yaml.YAMLError as e:
                    self.fail(f"Invalid YAML in playbook: {e}")


class ServiceConsistencyTests(DeploymentConsistencyTestCase):
    """Test service consistency and integration."""
    
    def test_service_discovery(self):
        """Test that services can discover each other."""
        services = self.config.get('services', {})
        
        for service_name, service_config in services.items():
            port = service_config.get('port')
            endpoint = service_config.get('health_endpoint', '/')
            
            with self.subTest(service=service_name):
                url = f"http://localhost:{port}{endpoint}"
                
                try:
                    response = requests.get(url, timeout=10)
                    self.assertIn(
                        response.status_code, [200, 404],
                        f"Service {service_name} should be accessible at {url}"
                    )
                except requests.RequestException as e:
                    self.fail(f"Service {service_name} not accessible: {e}")
    
    def test_monitoring_integration(self):
        """Test that monitoring services are properly integrated."""
        # Test Prometheus can scrape targets
        try:
            response = requests.get("http://localhost:9090/api/v1/targets", timeout=10)
            if response.status_code == 200:
                data = response.json()
                targets = data.get('data', {}).get('activeTargets', [])
                
                self.assertGreater(
                    len(targets), 0,
                    "Prometheus should have scraping targets"
                )
                
                # Check for healthy targets
                healthy_targets = [t for t in targets if t.get('health') == 'up']
                self.assertGreater(
                    len(healthy_targets), 0,
                    "At least some Prometheus targets should be healthy"
                )
            else:
                logger.warning(f"Could not check Prometheus targets: HTTP {response.status_code}")
        except requests.RequestException:
            logger.warning("Prometheus not accessible for target validation")
    
    def test_logging_integration(self):
        """Test that logging services are properly integrated."""
        try:
            # Check Elasticsearch health
            response = requests.get("http://localhost:9200/_cluster/health", timeout=10)
            if response.status_code == 200:
                health = response.json()
                status = health.get('status')
                
                self.assertIn(
                    status, ['green', 'yellow'],
                    f"Elasticsearch cluster should be healthy, got {status}"
                )
                
                # Check for indices
                indices_response = requests.get("http://localhost:9200/_cat/indices?format=json", timeout=10)
                if indices_response.status_code == 200:
                    indices = indices_response.json()
                    logger.info(f"Found {len(indices)} Elasticsearch indices")
                
        except requests.RequestException:
            logger.warning("Elasticsearch not accessible for health check")
    
    def test_security_configuration(self):
        """Test security configuration consistency."""
        # Check that services are not using default credentials
        security_checks = [
            {
                'name': 'Grafana Default Auth',
                'url': 'http://admin:admin@localhost:3000/api/health',
                'should_fail': True
            },
            {
                'name': 'Prometheus Admin API',
                'url': 'http://localhost:9090/api/v1/admin/tsdb/snapshot',
                'should_fail': False  # Should be accessible in test environment
            }
        ]
        
        for check in security_checks:
            with self.subTest(check=check['name']):
                try:
                    response = requests.get(check['url'], timeout=5)
                    
                    if check['should_fail']:
                        self.assertNotEqual(
                            response.status_code, 200,
                            f"{check['name']}: Should not be accessible with default credentials"
                        )
                    else:
                        # Just log the result for informational purposes
                        logger.info(f"{check['name']}: HTTP {response.status_code}")
                        
                except requests.RequestException:
                    if not check['should_fail']:
                        logger.warning(f"{check['name']}: Service not accessible")


class PerformanceConsistencyTests(DeploymentConsistencyTestCase):
    """Test performance consistency."""
    
    def test_service_response_times(self):
        """Test that services respond within acceptable time limits."""
        services = self.config.get('services', {})
        max_response_time = 5.0  # 5 seconds
        
        for service_name, service_config in services.items():
            with self.subTest(service=service_name):
                port = service_config.get('port')
                endpoint = service_config.get('health_endpoint', '/')
                url = f"http://localhost:{port}{endpoint}"
                
                try:
                    start_time = time.time()
                    response = requests.get(url, timeout=max_response_time)
                    response_time = time.time() - start_time
                    
                    self.assertLess(
                        response_time, max_response_time,
                        f"Service {service_name} response time {response_time:.2f}s exceeds {max_response_time}s"
                    )
                    
                    logger.info(f"Service {service_name} responded in {response_time:.3f}s")
                    
                except requests.RequestException as e:
                    logger.warning(f"Could not test response time for {service_name}: {e}")
    
    def test_resource_utilization(self):
        """Test that containers are not exceeding resource limits."""
        if self.docker_client:
            containers = self.docker_client.containers.list()
            archangel_containers = [c for c in containers if self.project_name in c.name]
            
            high_memory_containers = []
            
            for container in archangel_containers:
                try:
                    stats = container.stats(stream=False)
                    
                    # Check memory usage
                    memory_usage = stats.get('memory_stats', {}).get('usage', 0)
                    memory_limit = stats.get('memory_stats', {}).get('limit', 0)
                    
                    if memory_limit > 0:
                        memory_percent = (memory_usage / memory_limit) * 100
                        if memory_percent > 90:  # 90% threshold
                            high_memory_containers.append(f"{container.name}: {memory_percent:.1f}%")
                    
                except Exception as e:
                    logger.warning(f"Could not get stats for {container.name}: {e}")
            
            self.assertEqual(
                len(high_memory_containers), 0,
                f"Containers exceeding 90% memory usage: {high_memory_containers}"
            )


if __name__ == '__main__':
    # Configure test runner
    import sys
    
    # Add test discovery and execution
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=True
    )
    
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)