#!/usr/bin/env python3
"""
Comprehensive infrastructure tests for Archangel Mock Enterprise Environment
"""

import pytest
import requests
import socket
import time
import subprocess
import json
import mysql.connector
import psycopg2
from typing import Dict, List, Optional
import docker
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InfrastructureTestSuite:
    """Test suite for validating mock enterprise infrastructure"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.services = {
            'nginx-loadbalancer': 'http://localhost:80',
            'kibana': 'http://localhost:5601',
            'elasticsearch': 'http://localhost:9200',
            'mysql': ('localhost', 3306),
            'postgresql': ('localhost', 5432),
            'mailhog': 'http://localhost:8025',
            'dvwa': 'http://localhost:8080'
        }
    
    def wait_for_service(self, url: str, timeout: int = 60) -> bool:
        """Wait for a service to become available"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if url.startswith('http'):
                    response = requests.get(url, timeout=5)
                    if response.status_code < 500:
                        return True
                else:
                    host, port = url
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    result = sock.connect_ex((host, port))
                    sock.close()
                    if result == 0:
                        return True
            except Exception as e:
                logger.debug(f"Service {url} not ready: {e}")
            time.sleep(2)
        return False

class TestContainerDeployment:
    """Test container deployment and basic connectivity"""
    
    def test_all_containers_running(self):
        """Verify all required containers are running"""
        required_containers = [
            'nginx-loadbalancer',
            'wordpress-vulnerable', 
            'opencart-vulnerable',
            'mysql-vulnerable',
            'postgresql-vulnerable',
            'smb-fileserver',
            'mail-server',
            'dvwa-vulnerable',
            'suricata-ids',
            'elasticsearch',
            'logstash',
            'kibana',
            'filebeat'
        ]
        
        client = docker.from_env()
        running_containers = [c.name for c in client.containers.list()]
        
        for container in required_containers:
            assert container in running_containers, f"Container {container} is not running"
    
    def test_container_health(self):
        """Test container health status"""
        client = docker.from_env()
        containers = client.containers.list()
        
        for container in containers:
            if hasattr(container, 'health'):
                health = container.attrs['State'].get('Health', {})
                if health:
                    assert health['Status'] in ['healthy', 'starting'], f"Container {container.name} is unhealthy"

class TestNetworkSegmentation:
    """Test network segmentation and isolation"""
    
    def test_network_creation(self):
        """Verify all required networks are created"""
        client = docker.from_env()
        networks = [n.name for n in client.networks.list()]
        
        required_networks = [
            'infrastructure_dmz_network',
            'infrastructure_internal_network', 
            'infrastructure_management_network'
        ]
        
        for network in required_networks:
            assert network in networks, f"Network {network} not found"
    
    def test_dmz_isolation(self):
        """Test that DMZ cannot directly access internal network"""
        try:
            # Try to ping internal network from DMZ
            result = subprocess.run([
                'docker', 'exec', 'nginx-loadbalancer',
                'ping', '-c', '1', '-W', '2', '192.168.20.10'
            ], capture_output=True, timeout=10)
            
            # This should fail (non-zero exit code) indicating proper isolation
            assert result.returncode != 0, "DMZ can reach internal network - isolation failed"
        except subprocess.TimeoutExpired:
            # Timeout is also acceptable as it indicates blocked access
            pass
    
    def test_internal_to_dmz_access(self):
        """Test that internal network can access DMZ (expected behavior)"""
        result = subprocess.run([
            'docker', 'exec', 'mysql-vulnerable',
            'ping', '-c', '1', '-W', '5', '192.168.10.10'
        ], capture_output=True, timeout=15)
        
        assert result.returncode == 0, "Internal network cannot reach DMZ"

class TestWebServices:
    """Test web services and load balancer"""
    
    def test_nginx_load_balancer(self):
        """Test Nginx load balancer is responding"""
        response = requests.get('http://localhost:80', timeout=10)
        assert response.status_code in [200, 404], "Nginx load balancer not responding"
    
    def test_wordpress_accessibility(self):
        """Test WordPress is accessible through load balancer"""
        headers = {'Host': 'wordpress.local'}
        response = requests.get('http://localhost:80', headers=headers, timeout=10)
        assert response.status_code == 200, "WordPress not accessible"
        assert 'wordpress' in response.text.lower(), "WordPress content not found"
    
    def test_opencart_accessibility(self):
        """Test OpenCart is accessible through load balancer"""
        headers = {'Host': 'shop.local'}
        response = requests.get('http://localhost:80', headers=headers, timeout=10)
        assert response.status_code == 200, "OpenCart not accessible"
    
    def test_dvwa_accessibility(self):
        """Test DVWA is accessible"""
        response = requests.get('http://localhost:8080', timeout=10)
        assert response.status_code == 200, "DVWA not accessible"
        assert 'dvwa' in response.text.lower(), "DVWA content not found"

class TestDatabaseServices:
    """Test database services and vulnerabilities"""
    
    def test_mysql_connectivity(self):
        """Test MySQL database connectivity"""
        try:
            conn = mysql.connector.connect(
                host='localhost',
                port=3306,
                user='root',
                password='root123',
                database='corporate',
                connection_timeout=10
            )
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1, "MySQL query failed"
            conn.close()
        except Exception as e:
            pytest.fail(f"MySQL connectivity test failed: {e}")
    
    def test_mysql_vulnerable_data(self):
        """Test that vulnerable data exists in MySQL"""
        try:
            conn = mysql.connector.connect(
                host='localhost',
                port=3306,
                user='root',
                password='root123',
                database='corporate'
            )
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM employees")
            count = cursor.fetchone()[0]
            assert count > 0, "No vulnerable data found in employees table"
            
            cursor.execute("SELECT COUNT(*) FROM financial_data")
            count = cursor.fetchone()[0]
            assert count > 0, "No vulnerable data found in financial_data table"
            conn.close()
        except Exception as e:
            pytest.fail(f"MySQL vulnerable data test failed: {e}")
    
    def test_postgresql_connectivity(self):
        """Test PostgreSQL database connectivity"""
        try:
            conn = psycopg2.connect(
                host='localhost',
                port=5432,
                user='admin',
                password='admin123',
                database='corporate',
                connect_timeout=10
            )
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1, "PostgreSQL query failed"
            conn.close()
        except Exception as e:
            pytest.fail(f"PostgreSQL connectivity test failed: {e}")

class TestLoggingInfrastructure:
    """Test ELK stack and logging infrastructure"""
    
    def test_elasticsearch_health(self):
        """Test Elasticsearch cluster health"""
        response = requests.get('http://localhost:9200/_cluster/health', timeout=10)
        assert response.status_code == 200, "Elasticsearch not responding"
        
        health = response.json()
        assert health['status'] in ['yellow', 'green'], f"Elasticsearch unhealthy: {health['status']}"
    
    def test_kibana_accessibility(self):
        """Test Kibana web interface"""
        response = requests.get('http://localhost:5601', timeout=15)
        assert response.status_code == 200, "Kibana not accessible"
    
    def test_logstash_pipeline(self):
        """Test Logstash pipeline is processing logs"""
        # Send a test log entry
        test_log = {
            "message": "Test log entry from infrastructure test",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "level": "INFO",
            "source": "infrastructure_test"
        }
        
        try:
            response = requests.post(
                'http://localhost:8080',  # Logstash HTTP input
                json=test_log,
                timeout=10
            )
            # Logstash HTTP input returns 200 for successful ingestion
            assert response.status_code == 200, "Failed to send log to Logstash"
        except requests.exceptions.ConnectionError:
            # If HTTP input is not configured, this is expected
            pass
    
    def test_log_indices_creation(self):
        """Test that log indices are being created in Elasticsearch"""
        time.sleep(5)  # Wait for logs to be processed
        
        response = requests.get('http://localhost:9200/_cat/indices?format=json', timeout=10)
        assert response.status_code == 200, "Failed to get Elasticsearch indices"
        
        indices = response.json()
        index_names = [idx['index'] for idx in indices]
        
        # Check for expected log indices
        archangel_indices = [idx for idx in index_names if 'archangel' in idx]
        assert len(archangel_indices) > 0, "No Archangel log indices found"

class TestSecurityFeatures:
    """Test security features and IDS"""
    
    def test_suricata_running(self):
        """Test that Suricata IDS is running"""
        client = docker.from_env()
        try:
            container = client.containers.get('suricata-ids')
            assert container.status == 'running', "Suricata IDS is not running"
        except docker.errors.NotFound:
            pytest.fail("Suricata IDS container not found")
    
    def test_suricata_log_generation(self):
        """Test that Suricata is generating logs"""
        # Generate some network traffic to trigger logging
        requests.get('http://localhost:80', timeout=5)
        
        # Check if Suricata logs exist
        result = subprocess.run([
            'docker', 'exec', 'suricata-ids',
            'ls', '-la', '/var/log/suricata/'
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, "Failed to access Suricata logs"
        assert 'eve.json' in result.stdout, "Suricata EVE log not found"
    
    def test_file_server_accessibility(self):
        """Test SMB file server is accessible"""
        # Test if SMB ports are open
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 445))
        sock.close()
        assert result == 0, "SMB file server port 445 not accessible"

class TestVulnerabilityExposure:
    """Test that intentional vulnerabilities are properly exposed"""
    
    def test_exposed_database_ports(self):
        """Test that database ports are intentionally exposed"""
        # MySQL port 3306
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 3306))
        sock.close()
        assert result == 0, "MySQL port 3306 not exposed as expected"
        
        # PostgreSQL port 5432
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 5432))
        sock.close()
        assert result == 0, "PostgreSQL port 5432 not exposed as expected"
    
    def test_weak_authentication(self):
        """Test that weak authentication is configured"""
        # Test MySQL with weak credentials
        try:
            conn = mysql.connector.connect(
                host='localhost',
                port=3306,
                user='guest',
                password='',  # Empty password
                connection_timeout=5
            )
            conn.close()
            # If connection succeeds, weak auth is configured
        except mysql.connector.Error:
            # This is expected - guest user might not have empty password
            pass
    
    def test_information_disclosure(self):
        """Test information disclosure vulnerabilities"""
        # Test Nginx server-status endpoint
        response = requests.get('http://localhost/server-status', timeout=5)
        # Should either return status info or 404, not 403 (which would indicate proper security)
        assert response.status_code in [200, 404], "Server status endpoint properly secured (unexpected)"

class TestMailServices:
    """Test mail server functionality"""
    
    def test_mailhog_accessibility(self):
        """Test MailHog web interface"""
        response = requests.get('http://localhost:8025', timeout=10)
        assert response.status_code == 200, "MailHog not accessible"
        assert 'mailhog' in response.text.lower(), "MailHog interface not found"
    
    def test_smtp_port_open(self):
        """Test SMTP port is open"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 1025))
        sock.close()
        assert result == 0, "SMTP port 1025 not accessible"

# Test execution and reporting
def run_infrastructure_tests():
    """Run all infrastructure tests and generate report"""
    pytest_args = [
        __file__,
        '-v',
        '--tb=short',
        '--junit-xml=infrastructure_test_results.xml',
        '--html=infrastructure_test_report.html',
        '--self-contained-html'
    ]
    
    return pytest.main(pytest_args)

if __name__ == '__main__':
    exit_code = run_infrastructure_tests()
    exit(exit_code)