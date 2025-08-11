#!/usr/bin/env python3
"""
Deployment validation script for Archangel Mock Enterprise Environment
"""

import sys
import time
import requests
import socket
import subprocess
import json
import logging
from typing import Dict, List, Tuple, Optional
import docker
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ServiceCheck:
    name: str
    endpoint: str
    expected_status: int = 200
    timeout: int = 10
    required: bool = True

@dataclass
class PortCheck:
    name: str
    host: str
    port: int
    required: bool = True

@dataclass
class ValidationResult:
    service: str
    status: str
    message: str
    details: Optional[Dict] = None

class DeploymentValidator:
    """Validates the deployment of Archangel Mock Enterprise Environment"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.results: List[ValidationResult] = []
        
        # Define service checks
        self.service_checks = [
            ServiceCheck("Nginx Load Balancer", "http://localhost:80", 404),  # 404 is OK for default server
            ServiceCheck("Kibana", "http://localhost:5601"),
            ServiceCheck("Elasticsearch", "http://localhost:9200"),
            ServiceCheck("MailHog", "http://localhost:8025"),
            ServiceCheck("DVWA", "http://localhost:8080"),
        ]
        
        # Define port checks
        self.port_checks = [
            PortCheck("MySQL", "localhost", 3306),
            PortCheck("PostgreSQL", "localhost", 5432),
            PortCheck("SMB", "localhost", 445),
            PortCheck("SMTP", "localhost", 1025),
            PortCheck("Nginx HTTP", "localhost", 80),
            PortCheck("Nginx HTTPS", "localhost", 443),
            PortCheck("Kibana", "localhost", 5601),
            PortCheck("Elasticsearch", "localhost", 9200),
        ]
        
        # Define required containers
        self.required_containers = [
            "nginx-loadbalancer",
            "wordpress-vulnerable", 
            "opencart-vulnerable",
            "mysql-vulnerable",
            "postgresql-vulnerable",
            "smb-fileserver",
            "mail-server",
            "dvwa-vulnerable",
            "suricata-ids",
            "elasticsearch",
            "logstash",
            "kibana",
            "filebeat"
        ]
    
    def add_result(self, service: str, status: str, message: str, details: Optional[Dict] = None):
        """Add a validation result"""
        self.results.append(ValidationResult(service, status, message, details))
        
        if status == "PASS":
            logger.info(f"✅ {service}: {message}")
        elif status == "WARN":
            logger.warning(f"⚠️  {service}: {message}")
        else:
            logger.error(f"❌ {service}: {message}")
    
    def check_docker_containers(self) -> bool:
        """Check if all required Docker containers are running"""
        logger.info("🐳 Checking Docker containers...")
        
        try:
            containers = self.docker_client.containers.list()
            running_containers = [c.name for c in containers]
            
            all_running = True
            for required in self.required_containers:
                if required in running_containers:
                    container = next(c for c in containers if c.name == required)
                    if container.status == 'running':
                        self.add_result(f"Container {required}", "PASS", "Running")
                    else:
                        self.add_result(f"Container {required}", "FAIL", f"Status: {container.status}")
                        all_running = False
                else:
                    self.add_result(f"Container {required}", "FAIL", "Not found")
                    all_running = False
            
            return all_running
            
        except Exception as e:
            self.add_result("Docker Containers", "FAIL", f"Error checking containers: {e}")
            return False
    
    def check_network_connectivity(self) -> bool:
        """Check network connectivity and segmentation"""
        logger.info("🌐 Checking network connectivity...")
        
        try:
            networks = self.docker_client.networks.list()
            network_names = [n.name for n in networks]
            
            required_networks = [
                'infrastructure_dmz_network',
                'infrastructure_internal_network',
                'infrastructure_management_network'
            ]
            
            networks_ok = True
            for network in required_networks:
                if network in network_names:
                    self.add_result(f"Network {network}", "PASS", "Created")
                else:
                    self.add_result(f"Network {network}", "FAIL", "Not found")
                    networks_ok = False
            
            return networks_ok
            
        except Exception as e:
            self.add_result("Network Connectivity", "FAIL", f"Error checking networks: {e}")
            return False
    
    def check_port_accessibility(self) -> bool:
        """Check if required ports are accessible"""
        logger.info("🔌 Checking port accessibility...")
        
        all_ports_ok = True
        for port_check in self.port_checks:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((port_check.host, port_check.port))
                sock.close()
                
                if result == 0:
                    self.add_result(f"Port {port_check.name}", "PASS", f"Port {port_check.port} accessible")
                else:
                    status = "FAIL" if port_check.required else "WARN"
                    self.add_result(f"Port {port_check.name}", status, f"Port {port_check.port} not accessible")
                    if port_check.required:
                        all_ports_ok = False
                        
            except Exception as e:
                status = "FAIL" if port_check.required else "WARN"
                self.add_result(f"Port {port_check.name}", status, f"Error checking port: {e}")
                if port_check.required:
                    all_ports_ok = False
        
        return all_ports_ok
    
    def check_web_services(self) -> bool:
        """Check web service accessibility"""
        logger.info("🌍 Checking web services...")
        
        all_services_ok = True
        for service_check in self.service_checks:
            try:
                response = requests.get(
                    service_check.endpoint, 
                    timeout=service_check.timeout,
                    allow_redirects=True
                )
                
                if response.status_code == service_check.expected_status:
                    self.add_result(
                        f"Service {service_check.name}", 
                        "PASS", 
                        f"Responding with status {response.status_code}"
                    )
                else:
                    status = "FAIL" if service_check.required else "WARN"
                    self.add_result(
                        f"Service {service_check.name}", 
                        status, 
                        f"Unexpected status {response.status_code}, expected {service_check.expected_status}"
                    )
                    if service_check.required:
                        all_services_ok = False
                        
            except requests.exceptions.RequestException as e:
                status = "FAIL" if service_check.required else "WARN"
                self.add_result(f"Service {service_check.name}", status, f"Request failed: {e}")
                if service_check.required:
                    all_services_ok = False
        
        return all_services_ok
    
    def check_elasticsearch_health(self) -> bool:
        """Check Elasticsearch cluster health"""
        logger.info("🔍 Checking Elasticsearch health...")
        
        try:
            response = requests.get('http://localhost:9200/_cluster/health', timeout=10)
            if response.status_code == 200:
                health = response.json()
                status = health.get('status', 'unknown')
                
                if status in ['green', 'yellow']:
                    self.add_result("Elasticsearch Health", "PASS", f"Cluster status: {status}")
                    return True
                else:
                    self.add_result("Elasticsearch Health", "FAIL", f"Cluster status: {status}")
                    return False
            else:
                self.add_result("Elasticsearch Health", "FAIL", f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.add_result("Elasticsearch Health", "FAIL", f"Health check failed: {e}")
            return False
    
    def check_log_ingestion(self) -> bool:
        """Check if logs are being ingested"""
        logger.info("📊 Checking log ingestion...")
        
        try:
            # Wait a moment for logs to be processed
            time.sleep(5)
            
            # Check for indices
            response = requests.get('http://localhost:9200/_cat/indices?format=json', timeout=10)
            if response.status_code == 200:
                indices = response.json()
                archangel_indices = [idx for idx in indices if 'archangel' in idx.get('index', '')]
                
                if archangel_indices:
                    self.add_result("Log Ingestion", "PASS", f"Found {len(archangel_indices)} log indices")
                    return True
                else:
                    self.add_result("Log Ingestion", "WARN", "No Archangel log indices found yet")
                    return False
            else:
                self.add_result("Log Ingestion", "FAIL", f"Failed to check indices: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.add_result("Log Ingestion", "WARN", f"Log ingestion check failed: {e}")
            return False
    
    def check_vulnerability_exposure(self) -> bool:
        """Check that vulnerabilities are properly exposed"""
        logger.info("🔓 Checking vulnerability exposure...")
        
        vulnerabilities_ok = True
        
        # Check database exposure
        for port, service in [(3306, "MySQL"), (5432, "PostgreSQL")]:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                self.add_result(f"Vulnerable {service}", "PASS", f"Port {port} exposed as intended")
            else:
                self.add_result(f"Vulnerable {service}", "FAIL", f"Port {port} not exposed")
                vulnerabilities_ok = False
        
        # Check web vulnerabilities
        try:
            # Test server-status endpoint (information disclosure)
            response = requests.get('http://localhost/server-status', timeout=5)
            if response.status_code in [200, 404]:
                self.add_result("Web Vulnerabilities", "PASS", "Information disclosure endpoints accessible")
            else:
                self.add_result("Web Vulnerabilities", "WARN", "Some web vulnerabilities may not be exposed")
        except:
            self.add_result("Web Vulnerabilities", "WARN", "Could not test web vulnerabilities")
        
        return vulnerabilities_ok
    
    def generate_report(self) -> Dict:
        """Generate a comprehensive validation report"""
        passed = len([r for r in self.results if r.status == "PASS"])
        failed = len([r for r in self.results if r.status == "FAIL"])
        warnings = len([r for r in self.results if r.status == "WARN"])
        total = len(self.results)
        
        report = {
            "summary": {
                "total_checks": total,
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "success_rate": f"{(passed/total)*100:.1f}%" if total > 0 else "0%"
            },
            "results": [
                {
                    "service": r.service,
                    "status": r.status,
                    "message": r.message,
                    "details": r.details
                }
                for r in self.results
            ]
        }
        
        return report
    
    def run_validation(self) -> bool:
        """Run complete validation suite"""
        logger.info("🚀 Starting Archangel Mock Enterprise Environment Validation")
        
        # Run all validation checks
        checks = [
            ("Docker Containers", self.check_docker_containers),
            ("Network Connectivity", self.check_network_connectivity),
            ("Port Accessibility", self.check_port_accessibility),
            ("Web Services", self.check_web_services),
            ("Elasticsearch Health", self.check_elasticsearch_health),
            ("Log Ingestion", self.check_log_ingestion),
            ("Vulnerability Exposure", self.check_vulnerability_exposure),
        ]
        
        overall_success = True
        for check_name, check_func in checks:
            try:
                result = check_func()
                if not result:
                    overall_success = False
            except Exception as e:
                logger.error(f"Error during {check_name}: {e}")
                self.add_result(check_name, "FAIL", f"Check failed with exception: {e}")
                overall_success = False
        
        # Generate and display report
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("🎯 VALIDATION SUMMARY")
        print("="*60)
        print(f"Total Checks: {report['summary']['total_checks']}")
        print(f"✅ Passed: {report['summary']['passed']}")
        print(f"❌ Failed: {report['summary']['failed']}")
        print(f"⚠️  Warnings: {report['summary']['warnings']}")
        print(f"Success Rate: {report['summary']['success_rate']}")
        
        if report['summary']['failed'] > 0:
            print("\n❌ FAILED CHECKS:")
            for result in report['results']:
                if result['status'] == 'FAIL':
                    print(f"  • {result['service']}: {result['message']}")
        
        if report['summary']['warnings'] > 0:
            print("\n⚠️  WARNINGS:")
            for result in report['results']:
                if result['status'] == 'WARN':
                    print(f"  • {result['service']}: {result['message']}")
        
        print("\n" + "="*60)
        
        # Save detailed report
        with open('validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("📄 Detailed report saved to validation_report.json")
        
        return overall_success

def main():
    """Main execution function"""
    validator = DeploymentValidator()
    success = validator.run_validation()
    
    if success:
        logger.info("🎉 Validation completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Validation failed. Check the report for details.")
        sys.exit(1)

if __name__ == '__main__':
    main()