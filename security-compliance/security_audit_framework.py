#!/usr/bin/env python3
"""
Archangel Security Audit Framework
Comprehensive security assessment and penetration testing for the autonomous AI system
"""

import asyncio
import json
import logging
import subprocess
import socket
import ssl
import docker
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
import psutil
import nmap
from cryptography import x509
from cryptography.hazmat.backends import default_backend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SecurityFinding:
    """Represents a security finding from the audit"""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    category: str  # CONTAINER, NETWORK, ENCRYPTION, AUTHENTICATION, BOUNDARY
    title: str
    description: str
    affected_component: str
    remediation: str
    cve_references: List[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.cve_references is None:
            self.cve_references = []

@dataclass
class AuditResult:
    """Complete audit results"""
    audit_id: str
    timestamp: datetime
    findings: List[SecurityFinding]
    summary: Dict[str, int]
    recommendations: List[str]
    compliance_status: Dict[str, bool]

class ContainerSecurityAuditor:
    """Audits Docker container security and isolation"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.findings = []
    
    async def audit_container_isolation(self) -> List[SecurityFinding]:
        """Test container isolation and escape prevention"""
        findings = []
        
        try:
            # Check for privileged containers
            containers = self.docker_client.containers.list(all=True)
            for container in containers:
                if container.attrs.get('HostConfig', {}).get('Privileged', False):
                    findings.append(SecurityFinding(
                        severity="HIGH",
                        category="CONTAINER",
                        title="Privileged Container Detected",
                        description=f"Container {container.name} is running in privileged mode",
                        affected_component=container.name,
                        remediation="Remove privileged flag and use specific capabilities instead"
                    ))
            
            # Check for host network mode
            for container in containers:
                network_mode = container.attrs.get('HostConfig', {}).get('NetworkMode', '')
                if network_mode == 'host':
                    findings.append(SecurityFinding(
                        severity="MEDIUM",
                        category="CONTAINER",
                        title="Host Network Mode Detected",
                        description=f"Container {container.name} uses host networking",
                        affected_component=container.name,
                        remediation="Use bridge networking with port mapping instead"
                    ))
            
            # Check for volume mounts
            for container in containers:
                mounts = container.attrs.get('Mounts', [])
                for mount in mounts:
                    if mount.get('Type') == 'bind' and mount.get('Source', '').startswith('/'):
                        if mount.get('Source') in ['/var/run/docker.sock', '/proc', '/sys']:
                            findings.append(SecurityFinding(
                                severity="CRITICAL",
                                category="CONTAINER",
                                title="Dangerous Host Mount Detected",
                                description=f"Container {container.name} mounts {mount.get('Source')}",
                                affected_component=container.name,
                                remediation="Remove dangerous host mounts or use read-only mounts"
                            ))
            
            # Test container escape attempts
            escape_findings = await self._test_container_escape()
            findings.extend(escape_findings)
            
        except Exception as e:
            logger.error(f"Container isolation audit failed: {e}")
            findings.append(SecurityFinding(
                severity="HIGH",
                category="CONTAINER",
                title="Container Audit Failed",
                description=f"Unable to complete container security audit: {e}",
                affected_component="Docker Engine",
                remediation="Investigate Docker daemon accessibility and permissions"
            ))
        
        return findings
    
    async def _test_container_escape(self) -> List[SecurityFinding]:
        """Test for container escape vulnerabilities"""
        findings = []
        
        # Test common escape techniques
        escape_tests = [
            {
                'name': 'Docker Socket Access',
                'command': 'ls -la /var/run/docker.sock',
                'severity': 'CRITICAL'
            },
            {
                'name': 'Proc Filesystem Access',
                'command': 'ls -la /proc/1/root',
                'severity': 'HIGH'
            },
            {
                'name': 'Sys Filesystem Access',
                'command': 'ls -la /sys/fs/cgroup',
                'severity': 'MEDIUM'
            }
        ]
        
        for test in escape_tests:
            try:
                # Run test in a temporary container
                result = self.docker_client.containers.run(
                    'alpine:latest',
                    command=f"sh -c '{test['command']}'",
                    remove=True,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    findings.append(SecurityFinding(
                        severity=test['severity'],
                        category="CONTAINER",
                        title=f"Container Escape Vector: {test['name']}",
                        description=f"Container can access host resources via {test['name']}",
                        affected_component="Container Runtime",
                        remediation="Implement proper container isolation and security policies"
                    ))
                    
            except Exception as e:
                logger.debug(f"Escape test {test['name']} failed (expected): {e}")
        
        return findings

class NetworkSecurityAuditor:
    """Audits network segmentation and security"""
    
    def __init__(self):
        self.nm = nmap.PortScanner()
        self.findings = []
    
    async def audit_network_segmentation(self) -> List[SecurityFinding]:
        """Test network segmentation effectiveness"""
        findings = []
        
        try:
            # Get Docker networks
            client = docker.from_env()
            networks = client.networks.list()
            
            for network in networks:
                network_info = network.attrs
                
                # Check for bridge networks without isolation
                if network_info.get('Driver') == 'bridge':
                    options = network_info.get('Options', {})
                    if options.get('com.docker.network.bridge.enable_icc', 'true') == 'true':
                        findings.append(SecurityFinding(
                            severity="MEDIUM",
                            category="NETWORK",
                            title="Inter-Container Communication Enabled",
                            description=f"Network {network.name} allows unrestricted container communication",
                            affected_component=network.name,
                            remediation="Disable ICC or implement network policies"
                        ))
                
                # Check for networks without custom subnets
                ipam = network_info.get('IPAM', {})
                if not ipam.get('Config'):
                    findings.append(SecurityFinding(
                        severity="LOW",
                        category="NETWORK",
                        title="Default Network Configuration",
                        description=f"Network {network.name} uses default IP configuration",
                        affected_component=network.name,
                        remediation="Configure custom subnets for better network control"
                    ))
            
            # Test cross-network communication
            cross_network_findings = await self._test_cross_network_access()
            findings.extend(cross_network_findings)
            
        except Exception as e:
            logger.error(f"Network segmentation audit failed: {e}")
            findings.append(SecurityFinding(
                severity="HIGH",
                category="NETWORK",
                title="Network Audit Failed",
                description=f"Unable to complete network security audit: {e}",
                affected_component="Network Infrastructure",
                remediation="Investigate network configuration and accessibility"
            ))
        
        return findings
    
    async def _test_cross_network_access(self) -> List[SecurityFinding]:
        """Test if containers can access networks they shouldn't"""
        findings = []
        
        try:
            client = docker.from_env()
            
            # Create test containers in different networks
            test_network1 = client.networks.create("audit_test_net1", driver="bridge")
            test_network2 = client.networks.create("audit_test_net2", driver="bridge")
            
            try:
                container1 = client.containers.run(
                    'alpine:latest',
                    command='sleep 30',
                    network=test_network1.name,
                    detach=True,
                    remove=True
                )
                
                container2 = client.containers.run(
                    'alpine:latest',
                    command='sleep 30',
                    network=test_network2.name,
                    detach=True,
                    remove=True
                )
                
                # Get container IPs
                container1.reload()
                container2.reload()
                
                net1_ip = container1.attrs['NetworkSettings']['Networks'][test_network1.name]['IPAddress']
                net2_ip = container2.attrs['NetworkSettings']['Networks'][test_network2.name]['IPAddress']
                
                # Test connectivity between networks
                result = container1.exec_run(f'ping -c 1 -W 1 {net2_ip}')
                if result.exit_code == 0:
                    findings.append(SecurityFinding(
                        severity="HIGH",
                        category="NETWORK",
                        title="Cross-Network Communication Possible",
                        description="Containers in different networks can communicate",
                        affected_component="Network Isolation",
                        remediation="Implement proper network segmentation and firewall rules"
                    ))
                
                # Clean up
                container1.stop()
                container2.stop()
                
            finally:
                test_network1.remove()
                test_network2.remove()
                
        except Exception as e:
            logger.debug(f"Cross-network test failed (may be expected): {e}")
        
        return findings

class EncryptionAuditor:
    """Audits encryption and authentication mechanisms"""
    
    def __init__(self):
        self.findings = []
    
    async def audit_encryption_mechanisms(self) -> List[SecurityFinding]:
        """Validate encryption implementations"""
        findings = []
        
        # Check TLS configurations
        tls_findings = await self._audit_tls_configuration()
        findings.extend(tls_findings)
        
        # Check certificate validity
        cert_findings = await self._audit_certificates()
        findings.extend(cert_findings)
        
        # Check encryption at rest
        storage_findings = await self._audit_storage_encryption()
        findings.extend(storage_findings)
        
        return findings
    
    async def _audit_tls_configuration(self) -> List[SecurityFinding]:
        """Audit TLS configuration and cipher suites"""
        findings = []
        
        # Common ports to check
        ports_to_check = [443, 8443, 9443, 6379, 5432, 3306]
        
        for port in ports_to_check:
            try:
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                
                with socket.create_connection(('localhost', port), timeout=5) as sock:
                    with context.wrap_socket(sock) as ssock:
                        cipher = ssock.cipher()
                        version = ssock.version()
                        
                        # Check for weak ciphers
                        if cipher and any(weak in cipher[0] for weak in ['RC4', 'DES', 'MD5']):
                            findings.append(SecurityFinding(
                                severity="HIGH",
                                category="ENCRYPTION",
                                title="Weak Cipher Suite Detected",
                                description=f"Port {port} uses weak cipher: {cipher[0]}",
                                affected_component=f"Service on port {port}",
                                remediation="Configure strong cipher suites (AES-256, ChaCha20)"
                            ))
                        
                        # Check TLS version
                        if version and version in ['TLSv1', 'TLSv1.1']:
                            findings.append(SecurityFinding(
                                severity="MEDIUM",
                                category="ENCRYPTION",
                                title="Outdated TLS Version",
                                description=f"Port {port} uses {version}",
                                affected_component=f"Service on port {port}",
                                remediation="Upgrade to TLS 1.2 or 1.3"
                            ))
                            
            except (socket.error, ssl.SSLError, ConnectionRefusedError):
                # Service not running or not using TLS - this is expected for many ports
                continue
            except Exception as e:
                logger.debug(f"TLS audit failed for port {port}: {e}")
        
        return findings
    
    async def _audit_certificates(self) -> List[SecurityFinding]:
        """Audit SSL/TLS certificates"""
        findings = []
        
        # Check for self-signed certificates in production
        cert_paths = [
            '/etc/ssl/certs/',
            './certs/',
            './ssl/',
            './infrastructure/config/ssl/'
        ]
        
        for cert_path in cert_paths:
            path = Path(cert_path)
            if path.exists():
                for cert_file in path.glob('*.crt'):
                    try:
                        with open(cert_file, 'rb') as f:
                            cert_data = f.read()
                            cert = x509.load_pem_x509_certificate(cert_data, default_backend())
                            
                            # Check if self-signed
                            if cert.issuer == cert.subject:
                                findings.append(SecurityFinding(
                                    severity="MEDIUM",
                                    category="ENCRYPTION",
                                    title="Self-Signed Certificate",
                                    description=f"Self-signed certificate found: {cert_file}",
                                    affected_component=str(cert_file),
                                    remediation="Use certificates from trusted CA for production"
                                ))
                            
                            # Check expiration
                            if cert.not_valid_after < datetime.now():
                                findings.append(SecurityFinding(
                                    severity="HIGH",
                                    category="ENCRYPTION",
                                    title="Expired Certificate",
                                    description=f"Expired certificate: {cert_file}",
                                    affected_component=str(cert_file),
                                    remediation="Renew expired certificates"
                                ))
                                
                    except Exception as e:
                        logger.debug(f"Certificate audit failed for {cert_file}: {e}")
        
        return findings
    
    async def _audit_storage_encryption(self) -> List[SecurityFinding]:
        """Audit encryption at rest"""
        findings = []
        
        # Check for unencrypted sensitive files
        sensitive_patterns = [
            '*.key',
            '*.pem',
            '*password*',
            '*secret*',
            '*.env'
        ]
        
        for pattern in sensitive_patterns:
            for file_path in Path('.').rglob(pattern):
                if file_path.is_file():
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read(100)  # Read first 100 chars
                            
                            # Simple check for plaintext secrets
                            if any(keyword in content.lower() for keyword in ['password', 'secret', 'key']):
                                findings.append(SecurityFinding(
                                    severity="HIGH",
                                    category="ENCRYPTION",
                                    title="Plaintext Secrets Detected",
                                    description=f"Potential plaintext secrets in: {file_path}",
                                    affected_component=str(file_path),
                                    remediation="Encrypt sensitive files or use secret management"
                                ))
                                
                    except Exception as e:
                        logger.debug(f"Storage encryption audit failed for {file_path}: {e}")
        
        return findings

class BoundaryTestingAuditor:
    """Tests simulation containment and boundary enforcement"""
    
    def __init__(self):
        self.findings = []
    
    async def audit_simulation_boundaries(self) -> List[SecurityFinding]:
        """Test that simulation cannot escape its boundaries"""
        findings = []
        
        # Test network boundary enforcement
        network_findings = await self._test_network_boundaries()
        findings.extend(network_findings)
        
        # Test file system boundaries
        fs_findings = await self._test_filesystem_boundaries()
        findings.extend(fs_findings)
        
        # Test process boundaries
        process_findings = await self._test_process_boundaries()
        findings.extend(process_findings)
        
        return findings
    
    async def _test_network_boundaries(self) -> List[SecurityFinding]:
        """Test network access restrictions"""
        findings = []
        
        # Test external network access
        external_hosts = [
            'google.com',
            '8.8.8.8',
            'github.com'
        ]
        
        try:
            client = docker.from_env()
            
            for host in external_hosts:
                try:
                    container = client.containers.run(
                        'alpine:latest',
                        command=f'ping -c 1 -W 2 {host}',
                        remove=True,
                        network_mode='bridge'
                    )
                    
                    # If ping succeeds, external access is possible
                    findings.append(SecurityFinding(
                        severity="MEDIUM",
                        category="BOUNDARY",
                        title="External Network Access Possible",
                        description=f"Containers can reach external host: {host}",
                        affected_component="Network Configuration",
                        remediation="Implement network policies to restrict external access"
                    ))
                    
                except docker.errors.ContainerError:
                    # Expected - external access should be blocked
                    pass
                    
        except Exception as e:
            logger.error(f"Network boundary test failed: {e}")
        
        return findings
    
    async def _test_filesystem_boundaries(self) -> List[SecurityFinding]:
        """Test file system access restrictions"""
        findings = []
        
        try:
            client = docker.from_env()
            
            # Test access to host filesystem
            dangerous_paths = [
                '/etc/passwd',
                '/etc/shadow',
                '/proc/version',
                '/sys/class/net'
            ]
            
            for path in dangerous_paths:
                try:
                    result = client.containers.run(
                        'alpine:latest',
                        command=f'cat {path}',
                        remove=True,
                        volumes={'/': {'bind': '/host', 'mode': 'ro'}}
                    )
                    
                    findings.append(SecurityFinding(
                        severity="HIGH",
                        category="BOUNDARY",
                        title="Host Filesystem Access",
                        description=f"Container can access host path: {path}",
                        affected_component="Container Configuration",
                        remediation="Remove host volume mounts and use proper isolation"
                    ))
                    
                except docker.errors.ContainerError:
                    # Expected - access should be blocked
                    pass
                    
        except Exception as e:
            logger.error(f"Filesystem boundary test failed: {e}")
        
        return findings
    
    async def _test_process_boundaries(self) -> List[SecurityFinding]:
        """Test process isolation and privilege boundaries"""
        findings = []
        
        try:
            client = docker.from_env()
            
            # Test privilege escalation
            priv_tests = [
                'whoami',
                'id',
                'ps aux',
                'mount'
            ]
            
            for test_cmd in priv_tests:
                try:
                    result = client.containers.run(
                        'alpine:latest',
                        command=test_cmd,
                        remove=True,
                        user='root'  # This should be restricted
                    )
                    
                    if 'root' in result.decode():
                        findings.append(SecurityFinding(
                            severity="MEDIUM",
                            category="BOUNDARY",
                            title="Root Access in Container",
                            description=f"Container runs with root privileges: {test_cmd}",
                            affected_component="Container Security",
                            remediation="Run containers with non-root user"
                        ))
                        
                except docker.errors.ContainerError:
                    pass
                    
        except Exception as e:
            logger.error(f"Process boundary test failed: {e}")
        
        return findings

class SecurityAuditOrchestrator:
    """Main orchestrator for security audits"""
    
    def __init__(self):
        self.auditors = {
            'container': ContainerSecurityAuditor(),
            'network': NetworkSecurityAuditor(),
            'encryption': EncryptionAuditor(),
            'boundary': BoundaryTestingAuditor()
        }
        self.findings = []
    
    async def run_comprehensive_audit(self) -> AuditResult:
        """Run complete security audit"""
        logger.info("Starting comprehensive security audit...")
        
        audit_id = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        all_findings = []
        
        # Run all auditors
        for auditor_name, auditor in self.auditors.items():
            logger.info(f"Running {auditor_name} security audit...")
            
            try:
                if auditor_name == 'container':
                    findings = await auditor.audit_container_isolation()
                elif auditor_name == 'network':
                    findings = await auditor.audit_network_segmentation()
                elif auditor_name == 'encryption':
                    findings = await auditor.audit_encryption_mechanisms()
                elif auditor_name == 'boundary':
                    findings = await auditor.audit_simulation_boundaries()
                
                all_findings.extend(findings)
                logger.info(f"Completed {auditor_name} audit: {len(findings)} findings")
                
            except Exception as e:
                logger.error(f"Auditor {auditor_name} failed: {e}")
                all_findings.append(SecurityFinding(
                    severity="HIGH",
                    category="AUDIT",
                    title=f"{auditor_name.title()} Audit Failed",
                    description=f"Security audit failed: {e}",
                    affected_component=auditor_name,
                    remediation="Investigate audit framework and system accessibility"
                ))
        
        # Generate summary
        summary = self._generate_summary(all_findings)
        recommendations = self._generate_recommendations(all_findings)
        compliance_status = self._assess_compliance(all_findings)
        
        result = AuditResult(
            audit_id=audit_id,
            timestamp=datetime.now(),
            findings=all_findings,
            summary=summary,
            recommendations=recommendations,
            compliance_status=compliance_status
        )
        
        logger.info(f"Security audit completed: {len(all_findings)} total findings")
        return result
    
    def _generate_summary(self, findings: List[SecurityFinding]) -> Dict[str, int]:
        """Generate findings summary by severity and category"""
        summary = {
            'total': len(findings),
            'by_severity': {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'INFO': 0},
            'by_category': {}
        }
        
        for finding in findings:
            summary['by_severity'][finding.severity] += 1
            
            if finding.category not in summary['by_category']:
                summary['by_category'][finding.category] = 0
            summary['by_category'][finding.category] += 1
        
        return summary
    
    def _generate_recommendations(self, findings: List[SecurityFinding]) -> List[str]:
        """Generate high-level security recommendations"""
        recommendations = []
        
        # Count findings by category
        categories = {}
        for finding in findings:
            if finding.category not in categories:
                categories[finding.category] = []
            categories[finding.category].append(finding)
        
        # Generate category-specific recommendations
        if 'CONTAINER' in categories:
            recommendations.append(
                "Implement container security best practices: run as non-root, "
                "remove unnecessary capabilities, use read-only filesystems"
            )
        
        if 'NETWORK' in categories:
            recommendations.append(
                "Strengthen network segmentation: implement network policies, "
                "disable inter-container communication, use custom networks"
            )
        
        if 'ENCRYPTION' in categories:
            recommendations.append(
                "Enhance encryption: use TLS 1.3, implement certificate management, "
                "encrypt sensitive data at rest"
            )
        
        if 'BOUNDARY' in categories:
            recommendations.append(
                "Enforce simulation boundaries: restrict network access, "
                "implement proper isolation, monitor for escape attempts"
            )
        
        # Add general recommendations based on severity
        critical_count = sum(1 for f in findings if f.severity == 'CRITICAL')
        high_count = sum(1 for f in findings if f.severity == 'HIGH')
        
        if critical_count > 0:
            recommendations.append(
                f"URGENT: Address {critical_count} critical security issues immediately"
            )
        
        if high_count > 0:
            recommendations.append(
                f"HIGH PRIORITY: Remediate {high_count} high-severity vulnerabilities"
            )
        
        return recommendations
    
    def _assess_compliance(self, findings: List[SecurityFinding]) -> Dict[str, bool]:
        """Assess compliance with security requirements"""
        compliance = {
            'container_isolation': True,
            'network_segmentation': True,
            'encryption_standards': True,
            'boundary_enforcement': True,
            'overall_security': True
        }
        
        for finding in findings:
            if finding.severity in ['CRITICAL', 'HIGH']:
                if finding.category == 'CONTAINER':
                    compliance['container_isolation'] = False
                elif finding.category == 'NETWORK':
                    compliance['network_segmentation'] = False
                elif finding.category == 'ENCRYPTION':
                    compliance['encryption_standards'] = False
                elif finding.category == 'BOUNDARY':
                    compliance['boundary_enforcement'] = False
                
                compliance['overall_security'] = False
        
        return compliance
    
    def save_audit_report(self, result: AuditResult, output_path: str = None) -> str:
        """Save audit report to file"""
        if output_path is None:
            output_path = f"security_audit_report_{result.audit_id}.json"
        
        # Convert to serializable format
        report_data = {
            'audit_id': result.audit_id,
            'timestamp': result.timestamp.isoformat(),
            'summary': result.summary,
            'recommendations': result.recommendations,
            'compliance_status': result.compliance_status,
            'findings': [asdict(finding) for finding in result.findings]
        }
        
        # Convert datetime objects in findings
        for finding in report_data['findings']:
            if 'timestamp' in finding and finding['timestamp']:
                finding['timestamp'] = finding['timestamp'].isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Audit report saved to: {output_path}")
        return output_path

async def main():
    """Main entry point for security audit"""
    orchestrator = SecurityAuditOrchestrator()
    
    try:
        # Run comprehensive audit
        result = await orchestrator.run_comprehensive_audit()
        
        # Save report
        report_path = orchestrator.save_audit_report(result)
        
        # Print summary
        print("\n" + "="*60)
        print("SECURITY AUDIT SUMMARY")
        print("="*60)
        print(f"Audit ID: {result.audit_id}")
        print(f"Timestamp: {result.timestamp}")
        print(f"Total Findings: {result.summary['total']}")
        print("\nFindings by Severity:")
        for severity, count in result.summary['by_severity'].items():
            if count > 0:
                print(f"  {severity}: {count}")
        
        print("\nFindings by Category:")
        for category, count in result.summary['by_category'].items():
            print(f"  {category}: {count}")
        
        print("\nCompliance Status:")
        for requirement, status in result.compliance_status.items():
            status_str = "✓ PASS" if status else "✗ FAIL"
            print(f"  {requirement}: {status_str}")
        
        print("\nTop Recommendations:")
        for i, rec in enumerate(result.recommendations[:3], 1):
            print(f"  {i}. {rec}")
        
        print(f"\nDetailed report saved to: {report_path}")
        print("="*60)
        
        return result
        
    except Exception as e:
        logger.error(f"Security audit failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())