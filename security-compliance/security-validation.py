#!/usr/bin/env python3
"""
Automated Security Validation and Compliance Checking
Comprehensive security validation framework for CI/CD pipeline integration
"""

import asyncio
import json
import logging
import os
import subprocess
import time
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import hashlib
import re
from datetime import datetime, timedelta
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SecurityCheck:
    """Individual security check definition."""
    name: str
    category: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    description: str
    check_function: str
    enabled: bool = True
    timeout: int = 300  # seconds
    retry_count: int = 3
    dependencies: List[str] = field(default_factory=list)

@dataclass
class SecurityResult:
    """Security check result."""
    check_name: str
    status: str  # PASS, FAIL, WARNING, SKIP, ERROR
    severity: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    remediation: Optional[str] = None
    references: List[str] = field(default_factory=list)

@dataclass
class ComplianceFramework:
    """Compliance framework definition."""
    name: str
    version: str
    requirements: List[Dict[str, Any]]
    controls: Dict[str, Any]
    scoring: Dict[str, int]

class SecurityValidationFramework:
    """Comprehensive security validation and compliance checking framework."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_configuration(config_file)
        self.results: List[SecurityResult] = []
        self.compliance_frameworks = self._load_compliance_frameworks()
        self.security_checks = self._initialize_security_checks()
        
        # Thresholds and limits
        self.critical_threshold = self.config.get('critical_threshold', 0)
        self.high_threshold = self.config.get('high_threshold', 5)
        self.medium_threshold = self.config.get('medium_threshold', 20)
        self.timeout_limit = self.config.get('timeout_limit', 1800)  # 30 minutes
        
        # Initialize scanners
        self.scanners = {
            'bandit': self._initialize_bandit(),
            'semgrep': self._initialize_semgrep(),
            'trivy': self._initialize_trivy(),
            'safety': self._initialize_safety(),
            'checkov': self._initialize_checkov(),
            'kubesec': self._initialize_kubesec()
        }
    
    def _load_configuration(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load security validation configuration."""
        default_config = {
            'project_root': '.',
            'reports_dir': './security-reports',
            'enable_all_checks': True,
            'parallel_execution': True,
            'max_workers': 4,
            'critical_threshold': 0,
            'high_threshold': 5,
            'medium_threshold': 20,
            'compliance_frameworks': ['CIS', 'NIST', 'OWASP'],
            'exclude_patterns': ['test_*', '*/tests/*', '*/venv/*'],
            'include_patterns': ['**/*.py', '**/*.yaml', '**/*.yml', '**/*.json'],
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                if config_file.endswith(('.yaml', '.yml')):
                    user_config = yaml.safe_load(f)
                else:
                    user_config = json.load(f)
                default_config.update(user_config)
        
        # Create reports directory
        os.makedirs(default_config['reports_dir'], exist_ok=True)
        
        return default_config
    
    def _load_compliance_frameworks(self) -> Dict[str, ComplianceFramework]:
        """Load compliance framework definitions."""
        frameworks = {}
        
        # CIS Controls
        cis_framework = ComplianceFramework(
            name="CIS Controls",
            version="8.0",
            requirements=[
                {
                    'control': 'CIS-3.3',
                    'title': 'Configure Data Access Control Lists',
                    'description': 'Configure data access control lists based on user need to know',
                    'checks': ['file_permissions', 'access_controls', 'least_privilege']
                },
                {
                    'control': 'CIS-11.1',
                    'title': 'Establish and Maintain a Data Recovery Process',
                    'description': 'Establish and maintain a data recovery process',
                    'checks': ['backup_validation', 'disaster_recovery']
                },
                {
                    'control': 'CIS-12.2',
                    'title': 'Establish and Maintain a Software Bill of Materials',
                    'description': 'Establish and maintain a software bill of materials',
                    'checks': ['dependency_scanning', 'vulnerability_management']
                }
            ],
            controls={
                'access_control': {'weight': 10, 'required': True},
                'data_protection': {'weight': 8, 'required': True},
                'incident_response': {'weight': 6, 'required': False}
            },
            scoring={'max_score': 100, 'pass_threshold': 80}
        )
        frameworks['CIS'] = cis_framework
        
        # NIST Cybersecurity Framework
        nist_framework = ComplianceFramework(
            name="NIST Cybersecurity Framework",
            version="1.1",
            requirements=[
                {
                    'function': 'IDENTIFY',
                    'category': 'ID.AM',
                    'subcategory': 'ID.AM-2',
                    'description': 'Software platforms and applications are inventoried',
                    'checks': ['asset_inventory', 'software_catalog']
                },
                {
                    'function': 'PROTECT',
                    'category': 'PR.AC',
                    'subcategory': 'PR.AC-1',
                    'description': 'Identities and credentials are issued, managed, verified',
                    'checks': ['authentication', 'authorization', 'credential_management']
                },
                {
                    'function': 'DETECT',
                    'category': 'DE.CM',
                    'subcategory': 'DE.CM-1',
                    'description': 'The network is monitored to detect potential cybersecurity events',
                    'checks': ['network_monitoring', 'intrusion_detection']
                }
            ],
            controls={
                'identify': {'weight': 20, 'required': True},
                'protect': {'weight': 25, 'required': True},
                'detect': {'weight': 20, 'required': True},
                'respond': {'weight': 20, 'required': True},
                'recover': {'weight': 15, 'required': False}
            },
            scoring={'max_score': 100, 'pass_threshold': 75}
        )
        frameworks['NIST'] = nist_framework
        
        # OWASP Top 10
        owasp_framework = ComplianceFramework(
            name="OWASP Top 10",
            version="2021",
            requirements=[
                {
                    'id': 'A01:2021',
                    'title': 'Broken Access Control',
                    'description': 'Access control enforces policy such that users cannot act outside of their intended permissions',
                    'checks': ['access_control', 'authorization', 'privilege_escalation']
                },
                {
                    'id': 'A02:2021',
                    'title': 'Cryptographic Failures',
                    'description': 'Protect data in transit and at rest',
                    'checks': ['encryption', 'tls_configuration', 'key_management']
                },
                {
                    'id': 'A03:2021',
                    'title': 'Injection',
                    'description': 'Application is vulnerable to injection attacks',
                    'checks': ['sql_injection', 'command_injection', 'input_validation']
                }
            ],
            controls={
                'access_control': {'weight': 15, 'required': True},
                'cryptography': {'weight': 12, 'required': True},
                'injection_prevention': {'weight': 12, 'required': True},
                'insecure_design': {'weight': 10, 'required': True},
                'security_misconfiguration': {'weight': 10, 'required': True}
            },
            scoring={'max_score': 100, 'pass_threshold': 85}
        )
        frameworks['OWASP'] = owasp_framework
        
        return frameworks
    
    def _initialize_security_checks(self) -> List[SecurityCheck]:
        """Initialize comprehensive security checks."""
        checks = [
            # Code Security Checks
            SecurityCheck(
                name="python_bandit_scan",
                category="code_security",
                severity="HIGH",
                description="Python security issues using Bandit",
                check_function="run_bandit_scan"
            ),
            SecurityCheck(
                name="semgrep_security_scan",
                category="code_security", 
                severity="HIGH",
                description="Multi-language security patterns using Semgrep",
                check_function="run_semgrep_scan"
            ),
            SecurityCheck(
                name="dependency_vulnerability_scan",
                category="dependencies",
                severity="CRITICAL",
                description="Known vulnerabilities in dependencies",
                check_function="run_dependency_scan"
            ),
            
            # Container Security Checks
            SecurityCheck(
                name="container_image_scan",
                category="container_security",
                severity="HIGH",
                description="Container image vulnerabilities using Trivy",
                check_function="run_container_scan"
            ),
            SecurityCheck(
                name="dockerfile_security_scan",
                category="container_security",
                severity="MEDIUM",
                description="Dockerfile security best practices",
                check_function="run_dockerfile_scan"
            ),
            
            # Infrastructure Security Checks
            SecurityCheck(
                name="terraform_security_scan",
                category="infrastructure",
                severity="HIGH",
                description="Infrastructure as Code security using Checkov",
                check_function="run_terraform_scan"
            ),
            SecurityCheck(
                name="kubernetes_security_scan", 
                category="infrastructure",
                severity="HIGH",
                description="Kubernetes manifest security using Kubesec",
                check_function="run_kubernetes_scan"
            ),
            SecurityCheck(
                name="ansible_security_scan",
                category="infrastructure",
                severity="MEDIUM",
                description="Ansible playbook security using ansible-lint",
                check_function="run_ansible_scan"
            ),
            
            # Configuration Security Checks
            SecurityCheck(
                name="secrets_detection",
                category="secrets",
                severity="CRITICAL",
                description="Hardcoded secrets and credentials detection",
                check_function="run_secrets_scan"
            ),
            SecurityCheck(
                name="file_permissions_check",
                category="configuration",
                severity="MEDIUM",
                description="File and directory permissions validation",
                check_function="check_file_permissions"
            ),
            SecurityCheck(
                name="network_configuration_check",
                category="configuration",
                severity="HIGH",
                description="Network security configuration validation",
                check_function="check_network_configuration"
            ),
            
            # Compliance Checks
            SecurityCheck(
                name="cis_compliance_check",
                category="compliance",
                severity="HIGH",
                description="CIS Controls compliance validation",
                check_function="run_cis_compliance_check"
            ),
            SecurityCheck(
                name="nist_compliance_check",
                category="compliance",
                severity="HIGH",
                description="NIST Cybersecurity Framework compliance",
                check_function="run_nist_compliance_check"
            ),
            SecurityCheck(
                name="owasp_compliance_check",
                category="compliance",
                severity="HIGH",
                description="OWASP Top 10 compliance validation",
                check_function="run_owasp_compliance_check"
            ),
            
            # Runtime Security Checks
            SecurityCheck(
                name="runtime_security_check",
                category="runtime",
                severity="MEDIUM",
                description="Runtime security configuration validation",
                check_function="check_runtime_security"
            ),
            SecurityCheck(
                name="api_security_check",
                category="api_security",
                severity="HIGH",
                description="API security configuration validation",
                check_function="check_api_security"
            )
        ]
        
        return checks
    
    def _initialize_bandit(self) -> Dict[str, Any]:
        """Initialize Bandit scanner configuration."""
        return {
            'command': 'bandit',
            'args': ['-r', '.', '-f', 'json', '-o'],
            'config_file': '.bandit.yml',
            'severity_levels': {
                'LOW': 1,
                'MEDIUM': 5,
                'HIGH': 10
            }
        }
    
    def _initialize_semgrep(self) -> Dict[str, Any]:
        """Initialize Semgrep scanner configuration."""
        return {
            'command': 'semgrep',
            'args': ['--config=auto', '--json', '--severity=ERROR', '--severity=WARNING'],
            'rulesets': [
                'p/security-audit',
                'p/secrets',
                'p/owasp-top-ten',
                'p/docker'
            ]
        }
    
    def _initialize_trivy(self) -> Dict[str, Any]:
        """Initialize Trivy scanner configuration."""
        return {
            'command': 'trivy',
            'image_args': ['image', '--format', 'json', '--severity', 'CRITICAL,HIGH,MEDIUM'],
            'fs_args': ['fs', '--format', 'json', '--security-checks', 'vuln,config,secret'],
            'severity_mapping': {
                'CRITICAL': 10,
                'HIGH': 8,
                'MEDIUM': 5,
                'LOW': 2
            }
        }
    
    def _initialize_safety(self) -> Dict[str, Any]:
        """Initialize Safety scanner configuration."""
        return {
            'command': 'safety',
            'args': ['check', '--json', '--full-report'],
            'database': 'latest'
        }
    
    def _initialize_checkov(self) -> Dict[str, Any]:
        """Initialize Checkov scanner configuration."""
        return {
            'command': 'checkov',
            'args': ['-d', '.', '--framework', 'terraform,kubernetes,dockerfile', '--output', 'json'],
            'severity_mapping': {
                'CRITICAL': 10,
                'HIGH': 8,
                'MEDIUM': 5,
                'LOW': 2
            }
        }
    
    def _initialize_kubesec(self) -> Dict[str, Any]:
        """Initialize Kubesec scanner configuration."""
        return {
            'command': 'kubesec',
            'args': ['scan'],
            'api_endpoint': 'https://v2.kubesec.io/scan'
        }
    
    async def run_comprehensive_security_validation(self) -> Dict[str, Any]:
        """Run comprehensive security validation."""
        logger.info("Starting comprehensive security validation")
        start_time = time.time()
        
        try:
            # Run all security checks
            if self.config.get('parallel_execution', True):
                await self._run_checks_parallel()
            else:
                await self._run_checks_sequential()
            
            # Generate compliance reports
            compliance_results = self._generate_compliance_reports()
            
            # Calculate overall security score
            security_score = self._calculate_security_score()
            
            # Generate final report
            final_report = self._generate_final_report(
                execution_time=time.time() - start_time,
                compliance_results=compliance_results,
                security_score=security_score
            )
            
            # Save reports
            self._save_reports(final_report)
            
            logger.info(f"Security validation completed in {time.time() - start_time:.2f}s")
            return final_report
            
        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            raise
    
    async def _run_checks_parallel(self):
        """Run security checks in parallel."""
        import concurrent.futures
        
        max_workers = self.config.get('max_workers', 4)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for check in self.security_checks:
                if not check.enabled:
                    continue
                
                future = executor.submit(self._execute_security_check, check)
                futures.append(future)
            
            # Wait for all checks to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=self.timeout_limit)
                    if result:
                        self.results.append(result)
                except Exception as e:
                    logger.error(f"Security check failed: {e}")
    
    async def _run_checks_sequential(self):
        """Run security checks sequentially."""
        for check in self.security_checks:
            if not check.enabled:
                continue
            
            try:
                result = self._execute_security_check(check)
                if result:
                    self.results.append(result)
            except Exception as e:
                logger.error(f"Security check {check.name} failed: {e}")
                self.results.append(SecurityResult(
                    check_name=check.name,
                    status="ERROR",
                    severity=check.severity,
                    message=f"Check execution failed: {e}"
                ))
    
    def _execute_security_check(self, check: SecurityCheck) -> Optional[SecurityResult]:
        """Execute a single security check."""
        logger.info(f"Running security check: {check.name}")
        start_time = time.time()
        
        try:
            # Get the check function
            check_function = getattr(self, check.check_function, None)
            if not check_function:
                return SecurityResult(
                    check_name=check.name,
                    status="ERROR",
                    severity=check.severity,
                    message=f"Check function {check.check_function} not found"
                )
            
            # Execute the check with timeout
            result = check_function()
            
            if result:
                result.execution_time = time.time() - start_time
                return result
            
        except subprocess.TimeoutExpired:
            return SecurityResult(
                check_name=check.name,
                status="ERROR",
                severity=check.severity,
                message=f"Check timed out after {check.timeout} seconds"
            )
        except Exception as e:
            return SecurityResult(
                check_name=check.name,
                status="ERROR",
                severity=check.severity,
                message=f"Check execution failed: {e}"
            )
        
        return None
    
    # Security Check Implementations
    
    def run_bandit_scan(self) -> SecurityResult:
        """Run Bandit Python security scan."""
        try:
            output_file = os.path.join(self.config['reports_dir'], 'bandit-report.json')
            cmd = ['bandit', '-r', '.', '-f', 'json', '-o', output_file]
            
            if os.path.exists('.bandit.yml'):
                cmd.extend(['-c', '.bandit.yml'])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Bandit returns non-zero exit codes for findings, which is expected
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    bandit_results = json.load(f)
                
                issues = bandit_results.get('results', [])
                metrics = bandit_results.get('metrics', {})
                
                high_issues = len([i for i in issues if i.get('issue_severity') == 'HIGH'])
                medium_issues = len([i for i in issues if i.get('issue_severity') == 'MEDIUM'])
                low_issues = len([i for i in issues if i.get('issue_severity') == 'LOW'])
                
                if high_issues > self.config.get('bandit_high_threshold', 0):
                    status = "FAIL"
                    message = f"Found {high_issues} high severity issues"
                elif medium_issues > self.config.get('bandit_medium_threshold', 5):
                    status = "WARNING"
                    message = f"Found {medium_issues} medium severity issues"
                else:
                    status = "PASS"
                    message = f"Found {len(issues)} total issues (acceptable)"
                
                return SecurityResult(
                    check_name="python_bandit_scan",
                    status=status,
                    severity="HIGH",
                    message=message,
                    details={
                        'total_issues': len(issues),
                        'high_issues': high_issues,
                        'medium_issues': medium_issues,
                        'low_issues': low_issues,
                        'files_scanned': metrics.get('_totals', {}).get('loc', 0),
                        'report_file': output_file
                    }
                )
            else:
                return SecurityResult(
                    check_name="python_bandit_scan",
                    status="ERROR",
                    severity="HIGH",
                    message="Bandit scan failed to generate report"
                )
                
        except subprocess.TimeoutExpired:
            return SecurityResult(
                check_name="python_bandit_scan",
                status="ERROR",
                severity="HIGH",
                message="Bandit scan timed out"
            )
        except Exception as e:
            return SecurityResult(
                check_name="python_bandit_scan",
                status="ERROR",
                severity="HIGH",
                message=f"Bandit scan failed: {e}"
            )
    
    def run_semgrep_scan(self) -> SecurityResult:
        """Run Semgrep security scan."""
        try:
            output_file = os.path.join(self.config['reports_dir'], 'semgrep-report.json')
            cmd = [
                'semgrep',
                '--config=auto',
                '--json',
                '--output', output_file,
                '--severity=ERROR',
                '--severity=WARNING',
                '.'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    semgrep_results = json.load(f)
                
                findings = semgrep_results.get('results', [])
                errors = semgrep_results.get('errors', [])
                
                error_findings = [f for f in findings if f.get('extra', {}).get('severity') == 'ERROR']
                warning_findings = [f for f in findings if f.get('extra', {}).get('severity') == 'WARNING']
                
                if len(error_findings) > 0:
                    status = "FAIL"
                    message = f"Found {len(error_findings)} error-level security issues"
                elif len(warning_findings) > self.config.get('semgrep_warning_threshold', 10):
                    status = "WARNING"
                    message = f"Found {len(warning_findings)} warning-level issues"
                else:
                    status = "PASS"
                    message = f"Found {len(findings)} total issues (acceptable)"
                
                return SecurityResult(
                    check_name="semgrep_security_scan",
                    status=status,
                    severity="HIGH",
                    message=message,
                    details={
                        'total_findings': len(findings),
                        'error_findings': len(error_findings),
                        'warning_findings': len(warning_findings),
                        'scan_errors': len(errors),
                        'report_file': output_file
                    }
                )
            else:
                return SecurityResult(
                    check_name="semgrep_security_scan",
                    status="ERROR",
                    severity="HIGH",
                    message="Semgrep scan failed to generate report"
                )
                
        except subprocess.TimeoutExpired:
            return SecurityResult(
                check_name="semgrep_security_scan",
                status="ERROR",
                severity="HIGH",
                message="Semgrep scan timed out"
            )
        except Exception as e:
            return SecurityResult(
                check_name="semgrep_security_scan",
                status="ERROR",
                severity="HIGH",
                message=f"Semgrep scan failed: {e}"
            )
    
    def run_dependency_scan(self) -> SecurityResult:
        """Run dependency vulnerability scan using Safety."""
        try:
            output_file = os.path.join(self.config['reports_dir'], 'safety-report.json')
            cmd = ['safety', 'check', '--json', '--full-report']
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Safety returns different exit codes for different scenarios
            if result.returncode == 0:
                status = "PASS"
                message = "No known vulnerabilities found in dependencies"
                vulnerabilities = []
            else:
                try:
                    if result.stdout:
                        safety_results = json.loads(result.stdout)
                        vulnerabilities = safety_results.get('vulnerabilities', [])
                    else:
                        vulnerabilities = []
                except json.JSONDecodeError:
                    vulnerabilities = []
                
                critical_vulns = len([v for v in vulnerabilities if v.get('severity', '').upper() == 'CRITICAL'])
                high_vulns = len([v for v in vulnerabilities if v.get('severity', '').upper() == 'HIGH'])
                
                if critical_vulns > 0:
                    status = "FAIL"
                    message = f"Found {critical_vulns} critical vulnerabilities in dependencies"
                elif high_vulns > self.config.get('safety_high_threshold', 3):
                    status = "WARNING"
                    message = f"Found {high_vulns} high severity vulnerabilities"
                else:
                    status = "PASS"
                    message = f"Found {len(vulnerabilities)} vulnerabilities (acceptable)"
            
            # Save results to file
            with open(output_file, 'w') as f:
                json.dump({
                    'vulnerabilities': vulnerabilities,
                    'scan_time': datetime.now().isoformat(),
                    'status': status
                }, f, indent=2)
            
            return SecurityResult(
                check_name="dependency_vulnerability_scan",
                status=status,
                severity="CRITICAL",
                message=message,
                details={
                    'total_vulnerabilities': len(vulnerabilities),
                    'critical_vulnerabilities': len([v for v in vulnerabilities if v.get('severity', '').upper() == 'CRITICAL']),
                    'high_vulnerabilities': len([v for v in vulnerabilities if v.get('severity', '').upper() == 'HIGH']),
                    'report_file': output_file
                }
            )
            
        except Exception as e:
            return SecurityResult(
                check_name="dependency_vulnerability_scan",
                status="ERROR",
                severity="CRITICAL",
                message=f"Dependency scan failed: {e}"
            )
    
    def run_secrets_scan(self) -> SecurityResult:
        """Run secrets detection scan."""
        try:
            secrets_found = []
            
            # Define patterns for common secrets
            secret_patterns = {
                'aws_access_key': r'AKIA[0-9A-Z]{16}',
                'aws_secret_key': r'[0-9a-zA-Z/+]{40}',
                'github_token': r'ghp_[0-9a-zA-Z]{36}',
                'slack_token': r'xox[baprs]-[0-9]{12}-[0-9]{12}-[0-9a-zA-Z]{24}',
                'generic_api_key': r'api[_-]?key["\']?\s*[:=]\s*["\'][0-9a-zA-Z]{32,}["\']',
                'password': r'password["\']?\s*[:=]\s*["\'][^"\']{8,}["\']',
                'private_key': r'-----BEGIN [A-Z ]+ PRIVATE KEY-----'
            }
            
            # Scan files for secrets
            for root, dirs, files in os.walk('.'):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if not any(pattern in os.path.join(root, d) for pattern in self.config.get('exclude_patterns', []))]
                
                for file in files:
                    if file.endswith(('.py', '.yaml', '.yml', '.json', '.env', '.config')):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                
                                for secret_type, pattern in secret_patterns.items():
                                    matches = re.finditer(pattern, content, re.IGNORECASE)
                                    for match in matches:
                                        line_num = content[:match.start()].count('\n') + 1
                                        secrets_found.append({
                                            'type': secret_type,
                                            'file': file_path,
                                            'line': line_num,
                                            'context': content[max(0, match.start()-50):match.end()+50]
                                        })
                        except Exception:
                            continue
            
            if len(secrets_found) > 0:
                status = "FAIL"
                message = f"Found {len(secrets_found)} potential secrets in code"
            else:
                status = "PASS"
                message = "No hardcoded secrets detected"
            
            return SecurityResult(
                check_name="secrets_detection",
                status=status,
                severity="CRITICAL",
                message=message,
                details={
                    'secrets_found': len(secrets_found),
                    'secret_types': list(set([s['type'] for s in secrets_found])),
                    'affected_files': list(set([s['file'] for s in secrets_found]))
                },
                remediation="Remove hardcoded secrets and use environment variables or secret management systems"
            )
            
        except Exception as e:
            return SecurityResult(
                check_name="secrets_detection",
                status="ERROR",
                severity="CRITICAL",
                message=f"Secrets scan failed: {e}"
            )
    
    def _generate_compliance_reports(self) -> Dict[str, Any]:
        """Generate compliance framework reports."""
        compliance_results = {}
        
        for framework_name, framework in self.compliance_frameworks.items():
            if framework_name not in self.config.get('compliance_frameworks', []):
                continue
            
            framework_score = 0
            max_score = framework.scoring['max_score']
            requirements_met = 0
            total_requirements = len(framework.requirements)
            
            # Calculate compliance score based on security check results
            for requirement in framework.requirements:
                requirement_checks = requirement.get('checks', [])
                requirement_passed = True
                
                for check_name in requirement_checks:
                    # Find corresponding security results
                    check_results = [r for r in self.results if check_name in r.check_name.lower()]
                    if not check_results or any(r.status == 'FAIL' for r in check_results):
                        requirement_passed = False
                        break
                
                if requirement_passed:
                    requirements_met += 1
            
            framework_score = (requirements_met / total_requirements) * max_score
            compliance_status = "COMPLIANT" if framework_score >= framework.scoring['pass_threshold'] else "NON_COMPLIANT"
            
            compliance_results[framework_name] = {
                'framework': framework.name,
                'version': framework.version,
                'score': framework_score,
                'max_score': max_score,
                'requirements_met': requirements_met,
                'total_requirements': total_requirements,
                'compliance_percentage': (requirements_met / total_requirements) * 100,
                'status': compliance_status,
                'pass_threshold': framework.scoring['pass_threshold']
            }
        
        return compliance_results
    
    def _calculate_security_score(self) -> Dict[str, Any]:
        """Calculate overall security score."""
        if not self.results:
            return {'score': 0, 'grade': 'F', 'details': 'No security checks performed'}
        
        total_checks = len(self.results)
        passed_checks = len([r for r in self.results if r.status == 'PASS'])
        failed_checks = len([r for r in self.results if r.status == 'FAIL'])
        warning_checks = len([r for r in self.results if r.status == 'WARNING'])
        error_checks = len([r for r in self.results if r.status == 'ERROR'])
        
        # Weight by severity
        severity_weights = {'CRITICAL': 10, 'HIGH': 5, 'MEDIUM': 3, 'LOW': 1}
        total_weight = sum(severity_weights.get(r.severity, 1) for r in self.results)
        weighted_passed = sum(severity_weights.get(r.severity, 1) for r in self.results if r.status == 'PASS')
        
        # Calculate weighted score
        weighted_score = (weighted_passed / total_weight * 100) if total_weight > 0 else 0
        
        # Determine grade
        if weighted_score >= 95:
            grade = 'A+'
        elif weighted_score >= 90:
            grade = 'A'
        elif weighted_score >= 85:
            grade = 'B+'
        elif weighted_score >= 80:
            grade = 'B'
        elif weighted_score >= 75:
            grade = 'C+'
        elif weighted_score >= 70:
            grade = 'C'
        elif weighted_score >= 65:
            grade = 'D'
        else:
            grade = 'F'
        
        return {
            'score': weighted_score,
            'grade': grade,
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': failed_checks,
            'warning_checks': warning_checks,
            'error_checks': error_checks,
            'pass_rate': (passed_checks / total_checks * 100) if total_checks > 0 else 0
        }
    
    def _generate_final_report(self, execution_time: float, compliance_results: Dict[str, Any], security_score: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final security report."""
        # Categorize results by severity and status
        critical_failures = [r for r in self.results if r.severity == 'CRITICAL' and r.status == 'FAIL']
        high_failures = [r for r in self.results if r.severity == 'HIGH' and r.status == 'FAIL']
        
        # Determine overall status
        if len(critical_failures) > self.critical_threshold:
            overall_status = "CRITICAL_FAILURE"
        elif len(high_failures) > self.high_threshold:
            overall_status = "HIGH_RISK"
        elif security_score['score'] < 70:
            overall_status = "MEDIUM_RISK"
        else:
            overall_status = "ACCEPTABLE"
        
        report = {
            'summary': {
                'scan_timestamp': datetime.now().isoformat(),
                'execution_time_seconds': execution_time,
                'overall_status': overall_status,
                'security_score': security_score,
                'total_checks_performed': len(self.results),
                'critical_failures': len(critical_failures),
                'high_failures': len(high_failures),
                'pass_rate_percentage': security_score['pass_rate']
            },
            'detailed_results': [
                {
                    'check_name': r.check_name,
                    'status': r.status,
                    'severity': r.severity,
                    'message': r.message,
                    'execution_time': r.execution_time,
                    'details': r.details,
                    'remediation': r.remediation,
                    'references': r.references
                }
                for r in self.results
            ],
            'compliance_results': compliance_results,
            'recommendations': self._generate_recommendations(),
            'next_steps': self._generate_next_steps(overall_status),
            'configuration': {
                'thresholds': {
                    'critical_threshold': self.critical_threshold,
                    'high_threshold': self.high_threshold,
                    'medium_threshold': self.medium_threshold
                },
                'frameworks_evaluated': list(compliance_results.keys()),
                'scanners_used': list(self.scanners.keys())
            }
        }
        
        return report
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate security recommendations based on findings."""
        recommendations = []
        
        # Analyze patterns in failures
        critical_issues = [r for r in self.results if r.severity == 'CRITICAL' and r.status == 'FAIL']
        high_issues = [r for r in self.results if r.severity == 'HIGH' and r.status == 'FAIL']
        
        if len(critical_issues) > 0:
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'immediate_action',
                'title': 'Address Critical Security Issues',
                'description': f'Immediately address {len(critical_issues)} critical security issues before deployment',
                'actions': ['Review all CRITICAL findings', 'Implement fixes', 'Re-run security validation']
            })
        
        if len(high_issues) > 5:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'security_hardening',
                'title': 'Implement Security Hardening',
                'description': 'Multiple high-severity issues indicate need for systematic security hardening',
                'actions': ['Conduct security architecture review', 'Implement defense-in-depth', 'Regular security training']
            })
        
        # Check for specific issue patterns
        dependency_issues = [r for r in self.results if 'dependency' in r.check_name and r.status == 'FAIL']
        if dependency_issues:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'dependency_management',
                'title': 'Implement Robust Dependency Management',
                'description': 'Vulnerable dependencies pose significant risk',
                'actions': ['Regular dependency updates', 'Automated vulnerability scanning', 'Dependency pinning']
            })
        
        return recommendations
    
    def _generate_next_steps(self, overall_status: str) -> List[str]:
        """Generate next steps based on overall security status."""
        if overall_status == "CRITICAL_FAILURE":
            return [
                "STOP: Do not proceed with deployment",
                "Address all critical security issues immediately",
                "Conduct emergency security review",
                "Re-run complete security validation",
                "Consider security incident response procedures"
            ]
        elif overall_status == "HIGH_RISK":
            return [
                "Address high-severity issues before production deployment",
                "Implement additional monitoring and logging",
                "Schedule security review meeting",
                "Plan remediation timeline",
                "Consider staged deployment with enhanced monitoring"
            ]
        elif overall_status == "MEDIUM_RISK":
            return [
                "Review medium-severity findings",
                "Plan remediation in next development cycle",
                "Implement additional security controls",
                "Schedule follow-up security assessment",
                "Proceed with enhanced monitoring"
            ]
        else:
            return [
                "Security validation passed successfully",
                "Monitor for any new security alerts",
                "Schedule regular security assessments",
                "Continue following security best practices",
                "Proceed with deployment"
            ]
    
    def _save_reports(self, final_report: Dict[str, Any]):
        """Save security reports to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON report
        json_report_file = os.path.join(self.config['reports_dir'], f'security_validation_report_{timestamp}.json')
        with open(json_report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        # Save summary report
        summary_file = os.path.join(self.config['reports_dir'], f'security_summary_{timestamp}.md')
        self._generate_markdown_summary(final_report, summary_file)
        
        # Save SARIF report for GitHub integration
        sarif_file = os.path.join(self.config['reports_dir'], f'security_results_{timestamp}.sarif')
        self._generate_sarif_report(final_report, sarif_file)
        
        logger.info(f"Security reports saved:")
        logger.info(f"  JSON Report: {json_report_file}")
        logger.info(f"  Summary: {summary_file}")
        logger.info(f"  SARIF: {sarif_file}")
    
    def _generate_markdown_summary(self, report: Dict[str, Any], output_file: str):
        """Generate markdown summary report."""
        summary = report['summary']
        
        markdown_content = f"""# Security Validation Report

**Scan Date**: {summary['scan_timestamp']}  
**Execution Time**: {summary['execution_time_seconds']:.2f} seconds  
**Overall Status**: {summary['overall_status']}  

## Security Score: {summary['security_score']['grade']} ({summary['security_score']['score']:.1f}/100)

### Summary Statistics
- **Total Checks**: {summary['total_checks_performed']}
- **Pass Rate**: {summary['pass_rate_percentage']:.1f}%
- **Critical Failures**: {summary['critical_failures']}
- **High-Risk Issues**: {summary['high_failures']}

### Compliance Results
"""
        
        for framework, result in report['compliance_results'].items():
            status_emoji = "✅" if result['status'] == 'COMPLIANT' else "❌"
            markdown_content += f"- {status_emoji} **{framework}**: {result['compliance_percentage']:.1f}% ({result['requirements_met']}/{result['total_requirements']} requirements)\n"
        
        markdown_content += "\n### Key Recommendations\n"
        for rec in report['recommendations'][:5]:  # Top 5 recommendations
            markdown_content += f"- **{rec['priority']}**: {rec['title']} - {rec['description']}\n"
        
        markdown_content += "\n### Next Steps\n"
        for step in report['next_steps']:
            markdown_content += f"- {step}\n"
        
        with open(output_file, 'w') as f:
            f.write(markdown_content)
    
    def _generate_sarif_report(self, report: Dict[str, Any], output_file: str):
        """Generate SARIF format report for GitHub integration."""
        sarif_report = {
            "version": "2.1.0",
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "Archangel Security Validator",
                            "version": "1.0.0",
                            "informationUri": "https://github.com/archangel/security-validator"
                        }
                    },
                    "results": []
                }
            ]
        }
        
        for result in report['detailed_results']:
            if result['status'] in ['FAIL', 'WARNING']:
                sarif_result = {
                    "ruleId": result['check_name'],
                    "message": {
                        "text": result['message']
                    },
                    "level": {
                        'CRITICAL': 'error',
                        'HIGH': 'error',
                        'MEDIUM': 'warning',
                        'LOW': 'note'
                    }.get(result['severity'], 'warning'),
                    "properties": {
                        "severity": result['severity'],
                        "executionTime": result['execution_time']
                    }
                }
                
                sarif_report["runs"][0]["results"].append(sarif_result)
        
        with open(output_file, 'w') as f:
            json.dump(sarif_report, f, indent=2)

    # Placeholder implementations for other security checks
    def run_container_scan(self) -> SecurityResult:
        """Placeholder for container security scan."""
        return SecurityResult(
            check_name="container_image_scan",
            status="SKIP",
            severity="HIGH",
            message="Container scan not implemented - placeholder"
        )
    
    def run_dockerfile_scan(self) -> SecurityResult:
        """Placeholder for Dockerfile security scan.""" 
        return SecurityResult(
            check_name="dockerfile_security_scan",
            status="SKIP",
            severity="MEDIUM",
            message="Dockerfile scan not implemented - placeholder"
        )
    
    def run_terraform_scan(self) -> SecurityResult:
        """Placeholder for Terraform security scan."""
        return SecurityResult(
            check_name="terraform_security_scan",
            status="SKIP", 
            severity="HIGH",
            message="Terraform scan not implemented - placeholder"
        )
    
    def run_kubernetes_scan(self) -> SecurityResult:
        """Placeholder for Kubernetes security scan."""
        return SecurityResult(
            check_name="kubernetes_security_scan",
            status="SKIP",
            severity="HIGH", 
            message="Kubernetes scan not implemented - placeholder"
        )
    
    def run_ansible_scan(self) -> SecurityResult:
        """Placeholder for Ansible security scan."""
        return SecurityResult(
            check_name="ansible_security_scan",
            status="SKIP",
            severity="MEDIUM",
            message="Ansible scan not implemented - placeholder"
        )
    
    def check_file_permissions(self) -> SecurityResult:
        """Placeholder for file permissions check."""
        return SecurityResult(
            check_name="file_permissions_check",
            status="SKIP",
            severity="MEDIUM",
            message="File permissions check not implemented - placeholder"
        )
    
    def check_network_configuration(self) -> SecurityResult:
        """Placeholder for network configuration check."""
        return SecurityResult(
            check_name="network_configuration_check",
            status="SKIP",
            severity="HIGH",
            message="Network configuration check not implemented - placeholder"
        )
    
    def run_cis_compliance_check(self) -> SecurityResult:
        """Placeholder for CIS compliance check."""
        return SecurityResult(
            check_name="cis_compliance_check",
            status="SKIP",
            severity="HIGH",
            message="CIS compliance check not implemented - placeholder"
        )
    
    def run_nist_compliance_check(self) -> SecurityResult:
        """Placeholder for NIST compliance check."""
        return SecurityResult(
            check_name="nist_compliance_check",
            status="SKIP",
            severity="HIGH",
            message="NIST compliance check not implemented - placeholder"
        )
    
    def run_owasp_compliance_check(self) -> SecurityResult:
        """Placeholder for OWASP compliance check."""
        return SecurityResult(
            check_name="owasp_compliance_check",
            status="SKIP",
            severity="HIGH",
            message="OWASP compliance check not implemented - placeholder"
        )
    
    def check_runtime_security(self) -> SecurityResult:
        """Placeholder for runtime security check."""
        return SecurityResult(
            check_name="runtime_security_check",
            status="SKIP",
            severity="MEDIUM",
            message="Runtime security check not implemented - placeholder"
        )
    
    def check_api_security(self) -> SecurityResult:
        """Placeholder for API security check."""
        return SecurityResult(
            check_name="api_security_check",
            status="SKIP",
            severity="HIGH",
            message="API security check not implemented - placeholder"
        )


async def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Archangel Security Validation Framework")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--output-dir", default="./security-reports", help="Output directory for reports")
    parser.add_argument("--parallel", action="store_true", help="Run checks in parallel")
    parser.add_argument("--frameworks", nargs='+', choices=['CIS', 'NIST', 'OWASP'], 
                       default=['CIS', 'NIST', 'OWASP'], help="Compliance frameworks to evaluate")
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'reports_dir': args.output_dir,
        'parallel_execution': args.parallel,
        'compliance_frameworks': args.frameworks
    }
    
    # Initialize and run security validation
    validator = SecurityValidationFramework(args.config)
    validator.config.update(config)
    
    try:
        final_report = await validator.run_comprehensive_security_validation()
        
        # Print summary
        summary = final_report['summary']
        print(f"\n{'='*60}")
        print("SECURITY VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Security Score: {summary['security_score']['grade']} ({summary['security_score']['score']:.1f}/100)")
        print(f"Total Checks: {summary['total_checks_performed']}")
        print(f"Pass Rate: {summary['pass_rate_percentage']:.1f}%")
        print(f"Critical Failures: {summary['critical_failures']}")
        print(f"High-Risk Issues: {summary['high_failures']}")
        print(f"Execution Time: {summary['execution_time_seconds']:.2f}s")
        print(f"{'='*60}\n")
        
        # Exit with appropriate code
        if summary['overall_status'] in ['CRITICAL_FAILURE', 'HIGH_RISK']:
            exit(1)
        else:
            exit(0)
            
    except Exception as e:
        logger.error(f"Security validation failed: {e}")
        exit(2)

if __name__ == "__main__":
    asyncio.run(main())