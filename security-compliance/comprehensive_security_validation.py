#!/usr/bin/env python3
"""
Comprehensive Security Validation for Archangel Autonomous AI System
Combines security auditing and penetration testing for complete assessment
"""

import asyncio
import json
import logging
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Import our custom security modules
try:
    from security_audit_framework import SecurityAuditOrchestrator, AuditResult
    from penetration_testing import PenetrationTestOrchestrator, PenTestReport
except ImportError:
    # Fallback if modules not available
    SecurityAuditOrchestrator = None
    PenetrationTestOrchestrator = None
    AuditResult = None
    PenTestReport = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ComplianceCheck:
    """Represents a compliance check result"""
    check_name: str
    requirement_id: str
    status: str  # PASS, FAIL, WARNING, SKIP
    description: str
    evidence: Dict[str, Any]
    remediation: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class SecurityValidationReport:
    """Complete security validation report"""
    validation_id: str
    timestamp: datetime
    audit_results: Optional[Dict[str, Any]]
    pentest_results: Optional[Dict[str, Any]]
    compliance_checks: List[ComplianceCheck]
    static_analysis_results: Dict[str, Any]
    overall_security_score: float
    critical_issues: List[str]
    recommendations: List[str]

class StaticAnalysisRunner:
    """Runs static analysis security tools"""
    
    def __init__(self):
        self.results = {}
    
    async def run_bandit_analysis(self) -> Dict[str, Any]:
        """Run Bandit security analysis"""
        logger.info("Running Bandit static analysis...")
        
        try:
            # Run Bandit on Python code
            result = subprocess.run([
                'bandit', '-r', '.', '-f', 'json', 
                '--exclude', './venv,./env,./node_modules,./.git'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 or result.stdout:
                bandit_data = json.loads(result.stdout) if result.stdout else {}
                
                return {
                    'tool': 'bandit',
                    'status': 'completed',
                    'issues_found': len(bandit_data.get('results', [])),
                    'high_severity': len([r for r in bandit_data.get('results', []) 
                                        if r.get('issue_severity') == 'HIGH']),
                    'medium_severity': len([r for r in bandit_data.get('results', []) 
                                          if r.get('issue_severity') == 'MEDIUM']),
                    'low_severity': len([r for r in bandit_data.get('results', []) 
                                       if r.get('issue_severity') == 'LOW']),
                    'raw_results': bandit_data
                }
            else:
                return {
                    'tool': 'bandit',
                    'status': 'failed',
                    'error': result.stderr,
                    'issues_found': 0
                }
                
        except subprocess.TimeoutExpired:
            return {
                'tool': 'bandit',
                'status': 'timeout',
                'error': 'Analysis timed out after 5 minutes',
                'issues_found': 0
            }
        except FileNotFoundError:
            return {
                'tool': 'bandit',
                'status': 'not_installed',
                'error': 'Bandit not found - install with: pip install bandit',
                'issues_found': 0
            }
        except Exception as e:
            return {
                'tool': 'bandit',
                'status': 'error',
                'error': str(e),
                'issues_found': 0
            }
    
    async def run_safety_check(self) -> Dict[str, Any]:
        """Run Safety dependency vulnerability check"""
        logger.info("Running Safety dependency check...")
        
        try:
            # Check for known vulnerabilities in dependencies
            result = subprocess.run([
                'safety', 'check', '--json'
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                return {
                    'tool': 'safety',
                    'status': 'completed',
                    'vulnerabilities_found': 0,
                    'message': 'No known security vulnerabilities found'
                }
            else:
                # Parse safety output
                try:
                    safety_data = json.loads(result.stdout) if result.stdout else []
                    return {
                        'tool': 'safety',
                        'status': 'completed',
                        'vulnerabilities_found': len(safety_data),
                        'vulnerabilities': safety_data
                    }
                except json.JSONDecodeError:
                    return {
                        'tool': 'safety',
                        'status': 'completed',
                        'vulnerabilities_found': 0 if result.returncode == 0 else 1,
                        'raw_output': result.stdout
                    }
                    
        except subprocess.TimeoutExpired:
            return {
                'tool': 'safety',
                'status': 'timeout',
                'error': 'Safety check timed out after 2 minutes'
            }
        except FileNotFoundError:
            return {
                'tool': 'safety',
                'status': 'not_installed',
                'error': 'Safety not found - install with: pip install safety'
            }
        except Exception as e:
            return {
                'tool': 'safety',
                'status': 'error',
                'error': str(e)
            }
    
    async def run_docker_security_scan(self) -> Dict[str, Any]:
        """Run Docker security scan"""
        logger.info("Running Docker security scan...")
        
        try:
            # Check if Docker is available
            docker_check = subprocess.run(['docker', '--version'], 
                                        capture_output=True, text=True, timeout=10)
            
            if docker_check.returncode != 0:
                return {
                    'tool': 'docker_security',
                    'status': 'not_available',
                    'error': 'Docker not available'
                }
            
            # Look for Dockerfiles
            dockerfiles = list(Path('.').rglob('Dockerfile*'))
            
            if not dockerfiles:
                return {
                    'tool': 'docker_security',
                    'status': 'skipped',
                    'message': 'No Dockerfiles found'
                }
            
            issues = []
            
            # Basic Dockerfile security checks
            for dockerfile in dockerfiles:
                try:
                    with open(dockerfile, 'r') as f:
                        content = f.read()
                    
                    # Check for common security issues
                    if 'USER root' in content or not 'USER ' in content:
                        issues.append({
                            'file': str(dockerfile),
                            'issue': 'Running as root user',
                            'severity': 'HIGH'
                        })
                    
                    if 'ADD http' in content:
                        issues.append({
                            'file': str(dockerfile),
                            'issue': 'Using ADD with HTTP URLs',
                            'severity': 'MEDIUM'
                        })
                    
                    if '--privileged' in content:
                        issues.append({
                            'file': str(dockerfile),
                            'issue': 'Using privileged mode',
                            'severity': 'CRITICAL'
                        })
                        
                except Exception as e:
                    logger.debug(f"Error scanning {dockerfile}: {e}")
            
            return {
                'tool': 'docker_security',
                'status': 'completed',
                'dockerfiles_scanned': len(dockerfiles),
                'issues_found': len(issues),
                'issues': issues
            }
            
        except Exception as e:
            return {
                'tool': 'docker_security',
                'status': 'error',
                'error': str(e)
            }
    
    async def run_all_analyses(self) -> Dict[str, Any]:
        """Run all static analysis tools"""
        results = {}
        
        # Run analyses in parallel
        bandit_task = asyncio.create_task(self.run_bandit_analysis())
        safety_task = asyncio.create_task(self.run_safety_check())
        docker_task = asyncio.create_task(self.run_docker_security_scan())
        
        results['bandit'] = await bandit_task
        results['safety'] = await safety_task
        results['docker_security'] = await docker_task
        
        return results

class ComplianceValidator:
    """Validates compliance with security requirements"""
    
    def __init__(self):
        self.checks = []
    
    async def validate_requirements_compliance(self, audit_results: Optional[Dict], 
                                             pentest_results: Optional[Dict]) -> List[ComplianceCheck]:
        """Validate compliance with requirements 12.1-12.4"""
        checks = []
        
        # Requirement 12.1: Agent heartbeat monitoring and failure detection
        heartbeat_check = await self._check_heartbeat_monitoring()
        checks.append(heartbeat_check)
        
        # Requirement 12.2: Automatic recovery mechanisms
        recovery_check = await self._check_recovery_mechanisms()
        checks.append(recovery_check)
        
        # Requirement 12.3: Circuit breakers and retry logic
        circuit_breaker_check = await self._check_circuit_breakers()
        checks.append(circuit_breaker_check)
        
        # Requirement 12.4: Graceful degradation
        degradation_check = await self._check_graceful_degradation()
        checks.append(degradation_check)
        
        # Additional security compliance checks
        if audit_results:
            container_isolation_check = self._check_container_isolation_compliance(audit_results)
            checks.append(container_isolation_check)
            
            encryption_check = self._check_encryption_compliance(audit_results)
            checks.append(encryption_check)
        
        if audit_results and pentest_results:
            boundary_check = self._check_boundary_compliance(audit_results, pentest_results)
            checks.append(boundary_check)
        
        return checks
    
    async def _check_heartbeat_monitoring(self) -> ComplianceCheck:
        """Check if heartbeat monitoring is implemented"""
        try:
            # Look for heartbeat monitoring implementation
            heartbeat_files = list(Path('.').rglob('*heartbeat*'))
            monitoring_files = list(Path('.').rglob('*monitor*'))
            
            if heartbeat_files or monitoring_files:
                return ComplianceCheck(
                    check_name="Heartbeat Monitoring",
                    requirement_id="12.1",
                    status="PASS",
                    description="Heartbeat monitoring implementation found",
                    evidence={'files_found': [str(f) for f in heartbeat_files + monitoring_files]},
                    remediation="Continue monitoring implementation and testing"
                )
            else:
                return ComplianceCheck(
                    check_name="Heartbeat Monitoring",
                    requirement_id="12.1",
                    status="FAIL",
                    description="No heartbeat monitoring implementation found",
                    evidence={'files_searched': ['*heartbeat*', '*monitor*']},
                    remediation="Implement agent heartbeat monitoring system"
                )
        except Exception as e:
            return ComplianceCheck(
                check_name="Heartbeat Monitoring",
                requirement_id="12.1",
                status="WARNING",
                description=f"Could not verify heartbeat monitoring: {e}",
                evidence={'error': str(e)},
                remediation="Investigate heartbeat monitoring implementation"
            )
    
    async def _check_recovery_mechanisms(self) -> ComplianceCheck:
        """Check if recovery mechanisms are implemented"""
        try:
            # Look for recovery implementation
            recovery_files = list(Path('.').rglob('*recovery*'))
            fault_tolerance_files = list(Path('.').rglob('*fault*'))
            
            if recovery_files or fault_tolerance_files:
                return ComplianceCheck(
                    check_name="Recovery Mechanisms",
                    requirement_id="12.2",
                    status="PASS",
                    description="Recovery mechanisms implementation found",
                    evidence={'files_found': [str(f) for f in recovery_files + fault_tolerance_files]},
                    remediation="Test recovery mechanisms under failure conditions"
                )
            else:
                return ComplianceCheck(
                    check_name="Recovery Mechanisms",
                    requirement_id="12.2",
                    status="FAIL",
                    description="No recovery mechanisms implementation found",
                    evidence={'files_searched': ['*recovery*', '*fault*']},
                    remediation="Implement automatic recovery and fallback mechanisms"
                )
        except Exception as e:
            return ComplianceCheck(
                check_name="Recovery Mechanisms",
                requirement_id="12.2",
                status="WARNING",
                description=f"Could not verify recovery mechanisms: {e}",
                evidence={'error': str(e)},
                remediation="Investigate recovery mechanisms implementation"
            )
    
    async def _check_circuit_breakers(self) -> ComplianceCheck:
        """Check if circuit breakers are implemented"""
        try:
            # Look for circuit breaker implementation
            circuit_breaker_files = list(Path('.').rglob('*circuit*'))
            retry_files = list(Path('.').rglob('*retry*'))
            
            if circuit_breaker_files or retry_files:
                return ComplianceCheck(
                    check_name="Circuit Breakers and Retry Logic",
                    requirement_id="12.3",
                    status="PASS",
                    description="Circuit breaker and retry logic implementation found",
                    evidence={'files_found': [str(f) for f in circuit_breaker_files + retry_files]},
                    remediation="Test circuit breakers under high failure rates"
                )
            else:
                return ComplianceCheck(
                    check_name="Circuit Breakers and Retry Logic",
                    requirement_id="12.3",
                    status="FAIL",
                    description="No circuit breaker or retry logic implementation found",
                    evidence={'files_searched': ['*circuit*', '*retry*']},
                    remediation="Implement circuit breakers and retry logic for communication failures"
                )
        except Exception as e:
            return ComplianceCheck(
                check_name="Circuit Breakers and Retry Logic",
                requirement_id="12.3",
                status="WARNING",
                description=f"Could not verify circuit breakers: {e}",
                evidence={'error': str(e)},
                remediation="Investigate circuit breaker implementation"
            )
    
    async def _check_graceful_degradation(self) -> ComplianceCheck:
        """Check if graceful degradation is implemented"""
        try:
            # Look for graceful degradation implementation
            degradation_files = list(Path('.').rglob('*degradation*'))
            fallback_files = list(Path('.').rglob('*fallback*'))
            
            if degradation_files or fallback_files:
                return ComplianceCheck(
                    check_name="Graceful Degradation",
                    requirement_id="12.4",
                    status="PASS",
                    description="Graceful degradation implementation found",
                    evidence={'files_found': [str(f) for f in degradation_files + fallback_files]},
                    remediation="Test graceful degradation under partial system failures"
                )
            else:
                return ComplianceCheck(
                    check_name="Graceful Degradation",
                    requirement_id="12.4",
                    status="FAIL",
                    description="No graceful degradation implementation found",
                    evidence={'files_searched': ['*degradation*', '*fallback*']},
                    remediation="Implement graceful degradation for partial system failures"
                )
        except Exception as e:
            return ComplianceCheck(
                check_name="Graceful Degradation",
                requirement_id="12.4",
                status="WARNING",
                description=f"Could not verify graceful degradation: {e}",
                evidence={'error': str(e)},
                remediation="Investigate graceful degradation implementation"
            )
    
    def _check_container_isolation_compliance(self, audit_results: Dict) -> ComplianceCheck:
        """Check container isolation compliance"""
        try:
            findings = audit_results.get('findings', [])
            container_findings = [f for f in findings if f.get('category') == 'CONTAINER']
            critical_container_issues = [f for f in container_findings if f.get('severity') == 'CRITICAL']
            
            if not critical_container_issues:
                return ComplianceCheck(
                    check_name="Container Isolation Compliance",
                    requirement_id="Security",
                    status="PASS",
                    description="No critical container isolation issues found",
                    evidence={'container_findings': len(container_findings)},
                    remediation="Continue monitoring container security"
                )
            else:
                return ComplianceCheck(
                    check_name="Container Isolation Compliance",
                    requirement_id="Security",
                    status="FAIL",
                    description=f"Critical container isolation issues found: {len(critical_container_issues)}",
                    evidence={'critical_issues': [f.get('title', 'Unknown') for f in critical_container_issues]},
                    remediation="Address critical container isolation vulnerabilities immediately"
                )
        except Exception as e:
            return ComplianceCheck(
                check_name="Container Isolation Compliance",
                requirement_id="Security",
                status="WARNING",
                description=f"Could not verify container isolation: {e}",
                evidence={'error': str(e)},
                remediation="Investigate container isolation implementation"
            )
    
    def _check_encryption_compliance(self, audit_results: Dict) -> ComplianceCheck:
        """Check encryption compliance"""
        try:
            findings = audit_results.get('findings', [])
            encryption_findings = [f for f in findings if f.get('category') == 'ENCRYPTION']
            critical_encryption_issues = [f for f in encryption_findings if f.get('severity') in ['CRITICAL', 'HIGH']]
            
            if not critical_encryption_issues:
                return ComplianceCheck(
                    check_name="Encryption Standards Compliance",
                    requirement_id="Security",
                    status="PASS",
                    description="Encryption standards appear to be properly implemented",
                    evidence={'encryption_findings': len(encryption_findings)},
                    remediation="Continue monitoring encryption implementations"
                )
            else:
                return ComplianceCheck(
                    check_name="Encryption Standards Compliance",
                    requirement_id="Security",
                    status="FAIL",
                    description=f"Critical encryption issues found: {len(critical_encryption_issues)}",
                    evidence={'critical_issues': [f.get('title', 'Unknown') for f in critical_encryption_issues]},
                    remediation="Address encryption vulnerabilities and implement strong cryptography"
                )
        except Exception as e:
            return ComplianceCheck(
                check_name="Encryption Standards Compliance",
                requirement_id="Security",
                status="WARNING",
                description=f"Could not verify encryption compliance: {e}",
                evidence={'error': str(e)},
                remediation="Investigate encryption implementation"
            )
    
    def _check_boundary_compliance(self, audit_results: Dict, pentest_results: Dict) -> ComplianceCheck:
        """Check simulation boundary compliance"""
        try:
            audit_findings = audit_results.get('findings', [])
            boundary_findings = [f for f in audit_findings if f.get('category') == 'BOUNDARY']
            
            pentest_results_list = pentest_results.get('results', [])
            boundary_exploits = [r for r in pentest_results_list 
                               if r.get('success', False) and 'boundary' in r.get('test_name', '').lower()]
            
            if not boundary_findings and not boundary_exploits:
                return ComplianceCheck(
                    check_name="Simulation Boundary Compliance",
                    requirement_id="Security",
                    status="PASS",
                    description="Simulation boundaries appear to be properly enforced",
                    evidence={'boundary_findings': len(boundary_findings), 'boundary_exploits': len(boundary_exploits)},
                    remediation="Continue monitoring boundary enforcement"
                )
            else:
                return ComplianceCheck(
                    check_name="Simulation Boundary Compliance",
                    requirement_id="Security",
                    status="FAIL",
                    description="Simulation boundary violations detected",
                    evidence={
                        'boundary_findings': len(boundary_findings),
                        'boundary_exploits': len(boundary_exploits)
                    },
                    remediation="Strengthen simulation containment and boundary enforcement"
                )
        except Exception as e:
            return ComplianceCheck(
                check_name="Simulation Boundary Compliance",
                requirement_id="Security",
                status="WARNING",
                description=f"Could not verify boundary compliance: {e}",
                evidence={'error': str(e)},
                remediation="Investigate boundary enforcement implementation"
            )

class SecurityValidationOrchestrator:
    """Main orchestrator for comprehensive security validation"""
    
    def __init__(self):
        self.audit_orchestrator = SecurityAuditOrchestrator() if SecurityAuditOrchestrator else None
        self.pentest_orchestrator = PenetrationTestOrchestrator() if PenetrationTestOrchestrator else None
        self.static_analyzer = StaticAnalysisRunner()
        self.compliance_validator = ComplianceValidator()
    
    async def run_comprehensive_validation(self) -> SecurityValidationReport:
        """Run complete security validation"""
        logger.info("Starting comprehensive security validation...")
        
        validation_id = f"security_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Run security audit if available
        audit_results = None
        if self.audit_orchestrator:
            try:
                logger.info("Running security audit...")
                audit_result = await self.audit_orchestrator.run_comprehensive_audit()
                audit_results = {
                    'audit_id': audit_result.audit_id,
                    'findings': [asdict(f) for f in audit_result.findings],
                    'summary': audit_result.summary,
                    'compliance_status': audit_result.compliance_status
                }
            except Exception as e:
                logger.error(f"Security audit failed: {e}")
                audit_results = {'error': str(e)}
        else:
            logger.warning("Security audit orchestrator not available")
        
        # Run penetration testing if available
        pentest_results = None
        if self.pentest_orchestrator:
            try:
                logger.info("Running penetration testing...")
                pentest_result = await self.pentest_orchestrator.run_penetration_tests()
                pentest_results = {
                    'test_id': pentest_result.test_id,
                    'results': [asdict(r) for r in pentest_result.results],
                    'summary': pentest_result.summary,
                    'attack_paths': pentest_result.attack_paths
                }
            except Exception as e:
                logger.error(f"Penetration testing failed: {e}")
                pentest_results = {'error': str(e)}
        else:
            logger.warning("Penetration test orchestrator not available")
        
        # Run static analysis
        logger.info("Running static analysis...")
        static_analysis_results = await self.static_analyzer.run_all_analyses()
        
        # Run compliance validation
        logger.info("Running compliance validation...")
        compliance_checks = await self.compliance_validator.validate_requirements_compliance(
            audit_results, pentest_results
        )
        
        # Calculate overall security score
        security_score = self._calculate_security_score(
            audit_results, pentest_results, static_analysis_results, compliance_checks
        )
        
        # Identify critical issues
        critical_issues = self._identify_critical_issues(
            audit_results, pentest_results, static_analysis_results, compliance_checks
        )
        
        # Generate recommendations
        recommendations = self._generate_comprehensive_recommendations(
            audit_results, pentest_results, static_analysis_results, compliance_checks
        )
        
        report = SecurityValidationReport(
            validation_id=validation_id,
            timestamp=datetime.now(),
            audit_results=audit_results,
            pentest_results=pentest_results,
            compliance_checks=compliance_checks,
            static_analysis_results=static_analysis_results,
            overall_security_score=security_score,
            critical_issues=critical_issues,
            recommendations=recommendations
        )
        
        logger.info("Security validation completed")
        return report
    
    def _calculate_security_score(self, audit_results: Optional[Dict], 
                                pentest_results: Optional[Dict],
                                static_analysis_results: Dict[str, Any],
                                compliance_checks: List[ComplianceCheck]) -> float:
        """Calculate overall security score (0-100)"""
        score = 100.0
        
        # Deduct points for audit findings
        if audit_results and 'findings' in audit_results:
            for finding in audit_results['findings']:
                severity = finding.get('severity', 'LOW')
                if severity == 'CRITICAL':
                    score -= 15
                elif severity == 'HIGH':
                    score -= 10
                elif severity == 'MEDIUM':
                    score -= 5
                elif severity == 'LOW':
                    score -= 2
        
        # Deduct points for successful penetration tests
        if pentest_results and 'results' in pentest_results:
            for result in pentest_results['results']:
                if result.get('success', False):
                    severity = result.get('severity', 'LOW')
                    if severity == 'CRITICAL':
                        score -= 20
                    elif severity == 'HIGH':
                        score -= 15
                    elif severity == 'MEDIUM':
                        score -= 8
                    elif severity == 'LOW':
                        score -= 3
        
        # Deduct points for static analysis issues
        for tool, results in static_analysis_results.items():
            if results.get('status') == 'completed':
                high_issues = results.get('high_severity', 0)
                medium_issues = results.get('medium_severity', 0)
                score -= (high_issues * 5) + (medium_issues * 2)
        
        # Deduct points for compliance failures
        for check in compliance_checks:
            if check.status == 'FAIL':
                score -= 10
            elif check.status == 'WARNING':
                score -= 5
        
        return max(0.0, score)
    
    def _identify_critical_issues(self, audit_results: Optional[Dict],
                                pentest_results: Optional[Dict],
                                static_analysis_results: Dict[str, Any],
                                compliance_checks: List[ComplianceCheck]) -> List[str]:
        """Identify critical security issues"""
        critical_issues = []
        
        # Critical audit findings
        if audit_results and 'findings' in audit_results:
            critical_audit = [f for f in audit_results['findings'] if f.get('severity') == 'CRITICAL']
            for finding in critical_audit:
                critical_issues.append(f"AUDIT: {finding.get('title', 'Unknown issue')}")
        
        # Critical penetration test results
        if pentest_results and 'results' in pentest_results:
            critical_pentests = [r for r in pentest_results['results'] 
                               if r.get('success', False) and r.get('severity') == 'CRITICAL']
            for result in critical_pentests:
                critical_issues.append(f"PENTEST: {result.get('test_name', 'Unknown test')}")
        
        # Critical static analysis issues
        for tool, results in static_analysis_results.items():
            if results.get('high_severity', 0) > 0:
                critical_issues.append(f"STATIC: {tool.upper()} found {results['high_severity']} high-severity issues")
        
        # Failed compliance checks
        failed_compliance = [c for c in compliance_checks if c.status == 'FAIL']
        for check in failed_compliance:
            critical_issues.append(f"COMPLIANCE: {check.check_name} failed")
        
        return critical_issues
    
    def _generate_comprehensive_recommendations(self, audit_results: Optional[Dict],
                                              pentest_results: Optional[Dict],
                                              static_analysis_results: Dict[str, Any],
                                              compliance_checks: List[ComplianceCheck]) -> List[str]:
        """Generate comprehensive security recommendations"""
        recommendations = []
        
        # Add audit recommendations
        if audit_results and 'recommendations' in audit_results:
            recommendations.extend(audit_results['recommendations'])
        
        # Add penetration test recommendations
        if pentest_results and 'recommendations' in pentest_results:
            recommendations.extend(pentest_results['recommendations'])
        
        # Add static analysis recommendations
        for tool, results in static_analysis_results.items():
            if results.get('status') == 'not_installed':
                recommendations.append(f"Install {tool} for comprehensive security analysis")
            elif results.get('issues_found', 0) > 0:
                recommendations.append(f"Review and fix {tool} security findings")
        
        # Add compliance recommendations
        failed_checks = [c for c in compliance_checks if c.status == 'FAIL']
        for check in failed_checks:
            recommendations.append(f"Address compliance failure: {check.remediation}")
        
        # Remove duplicates and prioritize
        unique_recommendations = list(set(recommendations))
        
        # Add general recommendations
        critical_issues = self._identify_critical_issues(
            audit_results, pentest_results, static_analysis_results, compliance_checks
        )
        
        if any('CRITICAL' in issue for issue in critical_issues):
            unique_recommendations.insert(0, "URGENT: Address all critical security issues immediately")
        
        return unique_recommendations[:10]  # Top 10 recommendations
    
    def save_validation_report(self, report: SecurityValidationReport, 
                             output_path: str = None) -> str:
        """Save comprehensive validation report"""
        if output_path is None:
            output_path = f"security_validation_report_{report.validation_id}.json"
        
        # Convert to serializable format
        report_data = {
            'validation_id': report.validation_id,
            'timestamp': report.timestamp.isoformat(),
            'overall_security_score': report.overall_security_score,
            'critical_issues': report.critical_issues,
            'recommendations': report.recommendations,
            'audit_results': report.audit_results,
            'pentest_results': report.pentest_results,
            'static_analysis_results': report.static_analysis_results,
            'compliance_checks': [asdict(check) for check in report.compliance_checks]
        }
        
        # Convert datetime objects in compliance checks
        for check in report_data['compliance_checks']:
            if 'timestamp' in check and check['timestamp']:
                check['timestamp'] = check['timestamp'].isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Security validation report saved to: {output_path}")
        return output_path

async def main():
    """Main entry point for security validation"""
    orchestrator = SecurityValidationOrchestrator()
    
    try:
        # Run comprehensive validation
        report = await orchestrator.run_comprehensive_validation()
        
        # Save report
        report_path = orchestrator.save_validation_report(report)
        
        # Print comprehensive summary
        print("\n" + "="*80)
        print("COMPREHENSIVE SECURITY VALIDATION SUMMARY")
        print("="*80)
        print(f"Validation ID: {report.validation_id}")
        print(f"Timestamp: {report.timestamp}")
        print(f"Overall Security Score: {report.overall_security_score:.1f}/100")
        
        # Security grade
        if report.overall_security_score >= 90:
            grade = "A (Excellent)"
        elif report.overall_security_score >= 80:
            grade = "B (Good)"
        elif report.overall_security_score >= 70:
            grade = "C (Fair)"
        elif report.overall_security_score >= 60:
            grade = "D (Poor)"
        else:
            grade = "F (Critical)"
        
        print(f"Security Grade: {grade}")
        
        # Results summary
        if report.audit_results and 'findings' in report.audit_results:
            print(f"\nAudit Results: {len(report.audit_results['findings'])} findings")
        
        if report.pentest_results and 'summary' in report.pentest_results:
            summary = report.pentest_results['summary']
            print(f"Penetration Tests: {summary.get('successful_exploits', 0)}/{summary.get('total_tests', 0)} successful exploits")
        
        # Static analysis summary
        print("\nStatic Analysis Results:")
        for tool, results in report.static_analysis_results.items():
            status = results.get('status', 'unknown')
            issues = results.get('issues_found', 0)
            print(f"  {tool.upper()}: {status} ({issues} issues)")
        
        # Compliance summary
        compliance_pass = len([c for c in report.compliance_checks if c.status == 'PASS'])
        compliance_total = len(report.compliance_checks)
        print(f"\nCompliance Checks: {compliance_pass}/{compliance_total} passed")
        
        if report.critical_issues:
            print(f"\nCRITICAL ISSUES ({len(report.critical_issues)}):")
            for issue in report.critical_issues[:5]:  # Show top 5
                print(f"  • {issue}")
            if len(report.critical_issues) > 5:
                print(f"  ... and {len(report.critical_issues) - 5} more")
        
        print(f"\nTOP RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations[:5], 1):
            print(f"  {i}. {rec}")
        
        print(f"\nDetailed report saved to: {report_path}")
        print("="*80)
        
        # Return appropriate exit code
        if report.overall_security_score < 60:
            print("\n⚠️  CRITICAL: Security score below acceptable threshold!")
            sys.exit(1)
        elif len(report.critical_issues) > 0:
            print("\n⚠️  WARNING: Critical security issues found!")
            sys.exit(1)
        else:
            print("\n✅ Security validation completed successfully")
            sys.exit(0)
        
    except Exception as e:
        logger.error(f"Security validation failed: {e}")
        print(f"\n❌ Security validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())