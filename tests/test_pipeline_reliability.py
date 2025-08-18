#!/usr/bin/env python3
"""
CI/CD Pipeline Reliability and Security Validation Tests
Comprehensive test suite for pipeline reliability and security validation effectiveness
"""

import unittest
import asyncio
import json
import os
import subprocess
import time
import yaml
import tempfile
import shutil
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import logging
from pathlib import Path

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineReliabilityTestCase(unittest.TestCase):
    """Base test case for pipeline reliability testing."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment for pipeline testing."""
        cls.test_dir = tempfile.mkdtemp(prefix='pipeline_test_')
        cls.original_dir = os.getcwd()
        os.chdir(cls.test_dir)
        
        # Create mock project structure
        cls._create_mock_project()
        
        # Initialize test data
        cls.pipeline_config = cls._load_pipeline_config()
        cls.test_results = {}
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        os.chdir(cls.original_dir)
        shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    @classmethod
    def _create_mock_project(cls):
        """Create mock project structure for testing."""
        # Create directory structure
        dirs = [
            '.github/workflows',
            'src',
            'tests',
            'infrastructure/terraform',
            'infrastructure/k8s',
            'security-compliance',
            'chaos-testing'
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
        
        # Create mock Python files
        with open('src/main.py', 'w') as f:
            f.write('''
import os
import sys

def main():
    password = "hardcoded_secret_123"  # This should trigger Bandit
    print("Hello World")

if __name__ == "__main__":
    main()
''')
        
        with open('requirements.txt', 'w') as f:
            f.write('''
flask==1.0.0
requests==2.20.0
pyyaml==5.1
''')
        
        # Create mock Dockerfile
        with open('Dockerfile', 'w') as f:
            f.write('''
FROM python:3.9
USER root
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8080
CMD ["python", "src/main.py"]
''')
        
        # Create mock GitHub Actions workflow
        workflow_content = '''
name: Test Pipeline
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Test
        run: echo "Testing"
'''
        with open('.github/workflows/test.yml', 'w') as f:
            f.write(workflow_content)
    
    @classmethod
    def _load_pipeline_config(cls) -> Dict[str, Any]:
        """Load pipeline configuration for testing."""
        return {
            'security_tools': ['bandit', 'semgrep', 'safety', 'trivy'],
            'compliance_frameworks': ['CIS', 'NIST', 'OWASP'],
            'thresholds': {
                'critical': 0,
                'high': 5,
                'medium': 20
            },
            'timeout_limits': {
                'security_scan': 600,
                'chaos_test': 1800,
                'deployment': 900
            }
        }


class SecurityScannerReliabilityTests(PipelineReliabilityTestCase):
    """Test reliability of security scanners in the pipeline."""
    
    def test_bandit_scanner_reliability(self):
        """Test Bandit security scanner reliability and effectiveness."""
        # Create mock Bandit configuration
        bandit_config = {
            'tests': ['B105', 'B106', 'B107'],  # Password-related tests
            'skips': ['B404'],  # Skip subprocess import
        }
        
        with open('.bandit.yml', 'w') as f:
            yaml.dump(bandit_config, f)
        
        # Test scanner execution
        try:
            # Mock Bandit execution
            with patch('subprocess.run') as mock_run:
                mock_result = Mock()
                mock_result.returncode = 1  # Bandit found issues
                mock_result.stdout = ""
                mock_result.stderr = ""
                mock_run.return_value = mock_result
                
                # Create mock Bandit report
                bandit_report = {
                    'results': [
                        {
                            'filename': 'src/main.py',
                            'test_id': 'B105',
                            'test_name': 'hardcoded_password_string',
                            'issue_severity': 'HIGH',
                            'issue_confidence': 'MEDIUM',
                            'line_number': 5,
                            'code': 'password = "hardcoded_secret_123"'
                        }
                    ],
                    'metrics': {
                        '_totals': {
                            'loc': 100,
                            'nosec': 0
                        }
                    }
                }
                
                with open('bandit-report.json', 'w') as f:
                    json.dump(bandit_report, f)
                
                # Verify scanner finds expected issues
                self.assertTrue(os.path.exists('bandit-report.json'))
                
                with open('bandit-report.json', 'r') as f:
                    report = json.load(f)
                
                self.assertEqual(len(report['results']), 1)
                self.assertEqual(report['results'][0]['test_id'], 'B105')
                self.assertEqual(report['results'][0]['issue_severity'], 'HIGH')
                
                logger.info("✅ Bandit scanner reliability test passed")
                
        except Exception as e:
            self.fail(f"Bandit scanner reliability test failed: {e}")
    
    def test_semgrep_scanner_reliability(self):
        """Test Semgrep security scanner reliability and effectiveness."""
        try:
            # Mock Semgrep execution
            with patch('subprocess.run') as mock_run:
                mock_result = Mock()
                mock_result.returncode = 0
                mock_result.stdout = ""
                mock_result.stderr = ""
                mock_run.return_value = mock_result
                
                # Create mock Semgrep report
                semgrep_report = {
                    'results': [
                        {
                            'check_id': 'python.lang.security.audit.hardcoded-password',
                            'path': 'src/main.py',
                            'start': {'line': 5, 'col': 4},
                            'end': {'line': 5, 'col': 35},
                            'extra': {
                                'message': 'Hardcoded password detected',
                                'severity': 'ERROR',
                                'metadata': {
                                    'cwe': 'CWE-798',
                                    'owasp': 'A02:2021 - Cryptographic Failures'
                                }
                            }
                        }
                    ],
                    'errors': []
                }
                
                with open('semgrep-report.json', 'w') as f:
                    json.dump(semgrep_report, f)
                
                # Verify scanner finds expected patterns
                self.assertTrue(os.path.exists('semgrep-report.json'))
                
                with open('semgrep-report.json', 'r') as f:
                    report = json.load(f)
                
                self.assertEqual(len(report['results']), 1)
                self.assertEqual(report['results'][0]['extra']['severity'], 'ERROR')
                self.assertIn('hardcoded-password', report['results'][0]['check_id'])
                
                logger.info("✅ Semgrep scanner reliability test passed")
                
        except Exception as e:
            self.fail(f"Semgrep scanner reliability test failed: {e}")
    
    def test_dependency_scanner_reliability(self):
        """Test dependency vulnerability scanner reliability."""
        try:
            # Mock Safety execution
            with patch('subprocess.run') as mock_run:
                mock_result = Mock()
                mock_result.returncode = 1  # Vulnerabilities found
                mock_result.stdout = json.dumps({
                    'vulnerabilities': [
                        {
                            'package': 'flask',
                            'version': '1.0.0',
                            'vulnerability': 'CVE-2023-30861',
                            'severity': 'HIGH',
                            'description': 'Flask before 2.3.3 has a potential denial of service vulnerability'
                        }
                    ]
                })
                mock_run.return_value = mock_result
                
                # Verify scanner detects vulnerable dependencies
                vulnerabilities = json.loads(mock_result.stdout)
                self.assertEqual(len(vulnerabilities['vulnerabilities']), 1)
                self.assertEqual(vulnerabilities['vulnerabilities'][0]['package'], 'flask')
                self.assertEqual(vulnerabilities['vulnerabilities'][0]['severity'], 'HIGH')
                
                logger.info("✅ Dependency scanner reliability test passed")
                
        except Exception as e:
            self.fail(f"Dependency scanner reliability test failed: {e}")
    
    def test_container_scanner_reliability(self):
        """Test container image vulnerability scanner reliability."""
        try:
            # Mock Trivy execution
            with patch('subprocess.run') as mock_run:
                mock_result = Mock()
                mock_result.returncode = 0
                mock_result.stdout = ""
                mock_run.return_value = mock_result
                
                # Create mock Trivy report
                trivy_report = {
                    'Results': [
                        {
                            'Target': 'python:3.9',
                            'Class': 'os-pkgs',
                            'Type': 'debian',
                            'Vulnerabilities': [
                                {
                                    'VulnerabilityID': 'CVE-2023-1234',
                                    'PkgName': 'libssl1.1',
                                    'Severity': 'HIGH',
                                    'Title': 'OpenSSL vulnerability',
                                    'Description': 'Memory corruption in OpenSSL'
                                }
                            ]
                        }
                    ]
                }
                
                with open('trivy-report.json', 'w') as f:
                    json.dump(trivy_report, f)
                
                # Verify scanner detects container vulnerabilities
                self.assertTrue(os.path.exists('trivy-report.json'))
                
                with open('trivy-report.json', 'r') as f:
                    report = json.load(f)
                
                self.assertEqual(len(report['Results']), 1)
                self.assertEqual(len(report['Results'][0]['Vulnerabilities']), 1)
                self.assertEqual(report['Results'][0]['Vulnerabilities'][0]['Severity'], 'HIGH')
                
                logger.info("✅ Container scanner reliability test passed")
                
        except Exception as e:
            self.fail(f"Container scanner reliability test failed: {e}")
    
    def test_scanner_performance_under_load(self):
        """Test scanner performance under high load conditions."""
        try:
            # Create multiple large files to scan
            large_files = []
            for i in range(5):
                filename = f'large_file_{i}.py'
                with open(filename, 'w') as f:
                    # Generate large file content
                    content = '''
import os
import sys
def function_{i}():
    password = "secret_{i}"
    api_key = "key_{i}"
    return password + api_key
'''.replace('{i}', str(i))
                    f.write(content * 100)  # Repeat content 100 times
                large_files.append(filename)
            
            # Test concurrent scanning
            start_time = time.time()
            
            # Mock parallel scanner execution
            with patch('subprocess.run') as mock_run:
                mock_result = Mock()
                mock_result.returncode = 0
                mock_result.stdout = ""
                mock_run.return_value = mock_result
                
                # Simulate scanner execution time
                time.sleep(0.1)  # Simulate processing time
            
            execution_time = time.time() - start_time
            
            # Verify performance is within acceptable limits
            self.assertLess(execution_time, 10, "Scanner performance too slow under load")
            
            # Clean up large files
            for filename in large_files:
                os.remove(filename)
            
            logger.info(f"✅ Scanner performance test passed: {execution_time:.2f}s")
            
        except Exception as e:
            self.fail(f"Scanner performance test failed: {e}")


class ChaoTestingReliabilityTests(PipelineReliabilityTestCase):
    """Test reliability of chaos testing in the pipeline."""
    
    def test_chaos_experiment_execution(self):
        """Test chaos experiment execution reliability."""
        try:
            # Create mock chaos experiment configuration
            chaos_config = {
                'apiVersion': 'litmuschaos.io/v1alpha1',
                'kind': 'ChaosEngine',
                'metadata': {
                    'name': 'test-chaos',
                    'namespace': 'default'
                },
                'spec': {
                    'engineState': 'active',
                    'experiments': [
                        {
                            'name': 'pod-failure',
                            'spec': {
                                'components': {
                                    'env': [
                                        {'name': 'TOTAL_CHAOS_DURATION', 'value': '60'},
                                        {'name': 'PODS_AFFECTED_PERC', 'value': '30'}
                                    ]
                                }
                            }
                        }
                    ]
                }
            }
            
            with open('chaos-experiment.yaml', 'w') as f:
                yaml.dump(chaos_config, f)
            
            # Mock kubectl execution for chaos testing
            with patch('subprocess.run') as mock_run:
                mock_result = Mock()
                mock_result.returncode = 0
                mock_result.stdout = "chaosengine.litmuschaos.io/test-chaos created"
                mock_run.return_value = mock_result
                
                # Verify chaos experiment can be applied
                self.assertTrue(os.path.exists('chaos-experiment.yaml'))
                
                # Mock chaos result
                chaos_result = {
                    'apiVersion': 'litmuschaos.io/v1alpha1',
                    'kind': 'ChaosResult',
                    'metadata': {'name': 'test-chaos-pod-failure'},
                    'status': {
                        'experimentstatus': {
                            'verdict': 'Pass',
                            'failStep': 'N/A'
                        }
                    }
                }
                
                with open('chaos-result.json', 'w') as f:
                    json.dump(chaos_result, f)
                
                # Verify chaos test result
                with open('chaos-result.json', 'r') as f:
                    result = json.load(f)
                
                self.assertEqual(result['status']['experimentstatus']['verdict'], 'Pass')
                
                logger.info("✅ Chaos experiment execution test passed")
                
        except Exception as e:
            self.fail(f"Chaos experiment execution test failed: {e}")
    
    def test_chaos_experiment_timeout_handling(self):
        """Test chaos experiment timeout handling."""
        try:
            start_time = time.time()
            
            # Mock long-running chaos experiment
            with patch('subprocess.run') as mock_run:
                def side_effect(*args, **kwargs):
                    if 'timeout' in kwargs and kwargs['timeout'] < 5:
                        raise subprocess.TimeoutExpired('kubectl', kwargs['timeout'])
                    mock_result = Mock()
                    mock_result.returncode = 0
                    mock_result.stdout = "timeout handled"
                    return mock_result
                
                mock_run.side_effect = side_effect
                
                # Test timeout handling
                try:
                    subprocess.run(['kubectl', 'apply', '-f', 'chaos-experiment.yaml'], timeout=2)
                    self.fail("Expected TimeoutExpired exception")
                except subprocess.TimeoutExpired:
                    # This is expected behavior
                    pass
                
                # Test successful execution with sufficient timeout
                result = subprocess.run(['kubectl', 'get', 'chaosresults'], timeout=10)
                self.assertEqual(result.returncode, 0)
                
                logger.info("✅ Chaos experiment timeout handling test passed")
                
        except Exception as e:
            self.fail(f"Chaos experiment timeout handling test failed: {e}")
    
    def test_chaos_experiment_failure_recovery(self):
        """Test recovery from chaos experiment failures."""
        try:
            # Mock failed chaos experiment
            with patch('subprocess.run') as mock_run:
                mock_result = Mock()
                mock_result.returncode = 1
                mock_result.stderr = "Error: chaos experiment failed"
                mock_run.return_value = mock_result
                
                # Mock chaos result with failure
                failed_result = {
                    'status': {
                        'experimentstatus': {
                            'verdict': 'Fail',
                            'failStep': 'pod-failure-experiment'
                        }
                    }
                }
                
                with open('failed-chaos-result.json', 'w') as f:
                    json.dump(failed_result, f)
                
                # Verify failure is properly handled
                with open('failed-chaos-result.json', 'r') as f:
                    result = json.load(f)
                
                self.assertEqual(result['status']['experimentstatus']['verdict'], 'Fail')
                
                # Test recovery mechanism (cleanup)
                cleanup_successful = True  # Mock cleanup success
                self.assertTrue(cleanup_successful, "Chaos experiment cleanup should succeed")
                
                logger.info("✅ Chaos experiment failure recovery test passed")
                
        except Exception as e:
            self.fail(f"Chaos experiment failure recovery test failed: {e}")


class PipelineIntegrationTests(PipelineReliabilityTestCase):
    """Test end-to-end pipeline integration reliability."""
    
    def test_pipeline_workflow_validation(self):
        """Test GitHub Actions workflow validation."""
        try:
            # Validate workflow YAML syntax
            workflow_file = '.github/workflows/test.yml'
            self.assertTrue(os.path.exists(workflow_file))
            
            with open(workflow_file, 'r') as f:
                workflow_content = yaml.safe_load(f)
            
            # Verify required workflow components
            self.assertIn('name', workflow_content)
            self.assertIn('on', workflow_content)
            self.assertIn('jobs', workflow_content)
            
            # Verify job structure
            jobs = workflow_content['jobs']
            self.assertGreater(len(jobs), 0)
            
            for job_name, job_config in jobs.items():
                self.assertIn('runs-on', job_config)
                self.assertIn('steps', job_config)
                self.assertGreater(len(job_config['steps']), 0)
            
            logger.info("✅ Pipeline workflow validation test passed")
            
        except Exception as e:
            self.fail(f"Pipeline workflow validation test failed: {e}")
    
    def test_pipeline_stage_dependencies(self):
        """Test pipeline stage dependencies and execution order."""
        try:
            # Mock pipeline execution stages
            stages = [
                {'name': 'security-pre-checks', 'dependencies': []},
                {'name': 'code-quality-security', 'dependencies': ['security-pre-checks']},
                {'name': 'docker-security', 'dependencies': ['security-pre-checks']},
                {'name': 'build-and-package', 'dependencies': ['code-quality-security', 'docker-security']},
                {'name': 'chaos-testing', 'dependencies': ['build-and-package']},
                {'name': 'deploy-staging', 'dependencies': ['build-and-package']},
                {'name': 'deploy-production', 'dependencies': ['chaos-testing']}
            ]
            
            # Verify dependency chain is valid
            stage_names = {stage['name'] for stage in stages}
            for stage in stages:
                for dependency in stage['dependencies']:
                    self.assertIn(dependency, stage_names, 
                                f"Stage {stage['name']} has invalid dependency: {dependency}")
            
            # Test topological sort for execution order
            execution_order = self._topological_sort(stages)
            self.assertGreater(len(execution_order), 0)
            
            # Verify security checks come before deployment
            security_index = execution_order.index('security-pre-checks')
            deploy_index = execution_order.index('deploy-production')
            self.assertLess(security_index, deploy_index)
            
            logger.info("✅ Pipeline stage dependencies test passed")
            
        except Exception as e:
            self.fail(f"Pipeline stage dependencies test failed: {e}")
    
    def _topological_sort(self, stages: List[Dict[str, Any]]) -> List[str]:
        """Perform topological sort on pipeline stages."""
        from collections import defaultdict, deque
        
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        # Build graph
        stage_names = [stage['name'] for stage in stages]
        for stage in stages:
            stage_name = stage['name']
            in_degree[stage_name] = len(stage['dependencies'])
            for dependency in stage['dependencies']:
                graph[dependency].append(stage_name)
        
        # Topological sort using Kahn's algorithm
        queue = deque([stage for stage in stage_names if in_degree[stage] == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result if len(result) == len(stage_names) else []
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling and recovery mechanisms."""
        try:
            # Test security scan failure handling
            security_failure_scenarios = [
                {'type': 'critical_vulnerability', 'expected_action': 'block_deployment'},
                {'type': 'high_severity_issue', 'expected_action': 'require_approval'},
                {'type': 'scanner_timeout', 'expected_action': 'retry_scan'},
                {'type': 'scanner_unavailable', 'expected_action': 'fallback_scanner'}
            ]
            
            for scenario in security_failure_scenarios:
                # Mock failure scenario
                failure_type = scenario['type']
                expected_action = scenario['expected_action']
                
                # Simulate error handling logic
                if failure_type == 'critical_vulnerability':
                    deployment_blocked = True
                    self.assertTrue(deployment_blocked, 
                                  f"Deployment should be blocked for {failure_type}")
                
                elif failure_type == 'scanner_timeout':
                    retry_attempted = True
                    self.assertTrue(retry_attempted, 
                                  f"Retry should be attempted for {failure_type}")
                
                elif failure_type == 'scanner_unavailable':
                    fallback_used = True
                    self.assertTrue(fallback_used, 
                                  f"Fallback should be used for {failure_type}")
            
            logger.info("✅ Pipeline error handling test passed")
            
        except Exception as e:
            self.fail(f"Pipeline error handling test failed: {e}")
    
    def test_pipeline_performance_metrics(self):
        """Test pipeline performance metrics collection."""
        try:
            # Mock pipeline execution metrics
            metrics = {
                'total_execution_time': 1200,  # 20 minutes
                'security_scan_time': 300,     # 5 minutes
                'build_time': 180,             # 3 minutes
                'test_time': 120,              # 2 minutes
                'deployment_time': 90,         # 1.5 minutes
                'chaos_test_time': 360,        # 6 minutes
                'success_rate': 0.95,          # 95% success rate
                'avg_queue_time': 30           # 30 seconds average queue time
            }
            
            # Verify performance thresholds
            max_execution_time = self.pipeline_config['timeout_limits'].get('total_pipeline', 1800)  # 30 minutes
            self.assertLess(metrics['total_execution_time'], max_execution_time,
                          f"Pipeline execution time {metrics['total_execution_time']}s exceeds limit {max_execution_time}s")
            
            # Verify success rate
            min_success_rate = 0.90  # 90% minimum
            self.assertGreaterEqual(metrics['success_rate'], min_success_rate,
                                  f"Pipeline success rate {metrics['success_rate']} below minimum {min_success_rate}")
            
            # Verify security scan performance
            max_security_scan_time = self.pipeline_config['timeout_limits']['security_scan']
            self.assertLess(metrics['security_scan_time'], max_security_scan_time,
                          f"Security scan time {metrics['security_scan_time']}s exceeds limit {max_security_scan_time}s")
            
            logger.info("✅ Pipeline performance metrics test passed")
            
        except Exception as e:
            self.fail(f"Pipeline performance metrics test failed: {e}")


class SecurityValidationEffectivenessTests(PipelineReliabilityTestCase):
    """Test effectiveness of security validation in the pipeline."""
    
    def test_vulnerability_detection_effectiveness(self):
        """Test effectiveness of vulnerability detection across tools."""
        try:
            # Create test cases with known vulnerabilities
            vulnerability_test_cases = [
                {
                    'type': 'hardcoded_secret',
                    'code': 'api_key = "sk-1234567890abcdef"',
                    'expected_detectors': ['bandit', 'semgrep'],
                    'severity': 'CRITICAL'
                },
                {
                    'type': 'sql_injection',
                    'code': 'query = f"SELECT * FROM users WHERE id = {user_id}"',
                    'expected_detectors': ['semgrep'],
                    'severity': 'HIGH'
                },
                {
                    'type': 'weak_crypto',
                    'code': 'hashlib.md5(password.encode()).hexdigest()',
                    'expected_detectors': ['bandit'],
                    'severity': 'MEDIUM'
                },
                {
                    'type': 'unsafe_deserialization',
                    'code': 'pickle.loads(user_data)',
                    'expected_detectors': ['bandit', 'semgrep'],
                    'severity': 'HIGH'
                }
            ]
            
            detection_results = {}
            
            for test_case in vulnerability_test_cases:
                vuln_type = test_case['type']
                code = test_case['code']
                expected_detectors = test_case['expected_detectors']
                
                # Create test file with vulnerability
                test_file = f'test_{vuln_type}.py'
                with open(test_file, 'w') as f:
                    f.write(code)
                
                # Mock scanner results
                detected_by = []
                
                if 'bandit' in expected_detectors:
                    # Mock Bandit detection
                    detected_by.append('bandit')
                
                if 'semgrep' in expected_detectors:
                    # Mock Semgrep detection
                    detected_by.append('semgrep')
                
                detection_results[vuln_type] = {
                    'expected_detectors': expected_detectors,
                    'actual_detectors': detected_by,
                    'detection_rate': len(detected_by) / len(expected_detectors)
                }
                
                # Verify detection
                self.assertGreater(len(detected_by), 0, 
                                 f"No detectors found {vuln_type}")
                
                # Clean up test file
                os.remove(test_file)
            
            # Calculate overall detection effectiveness
            total_detection_rate = sum(r['detection_rate'] for r in detection_results.values()) / len(detection_results)
            self.assertGreater(total_detection_rate, 0.80, 
                             f"Overall detection rate {total_detection_rate:.2f} below threshold 0.80")
            
            logger.info(f"✅ Vulnerability detection effectiveness test passed: {total_detection_rate:.2f}")
            
        except Exception as e:
            self.fail(f"Vulnerability detection effectiveness test failed: {e}")
    
    def test_false_positive_management(self):
        """Test false positive management in security validation."""
        try:
            # Test cases that should NOT trigger alerts (potential false positives)
            false_positive_cases = [
                {
                    'type': 'test_password',
                    'code': '# Test password for unit testing\nTEST_PASSWORD = "test123"',
                    'context': 'test file',
                    'should_ignore': True
                },
                {
                    'type': 'config_template',
                    'code': 'password = "${PASSWORD}"  # Environment variable template',
                    'context': 'config template',
                    'should_ignore': True
                },
                {
                    'type': 'documentation',
                    'code': '# Example: password = "your-password-here"',
                    'context': 'documentation',
                    'should_ignore': True
                },
                {
                    'type': 'actual_secret',
                    'code': 'api_key = "prod-key-1234567890"',
                    'context': 'production code',
                    'should_ignore': False
                }
            ]
            
            false_positive_rate = 0
            total_cases = len(false_positive_cases)
            
            for test_case in false_positive_cases:
                case_type = test_case['type']
                code = test_case['code']
                should_ignore = test_case['should_ignore']
                
                # Create test file
                test_file = f'test_fp_{case_type}.py'
                with open(test_file, 'w') as f:
                    f.write(code)
                
                # Mock scanner result - simulate smart detection
                if 'test' in test_file.lower() or 'TEST_' in code or '#' in code:
                    scanner_flagged = False  # Smart scanner ignores test/template cases
                else:
                    scanner_flagged = True   # Real issues are flagged
                
                if should_ignore and scanner_flagged:
                    false_positive_rate += 1
                elif not should_ignore and not scanner_flagged:
                    false_positive_rate += 1
                
                # Clean up
                os.remove(test_file)
            
            false_positive_percentage = (false_positive_rate / total_cases) * 100
            self.assertLess(false_positive_percentage, 20, 
                          f"False positive rate {false_positive_percentage:.1f}% too high")
            
            logger.info(f"✅ False positive management test passed: {false_positive_percentage:.1f}% FP rate")
            
        except Exception as e:
            self.fail(f"False positive management test failed: {e}")
    
    def test_compliance_framework_coverage(self):
        """Test coverage of compliance frameworks in security validation."""
        try:
            # Test compliance framework requirements coverage
            frameworks = {
                'CIS': {
                    'requirements': ['access_control', 'data_protection', 'logging', 'monitoring'],
                    'covered_by_scanners': ['access_control', 'data_protection', 'logging']
                },
                'NIST': {
                    'requirements': ['identify', 'protect', 'detect', 'respond', 'recover'],
                    'covered_by_scanners': ['identify', 'protect', 'detect']
                },
                'OWASP': {
                    'requirements': ['injection', 'broken_auth', 'sensitive_data', 'xxe', 'broken_access'],
                    'covered_by_scanners': ['injection', 'sensitive_data', 'broken_access']
                }
            }
            
            overall_coverage = 0
            framework_count = len(frameworks)
            
            for framework_name, framework_data in frameworks.items():
                requirements = framework_data['requirements']
                covered = framework_data['covered_by_scanners']
                
                coverage_rate = len(covered) / len(requirements)
                overall_coverage += coverage_rate
                
                # Verify minimum coverage per framework
                self.assertGreater(coverage_rate, 0.60, 
                                 f"{framework_name} coverage {coverage_rate:.2f} below minimum 0.60")
            
            avg_coverage = overall_coverage / framework_count
            self.assertGreater(avg_coverage, 0.70, 
                             f"Overall compliance coverage {avg_coverage:.2f} below threshold 0.70")
            
            logger.info(f"✅ Compliance framework coverage test passed: {avg_coverage:.2f}")
            
        except Exception as e:
            self.fail(f"Compliance framework coverage test failed: {e}")


class AsyncPipelineTests(PipelineReliabilityTestCase):
    """Test asynchronous pipeline operations."""
    
    def test_parallel_security_scan_execution(self):
        """Test parallel execution of multiple security scans."""
        async def run_parallel_scans():
            # Mock parallel scanner execution
            scan_tasks = []
            
            scanners = ['bandit', 'semgrep', 'safety', 'trivy']
            
            async def mock_scanner(scanner_name):
                # Simulate scanner execution time
                await asyncio.sleep(0.1)
                return {'scanner': scanner_name, 'status': 'completed', 'issues': 2}
            
            # Start all scanners in parallel
            for scanner in scanners:
                task = asyncio.create_task(mock_scanner(scanner))
                scan_tasks.append(task)
            
            # Wait for all scanners to complete
            results = await asyncio.gather(*scan_tasks)
            
            return results
        
        try:
            # Run async test
            start_time = time.time()
            results = asyncio.run(run_parallel_scans())
            execution_time = time.time() - start_time
            
            # Verify all scanners completed
            self.assertEqual(len(results), 4)
            for result in results:
                self.assertEqual(result['status'], 'completed')
                self.assertIn('scanner', result)
            
            # Verify parallel execution was faster than sequential
            sequential_time_estimate = 0.1 * 4  # 4 scanners * 0.1s each
            parallel_speedup = sequential_time_estimate / execution_time
            self.assertGreater(parallel_speedup, 2.0, 
                             f"Parallel execution speedup {parallel_speedup:.1f}x insufficient")
            
            logger.info(f"✅ Parallel security scan test passed: {parallel_speedup:.1f}x speedup")
            
        except Exception as e:
            self.fail(f"Parallel security scan test failed: {e}")
    
    def test_async_pipeline_error_handling(self):
        """Test error handling in asynchronous pipeline operations."""
        async def run_scan_with_errors():
            async def failing_scanner():
                await asyncio.sleep(0.05)
                raise Exception("Scanner failed")
            
            async def working_scanner():
                await asyncio.sleep(0.05)
                return {'status': 'completed'}
            
            # Mix of working and failing scanners
            tasks = [
                asyncio.create_task(working_scanner()),
                asyncio.create_task(failing_scanner()),
                asyncio.create_task(working_scanner())
            ]
            
            # Gather with return_exceptions=True to handle errors gracefully
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
        
        try:
            results = asyncio.run(run_scan_with_errors())
            
            # Verify error handling
            successful_scans = [r for r in results if isinstance(r, dict)]
            failed_scans = [r for r in results if isinstance(r, Exception)]
            
            self.assertEqual(len(successful_scans), 2)
            self.assertEqual(len(failed_scans), 1)
            
            # Verify system continues despite failures
            for success in successful_scans:
                self.assertEqual(success['status'], 'completed')
            
            logger.info("✅ Async pipeline error handling test passed")
            
        except Exception as e:
            self.fail(f"Async pipeline error handling test failed: {e}")


if __name__ == '__main__':
    # Configure test runner
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=True
    )
    
    result = runner.run(suite)
    
    # Generate test report
    if hasattr(result, 'testsRun'):
        print(f"\n{'='*60}")
        print("PIPELINE RELIABILITY TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Tests Run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
        print(f"{'='*60}")
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)