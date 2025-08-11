#!/usr/bin/env python3
"""
Simple tests for honeypot deception effectiveness and detection capabilities
Tests basic functionality without external dependencies
"""

import unittest
import json
import time
import socket
import os
import sqlite3
from datetime import datetime, timedelta
import tempfile

class TestHoneypotConfiguration(unittest.TestCase):
    """Test honeypot configuration files"""
    
    def test_cowrie_config_exists(self):
        """Test that Cowrie configuration file exists"""
        config_path = "infrastructure/config/honeypots/cowrie.cfg"
        self.assertTrue(os.path.exists(config_path), "Cowrie config should exist")
        
        # Check config content
        with open(config_path, 'r') as f:
            content = f.read()
            self.assertIn("[honeypot]", content, "Should have honeypot section")
            self.assertIn("ssh_port", content, "Should configure SSH port")
    
    def test_dionaea_config_exists(self):
        """Test that Dionaea configuration file exists"""
        config_path = "infrastructure/config/honeypots/dionaea.cfg"
        self.assertTrue(os.path.exists(config_path), "Dionaea config should exist")
        
        with open(config_path, 'r') as f:
            content = f.read()
            self.assertIn("[dionaea]", content, "Should have dionaea section")
    
    def test_glastopf_config_exists(self):
        """Test that Glastopf configuration file exists"""
        config_path = "infrastructure/config/honeypots/glastopf.cfg"
        self.assertTrue(os.path.exists(config_path), "Glastopf config should exist")
        
        with open(config_path, 'r') as f:
            content = f.read()
            self.assertIn("[webserver]", content, "Should have webserver section")
    
    def test_monitor_config_exists(self):
        """Test that monitor configuration file exists"""
        config_path = "infrastructure/config/honeypots/monitor_config.yaml"
        self.assertTrue(os.path.exists(config_path), "Monitor config should exist")
        
        with open(config_path, 'r') as f:
            content = f.read()
            self.assertIn("monitoring:", content, "Should have monitoring section")
            self.assertIn("alerting:", content, "Should have alerting section")

class TestHoneytokenGeneration(unittest.TestCase):
    """Test honeytoken generation functionality"""
    
    def test_honeytoken_script_exists(self):
        """Test that honeytoken script exists"""
        script_path = "infrastructure/config/honeypots/honeytokens.py"
        self.assertTrue(os.path.exists(script_path), "Honeytoken script should exist")
        
        # Check script content
        with open(script_path, 'r') as f:
            content = f.read()
            self.assertIn("class HoneytokenGenerator", content, "Should have HoneytokenGenerator class")
            self.assertIn("generate_fake_credentials", content, "Should have credential generation")
    
    def test_honeytoken_config_exists(self):
        """Test that honeytoken configuration exists"""
        config_path = "infrastructure/config/honeypots/config.yaml"
        self.assertTrue(os.path.exists(config_path), "Honeytoken config should exist")
        
        with open(config_path, 'r') as f:
            content = f.read()
            self.assertIn("credentials:", content, "Should have credentials section")
            self.assertIn("documents:", content, "Should have documents section")
            self.assertIn("api_keys:", content, "Should have api_keys section")

class TestDecoyServices(unittest.TestCase):
    """Test decoy services implementation"""
    
    def test_decoy_services_script_exists(self):
        """Test that decoy services script exists"""
        script_path = "infrastructure/config/honeypots/decoy_services.py"
        self.assertTrue(os.path.exists(script_path), "Decoy services script should exist")
        
        with open(script_path, 'r') as f:
            content = f.read()
            self.assertIn("class DecoyService", content, "Should have DecoyService base class")
            self.assertIn("class FakeSSHService", content, "Should have fake SSH service")
            self.assertIn("class FakeFTPService", content, "Should have fake FTP service")
            self.assertIn("class FakeWebAdminPanel", content, "Should have fake web admin panel")
    
    def test_fake_admin_panel_exists(self):
        """Test that fake admin panel HTML exists"""
        html_path = "infrastructure/config/honeypots/fake_admin_panel.html"
        self.assertTrue(os.path.exists(html_path), "Fake admin panel HTML should exist")
        
        with open(html_path, 'r') as f:
            content = f.read()
            self.assertIn("<title>Corporate Admin Panel", content, "Should have admin panel title")
            self.assertIn("username", content, "Should have username field")
            self.assertIn("password", content, "Should have password field")

class TestHoneypotMonitoring(unittest.TestCase):
    """Test honeypot monitoring system"""
    
    def test_monitor_script_exists(self):
        """Test that monitor script exists"""
        script_path = "infrastructure/config/honeypots/honeypot_monitor.py"
        self.assertTrue(os.path.exists(script_path), "Monitor script should exist")
        
        with open(script_path, 'r') as f:
            content = f.read()
            self.assertIn("class HoneypotMonitor", content, "Should have HoneypotMonitor class")
            self.assertIn("class HoneypotAlert", content, "Should have HoneypotAlert class")
            self.assertIn("start_monitoring", content, "Should have monitoring start method")
    
    def test_database_initialization(self):
        """Test that database can be initialized"""
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
        
        try:
            # Initialize database
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Create alerts table (from monitor script)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        alert_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        honeypot_type TEXT NOT NULL,
                        source_ip TEXT NOT NULL,
                        target_port INTEGER,
                        attack_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        details TEXT,
                        raw_data TEXT
                    )
                ''')
                
                # Insert test alert
                cursor.execute('''
                    INSERT INTO alerts 
                    (alert_id, timestamp, honeypot_type, source_ip, target_port, attack_type, severity, details, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', ("test_alert", datetime.now().isoformat(), "cowrie_ssh", "192.168.1.100", 2222, "ssh_brute_force", "medium", "{}", "test_data"))
                
                conn.commit()
                
                # Verify alert was stored
                cursor.execute("SELECT COUNT(*) FROM alerts")
                count = cursor.fetchone()[0]
                self.assertEqual(count, 1, "Should store alert in database")
                
        finally:
            # Clean up
            if os.path.exists(db_path):
                os.unlink(db_path)

class TestDockerConfiguration(unittest.TestCase):
    """Test Docker configuration files"""
    
    def test_dockerfiles_exist(self):
        """Test that Dockerfiles exist"""
        dockerfiles = [
            "infrastructure/config/honeypots/Dockerfile.decoy",
            "infrastructure/config/honeypots/Dockerfile.honeytokens",
            "infrastructure/config/honeypots/Dockerfile.monitor"
        ]
        
        for dockerfile in dockerfiles:
            with self.subTest(dockerfile=dockerfile):
                self.assertTrue(os.path.exists(dockerfile), f"Dockerfile {dockerfile} should exist")
                
                with open(dockerfile, 'r') as f:
                    content = f.read()
                    self.assertIn("FROM python:", content, "Should use Python base image")
                    self.assertIn("COPY", content, "Should copy files")
    
    def test_docker_compose_updated(self):
        """Test that docker-compose.yml includes honeypot services"""
        compose_path = "infrastructure/docker-compose.yml"
        self.assertTrue(os.path.exists(compose_path), "Docker compose file should exist")
        
        with open(compose_path, 'r') as f:
            content = f.read()
            
            # Check for honeypot services
            honeypot_services = [
                "cowrie-ssh",
                "dionaea-malware", 
                "glastopf-web",
                "decoy-services",
                "honeytoken-distributor",
                "honeypot-monitor"
            ]
            
            for service in honeypot_services:
                self.assertIn(service, content, f"Should include {service} service")
            
            # Check for deception network
            self.assertIn("deception_network", content, "Should include deception network")
            self.assertIn("192.168.50.0/24", content, "Should configure deception subnet")

class TestDeploymentScripts(unittest.TestCase):
    """Test deployment scripts"""
    
    def test_deployment_script_exists(self):
        """Test that deployment script exists"""
        script_path = "infrastructure/scripts/deploy_honeypots.sh"
        self.assertTrue(os.path.exists(script_path), "Deployment script should exist")
        
        with open(script_path, 'r') as f:
            content = f.read()
            self.assertIn("#!/bin/bash", content, "Should be bash script")
            self.assertIn("deploy_honeypots", content, "Should have deployment function")
            self.assertIn("verify_deployment", content, "Should have verification function")
    
    def test_script_is_executable(self):
        """Test that deployment script is executable"""
        script_path = "infrastructure/scripts/deploy_honeypots.sh"
        self.assertTrue(os.access(script_path, os.X_OK), "Deployment script should be executable")

class TestThreatIntelligence(unittest.TestCase):
    """Test threat intelligence integration"""
    
    def test_threat_intel_structure(self):
        """Test that threat intelligence structure is defined"""
        monitor_script = "infrastructure/config/honeypots/honeypot_monitor.py"
        
        with open(monitor_script, 'r') as f:
            content = f.read()
            self.assertIn("threat_intel", content, "Should include threat intelligence")
            self.assertIn("malicious_ips", content, "Should track malicious IPs")
            self.assertIn("known_scanners", content, "Should track known scanners")

class TestMITREMapping(unittest.TestCase):
    """Test MITRE ATT&CK mapping"""
    
    def test_mitre_mapping_exists(self):
        """Test that MITRE ATT&CK mapping is defined"""
        config_path = "infrastructure/config/honeypots/monitor_config.yaml"
        
        with open(config_path, 'r') as f:
            content = f.read()
            self.assertIn("mitre_mapping:", content, "Should have MITRE mapping section")
            self.assertIn("T1110", content, "Should map brute force attacks")
            self.assertIn("T1046", content, "Should map network scanning")
            self.assertIn("T1190", content, "Should map web exploits")

class TestAlertGeneration(unittest.TestCase):
    """Test alert generation functionality"""
    
    def test_alert_structure(self):
        """Test that alert structure is properly defined"""
        monitor_script = "infrastructure/config/honeypots/honeypot_monitor.py"
        
        with open(monitor_script, 'r') as f:
            content = f.read()
            self.assertIn("@dataclass", content, "Should use dataclass for alerts")
            self.assertIn("class HoneypotAlert", content, "Should define HoneypotAlert class")
            self.assertIn("alert_id", content, "Should have alert ID")
            self.assertIn("severity", content, "Should have severity classification")
    
    def test_severity_levels(self):
        """Test that severity levels are properly defined"""
        config_path = "infrastructure/config/honeypots/monitor_config.yaml"
        
        with open(config_path, 'r') as f:
            content = f.read()
            severity_levels = ["low", "medium", "high", "critical"]
            
            for level in severity_levels:
                self.assertIn(level, content, f"Should define {level} severity level")

def run_simple_tests():
    """Run all simple honeypot tests"""
    test_suites = [
        unittest.TestLoader().loadTestsFromTestCase(TestHoneypotConfiguration),
        unittest.TestLoader().loadTestsFromTestCase(TestHoneytokenGeneration),
        unittest.TestLoader().loadTestsFromTestCase(TestDecoyServices),
        unittest.TestLoader().loadTestsFromTestCase(TestHoneypotMonitoring),
        unittest.TestLoader().loadTestsFromTestCase(TestDockerConfiguration),
        unittest.TestLoader().loadTestsFromTestCase(TestDeploymentScripts),
        unittest.TestLoader().loadTestsFromTestCase(TestThreatIntelligence),
        unittest.TestLoader().loadTestsFromTestCase(TestMITREMapping),
        unittest.TestLoader().loadTestsFromTestCase(TestAlertGeneration)
    ]
    
    combined_suite = unittest.TestSuite(test_suites)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(combined_suite)
    
    # Generate test report
    report = {
        "timestamp": datetime.now().isoformat(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
    }
    
    print(f"\n=== Simple Honeypot Test Report ===")
    print(f"Tests Run: {report['tests_run']}")
    print(f"Failures: {report['failures']}")
    print(f"Errors: {report['errors']}")
    print(f"Success Rate: {report['success_rate']:.2%}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0] if 'AssertionError:' in traceback else 'Unknown failure'}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\\n')[-2] if traceback else 'Unknown error'}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_simple_tests()
    exit(0 if success else 1)