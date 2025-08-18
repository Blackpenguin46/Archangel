#!/usr/bin/env python3
"""
Advanced Network Infrastructure Simulation Demo

This demo showcases the comprehensive network simulation capabilities including:
- IoT devices and BYOD endpoints
- Legacy systems with outdated protocols
- Network service dependencies with realistic failure modes
- Network topology discovery and mapping
"""

import asyncio
import json
import logging
import time
from datetime import datetime
import argparse

# Import simulation modules
from infrastructure.network_simulation import AdvancedNetworkSimulator
from infrastructure.iot_simulation import IoTAndBYODSimulator
from infrastructure.legacy_systems import LegacySystemManager
from infrastructure.network_dependencies import NetworkDependencySimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NetworkSimulationDemo:
    """Comprehensive network simulation demo"""
    
    def __init__(self):
        self.network_sim = AdvancedNetworkSimulator()
        self.iot_sim = IoTAndBYODSimulator()
        self.legacy_sim = LegacySystemManager()
        self.dependency_sim = NetworkDependencySimulator()
        
        self.demo_running = False
    
    def run_basic_demo(self):
        """Run basic network simulation demo"""
        print("="*60)
        print("ADVANCED NETWORK INFRASTRUCTURE SIMULATION DEMO")
        print("="*60)
        
        try:
            # 1. Basic Network Topology Discovery
            print("\n1. NETWORK TOPOLOGY DISCOVERY")
            print("-" * 40)
            
            self.network_sim.start_simulation()
            time.sleep(3)  # Let it discover devices
            
            network_map = self.network_sim.get_network_map()
            stats = network_map["statistics"]
            
            print(f"✓ Discovered {stats['total_devices']} network devices")
            print(f"✓ Found {stats['vulnerable_devices']} vulnerable devices ({stats['vulnerability_percentage']:.1f}%)")
            print(f"✓ Identified {stats['legacy_devices']} legacy devices")
            print(f"✓ Detected {stats['iot_devices']} IoT devices")
            
            print("\nDevice Types Distribution:")
            for device_type, count in stats["device_types"].items():
                print(f"  - {device_type}: {count}")
            
            # 2. IoT and BYOD Device Simulation
            print("\n2. IoT AND BYOD DEVICE SIMULATION")
            print("-" * 40)
            
            self.iot_sim.populate_iot_network(count=15)
            self.iot_sim.populate_byod_network(count=10)
            self.iot_sim.start_simulation()
            
            time.sleep(2)  # Let devices generate activity
            
            iot_stats = self.iot_sim.get_iot_statistics()
            byod_stats = self.iot_sim.get_byod_statistics()
            
            print(f"✓ Simulated {iot_stats['total_iot_devices']} IoT devices")
            print(f"  - Vulnerable IoT devices: {iot_stats['vulnerable_iot_devices']}")
            print(f"  - Devices with default credentials: {iot_stats['default_credentials']}")
            print(f"  - Unencrypted devices: {iot_stats['unencrypted_devices']}")
            
            print(f"✓ Simulated {byod_stats['total_byod_devices']} BYOD devices")
            print(f"  - Managed devices: {byod_stats['managed_byod_devices']}")
            print(f"  - Compliant devices: {byod_stats['compliant_byod_devices']}")
            print(f"  - MDM enrolled: {byod_stats['mdm_enrolled_devices']}")
            
            print("\nIoT Device Categories:")
            for category, count in iot_stats["device_categories"].items():
                print(f"  - {category}: {count}")
            
            # 3. Legacy Systems Simulation
            print("\n3. LEGACY SYSTEMS SIMULATION")
            print("-" * 40)
            
            self.legacy_sim.populate_legacy_network(count=8)
            self.legacy_sim.start_simulation()
            
            time.sleep(2)
            
            legacy_stats = self.legacy_sim.get_legacy_statistics()
            
            print(f"✓ Simulated {legacy_stats['total_systems']} legacy systems")
            print(f"  - Critical systems: {legacy_stats['critical_systems']}")
            print(f"  - Unsupported systems: {legacy_stats['unsupported_systems']} ({legacy_stats['unsupported_percentage']:.1f}%)")
            print(f"  - Total vulnerabilities: {legacy_stats['total_vulnerabilities']}")
            print(f"  - Critical vulnerabilities: {legacy_stats['critical_vulnerabilities']}")
            print(f"  - Average risk score: {legacy_stats['average_risk_score']}/10")
            
            print("\nLegacy System Types:")
            for sys_type, count in legacy_stats["system_types"].items():
                print(f"  - {sys_type}: {count}")
            
            # 4. Network Dependencies and Failure Simulation
            print("\n4. NETWORK DEPENDENCIES AND FAILURE SIMULATION")
            print("-" * 40)
            
            self.dependency_sim.setup_enterprise_services()
            self.dependency_sim.start_simulation()
            
            time.sleep(3)  # Let it potentially generate some failures
            
            health = self.dependency_sim.get_system_health()
            dep_metrics = health["dependency_metrics"]
            
            print(f"✓ Configured {health['total_services']} enterprise services")
            print(f"  - System health: {health['health_status']} ({health['health_score']:.1f}%)")
            print(f"  - Running services: {health['running_services']}")
            print(f"  - Failed services: {health['failed_services']}")
            print(f"  - Active failures: {health['active_failures']}")
            print(f"  - Single points of failure: {health['single_points_of_failure']}")
            
            print(f"\nDependency Complexity:")
            print(f"  - Total dependencies: {dep_metrics['total_dependencies']}")
            print(f"  - Avg dependencies per service: {dep_metrics['average_dependencies_per_service']:.1f}")
            print(f"  - Graph density: {dep_metrics['graph_density']:.3f}")
            
            # 5. Demonstrate Attack Surface Analysis
            print("\n5. ATTACK SURFACE ANALYSIS")
            print("-" * 40)
            
            self._analyze_attack_surface()
            
            # 6. Export Comprehensive Report
            print("\n6. EXPORTING COMPREHENSIVE REPORTS")
            print("-" * 40)
            
            self._export_reports()
            
        except Exception as e:
            logger.error(f"Demo error: {e}")
            
        finally:
            self._cleanup_simulations()
    
    def run_advanced_demo(self):
        """Run advanced demo with failure simulation"""
        print("\n" + "="*60)
        print("ADVANCED FAILURE SIMULATION DEMO")
        print("="*60)
        
        try:
            # Setup all simulations
            self._setup_comprehensive_environment()
            
            # Demonstrate cascade failures
            print("\n1. CASCADE FAILURE SIMULATION")
            print("-" * 40)
            
            self._demonstrate_cascade_failures()
            
            # Demonstrate IoT security issues
            print("\n2. IoT SECURITY VULNERABILITY DEMONSTRATION")
            print("-" * 40)
            
            self._demonstrate_iot_vulnerabilities()
            
            # Demonstrate legacy system risks
            print("\n3. LEGACY SYSTEM RISK ASSESSMENT")
            print("-" * 40)
            
            self._demonstrate_legacy_risks()
            
            # Real-time monitoring
            print("\n4. REAL-TIME MONITORING DEMONSTRATION")
            print("-" * 40)
            
            self._demonstrate_real_time_monitoring()
            
        except Exception as e:
            logger.error(f"Advanced demo error: {e}")
            
        finally:
            self._cleanup_simulations()
    
    def _setup_comprehensive_environment(self):
        """Setup comprehensive simulation environment"""
        print("Setting up comprehensive network environment...")
        
        # Start all simulations
        self.network_sim.start_simulation()
        
        self.iot_sim.populate_iot_network(count=20)
        self.iot_sim.populate_byod_network(count=15)
        self.iot_sim.start_simulation()
        
        self.legacy_sim.populate_legacy_network(count=10)
        self.legacy_sim.start_simulation()
        
        self.dependency_sim.setup_enterprise_services()
        self.dependency_sim.start_simulation()
        
        # Let everything initialize
        time.sleep(5)
        
        print("✓ Comprehensive environment setup complete")
    
    def _demonstrate_cascade_failures(self):
        """Demonstrate cascade failure scenarios"""
        from infrastructure.network_dependencies import FailureType
        
        # Simulate DNS server failure
        print("Simulating DNS server failure...")
        
        failure = self.dependency_sim.failure_simulator.simulate_targeted_failure(
            "dns_primary", FailureType.HARDWARE_FAILURE
        )
        
        if failure:
            print(f"✓ Simulated failure: {failure.description}")
            print(f"  - Affected services: {len(failure.affected_services)}")
            
            # Show cascade effects
            for service_id in failure.affected_services[:5]:  # Show first 5
                service = self.dependency_sim.dependency_graph.services.get(service_id)
                if service:
                    print(f"    → {service.name} ({service.status.value})")
            
            # Wait a bit then resolve
            time.sleep(3)
            
            resolved = self.dependency_sim.failure_simulator.resolve_failure(failure.failure_id)
            if resolved:
                print("✓ Failure resolved, services recovering")
        else:
            print("⚠ No failure occurred (service may not exist)")
        
        # Show failure statistics
        failure_stats = self.dependency_sim.failure_simulator.get_failure_statistics()
        print(f"\nFailure Statistics:")
        print(f"  - Total failures: {failure_stats['total_failures']}")
        print(f"  - Cascade failures: {failure_stats['cascade_failures']}")
        print(f"  - MTTR: {failure_stats['mttr_minutes']:.1f} minutes")
    
    def _demonstrate_iot_vulnerabilities(self):
        """Demonstrate IoT security vulnerabilities"""
        print("Analyzing IoT device vulnerabilities...")
        
        vulnerable_devices = []
        
        for device in self.iot_sim.iot_devices.values():
            vulns = device.simulate_vulnerability_scan()
            if vulns:
                vulnerable_devices.append((device, vulns))
        
        print(f"✓ Found {len(vulnerable_devices)} vulnerable IoT devices")
        
        # Show examples
        for i, (device, vulns) in enumerate(vulnerable_devices[:3]):
            print(f"\n  Device: {device.name} ({device.ip_address})")
            print(f"    Category: {device.category.value}")
            print(f"    Manufacturer: {device.manufacturer}")
            print(f"    Vulnerabilities:")
            for vuln in vulns[:2]:  # Show first 2 vulnerabilities
                print(f"      - {vuln}")
        
        # Show statistics
        iot_stats = self.iot_sim.get_iot_statistics()
        print(f"\nIoT Security Summary:")
        print(f"  - Devices with default credentials: {iot_stats['default_credentials']}")
        print(f"  - Unencrypted devices: {iot_stats['unencrypted_devices']}")
        print(f"  - Vulnerability rate: {iot_stats['vulnerability_percentage']:.1f}%")
    
    def _demonstrate_legacy_risks(self):
        """Demonstrate legacy system risks"""
        print("Assessing legacy system risks...")
        
        # Generate vulnerability report
        vuln_report = self.legacy_sim.get_vulnerability_report()
        
        print(f"✓ Scanned {vuln_report['summary']['total_systems_scanned']} legacy systems")
        print(f"  - Vulnerable systems: {vuln_report['summary']['vulnerable_systems']}")
        print(f"  - Total vulnerabilities: {vuln_report['summary']['total_vulnerabilities']}")
        
        print("\nVulnerabilities by Severity:")
        for severity, count in vuln_report['summary']['by_severity'].items():
            if count > 0:
                print(f"  - {severity}: {count}")
        
        # Show highest risk systems
        high_risk_systems = []
        for system_id, system_data in vuln_report['systems'].items():
            if system_data['risk_score'] > 7.0:
                high_risk_systems.append((system_id, system_data))
        
        print(f"\nHigh Risk Systems (score > 7.0): {len(high_risk_systems)}")
        for system_id, system_data in high_risk_systems[:3]:
            print(f"  - {system_data['hostname']}: {system_data['risk_score']:.1f}/10")
            print(f"    OS: {system_data['os_name']} {system_data['os_version']}")
            print(f"    Critical vulns: {system_data['vulnerability_counts']['Critical']}")
    
    def _demonstrate_real_time_monitoring(self):
        """Demonstrate real-time monitoring capabilities"""
        print("Demonstrating real-time monitoring...")
        
        # Monitor for 30 seconds
        for i in range(6):
            print(f"\n--- Monitoring Cycle {i+1}/6 ---")
            
            # Network status
            network_map = self.network_sim.get_network_map()
            network_stats = network_map["statistics"]
            
            # System health
            health = self.dependency_sim.get_system_health()
            
            # IoT activity
            iot_stats = self.iot_sim.get_iot_statistics()
            
            print(f"Network: {network_stats['total_devices']} devices, {network_stats['vulnerable_devices']} vulnerable")
            print(f"System Health: {health['health_status']} ({health['health_score']:.1f}%)")
            print(f"IoT: {iot_stats['online_iot_devices']}/{iot_stats['total_iot_devices']} online")
            print(f"Active Failures: {health['active_failures']}")
            
            if i < 5:  # Don't sleep on last iteration
                time.sleep(5)
        
        print("\n✓ Real-time monitoring demonstration complete")
    
    def _analyze_attack_surface(self):
        """Analyze the overall attack surface"""
        # Collect data from all simulations
        network_stats = self.network_sim.get_network_map()["statistics"]
        iot_stats = self.iot_sim.get_iot_statistics()
        byod_stats = self.iot_sim.get_byod_statistics()
        legacy_stats = self.legacy_sim.get_legacy_statistics()
        
        total_devices = (
            network_stats["total_devices"] +
            iot_stats["total_iot_devices"] +
            byod_stats["total_byod_devices"] +
            legacy_stats["total_systems"]
        )
        
        total_vulnerabilities = (
            network_stats["vulnerable_devices"] +
            iot_stats["vulnerable_iot_devices"] +
            legacy_stats["total_vulnerabilities"]
        )
        
        print(f"✓ Total Attack Surface Analysis:")
        print(f"  - Total devices/systems: {total_devices}")
        print(f"  - Vulnerable assets: {total_vulnerabilities}")
        print(f"  - Legacy systems: {legacy_stats['total_systems']} ({legacy_stats['unsupported_percentage']:.1f}% unsupported)")
        print(f"  - IoT devices: {iot_stats['total_iot_devices']} ({iot_stats['default_credentials']} with default creds)")
        print(f"  - BYOD devices: {byod_stats['total_byod_devices']} ({byod_stats['compliant_byod_devices']} compliant)")
        
        # Risk assessment
        risk_factors = []
        if legacy_stats['unsupported_percentage'] > 50:
            risk_factors.append("High percentage of unsupported legacy systems")
        if iot_stats['vulnerability_percentage'] > 60:
            risk_factors.append("High IoT vulnerability rate")
        if byod_stats['compliance_percentage'] < 70:
            risk_factors.append("Low BYOD compliance rate")
        
        if risk_factors:
            print(f"\n⚠ Risk Factors Identified:")
            for factor in risk_factors:
                print(f"  - {factor}")
        else:
            print(f"\n✓ No major risk factors identified")
    
    def _export_reports(self):
        """Export comprehensive reports"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export network configuration
        network_file = f"network_simulation_{timestamp}.json"
        self.network_sim.export_network_config(network_file)
        print(f"✓ Network configuration exported to {network_file}")
        
        # Export IoT/BYOD inventory
        iot_file = f"iot_byod_inventory_{timestamp}.json"
        self.iot_sim.export_device_inventory(iot_file)
        print(f"✓ IoT/BYOD inventory exported to {iot_file}")
        
        # Export legacy systems inventory
        legacy_file = f"legacy_systems_{timestamp}.json"
        self.legacy_sim.export_legacy_inventory(legacy_file)
        print(f"✓ Legacy systems inventory exported to {legacy_file}")
        
        # Export dependency report
        dependency_file = f"network_dependencies_{timestamp}.json"
        self.dependency_sim.export_dependency_report(dependency_file)
        print(f"✓ Network dependencies report exported to {dependency_file}")
        
        # Create summary report
        summary_file = f"network_simulation_summary_{timestamp}.json"
        self._create_summary_report(summary_file)
        print(f"✓ Summary report exported to {summary_file}")
    
    def _create_summary_report(self, filename: str):
        """Create a comprehensive summary report"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "simulation_overview": {
                "network_topology": self.network_sim.get_network_map()["statistics"],
                "iot_devices": self.iot_sim.get_iot_statistics(),
                "byod_devices": self.iot_sim.get_byod_statistics(),
                "legacy_systems": self.legacy_sim.get_legacy_statistics(),
                "system_health": self.dependency_sim.get_system_health()
            },
            "security_assessment": {
                "vulnerability_summary": self.legacy_sim.get_vulnerability_report()["summary"],
                "risk_factors": self._identify_risk_factors()
            },
            "recommendations": self._generate_recommendations()
        }
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def _identify_risk_factors(self) -> list:
        """Identify key risk factors"""
        risk_factors = []
        
        legacy_stats = self.legacy_sim.get_legacy_statistics()
        iot_stats = self.iot_sim.get_iot_statistics()
        byod_stats = self.iot_sim.get_byod_statistics()
        
        if legacy_stats['unsupported_percentage'] > 30:
            risk_factors.append({
                "category": "Legacy Systems",
                "risk": "High percentage of unsupported systems",
                "value": f"{legacy_stats['unsupported_percentage']:.1f}%",
                "severity": "high"
            })
        
        if iot_stats['default_credentials'] > 0:
            risk_factors.append({
                "category": "IoT Security",
                "risk": "Devices with default credentials",
                "value": iot_stats['default_credentials'],
                "severity": "medium"
            })
        
        if byod_stats['compliance_percentage'] < 80:
            risk_factors.append({
                "category": "BYOD Compliance",
                "risk": "Low device compliance rate",
                "value": f"{byod_stats['compliance_percentage']:.1f}%",
                "severity": "medium"
            })
        
        return risk_factors
    
    def _generate_recommendations(self) -> list:
        """Generate security recommendations"""
        recommendations = [
            {
                "category": "Legacy Systems",
                "recommendation": "Prioritize migration or isolation of unsupported legacy systems",
                "priority": "high"
            },
            {
                "category": "IoT Security",
                "recommendation": "Implement network segmentation for IoT devices",
                "priority": "high"
            },
            {
                "category": "IoT Security",
                "recommendation": "Change default credentials on all IoT devices",
                "priority": "critical"
            },
            {
                "category": "BYOD Management",
                "recommendation": "Enforce MDM enrollment for all BYOD devices",
                "priority": "medium"
            },
            {
                "category": "Network Monitoring",
                "recommendation": "Implement continuous network monitoring and anomaly detection",
                "priority": "high"
            },
            {
                "category": "Dependency Management",
                "recommendation": "Identify and mitigate single points of failure",
                "priority": "high"
            }
        ]
        
        return recommendations
    
    def _cleanup_simulations(self):
        """Clean up all running simulations"""
        print("\nCleaning up simulations...")
        
        try:
            self.network_sim.stop_simulation()
            self.iot_sim.stop_simulation()
            self.legacy_sim.stop_simulation()
            self.dependency_sim.stop_simulation()
            print("✓ All simulations stopped")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Advanced Network Infrastructure Simulation Demo")
    parser.add_argument("--mode", choices=["basic", "advanced", "both"], default="basic",
                       help="Demo mode to run")
    parser.add_argument("--duration", type=int, default=60,
                       help="Demo duration in seconds (for advanced mode)")
    
    args = parser.parse_args()
    
    demo = NetworkSimulationDemo()
    
    try:
        if args.mode in ["basic", "both"]:
            demo.run_basic_demo()
        
        if args.mode in ["advanced", "both"]:
            demo.run_advanced_demo()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nGenerated files:")
        print("- network_simulation_*.json - Network topology and device inventory")
        print("- iot_byod_inventory_*.json - IoT and BYOD device details")
        print("- legacy_systems_*.json - Legacy system vulnerability assessment")
        print("- network_dependencies_*.json - Service dependency analysis")
        print("- network_simulation_summary_*.json - Comprehensive summary report")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        demo._cleanup_simulations()
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        demo._cleanup_simulations()


if __name__ == "__main__":
    main()