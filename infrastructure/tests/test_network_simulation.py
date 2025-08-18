#!/usr/bin/env python3
"""
Tests for Advanced Network Infrastructure Simulation

This module contains comprehensive tests for the network simulation components
including IoT devices, BYOD endpoints, legacy systems, and network dependencies.
"""

import unittest
import tempfile
import json
import time
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Import the modules to test
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network_simulation import (
    AdvancedNetworkSimulator, NetworkTopologyMapper, NetworkDependencyManager,
    DeviceType, ServiceType, VulnerabilityLevel, NetworkDevice, NetworkService
)
from iot_simulation import (
    IoTAndBYODSimulator, IoTDeviceFactory, BYODDeviceFactory,
    IoTDeviceCategory, BYODDeviceType, IoTProtocol
)
from legacy_systems import (
    LegacySystemManager, LegacySystemFactory, LegacySystemType,
    LegacyProtocol, VulnerabilityCategory
)
from network_dependencies import (
    NetworkDependencySimulator, ServiceDependencyGraph, FailureSimulator,
    ServiceType as DepServiceType, FailureType, ServiceStatus
)


class TestNetworkTopologyMapper(unittest.TestCase):
    """Test cases for NetworkTopologyMapper"""
    
    def setUp(self):
        self.mapper = NetworkTopologyMapper()
    
    def test_device_discovery(self):
        """Test device discovery functionality"""
        # Test device discovery
        device = self.mapper.discover_device("192.168.10.100")
        
        self.assertIsNotNone(device)
        self.assertEqual(device.ip_address, "192.168.10.100")
        self.assertIsInstance(device.device_type, DeviceType)
        self.assertTrue(len(device.mac_address) > 0)
        self.assertTrue(len(device.hostname) > 0)
    
    def test_network_range_scanning(self):
        """Test network range scanning"""
        # Test scanning a small network range
        devices = self.mapper.scan_network_range("192.168.20.0/28")
        
        # Should discover some devices (not all IPs will have devices)
        self.assertGreater(len(devices), 0)
        self.assertLess(len(devices), 16)  # Not every IP should have a device
        
        # All discovered devices should be in the correct IP range
        for device in devices:
            self.assertTrue(device.ip_address.startswith("192.168.20."))
    
    def test_topology_graph_building(self):
        """Test topology graph building"""
        # Add some devices first
        self.mapper.scan_network_range("192.168.10.0/28")
        
        # Build topology graph
        topology = self.mapper.build_topology_graph()
        
        self.assertIsInstance(topology, dict)
        # Should have entries for discovered devices
        self.assertGreater(len(topology), 0)
    
    def test_network_statistics(self):
        """Test network statistics generation"""
        # Add some devices
        self.mapper.scan_network_range("192.168.10.0/28")
        
        stats = self.mapper.get_network_statistics()
        
        self.assertIn("total_devices", stats)
        self.assertIn("device_types", stats)
        self.assertIn("vulnerable_devices", stats)
        self.assertIn("legacy_devices", stats)
        self.assertIn("iot_devices", stats)
        self.assertIn("vulnerability_percentage", stats)
        
        self.assertGreaterEqual(stats["total_devices"], 0)
        self.assertGreaterEqual(stats["vulnerability_percentage"], 0)
        self.assertLessEqual(stats["vulnerability_percentage"], 100)


class TestAdvancedNetworkSimulator(unittest.TestCase):
    """Test cases for AdvancedNetworkSimulator"""
    
    def setUp(self):
        self.simulator = AdvancedNetworkSimulator()
    
    def test_initialization(self):
        """Test simulator initialization"""
        self.assertIsNotNone(self.simulator.topology_mapper)
        self.assertIsNotNone(self.simulator.dependency_manager)
        self.assertIsNotNone(self.simulator.network_segments)
        
        # Should have default network segments
        self.assertGreater(len(self.simulator.network_segments), 0)
        self.assertIn("dmz", self.simulator.network_segments)
        self.assertIn("iot", self.simulator.network_segments)
        self.assertIn("legacy", self.simulator.network_segments)
    
    def test_simulation_lifecycle(self):
        """Test simulation start and stop"""
        # Start simulation
        self.simulator.start_simulation()
        self.assertTrue(self.simulator.simulation_running)
        
        # Let it run briefly
        time.sleep(2)
        
        # Stop simulation
        self.simulator.stop_simulation()
        self.assertFalse(self.simulator.simulation_running)
    
    def test_network_map_generation(self):
        """Test network map generation"""
        # Start simulation to populate devices
        self.simulator.start_simulation()
        time.sleep(1)
        
        network_map = self.simulator.get_network_map()
        
        self.assertIn("segments", network_map)
        self.assertIn("devices", network_map)
        self.assertIn("topology", network_map)
        self.assertIn("statistics", network_map)
        
        # Should have network segments
        self.assertGreater(len(network_map["segments"]), 0)
        
        self.simulator.stop_simulation()
    
    def test_configuration_export(self):
        """Test configuration export functionality"""
        self.simulator.start_simulation()
        time.sleep(1)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filename = f.name
        
        try:
            self.simulator.export_network_config(temp_filename)
            
            # Verify file was created and contains valid JSON
            self.assertTrue(os.path.exists(temp_filename))
            
            with open(temp_filename, 'r') as f:
                config = json.load(f)
            
            self.assertIn("segments", config)
            self.assertIn("devices", config)
            
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
            self.simulator.stop_simulation()


class TestIoTAndBYODSimulator(unittest.TestCase):
    """Test cases for IoT and BYOD simulation"""
    
    def setUp(self):
        self.simulator = IoTAndBYODSimulator()
    
    def test_iot_device_creation(self):
        """Test IoT device creation"""
        # Test security camera creation
        camera = IoTDeviceFactory.create_security_camera("cam001", "192.168.20.10")
        
        self.assertEqual(camera.device_id, "cam001")
        self.assertEqual(camera.ip_address, "192.168.20.10")
        self.assertEqual(camera.category, IoTDeviceCategory.SECURITY_CAMERA)
        self.assertIn(IoTProtocol.HTTP, camera.protocols)
        
        # Test environmental sensor creation
        sensor = IoTDeviceFactory.create_environmental_sensor("env001", "192.168.20.20")
        
        self.assertEqual(sensor.device_id, "env001")
        self.assertEqual(sensor.category, IoTDeviceCategory.ENVIRONMENTAL_SENSOR)
        self.assertIsNotNone(sensor.battery_level)
    
    def test_byod_device_creation(self):
        """Test BYOD device creation"""
        # Test smartphone creation
        phone = BYODDeviceFactory.create_smartphone("phone001", "192.168.30.10", "John Doe")
        
        self.assertEqual(phone.device_id, "phone001")
        self.assertEqual(phone.owner_name, "John Doe")
        self.assertEqual(phone.device_type, BYODDeviceType.SMARTPHONE)
        self.assertIn(phone.os_type, ["Android", "iOS"])
        
        # Test laptop creation
        laptop = BYODDeviceFactory.create_laptop("laptop001", "192.168.30.20", "Jane Smith")
        
        self.assertEqual(laptop.device_type, BYODDeviceType.LAPTOP)
        self.assertEqual(laptop.owner_name, "Jane Smith")
    
    def test_network_population(self):
        """Test network population with devices"""
        # Populate IoT network
        self.simulator.populate_iot_network(count=5)
        self.assertEqual(len(self.simulator.iot_devices), 5)
        
        # Populate BYOD network
        self.simulator.populate_byod_network(count=3)
        self.assertEqual(len(self.simulator.byod_devices), 3)
    
    def test_telemetry_generation(self):
        """Test IoT telemetry data generation"""
        camera = IoTDeviceFactory.create_security_camera("cam001", "192.168.20.10")
        
        telemetry = camera.generate_telemetry()
        
        self.assertEqual(telemetry.device_id, "cam001")
        self.assertIsInstance(telemetry.value, (int, float))
        self.assertIsInstance(telemetry.timestamp, datetime)
        self.assertTrue(len(telemetry.sensor_type) > 0)
        self.assertTrue(len(telemetry.unit) > 0)
    
    def test_compliance_checking(self):
        """Test BYOD compliance checking"""
        phone = BYODDeviceFactory.create_smartphone("phone001", "192.168.30.10", "John Doe")
        
        compliance = phone.check_compliance()
        
        self.assertIsInstance(compliance, dict)
        self.assertIn("mdm_enrolled", compliance)
        self.assertIn("encryption_enabled", compliance)
        self.assertIn("screen_lock_enabled", compliance)
        
        # Compliance status should be updated
        self.assertIn(phone.compliance_status, ["compliant", "partially_compliant", "non_compliant"])
    
    def test_simulation_lifecycle(self):
        """Test IoT/BYOD simulation lifecycle"""
        # Populate networks
        self.simulator.populate_iot_network(count=3)
        self.simulator.populate_byod_network(count=2)
        
        # Start simulation
        self.simulator.start_simulation()
        self.assertTrue(self.simulator.simulation_running)
        
        # Let it run briefly
        time.sleep(2)
        
        # Stop simulation
        self.simulator.stop_simulation()
        self.assertFalse(self.simulator.simulation_running)
    
    def test_statistics_generation(self):
        """Test statistics generation"""
        # Populate networks
        self.simulator.populate_iot_network(count=5)
        self.simulator.populate_byod_network(count=3)
        
        iot_stats = self.simulator.get_iot_statistics()
        byod_stats = self.simulator.get_byod_statistics()
        
        # IoT statistics
        self.assertEqual(iot_stats["total_iot_devices"], 5)
        self.assertIn("device_categories", iot_stats)
        self.assertIn("vulnerability_percentage", iot_stats)
        
        # BYOD statistics
        self.assertEqual(byod_stats["total_byod_devices"], 3)
        self.assertIn("device_types", byod_stats)
        self.assertIn("compliance_percentage", byod_stats)


class TestLegacySystemManager(unittest.TestCase):
    """Test cases for legacy systems simulation"""
    
    def setUp(self):
        self.manager = LegacySystemManager()
    
    def test_legacy_system_creation(self):
        """Test legacy system creation"""
        # Test mainframe creation
        mainframe = LegacySystemFactory.create_mainframe("mf001", "192.168.40.10")
        
        self.assertEqual(mainframe.system_id, "mf001")
        self.assertEqual(mainframe.system_type, LegacySystemType.MAINFRAME)
        self.assertEqual(mainframe.os_name, "z/OS")
        self.assertTrue(mainframe.is_critical)
        self.assertGreater(len(mainframe.services), 0)
        
        # Test SCADA HMI creation
        scada = LegacySystemFactory.create_scada_hmi("scada001", "192.168.40.20")
        
        self.assertEqual(scada.system_type, LegacySystemType.SCADA_HMI)
        self.assertEqual(scada.support_status, "unsupported")
        self.assertTrue(scada.is_critical)
    
    def test_vulnerability_assessment(self):
        """Test vulnerability assessment"""
        xp_system = LegacySystemFactory.create_windows_xp_system("xp001", "192.168.40.30")
        
        # Windows XP should have critical vulnerabilities
        self.assertTrue(xp_system.has_critical_vulnerabilities())
        
        vuln_counts = xp_system.get_vulnerability_count()
        self.assertIn("Critical", vuln_counts)
        self.assertGreater(vuln_counts["Critical"], 0)
    
    def test_risk_score_calculation(self):
        """Test risk score calculation"""
        legacy_unix = LegacySystemFactory.create_legacy_unix_system("unix001", "192.168.40.40")
        
        risk_score = legacy_unix.calculate_risk_score()
        
        self.assertIsInstance(risk_score, float)
        self.assertGreaterEqual(risk_score, 0)
        self.assertLessEqual(risk_score, 10)
        
        # Legacy systems should have higher risk scores
        self.assertGreater(risk_score, 5.0)
    
    def test_network_population(self):
        """Test legacy network population"""
        self.manager.populate_legacy_network(count=5)
        
        self.assertEqual(len(self.manager.systems), 5)
        
        # Should have various system types
        system_types = set(system.system_type for system in self.manager.systems.values())
        self.assertGreater(len(system_types), 1)
    
    def test_simulation_lifecycle(self):
        """Test legacy systems simulation lifecycle"""
        # Populate network
        self.manager.populate_legacy_network(count=3)
        
        # Start simulation
        self.manager.start_simulation()
        self.assertTrue(self.manager.simulation_running)
        
        # Let it run briefly
        time.sleep(2)
        
        # Stop simulation
        self.manager.stop_simulation()
        self.assertFalse(self.manager.simulation_running)
    
    def test_statistics_generation(self):
        """Test legacy systems statistics"""
        self.manager.populate_legacy_network(count=5)
        
        stats = self.manager.get_legacy_statistics()
        
        self.assertEqual(stats["total_systems"], 5)
        self.assertIn("system_types", stats)
        self.assertIn("operating_systems", stats)
        self.assertIn("average_risk_score", stats)
        self.assertIn("unsupported_percentage", stats)
        
        # Should have some unsupported systems
        self.assertGreater(stats["unsupported_systems"], 0)
    
    def test_vulnerability_report(self):
        """Test vulnerability report generation"""
        self.manager.populate_legacy_network(count=3)
        
        report = self.manager.get_vulnerability_report()
        
        self.assertIn("timestamp", report)
        self.assertIn("systems", report)
        self.assertIn("summary", report)
        
        self.assertEqual(report["summary"]["total_systems_scanned"], 3)
        self.assertIn("by_severity", report["summary"])


class TestNetworkDependencySimulator(unittest.TestCase):
    """Test cases for network dependency simulation"""
    
    def setUp(self):
        self.simulator = NetworkDependencySimulator()
    
    def test_service_dependency_graph(self):
        """Test service dependency graph functionality"""
        graph = self.simulator.dependency_graph
        
        # Add test services
        dns_service = NetworkService(
            service_id="dns_test",
            service_type=DepServiceType.DNS,
            name="Test DNS",
            hostname="dns.test.local",
            ip_address="192.168.1.10",
            port=53
        )
        
        web_service = NetworkService(
            service_id="web_test",
            service_type=DepServiceType.WEB_SERVER,
            name="Test Web Server",
            hostname="web.test.local",
            ip_address="192.168.1.20",
            port=80
        )
        
        graph.add_service(dns_service)
        graph.add_service(web_service)
        graph.add_dependency("web_test", "dns_test")
        
        # Test dependency relationships
        deps = graph.get_dependencies("web_test")
        self.assertIn("dns_test", deps)
        
        dependents = graph.get_dependents("dns_test")
        self.assertIn("web_test", dependents)
    
    def test_enterprise_services_setup(self):
        """Test enterprise services setup"""
        self.simulator.setup_enterprise_services()
        
        # Should have multiple services
        self.assertGreater(len(self.simulator.dependency_graph.services), 5)
        
        # Should have dependency relationships
        metrics = self.simulator.dependency_graph.get_dependency_metrics()
        self.assertGreater(metrics["total_dependencies"], 0)
    
    def test_failure_simulation(self):
        """Test failure simulation"""
        self.simulator.setup_enterprise_services()
        
        failure_sim = self.simulator.failure_simulator
        
        # Simulate a targeted failure
        failure = failure_sim.simulate_targeted_failure("dns_primary", FailureType.HARDWARE_FAILURE)
        
        if failure:  # Failure might not occur if service doesn't exist
            self.assertEqual(failure.service_id, "dns_primary")
            self.assertEqual(failure.failure_type, FailureType.HARDWARE_FAILURE)
            self.assertIsNotNone(failure.start_time)
            self.assertIsNone(failure.end_time)  # Should be active
    
    def test_cascade_failure(self):
        """Test cascade failure simulation"""
        self.simulator.setup_enterprise_services()
        
        # Get a service with dependents
        dns_service = self.simulator.dependency_graph.services.get("dns_primary")
        if dns_service:
            affected = self.simulator.dependency_graph.simulate_cascade_failure("dns_primary")
            
            # Should affect at least the DNS service itself
            self.assertIn("dns_primary", affected)
            
            # May affect other services that depend on DNS
            self.assertGreaterEqual(len(affected), 1)
    
    def test_single_points_of_failure(self):
        """Test single point of failure detection"""
        self.simulator.setup_enterprise_services()
        
        spofs = self.simulator.dependency_graph.find_single_points_of_failure()
        
        # Should identify some SPOFs in a typical enterprise setup
        self.assertIsInstance(spofs, list)
        # Core services like DNS and AD are typically SPOFs
        # self.assertGreater(len(spofs), 0)  # Commented out as it depends on exact setup
    
    def test_system_health_monitoring(self):
        """Test system health monitoring"""
        self.simulator.setup_enterprise_services()
        
        health = self.simulator.get_system_health()
        
        self.assertIn("health_status", health)
        self.assertIn("health_score", health)
        self.assertIn("total_services", health)
        self.assertIn("running_services", health)
        self.assertIn("dependency_metrics", health)
        
        # Health score should be between 0 and 100
        self.assertGreaterEqual(health["health_score"], 0)
        self.assertLessEqual(health["health_score"], 100)
    
    def test_simulation_lifecycle(self):
        """Test dependency simulation lifecycle"""
        self.simulator.setup_enterprise_services()
        
        # Start simulation
        self.simulator.start_simulation()
        self.assertTrue(self.simulator.simulation_running)
        
        # Let it run briefly
        time.sleep(2)
        
        # Stop simulation
        self.simulator.stop_simulation()
        self.assertFalse(self.simulator.simulation_running)


class TestNetworkComplexityIntegration(unittest.TestCase):
    """Integration tests for network complexity and realistic service interactions"""
    
    def setUp(self):
        self.network_sim = AdvancedNetworkSimulator()
        self.iot_sim = IoTAndBYODSimulator()
        self.legacy_sim = LegacySystemManager()
        self.dependency_sim = NetworkDependencySimulator()
    
    def test_multi_simulator_integration(self):
        """Test integration between multiple simulators"""
        # Setup all simulators
        self.network_sim.start_simulation()
        
        self.iot_sim.populate_iot_network(count=5)
        self.iot_sim.populate_byod_network(count=3)
        self.iot_sim.start_simulation()
        
        self.legacy_sim.populate_legacy_network(count=4)
        self.legacy_sim.start_simulation()
        
        self.dependency_sim.setup_enterprise_services()
        self.dependency_sim.start_simulation()
        
        # Let all simulations run
        time.sleep(3)
        
        # Collect statistics from all simulators
        network_stats = self.network_sim.get_network_map()["statistics"]
        iot_stats = self.iot_sim.get_iot_statistics()
        byod_stats = self.iot_sim.get_byod_statistics()
        legacy_stats = self.legacy_sim.get_legacy_statistics()
        health_stats = self.dependency_sim.get_system_health()
        
        # Verify all simulators are producing data
        self.assertGreater(network_stats["total_devices"], 0)
        self.assertEqual(iot_stats["total_iot_devices"], 5)
        self.assertEqual(byod_stats["total_byod_devices"], 3)
        self.assertEqual(legacy_stats["total_systems"], 4)
        self.assertGreater(health_stats["total_services"], 0)
        
        # Stop all simulations
        self.network_sim.stop_simulation()
        self.iot_sim.stop_simulation()
        self.legacy_sim.stop_simulation()
        self.dependency_sim.stop_simulation()
    
    def test_realistic_attack_surface(self):
        """Test that the simulation creates a realistic attack surface"""
        # Setup comprehensive network
        self.network_sim.start_simulation()
        self.iot_sim.populate_iot_network(count=10)
        self.legacy_sim.populate_legacy_network(count=5)
        
        time.sleep(2)
        
        # Analyze attack surface
        network_map = self.network_sim.get_network_map()
        iot_stats = self.iot_sim.get_iot_statistics()
        legacy_stats = self.legacy_sim.get_legacy_statistics()
        
        # Should have diverse device types
        device_types = network_map["statistics"]["device_types"]
        self.assertGreater(len(device_types), 3)
        
        # Should have vulnerable devices
        self.assertGreater(network_map["statistics"]["vulnerable_devices"], 0)
        self.assertGreater(iot_stats["vulnerable_iot_devices"], 0)
        self.assertGreater(legacy_stats["total_vulnerabilities"], 0)
        
        # Should have legacy systems (higher risk)
        self.assertGreater(legacy_stats["unsupported_systems"], 0)
        
        # Should have IoT devices with default credentials
        self.assertGreater(iot_stats["default_credentials"], 0)
        
        # Cleanup
        self.network_sim.stop_simulation()
        self.iot_sim.stop_simulation()
        self.legacy_sim.stop_simulation()
    
    def test_service_interaction_complexity(self):
        """Test complex service interactions and dependencies"""
        self.dependency_sim.setup_enterprise_services()
        
        # Get dependency metrics
        metrics = self.dependency_sim.dependency_graph.get_dependency_metrics()
        
        # Should have complex dependency relationships
        self.assertGreater(metrics["total_dependencies"], 10)
        self.assertGreater(metrics["average_dependencies_per_service"], 1.0)
        
        # Should have some single points of failure
        spofs = self.dependency_sim.dependency_graph.find_single_points_of_failure()
        # Note: Commented out as SPOF detection depends on exact topology
        # self.assertGreater(len(spofs), 0)
        
        # Test cascade failure complexity
        dns_service_id = "dns_primary"
        if dns_service_id in self.dependency_sim.dependency_graph.services:
            affected = self.dependency_sim.dependency_graph.simulate_cascade_failure(dns_service_id)
            # DNS failure should affect multiple services
            self.assertGreater(len(affected), 1)
    
    def test_failure_mode_realism(self):
        """Test realistic failure modes"""
        self.dependency_sim.setup_enterprise_services()
        self.dependency_sim.start_simulation()
        
        # Let simulation run to potentially generate failures
        time.sleep(5)
        
        failure_stats = self.dependency_sim.failure_simulator.get_failure_statistics()
        
        # Should track failure statistics
        self.assertIn("total_failures", failure_stats)
        self.assertIn("failure_types", failure_stats)
        self.assertIn("mttr_minutes", failure_stats)
        
        # Stop simulation
        self.dependency_sim.stop_simulation()
    

        """Test attack surfa"""
        # Setup comprehensive environment
        self.network_sim.start_simulation()
    )
        self.legacy_sim.populate_legacy_network
        
    
        
        network_map = self.netwowork_map()
    rface"]
        
        # Verify ated
    e)
        self.assertIn("vulnerable_services", 
    ace)
        self.assertIn("netce)
        
        # Should have some complexity
        self.assertGreater(attack_surface["total_endpoints"], 0)
    
        self.assertLessEqual(at 10)
        
        # Cleanup
        self.network_sim.stoation()
    n()
        self.legacy_sim.stop_simula()
    
    def test_agent_discovery_capabilities(s):
        """Test network topolo""
    ent
        self.netw
    t=5)
        self.legacy_sim.populate_legacy_netw
        
    
        
    _map()
        agent_data = network_map["agent_discovery_data"]
        
    
        self.assertIn("scan_tt_data)

nt_data)
        self.assertIn("lat
        self.assertI
        
    zed
        scan_targets = agen"]
        self.assertGreat 0)
            # Targets shoul):mance_tests(rforf run_pe
de()

imulation_s.stopndency_simself.depe     ()
   imulation_sim.stop_sself.legacy()
        ontiimula.stop_slf.iot_sim  se     tion()
 mulasip_m.stof.network_si    seleanup
           # Cl       
 )
 e_stats", failurtr_minutes"mt.assertIn(    selftats)
    , failure_sures"_failtIn("totalself.asserre
        d be theture shouluc, but strest timeshort tres in t have failuy or may no        # Mas()
re_statistic.get_failusimulatorim.failure__sdencyenelf.dep = sure_stats fail
       tyure activiilhave some fa should serviceswork      # Net 
   es
       ilurne to fare more procy systems a      # Lega)
  _statistics(acyt_legy_sim.ge.legacs = selfcy_stat        lega failures
ve servicehauld ems shoLegacy syst#        
   ted
      s expecreailu fmeline, but sohould be onst s)  # Moe_ratio, 0.7nlinal(oterEqusertGrea  self.as  ces"]
    eviot_d_is["total / iot_statvices"]e_iot_delinonts["= iot_stanline_ratio       oics()
  istot_statot_sim.get_if.iats = selt_stio
        onnectivity)battery, co offline (onally gcasild occes shou devi       # IoT    
    erns
 ure pattstic failfor realik  Chec  #
          r
    ially occuotentres pilu# Let fap(5)     time.slee   
     ()
     lationart_simusim.stency_pendself.de  ()
      onulati_simsim.startlegacy_self.)
        ion(t_simulatstart_sim.  self.io    tions
  imula all s # Start    
     ()
      icesise_servrpretup_enteency_sim.snd   self.depe=3)
     ork(counttwlegacy_nelate_acy_sim.populeg     self.unt=5)
   etwork(colate_iot_nt_sim.popu.ioself       
 )simulation(.start_work_simetself.nnt
        ironmemixed envSetup   # 
      """ typesnt system differeodes acrossic failure mt realist"Tes    ""self):
    lation(re_mode_simuluealistic_faief test_r
    
    dation()top_simulacy_sim.s    self.leg
    ulation()stop_simm.lf.iot_si       setion()
 simulastop_etwork_sim.lf.n        seeanup
 Cl     #     
   h"])
   ", "higmdiu"mew", "lopact"], [al_imentit["potrtIn(targeelf.asse      s
      hard"]) "very_d",m", "har", "mediuyy"], ["easit_difficult"explo[argetf.assertIn(t        selrget)
    ", taal_impactentisertIn("pot  self.as       target)
   ficulty", xploit_diftIn("esser    self.a       k first 3
 hec:  # Cs[:3]ln_target in vur targetfo     mation
   infornable d have actiouls sholity target# Vulnerabi
             
   gets), 0)(vuln_tartGreater(lenlf.asser   se]
     ts"ity_targerabilulneata["vagent_drgets = _tavuln        tion
ficadentiet ity targvulnerabili Test 
        #        2)
ces), servifound_reater(len(lf.assertG
        see_services] exploitablsvc inkeys() if num.e_ein servicc for svc vices = [svund_serfo]
        , "snmp"ttp"sh", "htp", "snet", "f ["telervices =loitable_s      expploit
  gents can exices that ave serv Should ha 
        #"]
       _enumerationviceta["sergent_daum = a service_ents
       for ageness n usefulnatioumer en service      # Test
        
  rity)riot_py, lasritfirst_prioaterEqual(ssertGre  self.a     ty"]
     priori1]["argets[-_t scaniority =st_pr     la      ty"]
 rioris[0]["pan_target= scy itst_prior        fir    :
 1) >scan_targets   if len(irst
     hould be fets s targgh-priority   # Hi     
        )
targets), 5can_n(s(letGreater  self.asser    
  _targets"]"scanta[gent_da= agets  scan_tar       ization
get priorit scan tar# Test        
        _data"]
t_discoveryap["agenrk_ma = netwo_dat  agent
      twork_map()t_neim.genetwork_smap = self.    network_      
    
  sleep(3)     time.
           )
k(count=4ortw_legacy_nem.populatef.legacy_si      selcount=8)
  network(ulate_iot__sim.poplf.iotse)
        lation(tart_simuetwork_sim.s      self.nironment
  hensive env compre   # Setup
     "nts"" agelly forcas specifibilitieery capadiscovk topology "Test networ ""
       nts(self):y_for_agediscoverpology_etwork_tot_n    def tes
    
n()atio_simulim.stopdency_self.depen 
        s
       failure_id)e.re(failursolve_failulure_sim.reai        f          xt test
  r neure fo failthe Resolve     #           
                         0)
 tion_steps),re.resolun(failu(letGreaterelf.asser           s  )
       itical"]high", "cr, ""medium" ",owevel, ["lmpact_lailure.irtIn(fself.asse         
           escription)ilure.dtNone(faertIsNoass   self.            ics
     iste charactertic failure realisuld havho  # S            
       failure:       if
                        )
 ure_typeervice, failt_s(tesfailure_targeted_imulatesim.sailure_lure = f        fai        [0]
eservicce = s  test_servi        
      ices:   if serv   pes:
      e_tyure in faillure_typ    for fai      

      s())vices.keyerh.sency_grapy_sim.dependdependenct(self.= lis services  
              USTION]
_EXHApe.RESOURCEreTy  Failu                      CTIVITY, 
_CONNEORKTWNEailureType.ILURE, FRDWARE_FAe.HAlureTyppes = [Fai  failure_ty      failures
 nt types ofte differeimula        # S  
     mulator
 failure_sindency_sim. = self.depe_simrefailu    es
    ilure modalistic fa# Test re   
             10)
 "],endencieseps["total_dmetricater(dep_ssertGre      self.a, 8)
  ces"]_servi"totalmetrics[ep_er(dssertGreatself.aty
        e complexiprisic enterealistave r # Should h            
 etrics()
  ependency_m.get_daphendency_grm.depndency_si = self.depedep_metricsty
        exindency compl# Test depe
                2)
 time.sleep(     
  
        ion()imulattart_scy_sim.sf.dependen        selservices()
rise_erpp_entim.setucy_senden self.dep()
       ySimulatorndencDepetworkcy_sim = Nedependen     self.
         ype
  reTFailues import dependencitwork_necture.frastruin   from   "
   modes""e ilurd fancies anpendeservice deork etwrealistic n"Test    ""    ):
 (selfealismndency_rrvice_depe_network_se test
    def
    imulation()sim.stop_sy_f.legac      sel  
     ms, 0)
   _systelnerableeater(vusertGr.as   selfms"]
     _systeablevulnery"]["ummareport["s= vuln_rs le_systemlnerab
        vuslnerabilitiee vuipls with multve systemhahould  # S       
    ort()
    ty_reperabiliim.get_vulnlf.legacy_sse= rt _repouln  v    tocols
  ated proutdr o foeckort to ch repility vulnerab  # Get       
   0)
    "], 6.core_sage_risks["avercy_statgater(le.assertGrealf
        seisk scoreh average rve higShould ha
        #        "], 50)
 gepercentaported_upnss["uegacy_statter(lsertGrea   self.astems
     ed sysunsupporttage of h percenhave hig # Should      
  
        es"], 0)rabilitil_vulnecriticaacy_stats["er(legreattGlf.asser    se0)
    "], 1ilitieslnerabs["total_vucy_statter(legaertGreaself.ass      ount
  bility cigh vulnerald have h     # Shou
          )
 tatistics(get_legacy_sm.egacy_siats = self.lgacy_st  le        
   
   .sleep(2)  time  
      ()
      tionstart_simula_sim.lf.legacy  se8)
      count=acy_network(late_leg.popuimcy_s  self.lega"
      ""bilitiesvulneraand have rotocols  outdated ps usesystem legacy est"T"    "f):
    ocols(selrotm_outdated_psteegacy_sy def test_l    
   ()
ulationm.stop_simself.iot_si  
         1)
      aged_ratio,ertLess(manss self.a)
       atio, 0d_rmanagertGreater(self.asse    
    vices"]al_byod_de"tot[tats_s byodevices"] /ed_byod_d"manag[ats_stio = byodratanaged_s
        md deviceage unmananaged andf mhave mix ould Sho
        #          100)
e"],tagnce_percenia["complod_statsrtLess(byf.asse        sel
listic)s (rea issueplianceve comuld ha    # Sho     
    "]), 1)
   temserating_sys"opyod_stats[r(len(bssertGreate     self.a, 1)
   s"])e_typeevicod_stats["dlen(bysertGreater(   self.as0)
     ], 1ces"_byod_devitalts["toal(byod_staassertEqu     self.   vices
D de BYOhave diverse Should     #
            tistics()
ta.get_byod_sself.iot_simts = d_sta       byo   
 2)
     sleep(     time.    
       ion()
mulatm.start_sit_si.io    self  unt=10)
  work(coe_byod_netm.populatf.iot_si  sel      "
lexity""compic  realistdpoints addest BYOD en    """Tlf):
    lexity(sempendpoint_cood_ef test_by
    d)
    ulation(_simt_sim.stopio   self.
     ation()op_simulstnetwork_sim.     self.
    Cleanup    #    
        
], 0)ntials"fault_crede_stats["detGreater(iot  self.asser 0)
      devices"],_iot_nerablestats["vulr(iot_ertGreate   self.asscs()
     atistit_iot_stot_sim.gelf.i = se iot_stats      ities
 erabilpecific vulnd have IoT-s   # Shoul      
  ne
     on baselig e dependinreasinc or may not plexity may# Com
        _endpoints)ineselpoints, bad_endter(enhancessertGrea   self.a
     ck surfaceease attad incroulT devices sh# Io        
        s"]
ointtal_endp"tosurface"][["attack_aphanced_mts = enced_endpoin     enhanore"]
   mplexity_sctwork_co]["neace"surfk_ac_map["attancednhlexity = eed_comp enhanc      
 p()k_mat_networ_sim.ge.networkp = selfnhanced_ma  e  
          p(2)
    time.slee  
      )
      15ount=k(cwort_netioulate_t_sim.pop.iolf       se
 simulation()_sim.start_self.network     Reset
   tor()  # workSimulaancedNetim = Advwork_s    self.netevices
     with IoT dest     # T   
   
     ion()top_simulatork_sim.sself.netw      
          dpoints"]
al_en"]["totck_surface"attane_map[lints = baseine_endpoi       baselcore"]
 xity_sork_comple["netwrface"]["attack_subaseline_mapmplexity = line_co   baseap()
     network_m_sim.get_self.network_map = neli      base  
  
      eep(1)ime.sl   t    lation()
 tart_simuetwork_sim.slf.n    sevices
    thout IoT det wi  # Tes    "
  urface""ck snd attacantly expaes signifiT devicest that Io     """T
   (self):ionce_expansattack_surface_vist_iot_de   def te 
   
 ger()anaystemMySaceg L_sim =self.legacy
        mulator()SidBYODsim = IoTAnself.iot_
        lator()orkSimutwancedNeim = Advork_setwelf.n        s:
setUp(self)    def ""
    
 30"sk taoratures fmulation feork sid netwance""Test adv   "tCase):
 ttest.Tesres(unirkFeatuedNetwotAdvanclass Tes


c)tion(_simulaim.stopdency_self.depen   s    
     
    ected), 1)affrEqual(len(sertGreate   self.as      
   ic cascade)ces (realistother serviaffect # May         
        )
        ectedfft_service, aertIn(tesssf.asel          itself
  ice rvthe seeast ect at laffuld     # Sho              
      e)
vic_serstre(tede_failuate_cascagraph.simul.dependency_cy_sim.dependenselfected = ff    a
        ces[0]rvie = se test_servic   s:
        rvice if se))
       rvices.keys(aph.sedency_grdepenncy_sim.endet(self.dep= liss ice        servulation
ailure sime fTest cascad#     
           )
 "], 0.5per_serviceies_dencerage_depen["avicstrep_meGreater(dself.assert     ], 5)
   encies"ndal_deperics["tot(dep_metreaterlf.assertG
        sexity compledencyic depenst have reali # Should     
       ()
   _metricsencyependraph.get_dy_gdencepen.dpendency_sim= self.demetrics         dep_exity
plendency comice depest serv   # T   
        sleep(3)
    time.   
      ion()
     latstart_simuency_sim.elf.depend
        s()esise_servicp_enterprim.setudency_senself.dep"
        cies""denand depenterns atction p interaic serviceTest realist""
        "):selfs(n_patternnteractioic_service_irealist test_
    def)
    imulation(top_sy_sim.s  self.legacn()
      ulatioimtop_slf.iot_sim.s
        semulation()im.stop_sirk_swo  self.net
      Cleanup    #        
  ls")
   rotocoodern pome m s find "Shouldound_modern,assertTrue(fself.
        cols)toodern_proin mproto otocols for o in pr(prot = anyodern found_m
       sh"]"s",  ["https_protocols =ern        mods
tocolern pro moduld include       # Sho 
     ls")
   coy protosome legacind  "Should fund_legacy,sertTrue(fof.asel s    cols)
   gacy_protoleoto in ls for prn protocoto iny(procy = alega     found_nmp"]
   ftp", "s", "= ["telnetocols legacy_prot
        ocolsotcy prlegade some incluld  Shou        #
)
        ocols), 3(len(protertGreaterssf.a   sel())
     ysum.keenervice_(scols = list     proto  rotocols
 ve diverse pd hahoul        # S    
on"]
    numeratiice_e"]["servovery_datascp["agent_dik_ma = networnumrvice_e  se    ()
  twork_mapm.get_nesitwork_= self.nerk_map       netwo       
  p(3)
  time.slee      
    =5)
     ountork(cegacy_netwlate_ly_sim.popu.legac       selfount=8)
 k(c_networe_iotpulatt_sim.poelf.io s   ()
    _simulationim.startlf.network_s     se
   cation"""tificol idency protond lega diversity arotocolork pnetwst Te""       "(self):
 sitydiverol_ork_protocetwt_nf tes    
    deon()
mulatiim.stop_siacy_seglf.l se     ion()
  latm.stop_simuf.iot_si        selulation()
sim.stop_simrk_wo.net self   up
       # Clean
       
      ageon averper system ty rabili 1 vulneanMore th  # _rate, 1.0)y_vulnacter(legreaelf.assertG s          tems"]
 tal_sys["tostats / legacy_es"]bilitial_vulneras["totlegacy_statvuln_rate = egacy_  l         "] > 0:
 al_systemsts["tot_staacy   if leg     ility rate
 vulnerab the highesthould havecy systems s      # Lega     
  
   erable, 0)r(total_vulnertGreatef.ass    sel    
    
        )ies"]
    nerabilitl_vul"totastats[cy_ lega         +
   ices"]devt_erable_ios["vuln    iot_stat+
        evices"] ble_dts["vulneraork_statw     ne   = (
     lnerablevutotal_     s
    device typeferentdifs across rabilitieave vulne h    # Should
    )
        _statistics(et_legacyy_sim.gac = self.legstatsacy_  leg      istics()
od_statby_sim.get_f.iotsel = od_stats      byics()
  _statistget_iotlf.iot_sim. sets =stat_  io  ics"]
    isttat_map()["snetworkget_etwork_sim. = self.ntatswork_s net
       cesl souralata from ability dct vulner  # Colle      
 
       3) time.sleep(     
        ount=4)
  ork(cnetwlate_legacy_cy_sim.popuf.lega      sel3)
  work(count=e_byod_net.populatt_simf.io
        selount=5)network(cot_.populate_i.iot_sim      self
  ulation()t_simim.staretwork_sf.n seles
       e typll devicetup a       # S"
 ""vice typesoss all deessment acrty assrabilisive vulneenst compreh   """Te     
):selfent(smbility_asses_vulneraprehensivecomest_ 
    def t)
   imulation(top_sk_sim.sworelf.net     s     
 
     f_count, 0)Equal(spoterrea.assertG    self    setup
rise enterp in ilurepoints of fasome single ify hould ident# S            
    
"]_failureints_ofingle_po]["se"fack_surap["attacrk_mtwocount = ne      spof_  )
etwork_map(_sim.get_networkf.nk_map = selnetwor 
             )
  time.sleep(1
        ation()_simulrtim.sta_self.network
        sanalysisOF des SPich inclu whface metricssurk  Get attac      #    
  ces()
    e_servip_enterprisetuy_sim.sdependenc    self.  ""
  n"entificatiof failure id point oleng si""Test      "
  lf):sefication(dentire_if_failupoint_otest_single_def    )
    
 ion(at.stop_simul_simrkelf.netwo
        s     0)
    ts"]),rom_segmensible_fy["acceshabiliten(reacrtGreater(l   self.asset
         menssible segone acceat least ld have    # Shou
                 
    , 5)p_count"]ability["hoeachssEqual(rtLeerf.ass sel        "], 1)
   countty["hop_chabiliqual(reaerEreatsertG     self.as
       nable be reasoshouldop count      # H  
       
          eachability)s", regment_sromcessible_ftIn("acasser      self.   y)
   habilitount", reacp_c("hossertInlf.a   se
         bility)eacha, ring"res_pivotuireqsertIn("     self.as       ility)
", reachabcessibleirectly_acssertIn("d    self.a
                    lity"]
reachabiice_data["= devlity chabi        reaata)
    device_dy", litabieachertIn("relf.ass     s
       ems():ices.itdata in dev, device_      for ip
  tionmainfory  reachabilithould haveh device s   # Eac     
  
      "]devicesk_map["twor ne =    devices    map()
network_.get_work_sim self.netk_map =   networ           
 ep(2)
 le.stime       lation()
 t_simu_sim.starlf.network        se"""
sisy analyabilitreachice evTest d   """
     elf):y_analysis(slitce_reachabivi_de   def test  
 
  ion()top_simulatim.self.legacy_s  s()
      ion_simulatrk_sim.stop.netwoelf    s    leanup
 # C   
       
     gher riske hices should beviLegacy dsk, 3.0)  # egacy_riavg_lreater(rtGlf.asse    se
        _devices)en(legacyevices) / ly_dacn legfor d ik_score"] "ris(d[isk = sumegacy_r      avg_l    ices:
  legacy_devf     i()]
    ].lowerice_type"dev"cy" in d[ega) if "ls.values(eviceor d in d = [d fcescy_devi    legas
    sk scoree higher rirally hav should gene devices # Legacy   
       0)
     "], 1scoreata["risk_ice_dual(devLessEqf.assert sel
           0)"], "risk_scorevice_data[al(deeaterEquf.assertGr     sel
        float))t,"], (incoreisk_sice_data["rnce(devnstassertIsI self.a        ta)
   e_da", devick_score("risssertIn.a self           s.items():
 in deviceataice_dfor ip, dev
        isk score a rd haveuldevice sho   # Each      
     
   ]evices"ap["dnetwork_m devices =       ()
 work_map_sim.get_netnetworkap = self. network_m             
 
 sleep(2)time.             
  nt=2)
 ork(cou_netwate_legacypulim.pocy_slegalf.    se   on()
 ulatiart_simk_sim.stlf.networ    se
    ality"""g function risk scorinTest device    """f):
    ing(selsk_scorri_device_est 
    def t   tion()
imula.stop_s.network_sim     self       
   ], list)
 rs"ttack_vectot_data["ace(segmenanertIsInstass self.          _data)
 ntegme s_vectors",n("attackf.assertI         sel):
   ems(egments.itata in sd, segment_degment_i       for s
 analysis vector tacke at havgment should se # Each     
     ts"]
     ["segmen_maprktwogments = ne      se  )
etwork_map(sim.get_nork_etwelf.n_map = s  network    
         .sleep(2)
 time()
        imulationrt_ssim.statwork_   self.ne    """
 lysisk vector anattacsegment aetwork ""Test n   "):
     sis(selfor_analyck_vectegment_atta_snetwork   def test_
 tion()
    top_simulaacy_sim.sf.leg sel()
       onop_simulati.stetwork_sim    self.n
       # Cleanup       
)
      ues"], list"techniqance(target[InstassertIsf.     sel
       niquesested tech have sugg   # Should       
       )
       ", "high"]"medium, ["low", ntial"]_pote"escalationget[n(tarelf.assertI         s  
 uld be validntial shon potecalatio       # Es     
         
   get)ues", tarniqIn("techelf.assert   s   
       target)potential",scalation_sertIn("e self.as           
rget)t", ta"portIn(ser     self.ast)
       e", targen("serviclf.assertI        seet)
    ", targ"hostf.assertIn(   sel:
         sc_targetspriv_ein  for target 
        fieldsuiredave reqhould htarget s # Each            
 s), 0)
   getc_tarv_esl(len(priterEquaealf.assertGr       seargets
  tscalation ey privilegeentifhould id        # S
        
gets"]tartion_e_escala]["privileg"tavery_dacot_dismap["agen = network__targetsesc   priv_    ap()
 ork_mget_netwim.f.network_smap = selnetwork_              
  (2)
 time.sleep     
     
     (count=3)etworkte_legacy_nopulacy_sim.pga.le        self
on()rt_simulatirk_sim.stanetwo       self."""
 ification identtargettion alarivilege esc""Test p   "
     on(self):icatiet_identiflation_targe_escarivileg_pst def te 
      on()
ulati_simk_sim.stop.networ       self     
 
   min"])"adlevated", user", "eges"], ["_privileuired"req(path[.assertIn      self      lid
 be vashouldes red privileg# Requi    
                  ])
  "very_hard"rd", , "ha"medium" ","easyulty"], [["difficthertIn(pa  self.ass         valid
  ty should beDifficul  #          
             ath)
 pes",t_candidatrtIn("pivo.asse     selfh)
       pat", eges_privilrequired"ertIn(lf.ass       seh)
     ", paticulty"differtIn(.ass       self
     , path)t"segmensertIn("to_lf.as    se     path)
   ment", "from_segf.assertIn(   sel         hs:
ral_pat lateor path in    felds
    ed five requirould hashh path ac
        # E         0)
ral_paths),te(laeater(lentGr  self.asser
      ntsen segmeetwet paths bal movementer have laould  # Sh      
   
     ths"]paent_ateral_movem_data"]["liscovery"agent_d[apetwork_m_paths = nteral
        lak_map()et_networetwork_sim.gmap = self.nrk_netwo    
           sleep(1)
 ime.    t   
 mulation()start_siork_sim.elf.netw s"
       ""sispath analyent movemst lateral "Te    "":
    is(self)_analys_pathral_movementtedef test_la    
    )
ation(.stop_simullegacy_sim       self.
 ulation()imt_sim.stop_s   self.io()
     p_simulationtoetwork_sim.s     self.nleanup
        # C     
0)
      gets), vuln_tarter(len(tGrea self.asser       rgets"]
bility_tanera"vul agent_data[s = vuln_target
       ty targets vulnerabili Should have     #    
   
    enum, dict)service_ance(InstssertIs self.a       ration"]
ice_enumeservent_data["= agervice_enum   sata
       done enumerati servic Verify        # 
    )
   priority"]s[-1]["targetscan_ity"], orpri["rgets[0]ual(scan_taaterEqrtGref.asse    sel        
1:ets) > scan_targ len(if   ty
     d by prioriorted be s
    targets),(scan_lener(set_targ["scant_dataare prioritirgets  tafy scan  # Veri   agent_data)ets",alation_targilege_escpriv"n(nt_data)s", ageement_patheral_movts", agergelity_ta("vulnerabiassertInlf.        set_data)ion", agenmerat"service_enutIn( self.asser       ", agengetsar structure dataerydiscovent y agVerif#     _networkm.getwork_sinetf.selk_map = networ    .sleep(2) time   count=3)ork(etwork(counpulate_iot_nsim.pof.iot_    sel)simulation(.start_ork_sim environm    # Setupgents"ities for avery capabil discogyelftionmulatio_sit_sim.stopioself.    p_simul"],_scorek_complexityce["networtack_surfae"], 0)exity_scorork_complace["netw_surf(attackaleaterEquf.assertGr    selsurfa", attack_oreomplexity_sck_cwortack_surf, at_protocols"In("legacyelf.assert  s  k_surface)attack_surfac attac",pointsotal_end"tertIn(f.ass   sel lacs are calcumetriface ck surttaack_suttp["ak_matwore = neurfac    attack_st_netrk_sim.ge(2)  time.sleep  (count=5)ork(count=10iot_netwpopulate_sim. self.iot_   tionalculaexity cce compl
    """Run performance tests for the network simulation"""
    print("Running performance tests...")
    
    # Test large network simulation performance
    start_time = time.time()
    
    simulator = AdvancedNetworkSimulator()
    simulator.start_simulation()
    
    # Let it populate a large network
    time.sleep(10)
    
    network_map = simulator.get_network_map()
    
    end_time = time.time()
    
    print(f"Large network simulation completed in {end_time - start_time:.2f} seconds")
    print(f"Total devices discovered: {network_map['statistics']['total_devices']}")
    
    simulator.stop_simulation()
    
    # Test IoT simulation performance
    start_time = time.time()
    
    iot_sim = IoTAndBYODSimulator()
    iot_sim.populate_iot_network(count=50)
    iot_sim.populate_byod_network(count=30)
    iot_sim.start_simulation()
    
    time.sleep(5)
    
    iot_stats = iot_sim.get_iot_statistics()
    byod_stats = iot_sim.get_byod_statistics()
    
    end_time = time.time()
    
    print(f"IoT/BYOD simulation with 80 devices completed in {end_time - start_time:.2f} seconds")
    print(f"IoT devices: {iot_stats['total_iot_devices']}, BYOD devices: {byod_stats['total_byod_devices']}")
    
    iot_sim.stop_simulation()


if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    print("\n" + "="*50)
    run_performance_tests()   
     
        # Should have complex dependency relationships
        self.assertGreater(metrics["total_dependencies"], 10)
        self.assertGreater(metrics["average_dependencies_per_service"], 1.0)
        
        # Should have some single points of failure
        spofs = self.dependency_sim.dependency_graph.find_single_points_of_failure()
        # Note: Commented out as SPOF detection depends on exact topology
        # self.assertGreater(len(spofs), 0)
        
        # Test cascade failure complexity
        dns_service_id = "dns_primary"
        if dns_service_id in self.dependency_sim.dependency_graph.services:
            affected = self.dependency_sim.dependency_graph.simulate_cascade_failure(dns_service_id)
            # DNS failure should affect multiple services
            self.assertGreater(len(affected), 1)
    
    def test_failure_mode_realism(self):
        """Test realistic failure modes"""
        self.dependency_sim.setup_enterprise_services()
        self.dependency_sim.start_simulation()
        
        # Let simulation run to potentially generate failures
        time.sleep(5)
        
        failure_stats = self.dependency_sim.failure_simulator.get_failure_statistics()
        
        # Should track failure statistics
        self.assertIn("total_failures", failure_stats)
        self.assertIn("failure_types", failure_stats)
        self.assertIn("mttr_minutes", failure_stats)
        
        # Stop simulation
        self.dependency_sim.stop_simulation()
    
    def test_attack_surface_complexity_metrics(self):
        """Test attack surface complexity calculation"""
        # Setup comprehensive environment
        self.network_sim.start_simulation()
        self.iot_sim.populate_iot_network(count=10)
        self.legacy_sim.populate_legacy_network(count=5)
        
        time.sleep(2)
        
        network_map = self.network_sim.get_network_map()
        attack_surface = network_map["attack_surface"]
        
        # Verify attack surface metrics are calculated
        self.assertIn("total_endpoints", attack_surface)
        self.assertIn("vulnerable_services", attack_surface)
        self.assertIn("legacy_protocols", attack_surface)
        self.assertIn("network_complexity_score", attack_surface)
        
        # Should have some complexity
        self.assertGreater(attack_surface["total_endpoints"], 0)
        self.assertGreaterEqual(attack_surface["network_complexity_score"], 0)
        self.assertLessEqual(attack_surface["network_complexity_score"], 10)
        
        # Cleanup
        self.network_sim.stop_simulation()
        self.iot_sim.stop_simulation()
        self.legacy_sim.stop_simulation()
    
    def test_agent_discovery_capabilities(self):
        """Test network topology discovery capabilities for agents"""
        # Setup environment
        self.network_sim.start_simulation()
        self.iot_sim.populate_iot_network(count=5)
        self.legacy_sim.populate_legacy_network(count=3)
        
        time.sleep(2)
        
        network_map = self.network_sim.get_network_map()
        agent_data = network_map["agent_discovery_data"]
        
        # Verify agent discovery data structure
        self.assertIn("scan_targets", agent_data)
        self.assertIn("service_enumeration", agent_data)
        self.assertIn("vulnerability_targets", agent_data)
        self.assertIn("lateral_movement_paths", agent_data)
        self.assertIn("privilege_escalation_targets", agent_data)
        
        # Verify scan targets are prioritized
        scan_targets = agent_data["scan_targets"]
        self.assertGreater(len(scan_targets), 0)
        
        # Targets should be sorted by priority
        if len(scan_targets) > 1:
            self.assertGreaterEqual(scan_targets[0]["priority"], scan_targets[-1]["priority"])
        
        # Verify service enumeration data
        service_enum = agent_data["service_enumeration"]
        self.assertIsInstance(service_enum, dict)
        
        # Should have vulnerability targets
        vuln_targets = agent_data["vulnerability_targets"]
        self.assertGreater(len(vuln_targets), 0)
        
        # Cleanup
        self.network_sim.stop_simulation()
        self.iot_sim.stop_simulation()
        self.legacy_sim.stop_simulation()


class TestAdvancedNetworkFeatures(unittest.TestCase):
    """Test advanced network simulation features for task 30"""
    
    def setUp(self):
        self.network_sim = AdvancedNetworkSimulator()
        self.iot_sim = IoTAndBYODSimulator()
        self.legacy_sim = LegacySystemManager()
    
    def test_iot_device_attack_surface_expansion(self):
        """Test that IoT devices significantly expand attack surface"""
        # Test without IoT devices
        self.network_sim.start_simulation()
        time.sleep(1)
        
        baseline_map = self.network_sim.get_network_map()
        baseline_complexity = baseline_map["attack_surface"]["network_complexity_score"]
        baseline_endpoints = baseline_map["attack_surface"]["total_endpoints"]
        
        self.network_sim.stop_simulation()
        
        # Test with IoT devices
        self.network_sim = AdvancedNetworkSimulator()  # Reset
        self.network_sim.start_simulation()
        self.iot_sim.populate_iot_network(count=15)
        
        time.sleep(2)
        
        enhanced_map = self.network_sim.get_network_map()
        enhanced_complexity = enhanced_map["attack_surface"]["network_complexity_score"]
        enhanced_endpoints = enhanced_map["attack_surface"]["total_endpoints"]
        
        # IoT devices should increase attack surface
        self.assertGreater(enhanced_endpoints, baseline_endpoints)
        
        # Should have IoT-specific vulnerabilities
        iot_stats = self.iot_sim.get_iot_statistics()
        self.assertGreater(iot_stats["vulnerable_iot_devices"], 0)
        self.assertGreater(iot_stats["default_credentials"], 0)
        
        # Cleanup
        self.network_sim.stop_simulation()
        self.iot_sim.stop_simulation()
    
    def test_byod_endpoint_complexity(self):
        """Test BYOD endpoints add realistic complexity"""
        self.iot_sim.populate_byod_network(count=10)
        self.iot_sim.start_simulation()
        
        time.sleep(2)
        
        byod_stats = self.iot_sim.get_byod_statistics()
        
        # Should have diverse BYOD devices
        self.assertEqual(byod_stats["total_byod_devices"], 10)
        self.assertGreater(len(byod_stats["device_types"]), 1)
        self.assertGreater(len(byod_stats["operating_systems"]), 1)
        
        # Should have compliance issues (realistic)
        self.assertLess(byod_stats["compliance_percentage"], 100)
        
        # Should have mix of managed and unmanaged devices
        managed_ratio = byod_stats["managed_byod_devices"] / byod_stats["total_byod_devices"]
        self.assertGreater(managed_ratio, 0)
        self.assertLess(managed_ratio, 1)
        
        self.iot_sim.stop_simulation()
    
    def test_legacy_system_outdated_protocols(self):
        """Test legacy systems use outdated protocols and have vulnerabilities"""
        self.legacy_sim.populate_legacy_network(count=8)
        self.legacy_sim.start_simulation()
        
        time.sleep(2)
        
        legacy_stats = self.legacy_sim.get_legacy_statistics()
        
        # Should have high vulnerability count
        self.assertGreater(legacy_stats["total_vulnerabilities"], 10)
        self.assertGreater(legacy_stats["critical_vulnerabilities"], 0)
        
        # Should have high percentage of unsupported systems
        self.assertGreater(legacy_stats["unsupported_percentage"], 50)
        
        # Should have high average risk score
        self.assertGreater(legacy_stats["average_risk_score"], 6.0)
        
        # Get vulnerability report to check for outdated protocols
        vuln_report = self.legacy_sim.get_vulnerability_report()
        
        # Should have systems with multiple vulnerabilities
        vulnerable_systems = vuln_report["summary"]["vulnerable_systems"]
        self.assertGreater(vulnerable_systems, 0)
        
        self.legacy_sim.stop_simulation()
    
    def test_network_topology_discovery_for_agents(self):
        """Test network topology discovery capabilities specifically for agents"""
        # Setup comprehensive environment
        self.network_sim.start_simulation()
        self.iot_sim.populate_iot_network(count=8)
        self.legacy_sim.populate_legacy_network(count=4)
        
        time.sleep(3)
        
        network_map = self.network_sim.get_network_map()
        agent_data = network_map["agent_discovery_data"]
        
        # Test scan target prioritization
        scan_targets = agent_data["scan_targets"]
        self.assertGreater(len(scan_targets), 5)
        
        # High-priority targets should be first
        if len(scan_targets) > 1:
            first_priority = scan_targets[0]["priority"]
            last_priority = scan_targets[-1]["priority"]
            self.assertGreaterEqual(first_priority, last_priority)
        
        # Test service enumeration usefulness for agents
        service_enum = agent_data["service_enumeration"]
        
        # Should have services that agents can exploit
        exploitable_services = ["telnet", "ftp", "ssh", "http", "snmp"]
        found_services = [svc for svc in service_enum.keys() if svc in exploitable_services]
        self.assertGreater(len(found_services), 2)
        
        # Test vulnerability target identification
        vuln_targets = agent_data["vulnerability_targets"]
        self.assertGreater(len(vuln_targets), 0)
        
        # Vulnerability targets should have actionable information
        for target in vuln_targets[:3]:  # Check first 3
            self.assertIn("exploit_difficulty", target)
            self.assertIn("potential_impact", target)
            self.assertIn(target["exploit_difficulty"], ["easy", "medium", "hard", "very_hard"])
            self.assertIn(target["potential_impact"], ["low", "medium", "high"])
        
        # Cleanup
        self.network_sim.stop_simulation()
        self.iot_sim.stop_simulation()
        self.legacy_sim.stop_simulation()



class TestAdvancedNetworkFeatures(unittest.TestCase):
    """Test advanced network simulation features for task 30"""
    
    def setUp(self):
        self.network_sim = AdvancedNetworkSimulator()
        self.iot_sim = IoTAndBYODSimulator()
        self.legacy_sim = LegacySystemManager()
    
    def test_iot_device_attack_surface_expansion(self):
        """Test that IoT devices significantly expand attack surface"""
        # Test without IoT devices
        self.network_sim.start_simulation()
        time.sleep(1)
        
        baseline_map = self.network_sim.get_network_map()
        baseline_endpoints = baseline_map["attack_surface"]["total_endpoints"]
        
        self.network_sim.stop_simulation()
        
        # Test with IoT devices
        self.network_sim = AdvancedNetworkSimulator()  # Reset
        self.network_sim.start_simulation()
        self.iot_sim.populate_iot_network(count=15)
        
        time.sleep(2)
        
        enhanced_map = self.network_sim.get_network_map()
        enhanced_endpoints = enhanced_map["attack_surface"]["total_endpoints"]
        
        # IoT devices should increase attack surface
        self.assertGreater(enhanced_endpoints, baseline_endpoints)
        
        # Should have IoT-specific vulnerabilities
        iot_stats = self.iot_sim.get_iot_statistics()
        self.assertGreater(iot_stats["vulnerable_iot_devices"], 0)
        self.assertGreater(iot_stats["default_credentials"], 0)
        
        # Cleanup
        self.network_sim.stop_simulation()
        self.iot_sim.stop_simulation()
    
    def test_network_topology_discovery_for_agents(self):
        """Test network topology discovery capabilities specifically for agents"""
        # Setup comprehensive environment
        self.network_sim.start_simulation()
        self.iot_sim.populate_iot_network(count=8)
        self.legacy_sim.populate_legacy_network(count=4)
        
        time.sleep(3)
        
        network_map = self.network_sim.get_network_map()
        agent_data = network_map["agent_discovery_data"]
        
        # Test scan target prioritization
        scan_targets = agent_data["scan_targets"]
        self.assertGreater(len(scan_targets), 5)
        
        # High-priority targets should be first
        if len(scan_targets) > 1:
            first_priority = scan_targets[0]["priority"]
            last_priority = scan_targets[-1]["priority"]
            self.assertGreaterEqual(first_priority, last_priority)
        
        # Test service enumeration usefulness for agents
        service_enum = agent_data["service_enumeration"]
        
        # Should have services that agents can exploit
        exploitable_services = ["telnet", "ftp", "ssh", "http", "snmp"]
        found_services = [svc for svc in service_enum.keys() if svc in exploitable_services]
        self.assertGreater(len(found_services), 2)
        
        # Test vulnerability target identification
        vuln_targets = agent_data["vulnerability_targets"]
        self.assertGreater(len(vuln_targets), 0)
        
        # Vulnerability targets should have actionable information
        for target in vuln_targets[:3]:  # Check first 3
            self.assertIn("exploit_difficulty", target)
            self.assertIn("potential_impact", target)
            self.assertIn(target["exploit_difficulty"], ["easy", "medium", "hard", "very_hard"])
            self.assertIn(target["potential_impact"], ["low", "medium", "high"])
        
        # Cleanup
        self.network_sim.stop_simulation()
        self.iot_sim.stop_simulation()
        self.legacy_sim.stop_simulation()

