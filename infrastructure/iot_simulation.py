#!/usr/bin/env python3
"""
IoT Device Simulation Module

This module simulates various IoT devices and BYOD endpoints with realistic
behaviors, vulnerabilities, and network interactions.
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
import threading
import socket
import struct

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IoTProtocol(Enum):
    """IoT communication protocols"""
    MQTT = "mqtt"
    COAP = "coap"
    HTTP = "http"
    MODBUS = "modbus"
    BACNET = "bacnet"
    ZIGBEE = "zigbee"
    BLUETOOTH = "bluetooth"
    WIFI = "wifi"
    LORA = "lora"


class IoTDeviceCategory(Enum):
    """Categories of IoT devices"""
    SECURITY_CAMERA = "security_camera"
    ENVIRONMENTAL_SENSOR = "environmental_sensor"
    SMART_THERMOSTAT = "smart_thermostat"
    SMART_LIGHTING = "smart_lighting"
    ACCESS_CONTROL = "access_control"
    INDUSTRIAL_SENSOR = "industrial_sensor"
    SMART_PRINTER = "smart_printer"
    NETWORK_ATTACHED_STORAGE = "network_attached_storage"
    SMART_TV = "smart_tv"
    VOICE_ASSISTANT = "voice_assistant"


class BYODDeviceType(Enum):
    """Types of BYOD devices"""
    SMARTPHONE = "smartphone"
    TABLET = "tablet"
    LAPTOP = "laptop"
    SMARTWATCH = "smartwatch"
    PERSONAL_HOTSPOT = "personal_hotspot"


@dataclass
class IoTTelemetryData:
    """Represents telemetry data from IoT devices"""
    device_id: str
    timestamp: datetime
    sensor_type: str
    value: float
    unit: str
    quality: str = "good"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "timestamp": self.timestamp.isoformat(),
            "sensor_type": self.sensor_type,
            "value": self.value,
            "unit": self.unit,
            "quality": self.quality
        }


@dataclass
class IoTDevice:
    """Represents an IoT device"""
    device_id: str
    category: IoTDeviceCategory
    name: str
    manufacturer: str
    model: str
    firmware_version: str
    ip_address: str
    mac_address: str
    protocols: List[IoTProtocol]
    is_online: bool = True
    last_seen: datetime = field(default_factory=datetime.now)
    battery_level: Optional[float] = None
    signal_strength: float = -50.0  # dBm
    data_transmission_interval: int = 60  # seconds
    vulnerabilities: List[str] = field(default_factory=list)
    default_credentials: bool = False
    encryption_enabled: bool = True
    firmware_update_available: bool = False
    telemetry_data: List[IoTTelemetryData] = field(default_factory=list)
    
    def generate_telemetry(self) -> IoTTelemetryData:
        """Generate realistic telemetry data based on device category"""
        sensor_configs = {
            IoTDeviceCategory.SECURITY_CAMERA: {
                "motion_detected": (0, 1),
                "image_quality": (720, 4096),
                "storage_used": (0, 100)
            },
            IoTDeviceCategory.ENVIRONMENTAL_SENSOR: {
                "temperature": (15, 35),
                "humidity": (30, 80),
                "air_quality": (0, 500)
            },
            IoTDeviceCategory.SMART_THERMOSTAT: {
                "temperature": (18, 28),
                "target_temperature": (20, 25),
                "energy_usage": (0, 5000)
            },
            IoTDeviceCategory.INDUSTRIAL_SENSOR: {
                "pressure": (0, 100),
                "vibration": (0, 10),
                "temperature": (0, 200)
            }
        }
        
        config = sensor_configs.get(self.category, {"generic_value": (0, 100)})
        sensor_type = random.choice(list(config.keys()))
        min_val, max_val = config[sensor_type]
        
        # Add some noise and trends
        base_value = random.uniform(min_val, max_val)
        noise = random.uniform(-0.1, 0.1) * base_value
        value = max(min_val, min(max_val, base_value + noise))
        
        units = {
            "temperature": "°C",
            "humidity": "%",
            "pressure": "bar",
            "energy_usage": "W",
            "air_quality": "ppm",
            "vibration": "mm/s",
            "motion_detected": "bool",
            "image_quality": "pixels",
            "storage_used": "%",
            "target_temperature": "°C",
            "generic_value": "units"
        }
        
        telemetry = IoTTelemetryData(
            device_id=self.device_id,
            timestamp=datetime.now(),
            sensor_type=sensor_type,
            value=value,
            unit=units.get(sensor_type, "units"),
            quality="good" if random.random() > 0.05 else "poor"
        )
        
        self.telemetry_data.append(telemetry)
        
        # Keep only last 100 readings
        if len(self.telemetry_data) > 100:
            self.telemetry_data = self.telemetry_data[-100:]
        
        return telemetry
    
    def simulate_vulnerability_scan(self) -> List[str]:
        """Simulate vulnerability scanning results"""
        found_vulns = []
        
        # Check for default credentials
        if self.default_credentials:
            found_vulns.append("Default credentials detected")
        
        # Check for outdated firmware
        if self.firmware_update_available:
            found_vulns.append("Outdated firmware version")
        
        # Check for weak encryption
        if not self.encryption_enabled:
            found_vulns.append("Unencrypted communication")
        
        # Add device-specific vulnerabilities
        found_vulns.extend(self.vulnerabilities)
        
        return found_vulns


@dataclass
class BYODDevice:
    """Represents a BYOD (Bring Your Own Device) endpoint"""
    device_id: str
    device_type: BYODDeviceType
    owner_name: str
    device_name: str
    os_type: str
    os_version: str
    ip_address: str
    mac_address: str
    is_managed: bool = False
    has_mdm: bool = False
    compliance_status: str = "unknown"
    last_policy_check: Optional[datetime] = None
    installed_apps: List[str] = field(default_factory=list)
    security_patches_current: bool = False
    encryption_enabled: bool = True
    screen_lock_enabled: bool = True
    remote_wipe_capable: bool = False
    network_access_level: str = "guest"  # guest, limited, full
    
    def check_compliance(self) -> Dict[str, bool]:
        """Check device compliance with security policies"""
        compliance_checks = {
            "mdm_enrolled": self.has_mdm,
            "encryption_enabled": self.encryption_enabled,
            "screen_lock_enabled": self.screen_lock_enabled,
            "patches_current": self.security_patches_current,
            "remote_wipe_capable": self.remote_wipe_capable
        }
        
        # Update compliance status
        passed_checks = sum(compliance_checks.values())
        total_checks = len(compliance_checks)
        
        if passed_checks == total_checks:
            self.compliance_status = "compliant"
        elif passed_checks >= total_checks * 0.7:
            self.compliance_status = "partially_compliant"
        else:
            self.compliance_status = "non_compliant"
        
        self.last_policy_check = datetime.now()
        
        return compliance_checks
    
    def simulate_app_installation(self) -> str:
        """Simulate app installation"""
        apps = [
            "Social Media App", "Gaming App", "Productivity App",
            "Banking App", "Shopping App", "News App", "Weather App"
        ]
        
        new_app = random.choice(apps)
        if new_app not in self.installed_apps:
            self.installed_apps.append(new_app)
            logger.info(f"App installed on {self.device_name}: {new_app}")
        
        return new_app


class IoTNetworkSimulator:
    """Simulates IoT network traffic and protocols"""
    
    def __init__(self):
        self.mqtt_broker_port = 1883
        self.coap_port = 5683
        self.modbus_port = 502
        self.bacnet_port = 47808
        
    def simulate_mqtt_traffic(self, device: IoTDevice) -> Dict[str, Any]:
        """Simulate MQTT protocol traffic"""
        topics = [
            f"sensors/{device.device_id}/temperature",
            f"sensors/{device.device_id}/humidity",
            f"devices/{device.device_id}/status",
            f"alerts/{device.device_id}/motion"
        ]
        
        topic = random.choice(topics)
        telemetry = device.generate_telemetry()
        
        message = {
            "protocol": "MQTT",
            "topic": topic,
            "payload": telemetry.to_dict(),
            "qos": random.choice([0, 1, 2]),
            "retain": random.choice([True, False])
        }
        
        return message
    
    def simulate_coap_traffic(self, device: IoTDevice) -> Dict[str, Any]:
        """Simulate CoAP protocol traffic"""
        resources = [
            f"/sensors/{device.device_id}/temp",
            f"/sensors/{device.device_id}/status",
            f"/actuators/{device.device_id}/control"
        ]
        
        resource = random.choice(resources)
        telemetry = device.generate_telemetry()
        
        message = {
            "protocol": "CoAP",
            "method": random.choice(["GET", "POST", "PUT"]),
            "resource": resource,
            "payload": telemetry.to_dict(),
            "message_type": random.choice(["CON", "NON", "ACK", "RST"])
        }
        
        return message
    
    def simulate_modbus_traffic(self, device: IoTDevice) -> Dict[str, Any]:
        """Simulate Modbus protocol traffic"""
        functions = [
            "Read Coils", "Read Discrete Inputs", "Read Holding Registers",
            "Read Input Registers", "Write Single Coil", "Write Single Register"
        ]
        
        message = {
            "protocol": "Modbus",
            "function": random.choice(functions),
            "unit_id": random.randint(1, 247),
            "address": random.randint(0, 65535),
            "value": random.randint(0, 65535)
        }
        
        return message


class IoTDeviceFactory:
    """Factory for creating various IoT devices"""
    
    @staticmethod
    def create_security_camera(device_id: str, ip_address: str) -> IoTDevice:
        """Create a security camera device"""
        return IoTDevice(
            device_id=device_id,
            category=IoTDeviceCategory.SECURITY_CAMERA,
            name=f"Security Camera {device_id}",
            manufacturer=random.choice(["Hikvision", "Dahua", "Axis", "Bosch"]),
            model=f"CAM-{random.randint(1000, 9999)}",
            firmware_version=f"{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
            ip_address=ip_address,
            mac_address=IoTDeviceFactory._generate_mac(),
            protocols=[IoTProtocol.HTTP, IoTProtocol.MQTT],
            vulnerabilities=[
                "CVE-2021-36260 (Web Interface RCE)",
                "CVE-2020-25078 (Authentication Bypass)"
            ],
            default_credentials=random.choice([True, False]),
            encryption_enabled=random.choice([True, False]),
            firmware_update_available=random.choice([True, False])
        )
    
    @staticmethod
    def create_environmental_sensor(device_id: str, ip_address: str) -> IoTDevice:
        """Create an environmental sensor device"""
        return IoTDevice(
            device_id=device_id,
            category=IoTDeviceCategory.ENVIRONMENTAL_SENSOR,
            name=f"Environmental Sensor {device_id}",
            manufacturer=random.choice(["Honeywell", "Siemens", "ABB", "Schneider"]),
            model=f"ENV-{random.randint(1000, 9999)}",
            firmware_version=f"{random.randint(1, 2)}.{random.randint(0, 5)}.{random.randint(0, 9)}",
            ip_address=ip_address,
            mac_address=IoTDeviceFactory._generate_mac(),
            protocols=[IoTProtocol.MQTT, IoTProtocol.COAP],
            battery_level=random.uniform(20, 100),
            data_transmission_interval=random.randint(30, 300),
            vulnerabilities=[
                "CVE-2022-12345 (Weak Encryption)",
                "CVE-2021-54321 (Buffer Overflow)"
            ],
            default_credentials=random.choice([True, False]),
            encryption_enabled=random.choice([True, False])
        )
    
    @staticmethod
    def create_smart_thermostat(device_id: str, ip_address: str) -> IoTDevice:
        """Create a smart thermostat device"""
        return IoTDevice(
            device_id=device_id,
            category=IoTDeviceCategory.SMART_THERMOSTAT,
            name=f"Smart Thermostat {device_id}",
            manufacturer=random.choice(["Nest", "Honeywell", "Ecobee", "Johnson Controls"]),
            model=f"THERM-{random.randint(1000, 9999)}",
            firmware_version=f"{random.randint(2, 4)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
            ip_address=ip_address,
            mac_address=IoTDeviceFactory._generate_mac(),
            protocols=[IoTProtocol.HTTP, IoTProtocol.WIFI],
            vulnerabilities=[
                "CVE-2023-11111 (Privilege Escalation)",
                "CVE-2022-22222 (Information Disclosure)"
            ],
            default_credentials=False,
            encryption_enabled=True,
            firmware_update_available=random.choice([True, False])
        )
    
    @staticmethod
    def create_industrial_sensor(device_id: str, ip_address: str) -> IoTDevice:
        """Create an industrial sensor device"""
        return IoTDevice(
            device_id=device_id,
            category=IoTDeviceCategory.INDUSTRIAL_SENSOR,
            name=f"Industrial Sensor {device_id}",
            manufacturer=random.choice(["Allen-Bradley", "Siemens", "Schneider", "Mitsubishi"]),
            model=f"IND-{random.randint(1000, 9999)}",
            firmware_version=f"{random.randint(1, 2)}.{random.randint(0, 3)}.{random.randint(0, 9)}",
            ip_address=ip_address,
            mac_address=IoTDeviceFactory._generate_mac(),
            protocols=[IoTProtocol.MODBUS, IoTProtocol.BACNET],
            vulnerabilities=[
                "CVE-2020-12345 (Authentication Bypass)",
                "CVE-2019-54321 (Remote Code Execution)",
                "CVE-2021-99999 (Default Credentials)"
            ],
            default_credentials=True,  # Industrial devices often have default creds
            encryption_enabled=False,  # Legacy industrial protocols often unencrypted
            firmware_update_available=True
        )
    
    @staticmethod
    def create_smart_printer(device_id: str, ip_address: str) -> IoTDevice:
        """Create a smart printer device"""
        return IoTDevice(
            device_id=device_id,
            category=IoTDeviceCategory.SMART_PRINTER,
            name=f"Smart Printer {device_id}",
            manufacturer=random.choice(["HP", "Canon", "Epson", "Brother"]),
            model=f"PRINT-{random.randint(1000, 9999)}",
            firmware_version=f"{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
            ip_address=ip_address,
            mac_address=IoTDeviceFactory._generate_mac(),
            protocols=[IoTProtocol.HTTP, IoTProtocol.WIFI],
            vulnerabilities=[
                "CVE-2022-33333 (Directory Traversal)",
                "CVE-2021-44444 (Cross-Site Scripting)"
            ],
            default_credentials=random.choice([True, False]),
            encryption_enabled=random.choice([True, False])
        )
    
    @staticmethod
    def _generate_mac() -> str:
        """Generate a random MAC address"""
        return ":".join([f"{random.randint(0, 255):02x}" for _ in range(6)])


class BYODDeviceFactory:
    """Factory for creating BYOD devices"""
    
    @staticmethod
    def create_smartphone(device_id: str, ip_address: str, owner_name: str) -> BYODDevice:
        """Create a smartphone device"""
        os_choices = [
            ("Android", "12.0"), ("Android", "11.0"), ("Android", "10.0"),
            ("iOS", "16.1"), ("iOS", "15.7"), ("iOS", "14.8")
        ]
        os_type, os_version = random.choice(os_choices)
        
        return BYODDevice(
            device_id=device_id,
            device_type=BYODDeviceType.SMARTPHONE,
            owner_name=owner_name,
            device_name=f"{owner_name}'s Phone",
            os_type=os_type,
            os_version=os_version,
            ip_address=ip_address,
            mac_address=BYODDeviceFactory._generate_mac(),
            is_managed=random.choice([True, False]),
            has_mdm=random.choice([True, False]),
            security_patches_current=random.choice([True, False]),
            encryption_enabled=random.choice([True, False]),
            screen_lock_enabled=random.choice([True, False]),
            remote_wipe_capable=random.choice([True, False]),
            network_access_level=random.choice(["guest", "limited", "full"]),
            installed_apps=[
                "Email", "Browser", "Messaging", "Camera",
                random.choice(["Banking App", "Social Media", "Games"])
            ]
        )
    
    @staticmethod
    def create_tablet(device_id: str, ip_address: str, owner_name: str) -> BYODDevice:
        """Create a tablet device"""
        os_choices = [
            ("Android", "12.0"), ("iOS", "16.1"), ("Windows", "11")
        ]
        os_type, os_version = random.choice(os_choices)
        
        return BYODDevice(
            device_id=device_id,
            device_type=BYODDeviceType.TABLET,
            owner_name=owner_name,
            device_name=f"{owner_name}'s Tablet",
            os_type=os_type,
            os_version=os_version,
            ip_address=ip_address,
            mac_address=BYODDeviceFactory._generate_mac(),
            is_managed=random.choice([True, False]),
            has_mdm=random.choice([True, False]),
            security_patches_current=random.choice([True, False]),
            encryption_enabled=True,  # Tablets usually encrypted by default
            screen_lock_enabled=random.choice([True, False]),
            remote_wipe_capable=random.choice([True, False]),
            network_access_level=random.choice(["guest", "limited"]),
            installed_apps=[
                "Productivity Suite", "PDF Reader", "Video Player",
                random.choice(["Design App", "Note Taking", "Reading App"])
            ]
        )
    
    @staticmethod
    def create_laptop(device_id: str, ip_address: str, owner_name: str) -> BYODDevice:
        """Create a laptop device"""
        os_choices = [
            ("Windows", "11"), ("Windows", "10"), ("macOS", "13.0"), ("Ubuntu", "22.04")
        ]
        os_type, os_version = random.choice(os_choices)
        
        return BYODDevice(
            device_id=device_id,
            device_type=BYODDeviceType.LAPTOP,
            owner_name=owner_name,
            device_name=f"{owner_name}'s Laptop",
            os_type=os_type,
            os_version=os_version,
            ip_address=ip_address,
            mac_address=BYODDeviceFactory._generate_mac(),
            is_managed=random.choice([True, False]),
            has_mdm=random.choice([True, False]),
            security_patches_current=random.choice([True, False]),
            encryption_enabled=random.choice([True, False]),
            screen_lock_enabled=random.choice([True, False]),
            remote_wipe_capable=random.choice([True, False]),
            network_access_level=random.choice(["limited", "full"]),
            installed_apps=[
                "Office Suite", "Web Browser", "Development Tools",
                "VPN Client", random.choice(["Design Software", "Gaming", "Media Tools"])
            ]
        )
    
    @staticmethod
    def _generate_mac() -> str:
        """Generate a random MAC address"""
        return ":".join([f"{random.randint(0, 255):02x}" for _ in range(6)])


class IoTAndBYODSimulator:
    """Main simulator for IoT and BYOD devices"""
    
    def __init__(self):
        self.iot_devices: Dict[str, IoTDevice] = {}
        self.byod_devices: Dict[str, BYODDevice] = {}
        self.network_simulator = IoTNetworkSimulator()
        self.simulation_running = False
        self.simulation_thread = None
        
    def add_iot_device(self, device: IoTDevice):
        """Add an IoT device to the simulation"""
        self.iot_devices[device.device_id] = device
        logger.info(f"Added IoT device: {device.name} ({device.ip_address})")
    
    def add_byod_device(self, device: BYODDevice):
        """Add a BYOD device to the simulation"""
        self.byod_devices[device.device_id] = device
        logger.info(f"Added BYOD device: {device.device_name} ({device.ip_address})")
    
    def populate_iot_network(self, base_ip: str = "192.168.20.", count: int = 20):
        """Populate the IoT network with various devices"""
        device_creators = [
            IoTDeviceFactory.create_security_camera,
            IoTDeviceFactory.create_environmental_sensor,
            IoTDeviceFactory.create_smart_thermostat,
            IoTDeviceFactory.create_industrial_sensor,
            IoTDeviceFactory.create_smart_printer
        ]
        
        for i in range(count):
            device_id = f"iot_{i:03d}"
            ip_address = f"{base_ip}{i + 10}"
            creator = random.choice(device_creators)
            device = creator(device_id, ip_address)
            self.add_iot_device(device)
    
    def populate_byod_network(self, base_ip: str = "192.168.30.", count: int = 15):
        """Populate the BYOD network with various devices"""
        device_creators = [
            BYODDeviceFactory.create_smartphone,
            BYODDeviceFactory.create_tablet,
            BYODDeviceFactory.create_laptop
        ]
        
        names = [
            "John Smith", "Jane Doe", "Bob Johnson", "Alice Brown", "Charlie Wilson",
            "Diana Prince", "Eve Adams", "Frank Miller", "Grace Lee", "Henry Davis",
            "Ivy Chen", "Jack Taylor", "Kate Anderson", "Liam O'Connor", "Mia Garcia"
        ]
        
        for i in range(count):
            device_id = f"byod_{i:03d}"
            ip_address = f"{base_ip}{i + 10}"
            owner_name = names[i % len(names)]
            creator = random.choice(device_creators)
            device = creator(device_id, ip_address, owner_name)
            self.add_byod_device(device)
    
    def start_simulation(self):
        """Start the IoT and BYOD simulation"""
        if self.simulation_running:
            logger.warning("Simulation already running")
            return
        
        logger.info("Starting IoT and BYOD simulation")
        self.simulation_running = True
        
        # Start background simulation thread
        self.simulation_thread = threading.Thread(target=self._simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        logger.info("IoT and BYOD simulation started successfully")
    
    def stop_simulation(self):
        """Stop the simulation"""
        logger.info("Stopping IoT and BYOD simulation")
        self.simulation_running = False
        
        if self.simulation_thread:
            self.simulation_thread.join(timeout=5)
    
    def _simulation_loop(self):
        """Main simulation loop"""
        while self.simulation_running:
            try:
                # Simulate IoT device activities
                self._simulate_iot_activities()
                
                # Simulate BYOD device activities
                self._simulate_byod_activities()
                
                # Sleep for simulation interval
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
    
    def _simulate_iot_activities(self):
        """Simulate IoT device activities"""
        for device in self.iot_devices.values():
            if not device.is_online:
                continue
            
            # Generate telemetry data
            if random.random() < 0.3:  # 30% chance per cycle
                telemetry = device.generate_telemetry()
                
                # Simulate network traffic based on protocols
                if IoTProtocol.MQTT in device.protocols:
                    message = self.network_simulator.simulate_mqtt_traffic(device)
                    logger.debug(f"MQTT traffic from {device.name}: {message['topic']}")
                
                if IoTProtocol.COAP in device.protocols:
                    message = self.network_simulator.simulate_coap_traffic(device)
                    logger.debug(f"CoAP traffic from {device.name}: {message['resource']}")
                
                if IoTProtocol.MODBUS in device.protocols:
                    message = self.network_simulator.simulate_modbus_traffic(device)
                    logger.debug(f"Modbus traffic from {device.name}: {message['function']}")
            
            # Simulate device going offline occasionally
            if random.random() < 0.001:  # 0.1% chance
                device.is_online = False
                logger.info(f"IoT device {device.name} went offline")
            
            # Simulate battery drain for battery-powered devices
            if device.battery_level is not None:
                device.battery_level -= random.uniform(0.1, 0.5)
                if device.battery_level <= 0:
                    device.battery_level = 0
                    device.is_online = False
                    logger.warning(f"IoT device {device.name} battery depleted")
    
    def _simulate_byod_activities(self):
        """Simulate BYOD device activities"""
        for device in self.byod_devices.values():
            # Simulate app installations
            if random.random() < 0.01:  # 1% chance per cycle
                device.simulate_app_installation()
            
            # Simulate compliance checks
            if random.random() < 0.05:  # 5% chance per cycle
                compliance = device.check_compliance()
                if device.compliance_status == "non_compliant":
                    logger.warning(f"BYOD device {device.device_name} is non-compliant")
            
            # Simulate security patch updates
            if random.random() < 0.02:  # 2% chance per cycle
                device.security_patches_current = random.choice([True, False])
                if device.security_patches_current:
                    logger.info(f"BYOD device {device.device_name} updated security patches")
    
    def get_iot_statistics(self) -> Dict[str, Any]:
        """Get IoT device statistics"""
        total_iot = len(self.iot_devices)
        online_iot = sum(1 for d in self.iot_devices.values() if d.is_online)
        vulnerable_iot = sum(1 for d in self.iot_devices.values() if d.vulnerabilities)
        default_creds = sum(1 for d in self.iot_devices.values() if d.default_credentials)
        unencrypted = sum(1 for d in self.iot_devices.values() if not d.encryption_enabled)
        
        categories = {}
        for device in self.iot_devices.values():
            cat = device.category.value
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_iot_devices": total_iot,
            "online_iot_devices": online_iot,
            "vulnerable_iot_devices": vulnerable_iot,
            "default_credentials": default_creds,
            "unencrypted_devices": unencrypted,
            "device_categories": categories,
            "vulnerability_percentage": (vulnerable_iot / total_iot * 100) if total_iot > 0 else 0
        }
    
    def get_byod_statistics(self) -> Dict[str, Any]:
        """Get BYOD device statistics"""
        total_byod = len(self.byod_devices)
        managed_byod = sum(1 for d in self.byod_devices.values() if d.is_managed)
        compliant_byod = sum(1 for d in self.byod_devices.values() if d.compliance_status == "compliant")
        mdm_enrolled = sum(1 for d in self.byod_devices.values() if d.has_mdm)
        
        device_types = {}
        os_types = {}
        for device in self.byod_devices.values():
            dt = device.device_type.value
            device_types[dt] = device_types.get(dt, 0) + 1
            
            os = device.os_type
            os_types[os] = os_types.get(os, 0) + 1
        
        return {
            "total_byod_devices": total_byod,
            "managed_byod_devices": managed_byod,
            "compliant_byod_devices": compliant_byod,
            "mdm_enrolled_devices": mdm_enrolled,
            "device_types": device_types,
            "operating_systems": os_types,
            "compliance_percentage": (compliant_byod / total_byod * 100) if total_byod > 0 else 0
        }
    
    def export_device_inventory(self, filename: str):
        """Export device inventory to file"""
        inventory = {
            "iot_devices": {
                device_id: {
                    "name": device.name,
                    "category": device.category.value,
                    "manufacturer": device.manufacturer,
                    "model": device.model,
                    "firmware_version": device.firmware_version,
                    "ip_address": device.ip_address,
                    "is_online": device.is_online,
                    "vulnerabilities": device.vulnerabilities,
                    "default_credentials": device.default_credentials,
                    "encryption_enabled": device.encryption_enabled
                }
                for device_id, device in self.iot_devices.items()
            },
            "byod_devices": {
                device_id: {
                    "device_name": device.device_name,
                    "device_type": device.device_type.value,
                    "owner_name": device.owner_name,
                    "os_type": device.os_type,
                    "os_version": device.os_version,
                    "ip_address": device.ip_address,
                    "is_managed": device.is_managed,
                    "compliance_status": device.compliance_status,
                    "has_mdm": device.has_mdm,
                    "network_access_level": device.network_access_level
                }
                for device_id, device in self.byod_devices.items()
            },
            "statistics": {
                "iot": self.get_iot_statistics(),
                "byod": self.get_byod_statistics()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(inventory, f, indent=2, default=str)
        
        logger.info(f"Device inventory exported to {filename}")


def main():
    """Main function for testing the IoT and BYOD simulation"""
    simulator = IoTAndBYODSimulator()
    
    try:
        # Populate networks
        simulator.populate_iot_network(count=15)
        simulator.populate_byod_network(count=10)
        
        # Start simulation
        simulator.start_simulation()
        
        # Let it run for a bit
        time.sleep(30)
        
        # Get statistics
        iot_stats = simulator.get_iot_statistics()
        byod_stats = simulator.get_byod_statistics()
        
        print(f"IoT Statistics: {iot_stats}")
        print(f"BYOD Statistics: {byod_stats}")
        
        # Export inventory
        simulator.export_device_inventory("iot_byod_inventory.json")
        
    finally:
        simulator.stop_simulation()


if __name__ == "__main__":
    main()