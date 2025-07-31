#!/usr/bin/env python3
"""
Apple Container CLI Setup and Kali Linux Integration
Archangel's containerized security testing environment using Apple's native virtualization
"""

import asyncio
import json
import logging
import subprocess
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime

class AppleContainerManager:
    """
    Manager for Apple Container CLI integration
    Provides safe containerized environments for security testing
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.container_cli_available = False
        self.active_containers: Dict[str, Dict[str, Any]] = {}
        self.container_configs_dir = Path("data/container_configs")
        self.container_configs_dir.mkdir(exist_ok=True)
        
    async def initialize(self) -> bool:
        """Initialize Apple Container CLI integration"""
        self.logger.info("🍎 Initializing Apple Container CLI integration...")
        
        try:
            # Check if container CLI is installed
            result = subprocess.run(["which", "container"], capture_output=True, text=True)
            if result.returncode == 0:
                self.container_cli_available = True
                self.logger.info("✅ Apple Container CLI found")
                
                # Check system service status
                await self._check_container_service()
                
                return True
            else:
                self.logger.warning("⚠️ Apple Container CLI not found - please install from Apple's GitHub")
                await self._provide_installation_instructions()
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Apple Container: {e}")
            return False
    
    async def _check_container_service(self):
        """Check if container system service is running"""
        try:
            result = subprocess.run(
                ["container", "system", "status"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                self.logger.info("✅ Container system service is running")
            else:
                self.logger.info("🔄 Starting container system service...")
                await self._start_container_service()
                
        except subprocess.TimeoutExpired:
            self.logger.warning("⚠️ Container system check timed out")
        except Exception as e:
            self.logger.warning(f"Container system check failed: {e}")
    
    async def _start_container_service(self):
        """Start the container system service"""
        try:
            result = subprocess.run(
                ["container", "system", "start"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self.logger.info("✅ Container system service started")
            else:
                self.logger.warning(f"Failed to start container service: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Failed to start container service: {e}")
    
    async def _provide_installation_instructions(self):
        """Provide installation instructions for Apple Container CLI"""
        instructions = """
🍎 Apple Container CLI Installation Instructions:

1. Download from GitHub:
   https://github.com/apple/container/releases

2. Install the .pkg file:
   - Double-click the downloaded .pkg file
   - Follow the installation prompts
   - Enter admin password when required

3. Start the system service:
   container system start

4. Verify installation:
   container --version

Requirements:
- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 15 (Sequoia) or later
- macOS 26 (Tahoe) recommended for full features

Note: On macOS 15, networking features are limited.
"""
        self.logger.info(instructions)
    
    async def create_kali_container(self, 
                                  container_name: str = None,
                                  security_tools: List[str] = None) -> Dict[str, Any]:
        """Create a Kali Linux container for security testing"""
        if not self.container_cli_available:
            return {"error": "Apple Container CLI not available"}
        
        container_name = container_name or f"archangel-kali-{uuid.uuid4().hex[:8]}"
        security_tools = security_tools or ["nmap", "metasploit", "burpsuite", "wireshark"]
        
        self.logger.info(f"🐧 Creating Kali Linux container: {container_name}")
        
        try:
            # Create container configuration
            config = await self._create_kali_config(container_name, security_tools)
            
            # Create the container
            result = await self._create_container_from_config(config)
            
            if result["success"]:
                self.active_containers[container_name] = {
                    "name": container_name,
                    "type": "kali_linux",
                    "config": config,
                    "created_at": datetime.now().isoformat(),
                    "status": "running",
                    "security_tools": security_tools
                }
                
                self.logger.info(f"✅ Kali container created: {container_name}")
                return {
                    "success": True,
                    "container_name": container_name,
                    "config": config,
                    "security_tools": security_tools
                }
            else:
                return {"error": f"Failed to create container: {result.get('error')}"}
                
        except Exception as e:
            self.logger.error(f"Failed to create Kali container: {e}")
            return {"error": str(e)}
    
    async def _create_kali_config(self, container_name: str, security_tools: List[str]) -> Dict[str, Any]:
        """Create Kali Linux container configuration"""
        
        # Note: This is a conceptual implementation
        # The actual Apple Container CLI uses different syntax
        config = {
            "name": container_name,
            "image": {
                "type": "debian_based",
                "base": "debian:bookworm-slim",
                "architecture": "arm64"
            },
            "resources": {
                "memory": "4GB",
                "cpu_cores": 2,
                "disk_size": "20GB"
            },
            "network": {
                "mode": "isolated",  # Safe for security testing
                "allow_host_access": False,
                "ports": []
            },
            "security": {
                "sandbox": True,
                "readonly_root": False,
                "capabilities": ["NET_ADMIN", "SYS_ADMIN"],  # Needed for security tools
                "user": "kali"
            },
            "setup_commands": [
                "apt-get update",
                "apt-get install -y curl wget git",
                "useradd -m -s /bin/bash kali",
                "usermod -aG sudo kali",
                *[f"apt-get install -y {tool}" for tool in security_tools if tool in ["nmap", "wireshark", "tcpdump"]],
                "echo 'Container ready for security testing' > /tmp/container_ready"
            ],
            "environment": {
                "DEBIAN_FRONTEND": "noninteractive",
                "KALI_USER": "kali",
                "SECURITY_TESTING": "true"
            }
        }
        
        return config
    
    async def _create_container_from_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create container from configuration"""
        try:
            # Save config to temporary file
            config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(config, config_file, indent=2)
            config_file.close()
            
            # For Apple Container CLI, we'd use commands like:
            # container create --config config.json
            # But since the actual syntax may differ, we'll simulate
            
            container_name = config["name"]
            
            # Simulate container creation
            self.logger.info(f"📦 Creating container with config: {config_file.name}")
            
            # In real implementation, this would be:
            # result = subprocess.run([
            #     "container", "create", 
            #     "--name", container_name,
            #     "--config", config_file.name
            # ], capture_output=True, text=True)
            
            # For now, simulate successful creation
            success = True
            
            # Cleanup temp file
            os.unlink(config_file.name)
            
            if success:
                return {
                    "success": True,
                    "container_name": container_name,
                    "message": "Container created successfully"
                }
            else:
                return {
                    "success": False,
                    "error": "Container creation failed"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def create_monitoring_container(self, container_name: str = None) -> Dict[str, Any]:
        """Create a monitoring container for blue team operations"""
        if not self.container_cli_available:
            return {"error": "Apple Container CLI not available"}
        
        container_name = container_name or f"archangel-monitor-{uuid.uuid4().hex[:8]}"
        
        self.logger.info(f"🛡️ Creating monitoring container: {container_name}")
        
        try:
            config = {
                "name": container_name,
                "image": {
                    "type": "ubuntu_based",
                    "base": "ubuntu:22.04",
                    "architecture": "arm64"
                },
                "resources": {
                    "memory": "2GB",
                    "cpu_cores": 1,
                    "disk_size": "10GB"
                },
                "network": {
                    "mode": "bridge",  # Can monitor host network
                    "allow_host_access": True,
                    "ports": ["8080:80", "9090:9090"]
                },
                "security": {
                    "sandbox": True,
                    "readonly_root": True,
                    "capabilities": ["NET_ADMIN"],
                    "user": "monitor"
                },
                "monitoring_tools": [
                    "htop", "netstat", "tcpdump", "iftop",
                    "prometheus", "grafana", "elk-stack"
                ],
                "setup_commands": [
                    "apt-get update",
                    "apt-get install -y htop netstat-nat tcpdump iftop",
                    "useradd -m -s /bin/bash monitor",
                    "echo 'Monitoring container ready' > /tmp/monitor_ready"
                ]
            }
            
            result = await self._create_container_from_config(config)
            
            if result["success"]:
                self.active_containers[container_name] = {
                    "name": container_name,
                    "type": "monitoring",
                    "config": config,
                    "created_at": datetime.now().isoformat(),
                    "status": "running"
                }
                
                return {
                    "success": True,
                    "container_name": container_name,
                    "config": config
                }
            else:
                return {"error": f"Failed to create monitoring container: {result.get('error')}"}
                
        except Exception as e:
            self.logger.error(f"Failed to create monitoring container: {e}")
            return {"error": str(e)}
    
    async def execute_in_container(self, 
                                 container_name: str, 
                                 command: str,
                                 timeout: int = 30) -> Dict[str, Any]:
        """Execute command in container safely"""
        if container_name not in self.active_containers:
            return {"error": f"Container {container_name} not found"}
        
        self.logger.info(f"🔄 Executing in {container_name}: {command}")
        
        try:
            # Safety check - ensure command is safe
            if not await self._is_safe_command(command):
                return {"error": f"Command blocked for safety: {command}"}
            
            # In real implementation:
            # result = subprocess.run([
            #     "container", "exec", container_name, "--", 
            #     "sh", "-c", command
            # ], capture_output=True, text=True, timeout=timeout)
            
            # Simulate command execution
            execution_result = {
                "success": True,
                "stdout": f"Simulated execution of: {command}",
                "stderr": "",
                "exit_code": 0,
                "execution_time": 1.5
            }
            
            return execution_result
            
        except subprocess.TimeoutExpired:
            return {"error": f"Command timed out after {timeout} seconds"}
        except Exception as e:
            return {"error": str(e)}
    
    async def _is_safe_command(self, command: str) -> bool:
        """Check if command is safe to execute"""
        # Block potentially dangerous commands
        dangerous_patterns = [
            "rm -rf", "mkfs", "dd if=",
            ":(){ :|:& };:", "chmod 777 /",
            "curl | sh", "wget | sh",
            "> /dev/sda", "cat /dev/urandom"
        ]
        
        command_lower = command.lower()
        
        for pattern in dangerous_patterns:
            if pattern in command_lower:
                self.logger.warning(f"🚨 Blocked dangerous command pattern: {pattern}")
                return False
        
        return True
    
    async def run_security_scan(self, 
                              container_name: str,
                              target: str,
                              scan_type: str = "basic") -> Dict[str, Any]:
        """Run security scan in container"""
        if container_name not in self.active_containers:
            return {"error": f"Container {container_name} not found"}
        
        container_info = self.active_containers[container_name]
        if container_info["type"] != "kali_linux":
            return {"error": "Security scans require Kali Linux container"}
        
        self.logger.info(f"🔍 Running {scan_type} scan on {target} from {container_name}")
        
        # Define scan commands based on type
        scan_commands = {
            "basic": f"nmap -sS -O {target}",
            "port_scan": f"nmap -p- {target}",
            "service_scan": f"nmap -sV -sC {target}",
            "vulnerability_scan": f"nmap --script vuln {target}"
        }
        
        command = scan_commands.get(scan_type, scan_commands["basic"])
        
        # Safety check - ensure target is safe
        if not await self._is_safe_target(target):
            return {"error": f"Target blocked for safety: {target}"}
        
        # Execute scan
        result = await self.execute_in_container(container_name, command, timeout=300)
        
        if result.get("success"):
            # Parse scan results
            scan_results = await self._parse_scan_results(result["stdout"], scan_type)
            
            return {
                "success": True,
                "scan_type": scan_type,
                "target": target,
                "container": container_name,
                "raw_output": result["stdout"],
                "parsed_results": scan_results,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {"error": f"Scan failed: {result.get('error')}"}
    
    async def _is_safe_target(self, target: str) -> bool:
        """Check if target is safe for scanning"""
        # Only allow scanning of test/local targets
        safe_targets = [
            "localhost", "127.0.0.1", "::1",
            "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
            "testbed", "sandbox", "demo"
        ]
        
        # Check if target matches safe patterns
        for safe_target in safe_targets:
            if safe_target in target.lower():
                return True
        
        # Block external/production targets
        self.logger.warning(f"🚨 Blocked external target: {target}")
        return False
    
    async def _parse_scan_results(self, output: str, scan_type: str) -> Dict[str, Any]:
        """Parse scan results into structured format"""
        # Simple parsing - could be enhanced with proper nmap XML parsing
        results = {
            "scan_type": scan_type,
            "ports_found": [],
            "services_identified": [],
            "vulnerabilities": [],
            "summary": "Scan completed"
        }
        
        # Mock parsing for demo
        if "22/tcp" in output:
            results["ports_found"].append({"port": 22, "service": "ssh", "state": "open"})
        if "80/tcp" in output:
            results["ports_found"].append({"port": 80, "service": "http", "state": "open"})
        
        return results
    
    async def list_containers(self) -> Dict[str, Any]:
        """List all active containers"""
        return {
            "active_containers": len(self.active_containers),
            "containers": list(self.active_containers.values())
        }
    
    async def stop_container(self, container_name: str) -> Dict[str, Any]:
        """Stop and remove container"""
        if container_name not in self.active_containers:
            return {"error": f"Container {container_name} not found"}
        
        self.logger.info(f"🛑 Stopping container: {container_name}")
        
        try:
            # In real implementation:
            # subprocess.run(["container", "stop", container_name])
            # subprocess.run(["container", "rm", container_name])
            
            # Remove from active containers
            del self.active_containers[container_name]
            
            return {
                "success": True,
                "message": f"Container {container_name} stopped and removed"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def get_container_logs(self, container_name: str, lines: int = 50) -> Dict[str, Any]:
        """Get container logs"""
        if container_name not in self.active_containers:
            return {"error": f"Container {container_name} not found"}
        
        try:
            # In real implementation:
            # result = subprocess.run([
            #     "container", "logs", container_name, "--tail", str(lines)
            # ], capture_output=True, text=True)
            
            # Mock logs
            mock_logs = [
                f"[{datetime.now().isoformat()}] Container {container_name} started",
                f"[{datetime.now().isoformat()}] Security tools initialized",
                f"[{datetime.now().isoformat()}] Ready for operations"
            ]
            
            return {
                "success": True,
                "container": container_name,
                "logs": mock_logs,
                "lines_requested": lines
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def cleanup_all_containers(self):
        """Cleanup all containers"""
        self.logger.info("🧹 Cleaning up all containers...")
        
        for container_name in list(self.active_containers.keys()):
            await self.stop_container(container_name)
        
        self.logger.info("✅ All containers cleaned up")


# Integration with Archangel's autonomous agents

class ContainerizedSecurityAgent:
    """
    Security agent that operates within Apple Container environments
    """
    
    def __init__(self, agent_id: str, container_manager: AppleContainerManager):
        self.agent_id = agent_id
        self.container_manager = container_manager
        self.logger = logging.getLogger(f"containerized_agent_{agent_id}")
        self.assigned_container: Optional[str] = None
    
    async def initialize_container_environment(self, agent_type: str = "kali") -> bool:
        """Initialize containerized environment for agent"""
        self.logger.info(f"🏗️ Initializing container environment for {agent_type} agent")
        
        try:
            if agent_type == "kali":
                result = await self.container_manager.create_kali_container(
                    container_name=f"agent-{self.agent_id}-kali",
                    security_tools=["nmap", "metasploit", "burpsuite"]
                )
            else:
                result = await self.container_manager.create_monitoring_container(
                    container_name=f"agent-{self.agent_id}-monitor"
                )
            
            if result.get("success"):
                self.assigned_container = result["container_name"]
                self.logger.info(f"✅ Container ready: {self.assigned_container}")
                return True
            else:
                self.logger.error(f"Failed to create container: {result.get('error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Container initialization failed: {e}")
            return False
    
    async def execute_security_operation(self, operation: str, target: str = None) -> Dict[str, Any]:
        """Execute security operation in container"""
        if not self.assigned_container:
            return {"error": "No container assigned"}
        
        self.logger.info(f"🎯 Executing operation: {operation}")
        
        if operation == "port_scan" and target:
            return await self.container_manager.run_security_scan(
                self.assigned_container, target, "port_scan"
            )
        elif operation == "vulnerability_scan" and target:
            return await self.container_manager.run_security_scan(
                self.assigned_container, target, "vulnerability_scan"
            )
        elif operation == "monitor_network":
            return await self.container_manager.execute_in_container(
                self.assigned_container, "netstat -tuln"
            )
        else:
            return {"error": f"Unknown operation: {operation}"}
    
    async def cleanup(self):
        """Cleanup agent container"""
        if self.assigned_container:
            await self.container_manager.stop_container(self.assigned_container)
            self.assigned_container = None


# Demo and testing functions

async def demo_apple_container_integration():
    """Demonstrate Apple Container integration"""
    print("🍎 Apple Container Integration Demo")
    print("=" * 50)
    
    # Initialize container manager
    container_manager = AppleContainerManager()
    
    if not await container_manager.initialize():
        print("❌ Apple Container CLI not available - please install first")
        return
    
    try:
        # Create Kali container for red team
        print("\n🐧 Creating Kali Linux container...")
        kali_result = await container_manager.create_kali_container(
            "demo-kali",
            ["nmap", "metasploit"]
        )
        
        if kali_result.get("success"):
            print(f"✅ Kali container created: {kali_result['container_name']}")
        else:
            print(f"❌ Kali container failed: {kali_result.get('error')}")
        
        # Create monitoring container for blue team
        print("\n🛡️ Creating monitoring container...")
        monitor_result = await container_manager.create_monitoring_container("demo-monitor")
        
        if monitor_result.get("success"):
            print(f"✅ Monitor container created: {monitor_result['container_name']}")
        else:
            print(f"❌ Monitor container failed: {monitor_result.get('error')}")
        
        # List containers
        print("\n📦 Active containers:")
        containers = await container_manager.list_containers()
        for container in containers["containers"]:
            print(f"  • {container['name']} ({container['type']}) - {container['status']}")
        
        # Run test scan (if Kali container available)
        if "demo-kali" in container_manager.active_containers:
            print("\n🔍 Running test security scan...")
            scan_result = await container_manager.run_security_scan(
                "demo-kali", "127.0.0.1", "basic"
            )
            
            if scan_result.get("success"):
                print(f"✅ Scan completed: {len(scan_result['parsed_results']['ports_found'])} ports found")
            else:
                print(f"❌ Scan failed: {scan_result.get('error')}")
        
        # Demonstrate containerized agent
        print("\n🤖 Testing containerized security agent...")
        agent = ContainerizedSecurityAgent("demo_agent", container_manager)
        
        if await agent.initialize_container_environment("kali"):
            print("✅ Containerized agent ready")
            
            # Test operation
            op_result = await agent.execute_security_operation("port_scan", "localhost")
            if op_result.get("success"):
                print("✅ Agent operation successful")
            else:
                print(f"❌ Agent operation failed: {op_result.get('error')}")
            
            await agent.cleanup()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up containers...")
        await container_manager.cleanup_all_containers()
        print("✅ Demo completed")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demo
    asyncio.run(demo_apple_container_integration())