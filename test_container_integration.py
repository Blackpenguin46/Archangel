#!/usr/bin/env python3
"""
Test script to demonstrate container integration with AI agents
Shows red team AI controlling Kali Linux and blue team AI monitoring Ubuntu SOC
"""

import asyncio
import json
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Any

class ContainerIntegrationTest:
    def __init__(self):
        self.red_container = "archangel-red-team"
        self.blue_container = "archangel-blue-team"
        self.target_container = "target-enterprise"
        
    async def run_integration_test(self):
        """Run a complete integration test of the containerized AI vs AI system"""
        print(f"🚀 Starting Archangel Container Integration Test")
        print(f"=" * 60)
        
        # Check container status
        await self._check_container_status()
        
        # Read agent status
        await self._read_agent_status()
        
        # Test red team autonomous scanning
        await self._test_red_team_scanning()
        
        # Test blue team monitoring
        await self._test_blue_team_monitoring()
        
        # Demonstrate AI vs AI interaction
        await self._demonstrate_ai_vs_ai()
        
        print(f"\\n✅ Container Integration Test Complete!")
        print(f"=" * 60)
    
    async def _check_container_status(self):
        """Check the status of all containers"""
        print(f"\\n📊 Container Status Check:")
        
        try:
            # Check container status
            result = subprocess.run([
                "docker", "ps", "--filter", "name=archangel-", 
                "--format", "table {{.Names}}\\t{{.Status}}\\t{{.Ports}}"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"\\n{result.stdout}")
            else:
                print(f"❌ Failed to check container status")
                
        except Exception as e:
            print(f"❌ Container status check error: {e}")
    
    async def _read_agent_status(self):
        """Read status from both AI agents"""
        print(f"\\n🤖 AI Agent Status:")
        
        try:
            # Read red team status
            with open("logs/red_team_status.log", "r") as f:
                red_status = json.load(f)
                print(f"\\n🔴 Red Team Agent:")
                print(f"   Status: {red_status.get('status', 'unknown')}")
                print(f"   Tools: {', '.join(red_status.get('tools_available', []))}")
                print(f"   Discovered Hosts: {len(red_status.get('discovered_hosts', []))}")
        except Exception as e:
            print(f"⚠️ Could not read red team status: {e}")
        
        try:
            # Read blue team status
            with open("logs/blue_team_status.log", "r") as f:
                blue_status = json.load(f)
                print(f"\\n🔵 Blue Team Agent:")
                print(f"   Status: {blue_status.get('status', 'unknown')}")
                print(f"   Tools: {', '.join(blue_status.get('tools_available', []))}")
                print(f"   Threats Detected: {blue_status.get('detected_threats', 0)}")
                print(f"   Blocked IPs: {blue_status.get('blocked_ips', 0)}")
                print(f"   Active Connections: {blue_status.get('active_connections', 0)}")
        except Exception as e:
            print(f"⚠️ Could not read blue team status: {e}")
    
    async def _test_red_team_scanning(self):
        """Test red team autonomous scanning capabilities"""
        print(f"\\n🔍 Testing Red Team Autonomous Scanning:")
        
        try:
            # Execute nmap scan inside red team container
            print(f"   Executing network scan from Kali container...")
            
            result = subprocess.run([
                "docker", "exec", self.red_container,
                "nmap", "-sn", "172.18.0.0/24"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                lines = result.stdout.split('\\n')
                discovered_hosts = []
                for line in lines:
                    if "Nmap scan report for" in line:
                        host = line.split()[-1]
                        discovered_hosts.append(host)
                
                print(f"   ✅ Scan successful!")
                print(f"   📡 Discovered {len(discovered_hosts)} hosts:")
                for host in discovered_hosts:
                    print(f"      • {host}")
            else:
                print(f"   ❌ Scan failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"   ⏱️ Scan timed out (normal for large networks)")
        except Exception as e:
            print(f"   ❌ Scan error: {e}")
    
    async def _test_blue_team_monitoring(self):
        """Test blue team monitoring capabilities"""
        print(f"\\n🛡️ Testing Blue Team Monitoring:")
        
        try:
            # Check network connections from blue team container
            print(f"   Checking network connections from Ubuntu SOC...")
            
            result = subprocess.run([
                "docker", "exec", self.blue_container,
                "ss", "-tuln"
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                lines = result.stdout.split('\\n')
                connections = []
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        connections.append(line.strip())
                
                print(f"   ✅ Monitoring successful!")
                print(f"   🌐 Active network connections: {len(connections)}")
                if connections:
                    print(f"   📊 Sample connections:")
                    for conn in connections[:3]:  # Show first 3
                        print(f"      • {conn}")
            else:
                print(f"   ❌ Monitoring failed: {result.stderr}")
                
        except Exception as e:
            print(f"   ❌ Monitoring error: {e}")
    
    async def _demonstrate_ai_vs_ai(self):
        """Demonstrate AI vs AI interaction"""
        print(f"\\n⚔️ AI vs AI Demonstration:")
        
        try:
            # Red team: Port scan the blue team
            print(f"   🔴 Red Team: Scanning Blue Team container...")
            
            # Get blue team IP
            blue_ip_result = subprocess.run([
                "docker", "inspect", self.blue_container,
                "--format", "{{.NetworkSettings.Networks.archangel-combat.IPAddress}}"
            ], capture_output=True, text=True)
            
            if blue_ip_result.returncode == 0:
                blue_ip = blue_ip_result.stdout.strip()
                print(f"   🎯 Blue Team IP: {blue_ip}")
                
                # Red team scans blue team
                scan_result = subprocess.run([
                    "docker", "exec", self.red_container,
                    "nmap", "-F", blue_ip
                ], capture_output=True, text=True, timeout=30)
                
                if scan_result.returncode == 0:
                    print(f"   ✅ Red Team scan completed")
                    # Extract open ports
                    lines = scan_result.stdout.split('\\n')
                    open_ports = []
                    for line in lines:
                        if "/tcp" in line and "open" in line:
                            port = line.split('/')[0].strip()
                            open_ports.append(port)
                    
                    if open_ports:
                        print(f"   🔓 Open ports discovered: {', '.join(open_ports)}")
                    else:
                        print(f"   🔒 No open ports found")
                else:
                    print(f"   ⚠️ Scan completed with warnings")
                
                # Blue team: Check for scanning activity
                print(f"\\n   🔵 Blue Team: Detecting scan activity...")
                
                # Check processes for nmap
                process_result = subprocess.run([
                    "docker", "exec", self.blue_container,
                    "ps", "aux"
                ], capture_output=True, text=True, timeout=10)
                
                if process_result.returncode == 0:
                    print(f"   ✅ Blue Team monitoring active")
                    print(f"   👁️ Processes monitored: {len(process_result.stdout.split(chr(10)))}")
                
            else:
                print(f"   ❌ Could not get blue team IP")
                
        except subprocess.TimeoutExpired:
            print(f"   ⏱️ AI interaction timed out")
        except Exception as e:
            print(f"   ❌ AI vs AI error: {e}")
    
    async def _show_container_logs(self):
        """Show recent logs from container agents"""
        print(f"\\n📋 Container Agent Logs:")
        
        try:
            # Red team logs
            red_logs = subprocess.run([
                "docker", "logs", "--tail", "10", self.red_container
            ], capture_output=True, text=True)
            
            if red_logs.stdout:
                print(f"\\n🔴 Red Team Recent Activity:")
                for line in red_logs.stdout.split('\\n')[-5:]:
                    if line.strip():
                        print(f"   {line}")
        except Exception as e:
            print(f"⚠️ Could not get red team logs: {e}")
        
        try:
            # Blue team logs
            blue_logs = subprocess.run([
                "docker", "logs", "--tail", "10", self.blue_container
            ], capture_output=True, text=True)
            
            if blue_logs.stdout:
                print(f"\\n🔵 Blue Team Recent Activity:")
                for line in blue_logs.stdout.split('\\n')[-5:]:
                    if line.strip():
                        print(f"   {line}")
        except Exception as e:
            print(f"⚠️ Could not get blue team logs: {e}")

async def main():
    """Main entry point for container integration test"""
    tester = ContainerIntegrationTest()
    
    try:
        await tester.run_integration_test()
    except KeyboardInterrupt:
        print(f"\\n🛑 Integration test stopped by user")
    except Exception as e:
        print(f"❌ Integration test error: {e}")

if __name__ == "__main__":
    asyncio.run(main())