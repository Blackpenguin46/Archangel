#!/usr/bin/env python3
"""
Autonomous Kali Linux Red Team Agent
AI agent that intelligently controls real Kali Linux penetration testing tools
"""

import asyncio
import json
import logging
import subprocess
import os
import sys
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import xml.etree.ElementTree as ET

# Add parent directory to path for imports
sys.path.append('/app')

try:
    from core.enhanced_hf_model_manager import get_model_manager
    from core.ai_enhanced_agents import AdvancedReasoningEngine
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ AI models not available, running in basic mode")

@dataclass
class KaliToolResult:
    """Result from executing a Kali Linux tool"""
    tool_name: str
    command: str
    success: bool
    output: str
    error: str
    execution_time: float
    findings: List[str]
    next_actions: List[str]

@dataclass
class PenetrationTarget:
    """Target information for penetration testing"""
    ip_address: str
    hostname: Optional[str] = None
    open_ports: List[int] = None
    services: Dict[int, str] = None
    vulnerabilities: List[str] = None
    os_info: Optional[str] = None
    web_technologies: List[str] = None

class KaliToolController:
    """Controller for Kali Linux penetration testing tools"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results_dir = "/data/results"
        self.logs_dir = "/data/logs"
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Available Kali tools organized by category
        self.kali_tools = {
            "reconnaissance": {
                "nmap": "/usr/bin/nmap",
                "masscan": "/usr/bin/masscan", 
                "amass": "/usr/bin/amass",
                "subfinder": "/usr/bin/subfinder",
                "theharvester": "/usr/bin/theharvester",
                "recon-ng": "/usr/bin/recon-ng",
                "whatweb": "/usr/bin/whatweb",
                "dirb": "/usr/bin/dirb",
                "gobuster": "/usr/bin/gobuster",
                "ffuf": "/usr/bin/ffuf"
            },
            "vulnerability_analysis": {
                "nikto": "/usr/bin/nikto",
                "sqlmap": "/usr/bin/sqlmap",
                "wpscan": "/usr/bin/wpscan",
                "nuclei": "/usr/bin/nuclei",
                "openvas": "/usr/bin/openvas"
            },
            "exploitation": {
                "msfconsole": "/usr/bin/msfconsole",
                "msfvenom": "/usr/bin/msfvenom",
                "searchsploit": "/usr/bin/searchsploit",
                "exploit-db": "/usr/bin/searchsploit"
            },
            "password_attacks": {
                "hydra": "/usr/bin/hydra",
                "john": "/usr/bin/john",
                "hashcat": "/usr/bin/hashcat",
                "crunch": "/usr/bin/crunch"
            },
            "post_exploitation": {
                "crackmapexec": "/usr/bin/crackmapexec",
                "impacket-psexec": "/usr/bin/impacket-psexec",
                "bloodhound": "/usr/bin/bloodhound",
                "mimikatz": "/usr/bin/mimikatz"
            },
            "social_engineering": {
                "setoolkit": "/usr/bin/setoolkit",
                "gophish": "/usr/bin/gophish",
                "beef-xss": "/usr/bin/beef-xss"
            }
        }
    
    async def execute_tool(self, tool_name: str, args: List[str], timeout: int = 300) -> KaliToolResult:
        """Execute a Kali Linux tool with AI-guided parameters"""
        start_time = datetime.now()
        
        # Find tool path
        tool_path = None
        tool_category = None
        for category, tools in self.kali_tools.items():
            if tool_name in tools:
                tool_path = tools[tool_name]
                tool_category = category
                break
        
        if not tool_path:
            return KaliToolResult(
                tool_name=tool_name,
                command="",
                success=False,
                output="",
                error=f"Tool {tool_name} not found",
                execution_time=0.0,
                findings=[],
                next_actions=[]
            )
        
        # Build command
        command = [tool_path] + args
        command_str = " ".join(command)
        
        self.logger.info(f"🔴 Executing: {command_str}")
        
        try:
            # Execute the tool
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.results_dir
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise Exception(f"Tool execution timed out after {timeout} seconds")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Decode output
            output = stdout.decode('utf-8', errors='ignore') if stdout else ""
            error = stderr.decode('utf-8', errors='ignore') if stderr else ""
            
            # Parse findings based on tool type
            findings = await self._parse_tool_output(tool_name, output)
            
            # Generate next actions
            next_actions = await self._generate_next_actions(tool_name, findings, tool_category)
            
            # Save results
            result_file = f"{self.results_dir}/{tool_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(result_file, 'w') as f:
                f.write(f"Command: {command_str}\n")
                f.write(f"Execution time: {execution_time:.2f}s\n")
                f.write(f"Exit code: {process.returncode}\n\n")
                f.write("=== STDOUT ===\n")
                f.write(output)
                f.write("\n=== STDERR ===\n")
                f.write(error)
                f.write(f"\n=== FINDINGS ===\n")
                for finding in findings:
                    f.write(f"• {finding}\n")
            
            return KaliToolResult(
                tool_name=tool_name,
                command=command_str,
                success=process.returncode == 0,
                output=output,
                error=error,
                execution_time=execution_time,
                findings=findings,
                next_actions=next_actions
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return KaliToolResult(
                tool_name=tool_name,
                command=command_str,
                success=False,
                output="",
                error=str(e),
                execution_time=execution_time,
                findings=[],
                next_actions=[]
            )
    
    async def _parse_tool_output(self, tool_name: str, output: str) -> List[str]:
        """Parse tool output to extract key findings"""
        findings = []
        
        if tool_name == "nmap":
            # Parse nmap output for open ports and services
            for line in output.split('\n'):
                if '/tcp' in line and 'open' in line:
                    findings.append(f"Open TCP port: {line.strip()}")
                elif '/udp' in line and 'open' in line:
                    findings.append(f"Open UDP port: {line.strip()}")
                elif 'OS details:' in line:
                    findings.append(f"OS detection: {line.split('OS details:')[1].strip()}")
        
        elif tool_name == "nikto":
            # Parse Nikto web vulnerability scan
            for line in output.split('\n'):
                if '+ OSVDB-' in line or '+ CVE-' in line:
                    findings.append(f"Web vulnerability: {line.strip()}")
                elif '+ /admin' in line or '+ /backup' in line:
                    findings.append(f"Interesting directory: {line.strip()}")
        
        elif tool_name == "gobuster":
            # Parse Gobuster directory brute force
            for line in output.split('\n'):
                if 'Status: 200' in line:
                    findings.append(f"Directory found: {line.strip()}")
                elif 'Status: 301' in line or 'Status: 302' in line:
                    findings.append(f"Redirect found: {line.strip()}")
        
        elif tool_name == "sqlmap":
            # Parse SQLMap injection results
            if 'Parameter:' in output and 'is vulnerable' in output:
                findings.append("SQL injection vulnerability confirmed")
            if 'back-end DBMS:' in output:
                dbms_match = re.search(r'back-end DBMS: (.+)', output)
                if dbms_match:
                    findings.append(f"Database: {dbms_match.group(1).strip()}")
        
        elif tool_name == "hydra":
            # Parse Hydra brute force results
            for line in output.split('\n'):
                if '[DATA]' in line and 'login' in line and 'password' in line:
                    findings.append(f"Credential found: {line.strip()}")
        
        elif tool_name == "theharvester":
            # Parse theHarvester email/subdomain results
            for line in output.split('\n'):
                if '@' in line and '.' in line:
                    findings.append(f"Email found: {line.strip()}")
                elif line.startswith('*'):
                    findings.append(f"Subdomain found: {line.strip()}")
        
        # Generic finding extraction
        if not findings:
            # Look for common indicators
            critical_patterns = [
                r'vulnerability',
                r'exploit',
                r'password',
                r'credential',
                r'backdoor',
                r'shell',
                r'admin',
                r'root'
            ]
            
            for pattern in critical_patterns:
                matches = re.findall(pattern, output, re.IGNORECASE)
                if matches:
                    findings.append(f"Pattern match ({pattern}): {len(matches)} occurrences")
        
        return findings
    
    async def _generate_next_actions(self, tool_name: str, findings: List[str], category: str) -> List[str]:
        """Generate intelligent next actions based on tool results"""
        next_actions = []
        
        if tool_name == "nmap" and findings:
            for finding in findings:
                if 'Open TCP port' in finding:
                    port = re.search(r'(\d+)/', finding)
                    if port:
                        port_num = port.group(1)
                        if port_num == "80" or port_num == "443":
                            next_actions.append(f"Run nikto scan on port {port_num}")
                            next_actions.append(f"Run gobuster directory brute force")
                        elif port_num == "22":
                            next_actions.append(f"Attempt SSH brute force with hydra")
                        elif port_num == "3389":
                            next_actions.append(f"Attempt RDP brute force")
        
        elif tool_name == "nikto" and findings:
            next_actions.append("Test identified web vulnerabilities manually")
            next_actions.append("Run SQL injection tests with sqlmap")
            
        elif tool_name == "gobuster" and findings:
            next_actions.append("Investigate discovered directories")
            next_actions.append("Look for backup files and config files")
        
        elif tool_name == "sqlmap" and "SQL injection" in str(findings):
            next_actions.append("Dump database contents")
            next_actions.append("Attempt privilege escalation via SQL")
        
        elif tool_name == "hydra" and findings:
            next_actions.append("Use discovered credentials for lateral movement")
            next_actions.append("Test credentials on other services")
        
        # Category-based next actions
        if category == "reconnaissance" and findings:
            next_actions.append("Move to vulnerability analysis phase")
            next_actions.append("Investigate most promising attack vectors")
        
        elif category == "vulnerability_analysis" and findings:
            next_actions.append("Prepare exploitation attempts")
            next_actions.append("Research available exploits")
        
        return next_actions

class AutonomousRedTeamAgent:
    """AI-powered autonomous red team agent controlling Kali Linux"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.kali_controller = KaliToolController()
        self.reasoning_engine = None
        self.model_manager = None
        self.current_targets = []
        self.attack_phase = "reconnaissance"
        self.findings_db = {}
        
        # Initialize AI components if available
        if HF_AVAILABLE:
            asyncio.create_task(self._initialize_ai())
    
    async def _initialize_ai(self):
        """Initialize AI components"""
        try:
            self.reasoning_engine = AdvancedReasoningEngine()
            self.model_manager = await get_model_manager()
            self.logger.info("✅ AI reasoning engine initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ AI initialization failed: {e}")
    
    async def start_autonomous_engagement(self, target_network: str, objectives: List[str]):
        """Start autonomous red team engagement"""
        self.logger.info(f"🔴 Starting autonomous red team engagement")
        self.logger.info(f"🎯 Target: {target_network}")
        self.logger.info(f"📋 Objectives: {', '.join(objectives)}")
        
        print("🔴 AUTONOMOUS KALI LINUX RED TEAM ACTIVATED")
        print(f"🎯 Target Network: {target_network}")
        print(f"📋 Mission Objectives: {', '.join(objectives)}")
        print("🤖 AI Agent now controlling Kali Linux tools...")
        print("=" * 60)
        
        # Phase 1: Reconnaissance
        await self._execute_reconnaissance_phase(target_network)
        
        # Phase 2: Vulnerability Analysis
        await self._execute_vulnerability_analysis_phase()
        
        # Phase 3: Exploitation (if vulnerabilities found)
        await self._execute_exploitation_phase()
        
        # Phase 4: Post-Exploitation
        await self._execute_post_exploitation_phase()
        
        # Generate final report
        await self._generate_engagement_report()
    
    async def _execute_reconnaissance_phase(self, target_network: str):
        """Execute reconnaissance phase with Kali tools"""
        print("\n🔍 PHASE 1: RECONNAISSANCE")
        print("-" * 30)
        
        # Network discovery with nmap
        print("🔴 Running network discovery scan...")
        nmap_result = await self.kali_controller.execute_tool(
            "nmap", 
            ["-sn", target_network]  # Ping scan
        )
        
        if nmap_result.success:
            print(f"✅ Network scan completed ({nmap_result.execution_time:.2f}s)")
            print(f"📊 Found {len(nmap_result.findings)} hosts")
            
            # Extract discovered hosts
            for line in nmap_result.output.split('\n'):
                if 'Nmap scan report for' in line:
                    ip_match = re.search(r'(\d+\.\d+\.\d+\.\d+)', line)
                    if ip_match:
                        target_ip = ip_match.group(1)
                        self.current_targets.append(PenetrationTarget(ip_address=target_ip))
        
        # Port scanning for each discovered target
        for target in self.current_targets[:3]:  # Limit to first 3 targets
            print(f"🔴 Port scanning {target.ip_address}...")
            
            port_scan_result = await self.kali_controller.execute_tool(
                "nmap",
                ["-sS", "-sV", "-O", "-A", target.ip_address]
            )
            
            if port_scan_result.success:
                print(f"✅ Port scan completed for {target.ip_address}")
                for finding in port_scan_result.findings:
                    print(f"   🔍 {finding}")
                
                # Store findings
                self.findings_db[target.ip_address] = {
                    "port_scan": port_scan_result,
                    "open_ports": [],
                    "services": {}
                }
    
    async def _execute_vulnerability_analysis_phase(self):
        """Execute vulnerability analysis phase"""
        print("\n🔬 PHASE 2: VULNERABILITY ANALYSIS")
        print("-" * 35)
        
        for target in self.current_targets:
            target_findings = self.findings_db.get(target.ip_address, {})
            
            # Web application testing if HTTP/HTTPS found
            has_web = any('80' in str(finding) or '443' in str(finding) 
                         for finding in target_findings.get("port_scan", {}).get("findings", []))
            
            if has_web:
                print(f"🔴 Web vulnerability scanning {target.ip_address}...")
                
                # Nikto web scanner
                nikto_result = await self.kali_controller.execute_tool(
                    "nikto",
                    ["-h", f"http://{target.ip_address}"]
                )
                
                if nikto_result.success:
                    print(f"✅ Nikto scan completed")
                    for finding in nikto_result.findings:
                        print(f"   🚨 {finding}")
                
                # Directory brute force
                print(f"🔴 Directory brute forcing {target.ip_address}...")
                gobuster_result = await self.kali_controller.execute_tool(
                    "gobuster",
                    ["dir", "-u", f"http://{target.ip_address}", "-w", "/usr/share/wordlists/dirb/common.txt"]
                )
                
                if gobuster_result.success:
                    print(f"✅ Directory scan completed")
                    for finding in gobuster_result.findings:
                        print(f"   📂 {finding}")
    
    async def _execute_exploitation_phase(self):
        """Execute exploitation phase"""
        print("\n💥 PHASE 3: EXPLOITATION")
        print("-" * 25)
        
        print("🔴 AI analyzing vulnerabilities for exploitation opportunities...")
        
        # This would contain actual exploitation logic
        # For demo purposes, we'll show the concept
        exploitation_targets = []
        
        for target_ip, findings in self.findings_db.items():
            if findings:
                exploitation_targets.append(target_ip)
        
        if exploitation_targets:
            print(f"🎯 Identified {len(exploitation_targets)} potential exploitation targets")
            for target in exploitation_targets:
                print(f"   • {target}: Analyzing attack vectors...")
        else:
            print("ℹ️ No immediate exploitation opportunities identified")
            print("🔄 Moving to credential-based attacks...")
    
    async def _execute_post_exploitation_phase(self):
        """Execute post-exploitation phase"""
        print("\n🚀 PHASE 4: POST-EXPLOITATION")
        print("-" * 30)
        
        print("🔴 Simulating post-exploitation activities...")
        print("   • Privilege escalation attempts")
        print("   • Lateral movement analysis")
        print("   • Data exfiltration simulation")
        print("   • Persistence mechanisms")
    
    async def _generate_engagement_report(self):
        """Generate comprehensive engagement report"""
        print("\n📊 ENGAGEMENT REPORT")
        print("=" * 25)
        
        total_targets = len(self.current_targets)
        total_findings = sum(len(findings.get("port_scan", {}).get("findings", [])) 
                           for findings in self.findings_db.values())
        
        print(f"🎯 Targets Analyzed: {total_targets}")
        print(f"🔍 Total Findings: {total_findings}")
        print(f"🕐 Engagement Duration: {datetime.now().strftime('%H:%M:%S')}")
        
        # Save detailed report
        report_file = f"/data/reports/red_team_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "targets": [asdict(target) for target in self.current_targets],
            "findings": self.findings_db,
            "phases_completed": ["reconnaissance", "vulnerability_analysis", "exploitation", "post_exploitation"],
            "total_findings": total_findings
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📄 Detailed report saved: {report_file}")

async def main():
    """Main entry point for autonomous red team agent"""
    logging.basicConfig(level=logging.INFO)
    
    # Get target from environment or use default
    target_network = os.getenv('TARGET_NETWORK', '192.168.100.0/24')
    objectives = [
        "Network reconnaissance and mapping",
        "Vulnerability identification and analysis", 
        "Exploitation of discovered weaknesses",
        "Demonstrate compromise capabilities"
    ]
    
    # Create and start autonomous agent
    agent = AutonomousRedTeamAgent()
    
    try:
        await agent.start_autonomous_engagement(target_network, objectives)
    except KeyboardInterrupt:
        print("\n🛑 Red team engagement stopped by user")
    except Exception as e:
        print(f"❌ Red team engagement failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())