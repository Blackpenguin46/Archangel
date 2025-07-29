"""
Archangel Tool Integration System
AI-driven security tool orchestration
"""

import asyncio
import subprocess
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import tempfile
import os

class ToolType(Enum):
    RECONNAISSANCE = "reconnaissance"
    SCANNING = "scanning"
    ENUMERATION = "enumeration"
    EXPLOITATION = "exploitation"
    ANALYSIS = "analysis"

@dataclass
class ToolResult:
    """Results from tool execution"""
    tool_name: str
    command: str
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    parsed_data: Optional[Dict[str, Any]] = None

class SecurityTool:
    """Base class for security tools"""
    
    def __init__(self, name: str, tool_type: ToolType):
        self.name = name
        self.tool_type = tool_type
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if tool is available on system"""
        try:
            result = subprocess.run(['which', self.name], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    async def execute(self, command: List[str], timeout: int = 60) -> ToolResult:
        """Execute tool command"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return ToolResult(
                tool_name=self.name,
                command=' '.join(command),
                exit_code=process.returncode,
                stdout=stdout.decode('utf-8', errors='ignore'),
                stderr=stderr.decode('utf-8', errors='ignore'),
                execution_time=execution_time
            )
            
        except asyncio.TimeoutError:
            return ToolResult(
                tool_name=self.name,
                command=' '.join(command),
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                execution_time=timeout
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                command=' '.join(command),
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=0
            )

class NmapTool(SecurityTool):
    """Nmap network scanning tool"""
    
    def __init__(self):
        super().__init__("nmap", ToolType.SCANNING)
    
    async def ping_scan(self, target: str) -> ToolResult:
        """Perform ping scan to discover live hosts"""
        command = ['nmap', '-sn', target]
        result = await self.execute(command)
        result.parsed_data = self._parse_ping_scan(result.stdout)
        return result
    
    async def port_scan(self, target: str, ports: str = "1-1000") -> ToolResult:
        """Perform port scan"""
        command = ['nmap', '-sS', '-p', ports, target]
        result = await self.execute(command)
        result.parsed_data = self._parse_port_scan(result.stdout)
        return result
    
    async def service_scan(self, target: str) -> ToolResult:
        """Perform service version detection"""
        command = ['nmap', '-sV', '-sC', target]
        result = await self.execute(command)
        result.parsed_data = self._parse_service_scan(result.stdout)
        return result
    
    def _parse_ping_scan(self, output: str) -> Dict[str, Any]:
        """Parse ping scan output"""
        lines = output.split('\n')
        hosts = []
        
        for line in lines:
            if 'Nmap scan report for' in line:
                host = line.split('for ')[-1].strip()
                hosts.append(host)
        
        return {"live_hosts": hosts, "count": len(hosts)}
    
    def _parse_port_scan(self, output: str) -> Dict[str, Any]:
        """Parse port scan output"""
        lines = output.split('\n')
        ports = []
        
        for line in lines:
            if '/tcp' in line and 'open' in line:
                parts = line.split()
                if len(parts) >= 2:
                    port_info = {
                        "port": parts[0],
                        "state": parts[1],
                        "service": parts[2] if len(parts) > 2 else "unknown"
                    }
                    ports.append(port_info)
        
        return {"open_ports": ports, "count": len(ports)}
    
    def _parse_service_scan(self, output: str) -> Dict[str, Any]:
        """Parse service scan output"""
        lines = output.split('\n')
        services = []
        
        for line in lines:
            if '/tcp' in line and 'open' in line:
                parts = line.split()
                if len(parts) >= 3:
                    service_info = {
                        "port": parts[0],
                        "state": parts[1], 
                        "service": parts[2],
                        "version": ' '.join(parts[3:]) if len(parts) > 3 else ""
                    }
                    services.append(service_info)
        
        return {"services": services, "count": len(services)}

class AIToolOrchestrator:
    """
    AI-driven tool orchestration system
    
    This is where the AI makes intelligent decisions about which tools to use
    and how to use them based on the target and context.
    """
    
    def __init__(self):
        self.tools = self._initialize_tools()
        self.execution_history: List[ToolResult] = []
    
    def _initialize_tools(self) -> Dict[str, SecurityTool]:
        """Initialize available security tools"""
        tools = {}
        
        # Add tools
        nmap = NmapTool()
        if nmap.available:
            tools['nmap'] = nmap
        
        return tools
    
    async def ai_select_tool(self, objective: str, target: str, context: Dict[str, Any]) -> Optional[SecurityTool]:
        """
        AI intelligently selects the best tool for the objective
        
        This simulates AI reasoning about tool selection
        """
        objective_lower = objective.lower()
        
        # Simple heuristic-based tool selection (in reality would use LLM)
        if any(keyword in objective_lower for keyword in ['scan', 'discover', 'enumerate']):
            if 'nmap' in self.tools:
                return self.tools['nmap']
        
        return None
    
    async def ai_execute_strategy(self, target: str, strategy: Dict[str, Any]) -> List[ToolResult]:
        """
        AI executes a multi-phase security strategy
        
        This demonstrates autonomous tool orchestration
        """
        results = []
        
        print(f"🤖 AI executing strategy for {target}")
        
        for phase_name, phase_data in strategy.items():
            print(f"📋 Executing {phase_data['name']}...")
            
            # AI selects appropriate tools for this phase
            for tool_name in phase_data['tools']:
                if tool_name == 'nmap' and 'nmap' in self.tools:
                    nmap_tool = self.tools['nmap']
                    
                    # AI decides which nmap scan to run based on phase
                    if 'reconnaissance' in phase_data['name'].lower():
                        result = await nmap_tool.ping_scan(target)
                    elif 'enumeration' in phase_data['name'].lower():
                        result = await nmap_tool.service_scan(target)
                    else:
                        result = await nmap_tool.port_scan(target)
                    
                    results.append(result)
                    self.execution_history.append(result)
                    
                    # AI analyzes results and adapts
                    await self._ai_analyze_result(result)
        
        return results
    
    async def _ai_analyze_result(self, result: ToolResult):
        """AI analyzes tool results and adapts strategy if needed"""
        print(f"🧠 AI analyzing results from {result.tool_name}...")
        
        if result.exit_code != 0:
            print(f"⚠️ AI detected issue: {result.stderr}")
            print("🔄 AI adapting strategy...")
        elif result.parsed_data:
            if 'open_ports' in result.parsed_data:
                port_count = result.parsed_data['count']
                print(f"📊 AI found {port_count} open ports")
                
                if port_count > 10:
                    print("🧠 AI reasoning: Many ports open, possible misconfiguration")
                elif port_count == 0:
                    print("🧠 AI reasoning: No ports found, target may be filtered")
            
            if 'services' in result.parsed_data:
                service_count = result.parsed_data['count']
                print(f"📊 AI identified {service_count} services")
                
                # AI looks for interesting services
                for service in result.parsed_data['services']:
                    if any(keyword in service['service'].lower() 
                          for keyword in ['http', 'ssh', 'ftp', 'mysql']):
                        print(f"🎯 AI flagged interesting service: {service['service']} on {service['port']}")
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return [tool.name for tool in self.tools.values() if tool.available]
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of tool executions"""
        total_executions = len(self.execution_history)
        successful_executions = len([r for r in self.execution_history if r.exit_code == 0])
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "tools_used": list(set(r.tool_name for r in self.execution_history))
        }

# Mock tool for demonstration when nmap isn't available
class MockNmapTool(SecurityTool):
    """Mock nmap tool for demonstration purposes"""
    
    def __init__(self):
        super().__init__("mock-nmap", ToolType.SCANNING)
        self.available = True  # Always available
    
    async def ping_scan(self, target: str) -> ToolResult:
        """Mock ping scan"""
        await asyncio.sleep(1)  # Simulate execution time
        
        mock_output = f"""
Starting Nmap 7.80 ( https://nmap.org ) at 2024-01-01 12:00 UTC
Nmap scan report for {target} ({target})
Host is up (0.050s latency).
Nmap done: 1 IP address (1 host up) scanned in 1.23 seconds
        """.strip()
        
        result = ToolResult(
            tool_name="mock-nmap",
            command=f"nmap -sn {target}",
            exit_code=0,
            stdout=mock_output,
            stderr="",
            execution_time=1.0
        )
        
        result.parsed_data = {"live_hosts": [target], "count": 1}
        return result
    
    async def port_scan(self, target: str, ports: str = "1-1000") -> ToolResult:
        """Mock port scan"""
        await asyncio.sleep(2)  # Simulate execution time
        
        mock_output = f"""
Starting Nmap 7.80 ( https://nmap.org ) at 2024-01-01 12:00 UTC
Nmap scan report for {target}
Host is up (0.050s latency).
PORT     STATE SERVICE
22/tcp   open  ssh
80/tcp   open  http
443/tcp  open  https

Nmap done: 1 IP address (1 host up) scanned in 2.45 seconds
        """.strip()
        
        result = ToolResult(
            tool_name="mock-nmap",
            command=f"nmap -sS -p {ports} {target}",
            exit_code=0,
            stdout=mock_output,
            stderr="",
            execution_time=2.0
        )
        
        result.parsed_data = {
            "open_ports": [
                {"port": "22/tcp", "state": "open", "service": "ssh"},
                {"port": "80/tcp", "state": "open", "service": "http"}, 
                {"port": "443/tcp", "state": "open", "service": "https"}
            ],
            "count": 3
        }
        return result
    
    async def service_scan(self, target: str) -> ToolResult:
        """Mock service scan"""
        await asyncio.sleep(3)  # Simulate execution time
        
        mock_output = f"""
Starting Nmap 7.80 ( https://nmap.org ) at 2024-01-01 12:00 UTC
Nmap scan report for {target}
Host is up (0.050s latency).
PORT    STATE SERVICE  VERSION
22/tcp  open  ssh      OpenSSH 8.2p1 Ubuntu 4ubuntu0.3
80/tcp  open  http     Apache httpd 2.4.41
443/tcp open  https    Apache httpd 2.4.41 ((Ubuntu))

Nmap done: 1 IP address (1 host up) scanned in 3.67 seconds
        """.strip()
        
        result = ToolResult(
            tool_name="mock-nmap",
            command=f"nmap -sV -sC {target}",
            exit_code=0,
            stdout=mock_output,
            stderr="",
            execution_time=3.0
        )
        
        result.parsed_data = {
            "services": [
                {"port": "22/tcp", "state": "open", "service": "ssh", "version": "OpenSSH 8.2p1 Ubuntu 4ubuntu0.3"},
                {"port": "80/tcp", "state": "open", "service": "http", "version": "Apache httpd 2.4.41"},
                {"port": "443/tcp", "state": "open", "service": "https", "version": "Apache httpd 2.4.41 ((Ubuntu))"}
            ],
            "count": 3
        }
        return result

# Add mock tool to orchestrator if real tools aren't available
def create_ai_orchestrator() -> AIToolOrchestrator:
    """Create AI tool orchestrator with available tools"""
    orchestrator = AIToolOrchestrator()
    
    # Add mock tools if real ones aren't available
    if 'nmap' not in orchestrator.tools:
        mock_nmap = MockNmapTool()
        orchestrator.tools['nmap'] = mock_nmap
    
    return orchestrator