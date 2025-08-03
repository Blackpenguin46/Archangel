"""
Red Team AI Agent - Autonomous Penetration Testing
"""

import asyncio
import json
import subprocess
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from ...core.llm_integration import get_llm_manager
from ...core.logging_system import ArchangelLogger

class RedTeamAgent:
    """Autonomous Red Team AI Agent for penetration testing"""
    
    def __init__(self, container_mode: bool = False, model_path: Optional[str] = None):
        self.agent_id = "red-team-ai"
        self.container_mode = container_mode
        self.llm_manager = get_llm_manager(model_path)
        self.logger = None
        
        # Agent state
        self.discovered_hosts = []
        self.compromised_hosts = []
        self.tools_available = []
        self.current_phase = "reconnaissance"
        self.action_history = []
        self.target_network = "172.18.0.0/16"
        
        # CVE Database (simplified for demo)
        self.cve_database = {
            "nginx": ["CVE-2019-20372", "CVE-2021-23017"],
            "apache": ["CVE-2021-44228", "CVE-2022-22965"], 
            "ssh": ["CVE-2020-15778", "CVE-2021-28041"],
            "mysql": ["CVE-2021-2154", "CVE-2022-21245"]
        }
        
    async def initialize(self, session_id: str):
        """Initialize the red team agent"""
        self.logger = ArchangelLogger(session_id)
        await self._check_environment()
        await self._log_initialization()
        
        print(f"🔴 Red Team AI Agent initialized")
        print(f"   Agent ID: {self.agent_id}")
        print(f"   Container Mode: {self.container_mode}")
        print(f"   Target Network: {self.target_network}")
        print(f"   Available Tools: {', '.join(self.tools_available)}")
        
    async def _check_environment(self):
        """Check available tools and environment"""
        tools_to_check = ["nmap", "curl", "nc", "python3"]
        
        if self.container_mode:
            # Running inside container
            for tool in tools_to_check:
                try:
                    result = subprocess.run(["which", tool], capture_output=True, text=True)
                    if result.returncode == 0:
                        self.tools_available.append(tool)
                except:
                    pass
        else:
            # Running on host - check local tools
            self.tools_available = ["python3", "curl"]  # Conservative list
            
    async def _log_initialization(self):
        """Log agent initialization"""
        if self.logger:
            await self.logger.log_system_event({
                'event_type': 'agent_initialization',
                'agent_id': self.agent_id,
                'mode': 'container' if self.container_mode else 'host',
                'tools_available': self.tools_available,
                'target_network': self.target_network
            })
    
    async def autonomous_operation_cycle(self) -> Dict[str, Any]:
        """Execute one autonomous operation cycle"""
        try:
            # Get current context
            context = await self._build_context()
            
            # Generate decision using LLM or intelligent fallback
            decision = await self.llm_manager.generate_red_team_decision(context)
            
            # Execute the decision
            result = await self._execute_action(decision)
            
            # Update agent state
            await self._update_state(decision, result)
            
            # Log the operation
            await self._log_operation(decision, result)
            
            return {
                'decision': decision,
                'result': result,
                'context': context,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_result = {
                'error': str(e),
                'agent_id': self.agent_id,
                'timestamp': datetime.now().isoformat()
            }
            
            if self.logger:
                await self.logger.log_system_event({
                    'event_type': 'agent_error',
                    'error': str(e),
                    'agent_id': self.agent_id
                })
                
            return error_result
    
    async def _build_context(self) -> Dict[str, Any]:
        """Build context for AI decision making"""
        return {
            'discovered_hosts': self.discovered_hosts,
            'compromised_hosts': self.compromised_hosts,
            'current_phase': self.current_phase,
            'previous_actions': self.action_history[-5:],  # Last 5 actions
            'tools_available': self.tools_available,
            'target_network': self.target_network,
            'environment': 'container' if self.container_mode else 'host'
        }
    
    async def _execute_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the AI decision"""
        action_type = decision.get('action_type', 'unknown')
        target = decision.get('target', 'unknown')
        
        if action_type == 'network_scan':
            return await self._execute_network_scan(target, decision.get('parameters', {}))
        elif action_type == 'port_scan':
            return await self._execute_port_scan(target, decision.get('parameters', {}))
        elif action_type == 'service_enumeration':
            return await self._execute_service_enum(target, decision.get('parameters', {}))
        elif action_type == 'vulnerability_scan':
            return await self._execute_vuln_scan(target, decision.get('parameters', {}))
        elif action_type == 'exploit_attempt':
            return await self._execute_exploit(target, decision.get('parameters', {}))
        else:
            return {
                'success': False,
                'error': f'Unknown action type: {action_type}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def _execute_network_scan(self, target: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute network discovery scan"""
        try:
            if "nmap" in self.tools_available:
                if self.container_mode:
                    cmd = ["nmap", "-sn", target]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    
                    hosts = []
                    if result.returncode == 0:
                        for line in result.stdout.split('\\n'):
                            if "Nmap scan report for" in line:
                                host = line.split()[-1].strip('()')
                                if host not in self.discovered_hosts:
                                    hosts.append(host)
                                    self.discovered_hosts.append(host)
                    
                    return {
                        'success': result.returncode == 0,
                        'hosts_discovered': hosts,
                        'total_discovered': len(self.discovered_hosts),
                        'output': result.stdout,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    # Simulated scan for host mode
                    simulated_hosts = ["172.18.0.3", "172.18.0.4"]
                    for host in simulated_hosts:
                        if host not in self.discovered_hosts:
                            self.discovered_hosts.append(host)
                    
                    return {
                        'success': True,
                        'hosts_discovered': simulated_hosts,
                        'total_discovered': len(self.discovered_hosts),
                        'output': f'Simulated scan discovered {len(simulated_hosts)} hosts',
                        'timestamp': datetime.now().isoformat()
                    }
            else:
                return {
                    'success': False,
                    'error': 'nmap not available',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _execute_port_scan(self, target: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute port scan on target"""
        try:
            if "nmap" in self.tools_available and self.container_mode:
                scan_type = params.get('scan_type', 'fast')
                ports_arg = "-F" if scan_type == 'fast' else "-p-"
                
                cmd = ["nmap", ports_arg, target]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                open_ports = []
                if result.returncode == 0:
                    for line in result.stdout.split('\\n'):
                        if "/tcp" in line and "open" in line:
                            port = line.split('/')[0].strip()
                            open_ports.append(port)
                
                return {
                    'success': result.returncode == 0,
                    'target': target,
                    'open_ports': open_ports,
                    'output': result.stdout,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Simulated port scan
                simulated_ports = ["80", "443", "22"]
                return {
                    'success': True,
                    'target': target,
                    'open_ports': simulated_ports,
                    'output': f'Simulated port scan found ports: {", ".join(simulated_ports)}',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _execute_service_enum(self, target: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute service enumeration"""
        try:
            if "nmap" in self.tools_available and self.container_mode:
                cmd = ["nmap", "-sV", "-F", target]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
                
                services = []
                if result.returncode == 0:
                    for line in result.stdout.split('\\n'):
                        if "/tcp" in line and "open" in line:
                            parts = line.split()
                            if len(parts) >= 3:
                                port = parts[0]
                                service = parts[2] if len(parts) > 2 else 'unknown'
                                services.append({'port': port, 'service': service})
                
                return {
                    'success': result.returncode == 0,
                    'target': target,
                    'services': services,
                    'output': result.stdout,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Simulated service enumeration
                simulated_services = [
                    {'port': '80/tcp', 'service': 'nginx'},
                    {'port': '443/tcp', 'service': 'nginx'},
                    {'port': '22/tcp', 'service': 'ssh'}
                ]
                return {
                    'success': True,
                    'target': target,
                    'services': simulated_services,
                    'output': 'Simulated service enumeration completed',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _execute_vuln_scan(self, target: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute vulnerability scan"""
        try:
            # Check for CVEs based on discovered services
            vulnerabilities = []
            
            # Simulated vulnerability assessment
            for service_info in params.get('services', []):
                service_name = service_info.get('service', '').lower()
                if service_name in self.cve_database:
                    cves = self.cve_database[service_name]
                    for cve in cves[:2]:  # Limit to 2 CVEs per service
                        vulnerabilities.append({
                            'cve': cve,
                            'service': service_name,
                            'severity': 'medium',
                            'exploitable': True
                        })
            
            return {
                'success': True,
                'target': target,
                'vulnerabilities': vulnerabilities,
                'vuln_count': len(vulnerabilities),
                'output': f'Vulnerability scan found {len(vulnerabilities)} potential issues',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _execute_exploit(self, target: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute exploit attempt (simulated for safety)"""
        # Simulated exploitation for demonstration
        vulnerabilities = params.get('vulnerabilities', [])
        
        if vulnerabilities:
            # Simulate exploitation attempt
            exploit_success = len(vulnerabilities) > 2  # More vulns = higher success chance
            
            if exploit_success and target not in self.compromised_hosts:
                self.compromised_hosts.append(target)
                self.current_phase = "post_exploitation"
                
                return {
                    'success': True,
                    'target': target,
                    'exploit_used': vulnerabilities[0].get('cve', 'unknown'),
                    'access_gained': 'user_level',
                    'compromised': True,
                    'output': f'Successfully compromised {target}',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'target': target,
                    'error': 'Exploitation attempt failed',
                    'compromised': False,
                    'output': f'Failed to compromise {target}',
                    'timestamp': datetime.now().isoformat()
                }
        else:
            return {
                'success': False,
                'error': 'No vulnerabilities available for exploitation',
                'timestamp': datetime.now().isoformat()
            }
    
    async def _update_state(self, decision: Dict[str, Any], result: Dict[str, Any]):
        """Update agent state based on action results"""
        self.action_history.append({
            'decision': decision,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update phase based on progress
        if len(self.discovered_hosts) > 0 and self.current_phase == "reconnaissance":
            self.current_phase = "enumeration"
        elif len(self.compromised_hosts) > 0 and self.current_phase == "enumeration":
            self.current_phase = "exploitation"
    
    async def _log_operation(self, decision: Dict[str, Any], result: Dict[str, Any]):
        """Log the operation to the system logger"""
        if self.logger:
            await self.logger.log_ai_reasoning({
                'agent_id': self.agent_id,
                'agent_type': 'red_team',
                'decision': decision,
                'result': result,
                'context': {
                    'phase': self.current_phase,
                    'discovered_hosts': len(self.discovered_hosts),
                    'compromised_hosts': len(self.compromised_hosts)
                }
            })
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'agent_id': self.agent_id,
            'agent_type': 'red_team',
            'status': 'operational',
            'current_phase': self.current_phase,
            'discovered_hosts': self.discovered_hosts,
            'compromised_hosts': self.compromised_hosts,
            'tools_available': self.tools_available,
            'actions_taken': len(self.action_history),
            'timestamp': datetime.now().isoformat()
        }