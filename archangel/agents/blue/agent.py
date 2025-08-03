"""
Blue Team AI Agent - Autonomous Security Monitoring and Defense
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

class BlueTeamAgent:
    """Autonomous Blue Team AI Agent for security monitoring and defense"""
    
    def __init__(self, container_mode: bool = False, model_path: Optional[str] = None):
        self.agent_id = "blue-team-ai"
        self.container_mode = container_mode
        self.llm_manager = get_llm_manager(model_path)
        self.logger = None
        
        # Agent state
        self.detected_threats = []
        self.blocked_ips = []
        self.active_connections = []
        self.monitoring_alerts = []
        self.tools_available = []
        self.defense_posture = "baseline"  # baseline, elevated, high_alert
        self.action_history = []
        
        # Threat signatures (simplified for demo)
        self.threat_signatures = {
            'port_scan': ['nmap', 'masscan', 'rapid_connections'],
            'brute_force': ['failed_login', 'multiple_attempts'],
            'malware': ['suspicious_process', 'network_beacon'],
            'data_exfil': ['large_transfer', 'unusual_traffic']
        }
        
        # Response playbooks
        self.response_playbooks = {
            'port_scan': ['block_source_ip', 'increase_monitoring'],
            'brute_force': ['block_source_ip', 'lockout_account'],
            'malware': ['isolate_host', 'kill_process'],
            'data_exfil': ['block_traffic', 'forensic_capture']
        }
        
    async def initialize(self, session_id: str):
        """Initialize the blue team agent"""
        self.logger = ArchangelLogger(session_id)
        await self._check_environment()
        await self._setup_monitoring()
        await self._log_initialization()
        
        print(f"🔵 Blue Team AI Agent initialized")
        print(f"   Agent ID: {self.agent_id}")
        print(f"   Container Mode: {self.container_mode}")
        print(f"   Defense Posture: {self.defense_posture}")
        print(f"   Available Tools: {', '.join(self.tools_available)}")
        
    async def _check_environment(self):
        """Check available defensive tools"""
        tools_to_check = ["iptables", "tcpdump", "ps", "ss", "netstat", "python3"]
        
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
            self.tools_available = ["python3", "ps"]  # Conservative list
            
    async def _setup_monitoring(self):
        """Setup security monitoring"""
        try:
            # Create monitoring directories
            if self.container_mode:
                os.makedirs("/var/log/security", exist_ok=True)
                
            # Initialize baseline monitoring
            await self._baseline_monitoring()
            
        except Exception as e:
            print(f"⚠️ Failed to setup monitoring: {e}")
    
    async def _log_initialization(self):
        """Log agent initialization"""
        if self.logger:
            await self.logger.log_system_event({
                'event_type': 'agent_initialization',
                'agent_id': self.agent_id,
                'mode': 'container' if self.container_mode else 'host',
                'tools_available': self.tools_available,
                'defense_posture': self.defense_posture
            })
    
    async def autonomous_operation_cycle(self) -> Dict[str, Any]:
        """Execute one autonomous operation cycle"""
        try:
            # Monitor environment for threats
            monitoring_data = await self._monitor_environment()
            
            # Analyze threats
            threat_analysis = await self._analyze_threats(monitoring_data)
            
            # Get current context
            context = await self._build_context(threat_analysis)
            
            # Generate decision using LLM or intelligent fallback
            decision = await self.llm_manager.generate_blue_team_decision(context)
            
            # Execute the decision
            result = await self._execute_action(decision)
            
            # Update agent state
            await self._update_state(decision, result, threat_analysis)
            
            # Log the operation
            await self._log_operation(decision, result, threat_analysis)
            
            return {
                'decision': decision,
                'result': result,
                'threat_analysis': threat_analysis,
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
    
    async def _monitor_environment(self) -> Dict[str, Any]:
        """Monitor environment for security events"""
        monitoring_data = {
            'network_connections': [],
            'running_processes': [],
            'system_logs': [],
            'suspicious_activity': []
        }
        
        # Monitor network connections
        try:
            connections = await self._get_network_connections()
            monitoring_data['network_connections'] = connections
            
            # Detect suspicious network activity
            suspicious_net = await self._detect_suspicious_network(connections)
            monitoring_data['suspicious_activity'].extend(suspicious_net)
            
        except Exception as e:
            print(f"⚠️ Network monitoring error: {e}")
        
        # Monitor running processes
        try:
            processes = await self._get_running_processes()
            monitoring_data['running_processes'] = processes
            
            # Detect suspicious processes
            suspicious_proc = await self._detect_suspicious_processes(processes)
            monitoring_data['suspicious_activity'].extend(suspicious_proc)
            
        except Exception as e:
            print(f"⚠️ Process monitoring error: {e}")
        
        return monitoring_data
    
    async def _get_network_connections(self) -> List[Dict[str, Any]]:
        """Get current network connections"""
        connections = []
        
        try:
            if "ss" in self.tools_available and self.container_mode:
                cmd = ["ss", "-tuln"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    for line in result.stdout.split('\\n')[1:]:  # Skip header
                        if line.strip():
                            parts = line.split()
                            if len(parts) >= 4:
                                connections.append({
                                    'protocol': parts[0],
                                    'local_address': parts[3],
                                    'state': parts[4] if len(parts) > 4 else 'unknown',
                                    'timestamp': datetime.now().isoformat()
                                })
            
            self.active_connections = connections
            
        except Exception as e:
            print(f"⚠️ Network connection monitoring error: {e}")
        
        return connections
    
    async def _get_running_processes(self) -> List[Dict[str, Any]]:
        """Get running processes"""
        processes = []
        
        try:
            if "ps" in self.tools_available:
                cmd = ["ps", "aux"] if self.container_mode else ["ps", "-ef"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    for line in result.stdout.split('\\n')[1:]:  # Skip header
                        if line.strip():
                            parts = line.split(None, 10)  # Split into max 11 parts
                            if len(parts) >= 8:
                                processes.append({
                                    'user': parts[0],
                                    'pid': parts[1],
                                    'cpu': parts[2],
                                    'mem': parts[3],
                                    'command': parts[-1] if len(parts) > 10 else 'unknown',
                                    'timestamp': datetime.now().isoformat()
                                })
                                
        except Exception as e:
            print(f"⚠️ Process monitoring error: {e}")
        
        return processes
    
    async def _detect_suspicious_network(self, connections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect suspicious network activity"""
        suspicious = []
        
        # Look for new connections that weren't in previous scan
        new_connections = []
        for conn in connections:
            local_addr = conn.get('local_address', '')
            if not any(c.get('local_address') == local_addr for c in self.active_connections):
                new_connections.append(conn)
        
        # Flag new listening services as potentially suspicious
        for conn in new_connections:
            if conn.get('state') == 'LISTEN':
                suspicious.append({
                    'type': 'new_listening_service',
                    'source': conn.get('local_address', 'unknown'),
                    'details': conn,
                    'severity': 'medium',
                    'timestamp': datetime.now().isoformat()
                })
        
        return suspicious
    
    async def _detect_suspicious_processes(self, processes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect suspicious processes"""
        suspicious = []
        
        for proc in processes:
            command = proc.get('command', '').lower()
            
            # Check for penetration testing tools
            if any(tool in command for tool in ['nmap', 'scan', 'masscan', 'nikto']):
                suspicious.append({
                    'type': 'pentest_tool_detected',
                    'source': proc.get('pid', 'unknown'),
                    'details': proc,
                    'severity': 'high',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Check for unusual network activity
            elif any(keyword in command for keyword in ['nc', 'netcat', 'curl', 'wget']):
                suspicious.append({
                    'type': 'network_tool_activity',
                    'source': proc.get('pid', 'unknown'),
                    'details': proc,
                    'severity': 'medium',
                    'timestamp': datetime.now().isoformat()
                })
        
        return suspicious
    
    async def _analyze_threats(self, monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze monitoring data for threats"""
        threats = monitoring_data.get('suspicious_activity', [])
        
        # Categorize threats
        threat_categories = {}
        for threat in threats:
            threat_type = threat.get('type', 'unknown')
            if threat_type not in threat_categories:
                threat_categories[threat_type] = []
            threat_categories[threat_type].append(threat)
        
        # Calculate threat level
        threat_level = self._calculate_threat_level(threats)
        
        # Update detected threats
        self.detected_threats.extend(threats)
        
        return {
            'new_threats': len(threats),
            'threat_categories': threat_categories,
            'threat_level': threat_level,
            'total_threats': len(self.detected_threats),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_threat_level(self, threats: List[Dict[str, Any]]) -> str:
        """Calculate overall threat level"""
        if not threats:
            return "low"
        
        high_severity_count = sum(1 for t in threats if t.get('severity') == 'high')
        medium_severity_count = sum(1 for t in threats if t.get('severity') == 'medium')
        
        if high_severity_count > 0 or medium_severity_count > 2:
            return "high"
        elif medium_severity_count > 0:
            return "medium"
        else:
            return "low"
    
    async def _build_context(self, threat_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build context for AI decision making"""
        return {
            'detected_threats': self.detected_threats[-10:],  # Last 10 threats
            'blocked_ips': self.blocked_ips,
            'active_connections': self.active_connections,
            'defense_posture': self.defense_posture,
            'threat_analysis': threat_analysis,
            'tools_available': self.tools_available,
            'previous_actions': self.action_history[-5:],  # Last 5 actions
            'environment': 'container' if self.container_mode else 'host'
        }
    
    async def _execute_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the AI decision"""
        action_type = decision.get('action_type', 'unknown')
        target = decision.get('target', 'unknown')
        
        if action_type == 'block_threat':
            return await self._execute_block_threat(target, decision.get('parameters', {}))
        elif action_type == 'increase_monitoring':
            return await self._execute_increase_monitoring(target, decision.get('parameters', {}))
        elif action_type == 'isolate_system':
            return await self._execute_isolate_system(target, decision.get('parameters', {}))
        elif action_type == 'baseline_monitor':
            return await self._execute_baseline_monitor(target, decision.get('parameters', {}))
        elif action_type == 'threat_investigation':
            return await self._execute_threat_investigation(target, decision.get('parameters', {}))
        else:
            return {
                'success': False,
                'error': f'Unknown action type: {action_type}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def _execute_block_threat(self, target: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Block threat source using iptables"""
        try:
            if "iptables" in self.tools_available and self.container_mode:
                # Extract IP from target
                ip = target.split(':')[0] if ':' in target else target
                
                if ip not in self.blocked_ips and ip not in ['127.0.0.1', '0.0.0.0', 'localhost']:
                    cmd = ["iptables", "-I", "INPUT", "-s", ip, "-j", "DROP"]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        self.blocked_ips.append(ip)
                        return {
                            'success': True,
                            'action': 'ip_blocked',
                            'target': ip,
                            'blocked_ips_total': len(self.blocked_ips),
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        return {
                            'success': False,
                            'error': f'iptables command failed: {result.stderr}',
                            'timestamp': datetime.now().isoformat()
                        }
                else:
                    return {
                        'success': False,
                        'error': f'IP {ip} already blocked or invalid',
                        'timestamp': datetime.now().isoformat()
                    }
            else:
                # Simulated blocking for host mode
                ip = target.split(':')[0] if ':' in target else target
                if ip not in self.blocked_ips:
                    self.blocked_ips.append(ip)
                
                return {
                    'success': True,
                    'action': 'ip_blocked_simulated',
                    'target': ip,
                    'blocked_ips_total': len(self.blocked_ips),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _execute_increase_monitoring(self, target: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Increase monitoring intensity"""
        try:
            self.defense_posture = "elevated"
            self.monitoring_alerts.append({
                'type': 'monitoring_increased',
                'target': target,
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'success': True,
                'action': 'monitoring_increased',
                'new_posture': self.defense_posture,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _execute_isolate_system(self, target: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Isolate potentially compromised system"""
        try:
            # Simulated system isolation
            self.defense_posture = "high_alert"
            
            return {
                'success': True,
                'action': 'system_isolated',
                'target': target,
                'new_posture': self.defense_posture,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _execute_baseline_monitor(self, target: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute baseline monitoring"""
        return await self._baseline_monitoring()
    
    async def _baseline_monitoring(self) -> Dict[str, Any]:
        """Perform baseline security monitoring"""
        try:
            monitoring_stats = {
                'connections_monitored': len(self.active_connections),
                'threats_detected': len(self.detected_threats),
                'ips_blocked': len(self.blocked_ips),
                'defense_posture': self.defense_posture
            }
            
            return {
                'success': True,
                'action': 'baseline_monitoring',
                'stats': monitoring_stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _execute_threat_investigation(self, target: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute detailed threat investigation"""
        try:
            # Analyze recent threats for patterns
            recent_threats = self.detected_threats[-5:]
            
            investigation_results = {
                'threats_analyzed': len(recent_threats),
                'threat_types': list(set(t.get('type', 'unknown') for t in recent_threats)),
                'pattern_detected': len(recent_threats) > 2,
                'recommendation': 'increase_monitoring' if len(recent_threats) > 2 else 'continue_baseline'
            }
            
            return {
                'success': True,
                'action': 'threat_investigation',
                'results': investigation_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _update_state(self, decision: Dict[str, Any], result: Dict[str, Any], threat_analysis: Dict[str, Any]):
        """Update agent state based on action results"""
        self.action_history.append({
            'decision': decision,
            'result': result,
            'threat_analysis': threat_analysis,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update defense posture based on threat level
        threat_level = threat_analysis.get('threat_level', 'low')
        if threat_level == 'high' and self.defense_posture != 'high_alert':
            self.defense_posture = 'high_alert'
        elif threat_level == 'medium' and self.defense_posture == 'baseline':
            self.defense_posture = 'elevated'
    
    async def _log_operation(self, decision: Dict[str, Any], result: Dict[str, Any], threat_analysis: Dict[str, Any]):
        """Log the operation to the system logger"""
        if self.logger:
            await self.logger.log_ai_reasoning({
                'agent_id': self.agent_id,
                'agent_type': 'blue_team',
                'decision': decision,
                'result': result,
                'threat_analysis': threat_analysis,
                'context': {
                    'defense_posture': self.defense_posture,
                    'threats_detected': len(self.detected_threats),
                    'ips_blocked': len(self.blocked_ips)
                }
            })
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'agent_id': self.agent_id,
            'agent_type': 'blue_team',
            'status': 'operational',
            'defense_posture': self.defense_posture,
            'detected_threats': len(self.detected_threats),
            'blocked_ips': len(self.blocked_ips),
            'active_connections': len(self.active_connections),
            'tools_available': self.tools_available,
            'actions_taken': len(self.action_history),
            'timestamp': datetime.now().isoformat()
        }