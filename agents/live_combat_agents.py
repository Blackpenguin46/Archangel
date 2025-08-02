#!/usr/bin/env python3
"""
Live Combat Agents - Autonomous Red Team vs Blue Team AI Agents
Real-time adversarial AI agents that learn from live hacking scenarios
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import subprocess
import threading

# Import our existing agent framework
import sys
sys.path.append('.')
from core.autonomous_security_agents import AutonomousSecurityAgent, BlueTeamDefenderAgent, RedTeamAttackerAgent
from environments.live_adversarial_environment import LiveAdversarialEnvironment, AttackScenario

@dataclass
class LiveCombatMetrics:
    """Real-time combat metrics"""
    attacks_attempted: int = 0
    attacks_successful: int = 0
    defenses_triggered: int = 0
    detection_time_avg: float = 0.0
    response_time_avg: float = 0.0
    learning_events: int = 0
    adaptation_count: int = 0

class LiveRedTeamAgent(RedTeamAttackerAgent):
    """Enhanced Red Team agent for live combat scenarios"""
    
    def __init__(self, agent_id: str, combat_environment: LiveAdversarialEnvironment):
        super().__init__(agent_id)
        self.combat_env = combat_environment
        self.live_metrics = LiveCombatMetrics()
        self.attack_history = []
        self.target_ip = "192.168.100.20"  # Blue team IP
        self.current_attack_plan = None
        self.learning_enabled = True
        
    async def autonomous_live_attack_operation(self, scenario: AttackScenario, duration_minutes: int = 30) -> Dict[str, Any]:
        """Execute autonomous live attack operation"""
        operation_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        self.logger.info(f"🔴 RED TEAM AGENT: Starting live attack operation")
        self.logger.info(f"🎯 Scenario: {scenario.name}")
        self.logger.info(f"⏱️ Duration: {duration_minutes} minutes")
        
        operation_results = {
            "operation_id": operation_id,
            "agent_id": self.agent_id,
            "scenario": scenario.name,
            "start_time": start_time,
            "status": "active",
            "attacks_executed": [],
            "learning_data": [],
            "adaptations_made": []
        }
        
        try:
            # Generate adaptive attack plan
            attack_plan = await self._generate_adaptive_attack_plan(scenario)
            self.current_attack_plan = attack_plan
            
            # Execute attacks in real-time
            end_time = start_time + timedelta(minutes=duration_minutes)
            
            while datetime.now() < end_time:
                # Select next attack based on current intelligence
                next_attack = await self._select_optimal_attack()
                
                if next_attack:
                    # Execute attack
                    attack_result = await self._execute_live_attack(next_attack)
                    operation_results["attacks_executed"].append(attack_result)
                    
                    # Learn from attack result
                    if self.learning_enabled:
                        learning_insight = await self._learn_from_attack_result(attack_result)
                        operation_results["learning_data"].append(learning_insight)
                    
                    # Adapt strategy if needed
                    if attack_result.get("blocked") or not attack_result.get("success"):
                        adaptation = await self._adapt_attack_strategy(attack_result)
                        operation_results["adaptations_made"].append(adaptation)
                
                # Wait before next attack (realistic timing)
                await asyncio.sleep(30)  # 30 seconds between attacks
            
            operation_results["status"] = "completed"
            operation_results["end_time"] = datetime.now()
            operation_results["live_metrics"] = asdict(self.live_metrics)
            
            self.logger.info(f"🔴 RED TEAM AGENT: Operation completed")
            self.logger.info(f"📊 Attacks: {self.live_metrics.attacks_attempted} attempted, {self.live_metrics.attacks_successful} successful")
            
        except Exception as e:
            self.logger.error(f"🔴 RED TEAM AGENT ERROR: {e}")
            operation_results["status"] = "failed"
            operation_results["error"] = str(e)
        
        return operation_results
    
    async def _generate_adaptive_attack_plan(self, scenario: AttackScenario) -> Dict[str, Any]:
        """Generate adaptive attack plan based on scenario and intelligence"""
        attack_plan = {
            "scenario": scenario.name,
            "phases": [],
            "target_services": scenario.target_services,
            "tools_available": scenario.attack_tools,
            "success_criteria": scenario.success_criteria,
            "adaptive_triggers": []
        }
        
        # Phase 1: Reconnaissance
        recon_phase = {
            "phase": "reconnaissance",
            "attacks": [
                {"type": "port_scan", "tool": "nmap", "target": self.target_ip, "stealth": True},
                {"type": "service_enum", "tool": "nmap", "target": self.target_ip, "ports": "22,80,21"},
                {"type": "banner_grab", "tool": "netcat", "target": self.target_ip, "port": 22}
            ],
            "success_threshold": 0.7,
            "fallback_strategy": "increase_stealth"
        }
        attack_plan["phases"].append(recon_phase)
        
        # Phase 2: Initial Access
        access_phase = {
            "phase": "initial_access",
            "attacks": [
                {"type": "ssh_bruteforce", "tool": "hydra", "target": self.target_ip, "wordlist": "common_passwords"},
                {"type": "web_vuln_scan", "tool": "nikto", "target": f"http://{self.target_ip}"},
                {"type": "directory_enum", "tool": "dirb", "target": f"http://{self.target_ip}"}
            ],
            "success_threshold": 0.5,
            "fallback_strategy": "try_alternative_services"
        }
        attack_plan["phases"].append(access_phase)
        
        # Phase 3: Exploitation
        exploit_phase = {
            "phase": "exploitation",
            "attacks": [
                {"type": "credential_reuse", "discovered_creds": True},
                {"type": "web_exploit", "tool": "sqlmap", "target": f"http://{self.target_ip}"},
                {"type": "privilege_escalation", "method": "sudo_abuse"}
            ],
            "success_threshold": 0.3,
            "fallback_strategy": "lateral_movement"
        }
        attack_plan["phases"].append(exploit_phase)
        
        self.logger.info(f"🔴 Generated adaptive attack plan with {len(attack_plan['phases'])} phases")
        return attack_plan
    
    async def _select_optimal_attack(self) -> Optional[Dict[str, Any]]:
        """Select optimal next attack based on current situation"""
        if not self.current_attack_plan:
            return None
        
        # Find current phase based on progress
        current_phase = None
        for phase in self.current_attack_plan["phases"]:
            phase_attacks = phase.get("attacks", [])
            executed_attacks = [a for a in phase_attacks if a.get("executed")]
            
            if len(executed_attacks) < len(phase_attacks):
                current_phase = phase
                break
        
        if not current_phase:
            return None  # All phases completed
        
        # Select next unexecuted attack from current phase
        for attack in current_phase["attacks"]:
            if not attack.get("executed"):
                return attack
        
        return None
    
    async def _execute_live_attack(self, attack: Dict[str, Any]) -> Dict[str, Any]:
        """Execute live attack against blue team"""
        attack_start_time = datetime.now()
        attack_result = {
            "attack_id": str(uuid.uuid4()),
            "timestamp": attack_start_time,
            "type": attack["type"],
            "tool": attack.get("tool", "custom"),
            "target": attack.get("target", self.target_ip),
            "success": False,
            "output": "",
            "detection_indicators": [],
            "response_observed": False
        }
        
        try:
            self.logger.info(f"🔴 Executing live attack: {attack['type']}")
            self.live_metrics.attacks_attempted += 1
            
            # Generate command based on attack type
            command = await self._generate_attack_command(attack)
            
            if command:
                # Execute attack in red team container
                result = subprocess.run([
                    "docker", "exec", self.combat_env.red_team_container,
                    "/bin/bash", "-c", command
                ], capture_output=True, text=True, timeout=60)
                
                attack_result["command"] = command
                attack_result["output"] = result.stdout
                attack_result["error"] = result.stderr
                attack_result["success"] = result.returncode == 0
                
                if attack_result["success"]:
                    self.live_metrics.attacks_successful += 1
                    self.logger.info(f"✅ Attack successful: {attack['type']}")
                else:
                    self.logger.warning(f"❌ Attack failed: {attack['type']}")
                
                # Check for defensive responses
                await asyncio.sleep(5)  # Wait for blue team response
                response_detected = await self._check_defensive_response()
                attack_result["response_observed"] = response_detected
                
                # Mark attack as executed
                attack["executed"] = True
                attack["result"] = attack_result
                
        except subprocess.TimeoutExpired:
            attack_result["error"] = "Attack timeout"
            self.logger.warning(f"⏰ Attack timeout: {attack['type']}")
        except Exception as e:
            attack_result["error"] = str(e)
            self.logger.error(f"❌ Attack error: {e}")
        
        attack_result["duration"] = (datetime.now() - attack_start_time).total_seconds()
        self.attack_history.append(attack_result)
        
        return attack_result
    
    async def _generate_attack_command(self, attack: Dict[str, Any]) -> str:
        """Generate attack command based on attack type"""
        attack_type = attack["type"]
        target = attack.get("target", self.target_ip)
        
        commands = {
            "port_scan": f"nmap -sS -T4 {target}",
            "service_enum": f"nmap -sV -p {attack.get('ports', '22,80,21')} {target}",
            "banner_grab": f"timeout 10 netcat -v {target} {attack.get('port', 22)}",
            "ssh_bruteforce": f"hydra -l root -P /tmp/passwords.txt ssh://{target}",
            "web_vuln_scan": f"nikto -h {target}",
            "directory_enum": f"dirb {target}",
            "credential_reuse": f"sshpass -p 'vulnerable123' ssh -o StrictHostKeyChecking=no root@{target} 'whoami'",
            "web_exploit": f"curl -s {target} | grep -i 'apache\\|nginx\\|server'",
            "privilege_escalation": f"sshpass -p 'vulnerable123' ssh -o StrictHostKeyChecking=no root@{target} 'sudo -l'"
        }
        
        return commands.get(attack_type, f"echo 'Unknown attack type: {attack_type}'")
    
    async def _check_defensive_response(self) -> bool:
        """Check if blue team responded to the attack"""
        try:
            # Check blue team container for defensive actions
            result = subprocess.run([
                "docker", "exec", self.combat_env.blue_team_container,
                "/bin/bash", "-c", "tail -5 /var/log/auth.log 2>/dev/null || echo 'No auth log'"
            ], capture_output=True, text=True, timeout=10)
            
            # Look for signs of defensive response
            output = result.stdout.lower()
            response_indicators = ["blocked", "banned", "fail2ban", "denied", "rejected"]
            
            for indicator in response_indicators:
                if indicator in output:
                    self.logger.info(f"🔵 Defensive response detected: {indicator}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Could not check defensive response: {e}")
            return False
    
    async def _learn_from_attack_result(self, attack_result: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from attack result to improve future attacks"""
        learning_insight = {
            "timestamp": datetime.now(),
            "attack_type": attack_result["type"],
            "success": attack_result["success"],
            "insights": [],
            "strategy_updates": []
        }
        
        # Analyze attack success/failure
        if attack_result["success"]:
            learning_insight["insights"].append(f"{attack_result['type']} successful against target")
            if attack_result.get("response_observed"):
                learning_insight["insights"].append("Blue team responded - increase stealth in future")
                learning_insight["strategy_updates"].append("increase_stealth_timing")
        else:
            learning_insight["insights"].append(f"{attack_result['type']} failed - analyze countermeasures")
            if "permission denied" in attack_result.get("error", "").lower():
                learning_insight["strategy_updates"].append("try_alternative_credentials")
            elif "connection refused" in attack_result.get("error", "").lower():
                learning_insight["strategy_updates"].append("service_may_be_blocked")
        
        # Update attack effectiveness scoring
        attack_type = attack_result["type"]
        if not hasattr(self, 'attack_effectiveness'):
            self.attack_effectiveness = {}
        
        if attack_type not in self.attack_effectiveness:
            self.attack_effectiveness[attack_type] = {"attempts": 0, "successes": 0}
        
        self.attack_effectiveness[attack_type]["attempts"] += 1
        if attack_result["success"]:
            self.attack_effectiveness[attack_type]["successes"] += 1
        
        self.live_metrics.learning_events += 1
        self.logger.info(f"🧠 Learning insight: {len(learning_insight['insights'])} new insights")
        
        return learning_insight
    
    async def _adapt_attack_strategy(self, failed_attack: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt attack strategy based on failures"""
        adaptation = {
            "timestamp": datetime.now(),
            "trigger": f"Failed attack: {failed_attack['type']}",
            "adaptations": [],
            "new_tactics": []
        }
        
        # Analyze failure reason and adapt
        error = failed_attack.get("error", "").lower()
        
        if "connection refused" in error:
            adaptation["adaptations"].append("Service blocked - try alternative ports")
            adaptation["new_tactics"].append("port_hopping")
        elif "permission denied" in error:
            adaptation["adaptations"].append("Authentication failed - try credential stuffing")
            adaptation["new_tactics"].append("credential_enumeration")
        elif "timeout" in error:
            adaptation["adaptations"].append("Network filtering detected - increase stealth")
            adaptation["new_tactics"].append("slow_scan_timing")
        
        # Update attack timing to avoid detection
        if failed_attack.get("response_observed"):
            adaptation["adaptations"].append("Blue team response detected - increase delay between attacks")
            adaptation["new_tactics"].append("extended_timing")
        
        self.live_metrics.adaptation_count += 1
        self.logger.info(f"🔄 Strategy adaptation: {len(adaptation['adaptations'])} changes")
        
        return adaptation

class LiveBlueTeamAgent(BlueTeamDefenderAgent):
    """Enhanced Blue Team agent for live combat scenarios"""
    
    def __init__(self, agent_id: str, combat_environment: LiveAdversarialEnvironment):
        super().__init__(agent_id)
        self.combat_env = combat_environment
        self.live_metrics = LiveCombatMetrics()
        self.defense_history = []
        self.monitoring_active = False
        self.alert_thresholds = {
            "failed_logins": 5,
            "port_scans": 10,
            "connection_attempts": 20
        }
        self.learning_enabled = True
    
    async def autonomous_live_defense_operation(self, duration_minutes: int = 30) -> Dict[str, Any]:
        """Execute autonomous live defense operation"""
        operation_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        self.logger.info(f"🔵 BLUE TEAM AGENT: Starting live defense operation")
        self.logger.info(f"⏱️ Duration: {duration_minutes} minutes")
        
        operation_results = {
            "operation_id": operation_id,
            "agent_id": self.agent_id,
            "start_time": start_time,
            "status": "active",
            "detections": [],
            "responses": [],
            "learning_data": [],
            "threat_intelligence": []
        }
        
        try:
            # Initialize defensive monitoring
            await self._initialize_live_monitoring()
            
            # Start continuous monitoring loop
            end_time = start_time + timedelta(minutes=duration_minutes)
            
            while datetime.now() < end_time:
                # Monitor for threats
                threats = await self._detect_live_threats()
                
                for threat in threats:
                    operation_results["detections"].append(threat)
                    
                    # Respond to threat
                    response = await self._respond_to_threat(threat)
                    operation_results["responses"].append(response)
                    
                    # Learn from threat patterns
                    if self.learning_enabled:
                        intelligence = await self._analyze_threat_intelligence(threat)
                        operation_results["threat_intelligence"].append(intelligence)
                
                # Update defensive posture
                await self._update_defensive_posture()
                
                # Monitor interval
                await asyncio.sleep(15)  # Check every 15 seconds
            
            operation_results["status"] = "completed"
            operation_results["end_time"] = datetime.now()
            operation_results["live_metrics"] = asdict(self.live_metrics)
            
            self.logger.info(f"🔵 BLUE TEAM AGENT: Defense operation completed")
            self.logger.info(f"📊 Detected: {len(operation_results['detections'])} threats")
            
        except Exception as e:
            self.logger.error(f"🔵 BLUE TEAM AGENT ERROR: {e}")
            operation_results["status"] = "failed"
            operation_results["error"] = str(e)
        finally:
            await self._stop_live_monitoring()
        
        return operation_results
    
    async def _initialize_live_monitoring(self):
        """Initialize live threat monitoring"""
        self.monitoring_active = True
        
        # Start monitoring services in blue team container
        monitoring_commands = [
            "service fail2ban start",
            "service rsyslog start",
            "iptables -I INPUT -j LOG --log-prefix 'ARCHANGEL-FW: '",
        ]
        
        for cmd in monitoring_commands:
            try:
                subprocess.run([
                    "docker", "exec", "-d", self.combat_env.blue_team_container,
                    "/bin/bash", "-c", cmd
                ], capture_output=True, timeout=10)
            except Exception as e:
                self.logger.warning(f"Monitoring setup warning: {e}")
        
        self.logger.info("🔵 Live monitoring initialized")
    
    async def _detect_live_threats(self) -> List[Dict[str, Any]]:
        """Detect live threats in real-time"""
        threats = []
        
        try:
            # Check authentication logs for brute force
            auth_threats = await self._check_authentication_threats()
            threats.extend(auth_threats)
            
            # Check network activity for port scans
            network_threats = await self._check_network_threats()
            threats.extend(network_threats)
            
            # Check system logs for suspicious activity
            system_threats = await self._check_system_threats()
            threats.extend(system_threats)
            
            # Update detection metrics
            self.live_metrics.defenses_triggered += len(threats)
            
        except Exception as e:
            self.logger.error(f"Threat detection error: {e}")
        
        return threats
    
    async def _check_authentication_threats(self) -> List[Dict[str, Any]]:
        """Check for authentication-based threats"""
        threats = []
        
        try:
            # Check auth logs for failed login attempts
            result = subprocess.run([
                "docker", "exec", self.combat_env.blue_team_container,
                "/bin/bash", "-c", 
                "tail -20 /var/log/auth.log 2>/dev/null | grep -i 'failed password' | tail -5"
            ], capture_output=True, text=True, timeout=10)
            
            if result.stdout:
                failed_attempts = result.stdout.strip().split('\n')
                if len(failed_attempts) >= self.alert_thresholds["failed_logins"]:
                    threat = {
                        "timestamp": datetime.now(),
                        "type": "brute_force_attack",
                        "severity": "high",
                        "source": "auth_logs",
                        "details": f"{len(failed_attempts)} failed login attempts detected",
                        "indicators": failed_attempts[-3:]  # Last 3 attempts
                    }
                    threats.append(threat)
                    self.logger.warning(f"🚨 Brute force attack detected: {len(failed_attempts)} attempts")
            
        except Exception as e:
            self.logger.warning(f"Auth threat check error: {e}")
        
        return threats
    
    async def _check_network_threats(self) -> List[Dict[str, Any]]:
        """Check for network-based threats"""
        threats = []
        
        try:
            # Check netstat for unusual connections
            result = subprocess.run([
                "docker", "exec", self.combat_env.blue_team_container,
                "/bin/bash", "-c",
                "netstat -tuln | grep -E '192.168.100.10|ESTABLISHED' | wc -l"
            ], capture_output=True, text=True, timeout=10)
            
            if result.stdout:
                connection_count = int(result.stdout.strip() or 0)
                if connection_count > self.alert_thresholds["connection_attempts"]:
                    threat = {
                        "timestamp": datetime.now(),
                        "type": "suspicious_network_activity",
                        "severity": "medium",
                        "source": "network_monitoring",
                        "details": f"{connection_count} active connections from red team",
                        "indicators": ["high_connection_count", "red_team_source"]
                    }
                    threats.append(threat)
                    self.logger.warning(f"🚨 Suspicious network activity: {connection_count} connections")
            
        except Exception as e:
            self.logger.warning(f"Network threat check error: {e}")
        
        return threats
    
    async def _check_system_threats(self) -> List[Dict[str, Any]]:
        """Check for system-level threats"""
        threats = []
        
        try:
            # Check for privilege escalation attempts
            result = subprocess.run([
                "docker", "exec", self.combat_env.blue_team_container,
                "/bin/bash", "-c",
                "tail -10 /var/log/auth.log 2>/dev/null | grep -i 'sudo\\|su ' | wc -l"
            ], capture_output=True, text=True, timeout=10)
            
            if result.stdout:
                sudo_attempts = int(result.stdout.strip() or 0)
                if sudo_attempts > 0:
                    threat = {
                        "timestamp": datetime.now(),
                        "type": "privilege_escalation_attempt", 
                        "severity": "high",
                        "source": "system_logs",
                        "details": f"{sudo_attempts} privilege escalation attempts",
                        "indicators": ["sudo_usage", "potential_lateral_movement"]
                    }
                    threats.append(threat)
                    self.logger.warning(f"🚨 Privilege escalation detected: {sudo_attempts} attempts")
            
        except Exception as e:
            self.logger.warning(f"System threat check error: {e}")
        
        return threats
    
    async def _respond_to_threat(self, threat: Dict[str, Any]) -> Dict[str, Any]:
        """Respond to detected threat"""
        response_start_time = datetime.now()
        response = {
            "timestamp": response_start_time,
            "threat_id": threat.get("timestamp", datetime.now()).isoformat(),
            "threat_type": threat["type"],
            "response_actions": [],
            "effectiveness": "unknown"
        }
        
        try:
            threat_type = threat["type"]
            
            # Determine response based on threat type
            if threat_type == "brute_force_attack":
                actions = await self._respond_to_brute_force()
            elif threat_type == "suspicious_network_activity":
                actions = await self._respond_to_network_threat()
            elif threat_type == "privilege_escalation_attempt":
                actions = await self._respond_to_escalation()
            else:
                actions = await self._generic_threat_response()
            
            response["response_actions"] = actions
            
            # Measure response time
            response_time = (datetime.now() - response_start_time).total_seconds()
            self.live_metrics.response_time_avg = (
                (self.live_metrics.response_time_avg + response_time) / 2
                if self.live_metrics.response_time_avg > 0 else response_time
            )
            
            self.logger.info(f"🔵 Threat response completed in {response_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Threat response error: {e}")
            response["error"] = str(e)
        
        self.defense_history.append(response)
        return response
    
    async def _respond_to_brute_force(self) -> List[str]:
        """Respond to brute force attack"""
        actions = []
        
        try:
            # Block red team IP using iptables
            block_cmd = "iptables -I INPUT -s 192.168.100.10 -j DROP"
            result = subprocess.run([
                "docker", "exec", self.combat_env.blue_team_container,
                "/bin/bash", "-c", block_cmd
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                actions.append("blocked_attacker_ip")
                self.logger.info("🔵 Blocked red team IP address")
            
            # Increase fail2ban sensitivity
            actions.append("increased_fail2ban_sensitivity")
            
            # Log security event
            actions.append("logged_security_incident")
            
        except Exception as e:
            self.logger.error(f"Brute force response error: {e}")
            actions.append(f"response_error: {e}")
        
        return actions
    
    async def _respond_to_network_threat(self) -> List[str]:
        """Respond to network threat"""
        actions = []
        
        try:
            # Monitor network connections more closely
            actions.append("enhanced_network_monitoring")
            
            # Rate limit connections
            rate_limit_cmd = "iptables -I INPUT -p tcp --dport 22 -m limit --limit 1/min -j ACCEPT"
            subprocess.run([
                "docker", "exec", self.combat_env.blue_team_container,
                "/bin/bash", "-c", rate_limit_cmd
            ], capture_output=True, timeout=10)
            actions.append("applied_rate_limiting")
            
        except Exception as e:
            actions.append(f"network_response_error: {e}")
        
        return actions
    
    async def _respond_to_escalation(self) -> List[str]:
        """Respond to privilege escalation attempt"""
        actions = []
        
        try:
            # Restrict sudo access
            actions.append("restricted_sudo_access")
            
            # Monitor privileged operations
            actions.append("enhanced_privilege_monitoring")
            
            # Alert security team (simulated)
            actions.append("alerted_security_team")
            
        except Exception as e:
            actions.append(f"escalation_response_error: {e}")
        
        return actions
    
    async def _generic_threat_response(self) -> List[str]:
        """Generic threat response"""
        return [
            "increased_logging_verbosity",
            "enhanced_monitoring",
            "documented_incident"
        ]
    
    async def _analyze_threat_intelligence(self, threat: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze threat for intelligence gathering"""
        intelligence = {
            "timestamp": datetime.now(),
            "threat_type": threat["type"],
            "attack_patterns": [],
            "iocs": [],  # Indicators of Compromise
            "attribution": "unknown",
            "recommendations": []
        }
        
        # Extract attack patterns
        if threat["type"] == "brute_force_attack":
            intelligence["attack_patterns"] = [
                "credential_stuffing",
                "password_spraying",
                "dictionary_attack"
            ]
            intelligence["iocs"] = ["multiple_failed_logins", "source_192.168.100.10"]
            intelligence["attribution"] = "red_team_agent"
            intelligence["recommendations"] = [
                "implement_account_lockout",
                "deploy_mfa",
                "monitor_authentication_logs"
            ]
        
        self.live_metrics.learning_events += 1
        return intelligence
    
    async def _update_defensive_posture(self):
        """Update defensive posture based on current threats"""
        try:
            # Adaptive threshold adjustment
            recent_threats = [d for d in self.defense_history if 
                            (datetime.now() - d["timestamp"]).total_seconds() < 300]  # Last 5 minutes
            
            if len(recent_threats) > 5:  # High threat activity
                # Lower detection thresholds
                self.alert_thresholds["failed_logins"] = max(3, self.alert_thresholds["failed_logins"] - 1)
                self.alert_thresholds["connection_attempts"] = max(10, self.alert_thresholds["connection_attempts"] - 5)
                self.logger.info("🔵 Increased defensive sensitivity due to high threat activity")
            
        except Exception as e:
            self.logger.warning(f"Defensive posture update error: {e}")
    
    async def _stop_live_monitoring(self):
        """Stop live monitoring"""
        self.monitoring_active = False
        self.logger.info("🔵 Live monitoring stopped")

class LiveCombatOrchestrator:
    """Orchestrates live combat between red and blue team agents"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.combat_env = LiveAdversarialEnvironment()
        self.red_team_agent = None
        self.blue_team_agent = None
        self.exercise_running = False
        self.live_learning_db = []
    
    async def setup_live_combat_simulation(self) -> bool:
        """Setup live combat simulation environment"""
        try:
            self.logger.info("🛡️ Setting up live adversarial combat simulation...")
            
            # Setup combat environment
            if not await self.combat_env.setup_combat_environment():
                return False
            
            # Create AI agents
            self.red_team_agent = LiveRedTeamAgent("red-agent-001", self.combat_env)
            self.blue_team_agent = LiveBlueTeamAgent("blue-agent-001", self.combat_env)
            
            self.logger.info("✅ Live combat simulation ready")
            return True
            
        except Exception as e:
            self.logger.error(f"Combat simulation setup failed: {e}")
            return False
    
    async def execute_live_adversarial_exercise(self, scenario_name: str = None, duration_minutes: int = 20) -> Dict[str, Any]:
        """Execute live adversarial exercise with learning"""
        exercise_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        self.logger.info(f"🚀 Starting live adversarial exercise: {exercise_id}")
        
        exercise_results = {
            "exercise_id": exercise_id,
            "start_time": start_time,
            "status": "active",
            "red_team_results": {},
            "blue_team_results": {},
            "learning_data": [],
            "combat_metrics": {}
        }
        
        try:
            self.exercise_running = True
            
            # Select scenario
            scenario = next((s for s in self.combat_env.attack_scenarios 
                           if s.name == scenario_name), None) or self.combat_env.attack_scenarios[0]
            
            # Launch both teams concurrently
            red_team_task = asyncio.create_task(
                self.red_team_agent.autonomous_live_attack_operation(scenario, duration_minutes)
            )
            
            blue_team_task = asyncio.create_task(
                self.blue_team_agent.autonomous_live_defense_operation(duration_minutes)
            )
            
            # Wait for both teams to complete
            red_results, blue_results = await asyncio.gather(
                red_team_task, blue_team_task, return_exceptions=True
            )
            
            exercise_results["red_team_results"] = red_results if not isinstance(red_results, Exception) else {"error": str(red_results)}
            exercise_results["blue_team_results"] = blue_results if not isinstance(blue_results, Exception) else {"error": str(blue_results)}
            
            # Analyze combat interaction
            combat_analysis = await self._analyze_combat_interaction(red_results, blue_results)
            exercise_results["combat_analysis"] = combat_analysis
            
            # Generate learning data
            learning_data = await self._extract_learning_data(red_results, blue_results, combat_analysis)
            exercise_results["learning_data"] = learning_data
            self.live_learning_db.extend(learning_data)
            
            exercise_results["status"] = "completed"
            exercise_results["end_time"] = datetime.now()
            exercise_results["duration"] = (exercise_results["end_time"] - start_time).total_seconds()
            
            self.logger.info(f"✅ Live adversarial exercise completed: {exercise_id}")
            
        except Exception as e:
            self.logger.error(f"Live exercise error: {e}")
            exercise_results["status"] = "failed"
            exercise_results["error"] = str(e)
        finally:
            self.exercise_running = False
        
        return exercise_results
    
    async def _analyze_combat_interaction(self, red_results: Dict, blue_results: Dict) -> Dict[str, Any]:
        """Analyze the interaction between red and blue teams"""
        analysis = {
            "timestamp": datetime.now(),
            "red_team_effectiveness": 0.0,
            "blue_team_effectiveness": 0.0,
            "interaction_patterns": [],
            "learning_opportunities": []
        }
        
        try:
            # Calculate red team effectiveness
            if isinstance(red_results, dict) and "live_metrics" in red_results:
                red_metrics = red_results["live_metrics"]
                attacks_attempted = red_metrics.get("attacks_attempted", 0)
                attacks_successful = red_metrics.get("attacks_successful", 0)
                analysis["red_team_effectiveness"] = (
                    attacks_successful / attacks_attempted if attacks_attempted > 0 else 0.0
                )
            
            # Calculate blue team effectiveness
            if isinstance(blue_results, dict) and "detections" in blue_results:
                detections = len(blue_results.get("detections", []))
                responses = len(blue_results.get("responses", []))
                analysis["blue_team_effectiveness"] = (
                    responses / detections if detections > 0 else 0.0
                )
            
            # Identify interaction patterns
            if analysis["red_team_effectiveness"] > 0.7:
                analysis["interaction_patterns"].append("red_team_dominant")
            elif analysis["blue_team_effectiveness"] > 0.7:
                analysis["interaction_patterns"].append("blue_team_dominant")
            else:
                analysis["interaction_patterns"].append("balanced_engagement")
            
            # Generate learning opportunities
            analysis["learning_opportunities"] = [
                "red_team_stealth_techniques",
                "blue_team_detection_patterns",
                "attack_defense_timing_analysis",
                "adaptive_strategy_effectiveness"
            ]
            
        except Exception as e:
            self.logger.error(f"Combat analysis error: {e}")
            analysis["error"] = str(e)
        
        return analysis
    
    async def _extract_learning_data(self, red_results: Dict, blue_results: Dict, combat_analysis: Dict) -> List[Dict[str, Any]]:
        """Extract learning data from live combat exercise"""
        learning_data = []
        
        try:
            # Red team learning data
            if isinstance(red_results, dict) and "learning_data" in red_results:
                for learning_event in red_results["learning_data"]:
                    learning_data.append({
                        "timestamp": learning_event.get("timestamp", datetime.now()),
                        "source": "red_team",
                        "type": "attack_learning",
                        "data": learning_event,
                        "combat_context": combat_analysis
                    })
            
            # Blue team learning data
            if isinstance(blue_results, dict) and "threat_intelligence" in blue_results:
                for intel_event in blue_results["threat_intelligence"]:
                    learning_data.append({
                        "timestamp": intel_event.get("timestamp", datetime.now()),
                        "source": "blue_team",
                        "type": "defense_intelligence",
                        "data": intel_event,
                        "combat_context": combat_analysis
                    })
            
            # Cross-team learning insights
            learning_data.append({
                "timestamp": datetime.now(),
                "source": "orchestrator",
                "type": "interaction_analysis",
                "data": {
                    "red_effectiveness": combat_analysis.get("red_team_effectiveness", 0),
                    "blue_effectiveness": combat_analysis.get("blue_team_effectiveness", 0),
                    "dominant_strategy": combat_analysis.get("interaction_patterns", []),
                    "improvement_areas": [
                        "red_team_stealth" if combat_analysis.get("blue_team_effectiveness", 0) > 0.5 else None,
                        "blue_team_detection" if combat_analysis.get("red_team_effectiveness", 0) > 0.5 else None
                    ]
                },
                "combat_context": combat_analysis
            })
            
        except Exception as e:
            self.logger.error(f"Learning data extraction error: {e}")
        
        return learning_data
    
    async def cleanup_combat_simulation(self):
        """Cleanup combat simulation"""
        try:
            await self.combat_env.cleanup_combat_environment()
            self.logger.info("✅ Combat simulation cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

async def main():
    """Demonstrate live adversarial combat with AI agents"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🛡️⚔️ Archangel Live Adversarial Combat - AI Agents")
    print("=" * 60)
    
    orchestrator = LiveCombatOrchestrator()
    
    try:
        # Setup live combat simulation
        print("\n🔧 Setting up live red team vs blue team combat...")
        if await orchestrator.setup_live_combat_simulation():
            print("✅ Live combat simulation ready")
            
            # Execute live adversarial exercise
            print("\n⚔️ Starting live adversarial exercise...")
            print("🔴 RED TEAM AI: Autonomous attack operations")
            print("🔵 BLUE TEAM AI: Autonomous defense operations")
            print("🧠 LEARNING: Real-time adaptation and intelligence gathering")
            
            results = await orchestrator.execute_live_adversarial_exercise(
                scenario_name="Network Reconnaissance",
                duration_minutes=15
            )
            
            print("\n📋 Live Exercise Results:")
            print(f"Status: {results.get('status')}")
            print(f"Duration: {results.get('duration', 0):.1f} seconds")
            
            if "red_team_results" in results:
                red_metrics = results["red_team_results"].get("live_metrics", {})
                print(f"🔴 Red Team: {red_metrics.get('attacks_attempted', 0)} attacks, {red_metrics.get('attacks_successful', 0)} successful")
            
            if "blue_team_results" in results:
                blue_detections = len(results["blue_team_results"].get("detections", []))
                blue_responses = len(results["blue_team_results"].get("responses", []))
                print(f"🔵 Blue Team: {blue_detections} threats detected, {blue_responses} responses executed")
            
            learning_count = len(results.get("learning_data", []))
            print(f"🧠 Learning Events: {learning_count} insights captured")
            
        else:
            print("❌ Failed to setup live combat simulation")
            
    except KeyboardInterrupt:
        print("\n🛑 Live exercise interrupted by user")
    except Exception as e:
        print(f"❌ Live exercise failed: {e}")
    finally:
        print("\n🧹 Cleaning up...")
        await orchestrator.cleanup_combat_simulation()
        print("✅ Cleanup completed")

if __name__ == "__main__":
    asyncio.run(main())