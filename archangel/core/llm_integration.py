"""
Local LLM Integration for Archangel AI vs AI System
Supports offline LLM models via llama-cpp-python for autonomous operation
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Fix HuggingFace tokenizers fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LocalLLMManager:
    """Manages local LLM models for autonomous AI agents"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.llm = None
        self.model_loaded = False
        self.fallback_mode = True
        
        # Try to import transformers for HuggingFace models
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            self.transformers_available = True
            if model_path:
                self._load_hf_model()
        except ImportError:
            print("⚠️ transformers not available, using fallback intelligent responses")
            self.transformers_available = False
    
    def _load_hf_model(self):
        """Load HuggingFace model"""
        try:
            from transformers import pipeline
            print(f"🧠 Loading HuggingFace model: {self.model_path}")
            
            # Use text generation pipeline for cybersecurity models
            self.llm = pipeline(
                "text-generation",
                model=self.model_path,
                tokenizer=self.model_path,
                max_length=512,
                truncation=True,
                do_sample=True,
                temperature=0.7,
                device="cpu"  # Use CPU for compatibility
            )
            self.model_loaded = True
            self.fallback_mode = False
            print(f"✅ HuggingFace model loaded successfully")
            
        except Exception as e:
            print(f"⚠️ Failed to load HuggingFace model: {e}")
            print(f"🔄 Falling back to intelligent rule-based responses")
            self.fallback_mode = True
    
    async def generate_red_team_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate red team decision using local LLM or intelligent fallback"""
        
        if not self.fallback_mode and self.llm:
            return await self._llm_red_team_decision(context)
        else:
            return await self._fallback_red_team_decision(context)
    
    async def generate_blue_team_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate blue team decision using local LLM or intelligent fallback"""
        
        if not self.fallback_mode and self.llm:
            return await self._llm_blue_team_decision(context)
        else:
            return await self._fallback_blue_team_decision(context)
    
    async def _llm_red_team_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate red team decision using HuggingFace model"""
        try:
            prompt = self._build_red_team_prompt(context)
            
            # Generate response using HuggingFace pipeline
            response = self.llm(prompt, max_length=len(prompt.split()) + 100, truncation=True, num_return_sequences=1)
            
            # Validate response format
            if not response or len(response) == 0:
                raise ValueError("LLM returned empty response")
            
            if 'generated_text' not in response[0]:
                raise ValueError("LLM response missing 'generated_text' field")
            
            # Parse LLM response
            full_text = response[0]['generated_text']
            decision_text = full_text[len(prompt):].strip() if len(full_text) > len(prompt) else full_text.strip()
            
            return self._parse_red_team_response(decision_text, context)
            
        except Exception as e:
            print(f"⚠️ LLM red team decision failed: {e}")
            print(f"🔄 Falling back to intelligent rule-based decision")
            return await self._fallback_red_team_decision(context)
    
    async def _llm_blue_team_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate blue team decision using HuggingFace model"""
        try:
            prompt = self._build_blue_team_prompt(context)
            
            # Generate response using HuggingFace pipeline
            response = self.llm(prompt, max_length=len(prompt.split()) + 100, truncation=True, num_return_sequences=1, temperature=0.3)
            
            # Validate response format
            if not response or len(response) == 0:
                raise ValueError("LLM returned empty response")
            
            if 'generated_text' not in response[0]:
                raise ValueError("LLM response missing 'generated_text' field")
            
            # Parse LLM response
            full_text = response[0]['generated_text']
            decision_text = full_text[len(prompt):].strip() if len(full_text) > len(prompt) else full_text.strip()
            
            return self._parse_blue_team_response(decision_text, context)
            
        except Exception as e:
            print(f"⚠️ LLM blue team decision failed: {e}")
            print(f"🔄 Falling back to intelligent rule-based decision")
            return await self._fallback_blue_team_decision(context)
    
    def _build_red_team_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for red team LLM"""
        discovered_hosts = context.get('discovered_hosts', [])
        previous_actions = context.get('previous_actions', [])
        target_info = context.get('target_info', {})
        
        prompt = f"""You are an autonomous red team penetration tester. Your goal is to discover and compromise targets.

Current situation:
- Discovered hosts: {discovered_hosts}
- Previous actions: {len(previous_actions)} completed
- Target network: 172.18.0.0/16

Available tools: nmap, curl, netcat, python3

Choose your next action:
1. SCAN - Network discovery and port scanning
2. ENUM - Service enumeration and vulnerability assessment  
3. EXPLOIT - Attempt to exploit discovered vulnerabilities
4. PERSIST - Establish persistence on compromised systems
5. LATERAL - Lateral movement to new targets

<decision>
Action:"""
        
        return prompt
    
    def _build_blue_team_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for blue team LLM"""
        threats = context.get('detected_threats', [])
        connections = context.get('active_connections', [])
        blocked_ips = context.get('blocked_ips', [])
        
        prompt = f"""You are an autonomous blue team security analyst. Your goal is to detect and respond to threats.

Current situation:
- Detected threats: {len(threats)}
- Active connections: {len(connections)}
- Blocked IPs: {len(blocked_ips)}

Available tools: iptables, tcpdump, process monitoring, log analysis

Choose your response:
1. MONITOR - Increase monitoring and log analysis
2. BLOCK - Block suspicious IP addresses
3. ISOLATE - Isolate compromised systems
4. PATCH - Apply security patches and hardening
5. INVESTIGATE - Deep dive threat analysis

<decision>
Action:"""
        
        return prompt
    
    async def _fallback_red_team_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligent fallback for red team decisions"""
        discovered_hosts = context.get('discovered_hosts', [])
        previous_actions = context.get('previous_actions', [])
        current_phase = context.get('current_phase', 'reconnaissance')
        
        # Intelligent decision tree based on current state
        if len(discovered_hosts) == 0:
            # No hosts discovered yet - start with network scan
            return {
                'action_type': 'network_scan',
                'target': '172.18.0.0/24',
                'tool': 'nmap',
                'parameters': {'scan_type': 'discovery', 'timing': 'normal'},
                'reasoning': 'No hosts discovered - performing network reconnaissance',
                'confidence': 0.9,
                'timestamp': datetime.now().isoformat()
            }
        
        elif len(previous_actions) < 3:
            # Early phase - enumerate discovered hosts
            target_host = discovered_hosts[len(previous_actions) % len(discovered_hosts)]
            return {
                'action_type': 'port_scan',
                'target': target_host,
                'tool': 'nmap',
                'parameters': {'scan_type': 'service_detection', 'ports': 'top-1000'},
                'reasoning': f'Enumerating services on discovered host {target_host}',
                'confidence': 0.8,
                'timestamp': datetime.now().isoformat()
            }
        
        else:
            # Later phase - attempt exploitation
            target_host = discovered_hosts[0]  # Focus on first discovered host
            return {
                'action_type': 'vulnerability_scan',
                'target': target_host,
                'tool': 'nmap',
                'parameters': {'scan_type': 'vuln_scripts', 'scripts': 'vuln'},
                'reasoning': f'Scanning for vulnerabilities on {target_host}',
                'confidence': 0.7,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _fallback_blue_team_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligent fallback for blue team decisions"""
        threats = context.get('detected_threats', [])
        connections = context.get('active_connections', [])
        suspicious_activity = context.get('suspicious_activity', False)
        
        # Intelligent defensive decision tree
        if len(threats) > 0:
            # Active threats detected - take defensive action
            latest_threat = threats[-1]
            threat_source = latest_threat.get('source', 'unknown')
            
            return {
                'action_type': 'block_threat',
                'target': threat_source,
                'tool': 'iptables',
                'parameters': {'action': 'DROP', 'protocol': 'all'},
                'reasoning': f'Blocking threat source {threat_source}',
                'confidence': 0.9,
                'timestamp': datetime.now().isoformat()
            }
        
        elif suspicious_activity:
            # Suspicious activity - increase monitoring
            return {
                'action_type': 'increase_monitoring',
                'target': 'all_interfaces',
                'tool': 'tcpdump',
                'parameters': {'duration': 300, 'filter': 'suspicious_patterns'},
                'reasoning': 'Increasing monitoring due to suspicious activity',
                'confidence': 0.7,
                'timestamp': datetime.now().isoformat()
            }
        
        else:
            # Normal operation - baseline monitoring
            return {
                'action_type': 'baseline_monitor',
                'target': 'system',
                'tool': 'ps_netstat',
                'parameters': {'interval': 30, 'log_level': 'info'},
                'reasoning': 'Performing baseline security monitoring',
                'confidence': 0.6,
                'timestamp': datetime.now().isoformat()
            }
    
    def _parse_red_team_response(self, response_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response for red team actions"""
        # Simple parsing logic - in production would be more sophisticated
        action_type = 'network_scan'  # Default
        
        if 'SCAN' in response_text.upper():
            action_type = 'network_scan'
        elif 'ENUM' in response_text.upper():
            action_type = 'service_enumeration' 
        elif 'EXPLOIT' in response_text.upper():
            action_type = 'exploit_attempt'
        elif 'PERSIST' in response_text.upper():
            action_type = 'persistence'
        elif 'LATERAL' in response_text.upper():
            action_type = 'lateral_movement'
        
        # Get target - use discovered hosts if available, otherwise default
        discovered_hosts = context.get('discovered_hosts', [])
        target = discovered_hosts[0] if discovered_hosts else '172.18.0.0/24'
        
        return {
            'action_type': action_type,
            'target': target,
            'reasoning': response_text,
            'confidence': 0.8,
            'llm_generated': True,
            'timestamp': datetime.now().isoformat()
        }
    
    def _parse_blue_team_response(self, response_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response for blue team actions"""
        action_type = 'baseline_monitor'  # Default
        
        if 'MONITOR' in response_text.upper():
            action_type = 'increase_monitoring'
        elif 'BLOCK' in response_text.upper():
            action_type = 'block_threat'
        elif 'ISOLATE' in response_text.upper():
            action_type = 'isolate_system'
        elif 'PATCH' in response_text.upper():
            action_type = 'apply_patches'
        elif 'INVESTIGATE' in response_text.upper():
            action_type = 'threat_investigation'
        
        return {
            'action_type': action_type,
            'target': 'system',
            'reasoning': response_text,
            'confidence': 0.8,
            'llm_generated': True,
            'timestamp': datetime.now().isoformat()
        }

# Singleton instance for global access
_local_llm_manager = None

def get_llm_manager(model_path: Optional[str] = None) -> LocalLLMManager:
    """Get singleton LLM manager instance"""
    global _local_llm_manager
    if _local_llm_manager is None:
        _local_llm_manager = LocalLLMManager(model_path)
    return _local_llm_manager