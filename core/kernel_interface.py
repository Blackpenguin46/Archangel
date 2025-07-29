"""
Archangel Kernel Interface
Python interface to communicate with the Archangel kernel module
"""

import os
import time
import struct
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import mmap
import logging

logger = logging.getLogger(__name__)

class ArchangelDecision(Enum):
    ALLOW = 0
    DENY = 1
    MONITOR = 2
    DEFER_TO_USERSPACE = 3
    UNKNOWN = 4

class ArchangelConfidence(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    VERY_HIGH = 3

class ArchangelMsgType(Enum):
    PING = 1
    PONG = 2
    ANALYSIS_REQUEST = 3
    ANALYSIS_RESPONSE = 4
    RULE_UPDATE = 5
    EVENT_NOTIFICATION = 6
    STATUS_REQUEST = 7
    STATUS_RESPONSE = 8
    SHUTDOWN = 9

@dataclass
class SecurityContext:
    """Security context from kernel"""
    pid: int
    uid: int
    syscall_nr: int
    timestamp: int
    flags: int
    comm: str
    data: bytes = b''

@dataclass
class SecurityRule:
    """Security rule for kernel"""
    rule_id: int
    priority: int
    condition_mask: int
    condition_values: int
    action: ArchangelDecision
    confidence: ArchangelConfidence
    description: str

@dataclass
class KernelStats:
    """Statistics from kernel module"""
    total_decisions: int = 0
    allow_decisions: int = 0
    deny_decisions: int = 0
    monitor_decisions: int = 0
    deferred_decisions: int = 0
    rule_matches: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    userspace_requests: int = 0
    userspace_responses: int = 0
    avg_decision_time_ns: int = 0
    max_decision_time_ns: int = 0
    uptime_seconds: int = 0

class ArchangelKernelInterface:
    """
    Interface to Archangel kernel module
    
    This class handles communication between the userspace AI system
    and the kernel module, providing high-speed message passing and
    shared memory communication.
    """
    
    def __init__(self):
        self.proc_root = "/proc/archangel"
        self.comm_device = "/proc/archangel_comm"
        self.module_loaded = False
        self.shared_memory = None
        self.comm_fd = None
        
        # Message handling
        self.message_handlers = {}
        self.message_sequence = 0
        
    def is_module_loaded(self) -> bool:
        """Check if Archangel kernel module is loaded"""
        try:
            return os.path.exists(self.proc_root) and os.path.exists(f"{self.proc_root}/status")
        except:
            return False
    
    def get_module_status(self) -> Dict[str, Any]:
        """Get kernel module status"""
        if not self.is_module_loaded():
            return {"status": "not_loaded", "error": "Module not loaded"}
        
        try:
            with open(f"{self.proc_root}/status", 'r') as f:
                status_lines = f.readlines()
            
            status = {}
            for line in status_lines:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    status[key.strip().lower().replace(' ', '_')] = value.strip()
            
            return {"status": "loaded", "details": status}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_kernel_stats(self) -> KernelStats:
        """Get kernel module statistics"""
        if not self.is_module_loaded():
            return KernelStats()
        
        try:
            with open(f"{self.proc_root}/stats", 'r') as f:
                stats_lines = f.readlines()
            
            stats_dict = {}
            for line in stats_lines:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    try:
                        stats_dict[key] = int(value.strip().split()[0])  # Take first number
                    except:
                        pass
            
            return KernelStats(
                total_decisions=stats_dict.get('total_decisions', 0),
                allow_decisions=stats_dict.get('allow', 0),
                deny_decisions=stats_dict.get('deny', 0),
                monitor_decisions=stats_dict.get('monitor', 0),
                deferred_decisions=stats_dict.get('deferred', 0),
                rule_matches=stats_dict.get('rule_matches', 0),
                cache_hits=stats_dict.get('cache_hits', 0),
                cache_misses=stats_dict.get('cache_misses', 0),
                userspace_requests=stats_dict.get('userspace_requests', 0),
                userspace_responses=stats_dict.get('userspace_responses', 0),
                avg_decision_time_ns=stats_dict.get('avg_decision_time', 0),
                max_decision_time_ns=stats_dict.get('max_decision_time', 0),
                uptime_seconds=stats_dict.get('uptime', 0)
            )
        except Exception as e:
            logger.error(f"Failed to get kernel stats: {e}")
            return KernelStats()
    
    def get_security_rules(self) -> List[Dict[str, Any]]:
        """Get current security rules from kernel"""
        if not self.is_module_loaded():
            return []
        
        try:
            with open(f"{self.proc_root}/rules", 'r') as f:
                lines = f.readlines()
            
            rules = []
            for line in lines[1:]:  # Skip header
                parts = line.strip().split('\t')
                if len(parts) >= 6:
                    rules.append({
                        'id': int(parts[0]),
                        'priority': int(parts[1]),
                        'action': int(parts[2]),
                        'confidence': int(parts[3]),
                        'matches': int(parts[4]),
                        'description': parts[5]
                    })
            
            return rules
        except Exception as e:
            logger.error(f"Failed to get security rules: {e}")
            return []
    
    def add_security_rule(self, rule: SecurityRule) -> bool:
        """Add a security rule to the kernel"""
        if not self.is_module_loaded():
            return False
        
        try:
            cmd = f"add_rule {rule.rule_id} {rule.priority} {rule.action.value} {rule.description}"
            with open(f"{self.proc_root}/control", 'w') as f:
                f.write(cmd)
            
            logger.info(f"Added security rule {rule.rule_id} to kernel")
            return True
        except Exception as e:
            logger.error(f"Failed to add security rule: {e}")
            return False
    
    def remove_security_rule(self, rule_id: int) -> bool:
        """Remove a security rule from the kernel"""
        if not self.is_module_loaded():
            return False
        
        try:
            cmd = f"del_rule {rule_id}"
            with open(f"{self.proc_root}/control", 'w') as f:
                f.write(cmd)
            
            logger.info(f"Removed security rule {rule_id} from kernel")
            return True
        except Exception as e:
            logger.error(f"Failed to remove security rule: {e}")
            return False
    
    def clear_security_rules(self) -> bool:
        """Clear all security rules from kernel"""
        if not self.is_module_loaded():
            return False
        
        try:
            cmd = "clear_rules"
            with open(f"{self.proc_root}/control", 'w') as f:
                f.write(cmd)
            
            logger.info("Cleared all security rules from kernel")
            return True
        except Exception as e:
            logger.error(f"Failed to clear security rules: {e}")
            return False
    
    def init_communication(self) -> bool:
        """Initialize communication with kernel module"""
        if not self.is_module_loaded():
            logger.error("Kernel module not loaded")
            return False
        
        try:
            # For now, we'll use proc filesystem communication
            # In a full implementation, this would open shared memory
            logger.info("Communication with kernel module initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize communication: {e}")
            return False
    
    def cleanup_communication(self):
        """Cleanup communication with kernel module"""
        if self.shared_memory:
            self.shared_memory.close()
            self.shared_memory = None
        
        if self.comm_fd:
            os.close(self.comm_fd)
            self.comm_fd = None
        
        logger.info("Communication with kernel module cleaned up")
    
    async def process_kernel_messages(self, ai_system):
        """
        Process messages from kernel module
        
        This runs in a background task and handles analysis requests
        from the kernel, passing them to the AI system for decisions.
        """
        logger.info("Starting kernel message processing")
        
        while True:
            try:
                # Check for analysis requests from kernel
                # In a full implementation, this would read from shared memory queues
                
                # For now, simulate message processing
                await asyncio.sleep(1)
                
                # Example: Create a simulated analysis request
                if hasattr(ai_system, 'handle_kernel_analysis_request'):
                    # Simulate a security context from kernel
                    context = SecurityContext(
                        pid=12345,
                        uid=1000,
                        syscall_nr=59,  # execve
                        timestamp=time.time_ns(),
                        flags=0x0001,
                        comm="suspicious_app"
                    )
                    
                    # Let AI system handle the analysis
                    decision = await ai_system.handle_kernel_analysis_request(context)
                    
                    # Send response back to kernel (would use shared memory)
                    logger.debug(f"AI decision for PID {context.pid}: {decision}")
                
            except Exception as e:
                logger.error(f"Error processing kernel messages: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    def register_message_handler(self, msg_type: ArchangelMsgType, handler):
        """Register a handler for specific message types"""
        self.message_handlers[msg_type] = handler
    
    def send_message_to_kernel(self, msg_type: ArchangelMsgType, data: bytes = b'') -> bool:
        """Send message to kernel module"""
        try:
            # In full implementation, this would use shared memory queues
            logger.debug(f"Sending message {msg_type.name} to kernel (simulated)")
            return True
        except Exception as e:
            logger.error(f"Failed to send message to kernel: {e}")
            return False

class MockKernelInterface(ArchangelKernelInterface):
    """
    Mock kernel interface for testing without actual kernel module
    
    This simulates kernel module behavior for development and testing.
    """
    
    def __init__(self):
        super().__init__()
        self.mock_stats = KernelStats()
        self.mock_rules = []
        self.mock_loaded = True
        
    def is_module_loaded(self) -> bool:
        return self.mock_loaded
    
    def get_module_status(self) -> Dict[str, Any]:
        return {
            "status": "loaded_mock",
            "details": {
                "archangel_ai_security_expert": "v0.1.0",
                "status": "Active (Mock)",
                "mode": "0x00000001",
                "rules": f"{len(self.mock_rules)}/1024",
                "shared_memory": "1048576 bytes"
            }
        }
    
    def get_kernel_stats(self) -> KernelStats:
        # Simulate some statistics
        self.mock_stats.total_decisions += 1
        self.mock_stats.allow_decisions += 1
        self.mock_stats.uptime_seconds = int(time.time()) - 1640995200  # Mock start time
        return self.mock_stats
    
    def get_security_rules(self) -> List[Dict[str, Any]]:
        return [
            {
                'id': rule.rule_id,
                'priority': rule.priority,
                'action': rule.action.value,
                'confidence': rule.confidence.value,
                'matches': 0,
                'description': rule.description
            }
            for rule in self.mock_rules
        ]
    
    def add_security_rule(self, rule: SecurityRule) -> bool:
        self.mock_rules.append(rule)
        logger.info(f"Added mock security rule {rule.rule_id}")
        return True
    
    def remove_security_rule(self, rule_id: int) -> bool:
        self.mock_rules = [r for r in self.mock_rules if r.rule_id != rule_id]
        logger.info(f"Removed mock security rule {rule_id}")
        return True
    
    def clear_security_rules(self) -> bool:
        self.mock_rules.clear()
        logger.info("Cleared all mock security rules")
        return True
    
    def init_communication(self) -> bool:
        logger.info("Mock communication initialized")
        return True

def create_kernel_interface() -> ArchangelKernelInterface:
    """
    Create appropriate kernel interface
    
    Returns real interface if kernel module is loaded,
    otherwise returns mock interface for testing.
    """
    real_interface = ArchangelKernelInterface()
    
    if real_interface.is_module_loaded():
        logger.info("Using real kernel interface")
        return real_interface
    else:
        logger.info("Kernel module not found, using mock interface")
        return MockKernelInterface()