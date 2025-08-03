"""
Archangel AI vs AI Cybersecurity Orchestrator (Simplified)
Central system for coordinating red and blue team AI agents
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from .logging_system import ArchangelLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, timedelta)):
        return obj.isoformat()
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    if hasattr(obj, '_asdict'):
        return obj._asdict()
    return str(obj)

def safe_json_dumps(obj, **kwargs):
    """Safe JSON dumps that handles datetime and other objects"""
    return json.dumps(obj, default=json_serial, **kwargs)

class ArchangelOrchestrator:
    """Simplified orchestrator for the modular Archangel system"""
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.logger = ArchangelLogger(self.session_id)
        self.red_agents = []
        self.blue_agents = []
        self.simulation_active = False
        self.tick_count = 0
        self.metrics = {
            'red_team_score': 0,
            'blue_team_score': 0,
            'total_actions': 0,
            'start_time': None,
            'end_time': None
        }
        
    async def initialize(self):
        """Initialize the orchestrator"""
        self.metrics['start_time'] = datetime.now().isoformat()
        await self.logger.log_system_event({
            'event_type': 'orchestrator_initialization',
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat()
        })
        
    async def add_red_agent(self, agent):
        """Add a red team agent"""
        self.red_agents.append(agent)
        
    async def add_blue_agent(self, agent):
        """Add a blue team agent"""
        self.blue_agents.append(agent)
        
    async def run_simulation_tick(self):
        """Run one simulation tick"""
        self.tick_count += 1
        
        # Execute red team operations
        red_results = []
        for agent in self.red_agents:
            try:
                result = await agent.autonomous_operation_cycle()
                red_results.append(result)
            except Exception as e:
                logger.error(f"Red team agent error: {e}")
                red_results.append({'error': str(e)})
            
        # Execute blue team operations
        blue_results = []
        for agent in self.blue_agents:
            try:
                result = await agent.autonomous_operation_cycle()
                blue_results.append(result)
            except Exception as e:
                logger.error(f"Blue team agent error: {e}")
                blue_results.append({'error': str(e)})
            
        # Update metrics
        self.metrics['total_actions'] += len(red_results) + len(blue_results)
        
        # Log tick results
        await self.logger.log_system_event({
            'event_type': 'simulation_tick',
            'tick_number': self.tick_count,
            'red_results_count': len(red_results),
            'blue_results_count': len(blue_results),
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'tick': self.tick_count,
            'red_results': red_results,
            'blue_results': blue_results,
            'metrics': self.metrics
        }
    
    async def get_status(self):
        """Get current orchestrator status"""
        return {
            'session_id': self.session_id,
            'tick_count': self.tick_count,
            'simulation_active': self.simulation_active,
            'red_agents': len(self.red_agents),
            'blue_agents': len(self.blue_agents),
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    async def shutdown(self):
        """Shutdown the orchestrator"""
        self.simulation_active = False
        self.metrics['end_time'] = datetime.now().isoformat()
        
        await self.logger.log_system_event({
            'event_type': 'orchestrator_shutdown',
            'session_id': self.session_id,
            'final_metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        })