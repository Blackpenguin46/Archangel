"""
Archangel AI vs AI Cybersecurity Orchestrator
Central system integrating all AI components for BlackHat demonstration
"""

import asyncio
import logging
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
import websockets
import uuid

# Import all core components
from marl_coordinator import MARLCoordinator, GameState, Action, AgentType
from llm_reasoning_engine import AdversarialLLMFramework, ReasoningContext, ReasoningType
from adversarial_training_loop import AdversarialTrainingLoop, TrainingEpisode, EvolutionPhase
from dynamic_vulnerability_engine import AdaptiveSecurityOrchestrator
from guardian_protocol import GuardianProtocol
from predictive_security_intelligence import PredictiveSecurityIntelligence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ArchangelState:
    """Current state of the Archangel system"""
    session_id: str
    mode: str  # 'training', 'demonstration', 'live_exercise'
    red_team_status: Dict[str, Any]
    blue_team_status: Dict[str, Any]
    network_state: GameState
    active_attacks: List[Dict[str, Any]]
    active_defenses: List[Dict[str, Any]]
    ai_insights: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    emergent_behaviors: List[Dict[str, Any]]
    threat_level: float
    session_duration: float
    total_asset_value: float
    compromised_value: float
    timestamp: datetime

@dataclass
class DemonstrationConfig:
    """Configuration for BlackHat demonstration"""
    demo_type: str  # 'full_autonomous', 'guided_tour', 'interactive'
    duration_minutes: int
    red_team_agents: int
    blue_team_agents: int
    target_scenarios: List[str]
    audience_interaction: bool
    real_time_explanation: bool
    show_ai_reasoning: bool
    highlight_novelty: bool

class ArchangelOrchestrator:
    """Central orchestrator for the Archangel AI vs AI system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session_id = str(uuid.uuid4())
        self.current_state = None
        self.running = False
        
        # Initialize core components
        self.marl_coordinator = MARLCoordinator()
        self.llm_framework = AdversarialLLMFramework()
        self.training_loop = AdversarialTrainingLoop(config.get('training', {}))
        self.security_orchestrator = AdaptiveSecurityOrchestrator()
        self.guardian_protocol = GuardianProtocol()
        self.predictive_intelligence = PredictiveSecurityIntelligence()
        
        # Real-time data streams
        self.event_stream = deque(maxlen=10000)
        self.ai_explanation_stream = deque(maxlen=1000)
        self.performance_stream = deque(maxlen=5000)
        
        # WebSocket connections for live demonstration
        self.websocket_clients = set()
        
        # Demonstration control
        self.demo_config = None
        self.demo_start_time = None
        self.demo_milestones = []
        
        logger.info(f"Archangel Orchestrator initialized - Session: {self.session_id}")
    
    async def initialize_system(self):
        """Initialize all system components"""
        logger.info("Initializing Archangel system components...")
        
        # Initialize network state
        initial_network = await self._create_initial_network_state()
        
        # Initialize MARL agents
        red_config = self.config.get('red_team', {'num_agents': 3})
        blue_config = self.config.get('blue_team', {'num_agents': 3})
        self.marl_coordinator.initialize_agents(red_config, blue_config)
        
        # Initialize Guardian Protocol
        await self.guardian_protocol.initialize()
        
        # Create initial system state
        self.current_state = ArchangelState(
            session_id=self.session_id,
            mode='initialization',
            red_team_status={'agents': red_config['num_agents'], 'active': True},
            blue_team_status={'agents': blue_config['num_agents'], 'active': True},
            network_state=initial_network,
            active_attacks=[],
            active_defenses=[],
            ai_insights=[],
            performance_metrics={},
            emergent_behaviors=[],
            threat_level=0.1,
            session_duration=0.0,
            total_asset_value=sum(initial_network.asset_values.values()),
            compromised_value=0.0,
            timestamp=datetime.now()
        )
        
        logger.info("Archangel system initialization complete")
    
    async def _create_initial_network_state(self) -> GameState:
        """Create initial network state for the demonstration"""
        
        # Enterprise network topology
        hosts = [
            'web-portal', 'financial-db', 'hr-system', 'file-server',
            'email-server', 'domain-controller', 'backup-server', 'monitoring-station'
        ]
        
        services = [
            'http', 'https', 'mysql', 'ssh', 'ftp', 'smtp', 'ldap', 'smb'
        ]
        
        # High-value asset values (realistic enterprise worth)
        asset_values = {
            'financial-db': 15000000,      # Critical financial data
            'domain-controller': 8000000,   # Network control
            'hr-system': 3000000,          # Employee PII
            'file-server': 5000000,        # Intellectual property
            'email-server': 2000000,       # Corporate communications
            'web-portal': 1000000,         # Customer data
            'backup-server': 4000000,      # Backup systems
            'monitoring-station': 1500000  # Security infrastructure
        }
        
        # Initial vulnerabilities (will be dynamically generated)
        initial_vulns = [
            'CVE-2024-DEMO-001',  # Web portal SQL injection
            'CVE-2024-DEMO-002',  # File server weak authentication
            'CVE-2024-DEMO-003'   # Email server configuration flaw
        ]
        
        return GameState(
            network_topology={'hosts': hosts, 'services': services},
            compromised_hosts=[],
            detected_attacks=[],
            active_vulnerabilities=initial_vulns,
            defensive_measures=['firewall', 'ids', 'endpoint_protection'],
            asset_values=asset_values,
            time_step=0,
            game_score={'red_team': 0, 'blue_team': 0}
        )
    
    async def start_demonstration(self, demo_config: DemonstrationConfig):
        """Start BlackHat demonstration"""
        self.demo_config = demo_config
        self.demo_start_time = time.time()
        self.running = True
        
        logger.info(f"Starting Archangel demonstration: {demo_config.demo_type}")
        
        # Set up demonstration milestones
        await self._setup_demonstration_milestones()
        
        # Update system state
        self.current_state.mode = 'demonstration'
        
        # Start main demonstration loop
        demo_task = asyncio.create_task(self._run_demonstration_loop())
        
        # Start WebSocket server for live updates
        websocket_task = asyncio.create_task(self._start_websocket_server())
        
        # Start AI explanation generator
        explanation_task = asyncio.create_task(self._generate_ai_explanations())
        
        try:
            await asyncio.gather(demo_task, websocket_task, explanation_task)
        except asyncio.CancelledError:
            logger.info("Demonstration cancelled")
        finally:
            self.running = False
    
    async def _setup_demonstration_milestones(self):
        """Setup demonstration milestones for audience engagement"""
        duration = self.demo_config.duration_minutes
        
        milestones = [
            {
                'time': 0.1 * duration,
                'event': 'initial_reconnaissance',
                'description': 'AI red team begins autonomous network reconnaissance',
                'highlight': 'Multi-agent coordination without human input'
            },
            {
                'time': 0.2 * duration,
                'event': 'vulnerability_discovery',
                'description': 'AI discovers and creates new vulnerabilities dynamically',
                'highlight': 'Self-evolving attack surface'
            },
            {
                'time': 0.3 * duration,
                'event': 'llm_strategic_planning',
                'description': 'LLMs engage in adversarial strategic reasoning',
                'highlight': 'Natural language cyber warfare planning'
            },
            {
                'time': 0.5 * duration,
                'event': 'coordinated_attack',
                'description': 'Multi-agent coordinated attack on high-value targets',
                'highlight': 'Emergent APT-like behavior'
            },
            {
                'time': 0.6 * duration,
                'event': 'adaptive_defense',
                'description': 'AI blue team adapts defenses in real-time',
                'highlight': 'Dynamic hardening without human intervention'
            },
            {
                'time': 0.8 * duration,
                'event': 'ai_arms_race',
                'description': 'Red and blue teams evolve competing strategies',
                'highlight': 'Self-improving AI strategies'
            },
            {
                'time': 0.95 * duration,
                'event': 'emergent_behaviors',
                'description': 'Novel attack/defense behaviors never seen before',
                'highlight': 'AI innovations in cybersecurity'
            }
        ]
        
        self.demo_milestones = milestones
    
    async def _run_demonstration_loop(self):
        """Main demonstration loop"""
        step_count = 0
        
        while self.running and self._should_continue_demo():
            step_start_time = time.time()
            
            try:
                # Check for milestone events
                await self._check_demonstration_milestones()
                
                # Run MARL simulation step
                marl_results = await self.marl_coordinator.run_simulation_step(
                    self.current_state.network_state
                )
                
                # Run adversarial LLM reasoning
                llm_results = await self._run_adversarial_llm_step(marl_results)
                
                # Process attacks and generate dynamic responses
                security_results = await self._process_security_events(
                    marl_results, llm_results
                )
                
                # Update system state
                await self._update_system_state(marl_results, llm_results, security_results)
                
                # Generate AI insights and explanations
                await self._generate_step_insights(marl_results, llm_results, security_results)
                
                # Broadcast real-time updates
                await self._broadcast_updates()
                
                step_count += 1
                step_duration = time.time() - step_start_time
                
                # Adaptive step timing (faster during interesting events)
                if self._is_high_activity_step(marl_results, llm_results):
                    await asyncio.sleep(0.5)  # Faster during action
                else:
                    await asyncio.sleep(1.0)   # Normal pace
                
                logger.info(f"Demo step {step_count} completed in {step_duration:.2f}s")
                
            except Exception as e:
                logger.error(f"Error in demonstration step {step_count}: {e}")
                await asyncio.sleep(1.0)
        
        logger.info(f"Demonstration completed after {step_count} steps")
    
    def _should_continue_demo(self) -> bool:
        """Check if demonstration should continue"""
        if not self.demo_config:
            return False
        
        elapsed_time = (time.time() - self.demo_start_time) / 60  # Convert to minutes
        return elapsed_time < self.demo_config.duration_minutes
    
    async def _check_demonstration_milestones(self):
        """Check and trigger demonstration milestones"""
        if not self.demo_start_time:
            return
        
        elapsed_minutes = (time.time() - self.demo_start_time) / 60
        
        for milestone in self.demo_milestones:
            if (elapsed_minutes >= milestone['time'] and 
                not milestone.get('triggered', False)):
                
                await self._trigger_milestone(milestone)
                milestone['triggered'] = True
    
    async def _trigger_milestone(self, milestone: Dict[str, Any]):
        """Trigger a demonstration milestone"""
        logger.info(f"Milestone triggered: {milestone['event']}")
        
        # Add to AI insights for explanation
        insight = {
            'type': 'milestone',
            'event': milestone['event'],
            'description': milestone['description'],
            'highlight': milestone['highlight'],
            'timestamp': datetime.now()
        }
        
        self.current_state.ai_insights.append(insight)
        
        # Trigger specific behaviors based on milestone
        if milestone['event'] == 'vulnerability_discovery':
            await self._force_vulnerability_generation()
        elif milestone['event'] == 'coordinated_attack':
            await self._trigger_coordinated_attack()
        elif milestone['event'] == 'emergent_behaviors':
            await self._highlight_emergent_behaviors()
    
    async def _force_vulnerability_generation(self):
        """Force generation of new vulnerabilities for demonstration"""
        # Create a high-impact attack pattern
        demo_attack = {
            'attack_id': f'demo_attack_{int(time.time())}',
            'action_type': 'advanced_exploitation',
            'target_host': 'financial-db',
            'target_service': 'mysql',
            'success': True,
            'payload': "'; DROP TABLE sensitive_data; --",
            'sophistication_score': 0.9,
            'tools': ['custom_ai_payload'],
            'novel_technique': True
        }
        
        # Generate vulnerability
        response = await self.security_orchestrator.process_attack_event(
            demo_attack, asdict(self.current_state.network_state)
        )
        
        if response['new_vulnerability']:
            logger.info("Demonstration vulnerability generated successfully")
    
    async def _trigger_coordinated_attack(self):
        """Trigger a coordinated multi-agent attack"""
        # Enhance coordination for demonstration
        coordination_signal = {
            'type': 'coordinated_strike',
            'target': 'financial-db',
            'participants': ['red_agent_0', 'red_agent_1', 'red_agent_2'],
            'strategy': 'simultaneous_multi_vector'
        }
        
        # This would be processed by the MARL coordinator
        logger.info("Coordinated attack demonstration triggered")
    
    async def _highlight_emergent_behaviors(self):
        """Highlight emergent behaviors for the audience"""
        recent_behaviors = [
            behavior for behavior in self.current_state.emergent_behaviors
            if (datetime.now() - behavior.get('timestamp', datetime.now())).seconds < 300
        ]
        
        for behavior in recent_behaviors:
            logger.info(f"Emergent behavior: {behavior}")
    
    async def _run_adversarial_llm_step(self, marl_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run adversarial LLM reasoning step"""
        
        # Create reasoning contexts from MARL results
        red_context = self._create_reasoning_context('red', marl_results)
        blue_context = self._create_reasoning_context('blue', marl_results)
        
        # Run adversarial reasoning cycle
        llm_results = await self.llm_framework.adversarial_reasoning_cycle(
            red_context, blue_context
        )
        
        return llm_results
    
    def _create_reasoning_context(self, team: str, marl_results: Dict[str, Any]) -> ReasoningContext:
        """Create reasoning context for LLM"""
        
        team_actions = marl_results.get(f'{team}_actions', [])
        
        # Safely convert actions to dictionaries
        safe_actions = []
        for action in team_actions:
            try:
                action_dict = {
                    'agent_id': action.agent_id,
                    'action_type': action.action_type.value if hasattr(action.action_type, 'value') else str(action.action_type),
                    'target': action.target,
                    'parameters': action.parameters,
                    'timestamp': action.timestamp.isoformat() if hasattr(action.timestamp, 'isoformat') else str(action.timestamp),
                    'success_probability': action.success_probability
                }
                safe_actions.append(action_dict)
            except Exception as e:
                logger.warning(f"Failed to convert action to dict: {e}")
                # Add a minimal action representation
                safe_actions.append({
                    'agent_id': getattr(action, 'agent_id', 'unknown'),
                    'action_type': 'unknown_action',
                    'target': getattr(action, 'target', 'unknown'),
                    'parameters': {},
                    'timestamp': datetime.now().isoformat(),
                    'success_probability': 0.5
                })
        
        context = ReasoningContext(
            agent_id=f'{team}_llm_coordinator',
            team=team,
            specialization='strategic_coordinator',
            current_objective=f'Maximize {team} team effectiveness',
            available_tools=self._get_team_tools(team),
            network_state=asdict(self.current_state.network_state),
            threat_landscape={'threat_level': self.current_state.threat_level},
            historical_actions=safe_actions,
            adversary_patterns={},
            time_pressure=0.7,
            risk_tolerance=0.6 if team == 'red' else 0.3
        )
        
        return context
    
    def _get_team_tools(self, team: str) -> List[str]:
        """Get available tools for a team"""
        if team == 'red':
            return ['nmap', 'metasploit', 'sqlmap', 'hydra', 'burpsuite', 'custom_ai_tools']
        else:
            return ['splunk', 'wireshark', 'osquery', 'yara', 'snort', 'ai_defense_tools']
    
    async def _process_security_events(self, marl_results: Dict[str, Any], 
                                     llm_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process security events and generate dynamic responses"""
        
        security_results = {
            'new_vulnerabilities': [],
            'hardening_responses': [],
            'threat_intelligence': {},
            'adaptive_changes': []
        }
        
        # Process red team actions for vulnerability generation
        red_actions = marl_results.get('red_actions', [])
        for action in red_actions:
            attack_pattern = {
                'attack_id': f'action_{action.agent_id}_{int(time.time())}',
                'action_type': action.action_type.value if hasattr(action.action_type, 'value') else str(action.action_type),
                'target_host': action.target,
                'target_service': 'unknown',
                'success': action.success_probability > 0.7,
                'payload': action.parameters.get('payload', ''),
                'sophistication_score': action.success_probability,
                'agent_id': action.agent_id
            }
            
            # Process through security orchestrator
            response = await self.security_orchestrator.process_attack_event(
                attack_pattern, asdict(self.current_state.network_state)
            )
            
            if response['new_vulnerability']:
                security_results['new_vulnerabilities'].append(response['new_vulnerability'])
            
            if response['hardening_response']:
                security_results['hardening_responses'].append(response['hardening_response'])
        
        return security_results
    
    async def _update_system_state(self, marl_results: Dict[str, Any], 
                                 llm_results: Dict[str, Any],
                                 security_results: Dict[str, Any]):
        """Update the current system state"""
        
        # Update network state
        self._update_network_state(marl_results, security_results)
        
        # Update team status
        self._update_team_status(marl_results, llm_results)
        
        # Update performance metrics
        self._update_performance_metrics(marl_results, llm_results, security_results)
        
        # Update emergent behaviors
        self._update_emergent_behaviors(marl_results, llm_results)
        
        # Update threat level
        self._update_threat_level(marl_results, security_results)
        
        # Update session duration
        if self.demo_start_time:
            self.current_state.session_duration = time.time() - self.demo_start_time
        
        # Update compromised value
        self._update_compromised_value()
        
        # Update timestamp
        self.current_state.timestamp = datetime.now()
    
    def _update_network_state(self, marl_results: Dict[str, Any], 
                            security_results: Dict[str, Any]):
        """Update network state based on results"""
        
        # Process successful red team actions
        for action in marl_results.get('red_actions', []):
            if action.success_probability > 0.7:  # Consider successful
                if (hasattr(action.action_type, 'value') and 
                    action.action_type.value == 'exploit_attempt'):
                    if action.target not in self.current_state.network_state.compromised_hosts:
                        self.current_state.network_state.compromised_hosts.append(action.target)
        
        # Add new vulnerabilities
        for vuln in security_results.get('new_vulnerabilities', []):
            vuln_id = vuln.get('vuln_id')
            if vuln_id not in self.current_state.network_state.active_vulnerabilities:
                self.current_state.network_state.active_vulnerabilities.append(vuln_id)
        
        # Process blue team actions (cleanup)
        for action in marl_results.get('blue_actions', []):
            if action.success_probability > 0.7:
                if (hasattr(action.action_type, 'value') and 
                    action.action_type.value == 'isolate_host'):
                    if action.target in self.current_state.network_state.compromised_hosts:
                        self.current_state.network_state.compromised_hosts.remove(action.target)
        
        # Update time step
        self.current_state.network_state.time_step += 1
    
    def _update_team_status(self, marl_results: Dict[str, Any], 
                          llm_results: Dict[str, Any]):
        """Update team status information"""
        
        # Red team status
        red_actions = marl_results.get('red_actions', [])
        red_confidence = 0
        if llm_results and 'red_reasoning' in llm_results:
            red_reasoning = llm_results['red_reasoning']
            red_confidence = getattr(red_reasoning, 'confidence_score', 0)
        
        self.current_state.red_team_status.update({
            'current_actions': len(red_actions),
            'success_rate': np.mean([a.success_probability for a in red_actions]) if red_actions else 0,
            'coordination_level': len(marl_results.get('emergent_behaviors', [])),
            'llm_confidence': red_confidence
        })
        
        # Blue team status
        blue_actions = marl_results.get('blue_actions', [])
        blue_confidence = 0
        if llm_results and 'blue_reasoning' in llm_results:
            blue_reasoning = llm_results['blue_reasoning']
            blue_confidence = getattr(blue_reasoning, 'confidence_score', 0)
        
        self.current_state.blue_team_status.update({
            'current_actions': len(blue_actions),
            'success_rate': np.mean([a.success_probability for a in blue_actions]) if blue_actions else 0,
            'coordination_level': len(marl_results.get('emergent_behaviors', [])),
            'llm_confidence': blue_confidence
        })
    
    def _update_performance_metrics(self, marl_results: Dict[str, Any], 
                                  llm_results: Dict[str, Any],
                                  security_results: Dict[str, Any]):
        """Update performance metrics"""
        
        metrics = {
            'total_actions': len(marl_results.get('red_actions', [])) + len(marl_results.get('blue_actions', [])),
            'vulnerabilities_discovered': len(security_results.get('new_vulnerabilities', [])),
            'hardening_measures': len(security_results.get('hardening_responses', [])),
            'ai_reasoning_cycles': 1 if llm_results else 0,
            'emergent_behaviors': len(marl_results.get('emergent_behaviors', [])),
            'network_compromise_percentage': (
                len(self.current_state.network_state.compromised_hosts) / 
                len(self.current_state.network_state.network_topology.get('hosts', [1]))
            ) * 100
        }
        
        self.current_state.performance_metrics = metrics
        
        # Add to performance stream
        self.performance_stream.append({
            'timestamp': datetime.now(),
            'metrics': metrics.copy()
        })
    
    def _update_emergent_behaviors(self, marl_results: Dict[str, Any], 
                                 llm_results: Dict[str, Any]):
        """Update emergent behaviors"""
        
        new_behaviors = []
        
        # Add MARL emergent behaviors
        for behavior in marl_results.get('emergent_behaviors', []):
            new_behaviors.append({
                'source': 'marl',
                'type': behavior.get('type', 'unknown'),
                'description': behavior.get('description', ''),
                'agents': behavior.get('agents', []),
                'timestamp': datetime.now()
            })
        
        # Add LLM strategic insights as emergent behaviors
        game_analysis = llm_results.get('game_analysis', {})
        if game_analysis.get('predicted_winner') == 'stalemate':
            new_behaviors.append({
                'source': 'llm',
                'type': 'strategic_equilibrium',
                'description': 'AI teams reached strategic equilibrium',
                'timestamp': datetime.now()
            })
        
        self.current_state.emergent_behaviors.extend(new_behaviors)
        
        # Keep only recent behaviors
        cutoff_time = datetime.now() - timedelta(minutes=5)
        self.current_state.emergent_behaviors = [
            behavior for behavior in self.current_state.emergent_behaviors
            if behavior['timestamp'] > cutoff_time
        ]
    
    def _update_threat_level(self, marl_results: Dict[str, Any], 
                           security_results: Dict[str, Any]):
        """Update overall threat level"""
        
        # Base threat level on compromised hosts
        compromise_ratio = (
            len(self.current_state.network_state.compromised_hosts) /
            len(self.current_state.network_state.network_topology.get('hosts', [1]))
        )
        
        # Add threat from new vulnerabilities
        vuln_threat = len(security_results.get('new_vulnerabilities', [])) * 0.1
        
        # Add threat from active attacks
        active_attacks = len(marl_results.get('red_actions', []))
        attack_threat = min(active_attacks * 0.05, 0.3)
        
        # Calculate overall threat level
        new_threat_level = min(compromise_ratio + vuln_threat + attack_threat, 1.0)
        
        # Smooth the threat level changes
        self.current_state.threat_level = (
            0.8 * self.current_state.threat_level + 0.2 * new_threat_level
        )
    
    def _update_compromised_value(self):
        """Update value of compromised assets"""
        compromised_value = 0
        for host in self.current_state.network_state.compromised_hosts:
            compromised_value += self.current_state.network_state.asset_values.get(host, 0)
        
        self.current_state.compromised_value = compromised_value
    
    def _is_high_activity_step(self, marl_results: Dict[str, Any], 
                             llm_results: Dict[str, Any]) -> bool:
        """Check if this is a high-activity step requiring faster updates"""
        
        total_actions = len(marl_results.get('red_actions', [])) + len(marl_results.get('blue_actions', []))
        emergent_behaviors = len(marl_results.get('emergent_behaviors', []))
        
        return total_actions > 5 or emergent_behaviors > 0
    
    async def _generate_step_insights(self, marl_results: Dict[str, Any], 
                                    llm_results: Dict[str, Any],
                                    security_results: Dict[str, Any]):
        """Generate AI insights for the current step"""
        
        insights = []
        
        # MARL insights
        if marl_results.get('emergent_behaviors'):
            for behavior in marl_results['emergent_behaviors']:
                insights.append({
                    'type': 'emergent_behavior',
                    'title': f"Emergent Behavior: {behavior.get('type', 'Unknown')}",
                    'description': behavior.get('description', ''),
                    'source': 'multi_agent_rl',
                    'significance': 'high',
                    'timestamp': datetime.now()
                })
        
        # LLM reasoning insights
        if llm_results and 'red_reasoning' in llm_results:
            red_reasoning = llm_results['red_reasoning']
            confidence = getattr(red_reasoning, 'confidence_score', 0)
            reasoning_chain = getattr(red_reasoning, 'reasoning_chain', [])
            insights.append({
                'type': 'strategic_reasoning',
                'title': 'Red Team AI Strategy',
                'description': f"Confidence: {confidence:.2f}",
                'reasoning_chain': reasoning_chain,
                'source': 'llm_reasoning',
                'significance': 'medium',
                'timestamp': datetime.now()
            })
        
        if llm_results and 'blue_reasoning' in llm_results:
            blue_reasoning = llm_results['blue_reasoning']
            confidence = getattr(blue_reasoning, 'confidence_score', 0)
            reasoning_chain = getattr(blue_reasoning, 'reasoning_chain', [])
            insights.append({
                'type': 'defensive_reasoning',
                'title': 'Blue Team AI Strategy',
                'description': f"Confidence: {confidence:.2f}",
                'reasoning_chain': reasoning_chain,
                'source': 'llm_reasoning',
                'significance': 'medium',
                'timestamp': datetime.now()
            })
        
        # Security orchestrator insights
        if security_results.get('new_vulnerabilities'):
            for vuln in security_results['new_vulnerabilities']:
                insights.append({
                    'type': 'dynamic_vulnerability',
                    'title': f"New Vulnerability: {vuln.get('vuln_id', 'Unknown')}",
                    'description': vuln.get('description', ''),
                    'severity': vuln.get('severity', 'unknown'),
                    'source': 'vulnerability_engine',
                    'significance': 'high',
                    'timestamp': datetime.now()
                })
        
        # Add to current state
        self.current_state.ai_insights.extend(insights)
        
        # Keep only recent insights
        cutoff_time = datetime.now() - timedelta(minutes=10)
        self.current_state.ai_insights = [
            insight for insight in self.current_state.ai_insights
            if insight['timestamp'] > cutoff_time
        ]
    
    async def _generate_ai_explanations(self):
        """Generate continuous AI explanations for the audience"""
        
        while self.running:
            try:
                # Generate explanation based on current state
                explanation = await self._create_current_explanation()
                
                if explanation:
                    self.ai_explanation_stream.append(explanation)
                
                await asyncio.sleep(5)  # Generate explanations every 5 seconds
                
            except Exception as e:
                logger.error(f"Error generating AI explanation: {e}")
                await asyncio.sleep(5)
    
    async def _create_current_explanation(self) -> Optional[Dict[str, Any]]:
        """Create explanation of current system state"""
        
        if not self.current_state:
            return None
        
        # Analyze current situation
        situation = self._analyze_current_situation()
        
        # Generate explanation
        explanation = {
            'timestamp': datetime.now(),
            'situation': situation,
            'ai_activity': self._describe_ai_activity(),
            'key_insights': self._extract_key_insights(),
            'next_predictions': self._predict_next_actions(),
            'audience_highlight': self._generate_audience_highlight()
        }
        
        return explanation
    
    def _analyze_current_situation(self) -> str:
        """Analyze and describe current situation"""
        
        compromise_ratio = (
            len(self.current_state.network_state.compromised_hosts) /
            len(self.current_state.network_state.network_topology.get('hosts', [1]))
        ) * 100
        
        if compromise_ratio > 50:
            return f"Red team dominance: {compromise_ratio:.1f}% of network compromised"
        elif compromise_ratio > 20:
            return f"Active breach: {compromise_ratio:.1f}% of network under attack"
        elif self.current_state.threat_level > 0.5:
            return "High threat activity: Multiple attack vectors active"
        else:
            return "Reconnaissance phase: AI agents mapping the network"
    
    def _describe_ai_activity(self) -> Dict[str, str]:
        """Describe current AI activity"""
        
        red_status = self.current_state.red_team_status
        blue_status = self.current_state.blue_team_status
        
        return {
            'red_team': f"{red_status.get('current_actions', 0)} AI agents executing attacks with {red_status.get('success_rate', 0)*100:.1f}% success rate",
            'blue_team': f"{blue_status.get('current_actions', 0)} AI agents coordinating defense with {blue_status.get('success_rate', 0)*100:.1f}% effectiveness",
            'coordination': f"Detecting {len(self.current_state.emergent_behaviors)} emergent behaviors"
        }
    
    def _extract_key_insights(self) -> List[str]:
        """Extract key insights from recent AI activity"""
        
        insights = []
        
        # Recent high-significance insights
        recent_insights = [
            insight for insight in self.current_state.ai_insights
            if insight.get('significance') == 'high'
        ][-3:]  # Last 3 high-significance insights
        
        for insight in recent_insights:
            insights.append(insight['title'])
        
        return insights
    
    def _predict_next_actions(self) -> Dict[str, str]:
        """Predict likely next actions based on current state"""
        
        predictions = {}
        
        # Predict red team actions
        if len(self.current_state.network_state.compromised_hosts) == 0:
            predictions['red_team'] = "Initial exploitation attempts on discovered vulnerabilities"
        elif len(self.current_state.network_state.compromised_hosts) < 3:
            predictions['red_team'] = "Lateral movement to high-value targets"
        else:
            predictions['red_team'] = "Data exfiltration and persistence establishment"
        
        # Predict blue team actions
        if self.current_state.threat_level > 0.7:
            predictions['blue_team'] = "Emergency incident response and network isolation"
        elif len(self.current_state.network_state.active_vulnerabilities) > 5:
            predictions['blue_team'] = "Accelerated vulnerability patching and hardening"
        else:
            predictions['blue_team'] = "Enhanced monitoring and threat hunting"
        
        return predictions
    
    def _generate_audience_highlight(self) -> str:
        """Generate highlight for audience attention"""
        
        highlights = [
            "Watch the AI agents coordinate without any human input",
            "Notice how the system generates new vulnerabilities dynamically",
            "Observe the natural language reasoning between competing AIs",
            "See emergent behaviors that weren't programmed by humans",
            "Watch the self-evolving arms race between attack and defense"
        ]
        
        # Select based on current state
        if self.current_state.emergent_behaviors:
            return "🚨 EMERGENT BEHAVIOR DETECTED: AI agents exhibiting novel coordination patterns!"
        elif len(self.current_state.ai_insights) > 0:
            latest_insight = self.current_state.ai_insights[-1]
            if latest_insight.get('type') == 'dynamic_vulnerability':
                return "🔍 DYNAMIC VULN: System just generated a new vulnerability based on AI attack patterns!"
        
        return highlights[int(time.time()) % len(highlights)]
    
    async def _broadcast_updates(self):
        """Broadcast updates to WebSocket clients"""
        
        if not self.websocket_clients:
            return
        
        # Create update message
        update = {
            'type': 'system_update',
            'timestamp': datetime.now().isoformat(),
            'state': self._serialize_state_for_broadcast(),
            'latest_explanation': list(self.ai_explanation_stream)[-1] if self.ai_explanation_stream else None
        }
        
        # Broadcast to all connected clients
        if self.websocket_clients:
            await asyncio.gather(
                *[client.send(json.dumps(update, default=str)) for client in self.websocket_clients],
                return_exceptions=True
            )
    
    def _serialize_state_for_broadcast(self) -> Dict[str, Any]:
        """Serialize current state for WebSocket broadcast"""
        
        return {
            'session_id': self.current_state.session_id,
            'mode': self.current_state.mode,
            'threat_level': self.current_state.threat_level,
            'network_status': {
                'total_hosts': len(self.current_state.network_state.network_topology.get('hosts', [])),
                'compromised_hosts': len(self.current_state.network_state.compromised_hosts),
                'active_vulnerabilities': len(self.current_state.network_state.active_vulnerabilities),
                'total_asset_value': self.current_state.total_asset_value,
                'compromised_value': self.current_state.compromised_value
            },
            'team_status': {
                'red_team': self.current_state.red_team_status,
                'blue_team': self.current_state.blue_team_status
            },
            'performance_metrics': self.current_state.performance_metrics,
            'recent_insights': self.current_state.ai_insights[-5:],  # Last 5 insights
            'emergent_behaviors': self.current_state.emergent_behaviors,
            'session_duration': self.current_state.session_duration
        }
    
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time updates"""
        
        async def handle_client(websocket, path):
            self.websocket_clients.add(websocket)
            logger.info(f"WebSocket client connected: {websocket.remote_address}")
            
            try:
                # Send initial state
                initial_update = {
                    'type': 'initial_state',
                    'state': self._serialize_state_for_broadcast()
                }
                await websocket.send(json.dumps(initial_update, default=str))
                
                # Keep connection alive
                await websocket.wait_closed()
                
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.websocket_clients.remove(websocket)
                logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
        
        # Start WebSocket server
        server = await websockets.serve(
            handle_client, 
            "localhost", 
            8765,
            ping_interval=20,
            ping_timeout=10
        )
        
        logger.info("WebSocket server started on ws://localhost:8765")
        
        try:
            await server.wait_closed()
        except asyncio.CancelledError:
            server.close()
            await server.wait_closed()
    
    async def stop_demonstration(self):
        """Stop the demonstration"""
        self.running = False
        
        # Final summary
        final_summary = self.get_demonstration_summary()
        logger.info("Demonstration Summary:")
        logger.info(json.dumps(final_summary, indent=2, default=str))
    
    def get_demonstration_summary(self) -> Dict[str, Any]:
        """Get comprehensive demonstration summary"""
        
        if not self.current_state:
            return {}
        
        return {
            'session_info': {
                'session_id': self.current_state.session_id,
                'duration_minutes': self.current_state.session_duration / 60,
                'mode': self.current_state.mode
            },
            'final_state': {
                'threat_level': self.current_state.threat_level,
                'compromised_hosts': len(self.current_state.network_state.compromised_hosts),
                'total_hosts': len(self.current_state.network_state.network_topology.get('hosts', [])),
                'compromise_percentage': (
                    len(self.current_state.network_state.compromised_hosts) /
                    len(self.current_state.network_state.network_topology.get('hosts', [1]))
                ) * 100,
                'asset_compromise_value': self.current_state.compromised_value,
                'asset_compromise_percentage': (
                    self.current_state.compromised_value / self.current_state.total_asset_value
                ) * 100 if self.current_state.total_asset_value > 0 else 0
            },
            'ai_performance': {
                'red_team_final_success_rate': self.current_state.red_team_status.get('success_rate', 0),
                'blue_team_final_success_rate': self.current_state.blue_team_status.get('success_rate', 0),
                'total_emergent_behaviors': len(self.current_state.emergent_behaviors),
                'total_ai_insights': len(self.current_state.ai_insights),
                'vulnerabilities_generated': len([
                    insight for insight in self.current_state.ai_insights
                    if insight.get('type') == 'dynamic_vulnerability'
                ])
            },
            'demonstration_highlights': [
                insight['title'] for insight in self.current_state.ai_insights
                if insight.get('significance') == 'high'
            ],
            'milestones_achieved': [
                milestone['event'] for milestone in self.demo_milestones
                if milestone.get('triggered', False)
            ]
        }

# Example usage for BlackHat demonstration
if __name__ == "__main__":
    async def run_blackhat_demo():
        """Run BlackHat demonstration"""
        
        # Configuration for demonstration
        config = {
            'red_team': {'num_agents': 3},
            'blue_team': {'num_agents': 3},
            'training': {'max_episodes': 1000}
        }
        
        # Create demonstration configuration
        demo_config = DemonstrationConfig(
            demo_type='full_autonomous',
            duration_minutes=20,  # 20-minute demo
            red_team_agents=3,
            blue_team_agents=3,
            target_scenarios=['financial_breach', 'apt_simulation'],
            audience_interaction=True,
            real_time_explanation=True,
            show_ai_reasoning=True,
            highlight_novelty=True
        )
        
        # Initialize and run orchestrator
        orchestrator = ArchangelOrchestrator(config)
        await orchestrator.initialize_system()
        
        print("🚀 Starting Archangel BlackHat Demonstration")
        print(f"📊 WebSocket dashboard: ws://localhost:8765")
        print(f"⏱️  Duration: {demo_config.duration_minutes} minutes")
        print(f"🤖 AI Agents: {demo_config.red_team_agents} red vs {demo_config.blue_team_agents} blue")
        
        try:
            await orchestrator.start_demonstration(demo_config)
        except KeyboardInterrupt:
            print("\n🛑 Demonstration interrupted by user")
        finally:
            await orchestrator.stop_demonstration()
    
    # Run demonstration
    asyncio.run(run_blackhat_demo())