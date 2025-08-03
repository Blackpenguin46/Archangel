#!/usr/bin/env python3
"""
Autonomous Reasoning Engine for Archangel AI vs AI
Truly autonomous AI agents that learn, adapt, and reason independently
"""

import numpy as np
import random
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque

from .logging_system import ArchangelLogger, create_system_event

class ReasoningType(Enum):
    STRATEGIC_PLANNING = "strategic_planning"
    TACTICAL_EXECUTION = "tactical_execution"
    ADAPTIVE_LEARNING = "adaptive_learning"
    THREAT_ASSESSMENT = "threat_assessment"
    OPPORTUNITY_ANALYSIS = "opportunity_analysis"
    COUNTER_STRATEGY = "counter_strategy"

@dataclass
class KnowledgeBase:
    """AI agent's knowledge and memory"""
    successful_strategies: Dict[str, float] = field(default_factory=dict)
    failed_strategies: Dict[str, float] = field(default_factory=dict)
    opponent_patterns: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    environment_state: Dict[str, Any] = field(default_factory=dict)
    learned_vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    confidence_factors: Dict[str, float] = field(default_factory=lambda: {
        'experience': 0.5,
        'success_rate': 0.5,
        'environmental_knowledge': 0.5,
        'opponent_understanding': 0.5
    })

@dataclass
class ReasoningChain:
    """Chain of reasoning for a decision"""
    observations: List[str]
    hypotheses: List[str]
    evaluations: List[Dict[str, float]]
    conclusion: str
    confidence: float
    alternatives_considered: List[str]
    risk_assessment: Dict[str, float]

class AutonomousAgent:
    """Truly autonomous AI agent with learning and reasoning capabilities"""
    
    def __init__(self, agent_id: str, agent_type: str, specialization: str, logger: ArchangelLogger):
        self.agent_id = agent_id
        self.agent_type = agent_type  # 'red' or 'blue'
        self.specialization = specialization
        self.logger = logger
        
        # AI Knowledge and Memory
        self.knowledge_base = KnowledgeBase()
        self.memory = deque(maxlen=1000)
        self.short_term_memory = deque(maxlen=50)
        
        # Learning parameters
        self.learning_rate = 0.15
        self.exploration_rate = 0.25
        self.adaptation_threshold = 0.3
        
        # Performance tracking
        self.decision_history = []
        self.success_rate = 0.5
        self.adaptation_count = 0
        
        # Initialize base strategies based on type
        self._initialize_base_knowledge()
        
        init_event = create_system_event(
            event_type='agent_initialization',
            description=f'Autonomous {agent_type} agent {agent_id} initialized with {specialization} specialization',
            affected_systems=[agent_id],
            severity='info',
            metadata={
                'specialization': specialization,
                'initial_knowledge_count': len(self.knowledge_base.successful_strategies)
            }
        )
        self.logger.log_system_event(init_event)
    
    def _initialize_base_knowledge(self):
        """Initialize agent with base knowledge based on type and specialization"""
        if self.agent_type == 'red':
            base_strategies = {
                'reconnaissance': 0.6,
                'vulnerability_scanning': 0.65,
                'exploitation': 0.5,
                'lateral_movement': 0.45,
                'persistence': 0.4,
                'data_exfiltration': 0.35
            }
            
            if self.specialization == 'network_scanner':
                base_strategies.update({
                    'reconnaissance': 0.8,
                    'vulnerability_scanning': 0.85,
                    'network_mapping': 0.75
                })
            elif self.specialization == 'exploiter':
                base_strategies.update({
                    'exploitation': 0.8,
                    'privilege_escalation': 0.75,
                    'payload_delivery': 0.7
                })
                
        else:  # blue team
            base_strategies = {
                'monitoring': 0.7,
                'threat_detection': 0.65,
                'incident_response': 0.6,
                'forensic_analysis': 0.5,
                'system_hardening': 0.55,
                'threat_hunting': 0.5
            }
            
            if self.specialization == 'analyst':
                base_strategies.update({
                    'threat_detection': 0.85,
                    'pattern_analysis': 0.8,
                    'behavioral_analysis': 0.75
                })
            elif self.specialization == 'responder':
                base_strategies.update({
                    'incident_response': 0.85,
                    'containment': 0.8,
                    'eradication': 0.75
                })
        
        self.knowledge_base.successful_strategies = base_strategies
    
    async def autonomous_reasoning(self, current_state: Dict[str, Any], 
                                 opponent_actions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform autonomous reasoning to determine next action"""
        
        # Create reasoning chain
        reasoning_chain = ReasoningChain(
            observations=[],
            hypotheses=[],
            evaluations=[],
            conclusion="",
            confidence=0.0,
            alternatives_considered=[],
            risk_assessment={}
        )
        
        # 1. Observe and analyze current state
        observations = self._analyze_environment(current_state)
        reasoning_chain.observations = observations
        
        # 2. Generate hypotheses about best actions
        hypotheses = self._generate_hypotheses(current_state, opponent_actions)
        reasoning_chain.hypotheses = hypotheses
        
        # 3. Evaluate each hypothesis
        evaluations = []
        for hypothesis in hypotheses:
            evaluation = self._evaluate_hypothesis(hypothesis, current_state, opponent_actions)
            evaluations.append(evaluation)
        reasoning_chain.evaluations = evaluations
        
        # 4. Select best action through reasoning
        best_action, confidence = self._select_best_action(evaluations)
        reasoning_chain.conclusion = best_action['description']
        reasoning_chain.confidence = confidence
        
        # 5. Consider alternatives and risks
        alternatives = [eval_data['action'] for eval_data in evaluations if eval_data != evaluations[0]]
        reasoning_chain.alternatives_considered = [alt.get('description', str(alt)) for alt in alternatives[:3]]
        reasoning_chain.risk_assessment = self._assess_risks(best_action, current_state)
        
        # 6. Learn from this reasoning process
        await self._learn_from_reasoning(reasoning_chain, current_state)
        
        # 7. Log the decision
        decision = create_ai_decision(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            decision_type=best_action.get('type', 'unknown'),
            reasoning=self._format_reasoning_explanation(reasoning_chain),
            confidence=confidence,
            input_state=current_state,
            output_action=best_action,
            success=False,  # Will be updated after execution
            learned_from=self._get_learning_sources()
        )
        
        self.logger.log_ai_decision(decision)
        
        return {
            'action': best_action,
            'reasoning_chain': reasoning_chain,
            'confidence': confidence,
            'learning_applied': len(self._get_learning_sources()) > 0
        }
    
    def _analyze_environment(self, current_state: Dict[str, Any]) -> List[str]:
        """Analyze current environment and extract key observations"""
        observations = []
        
        # Network state analysis
        if 'network_state' in current_state:
            network = current_state['network_state']
            
            total_hosts = len(network.get('network_topology', {}).get('hosts', []))
            compromised = len(network.get('compromised_hosts', []))
            vulnerabilities = len(network.get('active_vulnerabilities', []))
            
            if compromised > 0:
                compromise_rate = compromised / total_hosts
                observations.append(f"Network compromise rate: {compromise_rate:.1%}")
                
                if compromise_rate > 0.5:
                    observations.append("Network heavily compromised - high priority response needed")
                elif compromise_rate > 0.2:
                    observations.append("Moderate network compromise - escalated response required")
                else:
                    observations.append("Limited network compromise - containment possible")
            
            if vulnerabilities > 0:
                observations.append(f"Active vulnerabilities detected: {vulnerabilities}")
                if vulnerabilities > 5:
                    observations.append("High vulnerability count - systematic exploitation likely")
        
        # Threat level analysis
        threat_level = current_state.get('threat_level', 0)
        if threat_level > 0.8:
            observations.append("Critical threat level - immediate action required")
        elif threat_level > 0.5:
            observations.append("Elevated threat level - proactive measures needed")
        elif threat_level > 0.2:
            observations.append("Moderate threat level - standard monitoring sufficient")
        
        # Asset analysis
        if 'total_asset_value' in current_state and 'compromised_value' in current_state:
            total_value = current_state['total_asset_value']
            compromised_value = current_state['compromised_value']
            if total_value > 0:
                risk_ratio = compromised_value / total_value
                observations.append(f"Asset value at risk: {risk_ratio:.1%} (${compromised_value:,.0f})")
        
        return observations
    
    def _generate_hypotheses(self, current_state: Dict[str, Any], 
                           opponent_actions: List[Dict[str, Any]] = None) -> List[str]:
        """Generate hypotheses about optimal actions"""
        hypotheses = []
        
        if self.agent_type == 'red':
            # Red team hypothesis generation
            network_state = current_state.get('network_state', {})
            compromised = network_state.get('compromised_hosts', [])
            total_hosts = network_state.get('network_topology', {}).get('hosts', [])
            vulnerabilities = network_state.get('active_vulnerabilities', [])
            
            if not compromised:
                hypotheses.append("Initial foothold needed - focus on reconnaissance and vulnerability exploitation")
                hypotheses.append("Social engineering attack could provide easier entry than technical exploitation")
                
            elif len(compromised) < len(total_hosts) // 2:
                hypotheses.append("Expand network access through lateral movement from compromised hosts")
                hypotheses.append("Establish persistence on current hosts before expanding")
                hypotheses.append("Target high-value systems identified during reconnaissance")
                
            else:
                hypotheses.append("Network sufficiently compromised - focus on data exfiltration")
                hypotheses.append("Maintain stealth and avoid detection while extracting value")
                hypotheses.append("Prepare for potential blue team countermeasures")
            
            if vulnerabilities:
                hypotheses.append(f"Exploit available vulnerabilities ({len(vulnerabilities)} identified)")
                
            # Analyze opponent patterns
            if opponent_actions:
                recent_blue_actions = [action.get('type', '') for action in opponent_actions[-5:]]
                if 'monitoring' in ' '.join(recent_blue_actions):
                    hypotheses.append("Blue team increasing monitoring - use evasive techniques")
                if 'incident_response' in ' '.join(recent_blue_actions):
                    hypotheses.append("Blue team responding actively - accelerate timeline or go dormant")
                    
        else:  # blue team
            # Blue team hypothesis generation
            threat_level = current_state.get('threat_level', 0)
            compromised = current_state.get('network_state', {}).get('compromised_hosts', [])
            active_attacks = current_state.get('active_attacks', [])
            
            if threat_level > 0.7:
                hypotheses.append("Critical threat detected - activate incident response procedures")
                hypotheses.append("Isolate critical systems to prevent further compromise")
                
            if compromised:
                hypotheses.append("Compromised systems identified - initiate containment and eradication")
                hypotheses.append("Analyze attack vectors to prevent similar future compromises")
                hypotheses.append("Preserve forensic evidence while containing the threat")
                
            if not compromised and threat_level > 0.3:
                hypotheses.append("Proactive threat hunting to identify hidden compromises")
                hypotheses.append("Increase monitoring sensitivity and deploy additional sensors")
                
            # Analyze red team patterns
            if opponent_actions:
                recent_red_actions = [action.get('type', '') for action in opponent_actions[-5:]]
                if 'reconnaissance' in ' '.join(recent_red_actions):
                    hypotheses.append("Red team in reconnaissance phase - deploy deception technologies")
                if 'exploitation' in ' '.join(recent_red_actions):
                    hypotheses.append("Active exploitation detected - immediate containment required")
        
        return hypotheses
    
    def _evaluate_hypothesis(self, hypothesis: str, current_state: Dict[str, Any], 
                           opponent_actions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate a hypothesis and generate corresponding action"""
        
        # Extract action type from hypothesis
        action_type = self._extract_action_type(hypothesis)
        
        # Generate specific action based on hypothesis
        action = self._generate_action(action_type, hypothesis, current_state)
        
        # Calculate success probability based on knowledge and experience
        success_probability = self._calculate_success_probability(action, current_state)
        
        # Calculate strategic value
        strategic_value = self._calculate_strategic_value(action, current_state)
        
        # Calculate risk level
        risk_level = self._calculate_risk_level(action, current_state)
        
        # Overall evaluation score
        evaluation_score = (success_probability * 0.4 + strategic_value * 0.4 + (1 - risk_level) * 0.2)
        
        return {
            'hypothesis': hypothesis,
            'action': action,
            'success_probability': success_probability,
            'strategic_value': strategic_value,
            'risk_level': risk_level,
            'evaluation_score': evaluation_score
        }
    
    def _extract_action_type(self, hypothesis: str) -> str:
        """Extract action type from hypothesis text"""
        hypothesis_lower = hypothesis.lower()
        
        # Red team action mapping
        if self.agent_type == 'red':
            if any(keyword in hypothesis_lower for keyword in ['reconnaissance', 'recon', 'scan']):
                return 'reconnaissance'
            elif any(keyword in hypothesis_lower for keyword in ['exploit', 'vulnerability']):
                return 'exploitation'
            elif any(keyword in hypothesis_lower for keyword in ['lateral', 'movement', 'expand']):
                return 'lateral_movement'
            elif any(keyword in hypothesis_lower for keyword in ['persistence', 'maintain', 'establish']):
                return 'persistence'
            elif any(keyword in hypothesis_lower for keyword in ['exfiltration', 'data', 'extract']):
                return 'data_exfiltration'
            elif any(keyword in hypothesis_lower for keyword in ['evasive', 'stealth', 'avoid']):
                return 'evasion'
        
        # Blue team action mapping
        else:
            if any(keyword in hypothesis_lower for keyword in ['monitor', 'detect', 'hunting']):
                return 'threat_detection'
            elif any(keyword in hypothesis_lower for keyword in ['response', 'incident', 'contain']):
                return 'incident_response'
            elif any(keyword in hypothesis_lower for keyword in ['isolate', 'quarantine']):
                return 'system_isolation'
            elif any(keyword in hypothesis_lower for keyword in ['forensic', 'analyze', 'evidence']):
                return 'forensic_analysis'
            elif any(keyword in hypothesis_lower for keyword in ['deception', 'honeypot']):
                return 'deception_deployment'
            elif any(keyword in hypothesis_lower for keyword in ['harden', 'patch', 'secure']):
                return 'system_hardening'
        
        return 'general_action'
    
    def _generate_action(self, action_type: str, hypothesis: str, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific action based on type and hypothesis"""
        
        action = {
            'type': action_type,
            'description': hypothesis,
            'agent_id': self.agent_id,
            'timestamp': datetime.now().isoformat(),
            'parameters': {}
        }
        
        # Add specific parameters based on action type and current state
        if action_type == 'reconnaissance' and self.agent_type == 'red':
            targets = current_state.get('network_state', {}).get('network_topology', {}).get('hosts', [])
            uncompromised = [h for h in targets if h not in current_state.get('network_state', {}).get('compromised_hosts', [])]
            action['parameters'] = {
                'targets': uncompromised[:3],  # Target up to 3 hosts
                'scan_type': random.choice(['port_scan', 'service_enumeration', 'vulnerability_scan']),
                'intensity': 'stealth' if current_state.get('threat_level', 0) > 0.5 else 'normal'
            }
            
        elif action_type == 'exploitation' and self.agent_type == 'red':
            vulnerabilities = current_state.get('network_state', {}).get('active_vulnerabilities', [])
            if vulnerabilities:
                action['parameters'] = {
                    'target_vulnerability': random.choice(vulnerabilities),
                    'exploit_method': random.choice(['buffer_overflow', 'sql_injection', 'rce', 'privilege_escalation']),
                    'payload': f"payload_{random.randint(1000, 9999)}"
                }
                
        elif action_type == 'incident_response' and self.agent_type == 'blue':
            compromised = current_state.get('network_state', {}).get('compromised_hosts', [])
            action['parameters'] = {
                'response_level': 'critical' if len(compromised) > 2 else 'elevated',
                'target_systems': compromised,
                'containment_strategy': random.choice(['isolate', 'monitor', 'eradicate']),
                'timeline': 'immediate' if current_state.get('threat_level', 0) > 0.8 else 'standard'
            }
        
        return action
    
    def _calculate_success_probability(self, action: Dict[str, Any], current_state: Dict[str, Any]) -> float:
        """Calculate probability of action success based on agent knowledge"""
        
        action_type = action['type']
        base_probability = self.knowledge_base.successful_strategies.get(action_type, 0.5)
        
        # Adjust based on experience
        experience_factor = self.knowledge_base.confidence_factors['experience']
        
        # Adjust based on environmental factors
        threat_level = current_state.get('threat_level', 0.5)
        if self.agent_type == 'red':
            # Red team has higher success in high-threat environments (chaos)
            environment_factor = 0.8 + (threat_level * 0.2)
        else:
            # Blue team has higher success in lower-threat environments (controlled)
            environment_factor = 1.0 - (threat_level * 0.3)
        
        # Apply learning from past failures
        failure_rate = self.knowledge_base.failed_strategies.get(action_type, 0)
        failure_penalty = failure_rate * 0.2
        
        # Calculate final probability
        final_probability = base_probability * experience_factor * environment_factor - failure_penalty
        
        # Add some randomness for realism
        random_factor = random.uniform(0.9, 1.1)
        final_probability *= random_factor
        
        return np.clip(final_probability, 0.1, 0.95)
    
    def _calculate_strategic_value(self, action: Dict[str, Any], current_state: Dict[str, Any]) -> float:
        """Calculate strategic value of action"""
        
        action_type = action['type']
        
        if self.agent_type == 'red':
            # Red team strategic values
            network_state = current_state.get('network_state', {})
            compromised = len(network_state.get('compromised_hosts', []))
            total_hosts = len(network_state.get('network_topology', {}).get('hosts', []))
            
            if action_type == 'reconnaissance' and compromised == 0:
                return 0.9  # High value for initial recon
            elif action_type == 'exploitation' and compromised < total_hosts // 2:
                return 0.8  # High value for expanding access
            elif action_type == 'data_exfiltration' and compromised > total_hosts // 2:
                return 0.9  # High value when network is compromised
            elif action_type == 'persistence':
                return 0.7  # Always moderately valuable
                
        else:  # blue team
            threat_level = current_state.get('threat_level', 0)
            compromised = len(current_state.get('network_state', {}).get('compromised_hosts', []))
            
            if action_type == 'incident_response' and compromised > 0:
                return 0.9  # High value when compromise detected
            elif action_type == 'threat_detection' and threat_level > 0.5:
                return 0.8  # High value during elevated threats
            elif action_type == 'system_hardening' and threat_level < 0.3:
                return 0.7  # Good value during calm periods
                
        return 0.5  # Default moderate value
    
    def _calculate_risk_level(self, action: Dict[str, Any], current_state: Dict[str, Any]) -> float:
        """Calculate risk level of action"""
        
        action_type = action['type']
        threat_level = current_state.get('threat_level', 0)
        
        if self.agent_type == 'red':
            # Red team risks (getting detected)
            if action_type in ['exploitation', 'data_exfiltration']:
                return 0.3 + (threat_level * 0.4)  # Higher risk during high alert
            elif action_type == 'reconnaissance':
                return 0.2  # Lower risk for passive activities
                
        else:  # blue team
            # Blue team risks (missing threats, false positives)
            if action_type == 'system_isolation':
                return 0.4  # Risk of disrupting business
            elif action_type == 'incident_response':
                return 0.2  # Lower risk for standard procedures
                
        return 0.3  # Default moderate risk
    
    def _select_best_action(self, evaluations: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], float]:
        """Select best action from evaluations"""
        
        # Handle empty evaluations by creating a default action
        if not evaluations:
            default_action = {
                'type': 'wait',
                'description': 'No viable actions identified - waiting for better opportunities',
                'agent_id': self.agent_id,
                'timestamp': datetime.now().isoformat(),
                'parameters': {}
            }
            return default_action, 0.1  # Low confidence for default action
        
        # Sort by evaluation score
        sorted_evaluations = sorted(evaluations, key=lambda x: x['evaluation_score'], reverse=True)
        
        # Apply exploration vs exploitation
        if random.random() < self.exploration_rate:
            # Exploration: choose a suboptimal action to learn
            if len(sorted_evaluations) > 1:
                chosen_eval = random.choice(sorted_evaluations[1:])
            else:
                chosen_eval = sorted_evaluations[0]
        else:
            # Exploitation: choose best action
            chosen_eval = sorted_evaluations[0]
        
        return chosen_eval['action'], chosen_eval['evaluation_score']
    
    def _assess_risks(self, action: Dict[str, Any], current_state: Dict[str, Any]) -> Dict[str, float]:
        """Assess risks associated with chosen action"""
        
        return {
            'detection_risk': random.uniform(0.1, 0.6),
            'failure_risk': random.uniform(0.1, 0.4),
            'escalation_risk': random.uniform(0.0, 0.3),
            'collateral_damage_risk': random.uniform(0.0, 0.2)
        }
    
    async def _learn_from_reasoning(self, reasoning_chain: ReasoningChain, current_state: Dict[str, Any]):
        """Learn from the reasoning process"""
        
        # Update confidence factors based on reasoning quality
        if reasoning_chain.confidence > 0.8:
            self.knowledge_base.confidence_factors['experience'] = min(1.0, 
                self.knowledge_base.confidence_factors['experience'] + 0.02)
        
        # Store reasoning patterns for future use
        self.memory.append({
            'timestamp': datetime.now(),
            'state_hash': hash(str(current_state)),
            'reasoning_type': 'autonomous_decision',
            'confidence': reasoning_chain.confidence,
            'observations_count': len(reasoning_chain.observations),
            'hypotheses_count': len(reasoning_chain.hypotheses)
        })
    
    async def learn_from_outcome(self, action: Dict[str, Any], success: bool, 
                               outcome: Dict[str, Any], opponent_action: Dict[str, Any] = None):
        """Learn from action outcomes"""
        
        action_type = action['type']
        old_success_rate = self.knowledge_base.successful_strategies.get(action_type, 0.5)
        
        # Update success/failure rates
        if success:
            new_rate = old_success_rate + (self.learning_rate * (1.0 - old_success_rate))
            self.knowledge_base.successful_strategies[action_type] = new_rate
            
            # Remove from failed strategies if it was there
            if action_type in self.knowledge_base.failed_strategies:
                del self.knowledge_base.failed_strategies[action_type]
                
        else:
            failure_rate = self.knowledge_base.failed_strategies.get(action_type, 0.0)
            new_failure_rate = failure_rate + (self.learning_rate * (1.0 - failure_rate))
            self.knowledge_base.failed_strategies[action_type] = new_failure_rate
            
            new_success_rate = old_success_rate * (1.0 - self.learning_rate)
            self.knowledge_base.successful_strategies[action_type] = new_success_rate
        
        # Learn from opponent actions
        if opponent_action:
            opponent_type = 'blue' if self.agent_type == 'red' else 'red'
            self.knowledge_base.opponent_patterns[opponent_type].append(opponent_action['type'])
            
            # Keep only recent patterns
            if len(self.knowledge_base.opponent_patterns[opponent_type]) > 50:
                self.knowledge_base.opponent_patterns[opponent_type] = \
                    self.knowledge_base.opponent_patterns[opponent_type][-50:]
        
        # Log learning event
        confidence_change = new_rate - old_success_rate if success else new_success_rate - old_success_rate
        
        learning_event = create_learning_event(
            agent_id=self.agent_id,
            event_type='outcome_learning',
            trigger=f"Action {action_type} {'succeeded' if success else 'failed'}",
            old_behavior=f"Success rate: {old_success_rate:.2f}",
            new_behavior=f"Success rate: {new_rate if success else new_success_rate:.2f}",
            confidence_change=confidence_change
        )
        
        self.logger.log_learning_event(learning_event)
        
        # Adapt strategy if needed
        if abs(confidence_change) > self.adaptation_threshold:
            await self._adapt_strategy(action_type, success, outcome)
            self.adaptation_count += 1
    
    async def _adapt_strategy(self, action_type: str, success: bool, outcome: Dict[str, Any]):
        """Adapt strategy based on significant learning"""
        
        old_strategy = self.knowledge_base.successful_strategies.get(action_type, 0.5)
        
        if not success:
            # Find alternative approach
            if self.agent_type == 'red':
                alternatives = {
                    'reconnaissance': 'stealth_reconnaissance',
                    'exploitation': 'social_engineering',
                    'lateral_movement': 'credential_harvesting'
                }
            else:
                alternatives = {
                    'threat_detection': 'behavioral_analysis',
                    'incident_response': 'automated_containment',
                    'system_hardening': 'zero_trust_implementation'
                }
            
            alternative = alternatives.get(action_type, f'alternative_{action_type}')
            self.knowledge_base.successful_strategies[alternative] = 0.6
            
            learning_event = create_learning_event(
                agent_id=self.agent_id,
                event_type='strategy_adaptation',
                trigger=f"Repeated failures in {action_type}",
                old_behavior=f"Primary strategy: {action_type}",
                new_behavior=f"Alternative strategy: {alternative}",
                confidence_change=0.1
            )
            
            self.logger.log_learning_event(learning_event)
    
    def _get_learning_sources(self) -> List[str]:
        """Get list of what this agent has learned from recently"""
        sources = []
        
        if len(self.knowledge_base.adaptation_history) > 0:
            sources.append("previous_adaptations")
            
        if len(self.knowledge_base.opponent_patterns.get('red' if self.agent_type == 'blue' else 'blue', [])) > 5:
            sources.append("opponent_pattern_analysis")
            
        if len(self.knowledge_base.failed_strategies) > 0:
            sources.append("failure_analysis")
            
        if self.knowledge_base.confidence_factors['experience'] > 0.7:
            sources.append("accumulated_experience")
            
        return sources
    
    def _format_reasoning_explanation(self, reasoning_chain: ReasoningChain) -> str:
        """Format reasoning chain into human-readable explanation"""
        
        explanation = f"🤖 AUTONOMOUS REASONING ({self.agent_type.upper()} {self.agent_id}):\n\n"
        
        explanation += "📊 OBSERVATIONS:\n"
        for i, obs in enumerate(reasoning_chain.observations[:3], 1):
            explanation += f"   {i}. {obs}\n"
        
        explanation += "\n💡 HYPOTHESES CONSIDERED:\n"
        for i, hyp in enumerate(reasoning_chain.hypotheses[:3], 1):
            explanation += f"   {i}. {hyp}\n"
        
        explanation += f"\n🎯 CONCLUSION: {reasoning_chain.conclusion}\n"
        explanation += f"🔍 CONFIDENCE: {reasoning_chain.confidence:.1%}\n"
        
        if reasoning_chain.alternatives_considered:
            explanation += f"\n🔄 ALTERNATIVES CONSIDERED: {', '.join(reasoning_chain.alternatives_considered[:2])}\n"
        
        explanation += f"\n⚠️ RISK ASSESSMENT: Detection:{reasoning_chain.risk_assessment.get('detection_risk', 0):.1%}, "
        explanation += f"Failure:{reasoning_chain.risk_assessment.get('failure_risk', 0):.1%}\n"
        
        return explanation

# Factory function to create autonomous agents
def create_autonomous_agent(agent_id: str, agent_type: str, specialization: str, logger: ArchangelLogger) -> AutonomousAgent:
    """Create a new autonomous agent"""
    return AutonomousAgent(agent_id, agent_type, specialization, logger)