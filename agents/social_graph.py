"""
Social Graph Modeling and Trust Systems

This module implements dynamic social graph tracking for agent relationships,
trust scoring, influence pattern analysis, and collaboration metrics.
"""

import json
import logging
import networkx as nx
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import uuid4

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractionType(Enum):
    """Types of agent interactions"""
    INFORMATION_SHARING = "information_sharing"
    COORDINATION = "coordination"
    RESOURCE_REQUEST = "resource_request"
    TASK_DELEGATION = "task_delegation"
    FEEDBACK = "feedback"
    CONFLICT = "conflict"
    COLLABORATION = "collaboration"
    SUPPORT = "support"


class TrustLevel(Enum):
    """Trust levels between agents"""
    VERY_LOW = 0.0
    LOW = 0.25
    MEDIUM = 0.5
    HIGH = 0.75
    VERY_HIGH = 1.0


@dataclass
class Interaction:
    """Represents an interaction between two agents"""
    interaction_id: str = field(default_factory=lambda: str(uuid4()))
    source_agent: str = ""
    target_agent: str = ""
    interaction_type: InteractionType = InteractionType.INFORMATION_SHARING
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    value: float = 0.0  # Value/importance of the interaction
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrustScore:
    """Trust score between two agents"""
    source_agent: str = ""
    target_agent: str = ""
    trust_value: float = 0.5  # Initial neutral trust
    confidence: float = 0.0  # Confidence in the trust score
    last_updated: datetime = field(default_factory=datetime.now)
    interaction_count: int = 0
    positive_interactions: int = 0
    negative_interactions: int = 0


@dataclass
class InfluenceMetrics:
    """Influence metrics for an agent"""
    agent_id: str = ""
    centrality_score: float = 0.0
    betweenness_centrality: float = 0.0
    closeness_centrality: float = 0.0
    eigenvector_centrality: float = 0.0
    page_rank: float = 0.0
    influence_radius: int = 0
    information_broker_score: float = 0.0


@dataclass
class CollaborationMetrics:
    """Collaboration metrics between agents or teams"""
    participants: List[str] = field(default_factory=list)
    collaboration_strength: float = 0.0
    frequency: float = 0.0
    success_rate: float = 0.0
    average_response_time: float = 0.0
    information_flow_rate: float = 0.0
    task_completion_rate: float = 0.0


class SocialGraphManager:
    """Manages the social graph of agent relationships and interactions"""
    
    def __init__(self, decay_factor: float = 0.95, trust_threshold: float = 0.1):
        """
        Initialize the social graph manager
        
        Args:
            decay_factor: Factor for temporal decay of trust scores
            trust_threshold: Minimum threshold for trust score updates
        """
        self.graph = nx.DiGraph()  # Directed graph for asymmetric relationships
        self.interactions: List[Interaction] = []
        self.trust_scores: Dict[Tuple[str, str], TrustScore] = {}
        self.influence_metrics: Dict[str, InfluenceMetrics] = {}
        self.collaboration_metrics: Dict[str, CollaborationMetrics] = {}
        self.decay_factor = decay_factor
        self.trust_threshold = trust_threshold
        self.interaction_history = deque(maxlen=10000)  # Circular buffer
        
        logger.info("SocialGraphManager initialized")
    
    def add_agent(self, agent_id: str, team: str = "", role: str = "", 
                  attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an agent to the social graph"""
        if attributes is None:
            attributes = {}
            
        self.graph.add_node(agent_id, 
                           team=team, 
                           role=role, 
                           **attributes)
        
        # Initialize influence metrics
        self.influence_metrics[agent_id] = InfluenceMetrics(agent_id=agent_id)
        
        logger.info(f"Added agent {agent_id} to social graph")
    
    def record_interaction(self, interaction: Interaction) -> None:
        """Record an interaction between two agents"""
        # Ensure both agents exist in the graph
        if interaction.source_agent not in self.graph:
            self.add_agent(interaction.source_agent)
        if interaction.target_agent not in self.graph:
            self.add_agent(interaction.target_agent)
        
        # Add interaction to history
        self.interactions.append(interaction)
        self.interaction_history.append(interaction)
        
        # Update or create edge in graph
        if self.graph.has_edge(interaction.source_agent, interaction.target_agent):
            edge_data = self.graph[interaction.source_agent][interaction.target_agent]
            edge_data['interaction_count'] = edge_data.get('interaction_count', 0) + 1
            edge_data['last_interaction'] = interaction.timestamp
        else:
            self.graph.add_edge(interaction.source_agent, interaction.target_agent,
                              interaction_count=1,
                              last_interaction=interaction.timestamp)
        
        # Update trust scores
        self._update_trust_score(interaction)
        
        logger.debug(f"Recorded interaction: {interaction.source_agent} -> {interaction.target_agent}")
    
    def _update_trust_score(self, interaction: Interaction) -> None:
        """Update trust score based on interaction"""
        key = (interaction.source_agent, interaction.target_agent)
        
        if key not in self.trust_scores:
            self.trust_scores[key] = TrustScore(
                source_agent=interaction.source_agent,
                target_agent=interaction.target_agent
            )
        
        trust_score = self.trust_scores[key]
        trust_score.interaction_count += 1
        trust_score.last_updated = interaction.timestamp
        
        # Calculate trust update based on interaction success and type
        trust_delta = self._calculate_trust_delta(interaction)
        
        # Apply temporal decay
        time_decay = self._calculate_temporal_decay(trust_score.last_updated)
        trust_score.trust_value *= time_decay
        
        # Update trust value with weighted average
        weight = min(1.0, trust_score.interaction_count / 10.0)  # Stabilize after 10 interactions
        old_trust = trust_score.trust_value
        trust_score.trust_value = (1 - weight) * old_trust + weight * trust_delta
        trust_score.trust_value = max(0.0, min(1.0, trust_score.trust_value))
        
        # Update confidence
        trust_score.confidence = min(1.0, trust_score.interaction_count / 20.0)
        
        # Track positive/negative interactions
        if interaction.success and trust_delta > 0.5:
            trust_score.positive_interactions += 1
        elif not interaction.success or trust_delta < 0.5:
            trust_score.negative_interactions += 1
    
    def _calculate_trust_delta(self, interaction: Interaction) -> float:
        """Calculate trust change based on interaction"""
        base_trust = 0.6 if interaction.success else 0.3
        
        # Adjust based on interaction type
        type_modifiers = {
            InteractionType.INFORMATION_SHARING: 0.1,
            InteractionType.COORDINATION: 0.15,
            InteractionType.RESOURCE_REQUEST: 0.05,
            InteractionType.TASK_DELEGATION: 0.2,
            InteractionType.FEEDBACK: 0.1,
            InteractionType.CONFLICT: -0.3,
            InteractionType.COLLABORATION: 0.25,
            InteractionType.SUPPORT: 0.2
        }
        
        modifier = type_modifiers.get(interaction.interaction_type, 0.0)
        return max(0.0, min(1.0, base_trust + modifier))
    
    def _calculate_temporal_decay(self, last_update: datetime) -> float:
        """Calculate temporal decay factor"""
        time_diff = datetime.now() - last_update
        hours_passed = time_diff.total_seconds() / 3600
        return self.decay_factor ** (hours_passed / 24)  # Daily decay
    
    def get_trust_score(self, source_agent: str, target_agent: str) -> Optional[TrustScore]:
        """Get trust score between two agents"""
        key = (source_agent, target_agent)
        return self.trust_scores.get(key)
    
    def get_mutual_trust(self, agent1: str, agent2: str) -> Tuple[Optional[TrustScore], Optional[TrustScore]]:
        """Get mutual trust scores between two agents"""
        trust_1_to_2 = self.get_trust_score(agent1, agent2)
        trust_2_to_1 = self.get_trust_score(agent2, agent1)
        return trust_1_to_2, trust_2_to_1
    
    def calculate_influence_metrics(self) -> Dict[str, InfluenceMetrics]:
        """Calculate influence metrics for all agents"""
        if len(self.graph.nodes()) == 0:
            return {}
        
        # Calculate centrality measures
        try:
            centrality = nx.degree_centrality(self.graph)
            betweenness = nx.betweenness_centrality(self.graph)
            closeness = nx.closeness_centrality(self.graph)
            eigenvector = nx.eigenvector_centrality(self.graph, max_iter=1000)
            pagerank = nx.pagerank(self.graph)
        except (nx.NetworkXError, nx.PowerIterationFailedConvergence):
            # Fallback for disconnected or problematic graphs
            centrality = {node: 0.0 for node in self.graph.nodes()}
            betweenness = {node: 0.0 for node in self.graph.nodes()}
            closeness = {node: 0.0 for node in self.graph.nodes()}
            eigenvector = {node: 0.0 for node in self.graph.nodes()}
            pagerank = {node: 1.0/len(self.graph.nodes()) for node in self.graph.nodes()}
        
        # Update influence metrics for each agent
        for agent_id in self.graph.nodes():
            metrics = self.influence_metrics.get(agent_id, InfluenceMetrics(agent_id=agent_id))
            
            metrics.centrality_score = centrality.get(agent_id, 0.0)
            metrics.betweenness_centrality = betweenness.get(agent_id, 0.0)
            metrics.closeness_centrality = closeness.get(agent_id, 0.0)
            metrics.eigenvector_centrality = eigenvector.get(agent_id, 0.0)
            metrics.page_rank = pagerank.get(agent_id, 0.0)
            
            # Calculate influence radius (max distance to reach other nodes)
            try:
                if nx.is_strongly_connected(self.graph):
                    distances = nx.single_source_shortest_path_length(self.graph, agent_id)
                    metrics.influence_radius = max(distances.values()) if distances else 0
                else:
                    metrics.influence_radius = 0
            except nx.NetworkXError:
                metrics.influence_radius = 0
            
            # Calculate information broker score (based on betweenness and connections)
            metrics.information_broker_score = (
                metrics.betweenness_centrality * 0.7 + 
                metrics.centrality_score * 0.3
            )
            
            self.influence_metrics[agent_id] = metrics
        
        return self.influence_metrics
    
    def analyze_collaboration_patterns(self, time_window: timedelta = timedelta(hours=24)) -> Dict[str, CollaborationMetrics]:
        """Analyze collaboration patterns within a time window"""
        cutoff_time = datetime.now() - time_window
        recent_interactions = [
            interaction for interaction in self.interactions
            if interaction.timestamp >= cutoff_time
        ]
        
        # Group interactions by participant pairs
        collaboration_groups = defaultdict(list)
        for interaction in recent_interactions:
            key = tuple(sorted([interaction.source_agent, interaction.target_agent]))
            collaboration_groups[key].append(interaction)
        
        # Calculate metrics for each collaboration pair
        collaboration_metrics = {}
        for participants, interactions in collaboration_groups.items():
            if len(interactions) < 2:  # Need at least 2 interactions for meaningful metrics
                continue
            
            metrics = CollaborationMetrics(participants=list(participants))
            
            # Calculate collaboration strength (based on interaction frequency and success)
            successful_interactions = sum(1 for i in interactions if i.success)
            metrics.success_rate = successful_interactions / len(interactions)
            metrics.frequency = len(interactions) / time_window.total_seconds() * 3600  # per hour
            metrics.collaboration_strength = metrics.success_rate * min(1.0, metrics.frequency / 5.0)
            
            # Calculate average response time (simplified)
            response_times = []
            sorted_interactions = sorted(interactions, key=lambda x: x.timestamp)
            for i in range(1, len(sorted_interactions)):
                time_diff = (sorted_interactions[i].timestamp - sorted_interactions[i-1].timestamp).total_seconds()
                if time_diff < 3600:  # Only consider responses within an hour
                    response_times.append(time_diff)
            
            metrics.average_response_time = np.mean(response_times) if response_times else 0.0
            
            # Information flow rate (interactions per hour)
            metrics.information_flow_rate = metrics.frequency
            
            # Task completion rate (based on successful task delegations)
            task_interactions = [i for i in interactions if i.interaction_type == InteractionType.TASK_DELEGATION]
            if task_interactions:
                successful_tasks = sum(1 for i in task_interactions if i.success)
                metrics.task_completion_rate = successful_tasks / len(task_interactions)
            else:
                metrics.task_completion_rate = 0.0
            
            key_str = f"{participants[0]}-{participants[1]}"
            collaboration_metrics[key_str] = metrics
            self.collaboration_metrics[key_str] = metrics
        
        return collaboration_metrics
    
    def get_agent_network_position(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed network position information for an agent"""
        if agent_id not in self.graph:
            return {}
        
        # Get direct connections
        predecessors = list(self.graph.predecessors(agent_id))
        successors = list(self.graph.successors(agent_id))
        
        # Calculate unique connections (avoid double counting bidirectional connections)
        all_connections = set(predecessors + successors)
        all_connections.discard(agent_id)  # Remove self if present
        
        # Get trust relationships
        incoming_trust = []
        outgoing_trust = []
        
        for pred in predecessors:
            trust = self.get_trust_score(pred, agent_id)
            if trust:
                incoming_trust.append({
                    'agent': pred,
                    'trust_value': trust.trust_value,
                    'confidence': trust.confidence
                })
        
        for succ in successors:
            trust = self.get_trust_score(agent_id, succ)
            if trust:
                outgoing_trust.append({
                    'agent': succ,
                    'trust_value': trust.trust_value,
                    'confidence': trust.confidence
                })
        
        # Get influence metrics
        influence = self.influence_metrics.get(agent_id, InfluenceMetrics(agent_id=agent_id))
        
        return {
            'agent_id': agent_id,
            'direct_connections': len(all_connections),
            'incoming_connections': len(predecessors),
            'outgoing_connections': len(successors),
            'incoming_trust': incoming_trust,
            'outgoing_trust': outgoing_trust,
            'influence_metrics': asdict(influence),
            'team': self.graph.nodes[agent_id].get('team', ''),
            'role': self.graph.nodes[agent_id].get('role', '')
        }
    
    def detect_communities(self) -> Dict[str, List[str]]:
        """Detect communities/clusters in the social graph"""
        if len(self.graph.nodes()) < 3:
            return {}
        
        try:
            # Convert to undirected graph for community detection
            undirected_graph = self.graph.to_undirected()
            
            # Use Louvain algorithm for community detection
            import community as community_louvain
            partition = community_louvain.best_partition(undirected_graph)
            
            # Group agents by community
            communities = defaultdict(list)
            for agent, community_id in partition.items():
                if isinstance(agent, str):  # Ensure we only process agent IDs (strings)
                    communities[f"community_{community_id}"].append(agent)
            
            return dict(communities)
        
        except ImportError:
            # Fallback: simple clustering based on teams
            logger.warning("python-louvain not available, using team-based clustering")
            communities = defaultdict(list)
            for agent_id in self.graph.nodes():
                team = self.graph.nodes[agent_id].get('team', 'unknown')
                communities[team].append(agent_id)
            return dict(communities)
    
    def export_graph_data(self) -> Dict[str, Any]:
        """Export graph data for visualization"""
        nodes = []
        edges = []
        
        # Export nodes
        for agent_id in self.graph.nodes():
            node_data = self.graph.nodes[agent_id].copy()
            node_data['id'] = agent_id
            
            # Add influence metrics
            influence = self.influence_metrics.get(agent_id)
            if influence:
                node_data['influence'] = asdict(influence)
            
            nodes.append(node_data)
        
        # Export edges
        for source, target in self.graph.edges():
            edge_data = self.graph[source][target].copy()
            edge_data['source'] = source
            edge_data['target'] = target
            
            # Add trust score
            trust = self.get_trust_score(source, target)
            if trust:
                edge_data['trust'] = asdict(trust)
            
            edges.append(edge_data)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'communities': self.detect_communities(),
            'collaboration_metrics': {k: asdict(v) for k, v in self.collaboration_metrics.items()},
            'graph_stats': {
                'node_count': len(self.graph.nodes()),
                'edge_count': len(self.graph.edges()),
                'density': nx.density(self.graph),
                'is_connected': nx.is_weakly_connected(self.graph)
            }
        }
    
    def get_information_flow_paths(self, source_agent: str, target_agent: str, 
                                  max_paths: int = 5) -> List[List[str]]:
        """Find information flow paths between two agents"""
        if source_agent not in self.graph or target_agent not in self.graph:
            return []
        
        try:
            # First try to find shortest paths
            if nx.has_path(self.graph, source_agent, target_agent):
                paths = list(nx.all_shortest_paths(self.graph, source_agent, target_agent))
                return paths[:max_paths]
            else:
                # If no direct paths, try through intermediaries with longer paths
                all_paths = []
                try:
                    for path in nx.all_simple_paths(self.graph, source_agent, target_agent, cutoff=3):
                        all_paths.append(path)
                        if len(all_paths) >= max_paths:
                            break
                    return all_paths
                except nx.NetworkXNoPath:
                    return []
        
        except (nx.NetworkXNoPath, nx.NetworkXError):
            return []
    
    def calculate_team_cohesion(self, team_name: str) -> Dict[str, float]:
        """Calculate cohesion metrics for a team"""
        team_agents = [
            agent_id for agent_id in self.graph.nodes()
            if self.graph.nodes[agent_id].get('team') == team_name
        ]
        
        if len(team_agents) < 2:
            return {'cohesion_score': 0.0, 'internal_trust': 0.0, 'connectivity': 0.0}
        
        # Calculate internal connectivity
        internal_edges = 0
        possible_edges = len(team_agents) * (len(team_agents) - 1)
        
        for agent1 in team_agents:
            for agent2 in team_agents:
                if agent1 != agent2 and self.graph.has_edge(agent1, agent2):
                    internal_edges += 1
        
        connectivity = internal_edges / possible_edges if possible_edges > 0 else 0.0
        
        # Calculate average internal trust
        trust_scores = []
        for agent1 in team_agents:
            for agent2 in team_agents:
                if agent1 != agent2:
                    trust = self.get_trust_score(agent1, agent2)
                    if trust:
                        trust_scores.append(trust.trust_value)
        
        internal_trust = np.mean(trust_scores) if trust_scores else 0.0
        
        # Overall cohesion score
        cohesion_score = (connectivity * 0.6 + internal_trust * 0.4)
        
        return {
            'cohesion_score': cohesion_score,
            'internal_trust': internal_trust,
            'connectivity': connectivity,
            'team_size': len(team_agents)
        }


class SocialGraphVisualizer:
    """Handles visualization and analysis of social graphs"""
    
    def __init__(self, social_graph: SocialGraphManager):
        self.social_graph = social_graph
    
    def generate_network_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive network summary"""
        graph_data = self.social_graph.export_graph_data()
        influence_metrics = self.social_graph.calculate_influence_metrics()
        
        # Find most influential agents
        most_influential = sorted(
            influence_metrics.items(),
            key=lambda x: x[1].page_rank,
            reverse=True
        )[:5]
        
        # Find most trusted agents
        trust_scores = []
        for trust in self.social_graph.trust_scores.values():
            trust_scores.append((trust.target_agent, trust.trust_value, trust.confidence))
        
        most_trusted = sorted(trust_scores, key=lambda x: x[1] * x[2], reverse=True)[:5]
        
        # Analyze team dynamics
        teams = set()
        for node in graph_data['nodes']:
            if 'team' in node:
                teams.add(node['team'])
        
        team_cohesion = {}
        for team in teams:
            if team:  # Skip empty team names
                team_cohesion[team] = self.social_graph.calculate_team_cohesion(team)
        
        return {
            'network_overview': graph_data['graph_stats'],
            'most_influential_agents': [
                {'agent': agent, 'influence_score': metrics.page_rank}
                for agent, metrics in most_influential
            ],
            'most_trusted_agents': [
                {'agent': agent, 'trust_score': trust, 'confidence': conf}
                for agent, trust, conf in most_trusted
            ],
            'team_cohesion': team_cohesion,
            'communities': graph_data['communities'],
            'total_interactions': len(self.social_graph.interactions)
        }
    
    def export_for_visualization(self, format_type: str = "json") -> str:
        """Export graph data in various formats for visualization tools"""
        graph_data = self.social_graph.export_graph_data()
        
        if format_type.lower() == "json":
            return json.dumps(graph_data, indent=2, default=str)
        elif format_type.lower() == "graphml":
            # Export as GraphML for tools like Gephi
            try:
                import io
                # Create a copy of the graph with datetime objects converted to strings
                graph_copy = self.social_graph.graph.copy()
                for u, v, data in graph_copy.edges(data=True):
                    for key, value in data.items():
                        if isinstance(value, datetime):
                            data[key] = str(value)
                
                buffer = io.StringIO()
                nx.write_graphml(graph_copy, buffer)
                return buffer.getvalue()
            except (ImportError, Exception) as e:
                logger.warning(f"NetworkX GraphML export failed: {e}")
                return json.dumps(graph_data, indent=2, default=str)
        else:
            return json.dumps(graph_data, indent=2, default=str)


# Example usage and testing functions
def create_sample_social_graph() -> SocialGraphManager:
    """Create a sample social graph for testing"""
    sg = SocialGraphManager()
    
    # Add agents
    sg.add_agent("red_recon_1", team="red", role="reconnaissance")
    sg.add_agent("red_exploit_1", team="red", role="exploitation")
    sg.add_agent("blue_soc_1", team="blue", role="soc_analyst")
    sg.add_agent("blue_firewall_1", team="blue", role="firewall_admin")
    
    # Add some interactions
    interactions = [
        Interaction(
            source_agent="red_recon_1",
            target_agent="red_exploit_1",
            interaction_type=InteractionType.INFORMATION_SHARING,
            success=True,
            value=0.8
        ),
        Interaction(
            source_agent="blue_soc_1",
            target_agent="blue_firewall_1",
            interaction_type=InteractionType.COORDINATION,
            success=True,
            value=0.9
        ),
        Interaction(
            source_agent="red_exploit_1",
            target_agent="red_recon_1",
            interaction_type=InteractionType.FEEDBACK,
            success=True,
            value=0.7
        )
    ]
    
    for interaction in interactions:
        sg.record_interaction(interaction)
    
    return sg


if __name__ == "__main__":
    # Demo the social graph system
    print("Creating sample social graph...")
    sg = create_sample_social_graph()
    
    print("\nCalculating influence metrics...")
    influence = sg.calculate_influence_metrics()
    for agent_id, metrics in influence.items():
        print(f"{agent_id}: PageRank={metrics.page_rank:.3f}, Centrality={metrics.centrality_score:.3f}")
    
    print("\nAnalyzing collaboration patterns...")
    collaboration = sg.analyze_collaboration_patterns()
    for pair, metrics in collaboration.items():
        print(f"{pair}: Strength={metrics.collaboration_strength:.3f}, Success Rate={metrics.success_rate:.3f}")
    
    print("\nGenerating network summary...")
    visualizer = SocialGraphVisualizer(sg)
    summary = visualizer.generate_network_summary()
    print(json.dumps(summary, indent=2, default=str))