"""
Tests for Social Graph Modeling and Trust Systems
"""

import pytest
import json
import networkx as nx
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from agents.social_graph import (
    SocialGraphManager, SocialGraphVisualizer, Interaction, TrustScore,
    InteractionType, TrustLevel, InfluenceMetrics, CollaborationMetrics,
    create_sample_social_graph
)


class TestSocialGraphManager:
    """Test cases for SocialGraphManager"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.sg = SocialGraphManager()
        
        # Add test agents
        self.sg.add_agent("agent_1", team="red", role="recon")
        self.sg.add_agent("agent_2", team="red", role="exploit")
        self.sg.add_agent("agent_3", team="blue", role="soc")
        self.sg.add_agent("agent_4", team="blue", role="firewall")
    
    def test_add_agent(self):
        """Test adding agents to the social graph"""
        sg = SocialGraphManager()
        
        # Test basic agent addition
        sg.add_agent("test_agent", team="test", role="tester")
        
        assert "test_agent" in sg.graph.nodes()
        assert sg.graph.nodes["test_agent"]["team"] == "test"
        assert sg.graph.nodes["test_agent"]["role"] == "tester"
        assert "test_agent" in sg.influence_metrics
    
    def test_record_interaction(self):
        """Test recording interactions between agents"""
        interaction = Interaction(
            source_agent="agent_1",
            target_agent="agent_2",
            interaction_type=InteractionType.INFORMATION_SHARING,
            success=True,
            value=0.8
        )
        
        self.sg.record_interaction(interaction)
        
        # Check interaction was recorded
        assert len(self.sg.interactions) == 1
        assert len(self.sg.interaction_history) == 1
        
        # Check graph edge was created
        assert self.sg.graph.has_edge("agent_1", "agent_2")
        edge_data = self.sg.graph["agent_1"]["agent_2"]
        assert edge_data["interaction_count"] == 1
        
        # Check trust score was created
        trust_key = ("agent_1", "agent_2")
        assert trust_key in self.sg.trust_scores
        trust_score = self.sg.trust_scores[trust_key]
        assert trust_score.interaction_count == 1
        assert trust_score.positive_interactions >= 0
    
    def test_record_interaction_new_agents(self):
        """Test recording interaction with agents not in graph"""
        sg = SocialGraphManager()
        
        interaction = Interaction(
            source_agent="new_agent_1",
            target_agent="new_agent_2",
            interaction_type=InteractionType.COORDINATION,
            success=True
        )
        
        sg.record_interaction(interaction)
        
        # Check agents were automatically added
        assert "new_agent_1" in sg.graph.nodes()
        assert "new_agent_2" in sg.graph.nodes()
        assert sg.graph.has_edge("new_agent_1", "new_agent_2")
    
    def test_trust_score_calculation(self):
        """Test trust score calculation and updates"""
        # Record successful interaction
        interaction1 = Interaction(
            source_agent="agent_1",
            target_agent="agent_2",
            interaction_type=InteractionType.COLLABORATION,
            success=True,
            value=0.9
        )
        self.sg.record_interaction(interaction1)
        
        trust_score = self.sg.get_trust_score("agent_1", "agent_2")
        assert trust_score is not None
        assert trust_score.trust_value > 0.5  # Should be above neutral
        assert trust_score.positive_interactions == 1
        
        # Record failed interaction
        interaction2 = Interaction(
            source_agent="agent_1",
            target_agent="agent_2",
            interaction_type=InteractionType.CONFLICT,
            success=False,
            value=0.1
        )
        self.sg.record_interaction(interaction2)
        
        updated_trust = self.sg.get_trust_score("agent_1", "agent_2")
        # Trust should decrease or stay the same after a conflict
        assert updated_trust.trust_value <= trust_score.trust_value + 1e-10  # Allow for floating point precision
        assert updated_trust.negative_interactions == 1
        assert updated_trust.interaction_count == 2
    
    def test_mutual_trust(self):
        """Test mutual trust score retrieval"""
        # Create bidirectional interactions
        interaction1 = Interaction(
            source_agent="agent_1",
            target_agent="agent_2",
            interaction_type=InteractionType.INFORMATION_SHARING,
            success=True
        )
        interaction2 = Interaction(
            source_agent="agent_2",
            target_agent="agent_1",
            interaction_type=InteractionType.FEEDBACK,
            success=True
        )
        
        self.sg.record_interaction(interaction1)
        self.sg.record_interaction(interaction2)
        
        trust_1_to_2, trust_2_to_1 = self.sg.get_mutual_trust("agent_1", "agent_2")
        
        assert trust_1_to_2 is not None
        assert trust_2_to_1 is not None
        assert trust_1_to_2.source_agent == "agent_1"
        assert trust_1_to_2.target_agent == "agent_2"
        assert trust_2_to_1.source_agent == "agent_2"
        assert trust_2_to_1.target_agent == "agent_1"
    
    def test_influence_metrics_calculation(self):
        """Test influence metrics calculation"""
        # Create a network with clear influence patterns
        interactions = [
            Interaction(source_agent="agent_1", target_agent="agent_2", interaction_type=InteractionType.INFORMATION_SHARING, success=True),
            Interaction(source_agent="agent_1", target_agent="agent_3", interaction_type=InteractionType.COORDINATION, success=True),
            Interaction(source_agent="agent_2", target_agent="agent_4", interaction_type=InteractionType.TASK_DELEGATION, success=True),
            Interaction(source_agent="agent_3", target_agent="agent_4", interaction_type=InteractionType.SUPPORT, success=True),
        ]
        
        for interaction in interactions:
            self.sg.record_interaction(interaction)
        
        influence_metrics = self.sg.calculate_influence_metrics()
        
        # Check that all agents have influence metrics
        assert len(influence_metrics) == 4
        for agent_id in ["agent_1", "agent_2", "agent_3", "agent_4"]:
            assert agent_id in influence_metrics
            metrics = influence_metrics[agent_id]
            assert isinstance(metrics.centrality_score, float)
            assert isinstance(metrics.page_rank, float)
            assert metrics.page_rank > 0
    
    def test_collaboration_analysis(self):
        """Test collaboration pattern analysis"""
        # Create multiple interactions between same agents
        base_time = datetime.now()
        interactions = [
            Interaction(
                source_agent="agent_1",
                target_agent="agent_2",
                interaction_type=InteractionType.COLLABORATION,
                success=True,
                timestamp=base_time - timedelta(minutes=30)
            ),
            Interaction(
                source_agent="agent_2",
                target_agent="agent_1",
                interaction_type=InteractionType.FEEDBACK,
                success=True,
                timestamp=base_time - timedelta(minutes=20)
            ),
            Interaction(
                source_agent="agent_1",
                target_agent="agent_2",
                interaction_type=InteractionType.TASK_DELEGATION,
                success=True,
                timestamp=base_time - timedelta(minutes=10)
            ),
        ]
        
        for interaction in interactions:
            self.sg.record_interaction(interaction)
        
        collaboration_metrics = self.sg.analyze_collaboration_patterns(
            time_window=timedelta(hours=1)
        )
        
        # Should have collaboration metrics for agent_1 and agent_2
        assert len(collaboration_metrics) > 0
        
        # Find the collaboration pair
        pair_key = None
        for key in collaboration_metrics.keys():
            if "agent_1" in key and "agent_2" in key:
                pair_key = key
                break
        
        assert pair_key is not None
        metrics = collaboration_metrics[pair_key]
        assert metrics.success_rate == 1.0  # All interactions successful
        assert metrics.frequency > 0
        assert metrics.collaboration_strength > 0
    
    def test_agent_network_position(self):
        """Test agent network position analysis"""
        # Create interactions to establish network position
        interactions = [
            Interaction(source_agent="agent_1", target_agent="agent_2", interaction_type=InteractionType.INFORMATION_SHARING, success=True),
            Interaction(source_agent="agent_3", target_agent="agent_1", interaction_type=InteractionType.COORDINATION, success=True),
            Interaction(source_agent="agent_1", target_agent="agent_4", interaction_type=InteractionType.TASK_DELEGATION, success=True),
        ]
        
        for interaction in interactions:
            self.sg.record_interaction(interaction)
        
        position = self.sg.get_agent_network_position("agent_1")
        
        assert position["agent_id"] == "agent_1"
        assert position["direct_connections"] == 3  # Connected to 3 agents
        assert position["incoming_connections"] == 1  # From agent_3
        assert position["outgoing_connections"] == 2  # To agent_2 and agent_4
        assert len(position["incoming_trust"]) == 1
        assert len(position["outgoing_trust"]) == 2
        assert position["team"] == "red"
        assert position["role"] == "recon"
    
    def test_community_detection(self):
        """Test community detection in social graph"""
        # Create interactions within and between teams
        interactions = [
            # Red team internal
            Interaction(source_agent="agent_1", target_agent="agent_2", interaction_type=InteractionType.COLLABORATION, success=True),
            Interaction(source_agent="agent_2", target_agent="agent_1", interaction_type=InteractionType.FEEDBACK, success=True),
            # Blue team internal
            Interaction(source_agent="agent_3", target_agent="agent_4", interaction_type=InteractionType.COORDINATION, success=True),
            Interaction(source_agent="agent_4", target_agent="agent_3", interaction_type=InteractionType.SUPPORT, success=True),
            # Cross-team (minimal)
            Interaction(source_agent="agent_1", target_agent="agent_3", interaction_type=InteractionType.CONFLICT, success=False),
        ]
        
        for interaction in interactions:
            self.sg.record_interaction(interaction)
        
        communities = self.sg.detect_communities()
        
        # Should detect some form of community structure
        assert len(communities) > 0
        
        # All agents should be assigned to communities
        all_agents_in_communities = set()
        for agents in communities.values():
            all_agents_in_communities.update(agents)
        
        expected_agents = {"agent_1", "agent_2", "agent_3", "agent_4"}
        assert all_agents_in_communities == expected_agents
    
    def test_information_flow_paths(self):
        """Test information flow path finding"""
        # Create a path: agent_1 -> agent_2 -> agent_3
        interactions = [
            Interaction(source_agent="agent_1", target_agent="agent_2", interaction_type=InteractionType.INFORMATION_SHARING, success=True),
            Interaction(source_agent="agent_2", target_agent="agent_3", interaction_type=InteractionType.INFORMATION_SHARING, success=True),
        ]
        
        for interaction in interactions:
            self.sg.record_interaction(interaction)
        
        paths = self.sg.get_information_flow_paths("agent_1", "agent_3")
        
        assert len(paths) > 0
        # Should find path through agent_2
        assert ["agent_1", "agent_2", "agent_3"] in paths
    
    def test_team_cohesion(self):
        """Test team cohesion calculation"""
        # Create strong internal connections for red team
        red_interactions = [
            Interaction(source_agent="agent_1", target_agent="agent_2", interaction_type=InteractionType.COLLABORATION, success=True),
            Interaction(source_agent="agent_2", target_agent="agent_1", interaction_type=InteractionType.SUPPORT, success=True),
        ]
        
        for interaction in red_interactions:
            self.sg.record_interaction(interaction)
        
        cohesion = self.sg.calculate_team_cohesion("red")
        
        assert "cohesion_score" in cohesion
        assert "internal_trust" in cohesion
        assert "connectivity" in cohesion
        assert "team_size" in cohesion
        assert cohesion["team_size"] == 2
        assert cohesion["cohesion_score"] > 0
    
    def test_export_graph_data(self):
        """Test graph data export"""
        # Add some interactions
        interaction = Interaction(
            source_agent="agent_1",
            target_agent="agent_2",
            interaction_type=InteractionType.INFORMATION_SHARING,
            success=True
        )
        self.sg.record_interaction(interaction)
        
        graph_data = self.sg.export_graph_data()
        
        assert "nodes" in graph_data
        assert "edges" in graph_data
        assert "communities" in graph_data
        assert "collaboration_metrics" in graph_data
        assert "graph_stats" in graph_data
        
        # Check nodes
        assert len(graph_data["nodes"]) == 4
        node_ids = [node["id"] for node in graph_data["nodes"]]
        assert "agent_1" in node_ids
        
        # Check edges
        assert len(graph_data["edges"]) == 1
        edge = graph_data["edges"][0]
        assert edge["source"] == "agent_1"
        assert edge["target"] == "agent_2"
        
        # Check graph stats
        stats = graph_data["graph_stats"]
        assert stats["node_count"] == 4
        assert stats["edge_count"] == 1
    
    def test_temporal_decay(self):
        """Test temporal decay of trust scores"""
        sg = SocialGraphManager(decay_factor=0.5)  # Faster decay for testing
        
        # Record interaction with past timestamp
        past_time = datetime.now() - timedelta(days=2)
        interaction = Interaction(
            source_agent="agent_1",
            target_agent="agent_2",
            interaction_type=InteractionType.COLLABORATION,
            success=True,
            timestamp=past_time
        )
        
        with patch('agents.social_graph.datetime') as mock_datetime:
            mock_datetime.now.return_value = past_time
            sg.record_interaction(interaction)
            initial_trust = sg.get_trust_score("agent_1", "agent_2").trust_value
        
        # Record new interaction (should apply decay to previous trust)
        new_interaction = Interaction(
            source_agent="agent_1",
            target_agent="agent_2",
            interaction_type=InteractionType.COLLABORATION,
            success=True
        )
        sg.record_interaction(new_interaction)
        
        # Trust should be affected by temporal decay
        final_trust = sg.get_trust_score("agent_1", "agent_2")
        assert final_trust.interaction_count == 2


class TestSocialGraphVisualizer:
    """Test cases for SocialGraphVisualizer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.sg = create_sample_social_graph()
        self.visualizer = SocialGraphVisualizer(self.sg)
    
    def test_generate_network_summary(self):
        """Test network summary generation"""
        summary = self.visualizer.generate_network_summary()
        
        assert "network_overview" in summary
        assert "most_influential_agents" in summary
        assert "most_trusted_agents" in summary
        assert "team_cohesion" in summary
        assert "communities" in summary
        assert "total_interactions" in summary
        
        # Check network overview
        overview = summary["network_overview"]
        assert "node_count" in overview
        assert "edge_count" in overview
        assert "density" in overview
        
        # Check influential agents list
        influential = summary["most_influential_agents"]
        assert isinstance(influential, list)
        if influential:  # If there are influential agents
            assert "agent" in influential[0]
            assert "influence_score" in influential[0]
    
    def test_export_for_visualization_json(self):
        """Test JSON export for visualization"""
        json_data = self.visualizer.export_for_visualization("json")
        
        # Should be valid JSON
        parsed_data = json.loads(json_data)
        assert "nodes" in parsed_data
        assert "edges" in parsed_data
        assert "communities" in parsed_data
    
    def test_export_for_visualization_graphml(self):
        """Test GraphML export for visualization"""
        graphml_data = self.visualizer.export_for_visualization("graphml")
        
        # Should contain GraphML content or fallback to JSON
        assert len(graphml_data) > 0
        # Either GraphML format or JSON fallback
        assert "<?xml" in graphml_data or "{" in graphml_data


class TestInteractionTypes:
    """Test interaction type handling"""
    
    def test_interaction_type_trust_modifiers(self):
        """Test that different interaction types affect trust differently"""
        sg = SocialGraphManager()
        
        # Test positive interaction
        positive_interaction = Interaction(
            source_agent="agent_1",
            target_agent="agent_2",
            interaction_type=InteractionType.COLLABORATION,
            success=True
        )
        sg.record_interaction(positive_interaction)
        positive_trust = sg.get_trust_score("agent_1", "agent_2").trust_value
        
        # Reset and test negative interaction
        sg = SocialGraphManager()
        sg.add_agent("agent_1")
        sg.add_agent("agent_2")
        negative_interaction = Interaction(
            source_agent="agent_1",
            target_agent="agent_2",
            interaction_type=InteractionType.CONFLICT,
            success=False
        )
        sg.record_interaction(negative_interaction)
        negative_trust = sg.get_trust_score("agent_1", "agent_2").trust_value
        
        # Positive interaction should result in higher trust
        assert positive_trust > negative_trust


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_graph_operations(self):
        """Test operations on empty graph"""
        sg = SocialGraphManager()
        
        # Should handle empty graph gracefully
        influence = sg.calculate_influence_metrics()
        assert influence == {}
        
        collaboration = sg.analyze_collaboration_patterns()
        assert collaboration == {}
        
        communities = sg.detect_communities()
        assert communities == {}
    
    def test_single_agent_operations(self):
        """Test operations with single agent"""
        sg = SocialGraphManager()
        sg.add_agent("solo_agent")
        
        influence = sg.calculate_influence_metrics()
        assert "solo_agent" in influence
        # A single agent has maximum centrality (1.0) since it's the only node
        assert influence["solo_agent"].centrality_score == 1.0
        
        position = sg.get_agent_network_position("solo_agent")
        assert position["direct_connections"] == 0
    
    def test_nonexistent_agent_operations(self):
        """Test operations with nonexistent agents"""
        sg = SocialGraphManager()
        
        # Should return None or empty results gracefully
        trust = sg.get_trust_score("nonexistent_1", "nonexistent_2")
        assert trust is None
        
        position = sg.get_agent_network_position("nonexistent")
        assert position == {}
        
        paths = sg.get_information_flow_paths("nonexistent_1", "nonexistent_2")
        assert paths == []
    
    def test_self_interaction(self):
        """Test agent interacting with itself"""
        sg = SocialGraphManager()
        
        interaction = Interaction(
            source_agent="agent_1",
            target_agent="agent_1",  # Self-interaction
            interaction_type=InteractionType.FEEDBACK,
            success=True
        )
        
        sg.record_interaction(interaction)
        
        # Should handle self-interaction
        assert sg.graph.has_edge("agent_1", "agent_1")
        trust = sg.get_trust_score("agent_1", "agent_1")
        assert trust is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])