#!/usr/bin/env python3
"""
Demo script for Social Graph Modeling and Trust Systems

This script demonstrates the capabilities of the social graph system
including relationship tracking, trust scoring, influence analysis,
and collaboration metrics.
"""

import json
import time
import random
from datetime import datetime, timedelta
from agents.social_graph import (
    SocialGraphManager, SocialGraphVisualizer, Interaction,
    InteractionType, create_sample_social_graph
)


def create_realistic_scenario():
    """Create a realistic multi-agent scenario"""
    print("🔧 Creating realistic multi-agent scenario...")
    
    sg = SocialGraphManager()
    
    # Add Red Team agents
    red_agents = [
        ("red_recon_alpha", "red", "reconnaissance"),
        ("red_exploit_beta", "red", "exploitation"),
        ("red_persist_gamma", "red", "persistence"),
        ("red_exfil_delta", "red", "exfiltration"),
        ("red_social_epsilon", "red", "social_engineering")
    ]
    
    # Add Blue Team agents
    blue_agents = [
        ("blue_soc_alpha", "blue", "soc_analyst"),
        ("blue_firewall_beta", "blue", "firewall_admin"),
        ("blue_siem_gamma", "blue", "siem_operator"),
        ("blue_incident_delta", "blue", "incident_responder"),
        ("blue_threat_epsilon", "blue", "threat_hunter")
    ]
    
    # Add all agents
    for agent_id, team, role in red_agents + blue_agents:
        sg.add_agent(agent_id, team=team, role=role)
    
    print(f"✅ Added {len(red_agents)} Red Team agents and {len(blue_agents)} Blue Team agents")
    return sg, red_agents, blue_agents


def simulate_red_team_operations(sg, red_agents):
    """Simulate Red Team coordination and operations"""
    print("\n🔴 Simulating Red Team operations...")
    
    interactions = []
    
    # Reconnaissance phase - information sharing
    interactions.extend([
        Interaction(
            source_agent="red_recon_alpha",
            target_agent="red_exploit_beta",
            interaction_type=InteractionType.INFORMATION_SHARING,
            success=True,
            value=0.9,
            context={"phase": "reconnaissance", "intel_type": "vulnerability_scan"}
        ),
        Interaction(
            source_agent="red_recon_alpha",
            target_agent="red_social_epsilon",
            interaction_type=InteractionType.INFORMATION_SHARING,
            success=True,
            value=0.8,
            context={"phase": "reconnaissance", "intel_type": "employee_profiles"}
        )
    ])
    
    # Exploitation phase - coordination
    interactions.extend([
        Interaction(
            source_agent="red_exploit_beta",
            target_agent="red_persist_gamma",
            interaction_type=InteractionType.COORDINATION,
            success=True,
            value=0.85,
            context={"phase": "exploitation", "action": "establish_foothold"}
        ),
        Interaction(
            source_agent="red_persist_gamma",
            target_agent="red_exploit_beta",
            interaction_type=InteractionType.FEEDBACK,
            success=True,
            value=0.7,
            context={"phase": "exploitation", "status": "backdoor_established"}
        )
    ])
    
    # Persistence phase - task delegation
    interactions.extend([
        Interaction(
            source_agent="red_persist_gamma",
            target_agent="red_exfil_delta",
            interaction_type=InteractionType.TASK_DELEGATION,
            success=True,
            value=0.9,
            context={"phase": "persistence", "task": "data_exfiltration"}
        ),
        Interaction(
            source_agent="red_exfil_delta",
            target_agent="red_persist_gamma",
            interaction_type=InteractionType.FEEDBACK,
            success=True,
            value=0.8,
            context={"phase": "persistence", "status": "exfiltration_complete"}
        )
    ])
    
    # Some failed interactions (realistic scenario)
    interactions.extend([
        Interaction(
            source_agent="red_social_epsilon",
            target_agent="red_exploit_beta",
            interaction_type=InteractionType.SUPPORT,
            success=False,
            value=0.3,
            context={"phase": "exploitation", "issue": "phishing_detected"}
        )
    ])
    
    # Record all interactions with slight time delays
    base_time = datetime.now() - timedelta(hours=2)
    for i, interaction in enumerate(interactions):
        interaction.timestamp = base_time + timedelta(minutes=i*5)
        sg.record_interaction(interaction)
    
    print(f"✅ Recorded {len(interactions)} Red Team interactions")


def simulate_blue_team_operations(sg, blue_agents):
    """Simulate Blue Team coordination and response"""
    print("\n🔵 Simulating Blue Team operations...")
    
    interactions = []
    
    # Detection phase - alert sharing
    interactions.extend([
        Interaction(
            source_agent="blue_soc_alpha",
            target_agent="blue_siem_gamma",
            interaction_type=InteractionType.INFORMATION_SHARING,
            success=True,
            value=0.9,
            context={"phase": "detection", "alert_type": "suspicious_network_activity"}
        ),
        Interaction(
            source_agent="blue_siem_gamma",
            target_agent="blue_threat_epsilon",
            interaction_type=InteractionType.COORDINATION,
            success=True,
            value=0.85,
            context={"phase": "detection", "action": "threat_hunting"}
        )
    ])
    
    # Response phase - incident coordination
    interactions.extend([
        Interaction(
            source_agent="blue_threat_epsilon",
            target_agent="blue_incident_delta",
            interaction_type=InteractionType.TASK_DELEGATION,
            success=True,
            value=0.9,
            context={"phase": "response", "task": "incident_containment"}
        ),
        Interaction(
            source_agent="blue_incident_delta",
            target_agent="blue_firewall_beta",
            interaction_type=InteractionType.COORDINATION,
            success=True,
            value=0.8,
            context={"phase": "response", "action": "firewall_rule_update"}
        )
    ])
    
    # Containment phase - collaborative response
    interactions.extend([
        Interaction(
            source_agent="blue_firewall_beta",
            target_agent="blue_soc_alpha",
            interaction_type=InteractionType.FEEDBACK,
            success=True,
            value=0.75,
            context={"phase": "containment", "status": "rules_deployed"}
        ),
        Interaction(
            source_agent="blue_soc_alpha",
            target_agent="blue_incident_delta",
            interaction_type=InteractionType.COLLABORATION,
            success=True,
            value=0.9,
            context={"phase": "containment", "action": "forensic_analysis"}
        )
    ])
    
    # Some coordination challenges
    interactions.extend([
        Interaction(
            source_agent="blue_siem_gamma",
            target_agent="blue_firewall_beta",
            interaction_type=InteractionType.RESOURCE_REQUEST,
            success=False,
            value=0.4,
            context={"phase": "response", "issue": "resource_unavailable"}
        )
    ])
    
    # Record all interactions
    base_time = datetime.now() - timedelta(hours=1)
    for i, interaction in enumerate(interactions):
        interaction.timestamp = base_time + timedelta(minutes=i*3)
        sg.record_interaction(interaction)
    
    print(f"✅ Recorded {len(interactions)} Blue Team interactions")


def simulate_cross_team_interactions(sg):
    """Simulate limited cross-team interactions (conflicts/detection)"""
    print("\n⚔️  Simulating cross-team interactions...")
    
    interactions = [
        # Blue team detecting Red team activities
        Interaction(
            source_agent="blue_soc_alpha",
            target_agent="red_exploit_beta",
            interaction_type=InteractionType.CONFLICT,
            success=False,  # From Red team perspective
            value=0.1,
            context={"type": "detection", "action": "exploit_blocked"}
        ),
        Interaction(
            source_agent="blue_threat_epsilon",
            target_agent="red_persist_gamma",
            interaction_type=InteractionType.CONFLICT,
            success=False,
            value=0.2,
            context={"type": "detection", "action": "persistence_disrupted"}
        )
    ]
    
    for interaction in interactions:
        sg.record_interaction(interaction)
    
    print(f"✅ Recorded {len(interactions)} cross-team interactions")


def analyze_social_dynamics(sg):
    """Analyze and display social dynamics"""
    print("\n📊 Analyzing social dynamics...")
    
    # Calculate influence metrics
    print("\n🎯 Influence Analysis:")
    influence_metrics = sg.calculate_influence_metrics()
    
    # Sort by PageRank score
    sorted_influence = sorted(
        influence_metrics.items(),
        key=lambda x: x[1].page_rank,
        reverse=True
    )
    
    print("Top 5 Most Influential Agents:")
    for i, (agent_id, metrics) in enumerate(sorted_influence[:5], 1):
        team = sg.graph.nodes[agent_id].get('team', 'unknown')
        role = sg.graph.nodes[agent_id].get('role', 'unknown')
        print(f"  {i}. {agent_id} ({team}/{role})")
        print(f"     PageRank: {metrics.page_rank:.3f}")
        print(f"     Centrality: {metrics.centrality_score:.3f}")
        print(f"     Betweenness: {metrics.betweenness_centrality:.3f}")
        print()
    
    # Analyze trust relationships
    print("🤝 Trust Analysis:")
    high_trust_pairs = []
    low_trust_pairs = []
    
    for (source, target), trust_score in sg.trust_scores.items():
        if trust_score.confidence > 0.3:  # Only consider established relationships
            if trust_score.trust_value > 0.7:
                high_trust_pairs.append((source, target, trust_score.trust_value))
            elif trust_score.trust_value < 0.4:
                low_trust_pairs.append((source, target, trust_score.trust_value))
    
    print("High Trust Relationships:")
    for source, target, trust in sorted(high_trust_pairs, key=lambda x: x[2], reverse=True)[:5]:
        print(f"  {source} → {target}: {trust:.3f}")
    
    print("\nLow Trust Relationships:")
    for source, target, trust in sorted(low_trust_pairs, key=lambda x: x[2])[:5]:
        print(f"  {source} → {target}: {trust:.3f}")
    
    # Analyze collaboration patterns
    print("\n🤝 Collaboration Analysis:")
    collaboration_metrics = sg.analyze_collaboration_patterns(timedelta(hours=3))
    
    if collaboration_metrics:
        print("Top Collaboration Pairs:")
        sorted_collab = sorted(
            collaboration_metrics.items(),
            key=lambda x: x[1].collaboration_strength,
            reverse=True
        )
        
        for pair, metrics in sorted_collab[:5]:
            print(f"  {pair}:")
            print(f"    Strength: {metrics.collaboration_strength:.3f}")
            print(f"    Success Rate: {metrics.success_rate:.3f}")
            print(f"    Frequency: {metrics.frequency:.2f}/hour")
            print()
    
    # Team cohesion analysis
    print("🏆 Team Cohesion Analysis:")
    for team in ["red", "blue"]:
        cohesion = sg.calculate_team_cohesion(team)
        print(f"  {team.upper()} Team:")
        print(f"    Cohesion Score: {cohesion['cohesion_score']:.3f}")
        print(f"    Internal Trust: {cohesion['internal_trust']:.3f}")
        print(f"    Connectivity: {cohesion['connectivity']:.3f}")
        print(f"    Team Size: {cohesion['team_size']}")
        print()


def demonstrate_information_flow(sg):
    """Demonstrate information flow analysis"""
    print("\n🌊 Information Flow Analysis:")
    
    # Find paths between key agents
    test_paths = [
        ("red_recon_alpha", "red_exfil_delta"),
        ("blue_soc_alpha", "blue_firewall_beta"),
        ("red_exploit_beta", "blue_threat_epsilon")
    ]
    
    for source, target in test_paths:
        paths = sg.get_information_flow_paths(source, target, max_paths=3)
        print(f"\nPaths from {source} to {target}:")
        if paths:
            for i, path in enumerate(paths, 1):
                print(f"  Path {i}: {' → '.join(path)}")
        else:
            print("  No direct paths found")


def demonstrate_community_detection(sg):
    """Demonstrate community detection"""
    print("\n🏘️  Community Detection:")
    
    communities = sg.detect_communities()
    
    print(f"Detected {len(communities)} communities:")
    for community_id, members in communities.items():
        print(f"\n{community_id}:")
        for member in members:
            team = sg.graph.nodes[member].get('team', 'unknown')
            role = sg.graph.nodes[member].get('role', 'unknown')
            print(f"  - {member} ({team}/{role})")


def generate_visualization_data(sg):
    """Generate data for visualization"""
    print("\n🎨 Generating Visualization Data...")
    
    visualizer = SocialGraphVisualizer(sg)
    
    # Generate comprehensive network summary
    summary = visualizer.generate_network_summary()
    
    print("Network Summary:")
    print(f"  Total Nodes: {summary['network_overview']['node_count']}")
    print(f"  Total Edges: {summary['network_overview']['edge_count']}")
    print(f"  Network Density: {summary['network_overview']['density']:.3f}")
    print(f"  Total Interactions: {summary['total_interactions']}")
    
    # Export visualization data
    viz_data = visualizer.export_for_visualization("json")
    
    # Save to file for external visualization tools
    with open("social_graph_visualization.json", "w") as f:
        f.write(viz_data)
    
    print("✅ Visualization data saved to 'social_graph_visualization.json'")
    
    return summary


def simulate_temporal_evolution(sg):
    """Simulate how relationships evolve over time"""
    print("\n⏰ Simulating Temporal Evolution...")
    
    # Record some interactions over time
    agents = list(sg.graph.nodes())
    base_time = datetime.now() - timedelta(days=7)
    
    # Simulate a week of interactions
    for day in range(7):
        day_time = base_time + timedelta(days=day)
        
        # Random interactions each day
        for _ in range(random.randint(3, 8)):
            source = random.choice(agents)
            # Prefer same-team interactions
            source_team = sg.graph.nodes[source].get('team', '')
            same_team_agents = [
                a for a in agents 
                if sg.graph.nodes[a].get('team', '') == source_team and a != source
            ]
            
            if same_team_agents and random.random() < 0.8:  # 80% same team
                target = random.choice(same_team_agents)
            else:
                target = random.choice([a for a in agents if a != source])
            
            interaction = Interaction(
                source_agent=source,
                target_agent=target,
                interaction_type=random.choice(list(InteractionType)),
                success=random.random() > 0.2,  # 80% success rate
                timestamp=day_time + timedelta(hours=random.randint(0, 23))
            )
            
            sg.record_interaction(interaction)
    
    print(f"✅ Simulated {len(sg.interactions)} total interactions over 7 days")
    
    # Show how trust has evolved
    print("\nTrust Evolution Sample:")
    trust_samples = list(sg.trust_scores.items())[:3]
    for (source, target), trust in trust_samples:
        print(f"  {source} → {target}: {trust.trust_value:.3f} "
              f"(confidence: {trust.confidence:.3f}, interactions: {trust.interaction_count})")


def main():
    """Main demo function"""
    print("🚀 Social Graph Modeling and Trust Systems Demo")
    print("=" * 60)
    
    # Create scenario
    sg, red_agents, blue_agents = create_realistic_scenario()
    
    # Simulate operations
    simulate_red_team_operations(sg, red_agents)
    simulate_blue_team_operations(sg, blue_agents)
    simulate_cross_team_interactions(sg)
    
    # Analyze dynamics
    analyze_social_dynamics(sg)
    
    # Demonstrate advanced features
    demonstrate_information_flow(sg)
    demonstrate_community_detection(sg)
    
    # Generate visualization
    summary = generate_visualization_data(sg)
    
    # Simulate temporal evolution
    simulate_temporal_evolution(sg)
    
    print("\n" + "=" * 60)
    print("🎯 Demo completed successfully!")
    print("\nKey Features Demonstrated:")
    print("✅ Dynamic relationship tracking")
    print("✅ Trust score calculation and evolution")
    print("✅ Influence pattern analysis")
    print("✅ Collaboration metrics")
    print("✅ Community detection")
    print("✅ Information flow analysis")
    print("✅ Team cohesion measurement")
    print("✅ Temporal evolution simulation")
    print("✅ Visualization data export")
    
    print(f"\n📊 Final Statistics:")
    print(f"   Agents: {len(sg.graph.nodes())}")
    print(f"   Relationships: {len(sg.graph.edges())}")
    print(f"   Trust Scores: {len(sg.trust_scores)}")
    print(f"   Total Interactions: {len(sg.interactions)}")
    
    return sg


if __name__ == "__main__":
    social_graph = main()