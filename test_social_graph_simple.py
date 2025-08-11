#!/usr/bin/env python3
"""
Simple test script for Social Graph functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.social_graph import (
    SocialGraphManager, SocialGraphVisualizer, Interaction,
    InteractionType, create_sample_social_graph
)


def test_basic_functionality():
    """Test basic social graph functionality"""
    print("Testing basic social graph functionality...")
    
    # Create social graph
    sg = SocialGraphManager()
    
    # Add agents
    sg.add_agent("agent_1", team="red", role="recon")
    sg.add_agent("agent_2", team="red", role="exploit")
    sg.add_agent("agent_3", team="blue", role="soc")
    
    print(f"✅ Added {len(sg.graph.nodes())} agents")
    
    # Record interactions
    interaction = Interaction(
        source_agent="agent_1",
        target_agent="agent_2",
        interaction_type=InteractionType.INFORMATION_SHARING,
        success=True,
        value=0.8
    )
    
    sg.record_interaction(interaction)
    print("✅ Recorded interaction")
    
    # Check trust score
    trust = sg.get_trust_score("agent_1", "agent_2")
    assert trust is not None
    assert trust.interaction_count == 1
    print(f"✅ Trust score: {trust.trust_value:.3f}")
    
    # Calculate influence
    influence = sg.calculate_influence_metrics()
    assert len(influence) == 3
    print("✅ Calculated influence metrics")
    
    # Test collaboration analysis
    collaboration = sg.analyze_collaboration_patterns()
    print(f"✅ Found {len(collaboration)} collaboration patterns")
    
    # Test network position
    position = sg.get_agent_network_position("agent_1")
    assert position["agent_id"] == "agent_1"
    print("✅ Retrieved network position")
    
    # Test visualization
    visualizer = SocialGraphVisualizer(sg)
    summary = visualizer.generate_network_summary()
    assert "network_overview" in summary
    print("✅ Generated network summary")
    
    print("🎉 All basic tests passed!")
    return True


def test_sample_graph():
    """Test the sample graph creation"""
    print("\nTesting sample graph creation...")
    
    sg = create_sample_social_graph()
    
    # Verify sample graph
    assert len(sg.graph.nodes()) > 0
    assert len(sg.interactions) > 0
    print(f"✅ Sample graph: {len(sg.graph.nodes())} nodes, {len(sg.interactions)} interactions")
    
    # Test influence calculation
    influence = sg.calculate_influence_metrics()
    print("✅ Influence metrics calculated")
    
    # Test visualization
    visualizer = SocialGraphVisualizer(sg)
    json_data = visualizer.export_for_visualization("json")
    assert len(json_data) > 0
    print("✅ Visualization data exported")
    
    print("🎉 Sample graph tests passed!")
    return True


def test_trust_evolution():
    """Test trust score evolution"""
    print("\nTesting trust evolution...")
    
    sg = SocialGraphManager()
    sg.add_agent("agent_a", team="test")
    sg.add_agent("agent_b", team="test")
    
    # Record successful interaction
    interaction1 = Interaction(
        source_agent="agent_a",
        target_agent="agent_b",
        interaction_type=InteractionType.COLLABORATION,
        success=True
    )
    sg.record_interaction(interaction1)
    
    trust1 = sg.get_trust_score("agent_a", "agent_b").trust_value
    
    # Record another successful interaction
    interaction2 = Interaction(
        source_agent="agent_a",
        target_agent="agent_b",
        interaction_type=InteractionType.SUPPORT,
        success=True
    )
    sg.record_interaction(interaction2)
    
    trust2 = sg.get_trust_score("agent_a", "agent_b").trust_value
    
    print(f"✅ Trust evolution: {trust1:.3f} → {trust2:.3f}")
    
    # Record failed interaction
    interaction3 = Interaction(
        source_agent="agent_a",
        target_agent="agent_b",
        interaction_type=InteractionType.CONFLICT,
        success=False
    )
    sg.record_interaction(interaction3)
    
    trust3 = sg.get_trust_score("agent_a", "agent_b").trust_value
    print(f"✅ After conflict: {trust3:.3f}")
    
    print("🎉 Trust evolution tests passed!")
    return True


def main():
    """Run all simple tests"""
    print("🧪 Running Social Graph Simple Tests")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_sample_graph()
        test_trust_evolution()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)