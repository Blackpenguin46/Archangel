#!/usr/bin/env python3
"""
Synthetic User Simulation Demo

This script demonstrates the synthetic user simulation system in action,
showing how autonomous synthetic users generate realistic background activity
to complicate Red Team reconnaissance and provide realistic enterprise behavior.
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timedelta
import signal

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules directly to avoid dependency issues
from agents.synthetic_user_config import SyntheticUserConfigManager


async def demo_synthetic_user_system():
    """Demonstrate the synthetic user simulation system"""
    
    print("🤖 Archangel Synthetic User Simulation Demo")
    print("=" * 50)
    print()
    
    # Create configuration manager
    print("📋 Setting up configuration...")
    config_manager = SyntheticUserConfigManager()
    config_manager.create_default_config()
    
    print(f"✅ Created configuration with {len(config_manager.user_profiles)} user profiles")
    print()
    
    # Show configuration summary
    print("📊 Configuration Summary:")
    summary = config_manager.get_config_summary()
    print(json.dumps(summary, indent=2))
    print()
    
    # Show user profiles
    print("👥 Configured User Profiles:")
    print("-" * 40)
    for profile in config_manager.user_profiles:
        print(f"🔹 {profile.username} ({profile.role.value})")
        print(f"   Department: {profile.department}")
        print(f"   Work Hours: {profile.work_hours[0]:02d}:00 - {profile.work_hours[1]:02d}:00")
        print(f"   Activity Level: {profile.activity_frequency}")
        print(f"   Risk Profile: {profile.risk_profile}")
        print()
    
    # Demonstrate role-based behavior differences
    print("🎭 Role-Based Behavior Patterns:")
    print("-" * 40)
    
    # Show web browsing patterns for different roles
    from agents.synthetic_users import WebBrowsingSimulator, UserRole
    
    environment_config = {
        'web_server_url': config_manager.environment_config.web_server_url,
        'user_ip': config_manager.environment_config.default_user_ip
    }
    
    # Create simulators for different roles
    admin_profile = next(p for p in config_manager.user_profiles if p.role == UserRole.ADMIN)
    dev_profile = next(p for p in config_manager.user_profiles if p.role == UserRole.DEVELOPER)
    sales_profile = next(p for p in config_manager.user_profiles if p.role == UserRole.SALES)
    
    admin_sim = WebBrowsingSimulator(admin_profile, environment_config)
    dev_sim = WebBrowsingSimulator(dev_profile, environment_config)
    sales_sim = WebBrowsingSimulator(sales_profile, environment_config)
    
    print("🔧 Admin browsing patterns:")
    admin_patterns = admin_sim.browsing_patterns[UserRole.ADMIN][:5]
    for pattern in admin_patterns:
        print(f"   • {pattern}")
    
    print("\n💻 Developer browsing patterns:")
    dev_patterns = dev_sim.browsing_patterns[UserRole.DEVELOPER][:5]
    for pattern in dev_patterns:
        print(f"   • {pattern}")
    
    print("\n💼 Sales browsing patterns:")
    sales_patterns = sales_sim.browsing_patterns[UserRole.SALES][:5]
    for pattern in sales_patterns:
        print(f"   • {pattern}")
    print()
    
    # Demonstrate activity simulation
    print("⚡ Activity Simulation Demo:")
    print("-" * 40)
    
    # Import and create a comprehensive synthetic user
    from agents.synthetic_users import ComprehensiveSyntheticUser
    
    # Use the developer profile for demo
    demo_user = ComprehensiveSyntheticUser(dev_profile, environment_config)
    
    print(f"🚀 Starting activity simulation for {dev_profile.username}...")
    
    # Simulate some activities manually for demo
    print("\n📝 Simulating login...")
    await demo_user._simulate_login()
    
    print("🌐 Simulating web browsing...")
    # Mock web browsing activities
    for i in range(3):
        demo_user.log_activity(
            demo_user.web_simulator.ActivityType.WEB_BROWSING,
            {"url": f"/api/endpoint_{i+1}", "method": "GET"},
            True,
            1.2 + i * 0.3,
            "Mozilla/5.0 (Developer Browser)"
        )
    
    print("📁 Simulating file access...")
    # Mock file access activities
    for i in range(2):
        demo_user.log_activity(
            demo_user.file_simulator.ActivityType.FILE_ACCESS,
            {"operation": "read", "file_path": f"/home/dev/project_{i+1}.py"},
            True,
            0.8 + i * 0.2
        )
    
    print("📧 Simulating email activity...")
    # Mock email activities
    demo_user.log_activity(
        demo_user.email_simulator.ActivityType.EMAIL_ACTIVITY,
        {"action": "send", "recipient": "team@company.com", "subject": "Code Review"},
        True,
        2.5
    )
    
    print("📤 Simulating logout...")
    await demo_user._simulate_logout()
    
    # Show activity summary
    print("\n📊 Activity Summary:")
    summary = demo_user.get_activity_summary()
    print(f"   Total Activities: {summary['total_activities']}")
    print(f"   Success Rate: {summary['success_rate']:.1%}")
    print(f"   Average Duration: {summary['average_duration_seconds']:.2f}s")
    print("   Activity Breakdown:")
    for activity_type, count in summary['activity_counts'].items():
        print(f"     • {activity_type}: {count}")
    print()
    
    # Show detailed activity log
    print("📋 Detailed Activity Log:")
    print("-" * 40)
    for i, activity in enumerate(demo_user.activity_log, 1):
        status = "✅" if activity.success else "❌"
        print(f"{i:2d}. {activity.timestamp.strftime('%H:%M:%S')} | "
              f"{activity.activity_type.value:15s} | {status} | "
              f"{activity.duration_seconds:.2f}s")
        if activity.details:
            detail_str = ", ".join(f"{k}: {v}" for k, v in activity.details.items())
            print(f"     Details: {detail_str}")
    print()
    
    # Demonstrate detection evasion characteristics
    print("🕵️ Detection Evasion Characteristics:")
    print("-" * 40)
    
    print("🔍 Realistic User Agents:")
    user_agents = demo_user.web_simulator.user_agents[UserRole.DEVELOPER]
    for ua in user_agents[:2]:
        print(f"   • {ua}")
    
    print("\n⏰ Realistic Timing Patterns:")
    durations = [a.duration_seconds for a in demo_user.activity_log]
    avg_duration = sum(durations) / len(durations)
    print(f"   • Average activity duration: {avg_duration:.2f}s")
    print(f"   • Duration range: {min(durations):.2f}s - {max(durations):.2f}s")
    
    print("\n🌐 Network Characteristics:")
    print(f"   • Source IP: {demo_user.activity_log[0].source_ip}")
    print(f"   • Internal network range: 192.168.20.0/24")
    
    print("\n📈 Background Noise Generation:")
    print("   • Multiple user roles generating diverse activities")
    print("   • Realistic timing and frequency patterns")
    print("   • Role-appropriate behavior patterns")
    print("   • High success rates mimicking normal users")
    print()
    
    # Show how this helps Red Team evasion
    print("🛡️ How This Helps Red Team Reconnaissance:")
    print("-" * 40)
    print("✅ Masks attacker activities in legitimate traffic")
    print("✅ Creates realistic baseline for anomaly detection")
    print("✅ Provides cover for reconnaissance activities")
    print("✅ Complicates log analysis and forensics")
    print("✅ Simulates real enterprise user behavior")
    print("✅ Tests Blue Team detection capabilities")
    print()
    
    # Configuration management demo
    print("⚙️ Configuration Management:")
    print("-" * 40)
    
    # Validate configuration
    errors = config_manager.validate_config()
    if errors:
        print("❌ Configuration errors found:")
        for error in errors:
            print(f"   • {error}")
    else:
        print("✅ Configuration is valid")
    
    # Show how to save/load configuration
    print("\n💾 Configuration Persistence:")
    print("   • Save configuration: config_manager.save_config()")
    print("   • Load configuration: config_manager.load_config()")
    print("   • Create sample config: create_sample_config_file()")
    print()
    
    # CLI usage examples
    print("🖥️ CLI Usage Examples:")
    print("-" * 40)
    print("   # Create new configuration")
    print("   python3 agents/synthetic_user_cli.py create-config myconfig.yaml")
    print()
    print("   # Initialize and validate configuration")
    print("   python3 agents/synthetic_user_cli.py init --config myconfig.yaml")
    print("   python3 agents/synthetic_user_cli.py validate")
    print()
    print("   # List configured users")
    print("   python3 agents/synthetic_user_cli.py list-users")
    print()
    print("   # Run simulation")
    print("   python3 agents/synthetic_user_cli.py run --duration 3600  # 1 hour")
    print()
    print("   # Test specific user")
    print("   python3 agents/synthetic_user_cli.py test-user dev001 --duration 60")
    print()
    print("   # View activity logs")
    print("   python3 agents/synthetic_user_cli.py logs --lines 100")
    print()
    
    print("🎯 Integration with Archangel System:")
    print("-" * 40)
    print("✅ Provides realistic background activity for Red vs Blue scenarios")
    print("✅ Complicates Red Team reconnaissance and lateral movement")
    print("✅ Tests Blue Team detection and response capabilities")
    print("✅ Generates realistic logs for SIEM and monitoring systems")
    print("✅ Supports autonomous operation without human intervention")
    print("✅ Configurable behavior patterns for different scenarios")
    print()
    
    print("🏁 Demo Complete!")
    print("The synthetic user simulation system is ready for integration")
    print("with the Archangel autonomous AI evolution framework.")


def main():
    """Main demo entry point"""
    try:
        asyncio.run(demo_synthetic_user_system())
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()