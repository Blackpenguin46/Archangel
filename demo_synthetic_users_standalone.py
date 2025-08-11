#!/usr/bin/env python3
"""
Standalone Synthetic User Simulation Demo

This script demonstrates the synthetic user simulation system without
importing through the agents package to avoid dependency issues.
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import random
import time


class UserRole(Enum):
    """User roles with different access patterns and behaviors"""
    ADMIN = "admin"
    DEVELOPER = "developer"
    SALES = "sales"
    HR = "hr"
    FINANCE = "finance"
    MARKETING = "marketing"
    SUPPORT = "support"


class ActivityType(Enum):
    """Types of user activities"""
    WEB_BROWSING = "web_browsing"
    FILE_ACCESS = "file_access"
    EMAIL_ACTIVITY = "email_activity"
    LOGIN_LOGOUT = "login_logout"
    APPLICATION_USAGE = "application_usage"


@dataclass
class UserProfile:
    """Profile defining a synthetic user's characteristics and behavior patterns"""
    user_id: str
    username: str
    role: UserRole
    department: str
    email: str
    work_hours: Tuple[int, int]
    activity_frequency: float
    web_browsing_patterns: List[str]
    file_access_patterns: List[str]
    email_patterns: Dict[str, Any]
    risk_profile: str


@dataclass
class ActivityEvent:
    """Record of a synthetic user activity"""
    timestamp: datetime
    user_id: str
    activity_type: ActivityType
    details: Dict[str, Any]
    success: bool
    duration_seconds: float
    source_ip: str
    user_agent: str = None


def create_sample_user_profiles() -> List[UserProfile]:
    """Create sample user profiles for demonstration"""
    return [
        UserProfile(
            user_id="admin001",
            username="admin",
            role=UserRole.ADMIN,
            department="IT",
            email="admin@company.com",
            work_hours=(8, 17),
            activity_frequency=0.5,
            web_browsing_patterns=[
                "/admin", "/dashboard", "/system", "/logs", "/monitoring",
                "/users", "/settings", "/backup", "/security"
            ],
            file_access_patterns=[
                "/var/log/", "/etc/", "/home/admin/", "/opt/", "/usr/local/",
                "/backup/", "/scripts/", "/config/"
            ],
            email_patterns={
                "send_frequency": 0.3,
                "receive_frequency": 0.8,
                "common_subjects": [
                    "System Maintenance", "Security Alert", "Backup Report"
                ]
            },
            risk_profile="low"
        ),
        UserProfile(
            user_id="dev001",
            username="developer",
            role=UserRole.DEVELOPER,
            department="Engineering",
            email="dev@company.com",
            work_hours=(9, 18),
            activity_frequency=0.8,
            web_browsing_patterns=[
                "/api", "/docs", "/git", "/jenkins", "/jira", "/confluence",
                "/stackoverflow.com", "/github.com", "/dev-tools"
            ],
            file_access_patterns=[
                "/home/dev/", "/var/www/", "/opt/code/", "/git/", "/build/",
                "/src/", "/docs/", "/tests/"
            ],
            email_patterns={
                "send_frequency": 0.5,
                "receive_frequency": 1.2,
                "common_subjects": [
                    "Code Review", "Bug Report", "Feature Request"
                ]
            },
            risk_profile="medium"
        ),
        UserProfile(
            user_id="sales001",
            username="salesperson",
            role=UserRole.SALES,
            department="Sales",
            email="sales@company.com",
            work_hours=(8, 18),
            activity_frequency=1.0,
            web_browsing_patterns=[
                "/crm", "/leads", "/reports", "/customers", "/deals",
                "/linkedin.com", "/salesforce.com", "/proposals"
            ],
            file_access_patterns=[
                "/home/sales/", "/shared/sales/", "/crm/", "/proposals/",
                "/contracts/", "/leads/", "/reports/"
            ],
            email_patterns={
                "send_frequency": 1.5,
                "receive_frequency": 2.0,
                "common_subjects": [
                    "Proposal", "Client Meeting", "Quote Request"
                ]
            },
            risk_profile="high"
        ),
        UserProfile(
            user_id="hr001",
            username="hr_manager",
            role=UserRole.HR,
            department="Human Resources",
            email="hr@company.com",
            work_hours=(8, 17),
            activity_frequency=0.6,
            web_browsing_patterns=[
                "/hr", "/employees", "/payroll", "/benefits", "/recruiting",
                "/workday.com", "/bamboohr.com", "/applicants"
            ],
            file_access_patterns=[
                "/home/hr/", "/shared/hr/", "/employees/", "/payroll/",
                "/benefits/", "/policies/", "/recruiting/"
            ],
            email_patterns={
                "send_frequency": 0.8,
                "receive_frequency": 1.5,
                "common_subjects": [
                    "Employee Onboarding", "Policy Update", "Benefits"
                ]
            },
            risk_profile="low"
        )
    ]


class SyntheticUserSimulator:
    """Simplified synthetic user simulator for demonstration"""
    
    def __init__(self, profile: UserProfile):
        self.profile = profile
        self.activity_log: List[ActivityEvent] = []
        
        # User agents by role
        self.user_agents = {
            UserRole.ADMIN: [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            ],
            UserRole.DEVELOPER: [
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            ],
            UserRole.SALES: [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            ],
            UserRole.HR: [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            ]
        }
    
    def log_activity(self, activity_type: ActivityType, details: Dict[str, Any], 
                    success: bool, duration: float, user_agent: str = None) -> None:
        """Log a user activity event"""
        event = ActivityEvent(
            timestamp=datetime.now(),
            user_id=self.profile.user_id,
            activity_type=activity_type,
            details=details,
            success=success,
            duration_seconds=duration,
            source_ip="192.168.20.100",
            user_agent=user_agent
        )
        self.activity_log.append(event)
    
    async def simulate_login(self) -> None:
        """Simulate user login"""
        await asyncio.sleep(random.uniform(0.5, 2.0))
        self.log_activity(
            ActivityType.LOGIN_LOGOUT,
            {"action": "login", "method": "password"},
            True,
            1.2
        )
    
    async def simulate_logout(self) -> None:
        """Simulate user logout"""
        await asyncio.sleep(random.uniform(0.1, 0.5))
        self.log_activity(
            ActivityType.LOGIN_LOGOUT,
            {"action": "logout"},
            True,
            0.3
        )
    
    async def simulate_web_browsing(self, num_pages: int = 3) -> None:
        """Simulate web browsing session"""
        user_agent = random.choice(self.user_agents.get(self.profile.role, 
                                   self.user_agents[UserRole.ADMIN]))
        
        for i in range(num_pages):
            page = random.choice(self.profile.web_browsing_patterns)
            await asyncio.sleep(random.uniform(0.5, 3.0))
            
            self.log_activity(
                ActivityType.WEB_BROWSING,
                {"url": page, "method": "GET"},
                random.random() < 0.95,  # 95% success rate
                random.uniform(0.8, 4.0),
                user_agent
            )
    
    async def simulate_file_access(self, num_operations: int = 2) -> None:
        """Simulate file access operations"""
        operations = ["read", "write", "list", "search"]
        
        for i in range(num_operations):
            operation = random.choice(operations)
            base_path = random.choice(self.profile.file_access_patterns)
            file_path = f"{base_path}document_{i+1}.txt"
            
            await asyncio.sleep(random.uniform(0.2, 1.5))
            
            self.log_activity(
                ActivityType.FILE_ACCESS,
                {"operation": operation, "file_path": file_path},
                random.random() < 0.90,  # 90% success rate
                random.uniform(0.3, 2.0)
            )
    
    async def simulate_email_activity(self, num_emails: int = 1) -> None:
        """Simulate email activity"""
        for i in range(num_emails):
            action = random.choice(["send", "receive"])
            subject = random.choice(self.profile.email_patterns["common_subjects"])
            
            await asyncio.sleep(random.uniform(1.0, 5.0))
            
            details = {
                "action": action,
                "subject": subject
            }
            
            if action == "send":
                details["recipient"] = "colleague@company.com"
            else:
                details["sender"] = "external@company.com"
            
            self.log_activity(
                ActivityType.EMAIL_ACTIVITY,
                details,
                random.random() < 0.98,  # 98% success rate
                random.uniform(1.0, 8.0)
            )
    
    def get_activity_summary(self) -> Dict[str, Any]:
        """Get summary of user activities"""
        if not self.activity_log:
            return {"total_activities": 0}
        
        total_activities = len(self.activity_log)
        successful_activities = sum(1 for a in self.activity_log if a.success)
        success_rate = successful_activities / total_activities
        
        activity_counts = {}
        for activity in self.activity_log:
            activity_type = activity.activity_type.value
            activity_counts[activity_type] = activity_counts.get(activity_type, 0) + 1
        
        total_duration = sum(a.duration_seconds for a in self.activity_log)
        avg_duration = total_duration / total_activities
        
        return {
            "total_activities": total_activities,
            "activity_counts": activity_counts,
            "success_rate": success_rate,
            "average_duration_seconds": avg_duration,
            "time_range": {
                "start": min(a.timestamp for a in self.activity_log).isoformat(),
                "end": max(a.timestamp for a in self.activity_log).isoformat()
            }
        }


async def demo_synthetic_user_system():
    """Demonstrate the synthetic user simulation system"""
    
    print("🤖 Archangel Synthetic User Simulation Demo")
    print("=" * 50)
    print()
    
    # Create sample user profiles
    print("📋 Creating user profiles...")
    user_profiles = create_sample_user_profiles()
    print(f"✅ Created {len(user_profiles)} user profiles")
    print()
    
    # Show user profiles
    print("👥 User Profiles:")
    print("-" * 40)
    for profile in user_profiles:
        print(f"🔹 {profile.username} ({profile.role.value})")
        print(f"   Department: {profile.department}")
        print(f"   Work Hours: {profile.work_hours[0]:02d}:00 - {profile.work_hours[1]:02d}:00")
        print(f"   Activity Level: {profile.activity_frequency}")
        print(f"   Risk Profile: {profile.risk_profile}")
        print()
    
    # Demonstrate role-based behavior differences
    print("🎭 Role-Based Behavior Patterns:")
    print("-" * 40)
    
    for profile in user_profiles:
        print(f"\n{profile.role.value.upper()} - Web Browsing Patterns:")
        for pattern in profile.web_browsing_patterns[:4]:
            print(f"   • {pattern}")
        
        print(f"\n{profile.role.value.upper()} - File Access Patterns:")
        for pattern in profile.file_access_patterns[:3]:
            print(f"   • {pattern}")
    print()
    
    # Demonstrate activity simulation
    print("⚡ Activity Simulation Demo:")
    print("-" * 40)
    
    # Use the developer profile for detailed demo
    dev_profile = next(p for p in user_profiles if p.role == UserRole.DEVELOPER)
    demo_user = SyntheticUserSimulator(dev_profile)
    
    print(f"🚀 Starting activity simulation for {dev_profile.username}...")
    print()
    
    # Simulate a complete user session
    print("📝 Simulating login...")
    await demo_user.simulate_login()
    
    print("🌐 Simulating web browsing...")
    await demo_user.simulate_web_browsing(4)
    
    print("📁 Simulating file access...")
    await demo_user.simulate_file_access(3)
    
    print("📧 Simulating email activity...")
    await demo_user.simulate_email_activity(2)
    
    print("📤 Simulating logout...")
    await demo_user.simulate_logout()
    
    print("✅ Simulation complete!")
    print()
    
    # Show activity summary
    print("📊 Activity Summary:")
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
    print("-" * 60)
    print(f"{'#':<3} {'Time':<8} {'Activity':<15} {'Status':<6} {'Duration':<8} {'Details'}")
    print("-" * 60)
    
    for i, activity in enumerate(demo_user.activity_log, 1):
        status = "✅ OK" if activity.success else "❌ FAIL"
        time_str = activity.timestamp.strftime('%H:%M:%S')
        activity_type = activity.activity_type.value
        duration_str = f"{activity.duration_seconds:.2f}s"
        
        # Format details
        if activity.activity_type == ActivityType.WEB_BROWSING:
            details = activity.details.get('url', 'N/A')
        elif activity.activity_type == ActivityType.FILE_ACCESS:
            op = activity.details.get('operation', 'N/A')
            file_path = activity.details.get('file_path', 'N/A')
            details = f"{op}: {file_path}"
        elif activity.activity_type == ActivityType.EMAIL_ACTIVITY:
            action = activity.details.get('action', 'N/A')
            subject = activity.details.get('subject', 'N/A')
            details = f"{action}: {subject}"
        else:
            action = activity.details.get('action', 'N/A')
            details = action
        
        # Truncate details if too long
        if len(details) > 30:
            details = details[:27] + "..."
        
        print(f"{i:<3} {time_str:<8} {activity_type:<15} {status:<6} {duration_str:<8} {details}")
    print()
    
    # Demonstrate multi-user simulation
    print("👥 Multi-User Background Activity Demo:")
    print("-" * 40)
    
    # Create simulators for all users
    simulators = [SyntheticUserSimulator(profile) for profile in user_profiles]
    
    print("🚀 Simulating concurrent user activities...")
    
    # Run concurrent simulations
    tasks = []
    for sim in simulators:
        async def user_session(simulator):
            await simulator.simulate_login()
            await simulator.simulate_web_browsing(2)
            await simulator.simulate_file_access(1)
            await simulator.simulate_email_activity(1)
            await simulator.simulate_logout()
        
        tasks.append(asyncio.create_task(user_session(sim)))
    
    # Wait for all simulations to complete
    await asyncio.gather(*tasks)
    
    print("✅ Multi-user simulation complete!")
    print()
    
    # Show aggregated statistics
    print("📊 System-Wide Activity Statistics:")
    print("-" * 40)
    
    total_activities = sum(len(sim.activity_log) for sim in simulators)
    total_successful = sum(sum(1 for a in sim.activity_log if a.success) for sim in simulators)
    overall_success_rate = total_successful / total_activities if total_activities > 0 else 0
    
    print(f"   Total Users: {len(simulators)}")
    print(f"   Total Activities: {total_activities}")
    print(f"   Overall Success Rate: {overall_success_rate:.1%}")
    print()
    
    # Activity breakdown by type
    activity_type_counts = {}
    for sim in simulators:
        for activity in sim.activity_log:
            activity_type = activity.activity_type.value
            activity_type_counts[activity_type] = activity_type_counts.get(activity_type, 0) + 1
    
    print("   Activity Type Distribution:")
    for activity_type, count in sorted(activity_type_counts.items()):
        percentage = (count / total_activities) * 100
        print(f"     • {activity_type}: {count} ({percentage:.1f}%)")
    print()
    
    # Show per-user summaries
    print("   Per-User Activity Summary:")
    for sim in simulators:
        user_summary = sim.get_activity_summary()
        print(f"     • {sim.profile.username} ({sim.profile.role.value}): "
              f"{user_summary['total_activities']} activities, "
              f"{user_summary['success_rate']:.1%} success rate")
    print()
    
    # Demonstrate detection evasion characteristics
    print("🕵️ Detection Evasion Characteristics:")
    print("-" * 40)
    
    print("🔍 Realistic User Agents:")
    for role, agents in demo_user.user_agents.items():
        print(f"   {role.value}: {agents[0][:60]}...")
    
    print("\n⏰ Realistic Timing Patterns:")
    all_durations = []
    for sim in simulators:
        all_durations.extend([a.duration_seconds for a in sim.activity_log])
    
    if all_durations:
        avg_duration = sum(all_durations) / len(all_durations)
        min_duration = min(all_durations)
        max_duration = max(all_durations)
        print(f"   • Average activity duration: {avg_duration:.2f}s")
        print(f"   • Duration range: {min_duration:.2f}s - {max_duration:.2f}s")
        print(f"   • Most activities under 5s: {sum(1 for d in all_durations if d < 5.0) / len(all_durations):.1%}")
    
    print("\n🌐 Network Characteristics:")
    print("   • Source IP range: 192.168.20.0/24 (internal network)")
    print("   • Realistic internal traffic patterns")
    print("   • Role-appropriate access patterns")
    
    print("\n📈 Background Noise Benefits:")
    print("   ✅ Masks attacker activities in legitimate traffic")
    print("   ✅ Creates realistic baseline for anomaly detection")
    print("   ✅ Provides cover for reconnaissance activities")
    print("   ✅ Complicates log analysis and forensics")
    print("   ✅ Tests Blue Team detection capabilities")
    print()
    
    # Show integration benefits
    print("🎯 Integration with Archangel System:")
    print("-" * 40)
    print("✅ Autonomous operation without human intervention")
    print("✅ Configurable behavior patterns for different scenarios")
    print("✅ Role-based differentiation for realistic enterprise simulation")
    print("✅ Comprehensive activity logging for analysis")
    print("✅ Scalable to support multiple concurrent users")
    print("✅ Realistic timing and success rate patterns")
    print("✅ Detection evasion characteristics")
    print()
    
    # Show sample log output in JSONL format
    print("📄 Sample Activity Log (JSONL Format):")
    print("-" * 40)
    
    # Show first few activities in JSON format
    sample_activities = demo_user.activity_log[:3]
    for activity in sample_activities:
        activity_dict = {
            'timestamp': activity.timestamp.isoformat(),
            'user_id': activity.user_id,
            'activity_type': activity.activity_type.value,
            'details': activity.details,
            'success': activity.success,
            'duration_seconds': activity.duration_seconds,
            'source_ip': activity.source_ip,
            'user_agent': activity.user_agent
        }
        print(json.dumps(activity_dict, indent=None))
    print("...")
    print()
    
    print("🏁 Demo Complete!")
    print("The synthetic user simulation system demonstrates:")
    print("• Realistic user behavior simulation")
    print("• Role-based activity differentiation")
    print("• Background noise generation for Red Team cover")
    print("• Comprehensive activity logging and analysis")
    print("• Detection evasion characteristics")
    print("• Multi-user concurrent operation")
    print("\nReady for integration with the Archangel framework!")


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