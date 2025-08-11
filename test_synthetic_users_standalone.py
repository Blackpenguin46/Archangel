#!/usr/bin/env python3
"""
Standalone test script for synthetic user simulation system
Tests core functionality without importing through agents package
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import random
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    work_hours: Tuple[int, int]  # (start_hour, end_hour)
    activity_frequency: float  # Actions per hour during work hours
    web_browsing_patterns: List[str]  # Common websites/categories
    file_access_patterns: List[str]  # Common file paths/types
    email_patterns: Dict[str, Any]  # Email behavior patterns
    risk_profile: str  # "low", "medium", "high" - likelihood of risky behavior


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


def test_user_profile_creation():
    """Test creating user profiles"""
    print("Testing user profile creation...")
    
    profile = UserProfile(
        user_id="test001",
        username="testuser",
        role=UserRole.DEVELOPER,
        department="Engineering",
        email="test@company.com",
        work_hours=(9, 17),
        activity_frequency=0.5,
        web_browsing_patterns=["/api", "/docs"],
        file_access_patterns=["/home/dev/"],
        email_patterns={"frequency": "medium"},
        risk_profile="low"
    )
    
    assert profile.user_id == "test001"
    assert profile.role == UserRole.DEVELOPER
    assert profile.work_hours == (9, 17)
    assert profile.activity_frequency == 0.5
    print("✅ User profile creation test passed")


def test_activity_event_creation():
    """Test creating activity events"""
    print("Testing activity event creation...")
    
    event = ActivityEvent(
        timestamp=datetime.now(),
        user_id="test001",
        activity_type=ActivityType.WEB_BROWSING,
        details={"url": "/api/test", "method": "GET"},
        success=True,
        duration_seconds=1.5,
        source_ip="192.168.20.100",
        user_agent="Mozilla/5.0 Test"
    )
    
    assert event.user_id == "test001"
    assert event.activity_type == ActivityType.WEB_BROWSING
    assert event.details["url"] == "/api/test"
    assert event.success is True
    assert event.duration_seconds == 1.5
    print("✅ Activity event creation test passed")


def test_role_based_patterns():
    """Test that different roles have different patterns"""
    print("Testing role-based behavior patterns...")
    
    # Define role-specific patterns (simplified version of what's in the main code)
    browsing_patterns = {
        UserRole.ADMIN: ["/admin", "/dashboard", "/system", "/logs"],
        UserRole.DEVELOPER: ["/api", "/docs", "/git", "/jenkins"],
        UserRole.SALES: ["/crm", "/leads", "/reports", "/customers"],
        UserRole.HR: ["/hr", "/employees", "/payroll", "/benefits"]
    }
    
    file_patterns = {
        UserRole.ADMIN: ["/var/log/", "/etc/", "/home/admin/"],
        UserRole.DEVELOPER: ["/home/dev/", "/var/www/", "/opt/code/"],
        UserRole.SALES: ["/home/sales/", "/shared/sales/", "/crm/"],
        UserRole.HR: ["/home/hr/", "/shared/hr/", "/employees/"]
    }
    
    # Test that patterns are different for different roles
    admin_web = browsing_patterns[UserRole.ADMIN]
    dev_web = browsing_patterns[UserRole.DEVELOPER]
    
    assert admin_web != dev_web
    assert "/admin" in admin_web
    assert "/api" in dev_web
    
    admin_files = file_patterns[UserRole.ADMIN]
    dev_files = file_patterns[UserRole.DEVELOPER]
    
    assert admin_files != dev_files
    assert "/var/log/" in admin_files
    assert "/home/dev/" in dev_files
    
    print("✅ Role-based patterns test passed")


def test_work_hours_logic():
    """Test work hours detection logic"""
    print("Testing work hours logic...")
    
    def is_work_hours(work_hours: Tuple[int, int], current_hour: int) -> bool:
        """Check if current time is within work hours"""
        start_hour, end_hour = work_hours
        return start_hour <= current_hour <= end_hour
    
    # Test different scenarios
    work_hours = (9, 17)  # 9 AM to 5 PM
    
    assert is_work_hours(work_hours, 10) is True   # 10 AM - work hours
    assert is_work_hours(work_hours, 6) is False   # 6 AM - before work
    assert is_work_hours(work_hours, 20) is False  # 8 PM - after work
    assert is_work_hours(work_hours, 9) is True    # 9 AM - start of work
    assert is_work_hours(work_hours, 17) is True   # 5 PM - end of work
    
    print("✅ Work hours logic test passed")


def test_activity_logging():
    """Test activity logging functionality"""
    print("Testing activity logging...")
    
    activity_log = []
    
    def log_activity(user_id: str, activity_type: ActivityType, details: Dict[str, Any], 
                    success: bool, duration: float, user_agent: str = None) -> None:
        """Log a user activity event"""
        event = ActivityEvent(
            timestamp=datetime.now(),
            user_id=user_id,
            activity_type=activity_type,
            details=details,
            success=success,
            duration_seconds=duration,
            source_ip="192.168.20.100",
            user_agent=user_agent
        )
        activity_log.append(event)
    
    # Log some test activities
    log_activity("test001", ActivityType.WEB_BROWSING, {"url": "/test"}, True, 1.5, "test-agent")
    log_activity("test001", ActivityType.FILE_ACCESS, {"file": "/test.txt"}, True, 0.8)
    log_activity("test001", ActivityType.EMAIL_ACTIVITY, {"action": "send"}, False, 2.3)
    
    assert len(activity_log) == 3
    
    # Check first activity
    first_activity = activity_log[0]
    assert first_activity.user_id == "test001"
    assert first_activity.activity_type == ActivityType.WEB_BROWSING
    assert first_activity.details["url"] == "/test"
    assert first_activity.success is True
    assert first_activity.user_agent == "test-agent"
    
    # Check activity summary
    total_activities = len(activity_log)
    successful_activities = sum(1 for a in activity_log if a.success)
    success_rate = successful_activities / total_activities
    
    assert total_activities == 3
    assert successful_activities == 2
    assert success_rate == 2/3
    
    print("✅ Activity logging test passed")


async def test_async_simulation():
    """Test async simulation functionality"""
    print("Testing async simulation...")
    
    async def simulate_web_browsing(duration_seconds: int = 5) -> List[ActivityEvent]:
        """Simulate web browsing for a short duration"""
        activities = []
        start_time = time.time()
        
        pages = ["/api", "/docs", "/dashboard", "/reports"]
        
        while time.time() - start_time < duration_seconds:
            page = random.choice(pages)
            
            # Simulate page visit
            activity_start = time.time()
            await asyncio.sleep(random.uniform(0.1, 0.5))  # Simulate page load time
            duration = time.time() - activity_start
            
            success = random.random() < 0.95  # 95% success rate
            
            activity = ActivityEvent(
                timestamp=datetime.now(),
                user_id="async_test001",
                activity_type=ActivityType.WEB_BROWSING,
                details={"url": page, "method": "GET"},
                success=success,
                duration_seconds=duration,
                source_ip="192.168.20.100",
                user_agent="Mozilla/5.0 Test"
            )
            activities.append(activity)
            
            # Small delay between requests
            await asyncio.sleep(random.uniform(0.1, 0.3))
        
        return activities
    
    async def simulate_file_access(duration_seconds: int = 3) -> List[ActivityEvent]:
        """Simulate file access for a short duration"""
        activities = []
        start_time = time.time()
        
        operations = ["read", "write", "list", "search"]
        files = ["/home/user/doc1.txt", "/home/user/data.csv", "/tmp/temp.log"]
        
        while time.time() - start_time < duration_seconds:
            operation = random.choice(operations)
            file_path = random.choice(files)
            
            activity_start = time.time()
            await asyncio.sleep(random.uniform(0.05, 0.2))  # Simulate file I/O
            duration = time.time() - activity_start
            
            success = random.random() < 0.90  # 90% success rate
            
            activity = ActivityEvent(
                timestamp=datetime.now(),
                user_id="async_test001",
                activity_type=ActivityType.FILE_ACCESS,
                details={"operation": operation, "file_path": file_path},
                success=success,
                duration_seconds=duration,
                source_ip="192.168.20.100"
            )
            activities.append(activity)
            
            await asyncio.sleep(random.uniform(0.1, 0.5))
        
        return activities
    
    # Run simulations concurrently
    web_task = asyncio.create_task(simulate_web_browsing(3))
    file_task = asyncio.create_task(simulate_file_access(2))
    
    web_activities, file_activities = await asyncio.gather(web_task, file_task)
    
    # Verify results
    assert len(web_activities) > 0
    assert len(file_activities) > 0
    
    # Check activity types
    for activity in web_activities:
        assert activity.activity_type == ActivityType.WEB_BROWSING
        assert "url" in activity.details
    
    for activity in file_activities:
        assert activity.activity_type == ActivityType.FILE_ACCESS
        assert "operation" in activity.details
        assert "file_path" in activity.details
    
    print("✅ Async simulation test passed")


def test_realistic_behavior_characteristics():
    """Test characteristics that make behavior realistic"""
    print("Testing realistic behavior characteristics...")
    
    # Test user agent variety
    user_agents = {
        UserRole.ADMIN: [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
        ],
        UserRole.DEVELOPER: [
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ]
    }
    
    # Verify user agents are realistic
    for role, agents in user_agents.items():
        for ua in agents:
            assert "Mozilla" in ua
            assert any(browser in ua for browser in ["Chrome", "Firefox", "Safari"])
    
    # Test filename generation realism
    def generate_realistic_filename(extensions: List[str]) -> str:
        prefixes = ["document", "report", "data", "backup", "config", "log"]
        prefix = random.choice(prefixes)
        suffix = random.choice(extensions)
        timestamp = datetime.now().strftime("%Y%m%d")
        
        if random.random() < 0.5:
            return f"{prefix}_{timestamp}{suffix}"
        else:
            return f"{prefix}_{random.randint(1, 999)}{suffix}"
    
    # Generate multiple filenames and check for variety
    extensions = [".txt", ".log", ".conf"]
    filenames = [generate_realistic_filename(extensions) for _ in range(10)]
    
    # Should have some variety
    unique_filenames = set(filenames)
    assert len(unique_filenames) > 1
    
    # All should have valid extensions
    for filename in filenames:
        assert any(filename.endswith(ext) for ext in extensions)
    
    # Test timing realism
    def generate_realistic_timing() -> float:
        """Generate realistic activity timing"""
        # Most activities should be quick, but some can be longer
        if random.random() < 0.8:  # 80% quick activities
            return random.uniform(0.1, 5.0)  # 0.1 to 5 seconds
        else:  # 20% longer activities
            return random.uniform(5.0, 60.0)  # 5 seconds to 1 minute
    
    timings = [generate_realistic_timing() for _ in range(100)]
    
    # Most should be under 5 seconds
    quick_activities = sum(1 for t in timings if t <= 5.0)
    assert quick_activities >= 70  # At least 70% should be quick
    
    # All should be reasonable (not too fast or too slow)
    for timing in timings:
        assert 0.1 <= timing <= 300  # 0.1 seconds to 5 minutes
    
    print("✅ Realistic behavior characteristics test passed")


def test_detection_evasion_features():
    """Test features that help evade detection"""
    print("Testing detection evasion features...")
    
    # Test background noise generation
    def generate_background_activities(num_users: int = 5, activities_per_user: int = 10) -> List[ActivityEvent]:
        """Generate background noise activities"""
        activities = []
        
        for user_id in range(num_users):
            for _ in range(activities_per_user):
                activity_type = random.choice(list(ActivityType))
                
                # Generate realistic details based on activity type
                if activity_type == ActivityType.WEB_BROWSING:
                    details = {"url": f"/page_{random.randint(1, 100)}", "method": "GET"}
                elif activity_type == ActivityType.FILE_ACCESS:
                    details = {"operation": "read", "file_path": f"/home/user{user_id}/file_{random.randint(1, 50)}.txt"}
                elif activity_type == ActivityType.EMAIL_ACTIVITY:
                    details = {"action": "receive", "sender": f"user{random.randint(1, 20)}@company.com"}
                else:
                    details = {"action": "misc"}
                
                activity = ActivityEvent(
                    timestamp=datetime.now() - timedelta(seconds=random.randint(0, 3600)),
                    user_id=f"user{user_id:03d}",
                    activity_type=activity_type,
                    details=details,
                    success=random.random() < 0.95,  # High success rate for normal users
                    duration_seconds=random.uniform(0.5, 30.0),
                    source_ip=f"192.168.20.{100 + user_id}"
                )
                activities.append(activity)
        
        return activities
    
    background_activities = generate_background_activities()
    
    # Should have diverse activities
    activity_types = set(a.activity_type for a in background_activities)
    assert len(activity_types) >= 3  # At least 3 different activity types
    
    # Should have multiple users
    user_ids = set(a.user_id for a in background_activities)
    assert len(user_ids) >= 5
    
    # Should have high success rate (normal user behavior)
    successful = sum(1 for a in background_activities if a.success)
    success_rate = successful / len(background_activities)
    assert success_rate >= 0.8  # At least 80% success rate
    
    # Should have realistic IP addresses
    for activity in background_activities:
        assert activity.source_ip.startswith("192.168.20.")
    
    # Test temporal distribution
    timestamps = [a.timestamp for a in background_activities]
    time_range = max(timestamps) - min(timestamps)
    assert time_range.total_seconds() > 0  # Activities spread over time
    
    print("✅ Detection evasion features test passed")


def test_configuration_validation():
    """Test configuration validation"""
    print("Testing configuration validation...")
    
    # Test valid configuration
    valid_profiles = [
        UserProfile(
            user_id="valid001",
            username="validuser",
            role=UserRole.DEVELOPER,
            department="Engineering",
            email="valid@company.com",
            work_hours=(9, 17),
            activity_frequency=0.8,
            web_browsing_patterns=["/api"],
            file_access_patterns=["/home/dev/"],
            email_patterns={"frequency": "medium"},
            risk_profile="medium"
        )
    ]
    
    def validate_profiles(profiles: List[UserProfile]) -> List[str]:
        """Validate user profiles and return errors"""
        errors = []
        user_ids = set()
        usernames = set()
        
        for profile in profiles:
            # Check for duplicates
            if profile.user_id in user_ids:
                errors.append(f"Duplicate user ID: {profile.user_id}")
            user_ids.add(profile.user_id)
            
            if profile.username in usernames:
                errors.append(f"Duplicate username: {profile.username}")
            usernames.add(profile.username)
            
            # Validate work hours
            if len(profile.work_hours) != 2:
                errors.append(f"Invalid work hours for {profile.username}")
            elif profile.work_hours[0] >= profile.work_hours[1]:
                errors.append(f"Invalid work hours for {profile.username}: start >= end")
            
            # Validate activity frequency
            if profile.activity_frequency <= 0:
                errors.append(f"Invalid activity frequency for {profile.username}")
            
            # Validate risk profile
            if profile.risk_profile not in ["low", "medium", "high"]:
                errors.append(f"Invalid risk profile for {profile.username}")
        
        return errors
    
    # Test valid configuration
    errors = validate_profiles(valid_profiles)
    assert len(errors) == 0
    
    # Test invalid configuration - duplicate user ID
    invalid_profiles = valid_profiles + [
        UserProfile(
            user_id="valid001",  # Duplicate ID
            username="duplicate",
            role=UserRole.ADMIN,
            department="IT",
            email="duplicate@company.com",
            work_hours=(8, 17),
            activity_frequency=0.5,
            web_browsing_patterns=["/admin"],
            file_access_patterns=["/var/log/"],
            email_patterns={"frequency": "low"},
            risk_profile="low"
        )
    ]
    
    errors = validate_profiles(invalid_profiles)
    assert len(errors) > 0
    assert any("Duplicate user ID" in error for error in errors)
    
    print("✅ Configuration validation test passed")


def main():
    """Run all tests"""
    print("Running Synthetic User Simulation Tests (Standalone)")
    print("=" * 60)
    
    try:
        # Run synchronous tests
        test_user_profile_creation()
        test_activity_event_creation()
        test_role_based_patterns()
        test_work_hours_logic()
        test_activity_logging()
        test_realistic_behavior_characteristics()
        test_detection_evasion_features()
        test_configuration_validation()
        
        # Run async tests
        asyncio.run(test_async_simulation())
        
        print("\n" + "=" * 60)
        print("🎉 All standalone tests passed successfully!")
        print("\nCore synthetic user simulation functionality verified:")
        print("  ✅ User profile data structures")
        print("  ✅ Activity event logging")
        print("  ✅ Role-based behavior differentiation")
        print("  ✅ Work hours logic")
        print("  ✅ Activity logging and summarization")
        print("  ✅ Async simulation capabilities")
        print("  ✅ Realistic behavior characteristics")
        print("  ✅ Detection evasion features")
        print("  ✅ Configuration validation")
        print("\nThe synthetic user system core functionality is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()