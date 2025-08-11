#!/usr/bin/env python3
"""
Simple test script for synthetic user simulation system
Tests core functionality without external dependencies
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from agents.synthetic_users import (
    UserProfile, UserRole, ActivityType, ActivityEvent,
    WebBrowsingSimulator, FileAccessSimulator, EmailActivitySimulator,
    ComprehensiveSyntheticUser, SyntheticUserManager
)
from agents.synthetic_user_config import SyntheticUserConfigManager


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
    print("✅ User profile creation test passed")


def test_web_browsing_simulator():
    """Test web browsing simulator"""
    print("Testing web browsing simulator...")
    
    profile = UserProfile(
        user_id="web001", username="webuser", role=UserRole.DEVELOPER,
        department="Engineering", email="web@company.com", work_hours=(9, 17),
        activity_frequency=0.8, web_browsing_patterns=["/api", "/docs"],
        file_access_patterns=["/home/dev/"], email_patterns={"frequency": "medium"},
        risk_profile="medium"
    )
    
    environment_config = {
        'web_server_url': 'http://192.168.10.10',
        'user_ip': '192.168.20.100'
    }
    
    simulator = WebBrowsingSimulator(profile, environment_config)
    
    # Test initialization
    assert simulator.profile == profile
    assert simulator.environment_config == environment_config
    assert UserRole.DEVELOPER in simulator.browsing_patterns
    
    # Test role-specific patterns
    dev_patterns = simulator.browsing_patterns[UserRole.DEVELOPER]
    assert "/api" in dev_patterns
    assert "/docs" in dev_patterns
    
    print("✅ Web browsing simulator test passed")


def test_file_access_simulator():
    """Test file access simulator"""
    print("Testing file access simulator...")
    
    profile = UserProfile(
        user_id="file001", username="fileuser", role=UserRole.ADMIN,
        department="IT", email="file@company.com", work_hours=(8, 17),
        activity_frequency=0.5, web_browsing_patterns=[], file_access_patterns=["/var/log/", "/etc/"],
        email_patterns={}, risk_profile="low"
    )
    
    environment_config = {'user_ip': '192.168.20.100'}
    
    simulator = FileAccessSimulator(profile, environment_config)
    
    # Test initialization
    assert simulator.profile == profile
    assert UserRole.ADMIN in simulator.file_patterns
    
    # Test filename generation
    extensions = [".log", ".conf", ".txt"]
    filename = simulator._generate_filename(extensions)
    assert any(filename.endswith(ext) for ext in extensions)
    
    # Test role-specific patterns
    admin_patterns = simulator.file_patterns[UserRole.ADMIN]
    assert "/var/log/" in admin_patterns
    assert "/etc/" in admin_patterns
    
    print("✅ File access simulator test passed")


def test_email_activity_simulator():
    """Test email activity simulator"""
    print("Testing email activity simulator...")
    
    profile = UserProfile(
        user_id="email001", username="emailuser", role=UserRole.SALES,
        department="Sales", email="email@company.com", work_hours=(8, 18),
        activity_frequency=1.0, web_browsing_patterns=[], file_access_patterns=[],
        email_patterns={"frequency": "high"}, risk_profile="high"
    )
    
    environment_config = {'user_ip': '192.168.20.100'}
    
    simulator = EmailActivitySimulator(profile, environment_config)
    
    # Test initialization
    assert simulator.profile == profile
    assert UserRole.SALES in simulator.email_patterns
    
    # Test role-specific patterns
    sales_patterns = simulator.email_patterns[UserRole.SALES]
    assert "send_frequency" in sales_patterns
    assert "common_subjects" in sales_patterns
    
    print("✅ Email activity simulator test passed")


def test_comprehensive_synthetic_user():
    """Test comprehensive synthetic user"""
    print("Testing comprehensive synthetic user...")
    
    profile = UserProfile(
        user_id="comp001", username="compuser", role=UserRole.DEVELOPER,
        department="Engineering", email="comp@company.com", work_hours=(9, 17),
        activity_frequency=0.8, web_browsing_patterns=["/api", "/docs"],
        file_access_patterns=["/home/dev/"], email_patterns={"frequency": "medium"},
        risk_profile="medium"
    )
    
    environment_config = {
        'web_server_url': 'http://192.168.10.10',
        'user_ip': '192.168.20.100'
    }
    
    user = ComprehensiveSyntheticUser(profile, environment_config)
    
    # Test initialization
    assert user.profile == profile
    assert user.web_simulator is not None
    assert user.file_simulator is not None
    assert user.email_simulator is not None
    assert user.is_running is False
    
    # Test activity logging
    user.log_activity(
        ActivityType.WEB_BROWSING,
        {"url": "/test"},
        True,
        1.5,
        "test-user-agent"
    )
    
    assert len(user.activity_log) == 1
    activity = user.activity_log[0]
    assert activity.user_id == profile.user_id
    assert activity.activity_type == ActivityType.WEB_BROWSING
    assert activity.details["url"] == "/test"
    assert activity.success is True
    
    # Test activity summary
    summary = user.get_activity_summary()
    assert summary["total_activities"] == 1
    assert summary["success_rate"] == 1.0
    
    print("✅ Comprehensive synthetic user test passed")


def test_synthetic_user_manager():
    """Test synthetic user manager"""
    print("Testing synthetic user manager...")
    
    environment_config = {
        'web_server_url': 'http://192.168.10.10',
        'user_ip': '192.168.20.100'
    }
    
    manager = SyntheticUserManager(environment_config)
    
    # Test initialization
    assert manager.environment_config == environment_config
    assert len(manager.users) == 0
    assert manager.is_running is False
    
    # Test default user profiles
    profiles = manager.create_default_user_profiles()
    assert len(profiles) >= 4
    
    # Check that we have different roles
    roles = [profile.role for profile in profiles]
    assert UserRole.ADMIN in roles
    assert UserRole.DEVELOPER in roles
    assert UserRole.SALES in roles
    assert UserRole.HR in roles
    
    print("✅ Synthetic user manager test passed")


async def test_async_functionality():
    """Test async functionality"""
    print("Testing async functionality...")
    
    profile = UserProfile(
        user_id="async001", username="asyncuser", role=UserRole.DEVELOPER,
        department="Engineering", email="async@company.com", work_hours=(9, 17),
        activity_frequency=0.8, web_browsing_patterns=["/api"], file_access_patterns=["/home/dev/"],
        email_patterns={"frequency": "medium"}, risk_profile="medium"
    )
    
    environment_config = {'user_ip': '192.168.20.100'}
    
    user = ComprehensiveSyntheticUser(profile, environment_config)
    
    # Test login/logout simulation
    await user._simulate_login()
    assert len(user.activity_log) == 1
    assert user.activity_log[0].details["action"] == "login"
    
    await user._simulate_logout()
    assert len(user.activity_log) == 2
    assert user.activity_log[1].details["action"] == "logout"
    
    # Test file operation simulation
    file_sim = user.file_simulator
    success = await file_sim._perform_file_operation("read", "/test/file.txt")
    assert isinstance(success, bool)
    
    print("✅ Async functionality test passed")


def test_configuration_manager():
    """Test configuration manager"""
    print("Testing configuration manager...")
    
    config_manager = SyntheticUserConfigManager()
    
    # Test default config creation
    config_manager.create_default_config()
    
    assert len(config_manager.user_profiles) > 0
    assert config_manager.environment_config.web_server_url is not None
    assert config_manager.behavior_config.web_browsing_success_rate > 0
    
    # Test validation
    errors = config_manager.validate_config()
    assert len(errors) == 0, f"Configuration validation failed: {errors}"
    
    # Test user profile operations
    original_count = len(config_manager.user_profiles)
    
    test_profile = UserProfile(
        user_id="config_test001", username="configtest", role=UserRole.SUPPORT,
        department="Support", email="configtest@company.com", work_hours=(8, 20),
        activity_frequency=1.2, web_browsing_patterns=["/support"], file_access_patterns=["/home/support/"],
        email_patterns={"frequency": "high"}, risk_profile="medium"
    )
    
    config_manager.add_user_profile(test_profile)
    assert len(config_manager.user_profiles) == original_count + 1
    
    retrieved_profile = config_manager.get_user_profile("config_test001")
    assert retrieved_profile is not None
    assert retrieved_profile.username == "configtest"
    
    # Test role filtering
    support_profiles = config_manager.get_profiles_by_role(UserRole.SUPPORT)
    assert len(support_profiles) >= 1
    assert any(p.user_id == "config_test001" for p in support_profiles)
    
    print("✅ Configuration manager test passed")


def test_behavior_realism():
    """Test behavior realism characteristics"""
    print("Testing behavior realism...")
    
    # Test role-based differences
    admin_profile = UserProfile(
        user_id="admin001", username="admin", role=UserRole.ADMIN,
        department="IT", email="admin@company.com", work_hours=(8, 17),
        activity_frequency=0.5, web_browsing_patterns=[], file_access_patterns=[],
        email_patterns={}, risk_profile="low"
    )
    
    sales_profile = UserProfile(
        user_id="sales001", username="sales", role=UserRole.SALES,
        department="Sales", email="sales@company.com", work_hours=(8, 18),
        activity_frequency=1.0, web_browsing_patterns=[], file_access_patterns=[],
        email_patterns={}, risk_profile="high"
    )
    
    environment_config = {'user_ip': '192.168.20.100'}
    
    admin_user = ComprehensiveSyntheticUser(admin_profile, environment_config)
    sales_user = ComprehensiveSyntheticUser(sales_profile, environment_config)
    
    # Check that web browsing patterns are different
    admin_patterns = admin_user.web_simulator.browsing_patterns[UserRole.ADMIN]
    sales_patterns = sales_user.web_simulator.browsing_patterns[UserRole.SALES]
    
    assert admin_patterns != sales_patterns
    assert "/admin" in admin_patterns
    assert "/crm" in sales_patterns
    
    # Check activity frequencies are different
    assert admin_profile.activity_frequency != sales_profile.activity_frequency
    assert sales_profile.activity_frequency > admin_profile.activity_frequency
    
    print("✅ Behavior realism test passed")


def main():
    """Run all tests"""
    print("Running Synthetic User Simulation Tests")
    print("=" * 50)
    
    try:
        # Run synchronous tests
        test_user_profile_creation()
        test_web_browsing_simulator()
        test_file_access_simulator()
        test_email_activity_simulator()
        test_comprehensive_synthetic_user()
        test_synthetic_user_manager()
        test_configuration_manager()
        test_behavior_realism()
        
        # Run async tests
        asyncio.run(test_async_functionality())
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        print("\nThe synthetic user simulation system is working correctly.")
        print("Key features verified:")
        print("  ✅ User profile management")
        print("  ✅ Web browsing simulation")
        print("  ✅ File access simulation")
        print("  ✅ Email activity simulation")
        print("  ✅ Comprehensive user behavior")
        print("  ✅ Multi-user management")
        print("  ✅ Configuration management")
        print("  ✅ Behavior realism and role differentiation")
        print("  ✅ Async functionality")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()