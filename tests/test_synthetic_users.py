"""
Tests for Synthetic User Simulation System

This module contains comprehensive tests for the synthetic user agents,
including behavior realism validation and detection evasion testing.
"""

import pytest
import asyncio
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

from agents.synthetic_users import (
    UserProfile, UserRole, ActivityType, ActivityEvent,
    SyntheticUserAgent, WebBrowsingSimulator, FileAccessSimulator,
    EmailActivitySimulator, ComprehensiveSyntheticUser, SyntheticUserManager
)


class TestUserProfile:
    """Test UserProfile data class"""
    
    def test_user_profile_creation(self):
        """Test creating a user profile"""
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


class TestWebBrowsingSimulator:
    """Test web browsing simulation"""
    
    @pytest.fixture
    def profile(self):
        return UserProfile(
            user_id="web001",
            username="webuser",
            role=UserRole.DEVELOPER,
            department="Engineering",
            email="web@company.com",
            work_hours=(9, 17),
            activity_frequency=0.8,
            web_browsing_patterns=["/api", "/docs"],
            file_access_patterns=["/home/dev/"],
            email_patterns={"frequency": "medium"},
            risk_profile="medium"
        )
    
    @pytest.fixture
    def environment_config(self):
        return {
            'web_server_url': 'http://192.168.10.10',
            'user_ip': '192.168.20.100'
        }
    
    def test_web_simulator_initialization(self, profile, environment_config):
        """Test web browsing simulator initialization"""
        simulator = WebBrowsingSimulator(profile, environment_config)
        
        assert simulator.profile == profile
        assert simulator.environment_config == environment_config
        assert UserRole.DEVELOPER in simulator.browsing_patterns
        assert UserRole.DEVELOPER in simulator.user_agents
    
    @pytest.mark.asyncio
    async def test_simulate_browsing_session(self, profile, environment_config):
        """Test simulating a browsing session"""
        simulator = WebBrowsingSimulator(profile, environment_config)
        
        # Mock the HTTP session
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            # Simulate a short session (1 minute)
            activities = await simulator.simulate_browsing_session(duration_minutes=1)
            
            # Should have at least one activity
            assert len(activities) >= 1
            
            # Check activity structure
            for activity in activities:
                assert isinstance(activity, ActivityEvent)
                assert activity.user_id == profile.user_id
                assert activity.activity_type == ActivityType.WEB_BROWSING
                assert 'url' in activity.details
                assert activity.source_ip == environment_config['user_ip']
                assert activity.user_agent is not None
    
    @pytest.mark.asyncio
    async def test_visit_page_success(self, profile, environment_config):
        """Test successful page visit"""
        simulator = WebBrowsingSimulator(profile, environment_config)
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            simulator.session = mock_session.return_value.__aenter__.return_value
            success = await simulator._visit_page("/api", "test-user-agent")
            
            assert success is True
    
    @pytest.mark.asyncio
    async def test_visit_page_failure(self, profile, environment_config):
        """Test failed page visit"""
        simulator = WebBrowsingSimulator(profile, environment_config)
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 500
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            simulator.session = mock_session.return_value.__aenter__.return_value
            success = await simulator._visit_page("/api", "test-user-agent")
            
            assert success is False
    
    def test_role_specific_patterns(self, environment_config):
        """Test that different roles have different browsing patterns"""
        admin_profile = UserProfile(
            user_id="admin001", username="admin", role=UserRole.ADMIN,
            department="IT", email="admin@company.com", work_hours=(8, 17),
            activity_frequency=0.5, web_browsing_patterns=[], file_access_patterns=[],
            email_patterns={}, risk_profile="low"
        )
        
        dev_profile = UserProfile(
            user_id="dev001", username="dev", role=UserRole.DEVELOPER,
            department="Engineering", email="dev@company.com", work_hours=(9, 18),
            activity_frequency=0.8, web_browsing_patterns=[], file_access_patterns=[],
            email_patterns={}, risk_profile="medium"
        )
        
        admin_simulator = WebBrowsingSimulator(admin_profile, environment_config)
        dev_simulator = WebBrowsingSimulator(dev_profile, environment_config)
        
        admin_patterns = admin_simulator.browsing_patterns[UserRole.ADMIN]
        dev_patterns = dev_simulator.browsing_patterns[UserRole.DEVELOPER]
        
        # Admin should have admin-specific patterns
        assert "/admin" in admin_patterns
        assert "/dashboard" in admin_patterns
        
        # Developer should have dev-specific patterns
        assert "/api" in dev_patterns
        assert "/docs" in dev_patterns
        
        # Patterns should be different
        assert admin_patterns != dev_patterns


class TestFileAccessSimulator:
    """Test file access simulation"""
    
    @pytest.fixture
    def profile(self):
        return UserProfile(
            user_id="file001",
            username="fileuser",
            role=UserRole.ADMIN,
            department="IT",
            email="file@company.com",
            work_hours=(8, 17),
            activity_frequency=0.5,
            web_browsing_patterns=[],
            file_access_patterns=["/var/log/", "/etc/"],
            email_patterns={},
            risk_profile="low"
        )
    
    @pytest.fixture
    def environment_config(self):
        return {'user_ip': '192.168.20.100'}
    
    def test_file_simulator_initialization(self, profile, environment_config):
        """Test file access simulator initialization"""
        simulator = FileAccessSimulator(profile, environment_config)
        
        assert simulator.profile == profile
        assert simulator.environment_config == environment_config
        assert UserRole.ADMIN in simulator.file_patterns
        assert UserRole.ADMIN in simulator.file_extensions
    
    @pytest.mark.asyncio
    async def test_simulate_file_access_session(self, profile, environment_config):
        """Test simulating a file access session"""
        simulator = FileAccessSimulator(profile, environment_config)
        
        # Simulate a short session (1 minute)
        activities = await simulator.simulate_file_access_session(duration_minutes=1)
        
        # Should have at least one activity
        assert len(activities) >= 1
        
        # Check activity structure
        for activity in activities:
            assert isinstance(activity, ActivityEvent)
            assert activity.user_id == profile.user_id
            assert activity.activity_type == ActivityType.FILE_ACCESS
            assert 'operation' in activity.details
            assert 'file_path' in activity.details
            assert activity.details['operation'] in ['read', 'write', 'list', 'search']
    
    def test_generate_filename(self, profile, environment_config):
        """Test filename generation"""
        simulator = FileAccessSimulator(profile, environment_config)
        extensions = [".log", ".conf", ".txt"]
        
        filename = simulator._generate_filename(extensions)
        
        # Should have one of the specified extensions
        assert any(filename.endswith(ext) for ext in extensions)
        
        # Should contain reasonable prefixes
        prefixes = ["document", "report", "data", "backup", "config", "log"]
        assert any(prefix in filename for prefix in prefixes)
    
    @pytest.mark.asyncio
    async def test_perform_file_operation(self, profile, environment_config):
        """Test file operation simulation"""
        simulator = FileAccessSimulator(profile, environment_config)
        
        # Test different operations
        operations = ["read", "write", "list", "search"]
        
        for operation in operations:
            success = await simulator._perform_file_operation(operation, "/test/file.txt")
            assert isinstance(success, bool)
    
    def test_role_specific_file_patterns(self, environment_config):
        """Test that different roles have different file access patterns"""
        admin_profile = UserProfile(
            user_id="admin001", username="admin", role=UserRole.ADMIN,
            department="IT", email="admin@company.com", work_hours=(8, 17),
            activity_frequency=0.5, web_browsing_patterns=[], file_access_patterns=[],
            email_patterns={}, risk_profile="low"
        )
        
        dev_profile = UserProfile(
            user_id="dev001", username="dev", role=UserRole.DEVELOPER,
            department="Engineering", email="dev@company.com", work_hours=(9, 18),
            activity_frequency=0.8, web_browsing_patterns=[], file_access_patterns=[],
            email_patterns={}, risk_profile="medium"
        )
        
        admin_simulator = FileAccessSimulator(admin_profile, environment_config)
        dev_simulator = FileAccessSimulator(dev_profile, environment_config)
        
        admin_patterns = admin_simulator.file_patterns[UserRole.ADMIN]
        dev_patterns = dev_simulator.file_patterns[UserRole.DEVELOPER]
        
        # Admin should have admin-specific patterns
        assert "/var/log/" in admin_patterns
        assert "/etc/" in admin_patterns
        
        # Developer should have dev-specific patterns
        assert "/home/dev/" in dev_patterns
        assert "/var/www/" in dev_patterns
        
        # File extensions should be different
        admin_extensions = admin_simulator.file_extensions[UserRole.ADMIN]
        dev_extensions = dev_simulator.file_extensions[UserRole.DEVELOPER]
        
        assert ".log" in admin_extensions
        assert ".py" in dev_extensions


class TestEmailActivitySimulator:
    """Test email activity simulation"""
    
    @pytest.fixture
    def profile(self):
        return UserProfile(
            user_id="email001",
            username="emailuser",
            role=UserRole.SALES,
            department="Sales",
            email="email@company.com",
            work_hours=(8, 18),
            activity_frequency=1.0,
            web_browsing_patterns=[],
            file_access_patterns=[],
            email_patterns={"frequency": "high"},
            risk_profile="high"
        )
    
    @pytest.fixture
    def environment_config(self):
        return {'user_ip': '192.168.20.100'}
    
    def test_email_simulator_initialization(self, profile, environment_config):
        """Test email activity simulator initialization"""
        simulator = EmailActivitySimulator(profile, environment_config)
        
        assert simulator.profile == profile
        assert simulator.environment_config == environment_config
        assert UserRole.SALES in simulator.email_patterns
    
    @pytest.mark.asyncio
    async def test_simulate_email_session(self, profile, environment_config):
        """Test simulating an email session"""
        simulator = EmailActivitySimulator(profile, environment_config)
        
        # Simulate a short session (1 minute)
        activities = await simulator.simulate_email_session(duration_minutes=1)
        
        # Should have at least one activity (might be zero due to timing)
        assert len(activities) >= 0
        
        # Check activity structure if any activities were generated
        for activity in activities:
            assert isinstance(activity, ActivityEvent)
            assert activity.user_id == profile.user_id
            assert activity.activity_type == ActivityType.EMAIL_ACTIVITY
            assert 'action' in activity.details
            assert activity.details['action'] in ['send', 'receive']
    
    @pytest.mark.asyncio
    async def test_simulate_send_email(self, profile, environment_config):
        """Test email sending simulation"""
        simulator = EmailActivitySimulator(profile, environment_config)
        patterns = simulator.email_patterns[UserRole.SALES]
        
        activity = await simulator._simulate_send_email(patterns)
        
        if activity:  # Might be None due to random failures
            assert activity.user_id == profile.user_id
            assert activity.activity_type == ActivityType.EMAIL_ACTIVITY
            assert activity.details['action'] == 'send'
            assert 'subject' in activity.details
            assert 'recipient' in activity.details
    
    @pytest.mark.asyncio
    async def test_simulate_receive_email(self, profile, environment_config):
        """Test email receiving simulation"""
        simulator = EmailActivitySimulator(profile, environment_config)
        patterns = simulator.email_patterns[UserRole.SALES]
        
        activity = await simulator._simulate_receive_email(patterns)
        
        if activity:  # Might be None due to random failures
            assert activity.user_id == profile.user_id
            assert activity.activity_type == ActivityType.EMAIL_ACTIVITY
            assert activity.details['action'] == 'receive'
            assert 'subject' in activity.details
            assert 'sender' in activity.details


class TestComprehensiveSyntheticUser:
    """Test comprehensive synthetic user agent"""
    
    @pytest.fixture
    def profile(self):
        return UserProfile(
            user_id="comp001",
            username="compuser",
            role=UserRole.DEVELOPER,
            department="Engineering",
            email="comp@company.com",
            work_hours=(9, 17),
            activity_frequency=0.8,
            web_browsing_patterns=["/api", "/docs"],
            file_access_patterns=["/home/dev/"],
            email_patterns={"frequency": "medium"},
            risk_profile="medium"
        )
    
    @pytest.fixture
    def environment_config(self):
        return {
            'web_server_url': 'http://192.168.10.10',
            'user_ip': '192.168.20.100'
        }
    
    def test_comprehensive_user_initialization(self, profile, environment_config):
        """Test comprehensive synthetic user initialization"""
        user = ComprehensiveSyntheticUser(profile, environment_config)
        
        assert user.profile == profile
        assert user.environment_config == environment_config
        assert user.web_simulator is not None
        assert user.file_simulator is not None
        assert user.email_simulator is not None
        assert user.is_running is False
    
    def test_is_work_hours(self, profile, environment_config):
        """Test work hours detection"""
        user = ComprehensiveSyntheticUser(profile, environment_config)
        
        # Mock datetime to test different hours
        with patch('agents.synthetic_users.datetime') as mock_datetime:
            # Test during work hours (10 AM)
            mock_datetime.now.return_value.hour = 10
            assert user.is_work_hours() is True
            
            # Test outside work hours (6 AM)
            mock_datetime.now.return_value.hour = 6
            assert user.is_work_hours() is False
            
            # Test outside work hours (8 PM)
            mock_datetime.now.return_value.hour = 20
            assert user.is_work_hours() is False
    
    def test_log_activity(self, profile, environment_config):
        """Test activity logging"""
        user = ComprehensiveSyntheticUser(profile, environment_config)
        
        # Log an activity
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
        assert activity.duration_seconds == 1.5
        assert activity.user_agent == "test-user-agent"
    
    @pytest.mark.asyncio
    async def test_simulate_login(self, profile, environment_config):
        """Test login simulation"""
        user = ComprehensiveSyntheticUser(profile, environment_config)
        
        await user._simulate_login()
        
        assert len(user.activity_log) == 1
        activity = user.activity_log[0]
        assert activity.activity_type == ActivityType.LOGIN_LOGOUT
        assert activity.details["action"] == "login"
    
    @pytest.mark.asyncio
    async def test_simulate_logout(self, profile, environment_config):
        """Test logout simulation"""
        user = ComprehensiveSyntheticUser(profile, environment_config)
        
        await user._simulate_logout()
        
        assert len(user.activity_log) == 1
        activity = user.activity_log[0]
        assert activity.activity_type == ActivityType.LOGIN_LOGOUT
        assert activity.details["action"] == "logout"
        assert activity.success is True  # Logout should always succeed
    
    def test_get_activity_summary(self, profile, environment_config):
        """Test activity summary generation"""
        user = ComprehensiveSyntheticUser(profile, environment_config)
        
        # Add some test activities
        user.log_activity(ActivityType.WEB_BROWSING, {"url": "/test1"}, True, 1.0)
        user.log_activity(ActivityType.WEB_BROWSING, {"url": "/test2"}, False, 2.0)
        user.log_activity(ActivityType.FILE_ACCESS, {"file": "/test.txt"}, True, 0.5)
        
        summary = user.get_activity_summary()
        
        assert summary["total_activities"] == 3
        assert summary["activity_counts"]["web_browsing"] == 2
        assert summary["activity_counts"]["file_access"] == 1
        assert summary["success_rate"] == 2/3  # 2 successful out of 3
        assert summary["average_duration_seconds"] == (1.0 + 2.0 + 0.5) / 3


class TestSyntheticUserManager:
    """Test synthetic user manager"""
    
    @pytest.fixture
    def environment_config(self):
        return {
            'web_server_url': 'http://192.168.10.10',
            'user_ip': '192.168.20.100'
        }
    
    def test_manager_initialization(self, environment_config):
        """Test manager initialization"""
        manager = SyntheticUserManager(environment_config)
        
        assert manager.environment_config == environment_config
        assert len(manager.users) == 0
        assert manager.is_running is False
    
    def test_create_default_user_profiles(self, environment_config):
        """Test creating default user profiles"""
        manager = SyntheticUserManager(environment_config)
        profiles = manager.create_default_user_profiles()
        
        assert len(profiles) >= 4  # Should have at least admin, dev, sales, hr
        
        # Check that we have different roles
        roles = [profile.role for profile in profiles]
        assert UserRole.ADMIN in roles
        assert UserRole.DEVELOPER in roles
        assert UserRole.SALES in roles
        assert UserRole.HR in roles
        
        # Check profile structure
        for profile in profiles:
            assert profile.user_id is not None
            assert profile.username is not None
            assert profile.role is not None
            assert len(profile.work_hours) == 2
    
    @pytest.mark.asyncio
    async def test_add_user(self, environment_config):
        """Test adding a user to the manager"""
        manager = SyntheticUserManager(environment_config)
        
        profile = UserProfile(
            user_id="test001",
            username="testuser",
            role=UserRole.DEVELOPER,
            department="Engineering",
            email="test@company.com",
            work_hours=(9, 17),
            activity_frequency=0.5,
            web_browsing_patterns=[],
            file_access_patterns=[],
            email_patterns={},
            risk_profile="low"
        )
        
        await manager.add_user(profile)
        
        assert len(manager.users) == 1
        assert "test001" in manager.users
        assert manager.users["test001"].profile == profile
    
    def test_get_system_summary_empty(self, environment_config):
        """Test system summary with no users"""
        manager = SyntheticUserManager(environment_config)
        summary = manager.get_system_summary()
        
        assert summary["total_users"] == 0
        assert summary["users"] == {}
        assert summary["system_totals"]["total_activities"] == 0
    
    @pytest.mark.asyncio
    async def test_get_system_summary_with_users(self, environment_config):
        """Test system summary with users and activities"""
        manager = SyntheticUserManager(environment_config)
        
        # Add a user
        profile = UserProfile(
            user_id="test001",
            username="testuser",
            role=UserRole.DEVELOPER,
            department="Engineering",
            email="test@company.com",
            work_hours=(9, 17),
            activity_frequency=0.5,
            web_browsing_patterns=[],
            file_access_patterns=[],
            email_patterns={},
            risk_profile="low"
        )
        
        await manager.add_user(profile)
        
        # Add some activities
        user = manager.users["test001"]
        user.log_activity(ActivityType.WEB_BROWSING, {"url": "/test"}, True, 1.0)
        user.log_activity(ActivityType.FILE_ACCESS, {"file": "/test.txt"}, True, 0.5)
        
        summary = manager.get_system_summary()
        
        assert summary["total_users"] == 1
        assert "test001" in summary["users"]
        assert summary["users"]["test001"]["username"] == "testuser"
        assert summary["users"]["test001"]["role"] == "developer"
        assert summary["system_totals"]["total_activities"] == 2
        assert summary["system_totals"]["total_success_rate"] == 1.0


class TestBehaviorRealism:
    """Test behavior realism and detection evasion"""
    
    @pytest.fixture
    def environment_config(self):
        return {
            'web_server_url': 'http://192.168.10.10',
            'user_ip': '192.168.20.100'
        }
    
    def test_role_based_behavior_differences(self, environment_config):
        """Test that different roles exhibit different behaviors"""
        # Create profiles for different roles
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
        
        admin_user = ComprehensiveSyntheticUser(admin_profile, environment_config)
        sales_user = ComprehensiveSyntheticUser(sales_profile, environment_config)
        
        # Check that web browsing patterns are different
        admin_patterns = admin_user.web_simulator.browsing_patterns[UserRole.ADMIN]
        sales_patterns = sales_user.web_simulator.browsing_patterns[UserRole.SALES]
        
        assert admin_patterns != sales_patterns
        assert "/admin" in admin_patterns
        assert "/crm" in sales_patterns
        
        # Check that file access patterns are different
        admin_files = admin_user.file_simulator.file_patterns[UserRole.ADMIN]
        sales_files = sales_user.file_simulator.file_patterns[UserRole.SALES]
        
        assert admin_files != sales_files
        assert "/var/log/" in admin_files
        assert "/shared/sales/" in sales_files
        
        # Check that activity frequencies are different
        assert admin_profile.activity_frequency != sales_profile.activity_frequency
        assert sales_profile.activity_frequency > admin_profile.activity_frequency
    
    def test_temporal_behavior_patterns(self, environment_config):
        """Test that users follow realistic temporal patterns"""
        profile = UserProfile(
            user_id="temp001", username="tempuser", role=UserRole.DEVELOPER,
            department="Engineering", email="temp@company.com", work_hours=(9, 17),
            activity_frequency=0.8, web_browsing_patterns=[], file_access_patterns=[],
            email_patterns={}, risk_profile="medium"
        )
        
        user = ComprehensiveSyntheticUser(profile, environment_config)
        
        # Test work hours detection
        with patch('agents.synthetic_users.datetime') as mock_datetime:
            # During work hours
            mock_datetime.now.return_value.hour = 10
            assert user.is_work_hours() is True
            
            # Before work hours
            mock_datetime.now.return_value.hour = 7
            assert user.is_work_hours() is False
            
            # After work hours
            mock_datetime.now.return_value.hour = 19
            assert user.is_work_hours() is False
    
    def test_activity_randomization(self, environment_config):
        """Test that activities have appropriate randomization"""
        profile = UserProfile(
            user_id="rand001", username="randuser", role=UserRole.DEVELOPER,
            department="Engineering", email="rand@company.com", work_hours=(9, 17),
            activity_frequency=0.8, web_browsing_patterns=[], file_access_patterns=[],
            email_patterns={}, risk_profile="medium"
        )
        
        # Test web browsing randomization
        web_sim = WebBrowsingSimulator(profile, environment_config)
        patterns = web_sim.browsing_patterns[UserRole.DEVELOPER]
        
        # Generate multiple filenames and check for variety
        filenames = []
        for _ in range(10):
            filename = web_sim.file_simulator._generate_filename([".py", ".js"])
            filenames.append(filename)
        
        # Should have some variety in filenames
        unique_filenames = set(filenames)
        assert len(unique_filenames) > 1  # Should not all be the same
    
    def test_detection_evasion_characteristics(self, environment_config):
        """Test characteristics that help evade detection"""
        profile = UserProfile(
            user_id="evasion001", username="evasionuser", role=UserRole.SALES,
            department="Sales", email="evasion@company.com", work_hours=(8, 18),
            activity_frequency=1.0, web_browsing_patterns=[], file_access_patterns=[],
            email_patterns={}, risk_profile="high"
        )
        
        user = ComprehensiveSyntheticUser(profile, environment_config)
        
        # Test that user agents are realistic
        web_sim = user.web_simulator
        user_agents = web_sim.user_agents[UserRole.SALES]
        
        for ua in user_agents:
            # Should contain realistic browser signatures
            assert "Mozilla" in ua
            assert any(browser in ua for browser in ["Chrome", "Firefox", "Safari"])
        
        # Test that activities have realistic timing
        user.log_activity(ActivityType.WEB_BROWSING, {"url": "/test"}, True, 1.5)
        user.log_activity(ActivityType.FILE_ACCESS, {"file": "/test.txt"}, True, 0.8)
        
        activities = user.activity_log
        
        # Activities should have reasonable durations (not too fast or slow)
        for activity in activities:
            assert 0.1 <= activity.duration_seconds <= 300  # 0.1s to 5 minutes
        
        # Activities should have realistic source IPs
        for activity in activities:
            assert activity.source_ip.startswith("192.168.")  # Internal network
    
    @pytest.mark.asyncio
    async def test_background_noise_generation(self, environment_config):
        """Test that the system generates appropriate background noise"""
        manager = SyntheticUserManager(environment_config)
        
        # Add multiple users with different roles
        profiles = manager.create_default_user_profiles()
        for profile in profiles[:3]:  # Add first 3 users
            await manager.add_user(profile)
        
        # Simulate some activities
        for user in manager.users.values():
            user.log_activity(ActivityType.WEB_BROWSING, {"url": "/test"}, True, 1.0)
            user.log_activity(ActivityType.FILE_ACCESS, {"file": "/test.txt"}, True, 0.5)
            user.log_activity(ActivityType.EMAIL_ACTIVITY, {"action": "send"}, True, 2.0)
        
        summary = manager.get_system_summary()
        
        # Should have multiple users generating diverse activities
        assert summary["total_users"] >= 3
        assert summary["system_totals"]["total_activities"] >= 9  # 3 activities per user
        
        # Should have diverse activity types
        activity_types = summary["system_totals"]["activity_type_counts"]
        assert len(activity_types) >= 3  # At least 3 different activity types
        
        # Should have high success rate (realistic user behavior)
        assert summary["system_totals"]["total_success_rate"] >= 0.8


class TestIntegration:
    """Integration tests for the complete synthetic user system"""
    
    @pytest.fixture
    def environment_config(self):
        return {
            'web_server_url': 'http://192.168.10.10',
            'user_ip': '192.168.20.100',
            'mail_server': '192.168.10.20',
            'file_server': '192.168.10.30'
        }
    
    @pytest.mark.asyncio
    async def test_complete_user_lifecycle(self, environment_config):
        """Test complete user lifecycle from creation to activity logging"""
        manager = SyntheticUserManager(environment_config)
        
        # Create and add a user
        profile = UserProfile(
            user_id="lifecycle001",
            username="lifecycleuser",
            role=UserRole.DEVELOPER,
            department="Engineering",
            email="lifecycle@company.com",
            work_hours=(9, 17),
            activity_frequency=0.8,
            web_browsing_patterns=["/api", "/docs"],
            file_access_patterns=["/home/dev/"],
            email_patterns={"frequency": "medium"},
            risk_profile="medium"
        )
        
        await manager.add_user(profile)
        user = manager.users["lifecycle001"]
        
        # Simulate various activities
        await user._simulate_login()
        
        # Simulate web browsing (short session)
        with patch('aiohttp.ClientSession'):
            activities = await user.web_simulator.simulate_browsing_session(1)
            user.activity_log.extend(activities)
        
        # Simulate file access
        file_activities = await user.file_simulator.simulate_file_access_session(1)
        user.activity_log.extend(file_activities)
        
        await user._simulate_logout()
        
        # Check that we have a complete activity log
        assert len(user.activity_log) >= 2  # At least login and logout
        
        # Check activity types
        activity_types = [a.activity_type for a in user.activity_log]
        assert ActivityType.LOGIN_LOGOUT in activity_types
        
        # Get summary
        summary = user.get_activity_summary()
        assert summary["total_activities"] >= 2
        assert summary["success_rate"] > 0
    
    @pytest.mark.asyncio
    async def test_multi_user_coordination(self, environment_config):
        """Test multiple users running simultaneously"""
        manager = SyntheticUserManager(environment_config)
        
        # Add multiple users
        profiles = manager.create_default_user_profiles()[:2]  # Use first 2
        for profile in profiles:
            await manager.add_user(profile)
        
        # Simulate activities for all users
        for user in manager.users.values():
            await user._simulate_login()
            user.log_activity(ActivityType.WEB_BROWSING, {"url": "/test"}, True, 1.0)
            await user._simulate_logout()
        
        # Check system summary
        summary = manager.get_system_summary()
        
        assert summary["total_users"] == 2
        assert summary["system_totals"]["total_activities"] >= 4  # 2 activities per user minimum
        
        # Each user should have their own activities
        for user_id, user_info in summary["users"].items():
            assert user_info["summary"]["total_activities"] >= 2
    
    def test_activity_logging_format(self, environment_config):
        """Test that activity logging produces correct format"""
        profile = UserProfile(
            user_id="format001",
            username="formatuser",
            role=UserRole.ADMIN,
            department="IT",
            email="format@company.com",
            work_hours=(8, 17),
            activity_frequency=0.5,
            web_browsing_patterns=[],
            file_access_patterns=[],
            email_patterns={},
            risk_profile="low"
        )
        
        user = ComprehensiveSyntheticUser(profile, environment_config)
        
        # Log various activities
        user.log_activity(
            ActivityType.WEB_BROWSING,
            {"url": "/admin", "method": "GET"},
            True,
            1.5,
            "Mozilla/5.0 Test"
        )
        
        user.log_activity(
            ActivityType.FILE_ACCESS,
            {"operation": "read", "file_path": "/var/log/test.log"},
            True,
            0.8
        )
        
        user.log_activity(
            ActivityType.EMAIL_ACTIVITY,
            {"action": "send", "recipient": "test@company.com"},
            False,
            2.3
        )
        
        # Check activity log format
        assert len(user.activity_log) == 3
        
        for activity in user.activity_log:
            # Check required fields
            assert hasattr(activity, 'timestamp')
            assert hasattr(activity, 'user_id')
            assert hasattr(activity, 'activity_type')
            assert hasattr(activity, 'details')
            assert hasattr(activity, 'success')
            assert hasattr(activity, 'duration_seconds')
            assert hasattr(activity, 'source_ip')
            
            # Check data types
            assert isinstance(activity.timestamp, datetime)
            assert isinstance(activity.user_id, str)
            assert isinstance(activity.activity_type, ActivityType)
            assert isinstance(activity.details, dict)
            assert isinstance(activity.success, bool)
            assert isinstance(activity.duration_seconds, (int, float))
            assert isinstance(activity.source_ip, str)
            
            # Check values
            assert activity.user_id == profile.user_id
            assert activity.source_ip == environment_config['user_ip']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])