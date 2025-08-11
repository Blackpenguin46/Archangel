"""
Configuration for Synthetic User Simulation System

This module provides configuration management for synthetic users,
including environment settings, user profiles, and behavior patterns.
"""

import json
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from agents.synthetic_users import UserProfile, UserRole

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """Configuration for the mock enterprise environment"""
    web_server_url: str = "http://192.168.10.10"
    mail_server_url: str = "http://192.168.10.20"
    file_server_url: str = "http://192.168.10.30"
    database_server_url: str = "http://192.168.10.40"
    user_network_range: str = "192.168.20.0/24"
    default_user_ip: str = "192.168.20.100"
    
    # Service ports
    web_port: int = 80
    mail_port: int = 25
    file_port: int = 445
    database_port: int = 3306
    
    # Authentication settings
    domain_name: str = "company.local"
    default_password: str = "Password123!"
    
    # Logging settings
    log_level: str = "INFO"
    activity_log_file: str = "synthetic_user_activities.jsonl"
    enable_file_logging: bool = True
    
    # Simulation settings
    simulation_speed: float = 1.0  # 1.0 = real-time, 2.0 = 2x speed
    enable_background_noise: bool = True
    max_concurrent_users: int = 50


@dataclass
class BehaviorConfig:
    """Configuration for user behavior patterns"""
    # Activity timing (in seconds)
    min_activity_interval: int = 30
    max_activity_interval: int = 3600
    
    # Session durations (in minutes)
    min_web_session_duration: int = 5
    max_web_session_duration: int = 60
    min_file_session_duration: int = 2
    max_file_session_duration: int = 30
    min_email_session_duration: int = 10
    max_email_session_duration: int = 120
    
    # Success rates
    web_browsing_success_rate: float = 0.95
    file_access_success_rate: float = 0.90
    email_success_rate: float = 0.98
    login_success_rate: float = 0.98
    
    # Risk behavior settings
    risky_behavior_probability: Dict[str, float] = None
    
    def __post_init__(self):
        if self.risky_behavior_probability is None:
            self.risky_behavior_probability = {
                "low": 0.05,    # 5% chance of risky behavior
                "medium": 0.15, # 15% chance of risky behavior
                "high": 0.30    # 30% chance of risky behavior
            }


class SyntheticUserConfigManager:
    """Manages configuration for synthetic users"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "synthetic_user_config.yaml"
        self.environment_config = EnvironmentConfig()
        self.behavior_config = BehaviorConfig()
        self.user_profiles: List[UserProfile] = []
        
        # Load configuration if file exists
        if Path(self.config_file).exists():
            self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Load environment config
            if 'environment' in config_data:
                env_data = config_data['environment']
                self.environment_config = EnvironmentConfig(**env_data)
            
            # Load behavior config
            if 'behavior' in config_data:
                behavior_data = config_data['behavior']
                self.behavior_config = BehaviorConfig(**behavior_data)
            
            # Load user profiles
            if 'user_profiles' in config_data:
                self.user_profiles = []
                for profile_data in config_data['user_profiles']:
                    # Convert role string to enum
                    if 'role' in profile_data:
                        profile_data['role'] = UserRole(profile_data['role'])
                    
                    profile = UserProfile(**profile_data)
                    self.user_profiles.append(profile)
            
            logger.info(f"Loaded configuration from {self.config_file}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
    
    def save_config(self) -> None:
        """Save current configuration to file"""
        try:
            config_data = {
                'environment': asdict(self.environment_config),
                'behavior': asdict(self.behavior_config),
                'user_profiles': []
            }
            
            # Convert user profiles to dict format
            for profile in self.user_profiles:
                profile_dict = asdict(profile)
                profile_dict['role'] = profile.role.value  # Convert enum to string
                config_data['user_profiles'].append(profile_dict)
            
            with open(self.config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved configuration to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def create_default_config(self) -> None:
        """Create default configuration with sample user profiles"""
        self.environment_config = EnvironmentConfig()
        self.behavior_config = BehaviorConfig()
        
        # Create default user profiles
        self.user_profiles = [
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
                        "System Maintenance", "Security Alert", "Backup Report",
                        "Server Status", "User Access Request", "Policy Update"
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
                        "Code Review", "Bug Report", "Feature Request", "Deployment",
                        "Meeting Notes", "Technical Discussion"
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
                        "Proposal", "Client Meeting", "Quote Request", "Follow-up",
                        "Contract", "Sales Report"
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
                        "Employee Onboarding", "Policy Update", "Benefits Enrollment",
                        "Performance Review", "Recruitment", "Training Schedule"
                    ]
                },
                risk_profile="low"
            ),
            UserProfile(
                user_id="finance001",
                username="accountant",
                role=UserRole.FINANCE,
                department="Finance",
                email="finance@company.com",
                work_hours=(8, 17),
                activity_frequency=0.7,
                web_browsing_patterns=[
                    "/finance", "/accounting", "/reports", "/budgets", "/expenses",
                    "/quickbooks.com", "/banking", "/invoices"
                ],
                file_access_patterns=[
                    "/home/finance/", "/shared/finance/", "/accounting/", "/budgets/",
                    "/expenses/", "/reports/", "/invoices/"
                ],
                email_patterns={
                    "send_frequency": 0.6,
                    "receive_frequency": 1.0,
                    "common_subjects": [
                        "Budget Report", "Expense Approval", "Invoice Processing",
                        "Financial Analysis", "Audit Request", "Payment Authorization"
                    ]
                },
                risk_profile="low"
            ),
            UserProfile(
                user_id="marketing001",
                username="marketer",
                role=UserRole.MARKETING,
                department="Marketing",
                email="marketing@company.com",
                work_hours=(9, 18),
                activity_frequency=0.9,
                web_browsing_patterns=[
                    "/marketing", "/campaigns", "/analytics", "/social",
                    "/google.com/analytics", "/facebook.com", "/twitter.com"
                ],
                file_access_patterns=[
                    "/home/marketing/", "/shared/marketing/", "/campaigns/", "/assets/",
                    "/analytics/", "/social/", "/content/"
                ],
                email_patterns={
                    "send_frequency": 1.2,
                    "receive_frequency": 1.8,
                    "common_subjects": [
                        "Campaign Launch", "Analytics Report", "Content Review",
                        "Social Media Update", "Brand Guidelines", "Event Planning"
                    ]
                },
                risk_profile="medium"
            ),
            UserProfile(
                user_id="support001",
                username="support_agent",
                role=UserRole.SUPPORT,
                department="Customer Support",
                email="support@company.com",
                work_hours=(8, 20),  # Extended hours for support
                activity_frequency=1.2,
                web_browsing_patterns=[
                    "/support", "/tickets", "/knowledge-base", "/chat",
                    "/zendesk.com", "/freshdesk.com", "/customer-portal"
                ],
                file_access_patterns=[
                    "/home/support/", "/shared/support/", "/tickets/", "/knowledge/",
                    "/scripts/", "/logs/", "/customer-data/"
                ],
                email_patterns={
                    "send_frequency": 2.0,
                    "receive_frequency": 3.0,
                    "common_subjects": [
                        "Ticket Update", "Customer Inquiry", "Issue Resolution",
                        "Escalation", "Knowledge Base Update", "Customer Feedback"
                    ]
                },
                risk_profile="medium"
            )
        ]
    
    def add_user_profile(self, profile: UserProfile) -> None:
        """Add a user profile to the configuration"""
        self.user_profiles.append(profile)
        logger.info(f"Added user profile: {profile.username}")
    
    def remove_user_profile(self, user_id: str) -> bool:
        """Remove a user profile by user_id"""
        for i, profile in enumerate(self.user_profiles):
            if profile.user_id == user_id:
                removed_profile = self.user_profiles.pop(i)
                logger.info(f"Removed user profile: {removed_profile.username}")
                return True
        return False
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get a user profile by user_id"""
        for profile in self.user_profiles:
            if profile.user_id == user_id:
                return profile
        return None
    
    def get_profiles_by_role(self, role: UserRole) -> List[UserProfile]:
        """Get all user profiles with a specific role"""
        return [profile for profile in self.user_profiles if profile.role == role]
    
    def get_profiles_by_department(self, department: str) -> List[UserProfile]:
        """Get all user profiles in a specific department"""
        return [profile for profile in self.user_profiles if profile.department == department]
    
    def update_environment_config(self, **kwargs) -> None:
        """Update environment configuration"""
        for key, value in kwargs.items():
            if hasattr(self.environment_config, key):
                setattr(self.environment_config, key, value)
                logger.info(f"Updated environment config: {key} = {value}")
    
    def update_behavior_config(self, **kwargs) -> None:
        """Update behavior configuration"""
        for key, value in kwargs.items():
            if hasattr(self.behavior_config, key):
                setattr(self.behavior_config, key, value)
                logger.info(f"Updated behavior config: {key} = {value}")
    
    def validate_config(self) -> List[str]:
        """Validate the current configuration and return any errors"""
        errors = []
        
        # Validate environment config
        if not self.environment_config.web_server_url:
            errors.append("Web server URL is required")
        
        if self.environment_config.simulation_speed <= 0:
            errors.append("Simulation speed must be positive")
        
        if self.environment_config.max_concurrent_users <= 0:
            errors.append("Max concurrent users must be positive")
        
        # Validate behavior config
        if self.behavior_config.min_activity_interval >= self.behavior_config.max_activity_interval:
            errors.append("Min activity interval must be less than max activity interval")
        
        if not (0 <= self.behavior_config.web_browsing_success_rate <= 1):
            errors.append("Web browsing success rate must be between 0 and 1")
        
        # Validate user profiles
        user_ids = set()
        usernames = set()
        
        for profile in self.user_profiles:
            # Check for duplicate user IDs
            if profile.user_id in user_ids:
                errors.append(f"Duplicate user ID: {profile.user_id}")
            user_ids.add(profile.user_id)
            
            # Check for duplicate usernames
            if profile.username in usernames:
                errors.append(f"Duplicate username: {profile.username}")
            usernames.add(profile.username)
            
            # Validate work hours
            if len(profile.work_hours) != 2:
                errors.append(f"Invalid work hours for {profile.username}: must be (start, end)")
            elif profile.work_hours[0] >= profile.work_hours[1]:
                errors.append(f"Invalid work hours for {profile.username}: start must be before end")
            
            # Validate activity frequency
            if profile.activity_frequency <= 0:
                errors.append(f"Activity frequency must be positive for {profile.username}")
            
            # Validate risk profile
            if profile.risk_profile not in ["low", "medium", "high"]:
                errors.append(f"Invalid risk profile for {profile.username}: must be low, medium, or high")
        
        return errors
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration"""
        return {
            "environment": {
                "web_server_url": self.environment_config.web_server_url,
                "user_network_range": self.environment_config.user_network_range,
                "simulation_speed": self.environment_config.simulation_speed,
                "max_concurrent_users": self.environment_config.max_concurrent_users
            },
            "behavior": {
                "activity_intervals": f"{self.behavior_config.min_activity_interval}-{self.behavior_config.max_activity_interval}s",
                "success_rates": {
                    "web_browsing": self.behavior_config.web_browsing_success_rate,
                    "file_access": self.behavior_config.file_access_success_rate,
                    "email": self.behavior_config.email_success_rate
                }
            },
            "users": {
                "total_profiles": len(self.user_profiles),
                "roles": {role.value: len(self.get_profiles_by_role(role)) 
                         for role in UserRole},
                "departments": list(set(profile.department for profile in self.user_profiles))
            }
        }


def create_sample_config_file(filename: str = "synthetic_user_config.yaml") -> None:
    """Create a sample configuration file"""
    config_manager = SyntheticUserConfigManager()
    config_manager.create_default_config()
    config_manager.config_file = filename
    config_manager.save_config()
    print(f"Created sample configuration file: {filename}")


if __name__ == "__main__":
    # Create a sample configuration file
    create_sample_config_file()
    
    # Load and validate the configuration
    config_manager = SyntheticUserConfigManager("synthetic_user_config.yaml")
    
    errors = config_manager.validate_config()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid")
    
    # Print configuration summary
    summary = config_manager.get_config_summary()
    print("\nConfiguration Summary:")
    print(json.dumps(summary, indent=2))