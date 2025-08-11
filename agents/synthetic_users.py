"""
Synthetic User Simulation System

This module implements autonomous synthetic user agents that simulate realistic
employee behaviors including web browsing, file access, and email activity.
The goal is to create background noise that makes Red Team reconnaissance
more challenging while providing realistic enterprise activity patterns.
"""

import asyncio
import random
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import aiofiles
from pathlib import Path
import smtplib
import imaplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import schedule
import time
from abc import ABC, abstractmethod

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
    user_agent: Optional[str] = None


class SyntheticUserAgent(ABC):
    """Base class for synthetic user agents"""
    
    def __init__(self, profile: UserProfile, environment_config: Dict[str, Any]):
        self.profile = profile
        self.environment_config = environment_config
        self.is_active = False
        self.current_session = None
        self.activity_log: List[ActivityEvent] = []
        
    @abstractmethod
    async def start_activity_cycle(self) -> None:
        """Start the user's activity cycle"""
        pass
    
    @abstractmethod
    async def stop_activity_cycle(self) -> None:
        """Stop the user's activity cycle"""
        pass
    
    def is_work_hours(self) -> bool:
        """Check if current time is within user's work hours"""
        current_hour = datetime.now().hour
        start_hour, end_hour = self.profile.work_hours
        return start_hour <= current_hour <= end_hour
    
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
            source_ip=self.environment_config.get('user_ip', '192.168.20.100'),
            user_agent=user_agent
        )
        self.activity_log.append(event)
        logger.info(f"User {self.profile.username} performed {activity_type.value}: {success}")


class WebBrowsingSimulator:
    """Simulates realistic web browsing behavior"""
    
    def __init__(self, profile: UserProfile, environment_config: Dict[str, Any]):
        self.profile = profile
        self.environment_config = environment_config
        self.session = None
        
        # Common user agents for different roles
        self.user_agents = {
            UserRole.ADMIN: [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
            ],
            UserRole.DEVELOPER: [
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            ],
            UserRole.SALES: [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            ],
            UserRole.HR: [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            ]
        }
        
        # Role-specific browsing patterns
        self.browsing_patterns = {
            UserRole.ADMIN: [
                "/admin", "/dashboard", "/system", "/logs", "/monitoring",
                "/users", "/settings", "/backup", "/security"
            ],
            UserRole.DEVELOPER: [
                "/api", "/docs", "/git", "/jenkins", "/jira", "/confluence",
                "/stackoverflow.com", "/github.com", "/dev-tools"
            ],
            UserRole.SALES: [
                "/crm", "/leads", "/reports", "/customers", "/deals",
                "/linkedin.com", "/salesforce.com", "/proposals"
            ],
            UserRole.HR: [
                "/hr", "/employees", "/payroll", "/benefits", "/recruiting",
                "/workday.com", "/bamboohr.com", "/applicants"
            ],
            UserRole.FINANCE: [
                "/finance", "/accounting", "/reports", "/budgets", "/expenses",
                "/quickbooks.com", "/banking", "/invoices"
            ],
            UserRole.MARKETING: [
                "/marketing", "/campaigns", "/analytics", "/social",
                "/google.com/analytics", "/facebook.com", "/twitter.com"
            ]
        }
    
    async def simulate_browsing_session(self, duration_minutes: int = 30) -> List[ActivityEvent]:
        """Simulate a web browsing session"""
        activities = []
        session_start = datetime.now()
        
        try:
            async with aiohttp.ClientSession() as session:
                self.session = session
                
                # Get role-specific patterns
                patterns = self.browsing_patterns.get(self.profile.role, ["/"])
                user_agent = random.choice(self.user_agents.get(self.profile.role, 
                                         self.user_agents[UserRole.ADMIN]))
                
                # Simulate browsing for the specified duration
                end_time = session_start + timedelta(minutes=duration_minutes)
                
                while datetime.now() < end_time:
                    # Select a random page to visit
                    page = random.choice(patterns)
                    
                    # Add some randomness to the URL
                    if random.random() < 0.3:  # 30% chance of adding parameters
                        page += f"?id={random.randint(1, 1000)}"
                    
                    # Simulate page visit
                    activity_start = time.time()
                    success = await self._visit_page(page, user_agent)
                    duration = time.time() - activity_start
                    
                    # Log the activity
                    activity = ActivityEvent(
                        timestamp=datetime.now(),
                        user_id=self.profile.user_id,
                        activity_type=ActivityType.WEB_BROWSING,
                        details={"url": page, "method": "GET"},
                        success=success,
                        duration_seconds=duration,
                        source_ip=self.environment_config.get('user_ip', '192.168.20.100'),
                        user_agent=user_agent
                    )
                    activities.append(activity)
                    
                    # Random delay between page visits (1-10 seconds)
                    await asyncio.sleep(random.uniform(1, 10))
                    
        except Exception as e:
            logger.error(f"Error in browsing session for {self.profile.username}: {e}")
        
        return activities
    
    async def _visit_page(self, page: str, user_agent: str) -> bool:
        """Visit a specific page"""
        try:
            # Construct full URL
            base_url = self.environment_config.get('web_server_url', 'http://192.168.10.10')
            url = f"{base_url}{page}"
            
            headers = {
                'User-Agent': user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            async with self.session.get(url, headers=headers, timeout=10) as response:
                # Consider 2xx and 3xx as success
                return 200 <= response.status < 400
                
        except Exception as e:
            logger.debug(f"Failed to visit {page}: {e}")
            return False


class FileAccessSimulator:
    """Simulates realistic file system access patterns"""
    
    def __init__(self, profile: UserProfile, environment_config: Dict[str, Any]):
        self.profile = profile
        self.environment_config = environment_config
        
        # Role-specific file access patterns
        self.file_patterns = {
            UserRole.ADMIN: [
                "/var/log/", "/etc/", "/home/admin/", "/opt/", "/usr/local/",
                "/backup/", "/scripts/", "/config/"
            ],
            UserRole.DEVELOPER: [
                "/home/dev/", "/var/www/", "/opt/code/", "/git/", "/build/",
                "/src/", "/docs/", "/tests/"
            ],
            UserRole.SALES: [
                "/home/sales/", "/shared/sales/", "/crm/", "/proposals/",
                "/contracts/", "/leads/", "/reports/"
            ],
            UserRole.HR: [
                "/home/hr/", "/shared/hr/", "/employees/", "/payroll/",
                "/benefits/", "/policies/", "/recruiting/"
            ],
            UserRole.FINANCE: [
                "/home/finance/", "/shared/finance/", "/accounting/", "/budgets/",
                "/expenses/", "/reports/", "/invoices/"
            ]
        }
        
        # Common file extensions by role
        self.file_extensions = {
            UserRole.ADMIN: [".log", ".conf", ".sh", ".txt", ".cfg"],
            UserRole.DEVELOPER: [".py", ".js", ".html", ".css", ".json", ".md", ".sql"],
            UserRole.SALES: [".pdf", ".docx", ".xlsx", ".pptx", ".txt"],
            UserRole.HR: [".pdf", ".docx", ".xlsx", ".txt"],
            UserRole.FINANCE: [".xlsx", ".pdf", ".csv", ".txt"]
        }
    
    async def simulate_file_access_session(self, duration_minutes: int = 20) -> List[ActivityEvent]:
        """Simulate a file access session"""
        activities = []
        session_start = datetime.now()
        end_time = session_start + timedelta(minutes=duration_minutes)
        
        # Get role-specific patterns
        patterns = self.file_patterns.get(self.profile.role, ["/home/user/"])
        extensions = self.file_extensions.get(self.profile.role, [".txt"])
        
        while datetime.now() < end_time:
            # Select random file operation
            operation = random.choice(["read", "write", "list", "search"])
            
            # Generate file path
            base_path = random.choice(patterns)
            filename = self._generate_filename(extensions)
            file_path = f"{base_path}{filename}"
            
            # Simulate file operation
            activity_start = time.time()
            success = await self._perform_file_operation(operation, file_path)
            duration = time.time() - activity_start
            
            # Log the activity
            activity = ActivityEvent(
                timestamp=datetime.now(),
                user_id=self.profile.user_id,
                activity_type=ActivityType.FILE_ACCESS,
                details={
                    "operation": operation,
                    "file_path": file_path,
                    "file_size": random.randint(1024, 1024*1024)  # 1KB to 1MB
                },
                success=success,
                duration_seconds=duration,
                source_ip=self.environment_config.get('user_ip', '192.168.20.100')
            )
            activities.append(activity)
            
            # Random delay between operations (5-30 seconds)
            await asyncio.sleep(random.uniform(5, 30))
        
        return activities
    
    def _generate_filename(self, extensions: List[str]) -> str:
        """Generate a realistic filename"""
        prefixes = [
            "document", "report", "data", "backup", "config", "log",
            "project", "meeting", "proposal", "analysis", "summary"
        ]
        
        prefix = random.choice(prefixes)
        suffix = random.choice(extensions)
        timestamp = datetime.now().strftime("%Y%m%d")
        
        # Sometimes add timestamp, sometimes add random number
        if random.random() < 0.5:
            return f"{prefix}_{timestamp}{suffix}"
        else:
            return f"{prefix}_{random.randint(1, 999)}{suffix}"
    
    async def _perform_file_operation(self, operation: str, file_path: str) -> bool:
        """Simulate a file operation"""
        try:
            # Simulate different success rates based on operation
            success_rates = {
                "read": 0.95,
                "write": 0.90,
                "list": 0.98,
                "search": 0.85
            }
            
            # Add some delay to simulate actual file I/O
            await asyncio.sleep(random.uniform(0.1, 2.0))
            
            # Determine success based on operation type and random chance
            success_rate = success_rates.get(operation, 0.90)
            return random.random() < success_rate
            
        except Exception as e:
            logger.debug(f"File operation {operation} on {file_path} failed: {e}")
            return False


class EmailActivitySimulator:
    """Simulates realistic email activity patterns"""
    
    def __init__(self, profile: UserProfile, environment_config: Dict[str, Any]):
        self.profile = profile
        self.environment_config = environment_config
        
        # Role-specific email patterns
        self.email_patterns = {
            UserRole.ADMIN: {
                "send_frequency": 0.3,  # emails per hour
                "receive_frequency": 0.8,
                "common_subjects": [
                    "System Maintenance", "Security Alert", "Backup Report",
                    "Server Status", "User Access Request", "Policy Update"
                ],
                "common_recipients": ["it-team@company.com", "security@company.com"]
            },
            UserRole.DEVELOPER: {
                "send_frequency": 0.5,
                "receive_frequency": 1.2,
                "common_subjects": [
                    "Code Review", "Bug Report", "Feature Request", "Deployment",
                    "Meeting Notes", "Technical Discussion"
                ],
                "common_recipients": ["dev-team@company.com", "qa@company.com"]
            },
            UserRole.SALES: {
                "send_frequency": 1.5,
                "receive_frequency": 2.0,
                "common_subjects": [
                    "Proposal", "Client Meeting", "Quote Request", "Follow-up",
                    "Contract", "Sales Report"
                ],
                "common_recipients": ["sales@company.com", "clients@company.com"]
            }
        }
    
    async def simulate_email_session(self, duration_minutes: int = 60) -> List[ActivityEvent]:
        """Simulate an email activity session"""
        activities = []
        session_start = datetime.now()
        end_time = session_start + timedelta(minutes=duration_minutes)
        
        # Get role-specific patterns
        patterns = self.email_patterns.get(self.profile.role, self.email_patterns[UserRole.ADMIN])
        
        while datetime.now() < end_time:
            # Decide whether to send or receive email
            if random.random() < 0.4:  # 40% chance to send
                activity = await self._simulate_send_email(patterns)
            else:  # 60% chance to receive/read
                activity = await self._simulate_receive_email(patterns)
            
            if activity:
                activities.append(activity)
            
            # Random delay between email activities (10-60 minutes)
            await asyncio.sleep(random.uniform(600, 3600))
        
        return activities
    
    async def _simulate_send_email(self, patterns: Dict[str, Any]) -> Optional[ActivityEvent]:
        """Simulate sending an email"""
        try:
            activity_start = time.time()
            
            # Generate email details
            subject = random.choice(patterns["common_subjects"])
            recipient = random.choice(patterns["common_recipients"])
            
            # Simulate email composition and sending
            await asyncio.sleep(random.uniform(30, 180))  # 30 seconds to 3 minutes
            
            # Simulate SMTP connection (without actually sending)
            success = random.random() < 0.95  # 95% success rate
            
            duration = time.time() - activity_start
            
            return ActivityEvent(
                timestamp=datetime.now(),
                user_id=self.profile.user_id,
                activity_type=ActivityType.EMAIL_ACTIVITY,
                details={
                    "action": "send",
                    "subject": subject,
                    "recipient": recipient,
                    "size_bytes": random.randint(1024, 10240)
                },
                success=success,
                duration_seconds=duration,
                source_ip=self.environment_config.get('user_ip', '192.168.20.100')
            )
            
        except Exception as e:
            logger.debug(f"Email send simulation failed: {e}")
            return None
    
    async def _simulate_receive_email(self, patterns: Dict[str, Any]) -> Optional[ActivityEvent]:
        """Simulate receiving/reading an email"""
        try:
            activity_start = time.time()
            
            # Simulate checking email and reading
            await asyncio.sleep(random.uniform(10, 60))  # 10 seconds to 1 minute
            
            success = random.random() < 0.98  # 98% success rate
            duration = time.time() - activity_start
            
            return ActivityEvent(
                timestamp=datetime.now(),
                user_id=self.profile.user_id,
                activity_type=ActivityType.EMAIL_ACTIVITY,
                details={
                    "action": "receive",
                    "subject": random.choice(patterns["common_subjects"]),
                    "sender": "external@company.com",
                    "size_bytes": random.randint(512, 5120)
                },
                success=success,
                duration_seconds=duration,
                source_ip=self.environment_config.get('user_ip', '192.168.20.100')
            )
            
        except Exception as e:
            logger.debug(f"Email receive simulation failed: {e}")
            return None


class ComprehensiveSyntheticUser(SyntheticUserAgent):
    """Comprehensive synthetic user that combines all activity types"""
    
    def __init__(self, profile: UserProfile, environment_config: Dict[str, Any]):
        super().__init__(profile, environment_config)
        
        # Initialize activity simulators
        self.web_simulator = WebBrowsingSimulator(profile, environment_config)
        self.file_simulator = FileAccessSimulator(profile, environment_config)
        self.email_simulator = EmailActivitySimulator(profile, environment_config)
        
        # Activity scheduling
        self.activity_tasks = []
        self.is_running = False
    
    async def start_activity_cycle(self) -> None:
        """Start the comprehensive activity cycle"""
        self.is_running = True
        logger.info(f"Starting activity cycle for user {self.profile.username}")
        
        # Start concurrent activity tasks
        self.activity_tasks = [
            asyncio.create_task(self._web_browsing_cycle()),
            asyncio.create_task(self._file_access_cycle()),
            asyncio.create_task(self._email_activity_cycle()),
            asyncio.create_task(self._login_logout_cycle())
        ]
        
        # Wait for all tasks to complete
        await asyncio.gather(*self.activity_tasks, return_exceptions=True)
    
    async def stop_activity_cycle(self) -> None:
        """Stop the activity cycle"""
        self.is_running = False
        logger.info(f"Stopping activity cycle for user {self.profile.username}")
        
        # Cancel all running tasks
        for task in self.activity_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to finish cancellation
        await asyncio.gather(*self.activity_tasks, return_exceptions=True)
    
    async def _web_browsing_cycle(self) -> None:
        """Continuous web browsing activity cycle"""
        while self.is_running:
            if self.is_work_hours():
                try:
                    # Random browsing session duration (15-45 minutes)
                    duration = random.randint(15, 45)
                    activities = await self.web_simulator.simulate_browsing_session(duration)
                    self.activity_log.extend(activities)
                    
                except Exception as e:
                    logger.error(f"Web browsing cycle error for {self.profile.username}: {e}")
            
            # Wait before next browsing session (30-120 minutes)
            await asyncio.sleep(random.uniform(1800, 7200))
    
    async def _file_access_cycle(self) -> None:
        """Continuous file access activity cycle"""
        while self.is_running:
            if self.is_work_hours():
                try:
                    # Random file access session duration (10-30 minutes)
                    duration = random.randint(10, 30)
                    activities = await self.file_simulator.simulate_file_access_session(duration)
                    self.activity_log.extend(activities)
                    
                except Exception as e:
                    logger.error(f"File access cycle error for {self.profile.username}: {e}")
            
            # Wait before next file access session (45-180 minutes)
            await asyncio.sleep(random.uniform(2700, 10800))
    
    async def _email_activity_cycle(self) -> None:
        """Continuous email activity cycle"""
        while self.is_running:
            if self.is_work_hours():
                try:
                    # Random email session duration (30-90 minutes)
                    duration = random.randint(30, 90)
                    activities = await self.email_simulator.simulate_email_session(duration)
                    self.activity_log.extend(activities)
                    
                except Exception as e:
                    logger.error(f"Email activity cycle error for {self.profile.username}: {e}")
            
            # Wait before next email session (60-240 minutes)
            await asyncio.sleep(random.uniform(3600, 14400))
    
    async def _login_logout_cycle(self) -> None:
        """Simulate login/logout patterns"""
        while self.is_running:
            try:
                current_hour = datetime.now().hour
                start_hour, end_hour = self.profile.work_hours
                
                # Login at start of work day
                if current_hour == start_hour:
                    await self._simulate_login()
                
                # Logout at end of work day
                elif current_hour == end_hour:
                    await self._simulate_logout()
                
                # Random mid-day logout/login (lunch break, meetings, etc.)
                elif self.is_work_hours() and random.random() < 0.1:  # 10% chance
                    await self._simulate_logout()
                    await asyncio.sleep(random.uniform(1800, 3600))  # 30-60 minutes
                    await self._simulate_login()
                
            except Exception as e:
                logger.error(f"Login/logout cycle error for {self.profile.username}: {e}")
            
            # Check every hour
            await asyncio.sleep(3600)
    
    async def _simulate_login(self) -> None:
        """Simulate user login"""
        activity_start = time.time()
        
        # Simulate login process
        await asyncio.sleep(random.uniform(5, 15))
        success = random.random() < 0.98  # 98% success rate
        
        duration = time.time() - activity_start
        
        self.log_activity(
            ActivityType.LOGIN_LOGOUT,
            {"action": "login", "method": "password"},
            success,
            duration
        )
    
    async def _simulate_logout(self) -> None:
        """Simulate user logout"""
        activity_start = time.time()
        
        # Simulate logout process
        await asyncio.sleep(random.uniform(1, 5))
        success = True  # Logout almost always succeeds
        
        duration = time.time() - activity_start
        
        self.log_activity(
            ActivityType.LOGIN_LOGOUT,
            {"action": "logout"},
            success,
            duration
        )
    
    def get_activity_summary(self) -> Dict[str, Any]:
        """Get summary of user activities"""
        total_activities = len(self.activity_log)
        if total_activities == 0:
            return {"total_activities": 0}
        
        # Count activities by type
        activity_counts = {}
        for activity in self.activity_log:
            activity_type = activity.activity_type.value
            activity_counts[activity_type] = activity_counts.get(activity_type, 0) + 1
        
        # Calculate success rate
        successful_activities = sum(1 for a in self.activity_log if a.success)
        success_rate = successful_activities / total_activities
        
        # Calculate average duration
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


class SyntheticUserManager:
    """Manages multiple synthetic users and their activities"""
    
    def __init__(self, environment_config: Dict[str, Any]):
        self.environment_config = environment_config
        self.users: Dict[str, ComprehensiveSyntheticUser] = {}
        self.is_running = False
        self.activity_log_file = "synthetic_user_activities.jsonl"
    
    def create_default_user_profiles(self) -> List[UserProfile]:
        """Create a set of default user profiles for testing"""
        profiles = [
            UserProfile(
                user_id="admin001",
                username="admin",
                role=UserRole.ADMIN,
                department="IT",
                email="admin@company.com",
                work_hours=(8, 17),
                activity_frequency=0.5,
                web_browsing_patterns=["/admin", "/dashboard", "/system"],
                file_access_patterns=["/var/log/", "/etc/", "/home/admin/"],
                email_patterns={"frequency": "low"},
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
                web_browsing_patterns=["/api", "/docs", "/git"],
                file_access_patterns=["/home/dev/", "/var/www/", "/opt/code/"],
                email_patterns={"frequency": "medium"},
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
                web_browsing_patterns=["/crm", "/leads", "/reports"],
                file_access_patterns=["/home/sales/", "/shared/sales/"],
                email_patterns={"frequency": "high"},
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
                web_browsing_patterns=["/hr", "/employees", "/payroll"],
                file_access_patterns=["/home/hr/", "/shared/hr/"],
                email_patterns={"frequency": "medium"},
                risk_profile="low"
            )
        ]
        return profiles
    
    async def add_user(self, profile: UserProfile) -> None:
        """Add a synthetic user"""
        user = ComprehensiveSyntheticUser(profile, self.environment_config)
        self.users[profile.user_id] = user
        logger.info(f"Added synthetic user: {profile.username} ({profile.role.value})")
    
    async def start_all_users(self) -> None:
        """Start activity cycles for all users"""
        if not self.users:
            # Create default users if none exist
            profiles = self.create_default_user_profiles()
            for profile in profiles:
                await self.add_user(profile)
        
        self.is_running = True
        logger.info(f"Starting activity cycles for {len(self.users)} synthetic users")
        
        # Start all user activity cycles concurrently
        tasks = []
        for user in self.users.values():
            task = asyncio.create_task(user.start_activity_cycle())
            tasks.append(task)
        
        # Also start the activity logging task
        tasks.append(asyncio.create_task(self._periodic_log_activities()))
        
        # Wait for all tasks
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop_all_users(self) -> None:
        """Stop activity cycles for all users"""
        self.is_running = False
        logger.info("Stopping all synthetic user activities")
        
        # Stop all users
        tasks = []
        for user in self.users.values():
            task = asyncio.create_task(user.stop_activity_cycle())
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _periodic_log_activities(self) -> None:
        """Periodically log all user activities to file"""
        while self.is_running:
            try:
                # Collect activities from all users
                all_activities = []
                for user in self.users.values():
                    all_activities.extend(user.activity_log)
                
                # Write to log file
                if all_activities:
                    await self._write_activities_to_file(all_activities)
                    
                    # Clear activity logs to prevent memory buildup
                    for user in self.users.values():
                        user.activity_log.clear()
                
            except Exception as e:
                logger.error(f"Error logging activities: {e}")
            
            # Log every 5 minutes
            await asyncio.sleep(300)
    
    async def _write_activities_to_file(self, activities: List[ActivityEvent]) -> None:
        """Write activities to JSONL file"""
        try:
            async with aiofiles.open(self.activity_log_file, 'a') as f:
                for activity in activities:
                    # Convert to dict and handle datetime serialization
                    activity_dict = asdict(activity)
                    activity_dict['timestamp'] = activity.timestamp.isoformat()
                    activity_dict['activity_type'] = activity.activity_type.value
                    
                    # Write as JSON line
                    await f.write(json.dumps(activity_dict) + '\n')
                    
        except Exception as e:
            logger.error(f"Error writing activities to file: {e}")
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get summary of all synthetic user activities"""
        summary = {
            "total_users": len(self.users),
            "users": {},
            "system_totals": {
                "total_activities": 0,
                "total_success_rate": 0,
                "activity_type_counts": {}
            }
        }
        
        total_activities = 0
        total_successful = 0
        
        for user_id, user in self.users.items():
            user_summary = user.get_activity_summary()
            summary["users"][user_id] = {
                "username": user.profile.username,
                "role": user.profile.role.value,
                "summary": user_summary
            }
            
            # Aggregate system totals
            user_total = user_summary.get("total_activities", 0)
            total_activities += user_total
            total_successful += int(user_total * user_summary.get("success_rate", 0))
            
            # Aggregate activity type counts
            for activity_type, count in user_summary.get("activity_counts", {}).items():
                summary["system_totals"]["activity_type_counts"][activity_type] = \
                    summary["system_totals"]["activity_type_counts"].get(activity_type, 0) + count
        
        summary["system_totals"]["total_activities"] = total_activities
        summary["system_totals"]["total_success_rate"] = (
            total_successful / total_activities if total_activities > 0 else 0
        )
        
        return summary


# Example usage and testing
async def main():
    """Example usage of the synthetic user system"""
    
    # Environment configuration
    environment_config = {
        'web_server_url': 'http://192.168.10.10',
        'user_ip': '192.168.20.100',
        'mail_server': '192.168.10.20',
        'file_server': '192.168.10.30'
    }
    
    # Create and start synthetic user manager
    manager = SyntheticUserManager(environment_config)
    
    try:
        # Start all users (this will run indefinitely)
        await manager.start_all_users()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, stopping synthetic users...")
        await manager.stop_all_users()
        
        # Print final summary
        summary = manager.get_system_summary()
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())