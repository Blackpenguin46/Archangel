#!/usr/bin/env python3
"""
Synthetic User CLI Tool

Command-line interface for managing and running synthetic user simulations.
Provides commands for configuration, user management, and simulation control.
"""

import asyncio
import argparse
import json
import sys
import signal
from pathlib import Path
from typing import Optional
import logging

from agents.synthetic_users import SyntheticUserManager
from agents.synthetic_user_config import SyntheticUserConfigManager, create_sample_config_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SyntheticUserCLI:
    """Command-line interface for synthetic user management"""
    
    def __init__(self):
        self.config_manager = None
        self.user_manager = None
        self.running = False
    
    def setup_config(self, config_file: Optional[str] = None) -> None:
        """Setup configuration manager"""
        self.config_manager = SyntheticUserConfigManager(config_file)
        
        # Create environment config dict for user manager
        env_config = {
            'web_server_url': self.config_manager.environment_config.web_server_url,
            'mail_server_url': self.config_manager.environment_config.mail_server_url,
            'file_server_url': self.config_manager.environment_config.file_server_url,
            'database_server_url': self.config_manager.environment_config.database_server_url,
            'user_ip': self.config_manager.environment_config.default_user_ip,
            'domain_name': self.config_manager.environment_config.domain_name,
            'simulation_speed': self.config_manager.environment_config.simulation_speed
        }
        
        self.user_manager = SyntheticUserManager(env_config)
    
    async def run_simulation(self, duration: Optional[int] = None) -> None:
        """Run the synthetic user simulation"""
        if not self.config_manager or not self.user_manager:
            logger.error("Configuration not initialized. Run 'init' command first.")
            return
        
        # Add users from configuration
        for profile in self.config_manager.user_profiles:
            await self.user_manager.add_user(profile)
        
        logger.info(f"Starting simulation with {len(self.user_manager.users)} users")
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("Received interrupt signal, stopping simulation...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        self.running = True
        
        try:
            if duration:
                # Run for specified duration
                logger.info(f"Running simulation for {duration} seconds")
                task = asyncio.create_task(self.user_manager.start_all_users())
                await asyncio.wait_for(task, timeout=duration)
            else:
                # Run indefinitely
                logger.info("Running simulation indefinitely (Ctrl+C to stop)")
                await self.user_manager.start_all_users()
                
        except asyncio.TimeoutError:
            logger.info("Simulation duration completed")
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        finally:
            await self.user_manager.stop_all_users()
            
            # Print final summary
            summary = self.user_manager.get_system_summary()
            logger.info("Final simulation summary:")
            print(json.dumps(summary, indent=2))
    
    def list_users(self) -> None:
        """List all configured users"""
        if not self.config_manager:
            logger.error("Configuration not initialized. Run 'init' command first.")
            return
        
        if not self.config_manager.user_profiles:
            print("No users configured.")
            return
        
        print(f"Configured Users ({len(self.config_manager.user_profiles)}):")
        print("-" * 80)
        
        for profile in self.config_manager.user_profiles:
            print(f"ID: {profile.user_id}")
            print(f"  Username: {profile.username}")
            print(f"  Role: {profile.role.value}")
            print(f"  Department: {profile.department}")
            print(f"  Work Hours: {profile.work_hours[0]:02d}:00 - {profile.work_hours[1]:02d}:00")
            print(f"  Activity Frequency: {profile.activity_frequency}")
            print(f"  Risk Profile: {profile.risk_profile}")
            print()
    
    def show_config(self) -> None:
        """Show current configuration"""
        if not self.config_manager:
            logger.error("Configuration not initialized. Run 'init' command first.")
            return
        
        summary = self.config_manager.get_config_summary()
        print("Current Configuration:")
        print(json.dumps(summary, indent=2))
    
    def validate_config(self) -> None:
        """Validate current configuration"""
        if not self.config_manager:
            logger.error("Configuration not initialized. Run 'init' command first.")
            return
        
        errors = self.config_manager.validate_config()
        
        if errors:
            print("Configuration Errors:")
            for error in errors:
                print(f"  ❌ {error}")
            sys.exit(1)
        else:
            print("✅ Configuration is valid")
    
    def create_config(self, config_file: str) -> None:
        """Create a new configuration file"""
        if Path(config_file).exists():
            response = input(f"Configuration file '{config_file}' already exists. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("Configuration creation cancelled.")
                return
        
        create_sample_config_file(config_file)
        print(f"✅ Created configuration file: {config_file}")
    
    async def test_user(self, user_id: str, duration: int = 60) -> None:
        """Test a specific user for a short duration"""
        if not self.config_manager or not self.user_manager:
            logger.error("Configuration not initialized. Run 'init' command first.")
            return
        
        # Find the user profile
        profile = self.config_manager.get_user_profile(user_id)
        if not profile:
            logger.error(f"User '{user_id}' not found in configuration")
            return
        
        # Add only this user
        await self.user_manager.add_user(profile)
        
        logger.info(f"Testing user '{profile.username}' for {duration} seconds")
        
        try:
            task = asyncio.create_task(self.user_manager.start_all_users())
            await asyncio.wait_for(task, timeout=duration)
        except asyncio.TimeoutError:
            logger.info("Test completed")
        finally:
            await self.user_manager.stop_all_users()
            
            # Show user summary
            summary = self.user_manager.get_system_summary()
            if user_id in summary['users']:
                user_summary = summary['users'][user_id]['summary']
                print(f"\nTest Results for {profile.username}:")
                print(f"  Total Activities: {user_summary.get('total_activities', 0)}")
                print(f"  Success Rate: {user_summary.get('success_rate', 0):.2%}")
                print(f"  Activity Types: {user_summary.get('activity_counts', {})}")
    
    def show_logs(self, lines: int = 50) -> None:
        """Show recent activity logs"""
        if not self.config_manager:
            logger.error("Configuration not initialized. Run 'init' command first.")
            return
        
        log_file = self.config_manager.environment_config.activity_log_file
        
        if not Path(log_file).exists():
            print(f"Log file '{log_file}' not found. Run a simulation first.")
            return
        
        try:
            with open(log_file, 'r') as f:
                log_lines = f.readlines()
            
            # Show last N lines
            recent_lines = log_lines[-lines:] if len(log_lines) > lines else log_lines
            
            print(f"Recent Activity Log ({len(recent_lines)} entries):")
            print("-" * 80)
            
            for line in recent_lines:
                try:
                    activity = json.loads(line.strip())
                    timestamp = activity.get('timestamp', 'Unknown')
                    user_id = activity.get('user_id', 'Unknown')
                    activity_type = activity.get('activity_type', 'Unknown')
                    success = "✅" if activity.get('success', False) else "❌"
                    
                    print(f"{timestamp} | {user_id} | {activity_type} | {success}")
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            logger.error(f"Error reading log file: {e}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Synthetic User Simulation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s init                          # Initialize with default config
  %(prog)s init --config custom.yaml    # Initialize with custom config
  %(prog)s create-config myconfig.yaml  # Create new config file
  %(prog)s list-users                   # List all configured users
  %(prog)s validate                     # Validate configuration
  %(prog)s run                          # Run simulation indefinitely
  %(prog)s run --duration 3600          # Run for 1 hour
  %(prog)s test-user dev001 --duration 60  # Test specific user for 1 minute
  %(prog)s logs --lines 100             # Show last 100 log entries
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Configuration file path (default: synthetic_user_config.yaml)',
        default='synthetic_user_config.yaml'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize configuration')
    
    # Create config command
    create_parser = subparsers.add_parser('create-config', help='Create new configuration file')
    create_parser.add_argument('filename', help='Configuration file name')
    
    # List users command
    list_parser = subparsers.add_parser('list-users', help='List configured users')
    
    # Show config command
    config_parser = subparsers.add_parser('show-config', help='Show current configuration')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run simulation')
    run_parser.add_argument(
        '--duration', '-d',
        type=int,
        help='Run duration in seconds (default: indefinite)'
    )
    
    # Test user command
    test_parser = subparsers.add_parser('test-user', help='Test specific user')
    test_parser.add_argument('user_id', help='User ID to test')
    test_parser.add_argument(
        '--duration', '-d',
        type=int,
        default=60,
        help='Test duration in seconds (default: 60)'
    )
    
    # Show logs command
    logs_parser = subparsers.add_parser('logs', help='Show activity logs')
    logs_parser.add_argument(
        '--lines', '-n',
        type=int,
        default=50,
        help='Number of recent log lines to show (default: 50)'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = SyntheticUserCLI()
    
    try:
        if args.command == 'create-config':
            cli.create_config(args.filename)
        
        elif args.command == 'init':
            cli.setup_config(args.config)
            print(f"✅ Initialized with configuration: {args.config}")
        
        elif args.command in ['list-users', 'show-config', 'validate', 'logs']:
            cli.setup_config(args.config)
            
            if args.command == 'list-users':
                cli.list_users()
            elif args.command == 'show-config':
                cli.show_config()
            elif args.command == 'validate':
                cli.validate_config()
            elif args.command == 'logs':
                cli.show_logs(args.lines)
        
        elif args.command == 'run':
            cli.setup_config(args.config)
            asyncio.run(cli.run_simulation(args.duration))
        
        elif args.command == 'test-user':
            cli.setup_config(args.config)
            asyncio.run(cli.test_user(args.user_id, args.duration))
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()