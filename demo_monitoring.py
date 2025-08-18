#!/usr/bin/env python3
"""
Demo script for the Archangel monitoring and alerting infrastructure.

This script demonstrates:
1. Prometheus metrics collection from agents
2. Health monitoring and status tracking
3. Automated recovery system
4. Grafana dashboard integration
5. Alert management with AlertManager

Run this script to see the monitoring system in action.
"""

import asyncio
import time
import random
import logging
import threading
from typing import Dict, List
import signal
import sys

from monitoring.metrics_collector import MetricsCollector, metrics_collector
from monitoring.health_monitor import HealthMonitor, HealthStatus, health_monitor
from monitoring.recovery_system import RecoverySystem, RecoveryAction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MonitoringDemo:
    """
    Comprehensive demonstration of the Archangel monitoring system.
    """
    
    def __init__(self):
        """Initialize the monitoring demo."""
        self.running = False
        self.metrics = MetricsCollector(port=8888)
        self.health_monitor = HealthMonitor(check_interval=10.0)
        self.recovery_system = RecoverySystem(self.health_monitor)
        
        # Demo agents
        self.agents = {
            "red_recon_alpha": {"type": "recon", "team": "red", "status": "active"},
            "red_exploit_beta": {"type": "exploit", "team": "red", "status": "active"},
            "red_persist_gamma": {"type": "persistence", "team": "red", "status": "active"},
            "blue_soc_delta": {"type": "soc_analyst", "team": "blue", "status": "active"},
            "blue_firewall_epsilon": {"type": "firewall_config", "team": "blue", "status": "active"},
            "blue_siem_zeta": {"type": "siem_integrator", "team": "blue", "status": "active"},
        }
        
        # Demo services
        self.services = {
            "mysql-backend": {"type": "database", "status": "healthy"},
            "elasticsearch": {"type": "search_engine", "status": "healthy"}, 
            "redis-cache": {"type": "cache", "status": "healthy"},
            "message-broker": {"type": "messaging", "status": "healthy"},
        }
        
        self.simulation_threads: List[threading.Thread] = []
        
    def setup_monitoring(self):
        """Set up all monitoring components."""
        logger.info("Setting up monitoring infrastructure...")
        
        # Start metrics collection server
        self.metrics.start_server()
        logger.info(f"Prometheus metrics available at http://localhost:8888/metrics")
        
        # Start health monitoring
        self.health_monitor.start_monitoring()
        
        # Register all agents for metrics collection
        for agent_id, config in self.agents.items():
            self.metrics.register_agent(agent_id, config["type"], config["team"])
            self.health_monitor.register_component(agent_id, "agent")
            logger.info(f"Registered {agent_id} ({config['type']}, {config['team']} team)")
            
        # Register services for health monitoring
        for service_id, config in self.services.items():
            self.health_monitor.register_component(service_id, config["type"])
            logger.info(f"Registered service {service_id} ({config['type']})")
            
        # Add custom recovery rules
        self.setup_recovery_rules()
        
    def setup_recovery_rules(self):
        """Set up automated recovery rules."""
        logger.info("Setting up automated recovery rules...")
        
        # Agent failure recovery
        self.recovery_system.add_recovery_rule(
            name="restart_failed_red_agents",
            component_pattern=r"red_.*",
            trigger_status=HealthStatus.CRITICAL,
            action=RecoveryAction.CUSTOM,
            cooldown_seconds=30.0,
            max_attempts=3,
            custom_function=self.restart_agent,
            description="Restart failed Red Team agents"
        )
        
        self.recovery_system.add_recovery_rule(
            name="restart_failed_blue_agents", 
            component_pattern=r"blue_.*",
            trigger_status=HealthStatus.CRITICAL,
            action=RecoveryAction.CUSTOM,
            cooldown_seconds=30.0,
            max_attempts=3,
            custom_function=self.restart_agent,
            description="Restart failed Blue Team agents"
        )
        
        # Service failure recovery
        self.recovery_system.add_recovery_rule(
            name="restart_failed_services",
            component_pattern=r".*-.*",  # Services have dashes in names
            trigger_status=HealthStatus.CRITICAL,
            action=RecoveryAction.CUSTOM,
            cooldown_seconds=60.0,
            max_attempts=2,
            custom_function=self.restart_service,
            description="Restart failed services"
        )
        
        # Warning alerts
        self.recovery_system.add_recovery_rule(
            name="alert_on_warnings",
            component_pattern=r".*",
            trigger_status=HealthStatus.WARNING,
            action=RecoveryAction.ALERT_ONLY,
            cooldown_seconds=300.0,
            max_attempts=1,
            description="Send alerts for warning states"
        )
        
    def restart_agent(self, component_id: str, params: Dict) -> bool:
        """Custom recovery function to restart agents."""
        logger.info(f"🔄 RECOVERY: Restarting agent {component_id}")
        
        # Simulate restart process
        time.sleep(2)
        
        # Update agent status
        if component_id in self.agents:
            self.agents[component_id]["status"] = "restarting"
            
            # Simulate restart completion
            def complete_restart():
                time.sleep(3)
                self.agents[component_id]["status"] = "active"
                self.health_monitor.update_component_health(
                    component_id, HealthStatus.HEALTHY,
                    {"restart_time": time.time(), "action": "auto_restart"}
                )
                logger.info(f"✅ RECOVERY: Agent {component_id} restart completed")
                
            threading.Thread(target=complete_restart, daemon=True).start()
            return True
            
        return False
        
    def restart_service(self, component_id: str, params: Dict) -> bool:
        """Custom recovery function to restart services."""
        logger.info(f"🔄 RECOVERY: Restarting service {component_id}")
        
        # Simulate service restart
        time.sleep(5)
        
        if component_id in self.services:
            self.services[component_id]["status"] = "restarting"
            
            # Simulate restart completion
            def complete_restart():
                time.sleep(5)
                self.services[component_id]["status"] = "healthy"
                self.health_monitor.update_component_health(
                    component_id, HealthStatus.HEALTHY,
                    {"restart_time": time.time(), "action": "auto_restart"}
                )
                logger.info(f"✅ RECOVERY: Service {component_id} restart completed")
                
            threading.Thread(target=complete_restart, daemon=True).start()
            return True
            
        return False
        
    def simulate_agent_activity(self):
        """Simulate realistic agent activity patterns."""
        logger.info("Starting agent activity simulation...")
        
        while self.running:
            for agent_id, config in self.agents.items():
                if config["status"] != "active":
                    continue
                    
                # Simulate decision making
                decision_types = {
                    "recon": ["port_scan", "service_enum", "vuln_scan"],
                    "exploit": ["exploit_attempt", "payload_delivery", "privilege_escalation"],
                    "persistence": ["backdoor_install", "cred_harvest", "lateral_move"],
                    "soc_analyst": ["alert_triage", "incident_create", "threat_analyze"],
                    "firewall_config": ["rule_update", "block_ip", "allow_service"],
                    "siem_integrator": ["log_correlate", "pattern_detect", "alert_generate"]
                }
                
                agent_type = config["type"]
                if agent_type in decision_types:
                    decision_type = random.choice(decision_types[agent_type])
                    success = random.random() > 0.15  # 85% success rate
                    response_time = random.uniform(0.1, 3.0)
                    
                    self.metrics.record_decision(agent_id, decision_type, success, response_time)
                    
                    # Occasionally record communication failures
                    if random.random() < 0.05:  # 5% chance
                        failure_types = ["timeout", "connection_refused", "authentication_failed"]
                        failure_type = random.choice(failure_types)
                        self.metrics.record_communication_failure(agent_id, failure_type)
                        
                    # Record team-specific actions
                    if config["team"] == "red":
                        if random.random() < 0.3:  # 30% chance
                            targets = ["web_server", "database", "file_server", "workstation"]
                            action_types = ["exploit", "recon", "persist", "exfiltrate"]
                            self.metrics.record_red_team_action(
                                agent_id, 
                                random.choice(action_types),
                                random.choice(targets),
                                success
                            )
                    elif config["team"] == "blue":
                        if random.random() < 0.4:  # 40% chance
                            detection_types = ["malware", "intrusion", "anomaly", "policy_violation"]
                            severities = ["low", "medium", "high", "critical"]
                            response_time = random.uniform(1.0, 30.0)
                            self.metrics.record_blue_team_detection(
                                agent_id,
                                random.choice(detection_types),
                                random.choice(severities),
                                response_time
                            )
                            
            # Update team coordination scores
            red_score = max(0.3, min(1.0, 0.8 + random.uniform(-0.2, 0.2)))
            blue_score = max(0.3, min(1.0, 0.75 + random.uniform(-0.2, 0.2)))
            
            self.metrics.update_team_coordination("red", red_score)
            self.metrics.update_team_coordination("blue", blue_score)
            
            # Record system performance metrics
            self.metrics.record_game_loop_duration(random.uniform(5.0, 45.0))
            self.metrics.record_scoring_duration(random.uniform(0.5, 8.0))
            self.metrics.record_vector_store_query("search", random.uniform(0.1, 2.0))
            
            time.sleep(random.uniform(1.0, 3.0))  # Variable activity interval
            
    def simulate_system_events(self):
        """Simulate system events including failures and recoveries."""
        logger.info("Starting system events simulation...")
        
        while self.running:
            time.sleep(random.uniform(30.0, 120.0))  # Random intervals
            
            # Randomly introduce issues
            event_type = random.choice([
                "agent_failure", "agent_performance_degradation", 
                "service_failure", "service_warning",
                "communication_issues", "resource_pressure"
            ])
            
            if event_type == "agent_failure":
                agent_id = random.choice(list(self.agents.keys()))
                if self.agents[agent_id]["status"] == "active":
                    logger.warning(f"🚨 INCIDENT: Agent {agent_id} has failed!")
                    self.agents[agent_id]["status"] = "failed"
                    self.health_monitor.update_component_health(
                        agent_id, HealthStatus.CRITICAL,
                        {"failure_reason": "simulated_failure", "timestamp": time.time()}
                    )
                    
            elif event_type == "agent_performance_degradation":
                agent_id = random.choice(list(self.agents.keys()))
                if self.agents[agent_id]["status"] == "active":
                    logger.warning(f"⚠️  WARNING: Agent {agent_id} performance degraded")
                    self.health_monitor.update_component_health(
                        agent_id, HealthStatus.WARNING,
                        {"issue": "high_response_time", "threshold_exceeded": True}
                    )
                    
                    # Auto-recovery after some time
                    def auto_recover():
                        time.sleep(random.uniform(60.0, 180.0))
                        if self.running:
                            logger.info(f"✅ AUTO-RECOVERY: Agent {agent_id} performance normalized")
                            self.health_monitor.update_component_health(
                                agent_id, HealthStatus.HEALTHY,
                                {"auto_recovery": True, "recovery_time": time.time()}
                            )
                    threading.Thread(target=auto_recover, daemon=True).start()
                    
            elif event_type == "service_failure":
                service_id = random.choice(list(self.services.keys()))
                if self.services[service_id]["status"] == "healthy":
                    logger.error(f"💥 CRITICAL: Service {service_id} has failed!")
                    self.services[service_id]["status"] = "failed"
                    self.health_monitor.update_component_health(
                        service_id, HealthStatus.CRITICAL,
                        {"failure_type": "connection_timeout", "last_seen": time.time()}
                    )
                    
            elif event_type == "communication_issues":
                # Simulate widespread communication issues
                logger.warning("🌐 NETWORK: Communication issues detected")
                for agent_id in random.sample(list(self.agents.keys()), 2):
                    self.metrics.record_communication_failure(agent_id, "network_partition")
                    
    def print_monitoring_status(self):
        """Print current monitoring status."""
        while self.running:
            time.sleep(30.0)  # Print status every 30 seconds
            
            logger.info("=" * 60)
            logger.info("📊 MONITORING STATUS REPORT")
            logger.info("=" * 60)
            
            # System health summary
            health_summary = self.health_monitor.get_system_health_summary()
            logger.info(f"Overall System Health: {health_summary['overall_status'].upper()}")
            logger.info(f"Components - Healthy: {health_summary['healthy']}, "
                       f"Warning: {health_summary['warning']}, "
                       f"Critical: {health_summary['critical']}, "
                       f"Unknown: {health_summary['unknown']}")
            
            # Agent metrics
            all_agents = self.metrics.get_all_agents()
            logger.info(f"\n🤖 AGENTS ({len(all_agents)} registered):")
            for agent_id, metrics in all_agents.items():
                logger.info(f"  {agent_id}: {metrics.decisions_total} decisions, "
                           f"{metrics.communication_failures} comm failures, "
                           f"last active: {time.time() - metrics.last_activity:.1f}s ago")
                           
            # Recovery statistics
            recovery_stats = self.recovery_system.get_recovery_stats()
            if recovery_stats["total_attempts"] > 0:
                logger.info(f"\n🔧 RECOVERY STATS:")
                logger.info(f"  Total attempts: {recovery_stats['total_attempts']}")
                logger.info(f"  Success rate: {recovery_stats['success_rate']:.1%}")
                logger.info(f"  Most common action: {recovery_stats['most_common_action']}")
                
            logger.info("=" * 60)
            
    def run_demo(self, duration: int = 300):
        """Run the complete monitoring demonstration.
        
        Args:
            duration: How long to run the demo in seconds (default: 5 minutes)
        """
        logger.info(f"🚀 Starting Archangel Monitoring System Demo ({duration}s)")
        
        # Set up monitoring
        self.setup_monitoring()
        
        # Start simulation
        self.running = True
        
        # Start simulation threads
        threads = [
            threading.Thread(target=self.simulate_agent_activity, daemon=True),
            threading.Thread(target=self.simulate_system_events, daemon=True),
            threading.Thread(target=self.print_monitoring_status, daemon=True),
        ]
        
        for thread in threads:
            thread.start()
            self.simulation_threads.append(thread)
            
        logger.info("📈 Demo running! Check the following URLs:")
        logger.info("  • Prometheus metrics: http://localhost:8888/metrics")
        logger.info("  • Prometheus UI: http://localhost:9090")  
        logger.info("  • Grafana dashboards: http://localhost:3000 (admin/admin123)")
        logger.info("  • AlertManager: http://localhost:9093")
        logger.info(f"  • Demo will run for {duration} seconds...")
        
        try:
            # Run for specified duration
            time.sleep(duration)
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
            
        self.stop_demo()
        
    def stop_demo(self):
        """Stop the monitoring demo."""
        logger.info("🛑 Stopping monitoring demo...")
        
        self.running = False
        
        # Stop monitoring components
        if hasattr(self.metrics, '_running') and self.metrics._running:
            self.metrics.stop_server()
            
        if self.health_monitor._running:
            self.health_monitor.stop_monitoring()
            
        logger.info("✅ Demo stopped successfully")
        
    def signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_demo()
        sys.exit(0)


def main():
    """Main entry point for the monitoring demo."""
    demo = MonitoringDemo()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, demo.signal_handler)
    signal.signal(signal.SIGTERM, demo.signal_handler)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Archangel Monitoring System Demo")
    parser.add_argument("--duration", "-d", type=int, default=300, 
                       help="Demo duration in seconds (default: 300)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Reduce log output")
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
        
    # Run the demo
    try:
        demo.run_demo(duration=args.duration)
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())