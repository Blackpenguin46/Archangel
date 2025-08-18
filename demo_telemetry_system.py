#!/usr/bin/env python3
"""
Comprehensive demo of the telemetry and observability system.

This demo showcases:
- OpenTelemetry integration with distributed tracing
- Real-time performance monitoring and profiling
- Time-warping capabilities for forensic analysis
- Alert management and notification system
- Log correlation and analysis
- System health monitoring and dashboards
"""

import asyncio
import time
import json
import random
from datetime import datetime, timedelta
from threading import Thread
import logging

from monitoring.telemetry import (
    initialize_telemetry, get_telemetry, TelemetrySystem,
    PerformanceMetrics, TraceContext
)
from monitoring.observability import (
    initialize_observability, get_observability, ObservabilitySystem,
    AlertSeverity, AlertStatus, LogEntry
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DemoAgent:
    """Demo agent that generates telemetry data."""
    
    def __init__(self, agent_id: str, telemetry: TelemetrySystem):
        self.agent_id = agent_id
        self.telemetry = telemetry
        self.is_running = False
    
    async def simulate_work(self):
        """Simulate agent work with telemetry tracking."""
        operations = [
            "data_processing", "decision_making", "communication",
            "learning", "planning", "execution"
        ]
        
        while self.is_running:
            operation = random.choice(operations)
            
            # Simulate operation with tracing
            with self.telemetry.trace_operation(
                operation, 
                f"agent_{self.agent_id}",
                agent_id=self.agent_id,
                operation_type=operation
            ) as span:
                
                # Record start event
                self.telemetry.record_event(
                    "operation_started",
                    f"agent_{self.agent_id}",
                    {
                        "operation": operation,
                        "agent_id": self.agent_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                # Simulate work duration
                work_duration = random.uniform(0.1, 2.0)
                await asyncio.sleep(work_duration)
                
                # Simulate occasional errors
                if random.random() < 0.1:  # 10% error rate
                    error_msg = f"Simulated error in {operation}"
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", error_msg)
                    
                    self.telemetry.record_event(
                        "error",
                        f"agent_{self.agent_id}",
                        {
                            "operation": operation,
                            "error_message": error_msg,
                            "agent_id": self.agent_id
                        }
                    )
                    logger.error(f"Agent {self.agent_id}: {error_msg}")
                else:
                    # Record success event
                    self.telemetry.record_event(
                        "operation_completed",
                        f"agent_{self.agent_id}",
                        {
                            "operation": operation,
                            "duration_ms": work_duration * 1000,
                            "agent_id": self.agent_id,
                            "success": True
                        }
                    )
                
                # Add some span events
                span.add_event("processing_data", {"data_size": random.randint(100, 1000)})
                span.add_event("operation_checkpoint", {"progress": "50%"})
                span.add_event("operation_complete", {"result": "success" if random.random() > 0.1 else "error"})
            
            # Random delay between operations
            await asyncio.sleep(random.uniform(0.5, 3.0))
    
    def start(self):
        """Start the demo agent."""
        self.is_running = True
    
    def stop(self):
        """Stop the demo agent."""
        self.is_running = False


class DemoScenarioManager:
    """Manages different demo scenarios."""
    
    def __init__(self, telemetry: TelemetrySystem, observability: ObservabilitySystem):
        self.telemetry = telemetry
        self.observability = observability
        self.agents = []
    
    def setup_notification_handlers(self):
        """Set up alert notification handlers."""
        def console_notification_handler(alert):
            print(f"\n🚨 ALERT: {alert.title}")
            print(f"   Severity: {alert.severity.value.upper()}")
            print(f"   Description: {alert.description}")
            print(f"   Source: {alert.source}")
            print(f"   Time: {alert.timestamp}")
            print()
        
        def log_notification_handler(alert):
            log_entry = LogEntry(
                timestamp=datetime.utcnow(),
                level="WARN" if alert.severity in [AlertSeverity.LOW, AlertSeverity.MEDIUM] else "ERROR",
                source="alert_system",
                message=f"Alert triggered: {alert.title}",
                context={
                    "alert_id": alert.alert_id,
                    "severity": alert.severity.value,
                    "source": alert.source
                }
            )
            self.observability.log_correlator.add_log_entry(log_entry)
        
        self.observability.alert_manager.add_notification_handler(console_notification_handler)
        self.observability.alert_manager.add_notification_handler(log_notification_handler)
    
    def setup_topology_discovery(self):
        """Set up topology discovery handlers."""
        def discover_agent_topology():
            nodes = []
            edges = []
            
            # Add agent nodes
            for i, agent in enumerate(self.agents):
                nodes.append({
                    "id": f"agent_{agent.agent_id}",
                    "type": "agent",
                    "label": f"Agent {agent.agent_id}",
                    "status": "active" if agent.is_running else "inactive",
                    "metadata": {
                        "agent_id": agent.agent_id,
                        "start_time": datetime.utcnow().isoformat()
                    }
                })
            
            # Add connections between agents (simulated)
            for i in range(len(self.agents) - 1):
                edges.append({
                    "source": f"agent_{self.agents[i].agent_id}",
                    "target": f"agent_{self.agents[i+1].agent_id}",
                    "type": "communication",
                    "weight": random.uniform(0.1, 1.0)
                })
            
            return {
                "nodes": nodes,
                "edges": edges,
                "metadata": {
                    "discovery_method": "agent_simulation",
                    "total_agents": len(self.agents)
                }
            }
        
        self.observability.topology_discovery.add_discovery_handler(discover_agent_topology)
    
    async def scenario_normal_operations(self, duration: int = 60):
        """Scenario: Normal system operations."""
        print(f"\n🔄 Starting Normal Operations Scenario ({duration}s)")
        print("   - Multiple agents performing routine tasks")
        print("   - Occasional errors and performance variations")
        print("   - Real-time monitoring and telemetry collection")
        
        # Create demo agents
        for i in range(3):
            agent = DemoAgent(f"demo_{i}", self.telemetry)
            self.agents.append(agent)
            agent.start()
        
        # Run agents concurrently
        tasks = [agent.simulate_work() for agent in self.agents]
        
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=duration)
        except asyncio.TimeoutError:
            pass
        finally:
            for agent in self.agents:
                agent.stop()
    
    async def scenario_performance_stress(self, duration: int = 30):
        """Scenario: Performance stress testing."""
        print(f"\n⚡ Starting Performance Stress Scenario ({duration}s)")
        print("   - High-frequency operations")
        print("   - Performance bottleneck simulation")
        print("   - Automatic bottleneck detection")
        
        stress_operations = ["cpu_intensive", "memory_intensive", "io_intensive"]
        
        start_time = time.time()
        while time.time() - start_time < duration:
            operation = random.choice(stress_operations)
            
            with self.telemetry.trace_operation(operation, "stress_test") as span:
                # Simulate different types of stress
                if operation == "cpu_intensive":
                    # CPU-bound work simulation
                    work_time = random.uniform(0.5, 2.0)
                    await asyncio.sleep(work_time)
                    
                elif operation == "memory_intensive":
                    # Memory allocation simulation
                    data_size = random.randint(1000, 10000)
                    span.set_attribute("memory.allocated_mb", data_size / 1000)
                    await asyncio.sleep(0.1)
                    
                elif operation == "io_intensive":
                    # I/O simulation
                    io_time = random.uniform(0.2, 1.5)
                    span.set_attribute("io.wait_time_ms", io_time * 1000)
                    await asyncio.sleep(io_time)
                
                # Record performance metrics
                self.telemetry.record_event(
                    "stress_operation",
                    "stress_test",
                    {
                        "operation": operation,
                        "duration_ms": time.time() * 1000,
                        "load_level": "high"
                    }
                )
            
            await asyncio.sleep(0.1)  # Brief pause between operations
    
    async def scenario_error_conditions(self, duration: int = 20):
        """Scenario: Error condition simulation."""
        print(f"\n❌ Starting Error Conditions Scenario ({duration}s)")
        print("   - Simulating various error types")
        print("   - Testing alert generation")
        print("   - Error correlation and analysis")
        
        error_types = [
            "connection_timeout", "authentication_failure", 
            "resource_exhaustion", "data_corruption", "service_unavailable"
        ]
        
        start_time = time.time()
        while time.time() - start_time < duration:
            error_type = random.choice(error_types)
            
            # Record error event
            self.telemetry.record_event(
                "error",
                "error_simulation",
                {
                    "error_type": error_type,
                    "severity": random.choice(["low", "medium", "high"]),
                    "component": f"component_{random.randint(1, 5)}",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # Add corresponding log entry
            log_entry = LogEntry(
                timestamp=datetime.utcnow(),
                level="ERROR",
                source="error_simulation",
                message=f"Simulated error: {error_type}",
                context={
                    "error_type": error_type,
                    "simulation": True
                }
            )
            self.observability.log_correlator.add_log_entry(log_entry)
            
            # Occasionally trigger alerts
            if random.random() < 0.3:  # 30% chance
                severity = random.choice([AlertSeverity.LOW, AlertSeverity.MEDIUM, AlertSeverity.HIGH])
                self.observability.alert_manager.create_alert(
                    f"Simulated {error_type.replace('_', ' ').title()}",
                    f"Error simulation triggered: {error_type}",
                    severity,
                    "error_simulation",
                    tags={"simulation": "true", "error_type": error_type}
                )
            
            await asyncio.sleep(random.uniform(1.0, 3.0))
    
    def scenario_forensic_analysis(self):
        """Scenario: Forensic analysis with time-warping."""
        print("\n🔍 Starting Forensic Analysis Scenario")
        print("   - Time-warped replay of past events")
        print("   - Event correlation and analysis")
        print("   - Historical performance review")
        
        # Get events from the last few minutes for replay
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=5)
        
        # Start forensic replay
        replay_data = self.telemetry.start_forensic_replay(start_time, end_time, speed=5.0)
        
        print(f"   📊 Replay started: {replay_data['event_count']} events")
        print(f"   ⏰ Time range: {start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}")
        print(f"   🚀 Playback speed: {replay_data['speed']}x")
        
        # Analyze events during replay
        events = self.telemetry.get_events(time_range=(start_time, end_time))
        
        # Event analysis
        event_types = {}
        error_count = 0
        sources = set()
        
        for event in events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            if event.event_type == "error":
                error_count += 1
            sources.add(event.source)
        
        print(f"\n   📈 Analysis Results:")
        print(f"      Total events: {len(events)}")
        print(f"      Error events: {error_count}")
        print(f"      Event sources: {len(sources)}")
        print(f"      Event types: {dict(list(event_types.items())[:5])}")
        
        # Log correlation analysis
        correlations = self.observability.log_correlator.correlate_logs(timedelta(minutes=5))
        if correlations:
            print(f"      Log correlations found: {len(correlations)}")
        
        # Stop replay
        self.telemetry.stop_forensic_replay()
        print("   ✅ Forensic analysis complete")
    
    def display_system_status(self):
        """Display current system status and metrics."""
        print("\n" + "="*60)
        print("📊 SYSTEM STATUS DASHBOARD")
        print("="*60)
        
        # System health
        health = self.telemetry.get_system_health()
        status_emoji = "🟢" if health["status"] == "healthy" else "🟡" if health["status"] == "degraded" else "🔴"
        print(f"{status_emoji} System Health: {health['status'].upper()} (Score: {health['health_score']}/100)")
        
        # Recent activity
        recent_events = self.telemetry.get_events(
            time_range=(datetime.utcnow() - timedelta(minutes=1), datetime.utcnow())
        )
        print(f"📈 Recent Activity: {len(recent_events)} events in last minute")
        
        # Active alerts
        active_alerts = self.observability.alert_manager.get_alerts(status=AlertStatus.ACTIVE)
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        print(f"🚨 Active Alerts: {len(active_alerts)} total, {len(critical_alerts)} critical")
        
        # Performance summary
        performance_report = self.telemetry.profiler.get_performance_report()
        print(f"⚡ Performance: {performance_report['summary']['total_operations']} operations tracked")
        
        # Bottlenecks
        bottlenecks = self.telemetry.profiler.get_bottlenecks("high")
        print(f"🐌 High-Severity Bottlenecks: {len(bottlenecks)}")
        
        # Topology
        topology = self.observability.topology_discovery.get_topology()
        if topology:
            print(f"🌐 Network Topology: {len(topology.nodes)} nodes, {len(topology.edges)} connections")
        
        print("="*60)
    
    def export_telemetry_report(self, filename: str = None):
        """Export comprehensive telemetry report."""
        if filename is None:
            filename = f"telemetry_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        print(f"\n📄 Exporting telemetry report to {filename}")
        
        # Get comprehensive report
        report = self.observability.get_observability_report()
        
        # Add additional demo-specific data
        report["demo_metadata"] = {
            "demo_duration": "multiple_scenarios",
            "agents_simulated": len(self.agents),
            "export_time": datetime.utcnow().isoformat(),
            "scenarios_run": [
                "normal_operations",
                "performance_stress", 
                "error_conditions",
                "forensic_analysis"
            ]
        }
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"   ✅ Report exported successfully")
        print(f"   📊 Report size: {len(json.dumps(report, default=str))} characters")
        
        return filename


async def main():
    """Main demo function."""
    print("🚀 Archangel Telemetry & Observability System Demo")
    print("="*60)
    
    # Initialize systems
    print("🔧 Initializing telemetry and observability systems...")
    telemetry = initialize_telemetry("archangel_demo")
    observability = initialize_observability()
    
    # Create scenario manager
    scenario_manager = DemoScenarioManager(telemetry, observability)
    
    # Set up notification handlers and topology discovery
    scenario_manager.setup_notification_handlers()
    scenario_manager.setup_topology_discovery()
    
    # Start continuous monitoring
    observability.start_monitoring(update_interval=10)
    
    print("✅ Systems initialized successfully")
    
    try:
        # Run demo scenarios
        print("\n🎬 Starting demo scenarios...")
        
        # Scenario 1: Normal Operations
        await scenario_manager.scenario_normal_operations(duration=30)
        scenario_manager.display_system_status()
        
        # Brief pause
        await asyncio.sleep(2)
        
        # Scenario 2: Performance Stress
        await scenario_manager.scenario_performance_stress(duration=20)
        scenario_manager.display_system_status()
        
        # Brief pause
        await asyncio.sleep(2)
        
        # Scenario 3: Error Conditions
        await scenario_manager.scenario_error_conditions(duration=15)
        scenario_manager.display_system_status()
        
        # Brief pause
        await asyncio.sleep(2)
        
        # Scenario 4: Forensic Analysis
        scenario_manager.scenario_forensic_analysis()
        
        # Final status display
        print("\n🏁 Demo scenarios completed!")
        scenario_manager.display_system_status()
        
        # Export final report
        report_file = scenario_manager.export_telemetry_report()
        
        print(f"\n✨ Demo completed successfully!")
        print(f"📄 Full telemetry report saved to: {report_file}")
        print("\nKey features demonstrated:")
        print("  ✅ OpenTelemetry integration with distributed tracing")
        print("  ✅ Real-time performance monitoring and profiling")
        print("  ✅ Automatic bottleneck detection")
        print("  ✅ Alert management and notifications")
        print("  ✅ Log correlation and analysis")
        print("  ✅ Time-warping for forensic analysis")
        print("  ✅ System topology discovery")
        print("  ✅ Comprehensive observability reporting")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        for agent in scenario_manager.agents:
            agent.stop()
        print("\n🧹 Cleanup completed")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())