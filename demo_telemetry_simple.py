#!/usr/bin/env python3
"""
Simple demo of the telemetry and observability system.
This demo works without external dependencies by using mocked OpenTelemetry components.
"""

import asyncio
import time
import json
import random
from datetime import datetime, timedelta
from threading import Thread
import sys
from unittest.mock import MagicMock

# Mock OpenTelemetry modules before importing our code
mock_span_context = MagicMock()
mock_span_context.span_id = 12345678901234567890
mock_span_context.trace_id = 98765432109876543210987654321098765432

mock_span = MagicMock()
mock_span.get_span_context.return_value = mock_span_context
mock_span.__enter__ = MagicMock(return_value=mock_span)
mock_span.__exit__ = MagicMock(return_value=None)

mock_tracer = MagicMock()
mock_tracer.start_as_current_span.return_value = mock_span

mock_otel_trace = MagicMock()
mock_otel_trace.get_tracer.return_value = mock_tracer

mock_meter = MagicMock()
mock_otel_metrics = MagicMock()
mock_otel_metrics.get_meter.return_value = mock_meter

# Mock the OpenTelemetry imports
sys.modules['opentelemetry'] = MagicMock()
sys.modules['opentelemetry.trace'] = mock_otel_trace
sys.modules['opentelemetry.metrics'] = mock_otel_metrics
sys.modules['opentelemetry.exporter'] = MagicMock()
sys.modules['opentelemetry.exporter.jaeger'] = MagicMock()
sys.modules['opentelemetry.exporter.jaeger.thrift'] = MagicMock()
sys.modules['opentelemetry.exporter.prometheus'] = MagicMock()
sys.modules['opentelemetry.sdk'] = MagicMock()
sys.modules['opentelemetry.sdk.trace'] = MagicMock()
sys.modules['opentelemetry.sdk.trace.export'] = MagicMock()
sys.modules['opentelemetry.sdk.metrics'] = MagicMock()
sys.modules['opentelemetry.sdk.resources'] = MagicMock()
sys.modules['opentelemetry.instrumentation'] = MagicMock()
sys.modules['opentelemetry.instrumentation.logging'] = MagicMock()
sys.modules['opentelemetry.instrumentation.requests'] = MagicMock()
sys.modules['opentelemetry.instrumentation.asyncio'] = MagicMock()
sys.modules['opentelemetry.trace.status'] = MagicMock()
sys.modules['opentelemetry.semconv'] = MagicMock()
sys.modules['opentelemetry.semconv.trace'] = MagicMock()

# Now import our telemetry modules
from monitoring.telemetry import initialize_telemetry, get_telemetry
from monitoring.observability import (
    initialize_observability, get_observability,
    AlertSeverity, AlertStatus, LogEntry
)


class SimpleDemoAgent:
    """Simple demo agent for telemetry demonstration."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.telemetry = get_telemetry()
        self.is_running = False
        self.operation_count = 0
    
    async def simulate_operations(self, duration: int = 30):
        """Simulate agent operations for demonstration."""
        print(f"🤖 Agent {self.agent_id} starting operations...")
        
        operations = [
            "data_analysis", "decision_making", "communication",
            "learning", "planning", "task_execution"
        ]
        
        start_time = time.time()
        self.is_running = True
        
        while self.is_running and (time.time() - start_time) < duration:
            operation = random.choice(operations)
            
            # Use telemetry tracing
            with self.telemetry.trace_operation(
                operation, 
                f"agent_{self.agent_id}",
                agent_id=self.agent_id
            ) as span:
                
                # Record operation start
                self.telemetry.record_event(
                    "operation_started",
                    f"agent_{self.agent_id}",
                    {
                        "operation": operation,
                        "agent_id": self.agent_id,
                        "sequence": self.operation_count
                    }
                )
                
                # Simulate work
                work_duration = random.uniform(0.1, 1.0)
                await asyncio.sleep(work_duration)
                
                # Simulate occasional errors (10% chance)
                if random.random() < 0.1:
                    error_msg = f"Simulated error in {operation}"
                    self.telemetry.record_event(
                        "error",
                        f"agent_{self.agent_id}",
                        {
                            "operation": operation,
                            "error_message": error_msg,
                            "agent_id": self.agent_id
                        }
                    )
                    print(f"   ❌ Agent {self.agent_id}: Error in {operation}")
                else:
                    # Record successful completion
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
                    print(f"   ✅ Agent {self.agent_id}: Completed {operation}")
                
                self.operation_count += 1
            
            # Random delay between operations
            await asyncio.sleep(random.uniform(0.5, 2.0))
        
        self.is_running = False
        print(f"🏁 Agent {self.agent_id} completed {self.operation_count} operations")
    
    def stop(self):
        """Stop the agent."""
        self.is_running = False


def display_telemetry_summary():
    """Display a summary of collected telemetry data."""
    telemetry = get_telemetry()
    observability = get_observability()
    
    print("\n" + "="*60)
    print("📊 TELEMETRY SUMMARY")
    print("="*60)
    
    # Get recent events
    recent_events = telemetry.get_events(
        time_range=(datetime.utcnow() - timedelta(minutes=5), datetime.utcnow())
    )
    
    # Analyze events
    event_types = {}
    sources = set()
    error_count = 0
    
    for event in recent_events:
        event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        sources.add(event.source)
        if event.event_type == "error":
            error_count += 1
    
    print(f"📈 Total Events: {len(recent_events)}")
    print(f"🔴 Error Events: {error_count}")
    print(f"🎯 Event Sources: {len(sources)}")
    print(f"📋 Event Types: {len(event_types)}")
    
    # Show top event types
    if event_types:
        print("\n🏆 Top Event Types:")
        sorted_types = sorted(event_types.items(), key=lambda x: x[1], reverse=True)
        for event_type, count in sorted_types[:5]:
            print(f"   {event_type}: {count}")
    
    # System health
    health = telemetry.get_system_health()
    status_emoji = "🟢" if health["status"] == "healthy" else "🟡" if health["status"] == "degraded" else "🔴"
    print(f"\n{status_emoji} System Health: {health['status'].upper()} (Score: {health['health_score']}/100)")
    
    # Performance data
    performance_report = telemetry.profiler.get_performance_report()
    print(f"⚡ Operations Tracked: {performance_report['summary']['total_operations']}")
    
    # Bottlenecks
    bottlenecks = telemetry.profiler.get_bottlenecks()
    if bottlenecks:
        print(f"🐌 Performance Bottlenecks: {len(bottlenecks)}")
        for bottleneck in bottlenecks[:3]:  # Show top 3
            print(f"   - {bottleneck['type']}: {bottleneck['component']}")
    
    # Active alerts
    active_alerts = observability.alert_manager.get_alerts(status=AlertStatus.ACTIVE)
    if active_alerts:
        print(f"\n🚨 Active Alerts: {len(active_alerts)}")
        for alert in active_alerts[:3]:  # Show top 3
            print(f"   - {alert.severity.value.upper()}: {alert.title}")
    else:
        print(f"\n✅ No Active Alerts")
    
    print("="*60)


def demonstrate_forensic_analysis():
    """Demonstrate forensic analysis capabilities."""
    print("\n🔍 FORENSIC ANALYSIS DEMONSTRATION")
    print("-" * 40)
    
    telemetry = get_telemetry()
    
    # Get events from the last few minutes
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=3)
    
    print(f"🕐 Analyzing events from {start_time.strftime('%H:%M:%S')} to {end_time.strftime('%H:%M:%S')}")
    
    # Start forensic replay
    replay_data = telemetry.start_forensic_replay(start_time, end_time, speed=3.0)
    print(f"🚀 Forensic replay started at {replay_data['speed']}x speed")
    print(f"📊 Replaying {replay_data['event_count']} events")
    
    # Analyze the events
    events = telemetry.get_events(time_range=(start_time, end_time))
    
    if events:
        print(f"\n📈 Event Analysis:")
        
        # Timeline analysis
        event_timeline = {}
        for event in events:
            minute_key = event.timestamp.strftime('%H:%M')
            if minute_key not in event_timeline:
                event_timeline[minute_key] = []
            event_timeline[minute_key].append(event)
        
        for time_key in sorted(event_timeline.keys()):
            events_in_minute = event_timeline[time_key]
            error_events = [e for e in events_in_minute if e.event_type == "error"]
            print(f"   {time_key}: {len(events_in_minute)} events ({len(error_events)} errors)")
        
        # Error correlation
        error_events = [e for e in events if e.event_type == "error"]
        if error_events:
            print(f"\n🔴 Error Analysis:")
            error_sources = {}
            for error in error_events:
                error_sources[error.source] = error_sources.get(error.source, 0) + 1
            
            for source, count in error_sources.items():
                print(f"   {source}: {count} errors")
    else:
        print("   No events found in the specified time range")
    
    # Stop forensic replay
    telemetry.stop_forensic_replay()
    print("✅ Forensic analysis complete")


def demonstrate_alert_system():
    """Demonstrate the alert management system."""
    print("\n🚨 ALERT SYSTEM DEMONSTRATION")
    print("-" * 40)
    
    observability = get_observability()
    
    # Create sample alerts
    alerts_created = []
    
    # High severity alert
    alert_id1 = observability.alert_manager.create_alert(
        "High CPU Usage Detected",
        "System CPU usage has exceeded 90% for more than 5 minutes",
        AlertSeverity.HIGH,
        "system_monitor",
        tags={"component": "cpu", "threshold": "90%"}
    )
    alerts_created.append(alert_id1)
    print("🔴 Created HIGH severity alert: High CPU Usage")
    
    # Medium severity alert
    alert_id2 = observability.alert_manager.create_alert(
        "Memory Usage Warning",
        "Memory usage is approaching 75% capacity",
        AlertSeverity.MEDIUM,
        "system_monitor",
        tags={"component": "memory", "threshold": "75%"}
    )
    alerts_created.append(alert_id2)
    print("🟡 Created MEDIUM severity alert: Memory Usage Warning")
    
    # Low severity alert
    alert_id3 = observability.alert_manager.create_alert(
        "Disk Space Notice",
        "Disk space usage is at 60% capacity",
        AlertSeverity.LOW,
        "system_monitor",
        tags={"component": "disk", "threshold": "60%"}
    )
    alerts_created.append(alert_id3)
    print("🟢 Created LOW severity alert: Disk Space Notice")
    
    # Demonstrate alert management
    print(f"\n📋 Alert Management:")
    
    # Show all alerts
    all_alerts = observability.alert_manager.get_alerts()
    print(f"   Total alerts: {len(all_alerts)}")
    
    # Filter by severity
    high_alerts = observability.alert_manager.get_alerts(severity=AlertSeverity.HIGH)
    print(f"   High severity alerts: {len(high_alerts)}")
    
    # Acknowledge an alert
    if alerts_created:
        ack_result = observability.alert_manager.acknowledge_alert(alerts_created[0], "demo_user")
        if ack_result:
            print(f"   ✅ Acknowledged alert: {alerts_created[0]}")
    
    # Resolve an alert
    if len(alerts_created) > 1:
        resolve_result = observability.alert_manager.resolve_alert(alerts_created[1])
        if resolve_result:
            print(f"   ✅ Resolved alert: {alerts_created[1]}")
    
    # Show final alert status
    active_alerts = observability.alert_manager.get_alerts(status=AlertStatus.ACTIVE)
    acknowledged_alerts = observability.alert_manager.get_alerts(status=AlertStatus.ACKNOWLEDGED)
    resolved_alerts = observability.alert_manager.get_alerts(status=AlertStatus.RESOLVED)
    
    print(f"\n📊 Final Alert Status:")
    print(f"   Active: {len(active_alerts)}")
    print(f"   Acknowledged: {len(acknowledged_alerts)}")
    print(f"   Resolved: {len(resolved_alerts)}")


async def main():
    """Main demo function."""
    print("🚀 Archangel Telemetry System - Simple Demo")
    print("="*60)
    
    # Initialize systems
    print("🔧 Initializing telemetry and observability systems...")
    telemetry = initialize_telemetry("archangel_simple_demo")
    observability = initialize_observability()
    print("✅ Systems initialized")
    
    # Create demo agents
    agents = [
        SimpleDemoAgent("alpha"),
        SimpleDemoAgent("beta"),
        SimpleDemoAgent("gamma")
    ]
    
    print(f"\n🤖 Starting {len(agents)} demo agents...")
    
    # Run agents for a short duration
    try:
        tasks = [agent.simulate_operations(duration=20) for agent in agents]
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
        for agent in agents:
            agent.stop()
    
    # Display results
    print("\n🏁 Agent simulation completed")
    display_telemetry_summary()
    
    # Demonstrate forensic analysis
    demonstrate_forensic_analysis()
    
    # Demonstrate alert system
    demonstrate_alert_system()
    
    # Export telemetry data
    print(f"\n📄 Exporting telemetry data...")
    export_data = telemetry.export_telemetry_data("json")
    
    filename = f"simple_demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        f.write(export_data)
    
    print(f"✅ Telemetry data exported to: {filename}")
    
    # Final summary
    print(f"\n✨ Demo completed successfully!")
    print("Key features demonstrated:")
    print("  ✅ Event recording and tracking")
    print("  ✅ Distributed tracing simulation")
    print("  ✅ Performance monitoring")
    print("  ✅ Alert management")
    print("  ✅ Forensic analysis with time-warping")
    print("  ✅ System health monitoring")
    print("  ✅ Telemetry data export")


if __name__ == "__main__":
    asyncio.run(main())