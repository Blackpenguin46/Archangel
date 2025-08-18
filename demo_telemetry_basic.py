#!/usr/bin/env python3
"""
Basic demo of the telemetry and observability system.
This demo focuses on core functionality without complex tracing.
"""

import asyncio
import time
import json
import random
from datetime import datetime, timedelta
import sys
from unittest.mock import MagicMock

# Mock OpenTelemetry modules
sys.modules['opentelemetry'] = MagicMock()
sys.modules['opentelemetry.trace'] = MagicMock()
sys.modules['opentelemetry.metrics'] = MagicMock()
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

# Import our modules
from monitoring.telemetry import initialize_telemetry, get_telemetry, PerformanceMetrics
from monitoring.observability import (
    initialize_observability, get_observability,
    AlertSeverity, AlertStatus, LogEntry
)


class BasicDemoAgent:
    """Basic demo agent that generates telemetry without complex tracing."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.telemetry = get_telemetry()
        self.is_running = False
        self.operation_count = 0
    
    async def simulate_operations(self, duration: int = 15):
        """Simulate agent operations."""
        print(f"🤖 Agent {self.agent_id} starting operations...")
        
        operations = [
            "data_analysis", "decision_making", "communication",
            "learning", "planning", "task_execution"
        ]
        
        start_time = time.time()
        self.is_running = True
        
        while self.is_running and (time.time() - start_time) < duration:
            operation = random.choice(operations)
            
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
            work_duration = random.uniform(0.1, 0.8)
            await asyncio.sleep(work_duration)
            
            # Record performance metrics
            performance_metric = PerformanceMetrics(
                component=f"agent_{self.agent_id}",
                operation=operation,
                duration_ms=work_duration * 1000,
                cpu_usage=random.uniform(10, 80),
                memory_usage=random.uniform(100, 1000),
                success=random.random() > 0.1  # 90% success rate
            )
            
            self.telemetry.metrics.record_performance_metric(performance_metric)
            
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
                print(f"   ✅ Agent {self.agent_id}: Completed {operation} ({work_duration:.2f}s)")
            
            self.operation_count += 1
            
            # Random delay between operations
            await asyncio.sleep(random.uniform(0.3, 1.5))
        
        self.is_running = False
        print(f"🏁 Agent {self.agent_id} completed {self.operation_count} operations")
    
    def stop(self):
        """Stop the agent."""
        self.is_running = False


def display_telemetry_dashboard():
    """Display a telemetry dashboard."""
    telemetry = get_telemetry()
    observability = get_observability()
    
    print("\n" + "="*60)
    print("📊 TELEMETRY DASHBOARD")
    print("="*60)
    
    # Get recent events
    recent_events = telemetry.get_events()
    
    # Analyze events
    event_types = {}
    sources = set()
    error_count = 0
    success_count = 0
    
    for event in recent_events:
        event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        sources.add(event.source)
        if event.event_type == "error":
            error_count += 1
        elif event.event_type == "operation_completed":
            success_count += 1
    
    print(f"📈 Total Events: {len(recent_events)}")
    print(f"✅ Successful Operations: {success_count}")
    print(f"❌ Error Events: {error_count}")
    print(f"🎯 Active Sources: {len(sources)}")
    
    # Calculate success rate
    if success_count + error_count > 0:
        success_rate = (success_count / (success_count + error_count)) * 100
        print(f"📊 Success Rate: {success_rate:.1f}%")
    
    # Show event types
    if event_types:
        print(f"\n📋 Event Types:")
        sorted_types = sorted(event_types.items(), key=lambda x: x[1], reverse=True)
        for event_type, count in sorted_types:
            print(f"   {event_type}: {count}")
    
    # System health
    health = telemetry.get_system_health()
    status_emoji = "🟢" if health["status"] == "healthy" else "🟡" if health["status"] == "degraded" else "🔴"
    print(f"\n{status_emoji} System Health: {health['status'].upper()} (Score: {health['health_score']}/100)")
    
    # Performance summary
    for source in sources:
        if source.startswith("agent_"):
            agent_id = source.replace("agent_", "")
            summary = telemetry.metrics.get_performance_summary(source)
            if summary:
                print(f"⚡ Agent {agent_id}: {summary['total_operations']} ops, "
                      f"{summary['success_rate']:.1%} success, "
                      f"{summary['duration_stats']['mean']:.1f}ms avg")
    
    print("="*60)


def demonstrate_alert_system():
    """Demonstrate alert management."""
    print("\n🚨 ALERT SYSTEM DEMO")
    print("-" * 30)
    
    observability = get_observability()
    
    # Create sample alerts
    print("Creating sample alerts...")
    
    alert1 = observability.alert_manager.create_alert(
        "High Error Rate",
        "Error rate has exceeded 15% threshold",
        AlertSeverity.HIGH,
        "error_monitor",
        tags={"type": "error_rate", "threshold": "15%"}
    )
    print("🔴 HIGH: High Error Rate alert created")
    
    alert2 = observability.alert_manager.create_alert(
        "Performance Degradation",
        "Average response time increased by 200%",
        AlertSeverity.MEDIUM,
        "performance_monitor",
        tags={"type": "performance", "metric": "response_time"}
    )
    print("🟡 MEDIUM: Performance Degradation alert created")
    
    alert3 = observability.alert_manager.create_alert(
        "Resource Usage Notice",
        "Memory usage approaching 70% capacity",
        AlertSeverity.LOW,
        "resource_monitor",
        tags={"type": "resource", "metric": "memory"}
    )
    print("🟢 LOW: Resource Usage Notice alert created")
    
    # Show alert summary
    all_alerts = observability.alert_manager.get_alerts()
    active_alerts = observability.alert_manager.get_alerts(status=AlertStatus.ACTIVE)
    
    print(f"\n📊 Alert Summary:")
    print(f"   Total alerts: {len(all_alerts)}")
    print(f"   Active alerts: {len(active_alerts)}")
    
    # Demonstrate alert management
    print(f"\n🔧 Alert Management:")
    
    # Acknowledge first alert
    if len(all_alerts) > 0:
        ack_result = observability.alert_manager.acknowledge_alert(alert1, "demo_operator")
        if ack_result:
            print(f"   ✅ Acknowledged alert: {alert1[:8]}...")
    
    # Resolve second alert
    if len(all_alerts) > 1:
        resolve_result = observability.alert_manager.resolve_alert(alert2)
        if resolve_result:
            print(f"   ✅ Resolved alert: {alert2[:8]}...")
    
    # Final status
    final_active = observability.alert_manager.get_alerts(status=AlertStatus.ACTIVE)
    final_ack = observability.alert_manager.get_alerts(status=AlertStatus.ACKNOWLEDGED)
    final_resolved = observability.alert_manager.get_alerts(status=AlertStatus.RESOLVED)
    
    print(f"\n📈 Final Status:")
    print(f"   Active: {len(final_active)}")
    print(f"   Acknowledged: {len(final_ack)}")
    print(f"   Resolved: {len(final_resolved)}")


async def demonstrate_forensic_analysis():
    """Demonstrate forensic analysis."""
    print("\n🔍 FORENSIC ANALYSIS DEMO")
    print("-" * 30)
    
    telemetry = get_telemetry()
    
    # Get all events for analysis
    all_events = telemetry.get_events()
    
    if not all_events:
        print("No events available for analysis")
        return
    
    print(f"📊 Analyzing {len(all_events)} events...")
    
    # Time-based analysis
    if len(all_events) > 1:
        start_time = min(event.timestamp for event in all_events)
        end_time = max(event.timestamp for event in all_events)
        duration = (end_time - start_time).total_seconds()
        
        print(f"⏰ Time Range: {duration:.1f} seconds")
        print(f"📈 Event Rate: {len(all_events)/max(duration, 1):.1f} events/second")
    
    # Error analysis
    error_events = [e for e in all_events if e.event_type == "error"]
    if error_events:
        print(f"\n🔴 Error Analysis:")
        print(f"   Total errors: {len(error_events)}")
        
        # Group errors by source
        error_sources = {}
        for error in error_events:
            error_sources[error.source] = error_sources.get(error.source, 0) + 1
        
        for source, count in error_sources.items():
            print(f"   {source}: {count} errors")
    
    # Operation analysis
    completed_ops = [e for e in all_events if e.event_type == "operation_completed"]
    if completed_ops:
        print(f"\n✅ Operation Analysis:")
        print(f"   Completed operations: {len(completed_ops)}")
        
        # Calculate average duration
        durations = []
        for op in completed_ops:
            if "duration_ms" in op.data:
                durations.append(op.data["duration_ms"])
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            min_duration = min(durations)
            
            print(f"   Average duration: {avg_duration:.1f}ms")
            print(f"   Max duration: {max_duration:.1f}ms")
            print(f"   Min duration: {min_duration:.1f}ms")
    
    # Demonstrate time-warp replay
    print(f"\n⏰ Time-Warp Replay Demo:")
    if len(all_events) > 0:
        # Use a small time window for demo
        recent_time = datetime.utcnow() - timedelta(minutes=1)
        replay_data = telemetry.start_forensic_replay(recent_time, datetime.utcnow(), speed=5.0)
        
        print(f"   🚀 Started replay at {replay_data['speed']}x speed")
        print(f"   📊 Replaying {replay_data['event_count']} events")
        
        # Simulate some replay time
        time.sleep(0.5)
        
        telemetry.stop_forensic_replay()
        print(f"   ✅ Replay completed")


async def main():
    """Main demo function."""
    print("🚀 Archangel Telemetry System - Basic Demo")
    print("="*60)
    
    # Initialize systems
    print("🔧 Initializing systems...")
    telemetry = initialize_telemetry("archangel_basic_demo")
    observability = initialize_observability()
    print("✅ Systems initialized")
    
    # Create demo agents
    agents = [
        BasicDemoAgent("alpha"),
        BasicDemoAgent("beta")
    ]
    
    print(f"\n🤖 Starting {len(agents)} demo agents for 15 seconds...")
    
    # Run agents
    try:
        tasks = [agent.simulate_operations(duration=15) for agent in agents]
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted")
        for agent in agents:
            agent.stop()
    
    print("\n🏁 Agent simulation completed")
    
    # Display results
    display_telemetry_dashboard()
    
    # Demonstrate alert system
    demonstrate_alert_system()
    
    # Demonstrate forensic analysis
    await demonstrate_forensic_analysis()
    
    # Export data
    print(f"\n📄 Exporting telemetry data...")
    export_data = telemetry.export_telemetry_data("json")
    
    filename = f"basic_demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        f.write(export_data)
    
    print(f"✅ Data exported to: {filename}")
    
    # Show file size
    file_size = len(export_data)
    print(f"📊 Export size: {file_size:,} characters")
    
    # Final summary
    print(f"\n✨ Demo completed successfully!")
    print("Features demonstrated:")
    print("  ✅ Event recording and tracking")
    print("  ✅ Performance metrics collection")
    print("  ✅ Real-time system health monitoring")
    print("  ✅ Alert creation and management")
    print("  ✅ Forensic analysis capabilities")
    print("  ✅ Time-warp replay functionality")
    print("  ✅ Comprehensive data export")


if __name__ == "__main__":
    asyncio.run(main())