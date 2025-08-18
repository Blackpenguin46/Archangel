# Agent API Reference

The Agent API provides comprehensive interfaces for creating, managing, and interacting with autonomous agents in the Archangel system.

## Table of Contents
- [Base Agent Interface](#base-agent-interface)
- [Red Team Agents](#red-team-agents)
- [Blue Team Agents](#blue-team-agents)
- [Agent Communication](#agent-communication)
- [Memory Integration](#memory-integration)
- [Examples](#examples)

## Base Agent Interface

All agents inherit from the `BaseAgent` class, providing standardized functionality for autonomous operation.

### Class: BaseAgent

```python
class BaseAgent:
    """
    Base class for all autonomous agents in the Archangel system.
    
    Provides core functionality for perception, reasoning, planning,
    and execution in cybersecurity simulation environments.
    """
    
    def __init__(self, agent_id: str, team: Team, config: AgentConfig):
        """
        Initialize a new agent instance.
        
        Args:
            agent_id: Unique identifier for the agent
            team: Team assignment (RED_TEAM or BLUE_TEAM)
            config: Agent configuration parameters
            
        Example:
            >>> config = AgentConfig(
            ...     llm_model="gpt-4-turbo",
            ...     memory_size=1000,
            ...     reasoning_depth=3
            ... )
            >>> agent = BaseAgent("recon-001", Team.RED_TEAM, config)
        """
```

### Core Methods

#### perceive_environment()
```python
def perceive_environment(self) -> EnvironmentState:
    """
    Gather information about the current environment state.
    
    Returns:
        EnvironmentState: Current state of the simulation environment
        
    Example:
        >>> state = agent.perceive_environment()
        >>> print(f"Active services: {state.services}")
        >>> print(f"Network topology: {state.network}")
    """
```

#### reason_about_situation()
```python
def reason_about_situation(self, state: EnvironmentState) -> ReasoningResult:
    """
    Apply LLM-powered reasoning to analyze the current situation.
    
    Args:
        state: Current environment state from perception
        
    Returns:
        ReasoningResult: Analysis and recommended actions
        
    Example:
        >>> reasoning = agent.reason_about_situation(state)
        >>> print(f"Threat level: {reasoning.threat_assessment}")
        >>> print(f"Recommended actions: {reasoning.actions}")
    """
```

#### plan_actions()
```python
def plan_actions(self, reasoning: ReasoningResult) -> ActionPlan:
    """
    Create a structured action plan based on reasoning results.
    
    Args:
        reasoning: Results from situation analysis
        
    Returns:
        ActionPlan: Ordered sequence of actions to execute
        
    Example:
        >>> plan = agent.plan_actions(reasoning)
        >>> for action in plan.actions:
        ...     print(f"Action: {action.type}, Priority: {action.priority}")
    """
```

#### execute_action()
```python
def execute_action(self, action: Action) -> ActionResult:
    """
    Execute a specific action in the environment.
    
    Args:
        action: Action to execute
        
    Returns:
        ActionResult: Results and outcomes of the action
        
    Example:
        >>> action = Action(type="network_scan", target="192.168.1.0/24")
        >>> result = agent.execute_action(action)
        >>> if result.success:
        ...     print(f"Scan found {len(result.data)} hosts")
    """
```

## Red Team Agents

Specialized agents for offensive cybersecurity operations.

### ReconAgent

```python
class ReconAgent(BaseAgent):
    """
    Reconnaissance agent specializing in target discovery and intelligence gathering.
    
    Capabilities:
    - Network scanning and enumeration
    - Service discovery and fingerprinting
    - Vulnerability identification
    - Intelligence correlation
    """
    
    def scan_network(self, target_range: str, scan_type: ScanType = ScanType.STEALTH) -> ScanResults:
        """
        Perform network reconnaissance on target range.
        
        Args:
            target_range: CIDR notation network range (e.g., "192.168.1.0/24")
            scan_type: Type of scan to perform (STEALTH, AGGRESSIVE, COMPREHENSIVE)
            
        Returns:
            ScanResults: Discovered hosts, services, and potential vulnerabilities
            
        Example:
            >>> recon = ReconAgent("recon-001", Team.RED_TEAM, config)
            >>> results = recon.scan_network("10.0.0.0/24", ScanType.STEALTH)
            >>> for host in results.live_hosts:
            ...     print(f"Host: {host.ip}, OS: {host.os_guess}")
        """
```

### ExploitAgent

```python
class ExploitAgent(BaseAgent):
    """
    Exploitation agent specializing in vulnerability exploitation and payload delivery.
    
    Capabilities:
    - Exploit selection and customization
    - Payload generation and delivery
    - Post-exploitation activities
    - Privilege escalation
    """
    
    def select_exploit(self, vulnerability: Vulnerability) -> Exploit:
        """
        Select appropriate exploit for discovered vulnerability.
        
        Args:
            vulnerability: Target vulnerability information
            
        Returns:
            Exploit: Selected exploit with configuration
            
        Example:
            >>> exploit_agent = ExploitAgent("exploit-001", Team.RED_TEAM, config)
            >>> vuln = Vulnerability(cve="CVE-2021-44228", service="log4j")
            >>> exploit = exploit_agent.select_exploit(vuln)
            >>> print(f"Selected: {exploit.name}, Success rate: {exploit.reliability}")
        """
```

## Blue Team Agents

Specialized agents for defensive cybersecurity operations.

### SOCAnalystAgent

```python
class SOCAnalystAgent(BaseAgent):
    """
    Security Operations Center analyst agent for threat detection and response.
    
    Capabilities:
    - Alert monitoring and triage
    - Incident correlation and analysis
    - Threat hunting and investigation
    - Response coordination
    """
    
    def monitor_alerts(self, time_window: timedelta = timedelta(minutes=5)) -> List[Alert]:
        """
        Monitor and retrieve security alerts from SIEM systems.
        
        Args:
            time_window: Time window for alert retrieval
            
        Returns:
            List[Alert]: Security alerts requiring analysis
            
        Example:
            >>> soc_agent = SOCAnalystAgent("soc-001", Team.BLUE_TEAM, config)
            >>> alerts = soc_agent.monitor_alerts(timedelta(minutes=10))
            >>> high_priority = [a for a in alerts if a.severity == "HIGH"]
            >>> print(f"High priority alerts: {len(high_priority)}")
        """
```

### FirewallConfiguratorAgent

```python
class FirewallConfiguratorAgent(BaseAgent):
    """
    Firewall management agent for dynamic security rule deployment.
    
    Capabilities:
    - Traffic pattern analysis
    - Dynamic rule generation
    - Rule deployment and validation
    - Performance monitoring
    """
    
    def generate_firewall_rules(self, threats: List[Threat]) -> List[FirewallRule]:
        """
        Generate firewall rules to mitigate identified threats.
        
        Args:
            threats: List of identified threats requiring mitigation
            
        Returns:
            List[FirewallRule]: Generated firewall rules
            
        Example:
            >>> fw_agent = FirewallConfiguratorAgent("fw-001", Team.BLUE_TEAM, config)
            >>> threats = [Threat(type="port_scan", source="192.168.1.100")]
            >>> rules = fw_agent.generate_firewall_rules(threats)
            >>> for rule in rules:
            ...     print(f"Rule: {rule.action} {rule.source} -> {rule.destination}")
        """
```

## Agent Communication

Agents communicate through a secure message bus using standardized protocols.

### Message Types

```python
@dataclass
class AgentMessage:
    """Base message structure for inter-agent communication."""
    sender_id: str
    recipient_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime
    priority: Priority
    encryption_key: Optional[str] = None

@dataclass
class IntelligenceMessage(AgentMessage):
    """Intelligence sharing message between team members."""
    intelligence_type: IntelligenceType
    target_info: TargetInfo
    confidence_level: float
    source_reliability: float
    
    # Example usage
    intel_msg = IntelligenceMessage(
        sender_id="recon-001",
        recipient_id="exploit-001",
        message_type=MessageType.INTELLIGENCE,
        content={"vulnerability": "CVE-2021-44228", "target": "10.0.0.5"},
        timestamp=datetime.now(),
        priority=Priority.HIGH,
        intelligence_type=IntelligenceType.VULNERABILITY,
        target_info=TargetInfo(ip="10.0.0.5", service="log4j"),
        confidence_level=0.95,
        source_reliability=0.9
    )
```

### Communication Methods

```python
def send_team_message(self, message: AgentMessage) -> bool:
    """
    Send message to team members.
    
    Args:
        message: Message to send to team
        
    Returns:
        bool: True if message sent successfully
        
    Example:
        >>> message = IntelligenceMessage(...)
        >>> success = agent.send_team_message(message)
        >>> if success:
        ...     print("Intelligence shared with team")
    """

def broadcast_alert(self, alert: Alert) -> None:
    """
    Broadcast security alert to all relevant agents.
    
    Args:
        alert: Security alert to broadcast
        
    Example:
        >>> alert = Alert(
        ...     type="intrusion_detected",
        ...     severity="HIGH",
        ...     source="192.168.1.100"
        ... )
        >>> agent.broadcast_alert(alert)
    """
```

## Memory Integration

Agents integrate with vector memory systems for persistent learning and knowledge storage.

### Memory Operations

```python
def store_experience(self, experience: Experience) -> str:
    """
    Store experience in agent's memory system.
    
    Args:
        experience: Experience to store
        
    Returns:
        str: Unique identifier for stored experience
        
    Example:
        >>> experience = Experience(
        ...     action="network_scan",
        ...     outcome="successful",
        ...     context={"target": "192.168.1.0/24"},
        ...     lessons_learned=["Stealth scan avoided detection"]
        ... )
        >>> exp_id = agent.store_experience(experience)
    """

def retrieve_similar_experiences(self, query: str, limit: int = 5) -> List[Experience]:
    """
    Retrieve similar past experiences for decision-making.
    
    Args:
        query: Natural language query describing situation
        limit: Maximum number of experiences to retrieve
        
    Returns:
        List[Experience]: Relevant past experiences
        
    Example:
        >>> experiences = agent.retrieve_similar_experiences(
        ...     "scanning web server for vulnerabilities",
        ...     limit=3
        ... )
        >>> for exp in experiences:
        ...     print(f"Previous action: {exp.action}, Success: {exp.success}")
    """
```

## Complete Usage Examples

### Red Team Reconnaissance Workflow

```python
# Initialize reconnaissance agent
config = AgentConfig(
    llm_model="gpt-4-turbo",
    memory_size=1000,
    reasoning_depth=3,
    stealth_mode=True
)

recon_agent = ReconAgent("recon-001", Team.RED_TEAM, config)

# Perform environment perception
env_state = recon_agent.perceive_environment()
print(f"Target network: {env_state.target_network}")

# Conduct network reconnaissance
scan_results = recon_agent.scan_network("192.168.1.0/24", ScanType.STEALTH)
print(f"Discovered {len(scan_results.live_hosts)} live hosts")

# Analyze results and plan next actions
reasoning = recon_agent.reason_about_situation(env_state)
action_plan = recon_agent.plan_actions(reasoning)

# Share intelligence with team
for host in scan_results.live_hosts:
    if host.vulnerabilities:
        intel_msg = IntelligenceMessage(
            sender_id="recon-001",
            recipient_id="exploit-001",
            message_type=MessageType.INTELLIGENCE,
            content={"host": host.ip, "vulns": host.vulnerabilities},
            timestamp=datetime.now(),
            priority=Priority.HIGH,
            intelligence_type=IntelligenceType.VULNERABILITY,
            target_info=TargetInfo(ip=host.ip, services=host.services),
            confidence_level=0.9,
            source_reliability=0.95
        )
        recon_agent.send_team_message(intel_msg)
```

### Blue Team Incident Response Workflow

```python
# Initialize SOC analyst agent
config = AgentConfig(
    llm_model="gpt-4-turbo",
    memory_size=2000,
    reasoning_depth=4,
    alert_threshold="MEDIUM"
)

soc_agent = SOCAnalystAgent("soc-001", Team.BLUE_TEAM, config)

# Monitor for security alerts
alerts = soc_agent.monitor_alerts(timedelta(minutes=5))
high_priority_alerts = [a for a in alerts if a.severity in ["HIGH", "CRITICAL"]]

if high_priority_alerts:
    # Correlate related events
    for alert in high_priority_alerts:
        correlation = soc_agent.correlate_events([alert])
        
        if correlation.confidence > 0.8:
            # Create incident
            incident = soc_agent.create_incident(correlation)
            
            # Coordinate response
            response_plan = soc_agent.coordinate_response(incident)
            
            # Notify team members
            response_msg = TeamCoordinationMessage(
                sender_id="soc-001",
                recipient_id="all_blue_team",
                message_type=MessageType.COORDINATION,
                content={"incident_id": incident.id, "response_plan": response_plan},
                timestamp=datetime.now(),
                priority=Priority.URGENT,
                coordination_type=CoordinationType.INCIDENT_RESPONSE,
                proposed_action=response_plan.primary_action,
                required_resources=response_plan.resources,
                timeline=response_plan.timeline
            )
            soc_agent.send_team_message(response_msg)
```

## Error Handling

All agent methods include comprehensive error handling:

```python
try:
    scan_results = recon_agent.scan_network("192.168.1.0/24")
except NetworkTimeoutError as e:
    logger.warning(f"Network scan timeout: {e}")
    # Implement fallback strategy
except PermissionDeniedError as e:
    logger.error(f"Insufficient permissions: {e}")
    # Request elevated privileges or alternative approach
except AgentCommunicationError as e:
    logger.error(f"Communication failure: {e}")
    # Retry with exponential backoff
```

## Performance Considerations

- **Async Operations**: All I/O operations are asynchronous for better performance
- **Caching**: Frequently accessed data is cached in memory
- **Rate Limiting**: API calls are rate-limited to prevent overwhelming systems
- **Resource Management**: Automatic cleanup of resources after operations

## Security Features

- **Encryption**: All inter-agent communication is encrypted using TLS 1.3
- **Authentication**: Mutual authentication between agents
- **Authorization**: Role-based access control for agent capabilities
- **Audit Logging**: Complete audit trail of all agent actions

---

*For more examples and advanced usage patterns, see the [Training Materials](../training/) section.*