"""
Archangel Autonomous AI Evolution - Agent Framework
Multi-agent system for autonomous cybersecurity simulation
"""

# Core components that should always be available
from .base_agent import BaseAgent, AgentConfig, AgentStatus, Team, Role

# Optional components that may have external dependencies
__all__ = [
    'BaseAgent',
    'AgentConfig', 
    'AgentStatus',
    'Team',
    'Role'
]

# Try to import optional components
try:
    from .coordinator import LangGraphCoordinator
    __all__.append('LangGraphCoordinator')
except ImportError:
    pass

try:
    from .communication import MessageBus, AgentMessage, TeamMessage
    __all__.extend(['MessageBus', 'AgentMessage', 'TeamMessage'])
except ImportError:
    pass

try:
    from .red_team import ReconAgent, ExploitAgent, PersistenceAgent, ExfiltrationAgent
    __all__.extend(['ReconAgent', 'ExploitAgent', 'PersistenceAgent', 'ExfiltrationAgent'])
except ImportError:
    pass