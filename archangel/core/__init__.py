"""
Core modules for Archangel AI vs AI system
"""

from .orchestrator import ArchangelOrchestrator
from .logging_system import ArchangelLogger
from .autonomous_reasoning_engine import AutonomousAgent
from .llm_integration import LocalLLMManager

__all__ = [
    "ArchangelOrchestrator",
    "ArchangelLogger", 
    "AutonomousAgent",
    "LocalLLMManager"
]