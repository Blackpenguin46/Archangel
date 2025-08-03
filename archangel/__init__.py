"""
Archangel AI vs AI Cyber Conflict System
A self-contained, AI-vs-AI cyber warfare simulation showcasing an autonomous 
Red Team attacking a fully autonomous Blue Team in isolated environments.
"""

__version__ = "1.0.0"
__author__ = "Blackpenguin46"
__description__ = "AI vs AI Autonomous Cyber Conflict System"

from .core.orchestrator import ArchangelOrchestrator
from .agents.red.agent import RedTeamAgent
from .agents.blue.agent import BlueTeamAgent

__all__ = [
    "ArchangelOrchestrator", 
    "RedTeamAgent", 
    "BlueTeamAgent"
]