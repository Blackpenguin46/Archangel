"""
Autonomous AI agents for red and blue teams
"""

from .red.agent import RedTeamAgent
from .blue.agent import BlueTeamAgent

__all__ = ["RedTeamAgent", "BlueTeamAgent"]