"""
Auto AI Agents Generator

An autonomous system that uses Claude Code CLI to generate multi-agent systems
based on free-form context, with auto-setup of MCPs and full autonomous iteration.
"""

__version__ = "0.1.0"
__author__ = "Auto Agent Generator"

from .context_parser import ContextParser
from .config_parser import ConfigParser
from .progress_manager import ProgressManager
from .claude_interface import ClaudeInterface
from .mcp_manager import MCPManager
from .agent_generator import AgentGenerator

__all__ = [
    "ContextParser",
    "ConfigParser",
    "ProgressManager",
    "ClaudeInterface",
    "MCPManager",
    "AgentGenerator",
]
