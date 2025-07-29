#!/usr/bin/env python3
"""
Archangel CLI - Interactive AI Security Expert Interface
The main entry point for users to interact with Archangel
"""

import asyncio
import sys
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.table import Table
from rich.live import Live
import time

from core.ai_security_expert import SecurityExpertAI
from core.conversational_ai import ConversationalSecurityAI

class ArchangelCLI:
    """
    Interactive CLI for Archangel AI Security Expert
    
    This provides the main user interface for interacting with
    the AI security expert system.
    """
    
    def __init__(self):
        self.console = Console()
        self.security_expert = SecurityExpertAI()
        self.conversational_ai = ConversationalSecurityAI(self.security_expert)
        self.running = False
        
    def display_banner(self):
        """Display the Archangel banner"""
        banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                          ARCHANGEL LINUX                          ║
    ║                     AI Security Expert System                     ║
    ║                                                                   ║
    ║    "While others automate hacking, Archangel understands it"      ║
    ╚═══════════════════════════════════════════════════════════════════╝
        """
        
        self.console.print(Panel(
            banner.strip(),
            style="bold blue",
            title="🛡️ Welcome to Archangel",
            subtitle="The AI That Thinks Like a Security Expert"
        ))
    
    def display_intro(self):
        """Display introduction and capabilities"""
        intro_text = """
🧠 **What makes me different:**
• I think step-by-step about security problems
• I explain my reasoning process clearly  
• I adapt strategies based on what I discover
• I teach you security concepts while we work

🎯 **What I can help with:**
• Security analysis and penetration testing
• Learning about vulnerabilities and exploits
• Planning security assessment strategies
• Explaining complex security concepts

💬 **How to interact with me:**
• Ask me to analyze a target: "analyze example.com"
• Ask for explanations: "why did you choose that approach?"
• Request teaching: "teach me about SQL injection"
• General questions: "what's the best way to test for XSS?"

🚨 **Important:** I'm designed for defensive security research and education only.
        """
        
        self.console.print(Panel(
            intro_text.strip(),
            title="🤖 About Archangel AI",
            style="green"
        ))
    
    def display_help(self):
        """Display help information"""
        help_table = Table(title="🔧 Archangel Commands")
        help_table.add_column("Command", style="cyan", no_wrap=True)
        help_table.add_column("Description", style="white")
        help_table.add_column("Example", style="yellow")
        
        help_table.add_row(
            "analyze <target>",
            "Perform AI security analysis",
            "analyze example.com"
        )
        help_table.add_row(
            "explain",
            "Explain my reasoning process", 
            "explain"
        )
        help_table.add_row(
            "teach <concept>",
            "Learn about security concepts",
            "teach sql injection"
        )
        help_table.add_row(
            "thoughts",
            "Show my current thinking process",
            "thoughts"
        )
        help_table.add_row(
            "status",
            "Display current analysis status",
            "status"
        )
        help_table.add_row(
            "help",
            "Show this help message",
            "help"
        )
        help_table.add_row(
            "exit",
            "Exit Archangel",
            "exit"
        )
        
        self.console.print(help_table)
    
    async def display_thinking_animation(self, message: str):
        """Display animated thinking process"""
        thinking_frames = ["🧠   ", "🧠.  ", "🧠.. ", "🧠..."]
        
        with Live(console=self.console, refresh_per_second=4) as live:
            for i in range(12):  # 3 seconds of animation
                frame = thinking_frames[i % len(thinking_frames)]
                live.update(Panel(
                    f"{frame} {message}",
                    title="AI Security Expert Thinking",
                    style="blue"
                ))
                await asyncio.sleep(0.25)
    
    async def process_command(self, user_input: str) -> str:
        """Process user commands and return responses"""
        command = user_input.strip().lower()
        
        # Handle special commands
        if command in ['exit', 'quit', 'q']:
            return "EXIT"
        elif command in ['help', 'h', '?']:
            self.display_help()
            return ""
        elif command == 'status':
            return self._get_status()
        elif command == 'thoughts':
            return self._get_thoughts()
        elif command == 'clear':
            self.console.clear()
            return ""
        
        # For all other input, use the conversational AI
        return await self.conversational_ai.discuss(user_input)
    
    def _get_status(self) -> str:
        """Get current system status"""
        status_info = {
            "AI Expert": "Online ✅",
            "Conversation Mode": self.conversational_ai.context.mode.value,
            "Current Target": self.conversational_ai.context.current_target or "None",
            "Analysis Complete": "Yes ✅" if self.conversational_ai.context.current_analysis else "No ❌",
            "Thoughts Tracked": len(self.security_expert.thought_chain),
            "Operation History": len(self.security_expert.operation_history)
        }
        
        status_table = Table(title="🔍 Archangel Status")
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", style="green")
        
        for component, status in status_info.items():
            status_table.add_row(component, status)
        
        self.console.print(status_table)
        return ""
    
    def _get_thoughts(self) -> str:
        """Display current AI thought process"""
        thoughts = self.security_expert.get_current_thoughts()
        
        if not thoughts:
            return "🤔 I haven't started thinking about anything yet. Give me a target to analyze!"
        
        thoughts_text = "🧠 **Current AI Thought Process:**\n\n"
        
        for i, thought in enumerate(thoughts, 1):
            thoughts_text += f"**Step {i}: {thought.step.replace('_', ' ').title()}**\n"
            thoughts_text += f"Confidence: {thought.confidence:.1%}\n"
            thoughts_text += f"Time: {time.ctime(thought.timestamp)}\n"
            thoughts_text += f"Alternatives: {', '.join(thought.alternatives_considered)}\n\n"
        
        return thoughts_text
    
    async def run(self):
        """Main CLI loop"""
        self.running = True
        
        # Display welcome
        self.display_banner()
        self.display_intro()
        
        self.console.print("\n💬 Type 'help' for commands or just start talking to me!\n")
        
        while self.running:
            try:
                # Get user input
                user_input = Prompt.ask(
                    "[bold cyan]Archangel[/bold cyan]",
                    default="help"
                ).strip()
                
                if not user_input:
                    continue
                
                # Show thinking animation for analysis requests
                if any(keyword in user_input.lower() for keyword in ['analyze', 'test', 'scan', 'pentest']):
                    await self.display_thinking_animation("Analyzing target and formulating strategy...")
                
                # Process the command
                response = await self.process_command(user_input)
                
                # Handle exit
                if response == "EXIT":
                    self.console.print("\n👋 Thanks for using Archangel! Stay secure! 🛡️\n")
                    break
                
                # Display response if not empty
                if response:
                    self.console.print(Panel(
                        response,
                        title="🤖 Archangel AI",
                        style="white"
                    ))
                
                # Add spacing
                self.console.print()
                
            except KeyboardInterrupt:
                self.console.print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except Exception as e:
                self.console.print(f"\n🚨 Error: {str(e)}")
                self.console.print("Please try again or type 'help' for assistance.\n")
    
    def demo_mode(self):
        """Run in demo mode with scripted interactions"""
        self.console.print(Panel(
            "🎬 **Demo Mode Activated**\n\nThis will demonstrate Archangel's AI reasoning capabilities.",
            title="Demo Mode",
            style="yellow"
        ))
        
        # Simulate demo interactions
        demo_commands = [
            "analyze example.com",
            "explain",
            "teach sql injection"
        ]
        
        for command in demo_commands:
            self.console.print(f"\n[bold cyan]Demo Command:[/bold cyan] {command}")
            self.console.print("Press Enter to continue...")
            input()
            
            # Process command
            response = asyncio.run(self.process_command(command))
            if response:
                self.console.print(Panel(
                    response,
                    title="🤖 Archangel AI Demo",
                    style="green"
                ))


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Archangel AI Security Expert")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")
    parser.add_argument("--version", action="version", version="Archangel 0.1.0")
    
    args = parser.parse_args()
    
    # Create and run CLI
    cli = ArchangelCLI()
    
    if args.demo:
        cli.demo_mode()
    else:
        asyncio.run(cli.run())


if __name__ == "__main__":
    main()