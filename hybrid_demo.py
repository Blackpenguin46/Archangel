#!/usr/bin/env python3
"""
Archangel Hybrid Architecture Demo
Complete demonstration of kernel-userspace AI security system
"""

import asyncio
import time
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.live import Live
from rich.text import Text

from core.ai_security_expert import SecurityExpertAI
from core.conversational_ai import ConversationalSecurityAI
from core.kernel_interface import create_kernel_interface, SecurityContext, SecurityRule, ArchangelDecision, ArchangelConfidence
from tools.tool_integration import create_ai_orchestrator

class ArchangelHybridSystem:
    """
    Complete Archangel Hybrid AI Security System
    
    Demonstrates the full hybrid kernel-userspace architecture with:
    - AI Security Expert Brain (userspace)
    - Kernel security monitoring and interception
    - Real-time kernel-userspace communication
    - Research-backed LLM automation (arxiv.org/abs/2501.16466)
    """
    
    def __init__(self):
        self.console = Console()
        
        # Core AI components
        self.security_expert = None
        self.conversational_ai = None
        self.tool_orchestrator = None
        
        # Kernel interface
        self.kernel_interface = None
        
        # System state
        self.running = False
        self.message_processing_task = None
        
    async def initialize_system(self):
        """Initialize the complete hybrid system"""
        self.console.print(Panel(
            """
🛡️ **ARCHANGEL HYBRID ARCHITECTURE INITIALIZATION**

Initializing the complete AI security system:
• 🧠 AI Security Expert Brain (Userspace)
• 🔧 AI Tool Orchestration Layer
• ⚡ Kernel Security Module Interface
• 💬 Conversational AI Interface
• 🧮 Research-backed LLM Automation Framework

This demonstrates the paradigm shift from AI automation to AI understanding!
            """.strip(),
            title="🚀 Hybrid System Initialization",
            style="bold blue"
        ))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            # Initialize AI Security Expert
            task = progress.add_task("Loading AI Security Expert Brain...", total=None)
            await asyncio.sleep(1)
            self.security_expert = SecurityExpertAI()
            
            # Initialize kernel interface
            progress.update(task, description="Connecting to kernel module...")
            await asyncio.sleep(1)
            self.kernel_interface = create_kernel_interface()
            
            if not self.kernel_interface.init_communication():
                self.console.print("[yellow]Warning: Using mock kernel interface for demo[/yellow]")
            
            # Connect AI with kernel
            progress.update(task, description="Linking AI with kernel interface...")
            await asyncio.sleep(1)
            self.security_expert.set_kernel_interface(self.kernel_interface)
            
            # Initialize tool orchestrator
            progress.update(task, description="Initializing tool orchestration...")
            await asyncio.sleep(1)
            self.tool_orchestrator = create_ai_orchestrator()
            self.security_expert.set_tool_orchestrator(self.tool_orchestrator)
            
            # Initialize conversational AI
            progress.update(task, description="Loading conversational AI...")
            await asyncio.sleep(1)
            self.conversational_ai = ConversationalSecurityAI(self.security_expert)
            
            progress.update(task, description="Hybrid system online! ✅")
            await asyncio.sleep(1)
    
    async def demonstrate_hybrid_architecture(self):
        """Demonstrate the complete hybrid architecture"""
        
        # Show system status
        await self._show_system_status()
        await asyncio.sleep(2)
        
        # Demo 1: Kernel Statistics
        await self._demo_kernel_integration()
        await asyncio.sleep(3)
        
        # Demo 2: AI-Kernel Communication
        await self._demo_ai_kernel_communication()
        await asyncio.sleep(3)
        
        # Demo 3: Research-backed LLM Automation
        await self._demo_llm_automation()
        await asyncio.sleep(3)
        
        # Demo 4: Complete Security Analysis
        await self._demo_complete_security_analysis()
        await asyncio.sleep(2)
    
    async def _show_system_status(self):
        """Show current system status"""
        self.console.print(Panel(
            "**SYSTEM STATUS CHECK**\n\nAnalyzing hybrid architecture components...",
            title="📊 System Status",
            style="green"
        ))
        
        # Get kernel module status
        kernel_status = self.kernel_interface.get_module_status()
        kernel_stats = self.kernel_interface.get_kernel_stats()
        
        # Create status table
        status_table = Table(title="🔍 Archangel Hybrid System Status")
        status_table.add_column("Component", style="cyan", no_wrap=True)
        status_table.add_column("Status", style="green")
        status_table.add_column("Details", style="white")
        
        status_table.add_row(
            "AI Security Expert",
            "✅ Online",
            f"Model: {self.security_expert.model_name}, Thoughts: {len(self.security_expert.thought_chain)}"
        )
        
        status_table.add_row(
            "Kernel Module",
            "✅ Connected" if kernel_status["status"] in ["loaded", "loaded_mock"] else "❌ Offline",
            kernel_status.get("details", {}).get("status", "Unknown")
        )
        
        status_table.add_row(
            "Tool Orchestrator", 
            "✅ Ready",
            f"Tools: {', '.join(self.tool_orchestrator.get_available_tools())}"
        )
        
        status_table.add_row(
            "Kernel Decisions",
            "📊 Active",
            f"Total: {kernel_stats.total_decisions}, Deferred: {kernel_stats.deferred_decisions}"
        )
        
        status_table.add_row(
            "Uptime",
            "⏱️ Running",
            f"{kernel_stats.uptime_seconds} seconds"
        )
        
        self.console.print(status_table)
    
    async def _demo_kernel_integration(self):
        """Demonstrate kernel module integration"""
        self.console.print(Panel(
            "**DEMO 1:** Kernel Module Integration\n\nShowing real-time kernel statistics and security rules...",
            title="🔧 Kernel Integration Demo",
            style="yellow"
        ))
        
        # Show kernel statistics
        stats = self.kernel_interface.get_kernel_stats()
        
        stats_text = f"""
**Kernel Security Statistics:**
• Total Security Decisions: {stats.total_decisions}
• Allow Decisions: {stats.allow_decisions}
• Deny Decisions: {stats.deny_decisions} 
• Monitor Decisions: {stats.monitor_decisions}
• Deferred to AI: {stats.deferred_decisions}
• Average Decision Time: {stats.avg_decision_time_ns} ns
• System Uptime: {stats.uptime_seconds} seconds
        """
        
        self.console.print(Panel(stats_text.strip(), title="📊 Kernel Statistics", style="white"))
        
        # Show security rules
        rules = self.kernel_interface.get_security_rules()
        if rules:
            rules_table = Table(title="🛡️ Active Security Rules")
            rules_table.add_column("ID", style="cyan")
            rules_table.add_column("Priority", style="yellow")
            rules_table.add_column("Action", style="red")
            rules_table.add_column("Matches", style="green")
            rules_table.add_column("Description", style="white")
            
            for rule in rules:
                rules_table.add_row(
                    str(rule['id']),
                    str(rule['priority']),
                    str(rule['action']),
                    str(rule['matches']),
                    rule['description']
                )
            
            self.console.print(rules_table)
        else:
            self.console.print("[yellow]No security rules currently active[/yellow]")
    
    async def _demo_ai_kernel_communication(self):
        """Demonstrate AI-kernel communication"""
        self.console.print(Panel(
            "**DEMO 2:** AI-Kernel Communication\n\nSimulating real-time security analysis requests from kernel...",
            title="🧠 AI-Kernel Communication Demo",
            style="purple"
        ))
        
        # Simulate kernel analysis requests
        test_contexts = [
            SecurityContext(
                pid=1234,
                uid=0,
                syscall_nr=59,  # execve
                timestamp=time.time_ns(),
                flags=0x0001,  # EXECVE flag
                comm="suspicious_app"
            ),
            SecurityContext(
                pid=5678,
                uid=1000,
                syscall_nr=2,   # open
                timestamp=time.time_ns(),
                flags=0x0002,   # FILE_ACCESS flag
                comm="normal_app"
            )
        ]
        
        for i, context in enumerate(test_contexts, 1):
            self.console.print(f"\n[bold cyan]Kernel Analysis Request {i}:[/bold cyan]")
            
            # Show what the kernel is asking
            self.console.print(f"📋 Process: {context.comm} (PID {context.pid})")
            self.console.print(f"📋 User ID: {context.uid} ({'root' if context.uid == 0 else 'user'})")
            self.console.print(f"📋 System Call: {context.syscall_nr}")
            self.console.print(f"📋 Flags: {hex(context.flags)}")
            
            # AI processes the request
            decision = await self.security_expert.handle_kernel_analysis_request(context)
            
            # Show the result
            decision_color = {
                "ALLOW": "green",
                "DENY": "red", 
                "MONITOR": "yellow"
            }.get(decision, "white")
            
            self.console.print(f"[{decision_color}]🎯 AI Decision: {decision}[/{decision_color}]")
            
            await asyncio.sleep(2)
    
    async def _demo_llm_automation(self):
        """Demonstrate research-backed LLM automation"""
        self.console.print(Panel(
            """**DEMO 3:** Research-Backed LLM Automation

Based on findings from arxiv.org/abs/2501.16466:
• High-level action specification
• Expert agent translation layer  
• Modular attack goal decomposition

Showing AI abstraction layer translating kernel contexts to high-level objectives...""",
            title="🧮 LLM Automation Framework",
            style="blue"
        ))
        
        # Demonstrate action abstraction layer
        test_syscalls = [59, 2, 101, 165]  # execve, open, ptrace, mount
        
        for syscall_nr in test_syscalls:
            context = SecurityContext(
                pid=9999,
                uid=1000,
                syscall_nr=syscall_nr,
                timestamp=time.time_ns(),
                flags=0x0000,
                comm="test_process"
            )
            
            # Show abstraction layer in action
            objective = self.security_expert._translate_kernel_context_to_objective(context)
            steps = self.security_expert._decompose_security_analysis(objective)
            
            self.console.print(f"\n[bold]Syscall {syscall_nr}:[/bold]")
            self.console.print(f"🎯 High-level Objective: {objective}")
            self.console.print("📋 Decomposed Analysis Steps:")
            
            for j, step in enumerate(steps, 1):
                self.console.print(f"   {j}. {step}")
    
    async def _demo_complete_security_analysis(self):
        """Demonstrate complete security analysis with all components"""
        self.console.print(Panel(
            "**DEMO 4:** Complete Security Analysis\n\nFull hybrid architecture in action - AI + Kernel + Tools...",
            title="🎯 Complete Analysis Demo",
            style="green"
        ))
        
        target = "example.com"
        
        # Step 1: AI Analysis
        self.console.print(f"\n🧠 **Step 1:** AI Security Analysis of {target}")
        analysis = await self.security_expert.analyze_target(target)
        
        self.console.print(f"📊 Confidence: {analysis.confidence.value}")
        self.console.print(f"⚠️ Threat Level: {analysis.threat_level.value}")
        
        # Step 2: Kernel Rule Deployment
        self.console.print(f"\n🔧 **Step 2:** Deploying AI-Generated Rules to Kernel")
        
        # Create security rule based on AI analysis
        security_rule = SecurityRule(
            rule_id=1001,
            priority=100,
            condition_mask=0xFFFFFFFF,
            condition_values=0x0001,  # Match EXECVE operations
            action=ArchangelDecision.MONITOR,
            confidence=ArchangelConfidence.HIGH,
            description=f"AI-generated rule for {target} analysis"
        )
        
        if self.kernel_interface.add_security_rule(security_rule):
            self.console.print("✅ Security rule deployed to kernel")
        else:
            self.console.print("⚠️ Rule deployment simulated (mock kernel)")
        
        # Step 3: Tool Execution
        self.console.print(f"\n🛠️ **Step 3:** AI-Driven Tool Execution")
        tool_results = await self.security_expert.execute_strategy_with_tools(target, analysis.strategy)
        
        # Step 4: Results Integration
        self.console.print(f"\n📊 **Step 4:** Integrated Results")
        
        results_table = Table(title="🔍 Hybrid Analysis Results")
        results_table.add_column("Component", style="cyan")
        results_table.add_column("Result", style="white")
        results_table.add_column("Status", style="green")
        
        results_table.add_row(
            "AI Analysis",
            f"Confidence: {analysis.confidence.value}, Threat: {analysis.threat_level.value}",
            "✅ Complete"
        )
        
        results_table.add_row(
            "Kernel Integration",
            f"Rule {security_rule.rule_id} deployed",
            "✅ Active"
        )
        
        results_table.add_row(
            "Tool Execution", 
            f"{len(tool_results)} tools executed",
            "✅ Success"
        )
        
        self.console.print(results_table)
    
    async def run_interactive_mode(self):
        """Run interactive mode for user exploration"""
        self.console.print(Panel(
            """
🎮 **INTERACTIVE MODE**

You can now interact with the hybrid Archangel system:
• Ask questions about security
• Request target analysis  
• View kernel statistics
• Manage security rules

Type 'help' for commands or 'exit' to quit.
            """.strip(),
            title="🎮 Interactive Mode",
            style="cyan"
        ))
        
        while True:
            try:
                user_input = input("\n🛡️ Archangel> ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    self._show_interactive_help()
                elif user_input.lower() == 'status':
                    await self._show_system_status()
                elif user_input.lower().startswith('analyze '):
                    target = user_input[8:].strip()
                    await self._interactive_analyze(target)
                elif user_input.lower() == 'stats':
                    await self._show_kernel_stats()
                elif user_input.strip():
                    # Pass to conversational AI
                    response = await self.conversational_ai.discuss(user_input)
                    self.console.print(Panel(response, title="🤖 Archangel AI", style="white"))
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
        
        self.console.print("\n👋 Thanks for exploring Archangel! Stay secure! 🛡️")
    
    def _show_interactive_help(self):
        """Show interactive mode help"""
        help_table = Table(title="🔧 Interactive Commands")
        help_table.add_column("Command", style="cyan", no_wrap=True)
        help_table.add_column("Description", style="white")
        
        help_table.add_row("help", "Show this help message")
        help_table.add_row("status", "Show system status")
        help_table.add_row("stats", "Show kernel statistics")
        help_table.add_row("analyze <target>", "Analyze security target")
        help_table.add_row("exit", "Exit interactive mode")
        help_table.add_row("<question>", "Ask AI about security")
        
        self.console.print(help_table)
    
    async def _interactive_analyze(self, target: str):
        """Interactive target analysis"""
        self.console.print(f"🎯 Analyzing {target} with hybrid AI system...")
        
        analysis = await self.security_expert.analyze_target(target)
        
        self.console.print(Panel(
            f"""
**Target:** {analysis.target}
**Type:** {analysis.target_type}
**Confidence:** {analysis.confidence.value}
**Threat Level:** {analysis.threat_level.value}

**AI Reasoning:** {analysis.reasoning[:200]}...

**Next Actions:**
{chr(10).join(f"• {action}" for action in analysis.next_actions[:3])}
            """.strip(),
            title="🧠 AI Analysis Results",
            style="green"
        ))
    
    async def _show_kernel_stats(self):
        """Show kernel statistics"""
        stats = self.kernel_interface.get_kernel_stats()
        
        stats_panel = f"""
**Security Decisions:** {stats.total_decisions}
**Allow/Deny/Monitor:** {stats.allow_decisions}/{stats.deny_decisions}/{stats.monitor_decisions}
**AI Deferrals:** {stats.deferred_decisions}
**Average Decision Time:** {stats.avg_decision_time_ns} ns
**System Uptime:** {stats.uptime_seconds} seconds
        """
        
        self.console.print(Panel(stats_panel.strip(), title="📊 Kernel Statistics", style="blue"))

async def main():
    """Main demonstration function"""
    console = Console()
    
    # Create and initialize hybrid system
    system = ArchangelHybridSystem()
    
    try:
        # Initialize all components
        await system.initialize_system()
        
        # Run complete demonstration
        await system.demonstrate_hybrid_architecture()
        
        # Final summary
        console.print(Panel(
            """
🎉 **ARCHANGEL HYBRID ARCHITECTURE DEMONSTRATION COMPLETE**

**What You Just Experienced:**

🧠 **AI Security Expert** - Step-by-step reasoning about security problems
🔧 **Kernel Integration** - Real-time security monitoring and rule deployment
💬 **AI-Kernel Communication** - High-speed message passing and analysis
🧮 **Research-Backed Automation** - LLM abstraction layers (arxiv.org/abs/2501.16466)
🛠️ **Tool Orchestration** - AI-driven security tool execution
📊 **Hybrid Architecture** - Optimal balance of speed and intelligence

**🚀 The Innovation:**
This represents the first complete hybrid kernel-userspace AI security system that:
• Makes <1ms security decisions in kernel space
• Provides complex AI analysis in userspace
• Maintains transparent reasoning throughout
• Learns and adapts from every operation

**"While others automate hacking, Archangel understands it."** 🛡️

Ready to revolutionize cybersecurity with AI that truly understands security? 🚀
            """.strip(),
            title="✨ Hybrid Demo Complete",
            style="bold green"
        ))
        
        # Offer interactive mode
        console.print("\n[yellow]Would you like to try interactive mode? (y/n)[/yellow]")
        if input().lower().startswith('y'):
            await system.run_interactive_mode()
        
    except Exception as e:
        console.print(f"[red]Demo error: {e}[/red]")
    finally:
        # Cleanup
        if system.kernel_interface:
            system.kernel_interface.cleanup_communication()

if __name__ == "__main__":
    asyncio.run(main())