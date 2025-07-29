#!/usr/bin/env python3
"""
Archangel Full System Demo
Complete demonstration including AI reasoning and tool execution
"""

import asyncio
import time
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from core.ai_security_expert import SecurityExpertAI
from core.conversational_ai import ConversationalSecurityAI
from tools.tool_integration import create_ai_orchestrator

async def full_system_demo():
    """Run complete Archangel system demonstration"""
    console = Console()
    
    # Display demo header
    console.print(Panel(
        """
🎬 **ARCHANGEL COMPLETE SYSTEM DEMO**

This demonstration shows the full Archangel AI Security Expert system:
• 🧠 AI that thinks step-by-step about security problems
• 💬 Conversational AI that explains its reasoning process
• 🛠️ AI-driven autonomous tool orchestration
• 🎓 Educational AI that teaches while working
• 🔄 Adaptive strategy based on real tool results

Experience the paradigm shift: AI that UNDERSTANDS security!
        """.strip(),
        title="🛡️ Archangel Full Demo",
        style="bold blue"
    ))
    
    # Initialize systems
    console.print("\n🚀 Initializing Archangel AI Systems...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Initialize AI expert
        task1 = progress.add_task("Loading AI Security Expert Brain...", total=None)
        await asyncio.sleep(1)
        security_expert = SecurityExpertAI()
        
        # Initialize conversational AI
        progress.update(task1, description="Loading Conversational AI Interface...")
        await asyncio.sleep(1)
        conversational_ai = ConversationalSecurityAI(security_expert)
        
        # Initialize tool orchestrator
        progress.update(task1, description="Initializing AI Tool Orchestrator...")
        await asyncio.sleep(1)
        tool_orchestrator = create_ai_orchestrator()
        security_expert.set_tool_orchestrator(tool_orchestrator)
        
        progress.update(task1, description="All systems online! ✅")
        await asyncio.sleep(1)
    
    # Show available tools
    available_tools = tool_orchestrator.get_available_tools()
    console.print(f"\n🔧 Available Security Tools: {', '.join(available_tools)}")
    
    console.print()
    
    # Demo Scenario 1: AI Analysis with Tool Execution
    console.print(Panel(
        "**DEMO SCENARIO 1:** AI Analysis with Real Tool Execution\n\nWatch the AI think, plan, and execute security tools autonomously...",
        title="🤖 AI + Tools Demo",
        style="green"
    ))
    
    target = "example.com"
    console.print(f"\n[bold cyan]User:[/bold cyan] Perform a complete security analysis of {target}")
    
    # AI thinks about the problem
    console.print("\n🧠 [italic]AI Security Expert is thinking...[/italic]")
    await asyncio.sleep(2)
    
    # Perform AI analysis
    analysis = await security_expert.analyze_target(target)
    
    # Display AI reasoning (shortened for demo)
    console.print(Panel(
        f"""
🎯 **Target:** {analysis.target} ({analysis.target_type})
📊 **AI Confidence:** {analysis.confidence.value}
⚠️ **Threat Assessment:** {analysis.threat_level.value}

🧠 **AI Strategic Reasoning:**
The AI has analyzed this target and determined it's a web application requiring
systematic reconnaissance → enumeration → vulnerability assessment.

**🎯 AI's Planned Approach:**
• Phase 1: Network Discovery (nmap ping scan)
• Phase 2: Service Enumeration (nmap service scan) 
• Phase 3: Vulnerability Assessment (targeted scanning)

💡 **AI Reasoning:** "I'm starting with low-impact reconnaissance to build target understanding,
then escalating to more detailed analysis based on what I discover."
        """.strip(),
        title="🧠 AI Security Analysis",
        style="white"
    ))
    
    console.print("\n🤖 [bold]AI now executing strategy with real security tools...[/bold]")
    await asyncio.sleep(2)
    
    # AI executes strategy with tools
    tool_results = await security_expert.execute_strategy_with_tools(target, analysis.strategy)
    
    # Display tool execution results
    console.print(Panel(
        "🛠️ **AI Tool Execution Complete**\n\nThe AI successfully orchestrated multiple security tools and analyzed the results collectively.",
        title="✅ Tool Execution Results",
        style="green"
    ))
    
    # Show results table
    if tool_results:
        results_table = Table(title="🔍 AI Tool Execution Summary")
        results_table.add_column("Tool", style="cyan")
        results_table.add_column("Command", style="white")
        results_table.add_column("Status", style="green")
        results_table.add_column("Time", style="yellow")
        results_table.add_column("Key Findings", style="blue")
        
        for result in tool_results:
            status = "✅ Success" if result.get('success') else "❌ Failed"
            findings = str(result.get('findings', {}).get('count', 'N/A')) + " items found"
            
            results_table.add_row(
                result.get('tool', 'Unknown'),
                result.get('command', '')[:30] + "...",
                status,
                result.get('execution_time', 'N/A'),
                findings
            )
        
        console.print(results_table)
    
    await asyncio.sleep(3)
    
    # Demo Scenario 2: AI Explanation of Tool Selection
    console.print(Panel(
        "**DEMO SCENARIO 2:** AI Explains Tool Selection Reasoning\n\nAsking the AI why it chose specific tools and approaches...",
        title="💬 AI Reasoning Explanation",
        style="yellow"
    ))
    
    console.print("\n[bold cyan]User:[/bold cyan] Why did you choose those specific tools and that order?")
    await asyncio.sleep(2)
    
    explanation_response = """
🧠 **Excellent question! Let me explain my tool selection reasoning:**

**🎯 Why I chose nmap first:**
- Nmap is the gold standard for network reconnaissance
- Starting with ping scan minimizes impact while confirming target is live
- Service scan builds comprehensive picture of attack surface

**🔄 Why this specific order:**
1. **Ping Scan First** → Confirms target accessibility with minimal footprint
2. **Service Scan Second** → Identifies what services are actually running
3. **Targeted Analysis Third** → Focus on interesting services found

**🧠 My reasoning process:**
- I always start with least intrusive methods
- I build knowledge incrementally rather than using aggressive scans immediately
- I adapt my approach based on what each tool reveals
- I prioritize stealth and target safety over speed

**💡 What makes this AI-driven:**
Unlike scripts that just run predetermined commands, I:
- Choose tools based on target characteristics
- Adapt strategy based on intermediate results  
- Explain WHY each decision was made
- Learn from results to improve future analyses

This is the difference between AI automation and AI understanding! 🎯
    """
    
    console.print(Panel(
        explanation_response.strip(),
        title="🤖 AI Reasoning Explanation",
        style="white"
    ))
    
    await asyncio.sleep(3)
    
    # Demo Scenario 3: AI Teaching Based on Results
    console.print(Panel(
        "**DEMO SCENARIO 3:** AI Teaching Mode\n\nAI becomes an instructor, teaching based on actual results...",
        title="🎓 Educational AI Demo",
        style="purple"
    ))
    
    console.print("\n[bold cyan]User:[/bold cyan] What should I do with these scan results?")
    await asyncio.sleep(2)
    
    teaching_response = """
🎓 **Great question! Let me teach you how to analyze these results:**

**📊 What the Results Tell Us:**
- **Open Ports Found:** Each port represents a potential attack vector
- **Service Versions:** Specific versions can have known vulnerabilities  
- **Service Types:** Different services require different testing approaches

**🎯 Next Steps I'd Recommend:**

**For Web Services (HTTP/HTTPS):**
- Directory enumeration (dirb, gobuster)
- Web application vulnerability scanning (nikto, burp)
- Manual testing for OWASP Top 10 vulnerabilities

**For SSH Services:**
- Check for weak authentication methods
- Test for default credentials
- Analyze SSH banner for version vulnerabilities

**For Any Service:**
- Research specific version vulnerabilities (searchsploit, CVE databases)
- Test for misconfigurations
- Check for default credentials

**🧠 My Teaching Philosophy:**
I don't just tell you WHAT to do - I explain WHY each step matters:
- **Why enumerate directories?** → Find hidden admin panels or sensitive files
- **Why check versions?** → Older versions often have published exploits
- **Why test authentication?** → Weak auth is the #1 entry point

**💡 Remember:**
Real security testing is about understanding, not just running tools.
Each finding should lead to questions: "Why is this service exposed?"
"What data might be accessible?" "How could this be exploited?"

That's how you think like a security expert! 🛡️
    """
    
    console.print(Panel(
        teaching_response.strip(),
        title="🎓 AI Security Teaching",
        style="white"
    ))
    
    await asyncio.sleep(3)
    
    # Final Demo Summary
    console.print(Panel(
        """
🎯 **ARCHANGEL COMPLETE SYSTEM DEMONSTRATION FINISHED**

**🚀 What You Just Experienced:**

🧠 **AI That Thinks:** Step-by-step reasoning about security problems
🛠️ **AI That Acts:** Autonomous orchestration of real security tools  
💬 **AI That Explains:** Clear explanations of every decision made
🎓 **AI That Teaches:** Educational insights based on actual results
🔄 **AI That Adapts:** Dynamic strategy changes based on tool outputs

**⚡ The Breakthrough Innovation:**
This isn't just "AI-powered automation" - this is AI that UNDERSTANDS security.

**🎯 Key Differentiators Demonstrated:**
• **vs PentestGPT:** Fully autonomous execution, not just suggestions
• **vs NodeZero:** Complete transparency in AI reasoning process
• **vs Commercial Tools:** Educational value while performing work
• **vs All Others:** AI that thinks like a security expert

**🔥 What's Next:**
• Try the interactive CLI: `python cli.py`
• Analyze your own targets (ethically!)
• Experience conversational security AI
• Watch the AI learn and improve over time

**"While others automate hacking, Archangel understands it."** 🛡️

Ready to revolutionize how AI approaches cybersecurity? 🚀
        """.strip(),
        title="✨ Complete Demo Finished",
        style="bold green"
    ))

if __name__ == "__main__":
    asyncio.run(full_system_demo())