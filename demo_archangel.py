#!/usr/bin/env python3
"""
Archangel Demo Script
Quick demonstration of AI security expert capabilities
"""

import asyncio
import time
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from core.ai_security_expert import SecurityExpertAI
from core.conversational_ai import ConversationalSecurityAI

async def demo_archangel():
    """Run Archangel demonstration"""
    console = Console()
    
    # Display demo header
    console.print(Panel(
        """
🎬 **ARCHANGEL AI SECURITY EXPERT DEMO**

This demonstration shows how Archangel thinks like a security expert:
• Transparent reasoning process
• Educational explanations
• Adaptive strategy planning
• Conversational AI interface

Ready to see AI that understands security, not just automates it?
        """.strip(),
        title="🛡️ Archangel Demo",
        style="bold blue"
    ))
    
    # Initialize AI systems
    console.print("\n🧠 Initializing AI Security Expert Brain...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Loading AI systems...", total=None)
        await asyncio.sleep(2)  # Simulate loading
        
        security_expert = SecurityExpertAI()
        conversational_ai = ConversationalSecurityAI(security_expert)
        
        progress.update(task, description="AI systems online! ✅")
        await asyncio.sleep(1)
    
    console.print()
    
    # Demo Scenario 1: AI Security Reasoning
    console.print(Panel(
        "**DEMO SCENARIO 1:** AI Security Reasoning\n\nWatching the AI think step-by-step about a security target...",
        title="🧠 AI Reasoning Demo",
        style="green"
    ))
    
    console.print("\n[bold cyan]User:[/bold cyan] analyze example.com")
    
    # Show thinking process
    console.print("\n🧠 [italic]AI is thinking...[/italic]")
    await asyncio.sleep(2)
    
    # Perform actual analysis
    analysis = await security_expert.analyze_target("example.com")
    
    # Display AI reasoning
    console.print(Panel(
        f"""
🎯 **Target:** {analysis.target}
📊 **Confidence:** {analysis.confidence.value}
⚠️ **Threat Level:** {analysis.threat_level.value}

🧠 **AI Reasoning Process:**
{analysis.reasoning[:500]}...

**🎯 AI's Strategic Plan:**
1. {analysis.next_actions[0] if analysis.next_actions else 'Web reconnaissance'}
2. {analysis.next_actions[1] if len(analysis.next_actions) > 1 else 'Service enumeration'}
3. {analysis.next_actions[2] if len(analysis.next_actions) > 2 else 'Vulnerability assessment'}

💡 **Key Insight:** The AI doesn't just run tools - it thinks about WHY and HOW to use them.
        """.strip(),
        title="🤖 Archangel AI Analysis",
        style="white"
    ))
    
    await asyncio.sleep(3)
    
    # Demo Scenario 2: AI Explanation
    console.print(Panel(
        "**DEMO SCENARIO 2:** AI Explanation\n\nAsking the AI to explain its reasoning...",
        title="💬 Conversational AI Demo", 
        style="yellow"
    ))
    
    console.print("\n[bold cyan]User:[/bold cyan] why did you choose that approach?")
    
    await asyncio.sleep(2)
    
    explanation = await conversational_ai.discuss("why did you choose that approach?")
    
    console.print(Panel(
        explanation[:800] + "...\n\n💡 **Key Feature:** Transparent AI reasoning - you always know WHY the AI made each decision.",
        title="🤖 Archangel Explanation",
        style="white"
    ))
    
    await asyncio.sleep(3)
    
    # Demo Scenario 3: AI Teaching
    console.print(Panel(
        "**DEMO SCENARIO 3:** AI Teaching Mode\n\nAI becomes a security instructor...",
        title="🎓 Educational AI Demo",
        style="purple"
    ))
    
    console.print("\n[bold cyan]User:[/bold cyan] teach me about SQL injection")
    
    await asyncio.sleep(2)
    
    teaching = await conversational_ai.discuss("teach me about SQL injection")
    
    console.print(Panel(
        teaching[:800] + "...\n\n💡 **Key Feature:** AI that teaches while it works - educational value built-in.",
        title="🎓 Archangel Teaching",
        style="white"
    ))
    
    await asyncio.sleep(3)
    
    # Demo Summary
    console.print(Panel(
        """
🎯 **ARCHANGEL DEMONSTRATION COMPLETE**

**What you just saw:**

🧠 **AI That Thinks:** Step-by-step reasoning about security problems
💬 **AI That Explains:** Clear explanations of decision-making process  
🎓 **AI That Teaches:** Educational value while performing security work
🔄 **AI That Adapts:** Dynamic strategy changes based on discoveries

**🚀 The Innovation:**
While other tools just execute commands, Archangel UNDERSTANDS security.

**🎯 Next Steps:**
• Try the full interactive CLI: `python cli.py`
• Explore the AI reasoning with real targets
• Experience conversational security AI

"While others automate hacking, Archangel understands it." 🛡️
        """.strip(),
        title="✨ Demo Complete",
        style="bold green"
    ))

if __name__ == "__main__":
    asyncio.run(demo_archangel())