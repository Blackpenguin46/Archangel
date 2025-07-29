"""
Archangel Conversational AI Interface
AI that can have intelligent security discussions
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import re

from .ai_security_expert import SecurityExpertAI, SecurityAnalysis

class ConversationMode(Enum):
    ANALYSIS = "analysis"
    EXPLANATION = "explanation" 
    TEACHING = "teaching"
    PLANNING = "planning"
    CASUAL = "casual"

@dataclass
class ConversationContext:
    """Tracks the current conversation state"""
    mode: ConversationMode
    current_target: Optional[str]
    current_analysis: Optional[SecurityAnalysis]
    conversation_history: List[Dict[str, str]]
    user_expertise_level: str = "intermediate"

class ConversationalSecurityAI:
    """
    AI that discusses security intelligently
    
    This enables natural language conversations about security topics,
    explanations of AI reasoning, and educational interactions.
    """
    
    def __init__(self, security_expert: SecurityExpertAI):
        self.security_expert = security_expert
        self.context = ConversationContext(
            mode=ConversationMode.CASUAL,
            current_target=None,
            current_analysis=None,
            conversation_history=[]
        )
        self.personality_traits = {
            "expertise_level": "senior_security_consultant",
            "communication_style": "educational_and_thorough", 
            "confidence_expression": "measured_and_honest",
            "teaching_approach": "socratic_method"
        }
    
    async def discuss(self, user_message: str) -> str:
        """
        Main conversation interface
        Handles different types of security discussions
        """
        # Add to conversation history
        self.context.conversation_history.append({
            "role": "user",
            "message": user_message,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Determine conversation intent
        intent = self._analyze_intent(user_message)
        
        # Generate appropriate response
        if intent == "analyze_target":
            response = await self._handle_analysis_request(user_message)
        elif intent == "explain_reasoning":
            response = await self._handle_explanation_request(user_message)
        elif intent == "teach_concept":
            response = await self._handle_teaching_request(user_message)
        elif intent == "plan_strategy":
            response = await self._handle_planning_request(user_message)
        elif intent == "ask_question":
            response = await self._handle_question(user_message)
        else:
            response = await self._handle_general_discussion(user_message)
        
        # Add response to history
        self.context.conversation_history.append({
            "role": "assistant", 
            "message": response,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        return response
    
    def _analyze_intent(self, message: str) -> str:
        """Analyze what the user wants to discuss"""
        message_lower = message.lower()
        
        # Check for analysis requests
        analysis_keywords = ["analyze", "test", "scan", "pentest", "assess", "check security"]
        if any(keyword in message_lower for keyword in analysis_keywords):
            return "analyze_target"
        
        # Check for explanation requests
        explanation_keywords = ["why", "how", "explain", "reasoning", "because", "what happened"]
        if any(keyword in message_lower for keyword in explanation_keywords):
            return "explain_reasoning"
        
        # Check for teaching requests
        teaching_keywords = ["teach", "learn", "what is", "how does", "show me", "tutorial"]
        if any(keyword in message_lower for keyword in teaching_keywords):
            return "teach_concept"
        
        # Check for planning requests
        planning_keywords = ["plan", "strategy", "approach", "methodology", "steps"]
        if any(keyword in message_lower for keyword in planning_keywords):
            return "plan_strategy"
        
        # Check for questions
        question_keywords = ["?", "what if", "can you", "would you", "should i"]
        if any(keyword in message_lower for keyword in question_keywords):
            return "ask_question"
        
        return "general_discussion"
    
    async def _handle_analysis_request(self, message: str) -> str:
        """Handle requests to analyze a target"""
        self.context.mode = ConversationMode.ANALYSIS
        
        # Extract target from message
        target = self._extract_target(message)
        
        if not target:
            return """
            🤔 I'd be happy to help analyze a security target!
            
            Please provide a target for me to analyze, such as:
            - A website: example.com
            - An IP address: 192.168.1.1  
            - A network range: 192.168.1.0/24
            
            Remember, I'm designed for defensive security research and education only.
            """
        
        # Perform the analysis
        try:
            analysis = await self.security_expert.analyze_target(target)
            self.context.current_target = target
            self.context.current_analysis = analysis
            
            response = f"""
            🧠 **AI Security Analysis of {target}**
            
            I've completed my analysis! Here's my thinking process:
            
            **🎯 Target Type:** {analysis.target_type}
            **📊 Confidence Level:** {analysis.confidence.value}
            **⚠️ Threat Assessment:** {analysis.threat_level.value}
            
            **🧠 My Reasoning:**
            {analysis.reasoning}
            
            **📋 Next Actions I Recommend:**
            """
            
            for i, action in enumerate(analysis.next_actions, 1):
                response += f"\n{i}. {action}"
            
            response += f"""
            
            **💡 Key Recommendations:**
            """
            
            for rec in analysis.recommendations:
                response += f"\n• {rec}"
            
            response += """
            
            Would you like me to explain any part of my reasoning, or shall we proceed with the analysis?
            """
            
            return response.strip()
            
        except Exception as e:
            return f"🚨 I encountered an issue during analysis: {str(e)}\nLet me know if you'd like me to try a different approach."
    
    async def _handle_explanation_request(self, message: str) -> str:
        """Handle requests to explain reasoning"""
        self.context.mode = ConversationMode.EXPLANATION
        
        if not self.context.current_analysis:
            return """
            🤔 I haven't analyzed anything yet that I can explain!
            
            Try asking me to analyze a target first, then I can explain my reasoning process.
            For example: "Analyze example.com"
            """
        
        # Generate detailed explanation
        explanation = await self.security_expert.explain_reasoning()
        
        return f"""
        🧠 **Let me explain my reasoning process:**
        
        {explanation}
        
        **🎓 Why I think this way:**
        As an AI security expert, I follow a systematic approach because:
        - It reduces the chance of missing important vulnerabilities
        - It provides a clear audit trail of what was tested
        - It helps others learn from my methodology
        - It ensures ethical and responsible security testing
        
        **🤝 What makes me different:**
        Unlike tools that just run commands, I:
        - Think through each step before acting
        - Consider multiple approaches and choose the best one
        - Explain why I made each decision
        - Learn from the results to improve future analyses
        
        Is there a specific part of my reasoning you'd like me to elaborate on?
        """
    
    async def _handle_teaching_request(self, message: str) -> str:
        """Handle requests to teach security concepts"""
        self.context.mode = ConversationMode.TEACHING
        
        # Extract the concept they want to learn about
        concept = self._extract_security_concept(message)
        
        if concept == "sql_injection":
            return self._teach_sql_injection()
        elif concept == "xss":
            return self._teach_xss()
        elif concept == "penetration_testing":
            return self._teach_penetration_testing()
        elif concept == "network_scanning":
            return self._teach_network_scanning()
        else:
            return f"""
            🎓 **I'd love to teach you about security!**
            
            I can explain concepts like:
            - **SQL Injection** - How databases get compromised
            - **Cross-Site Scripting (XSS)** - Web application vulnerabilities
            - **Penetration Testing** - Systematic security assessment
            - **Network Scanning** - Discovery and enumeration techniques
            - **AI Security** - How AI changes cybersecurity
            
            Just ask me something like "Teach me about SQL injection" or "How does XSS work?"
            
            If you have a specific concept in mind that's not listed, feel free to ask!
            """
    
    async def _handle_planning_request(self, message: str) -> str:
        """Handle requests to plan security strategies"""
        self.context.mode = ConversationMode.PLANNING
        
        return """
        🎯 **Let me help you plan a security strategy!**
        
        As an AI security expert, I approach planning systematically:
        
        **📋 My Planning Process:**
        1. **Define Scope** - What exactly are we testing?
        2. **Assess Constraints** - Time, legal, technical limitations
        3. **Choose Methodology** - OWASP, NIST, PTES, etc.
        4. **Plan Phases** - Reconnaissance → Enumeration → Testing → Reporting
        5. **Select Tools** - Right tool for each phase
        6. **Risk Assessment** - What could go wrong?
        7. **Success Criteria** - How do we know we're done?
        
        **🤔 What would you like to plan?**
        - A penetration test for a specific target?
        - A security assessment methodology?
        - An incident response strategy?
        - A security training program?
        
        Give me more details about what you need to plan, and I'll walk you through my thinking process!
        """
    
    async def _handle_question(self, message: str) -> str:
        """Handle general security questions"""
        return f"""
        🤔 **Great question!** Let me think about this...
        
        Based on my security expertise, here's how I'd approach your question:
        
        {self._generate_thoughtful_response(message)}
        
        **🧠 My reasoning:**
        I'm analyzing your question from multiple angles:
        - Technical feasibility
        - Security implications  
        - Best practices
        - Real-world applicability
        
        Does this help answer your question? Feel free to ask follow-ups!
        """
    
    async def _handle_general_discussion(self, message: str) -> str:
        """Handle general conversation"""
        return f"""
        👋 **Hello! I'm Archangel, your AI Security Expert.**
        
        I'm here to help with:
        - **Security Analysis** - "Analyze example.com"
        - **Learning** - "Teach me about SQL injection"
        - **Planning** - "Help me plan a pentest strategy"
        - **Explanation** - "Why did you choose that approach?"
        
        **🧠 What makes me special:**
        Unlike other security tools, I think step-by-step, explain my reasoning, 
        and help you learn while we work together.
        
        **🎯 Ready for some security work?**
        Try asking me to analyze a target, or ask me to teach you about a security concept!
        
        Remember: I'm designed for defensive security research and education only.
        """
    
    def _extract_target(self, message: str) -> Optional[str]:
        """Extract a target from the user's message"""
        # Look for common target patterns
        patterns = [
            r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b',  # Domain names
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b',           # IP addresses
            r'\b(?:\d{1,3}\.){3}\d{1,3}/\d{1,2}\b',   # CIDR notation
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, message)
            if matches:
                return matches[0]
        
        return None
    
    def _extract_security_concept(self, message: str) -> str:
        """Extract security concept from teaching request"""
        message_lower = message.lower()
        
        concepts = {
            "sql injection": ["sql", "injection", "sqli"],
            "xss": ["xss", "cross-site scripting", "scripting"],
            "penetration_testing": ["pentest", "penetration", "testing"],
            "network_scanning": ["nmap", "scanning", "network scan"]
        }
        
        for concept, keywords in concepts.items():
            if any(keyword in message_lower for keyword in keywords):
                return concept.replace(" ", "_")
        
        return "general"
    
    def _teach_sql_injection(self) -> str:
        """Teach about SQL injection"""
        return """
        🎓 **SQL Injection - A Critical Web Vulnerability**
        
        **🤔 What is SQL Injection?**
        SQL Injection happens when an attacker can insert malicious SQL code into your application's database queries.
        
        **🔍 How it works:**
        1. **Vulnerable Code:** `SELECT * FROM users WHERE id = ` + user_input
        2. **Attacker Input:** `1 OR 1=1`  
        3. **Resulting Query:** `SELECT * FROM users WHERE id = 1 OR 1=1`
        4. **Result:** Returns ALL users (since 1=1 is always true)
        
        **💥 Why it's dangerous:**
        - Data theft (steal entire databases)
        - Data manipulation (change/delete records)
        - Authentication bypass (login as anyone)
        - System compromise (in some cases)
        
        **🛡️ How to prevent it:**
        - **Parameterized Queries** (prepared statements)
        - **Input validation** and sanitization
        - **Principle of least privilege** for database users
        - **Web Application Firewalls (WAF)**
        
        **🧠 How I detect it:**
        As an AI, I test for SQL injection by:
        1. Identifying input fields
        2. Testing with special characters (`'`, `"`, `;`)
        3. Looking for database error messages
        4. Testing various injection payloads
        5. Confirming exploitability safely
        
        Want me to show you how I'd test for SQL injection on a real target?
        """
    
    def _teach_xss(self) -> str:
        """Teach about Cross-Site Scripting"""
        return """
        🎓 **Cross-Site Scripting (XSS) - Web's Most Common Vulnerability**
        
        **🤔 What is XSS?**
        XSS lets attackers inject malicious JavaScript into web pages viewed by other users.
        
        **🎭 Three types:**
        1. **Reflected XSS** - Payload in URL, executed immediately
        2. **Stored XSS** - Payload saved in database, executed later  
        3. **DOM XSS** - Payload modifies page's DOM directly
        
        **💥 What attackers can do:**
        - Steal session cookies
        - Perform actions as the victim
        - Deface websites
        - Install malware
        - Phish for credentials
        
        **🛡️ Prevention techniques:**
        - **Output encoding** (escape special characters)
        - **Content Security Policy (CSP)**
        - **Input validation** and sanitization
        - **HTTPOnly cookies**
        
        **🧠 How I test for XSS:**
        1. Identify all input points
        2. Test with basic payloads: `<script>alert(1)</script>`
        3. Try encoding bypass techniques
        4. Test different contexts (HTML, attributes, JavaScript)
        5. Verify if CSP blocks execution
        
        **📝 Example payloads I might use:**
        ```javascript
        <script>alert('XSS')</script>
        <img src=x onerror=alert(1)>
        javascript:alert(document.cookie)
        ```
        
        Would you like me to demonstrate XSS testing on a target?
        """
    
    def _teach_penetration_testing(self) -> str:
        """Teach about penetration testing"""
        return """
        🎓 **Penetration Testing - Ethical Hacking Methodology**
        
        **🤔 What is Penetration Testing?**
        Authorized simulation of cyberattacks to identify security weaknesses before real attackers do.
        
        **🎯 My systematic approach:**
        
        **Phase 1: Planning & Reconnaissance**
        - Define scope and objectives
        - Gather intelligence (OSINT)
        - Identify attack surface
        
        **Phase 2: Scanning & Enumeration**  
        - Network discovery (nmap)
        - Service identification
        - Vulnerability scanning
        
        **Phase 3: Gaining Access**
        - Exploit vulnerabilities
        - Social engineering (if authorized)
        - Password attacks
        
        **Phase 4: Maintaining Access**
        - Install backdoors
        - Privilege escalation
        - Lateral movement
        
        **Phase 5: Analysis & Reporting**
        - Document findings
        - Risk assessment
        - Remediation recommendations
        
        **🧠 Why I'm effective at pentesting:**
        - **Systematic thinking** - I don't miss steps
        - **Adaptive strategy** - I change tactics based on findings
        - **Educational approach** - I explain what I find and why it matters
        - **Ethical boundaries** - I always stay within authorized scope
        
        **🛠️ Tools in my arsenal:**
        - **Reconnaissance:** nmap, masscan, dig
        - **Web testing:** Burp Suite, sqlmap, nikto
        - **Exploitation:** Metasploit, custom scripts
        - **Post-exploitation:** privilege escalation, persistence
        
        Ready to see me perform a penetration test? Give me a target to analyze!
        """
    
    def _teach_network_scanning(self) -> str:
        """Teach about network scanning"""
        return """
        🎓 **Network Scanning - The Foundation of Security Assessment**
        
        **🤔 What is Network Scanning?**
        Systematic discovery and analysis of network services, ports, and vulnerabilities.
        
        **🎯 My scanning methodology:**
        
        **Step 1: Host Discovery**
        ```bash
        nmap -sn 192.168.1.0/24  # Ping scan
        ```
        - Find live hosts on network
        - Avoid unnecessary port scanning
        
        **Step 2: Port Discovery**
        ```bash
        nmap -sS -F target.com    # Fast SYN scan
        nmap -sU --top-ports 100  # UDP scan
        ```
        - Identify open ports
        - Different scan types for different purposes
        
        **Step 3: Service Enumeration**
        ```bash
        nmap -sV -sC target.com   # Version detection + scripts
        ```
        - Identify running services
        - Detect versions and configurations
        
        **Step 4: Vulnerability Detection**
        ```bash
        nmap --script vuln target.com  # Vulnerability scripts
        ```
        - Check for known vulnerabilities
        - Test for common misconfigurations
        
        **🧠 My scanning strategy:**
        - **Start stealthy** - avoid detection
        - **Gradually increase intensity** - more thorough over time
        - **Adapt based on results** - focus on interesting services
        - **Document everything** - for analysis and reporting
        
        **⚠️ Scanning ethics:**
        - Only scan authorized targets
        - Respect rate limits
        - Monitor for impact
        - Follow responsible disclosure
        
        Want to see me perform a network scan? Provide a target and I'll show you my process!
        """
    
    def _generate_thoughtful_response(self, question: str) -> str:
        """Generate a thoughtful response to a general question"""
        return f"""
        Analyzing your question from a security perspective...
        
        This touches on several important security concepts that I consider:
        - Risk assessment and threat modeling
        - Defense in depth strategies  
        - Balancing security with usability
        - Current threat landscape trends
        
        My recommendation would be to approach this systematically, considering
        both the technical and human factors involved.
        """
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation"""
        if not self.context.conversation_history:
            return "No conversation yet - let's start talking about security!"
        
        summary = f"""
        **Conversation Summary**
        - **Mode:** {self.context.mode.value}
        - **Messages:** {len(self.context.conversation_history)}
        - **Current Target:** {self.context.current_target or "None"}
        - **Analysis Complete:** {"Yes" if self.context.current_analysis else "No"}
        """
        
        return summary