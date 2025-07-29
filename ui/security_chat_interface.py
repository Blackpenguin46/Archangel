"""
Archangel Natural Language Security Discussion Interface
Advanced conversational AI for security education and analysis
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("⚠️ Gradio not available - web interface disabled")

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("⚠️ Streamlit not available - alternative interface disabled")

from ..core.huggingface_orchestrator import HuggingFaceAIOrchestrator, AIResponse
from ..tools.smolagents_security_tools import create_autonomous_security_agent

class ConversationMode(Enum):
    SECURITY_ANALYSIS = "security_analysis"
    EDUCATIONAL = "educational"
    THREAT_HUNTING = "threat_hunting"
    INCIDENT_RESPONSE = "incident_response"
    COMPLIANCE_DISCUSSION = "compliance"
    GENERAL_SECURITY = "general"

class UserExpertiseLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class ConversationContext:
    """Enhanced conversation context with security focus"""
    session_id: str
    user_expertise: UserExpertiseLevel
    current_mode: ConversationMode
    active_target: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = None
    security_findings: Dict[str, Any] = None
    learning_objectives: List[str] = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.security_findings is None:
            self.security_findings = {}
        if self.learning_objectives is None:
            self.learning_objectives = []

class SecurityChatInterface:
    """
    Advanced natural language security discussion interface
    
    Features:
    - Context-aware security conversations
    - Educational explanations adapted to user expertise
    - Real-time security analysis integration
    - Multi-modal interaction (text, analysis results, visualizations)
    - Learning progression tracking
    """
    
    def __init__(self, hf_orchestrator: HuggingFaceAIOrchestrator, hf_token: str):
        self.hf_orchestrator = hf_orchestrator
        self.hf_token = hf_token
        
        # Session management
        self.active_sessions: Dict[str, ConversationContext] = {}
        
        # Educational content
        self.security_curriculum = self._load_security_curriculum()
        self.assessment_questions = self._load_assessment_questions()
        
        # Analysis integration
        self.autonomous_agent = None
        
        # UI components
        self.gradio_interface = None
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize the chat interface"""
        self.logger.info("🎯 Initializing Security Chat Interface...")
        
        # Ensure HF orchestrator is ready
        if not hasattr(self.hf_orchestrator, 'active_models') or not self.hf_orchestrator.active_models:
            await self.hf_orchestrator.initialize()
        
        # Initialize autonomous agent for analysis
        await self._initialize_autonomous_agent()
        
        # Setup interfaces
        if GRADIO_AVAILABLE:
            self._setup_gradio_interface()
        
        self.logger.info("✅ Security Chat Interface ready!")
    
    async def _initialize_autonomous_agent(self):
        """Initialize autonomous security agent"""
        try:
            # Get best available model for autonomous operations
            model_key = self.hf_orchestrator._select_best_model(
                capability=self.hf_orchestrator.ModelCapability.TEXT_GENERATION,
                preferred_type=self.hf_orchestrator.ModelType.CYBERSECURITY_SPECIALIST
            )
            
            if model_key and model_key in self.hf_orchestrator.active_models:
                from ..tools.smolagents_security_tools import create_autonomous_security_agent
                from smolagents import LocalModel
                
                model = LocalModel(
                    model=self.hf_orchestrator.active_models[model_key],
                    tokenizer=self.hf_orchestrator.active_tokenizers[model_key]
                )
                
                self.autonomous_agent = create_autonomous_security_agent(model)
                self.logger.info("✅ Autonomous security agent initialized")
            else:
                self.logger.warning("⚠️ No suitable model for autonomous agent")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize autonomous agent: {e}")
    
    def _load_security_curriculum(self) -> Dict[str, Any]:
        """Load structured security curriculum"""
        return {
            "beginner": {
                "topics": [
                    "What is Cybersecurity?",
                    "Common Threats and Attacks",
                    "Basic Security Principles",
                    "Password Security",
                    "Safe Internet Practices"
                ],
                "learning_path": [
                    "introduction_to_security",
                    "threat_landscape",
                    "basic_defenses",
                    "personal_security"
                ]
            },
            "intermediate": {
                "topics": [
                    "Network Security Fundamentals",
                    "Web Application Security",
                    "Incident Response Basics",
                    "Security Assessment Methods",
                    "Risk Management"
                ],
                "learning_path": [
                    "network_security",
                    "application_security",
                    "assessment_methodologies",
                    "incident_handling"
                ]
            },
            "advanced": {
                "topics": [
                    "Penetration Testing Methodology",
                    "Advanced Threat Detection",
                    "Malware Analysis",
                    "Forensics Techniques",
                    "Security Architecture"
                ],
                "learning_path": [
                    "offensive_security",
                    "threat_hunting",
                    "malware_research",
                    "digital_forensics"
                ]
            },
            "expert": {
                "topics": [
                    "Zero-Day Research",
                    "Advanced Persistent Threats",
                    "AI in Cybersecurity",
                    "Security Research Methods",
                    "Threat Intelligence"
                ],
                "learning_path": [
                    "vulnerability_research",
                    "apt_analysis",
                    "ai_security",
                    "threat_intelligence"
                ]
            }
        }
    
    def _load_assessment_questions(self) -> Dict[str, List[str]]:
        """Load assessment questions for different expertise levels"""
        return {
            "beginner": [
                "What are the three main principles of information security?",
                "What makes a password strong?",
                "What is phishing and how can you recognize it?",
                "Why should you keep software updated?",
                "What is two-factor authentication?"
            ],
            "intermediate": [
                "Explain the difference between vulnerability assessment and penetration testing",
                "What are the phases of an incident response process?",
                "How does SSL/TLS protect web communications?",
                "What is the principle of least privilege?",
                "Describe common web application vulnerabilities"
            ],
            "advanced": [
                "Explain the kill chain methodology for APT analysis",
                "How would you detect lateral movement in a network?",
                "What are the key components of a threat hunting program?",
                "Describe advanced evasion techniques used by malware",
                "How do you establish attribution in cyber attacks?"
            ],
            "expert": [
                "Discuss the challenges of AI-powered attack detection",
                "How do nation-state actors differ from criminal groups?",
                "What are the ethical implications of offensive security research?",
                "Explain the concept of living-off-the-land attacks",
                "How is quantum computing expected to impact cryptography?"
            ]
        }
    
    async def start_conversation(self, 
                               user_message: str, 
                               session_id: str = "default",
                               expertise_level: str = "intermediate",
                               mode: str = "general") -> Tuple[str, Dict[str, Any]]:
        """Start or continue a security conversation"""
        
        # Get or create session context
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = ConversationContext(
                session_id=session_id,
                user_expertise=UserExpertiseLevel(expertise_level),
                current_mode=ConversationMode(mode)
            )
        
        context = self.active_sessions[session_id]
        
        # Add user message to history
        context.conversation_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": time.time()
        })
        
        # Analyze user intent and generate response
        response, metadata = await self._generate_contextual_response(user_message, context)
        
        # Add AI response to history
        context.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": time.time(),
            "metadata": metadata
        })
        
        return response, metadata
    
    async def _generate_contextual_response(self, 
                                          user_message: str, 
                                          context: ConversationContext) -> Tuple[str, Dict[str, Any]]:
        """Generate contextually appropriate response"""
        
        # Analyze user intent
        intent = await self._analyze_user_intent(user_message, context)
        
        # Route to appropriate handler
        if intent["type"] == "security_analysis":
            return await self._handle_security_analysis(user_message, context, intent)
        elif intent["type"] == "educational_question":
            return await self._handle_educational_question(user_message, context, intent)
        elif intent["type"] == "threat_discussion":
            return await self._handle_threat_discussion(user_message, context, intent)
        elif intent["type"] == "tool_request":
            return await self._handle_tool_request(user_message, context, intent)
        elif intent["type"] == "assessment":
            return await self._handle_assessment(user_message, context, intent)
        else:
            return await self._handle_general_discussion(user_message, context, intent)
    
    async def _analyze_user_intent(self, message: str, context: ConversationContext) -> Dict[str, Any]:
        """Analyze user intent using AI"""
        
        intent_analysis_prompt = f"""
        Analyze this user message in a cybersecurity context:
        
        User expertise level: {context.user_expertise.value}
        Current conversation mode: {context.current_mode.value}
        Message: "{message}"
        
        Classify the intent as one of:
        - security_analysis: User wants to analyze a target/system
        - educational_question: User is asking to learn about a security concept
        - threat_discussion: User wants to discuss threats, attacks, or incidents
        - tool_request: User wants to use security tools or see demonstrations
        - assessment: User is asking assessment/quiz questions
        - general_discussion: General conversation about security topics
        
        Also identify:
        - Target mentioned (if any)
        - Security concepts referenced
        - Urgency level (low/medium/high)
        - Learning objective
        
        Response format:
        Type: [intent_type]
        Target: [target_if_mentioned]
        Concepts: [list_of_concepts]
        Urgency: [urgency_level]
        Learning_objective: [what_user_wants_to_learn]
        """
        
        try:
            ai_response = await self.hf_orchestrator.security_analysis(intent_analysis_prompt)
            
            # Parse AI response (simplified parsing)
            response_text = ai_response.content.lower()
            
            intent = {
                "type": "general_discussion",  # default
                "target": None,
                "concepts": [],
                "urgency": "low",
                "learning_objective": None,
                "confidence": ai_response.confidence
            }
            
            # Extract intent type
            if "security_analysis" in response_text or "analyze" in message.lower():
                intent["type"] = "security_analysis"
            elif "teach" in message.lower() or "explain" in message.lower() or "what is" in message.lower():
                intent["type"] = "educational_question"
            elif "threat" in message.lower() or "attack" in message.lower() or "malware" in message.lower():
                intent["type"] = "threat_discussion"
            elif "tool" in message.lower() or "scan" in message.lower() or "test" in message.lower():
                intent["type"] = "tool_request"
            elif "quiz" in message.lower() or "test me" in message.lower():
                intent["type"] = "assessment"
            
            # Extract target if mentioned
            import re
            
            # Look for URLs, IPs, domains
            url_pattern = r'https?://[^\s]+'
            ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            domain_pattern = r'\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
            
            if re.search(url_pattern, message):
                intent["target"] = re.search(url_pattern, message).group()
            elif re.search(ip_pattern, message):
                intent["target"] = re.search(ip_pattern, message).group()
            elif re.search(domain_pattern, message):
                intent["target"] = re.search(domain_pattern, message).group()
            
            return intent
            
        except Exception as e:
            self.logger.error(f"Intent analysis failed: {e}")
            return {
                "type": "general_discussion",
                "target": None,
                "concepts": [],
                "urgency": "low",
                "learning_objective": None,
                "confidence": 0.0
            }
    
    async def _handle_security_analysis(self, 
                                      message: str, 
                                      context: ConversationContext, 
                                      intent: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Handle security analysis requests"""
        
        target = intent.get("target")
        if not target:
            return self._request_target_clarification(context.user_expertise)
        
        # Update context
        context.active_target = target
        context.current_mode = ConversationMode.SECURITY_ANALYSIS
        
        # Perform analysis using HF orchestrator
        analysis_response = await self.hf_orchestrator.security_analysis(
            f"Perform comprehensive security analysis of {target}",
            context=f"User expertise: {context.user_expertise.value}"
        )
        
        # Format response based on user expertise
        formatted_response = self._format_analysis_for_expertise(
            analysis_response, context.user_expertise, target
        )
        
        # Store findings
        context.security_findings[target] = {
            "analysis": analysis_response,
            "timestamp": time.time()
        }
        
        metadata = {
            "analysis_performed": True,
            "target": target,
            "model_used": analysis_response.model_used,
            "confidence": analysis_response.confidence
        }
        
        return formatted_response, metadata
    
    async def _handle_educational_question(self, 
                                         message: str, 
                                         context: ConversationContext, 
                                         intent: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Handle educational questions with adaptive explanations"""
        
        context.current_mode = ConversationMode.EDUCATIONAL
        
        # Generate educational content
        educational_prompt = f"""
        As an expert cybersecurity educator, answer this question for a {context.user_expertise.value}-level student:
        
        Question: {message}
        
        Provide:
        1. Clear explanation appropriate for {context.user_expertise.value} level
        2. Real-world examples
        3. Practical implications
        4. Related concepts to explore
        5. Hands-on exercises if applicable
        
        Make it engaging and educational.
        """
        
        ai_response = await self.hf_orchestrator.conversational_security_chat(
            educational_prompt, context.session_id
        )
        
        # Add curriculum connections
        curriculum = self.security_curriculum[context.user_expertise.value]
        related_topics = [topic for topic in curriculum["topics"] 
                         if any(word in topic.lower() for word in message.lower().split())]
        
        enhanced_response = ai_response.content
        
        if related_topics:
            enhanced_response += f"\n\n📚 **Related Topics to Explore:**\n"
            for topic in related_topics[:3]:
                enhanced_response += f"• {topic}\n"
        
        # Suggest progression
        if context.user_expertise == UserExpertiseLevel.BEGINNER:
            enhanced_response += f"\n💡 **Next Steps:** Ready to learn about network security fundamentals?"
        
        metadata = {
            "educational_content": True,
            "related_topics": related_topics,
            "expertise_level": context.user_expertise.value
        }
        
        return enhanced_response, metadata
    
    async def _handle_threat_discussion(self, 
                                      message: str, 
                                      context: ConversationContext, 
                                      intent: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Handle threat and attack discussions"""
        
        context.current_mode = ConversationMode.THREAT_HUNTING
        
        threat_analysis_prompt = f"""
        Discuss this cybersecurity threat topic for a {context.user_expertise.value}-level audience:
        
        Topic: {message}
        
        Cover:
        1. Threat description and characteristics
        2. Attack vectors and methods
        3. Impact and consequences
        4. Detection techniques
        5. Mitigation strategies
        6. Current threat landscape status
        
        Be educational and emphasize defensive approaches.
        """
        
        ai_response = await self.hf_orchestrator.security_analysis(threat_analysis_prompt)
        
        # Add current threat intelligence context
        enhanced_response = ai_response.content
        enhanced_response += "\n\n🔍 **Threat Intelligence Context:**\n"
        enhanced_response += "• Stay updated with latest threat reports from CISA, NIST\n"
        enhanced_response += "• Monitor security vendor threat intelligence feeds\n"
        enhanced_response += "• Join relevant cybersecurity communities and forums\n"
        
        metadata = {
            "threat_discussion": True,
            "threat_concepts": intent.get("concepts", []),
            "intelligence_provided": True
        }
        
        return enhanced_response, metadata
    
    async def _handle_tool_request(self, 
                                 message: str, 
                                 context: ConversationContext, 
                                 intent: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Handle security tool usage requests"""
        
        target = intent.get("target")
        
        if not target:
            return self._suggest_tool_demonstration(context.user_expertise)
        
        if not self.autonomous_agent:
            return ("Security tools are not available in this session. The autonomous agent couldn't be initialized.", {})
        
        # Use autonomous agent for tool execution
        try:
            tool_prompt = f"""
            Use available security tools to analyze {target}.
            
            Perform:
            1. Network reconnaissance
            2. Web application scanning (if applicable)  
            3. Threat intelligence gathering
            4. Generate comprehensive report
            
            Explain each step and finding clearly.
            """
            
            result = self.autonomous_agent.run(tool_prompt)
            
            formatted_response = f"""
            🛠️ **Autonomous Security Analysis of {target}**
            
            {str(result)}
            
            🧠 **AI Analysis Complete**
            The autonomous agent has completed its security assessment using multiple tools.
            All findings are for educational and defensive security purposes only.
            """
            
            metadata = {
                "autonomous_analysis": True,
                "target": target,
                "tools_used": True
            }
            
            return formatted_response, metadata
            
        except Exception as e:
            return (f"Tool execution failed: {str(e)}", {"error": True})
    
    async def _handle_assessment(self, 
                               message: str, 
                               context: ConversationContext, 
                               intent: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Handle assessment and quiz requests"""
        
        questions = self.assessment_questions[context.user_expertise.value]
        
        import random
        selected_questions = random.sample(questions, min(3, len(questions)))
        
        response = f"""
        📝 **Security Knowledge Assessment - {context.user_expertise.value.title()} Level**
        
        Here are some questions to test your understanding:
        
        """
        
        for i, question in enumerate(selected_questions, 1):
            response += f"{i}. {question}\n\n"
        
        response += """
        💡 **Tips:**
        - Take your time to think through each answer
        - Explain your reasoning
        - Ask for clarification if needed
        - I'll provide detailed feedback on your responses
        
        Go ahead and answer any or all of these questions!
        """
        
        metadata = {
            "assessment": True,
            "questions": selected_questions,
            "expertise_level": context.user_expertise.value
        }
        
        return response, metadata
    
    async def _handle_general_discussion(self, 
                                       message: str, 
                                       context: ConversationContext, 
                                       intent: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Handle general security discussions"""
        
        general_prompt = f"""
        Have a natural conversation about cybersecurity with a {context.user_expertise.value}-level person.
        
        Their message: {message}
        
        Be helpful, educational, and engaging. Relate to their expertise level and provide actionable insights.
        """
        
        ai_response = await self.hf_orchestrator.conversational_security_chat(
            general_prompt, context.session_id
        )
        
        metadata = {
            "general_discussion": True,
            "conversation_mode": context.current_mode.value
        }
        
        return ai_response.content, metadata
    
    def _request_target_clarification(self, expertise: UserExpertiseLevel) -> Tuple[str, Dict[str, Any]]:
        """Request clarification on target for analysis"""
        
        if expertise == UserExpertiseLevel.BEGINNER:
            response = """
            🎯 **Security Analysis Request**
            
            I'd be happy to help you analyze a target! Please provide:
            
            • **Website URL** (like https://example.com)
            • **IP Address** (like 192.168.1.1) 
            • **Domain name** (like example.com)
            
            **Remember:** Only analyze systems you own or have explicit permission to test!
            
            **Example:** "Analyze the security of https://example.com"
            """
        else:
            response = """
            🎯 **Security Analysis Ready**
            
            Please specify your target for security analysis:
            
            **Supported Targets:**
            • Web applications (URLs)
            • Network hosts (IP addresses) 
            • Domain names
            • Network ranges (CIDR notation)
            
            **Analysis Types Available:**
            • Network reconnaissance
            • Web application security testing
            • Threat intelligence gathering
            • Comprehensive security assessment
            
            **Example:** "Perform a comprehensive security analysis of example.com"
            """
        
        return response, {"clarification_needed": True}
    
    def _suggest_tool_demonstration(self, expertise: UserExpertiseLevel) -> Tuple[str, Dict[str, Any]]:
        """Suggest tool demonstration"""
        
        response = f"""
        🛠️ **Security Tools Available**
        
        I can demonstrate various security tools and techniques:
        
        **Available Tools:**
        • Network reconnaissance (nmap-style scanning)
        • Web application security testing
        • Threat intelligence gathering
        • Security report generation
        
        **To use tools, specify a target:**
        "Use security tools to analyze example.com"
        "Scan the ports on 192.168.1.1"
        "Perform web security testing on https://testsite.com"
        
        **Educational Mode:** I can also explain how these tools work without running them.
        """
        
        return response, {"tool_suggestion": True}
    
    def _format_analysis_for_expertise(self, 
                                     analysis: AIResponse, 
                                     expertise: UserExpertiseLevel, 
                                     target: str) -> str:
        """Format analysis results based on user expertise"""
        
        base_response = f"""
        🛡️ **Security Analysis Results for {target}**
        
        {analysis.content}
        
        **Model Used:** {analysis.model_used}
        **Confidence:** {analysis.confidence:.1%}
        """
        
        if expertise == UserExpertiseLevel.BEGINNER:
            base_response += f"""
            
            📚 **For Beginners:**
            • This analysis shows potential security issues
            • Green items are good security practices
            • Red items need attention
            • Ask me to explain any technical terms!
            """
        elif expertise == UserExpertiseLevel.EXPERT:
            base_response += f"""
            
            🔬 **Technical Details:**
            • Analysis performed using {analysis.model_used}
            • Execution time: {analysis.execution_time:.2f}s
            • Additional context available on request
            • Integration with autonomous tools available
            """
        
        return base_response
    
    def _setup_gradio_interface(self):
        """Setup Gradio web interface"""
        if not GRADIO_AVAILABLE:
            return
        
        def chat_wrapper(message, history, expertise, mode):
            """Wrapper for Gradio interface"""
            try:
                # Run async function in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                response, metadata = loop.run_until_complete(
                    self.start_conversation(message, "gradio_session", expertise, mode)
                )
                
                history.append([message, response])
                return history, ""
                
            except Exception as e:
                error_response = f"Error: {str(e)}"
                history.append([message, error_response])
                return history, ""
        
        # Create Gradio interface
        with gr.Blocks(title="Archangel Security AI", theme=gr.themes.Dark()) as interface:
            gr.Markdown("# 🛡️ Archangel Security AI Assistant")
            gr.Markdown("Interactive AI-powered cybersecurity education and analysis platform")
            
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        height=500,
                        show_label=False,
                        container=False
                    )
                    
                    msg = gr.Textbox(
                        placeholder="Ask about security, request analysis, or start learning...",
                        show_label=False,
                        container=False
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### Settings")
                    
                    expertise = gr.Dropdown(
                        choices=["beginner", "intermediate", "advanced", "expert"],
                        value="intermediate",
                        label="Expertise Level"
                    )
                    
                    mode = gr.Dropdown(
                        choices=["general", "security_analysis", "educational", "threat_hunting"],
                        value="general",
                        label="Conversation Mode"
                    )
                    
                    gr.Markdown("### Quick Actions")
                    
                    analyze_btn = gr.Button("🔍 Security Analysis")
                    learn_btn = gr.Button("📚 Learn Security")
                    tools_btn = gr.Button("🛠️ Use Tools")
                    quiz_btn = gr.Button("📝 Take Quiz")
            
            # Event handlers
            msg.submit(
                chat_wrapper,
                inputs=[msg, chatbot, expertise, mode],
                outputs=[chatbot, msg]
            )
            
            # Quick action buttons
            analyze_btn.click(
                lambda: "I want to perform a security analysis. What should I analyze?",
                outputs=msg
            )
            
            learn_btn.click(
                lambda: "Teach me about cybersecurity fundamentals",
                outputs=msg  
            )
            
            tools_btn.click(
                lambda: "Show me available security tools and how to use them",
                outputs=msg
            )
            
            quiz_btn.click(
                lambda: "Give me a cybersecurity quiz",
                outputs=msg
            )
        
        self.gradio_interface = interface
        
    def launch_gradio(self, port: int = 7860, share: bool = False):
        """Launch Gradio interface"""
        if self.gradio_interface:
            self.gradio_interface.launch(server_port=port, share=share)
        else:
            print("❌ Gradio interface not available")
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of conversation session"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        context = self.active_sessions[session_id]
        
        return {
            "session_id": session_id,
            "user_expertise": context.user_expertise.value,
            "current_mode": context.current_mode.value,
            "message_count": len(context.conversation_history),
            "active_target": context.active_target,
            "security_findings": len(context.security_findings),
            "learning_objectives": context.learning_objectives
        }

# Factory function
def create_security_chat_interface(hf_orchestrator: HuggingFaceAIOrchestrator, 
                                 hf_token: str) -> SecurityChatInterface:
    """Create and return security chat interface"""
    return SecurityChatInterface(hf_orchestrator, hf_token)