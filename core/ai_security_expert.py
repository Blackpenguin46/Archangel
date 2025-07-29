"""
Archangel AI Security Expert Brain
The core innovation - AI that thinks like a security expert
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time

class ConfidenceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class ThreatLevel(Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityAnalysis:
    """Results from AI security analysis"""
    target: str
    target_type: str
    reasoning: str
    strategy: Dict[str, Any]
    confidence: ConfidenceLevel
    threat_level: ThreatLevel
    recommendations: List[str]
    next_actions: List[str]
    timestamp: float

@dataclass
class AIThought:
    """Represents a single AI reasoning step"""
    step: str
    reasoning: str
    confidence: float
    alternatives_considered: List[str]
    timestamp: float

class SecurityExpertAI:
    """
    The main innovation - AI that thinks like a security expert
    
    This is the heart of Archangel - an AI that:
    - Thinks step by step about security problems
    - Explains its reasoning clearly
    - Adapts strategies based on discoveries
    - Learns from each operation
    """
    
    def __init__(self, model_name: str = "local-llm"):
        self.model_name = model_name
        self.thought_chain: List[AIThought] = []
        self.security_knowledge = self._load_security_knowledge()
        self.operation_history: List[SecurityAnalysis] = []
        self.tool_orchestrator = None  # Will be set by external system
        self.kernel_interface = None  # Will be set by external system
        
        # Research-backed enhancements from arxiv.org/abs/2501.16466
        self.action_abstraction_layer = self._init_action_abstraction()
        self.multi_step_decomposer = self._init_multi_step_decomposer()
        
    def _load_security_knowledge(self) -> Dict[str, Any]:
        """Load security domain knowledge"""
        return {
            "attack_vectors": {
                "web": ["sql_injection", "xss", "csrf", "rce", "file_upload"],
                "network": ["port_scan", "service_enum", "vuln_scan", "exploit"],
                "system": ["privilege_escalation", "persistence", "lateral_movement"]
            },
            "tools": {
                "reconnaissance": ["nmap", "masscan", "dig", "whois"],
                "web_testing": ["burp", "sqlmap", "nikto", "dirb"],
                "exploitation": ["metasploit", "custom_scripts", "public_exploits"]
            },
            "methodologies": ["owasp", "osstmm", "nist", "ptes"]
        }
    
    async def think_aloud(self, problem: str) -> str:
        """
        AI thinks step by step about a security problem
        This is where the magic happens - transparent AI reasoning
        """
        thoughts = []
        
        # Step 1: Problem analysis
        thought = await self._analyze_problem(problem)
        thoughts.append(thought)
        self.thought_chain.append(thought)
        
        # Step 2: Strategy formulation
        thought = await self._formulate_strategy(problem, thoughts[-1])
        thoughts.append(thought)
        self.thought_chain.append(thought)
        
        # Step 3: Risk assessment
        thought = await self._assess_risks(problem, thoughts)
        thoughts.append(thought)
        self.thought_chain.append(thought)
        
        # Compile reasoning
        reasoning = self._compile_reasoning(thoughts)
        return reasoning
    
    async def _analyze_problem(self, problem: str) -> AIThought:
        """First step: Analyze what we're dealing with"""
        # Simulate AI reasoning about the problem
        reasoning = f"""
        Let me analyze this problem: {problem}
        
        🧠 INITIAL ANALYSIS:
        - This appears to be a security assessment request
        - I need to identify the target type and scope
        - I should consider the ethical and legal implications
        - The approach must be systematic and thorough
        
        🎯 TARGET IDENTIFICATION:
        Based on the input, I'm identifying:
        - Target type: {self._identify_target_type(problem)}
        - Scope: Security assessment
        - Constraints: Defensive/educational only
        """
        
        return AIThought(
            step="problem_analysis",
            reasoning=reasoning.strip(),
            confidence=0.8,
            alternatives_considered=["passive reconnaissance", "active scanning", "documentation review"],
            timestamp=time.time()
        )
    
    async def _formulate_strategy(self, problem: str, analysis: AIThought) -> AIThought:
        """Second step: Create a strategic approach"""
        reasoning = f"""
        🧠 STRATEGY FORMULATION:
        Based on my analysis, I'm developing a systematic approach:
        
        1. RECONNAISSANCE PHASE:
           - Passive information gathering
           - Target surface mapping  
           - Technology identification
        
        2. ENUMERATION PHASE:
           - Service discovery
           - Version detection
           - Vulnerability identification
        
        3. ANALYSIS PHASE:
           - Risk assessment
           - Attack vector analysis
           - Impact evaluation
        
        🤔 MY REASONING:
        I'm choosing this phased approach because:
        - It minimizes risk to the target
        - Builds comprehensive understanding before action
        - Follows industry-standard methodologies
        - Maintains ethical boundaries
        """
        
        return AIThought(
            step="strategy_formulation",
            reasoning=reasoning.strip(),
            confidence=0.9,
            alternatives_considered=["aggressive scanning", "stealth approach", "documentation-first"],
            timestamp=time.time()
        )
    
    async def _assess_risks(self, problem: str, previous_thoughts: List[AIThought]) -> AIThought:
        """Third step: Assess risks and confidence"""
        reasoning = f"""
        🧠 RISK ASSESSMENT:
        Let me evaluate the risks and my confidence level:
        
        ⚠️ IDENTIFIED RISKS:
        - Target disruption: LOW (using passive techniques first)
        - Legal concerns: LOW (defensive security context)
        - Detection risk: MEDIUM (depends on target monitoring)
        
        📊 CONFIDENCE ANALYSIS:
        - Strategy soundness: HIGH (follows proven methodologies)
        - Target identification: {previous_thoughts[0].confidence}
        - Approach validity: {previous_thoughts[1].confidence}
        
        ✅ MITIGATION STRATEGIES:
        - Start with least intrusive methods
        - Monitor for any adverse effects
        - Maintain detailed operation logs
        - Ready to abort if issues detected
        """
        
        return AIThought(
            step="risk_assessment", 
            reasoning=reasoning.strip(),
            confidence=0.85,
            alternatives_considered=["high-risk approach", "ultra-conservative", "mixed strategy"],
            timestamp=time.time()
        )
    
    def _identify_target_type(self, target: str) -> str:
        """Identify what type of target we're dealing with"""
        if any(char in target for char in ['http', 'www', '.com', '.org']):
            return "web_application"
        elif any(char.isdigit() for char in target.replace('.', '')):
            return "ip_address"
        elif '/' in target:
            return "network_range"
        else:
            return "hostname"
    
    def _compile_reasoning(self, thoughts: List[AIThought]) -> str:
        """Compile all thoughts into coherent reasoning"""
        reasoning_parts = []
        
        for i, thought in enumerate(thoughts, 1):
            reasoning_parts.append(f"## Step {i}: {thought.step.replace('_', ' ').title()}")
            reasoning_parts.append(thought.reasoning)
            reasoning_parts.append(f"*Confidence: {thought.confidence:.1%}*")
            reasoning_parts.append("")
        
        return "\n".join(reasoning_parts)
    
    async def analyze_target(self, target: str) -> SecurityAnalysis:
        """
        Main method: AI analyzes a security target
        This combines thinking, reasoning, and strategic planning
        """
        print(f"🧠 AI Security Expert analyzing: {target}")
        
        # AI thinks about the problem
        reasoning = await self.think_aloud(f"Analyze the security of {target}")
        
        # Generate strategy based on reasoning
        strategy = await self._generate_strategy(target, reasoning)
        
        # Assess confidence and threat levels
        confidence = self._assess_confidence(reasoning)
        threat_level = self._assess_threat_level(target, reasoning)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(target, reasoning)
        
        # Plan next actions
        next_actions = await self._plan_next_actions(target, strategy)
        
        # Create analysis result
        analysis = SecurityAnalysis(
            target=target,
            target_type=self._identify_target_type(target),
            reasoning=reasoning,
            strategy=strategy,
            confidence=confidence,
            threat_level=threat_level,
            recommendations=recommendations,
            next_actions=next_actions,
            timestamp=time.time()
        )
        
        # Store in operation history for learning
        self.operation_history.append(analysis)
        
        return analysis
    
    async def _generate_strategy(self, target: str, reasoning: str) -> Dict[str, Any]:
        """Generate detailed strategy based on AI reasoning"""
        target_type = self._identify_target_type(target)
        
        if target_type == "web_application":
            return {
                "phase_1": {
                    "name": "Web Reconnaissance",
                    "tools": ["nmap", "dirb", "whatweb"],
                    "duration": "5-10 minutes",
                    "risk_level": "low"
                },
                "phase_2": { 
                    "name": "Application Analysis",
                    "tools": ["burp", "nikto", "sqlmap"],
                    "duration": "15-30 minutes",
                    "risk_level": "medium"
                },
                "phase_3": {
                    "name": "Vulnerability Assessment",
                    "tools": ["custom_scripts", "manual_testing"],
                    "duration": "30-60 minutes", 
                    "risk_level": "medium"
                }
            }
        elif target_type == "ip_address":
            return {
                "phase_1": {
                    "name": "Network Discovery",
                    "tools": ["nmap", "masscan"],
                    "duration": "2-5 minutes",
                    "risk_level": "low"
                },
                "phase_2": {
                    "name": "Service Enumeration", 
                    "tools": ["nmap_scripts", "banner_grabbing"],
                    "duration": "10-15 minutes",
                    "risk_level": "low"
                },
                "phase_3": {
                    "name": "Vulnerability Scanning",
                    "tools": ["nessus", "openvas", "custom_checks"],
                    "duration": "20-45 minutes",
                    "risk_level": "medium"
                }
            }
        else:
            return {
                "phase_1": {
                    "name": "Target Resolution",
                    "tools": ["dig", "whois", "dns_enum"],
                    "duration": "2-5 minutes",
                    "risk_level": "very_low"
                }
            }
    
    def _assess_confidence(self, reasoning: str) -> ConfidenceLevel:
        """Assess AI's confidence in the analysis"""
        # Simple heuristic based on reasoning complexity and completeness
        if len(reasoning) > 1000 and "HIGH" in reasoning:
            return ConfidenceLevel.HIGH
        elif len(reasoning) > 500:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _assess_threat_level(self, target: str, reasoning: str) -> ThreatLevel:
        """Assess potential threat level of target"""
        # Simple heuristic - in reality this would be much more sophisticated
        if "critical" in reasoning.lower():
            return ThreatLevel.CRITICAL
        elif "high" in reasoning.lower():
            return ThreatLevel.HIGH
        elif "medium" in reasoning.lower():
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    async def _generate_recommendations(self, target: str, reasoning: str) -> List[str]:
        """Generate security recommendations based on analysis"""
        return [
            "Start with passive reconnaissance to avoid detection",
            "Use rate limiting to prevent overwhelming the target", 
            "Monitor for any signs of impact during testing",
            "Document all findings for educational purposes",
            "Follow responsible disclosure if vulnerabilities found"
        ]
    
    async def _plan_next_actions(self, target: str, strategy: Dict[str, Any]) -> List[str]:
        """Plan the next actions based on strategy"""
        actions = []
        
        for phase_name, phase_data in strategy.items():
            actions.append(f"Execute {phase_data['name']} using {', '.join(phase_data['tools'])}")
            
        actions.append("Analyze results and adapt strategy as needed")
        actions.append("Generate comprehensive security report")
        
        return actions
    
    def get_current_thoughts(self) -> List[AIThought]:
        """Get the current chain of AI reasoning"""
        return self.thought_chain.copy()
    
    def clear_thoughts(self):
        """Clear the current thought chain"""
        self.thought_chain.clear()

    async def explain_reasoning(self, question: str = "") -> str:
        """
        AI explains its reasoning process
        This is key for the educational/conversational aspect
        """
        if not self.thought_chain:
            return "I haven't analyzed anything yet. Give me a target to analyze!"
        
        explanation = """
        🧠 Let me explain my reasoning process:
        
        """
        
        for i, thought in enumerate(self.thought_chain, 1):
            explanation += f"""
        **Step {i}: {thought.step.replace('_', ' ').title()}**
        {thought.reasoning}
        
        *Confidence in this step: {thought.confidence:.1%}*
        *Alternatives I considered: {', '.join(thought.alternatives_considered)}*
        
        ---
        """
        
        explanation += """
        
        This is how I think through security problems - step by step, with clear reasoning
        and consideration of alternatives. This transparency is what makes me different
        from tools that just execute commands without explanation.
        """
        
        return explanation.strip()
    
    def set_tool_orchestrator(self, orchestrator):
        """Set the tool orchestrator for AI-driven tool execution"""
        self.tool_orchestrator = orchestrator
    
    async def execute_strategy_with_tools(self, target: str, strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute strategy using actual security tools
        This demonstrates AI-driven autonomous tool execution
        """
        if not self.tool_orchestrator:
            return [{"error": "No tool orchestrator available"}]
        
        print(f"🤖 AI executing strategy with tools for {target}")
        
        # Use AI to orchestrate tool execution
        tool_results = await self.tool_orchestrator.ai_execute_strategy(target, strategy)
        
        # AI analyzes all results collectively
        await self._analyze_collective_results(tool_results)
        
        return [self._format_tool_result(result) for result in tool_results]
    
    def _format_tool_result(self, result) -> Dict[str, Any]:
        """Format tool result for display"""
        return {
            "tool": result.tool_name,
            "command": result.command,
            "success": result.exit_code == 0,
            "execution_time": f"{result.execution_time:.2f}s",
            "findings": result.parsed_data or {},
            "raw_output": result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout
        }
    
    async def _analyze_collective_results(self, results):
        """AI analyzes all tool results together to form conclusions"""
        print("🧠 AI performing collective analysis of all results...")
        
        # Simple analysis - in reality would be much more sophisticated
        total_tools = len(results)
        successful_tools = len([r for r in results if r.exit_code == 0])
        
        if successful_tools == total_tools:
            print("✅ AI assessment: All tools executed successfully")
        elif successful_tools > 0:
            print(f"⚠️ AI assessment: {successful_tools}/{total_tools} tools successful")
        else:
            print("❌ AI assessment: No tools executed successfully")
        
        # Look for interesting patterns across results
        all_ports = []
        all_services = []
        
        for result in results:
            if result.parsed_data:
                if 'open_ports' in result.parsed_data:
                    all_ports.extend(result.parsed_data['open_ports'])
                if 'services' in result.parsed_data:
                    all_services.extend(result.parsed_data['services'])
        
        if all_ports:
            print(f"🎯 AI identified {len(all_ports)} total open ports across all scans")
        
        if all_services:
            print(f"🔍 AI catalogued {len(all_services)} services")
            
            # AI looks for security implications
            web_services = [s for s in all_services if 'http' in s.get('service', '').lower()]
            ssh_services = [s for s in all_services if 'ssh' in s.get('service', '').lower()]
            
            if web_services:
                print(f"🌐 AI found {len(web_services)} web services - recommending web application testing")
            if ssh_services: 
                print(f"🔐 AI found {len(ssh_services)} SSH services - recommending authentication testing")
    
    def set_kernel_interface(self, kernel_interface):
        """Set the kernel interface for kernel-userspace communication"""
        self.kernel_interface = kernel_interface
    
    async def handle_kernel_analysis_request(self, security_context) -> str:
        """
        Handle analysis request from kernel module
        
        This is called when the kernel needs AI analysis for security decisions.
        Based on research from arxiv.org/abs/2501.16466 on LLM cyber automation.
        """
        print(f"🧠 AI handling kernel analysis request for PID {security_context.pid}")
        
        # Use action abstraction layer to translate kernel context to high-level security objective
        security_objective = self._translate_kernel_context_to_objective(security_context)
        
        # Apply multi-step decomposition as per research findings
        analysis_steps = self._decompose_security_analysis(security_objective)
        
        # Execute analysis with AI reasoning
        for step in analysis_steps:
            await self._execute_analysis_step(step, security_context)
        
        # Make final decision
        decision = self._make_kernel_security_decision(security_context)
        
        print(f"🎯 AI kernel decision: {decision}")
        return decision
    
    def _init_action_abstraction(self) -> Dict[str, Any]:
        """
        Initialize action abstraction layer
        Based on Incalmo framework from arxiv.org/abs/2501.16466
        """
        return {
            "syscall_patterns": {
                59: "process_execution",  # execve
                2: "file_access",         # open
                257: "file_access",       # openat
                101: "process_debug",     # ptrace
                165: "filesystem_mount",  # mount
            },
            "high_level_actions": {
                "process_execution": ["analyze_binary", "check_permissions", "validate_origin"],
                "file_access": ["check_file_permissions", "validate_path", "assess_sensitivity"],
                "process_debug": ["verify_debug_privileges", "check_target_process"],
                "filesystem_mount": ["validate_mount_source", "check_mount_permissions"]
            }
        }
    
    def _init_multi_step_decomposer(self) -> Dict[str, Any]:
        """
        Initialize multi-step task decomposer
        Research shows LLMs need modular goal decomposition for complex tasks
        """
        return {
            "analysis_phases": [
                "context_extraction",
                "threat_assessment", 
                "pattern_matching",
                "decision_synthesis"
            ],
            "decision_factors": [
                "process_legitimacy",
                "user_context",
                "system_state",
                "historical_patterns"
            ]
        }
    
    def _translate_kernel_context_to_objective(self, context) -> str:
        """
        Translate low-level kernel context to high-level security objective
        This implements the abstraction layer concept from the research
        """
        syscall_nr = context.syscall_nr
        action_type = self.action_abstraction_layer["syscall_patterns"].get(syscall_nr, "unknown")
        
        objectives = {
            "process_execution": f"Assess security implications of process execution by PID {context.pid}",
            "file_access": f"Evaluate file access request from PID {context.pid}",
            "process_debug": f"Analyze process debugging attempt by PID {context.pid}",
            "filesystem_mount": f"Review filesystem mount operation by PID {context.pid}",
            "unknown": f"Analyze unknown syscall {syscall_nr} from PID {context.pid}"
        }
        
        return objectives.get(action_type, objectives["unknown"])
    
    def _decompose_security_analysis(self, objective: str) -> List[str]:
        """
        Decompose high-level security objective into executable analysis steps
        Research shows this modular approach improves LLM task completion
        """
        base_steps = [
            f"Extract context information for: {objective}",
            f"Assess threat level for: {objective}",
            f"Match against known patterns for: {objective}",
            f"Synthesize security decision for: {objective}"
        ]
        
        return base_steps
    
    async def _execute_analysis_step(self, step: str, context):
        """Execute individual analysis step with AI reasoning"""
        print(f"🔍 AI executing: {step}")
        
        # Simulate AI analysis with reasoning
        if "context information" in step:
            print(f"   📋 Process: {context.comm}, UID: {context.uid}, Flags: {hex(context.flags)}")
        elif "threat level" in step:
            threat_indicators = self._assess_threat_indicators(context)
            print(f"   ⚠️ Threat indicators: {threat_indicators}")
        elif "known patterns" in step:
            patterns = self._match_security_patterns(context)
            print(f"   🔍 Pattern matches: {patterns}")
        elif "security decision" in step:
            print(f"   🎯 Preparing final security decision...")
    
    def _assess_threat_indicators(self, context) -> List[str]:
        """Assess threat indicators from security context"""
        indicators = []
        
        # Check for suspicious process names
        suspicious_names = ["nc", "netcat", "wget", "curl", "python", "perl", "bash"]
        if any(name in context.comm.lower() for name in suspicious_names):
            indicators.append("suspicious_process_name")
        
        # Check for privileged operations
        if context.uid == 0:
            indicators.append("root_execution")
        
        # Check syscall patterns
        if context.syscall_nr == 59:  # execve
            indicators.append("process_execution")
        
        return indicators
    
    def _match_security_patterns(self, context) -> List[str]:
        """Match against known security patterns"""
        patterns = []
        
        # Pattern matching based on historical data
        if context.flags & 0x0001:  # EXECVE flag
            patterns.append("binary_execution_pattern")
        
        if context.flags & 0x0004:  # NETWORK flag
            patterns.append("network_activity_pattern")
        
        return patterns
    
    def _make_kernel_security_decision(self, context) -> str:
        """
        Make final security decision for kernel
        
        Returns decision that kernel can understand and act upon
        """
        # Simple decision logic - in practice would be much more sophisticated
        threat_indicators = self._assess_threat_indicators(context)
        
        if "suspicious_process_name" in threat_indicators and "root_execution" in threat_indicators:
            return "DENY"
        elif len(threat_indicators) > 0:
            return "MONITOR"
        else:
            return "ALLOW"