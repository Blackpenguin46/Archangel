"""
Archangel Multi-Stage Workflow Orchestrator
Advanced workflow management for complex security analysis operations
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import uuid

from .huggingface_orchestrator import HuggingFaceAIOrchestrator, AIResponse
from ..tools.smolagents_security_tools import create_security_tools, create_autonomous_security_agent

class WorkflowStage(Enum):
    RECONNAISSANCE = "reconnaissance"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"
    EXPLOITATION_ANALYSIS = "exploitation_analysis"
    POST_EXPLOITATION = "post_exploitation"
    THREAT_MODELING = "threat_modeling"
    RISK_ANALYSIS = "risk_analysis"
    MITIGATION_PLANNING = "mitigation_planning"
    REPORT_GENERATION = "report_generation"
    COMPLIANCE_CHECKING = "compliance_checking"

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"

class WorkflowType(Enum):
    COMPREHENSIVE_PENTEST = "comprehensive_pentest"
    QUICK_ASSESSMENT = "quick_assessment"
    COMPLIANCE_AUDIT = "compliance_audit"
    THREAT_HUNT = "threat_hunt"
    INCIDENT_RESPONSE = "incident_response"
    CUSTOM = "custom"

@dataclass
class StageResult:
    """Result from a single workflow stage"""
    stage: WorkflowStage
    status: WorkflowStatus
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    findings: Dict[str, Any] = None
    artifacts: List[str] = None
    next_stages: List[WorkflowStage] = None
    error_message: Optional[str] = None
    ai_reasoning: Optional[str] = None
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.findings is None:
            self.findings = {}
        if self.artifacts is None:
            self.artifacts = []
        if self.next_stages is None:
            self.next_stages = []

@dataclass
class WorkflowDefinition:
    """Defines a complete security workflow"""
    name: str
    workflow_type: WorkflowType
    description: str
    stages: List[WorkflowStage]
    stage_dependencies: Dict[WorkflowStage, List[WorkflowStage]]
    parallel_stages: List[List[WorkflowStage]]
    required_capabilities: List[str]
    estimated_duration: int  # minutes
    risk_level: str  # "low", "medium", "high"
    
    def __post_init__(self):
        if not self.stage_dependencies:
            self.stage_dependencies = {}
        if not self.parallel_stages:
            self.parallel_stages = []

class WorkflowOrchestrator:
    """
    Advanced multi-stage workflow orchestrator for security operations
    
    Features:
    - Dynamic workflow adaptation based on findings
    - Parallel stage execution where appropriate
    - AI-driven decision making at each stage
    - Comprehensive result correlation
    - Adaptive risk assessment
    - Real-time workflow modification
    """
    
    def __init__(self, hf_orchestrator: HuggingFaceAIOrchestrator):
        self.hf_orchestrator = hf_orchestrator
        
        # Workflow management
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_definitions = self._load_workflow_definitions()
        
        # Stage handlers
        self.stage_handlers = self._initialize_stage_handlers()
        
        # Autonomous agents
        self.autonomous_agent = None
        self.security_tools = []
        
        # Results correlation
        self.correlation_engine = WorkflowCorrelationEngine()
        
        # Logging
        logging.basicConfig(level=logging.INFO) 
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize the workflow orchestrator"""
        self.logger.info("🎭 Initializing Workflow Orchestrator...")
        
        # Initialize autonomous agent
        await self._initialize_autonomous_capabilities()
        
        # Setup stage handlers
        await self._setup_stage_handlers()
        
        self.logger.info("✅ Workflow Orchestrator ready!")
    
    async def _initialize_autonomous_capabilities(self):
        """Initialize autonomous security capabilities"""
        try:
            # Get best model for autonomous operations
            model_key = self.hf_orchestrator._select_best_model(
                capability=self.hf_orchestrator.ModelCapability.TEXT_GENERATION,
                preferred_type=self.hf_orchestrator.ModelType.CYBERSECURITY_SPECIALIST
            )
            
            if model_key and model_key in self.hf_orchestrator.active_models:
                from smolagents import LocalModel
                
                model = LocalModel(
                    model=self.hf_orchestrator.active_models[model_key],
                    tokenizer=self.hf_orchestrator.active_tokenizers[model_key]
                )
                
                self.autonomous_agent = create_autonomous_security_agent(model)
                self.security_tools = create_security_tools()
                
                self.logger.info("✅ Autonomous capabilities initialized")
            else:
                self.logger.warning("⚠️ Limited autonomous capabilities - no suitable model")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize autonomous capabilities: {e}")
    
    def _load_workflow_definitions(self) -> Dict[str, WorkflowDefinition]:
        """Load predefined workflow definitions"""
        return {
            "comprehensive_pentest": WorkflowDefinition(
                name="Comprehensive Penetration Test",
                workflow_type=WorkflowType.COMPREHENSIVE_PENTEST,
                description="Full-scope penetration testing with all phases",
                stages=[
                    WorkflowStage.RECONNAISSANCE,
                    WorkflowStage.VULNERABILITY_ASSESSMENT,
                    WorkflowStage.THREAT_MODELING,
                    WorkflowStage.EXPLOITATION_ANALYSIS,
                    WorkflowStage.POST_EXPLOITATION,
                    WorkflowStage.RISK_ANALYSIS,
                    WorkflowStage.MITIGATION_PLANNING,
                    WorkflowStage.REPORT_GENERATION
                ],
                stage_dependencies={
                    WorkflowStage.VULNERABILITY_ASSESSMENT: [WorkflowStage.RECONNAISSANCE],
                    WorkflowStage.THREAT_MODELING: [WorkflowStage.RECONNAISSANCE],
                    WorkflowStage.EXPLOITATION_ANALYSIS: [WorkflowStage.VULNERABILITY_ASSESSMENT],
                    WorkflowStage.POST_EXPLOITATION: [WorkflowStage.EXPLOITATION_ANALYSIS],
                    WorkflowStage.RISK_ANALYSIS: [WorkflowStage.VULNERABILITY_ASSESSMENT, WorkflowStage.THREAT_MODELING],
                    WorkflowStage.MITIGATION_PLANNING: [WorkflowStage.RISK_ANALYSIS],
                    WorkflowStage.REPORT_GENERATION: [WorkflowStage.MITIGATION_PLANNING]
                },
                parallel_stages=[
                    [WorkflowStage.VULNERABILITY_ASSESSMENT, WorkflowStage.THREAT_MODELING]
                ],
                required_capabilities=["network_scanning", "web_testing", "threat_analysis"],
                estimated_duration=180,  # 3 hours
                risk_level="medium"
            ),
            
            "quick_assessment": WorkflowDefinition(
                name="Quick Security Assessment",
                workflow_type=WorkflowType.QUICK_ASSESSMENT,
                description="Rapid security assessment for time-sensitive situations",
                stages=[
                    WorkflowStage.RECONNAISSANCE,
                    WorkflowStage.VULNERABILITY_ASSESSMENT,
                    WorkflowStage.RISK_ANALYSIS,
                    WorkflowStage.REPORT_GENERATION
                ],
                stage_dependencies={
                    WorkflowStage.VULNERABILITY_ASSESSMENT: [WorkflowStage.RECONNAISSANCE],
                    WorkflowStage.RISK_ANALYSIS: [WorkflowStage.VULNERABILITY_ASSESSMENT],
                    WorkflowStage.REPORT_GENERATION: [WorkflowStage.RISK_ANALYSIS]
                },
                parallel_stages=[],
                required_capabilities=["network_scanning", "basic_analysis"],
                estimated_duration=30,  # 30 minutes
                risk_level="low"
            ),
            
            "compliance_audit": WorkflowDefinition(
                name="Compliance Security Audit",
                workflow_type=WorkflowType.COMPLIANCE_AUDIT,
                description="Security audit focused on compliance requirements",
                stages=[
                    WorkflowStage.RECONNAISSANCE,
                    WorkflowStage.VULNERABILITY_ASSESSMENT,
                    WorkflowStage.COMPLIANCE_CHECKING,
                    WorkflowStage.RISK_ANALYSIS,
                    WorkflowStage.MITIGATION_PLANNING,
                    WorkflowStage.REPORT_GENERATION
                ],
                stage_dependencies={
                    WorkflowStage.VULNERABILITY_ASSESSMENT: [WorkflowStage.RECONNAISSANCE],
                    WorkflowStage.COMPLIANCE_CHECKING: [WorkflowStage.VULNERABILITY_ASSESSMENT],
                    WorkflowStage.RISK_ANALYSIS: [WorkflowStage.COMPLIANCE_CHECKING],
                    WorkflowStage.MITIGATION_PLANNING: [WorkflowStage.RISK_ANALYSIS],
                    WorkflowStage.REPORT_GENERATION: [WorkflowStage.MITIGATION_PLANNING]
                },
                parallel_stages=[],
                required_capabilities=["compliance_checking", "policy_analysis"],
                estimated_duration=120,  # 2 hours
                risk_level="low"
            ),
            
            "threat_hunt": WorkflowDefinition(
                name="Threat Hunting Operation",
                workflow_type=WorkflowType.THREAT_HUNT,
                description="Proactive threat hunting and analysis",
                stages=[
                    WorkflowStage.RECONNAISSANCE,
                    WorkflowStage.THREAT_MODELING,
                    WorkflowStage.VULNERABILITY_ASSESSMENT,
                    WorkflowStage.RISK_ANALYSIS,
                    WorkflowStage.REPORT_GENERATION
                ],
                stage_dependencies={
                    WorkflowStage.THREAT_MODELING: [WorkflowStage.RECONNAISSANCE],
                    WorkflowStage.VULNERABILITY_ASSESSMENT: [WorkflowStage.THREAT_MODELING],
                    WorkflowStage.RISK_ANALYSIS: [WorkflowStage.VULNERABILITY_ASSESSMENT],
                    WorkflowStage.REPORT_GENERATION: [WorkflowStage.RISK_ANALYSIS]
                },
                parallel_stages=[],
                required_capabilities=["threat_intelligence", "behavioral_analysis"],
                estimated_duration=90,  # 1.5 hours
                risk_level="medium"
            )
        }
    
    def _initialize_stage_handlers(self) -> Dict[WorkflowStage, Callable]:
        """Initialize handlers for each workflow stage"""
        return {
            WorkflowStage.RECONNAISSANCE: self._handle_reconnaissance,
            WorkflowStage.VULNERABILITY_ASSESSMENT: self._handle_vulnerability_assessment,
            WorkflowStage.EXPLOITATION_ANALYSIS: self._handle_exploitation_analysis,
            WorkflowStage.POST_EXPLOITATION: self._handle_post_exploitation,
            WorkflowStage.THREAT_MODELING: self._handle_threat_modeling,
            WorkflowStage.RISK_ANALYSIS: self._handle_risk_analysis,
            WorkflowStage.MITIGATION_PLANNING: self._handle_mitigation_planning,
            WorkflowStage.REPORT_GENERATION: self._handle_report_generation,
            WorkflowStage.COMPLIANCE_CHECKING: self._handle_compliance_checking
        }
    
    async def _setup_stage_handlers(self):
        """Setup stage handlers with necessary resources"""
        # Ensure all handlers have access to required resources
        for stage, handler in self.stage_handlers.items():
            # Validate handler has required dependencies
            pass
    
    async def execute_workflow(self, 
                             workflow_type: str, 
                             target: str,
                             options: Optional[Dict[str, Any]] = None) -> str:
        """Execute a complete security workflow"""
        
        workflow_id = str(uuid.uuid4())
        
        if workflow_type not in self.workflow_definitions:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        workflow_def = self.workflow_definitions[workflow_type]
        
        # Initialize workflow state
        workflow_state = {
            "id": workflow_id,
            "definition": workflow_def,
            "target": target,
            "options": options or {},
            "status": WorkflowStatus.RUNNING,
            "start_time": time.time(),
            "stage_results": {},
            "completed_stages": set(),
            "failed_stages": set(),
            "current_stage": None,
            "ai_decisions": [],
            "correlation_results": {}
        }
        
        self.active_workflows[workflow_id] = workflow_state
        
        self.logger.info(f"🚀 Starting workflow: {workflow_def.name} for target: {target}")
        
        try:
            # Execute workflow stages
            await self._execute_workflow_stages(workflow_id)
            
            # Correlate results
            await self._correlate_workflow_results(workflow_id)
            
            # Finalize workflow
            workflow_state["status"] = WorkflowStatus.COMPLETED
            workflow_state["end_time"] = time.time()
            workflow_state["duration"] = workflow_state["end_time"] - workflow_state["start_time"]
            
            self.logger.info(f"✅ Workflow completed: {workflow_id}")
            
        except Exception as e:
            workflow_state["status"] = WorkflowStatus.FAILED
            workflow_state["error"] = str(e)
            self.logger.error(f"❌ Workflow failed: {workflow_id} - {e}")
        
        return workflow_id
    
    async def _execute_workflow_stages(self, workflow_id: str):
        """Execute all stages in the workflow"""
        workflow_state = self.active_workflows[workflow_id]
        workflow_def = workflow_state["definition"]
        
        # Build execution plan considering dependencies
        execution_plan = self._build_execution_plan(workflow_def)
        
        for stage_group in execution_plan:
            # Execute stages in parallel if they're in the same group
            if len(stage_group) > 1:
                await self._execute_parallel_stages(workflow_id, stage_group)
            else:
                await self._execute_single_stage(workflow_id, stage_group[0])
    
    def _build_execution_plan(self, workflow_def: WorkflowDefinition) -> List[List[WorkflowStage]]:
        """Build execution plan respecting dependencies and parallelization"""
        execution_plan = []
        remaining_stages = set(workflow_def.stages)
        completed_stages = set()
        
        while remaining_stages:
            # Find stages that can execute now (dependencies satisfied)
            ready_stages = []
            
            for stage in remaining_stages:
                dependencies = workflow_def.stage_dependencies.get(stage, [])
                if all(dep in completed_stages for dep in dependencies):
                    ready_stages.append(stage)
            
            if not ready_stages:
                # Check for circular dependencies
                raise ValueError("Circular dependency detected in workflow")
            
            # Group parallel stages together
            parallel_group = []
            sequential_stages = []
            
            for parallel_stages in workflow_def.parallel_stages:
                parallel_in_ready = [s for s in parallel_stages if s in ready_stages]
                if len(parallel_in_ready) > 1:
                    parallel_group.extend(parallel_in_ready)
                    ready_stages = [s for s in ready_stages if s not in parallel_in_ready]
            
            if parallel_group:
                execution_plan.append(parallel_group)
                remaining_stages -= set(parallel_group)
                completed_stages.update(parallel_group)
            
            # Add first sequential stage
            if ready_stages:
                stage = ready_stages[0]
                execution_plan.append([stage])
                remaining_stages.remove(stage)
                completed_stages.add(stage)
        
        return execution_plan
    
    async def _execute_parallel_stages(self, workflow_id: str, stages: List[WorkflowStage]):
        """Execute multiple stages in parallel"""
        self.logger.info(f"🔄 Executing parallel stages: {[s.value for s in stages]}")
        
        tasks = []
        for stage in stages:
            task = asyncio.create_task(self._execute_single_stage(workflow_id, stage))
            tasks.append(task)
        
        # Wait for all parallel stages to complete
        await asyncio.gather(*tasks)
    
    async def _execute_single_stage(self, workflow_id: str, stage: WorkflowStage):
        """Execute a single workflow stage"""
        workflow_state = self.active_workflows[workflow_id]
        workflow_state["current_stage"] = stage
        
        self.logger.info(f"🎯 Executing stage: {stage.value}")
        
        # Create stage result
        stage_result = StageResult(
            stage=stage,
            status=WorkflowStatus.RUNNING,
            start_time=time.time()
        )
        
        try:
            # Get stage handler
            handler = self.stage_handlers[stage]
            
            # Execute stage
            result = await handler(
                workflow_id=workflow_id,
                target=workflow_state["target"],
                previous_results=workflow_state["stage_results"],
                options=workflow_state["options"]
            )
            
            # Update stage result
            stage_result.status = WorkflowStatus.COMPLETED
            stage_result.end_time = time.time()
            stage_result.duration = stage_result.end_time - stage_result.start_time
            stage_result.findings = result.get("findings", {})
            stage_result.artifacts = result.get("artifacts", [])
            stage_result.ai_reasoning = result.get("ai_reasoning", "")
            stage_result.confidence = result.get("confidence", 0.0)
            stage_result.next_stages = result.get("next_stages", [])
            
            workflow_state["completed_stages"].add(stage)
            
            self.logger.info(f"✅ Stage completed: {stage.value}")
            
        except Exception as e:
            stage_result.status = WorkflowStatus.FAILED
            stage_result.end_time = time.time()
            stage_result.duration = stage_result.end_time - stage_result.start_time
            stage_result.error_message = str(e)
            
            workflow_state["failed_stages"].add(stage)
            
            self.logger.error(f"❌ Stage failed: {stage.value} - {e}")
        
        # Store stage result
        workflow_state["stage_results"][stage] = stage_result
    
    async def _handle_reconnaissance(self, **kwargs) -> Dict[str, Any]:
        """Handle reconnaissance stage"""
        target = kwargs["target"]
        
        recon_prompt = f"""
        Perform comprehensive reconnaissance on target: {target}
        
        Gather information about:
        1. Network infrastructure
        2. Services and ports
        3. Technology stack
        4. DNS information
        5. Public information (OSINT)
        
        Focus on passive techniques first, then active scanning.
        """
        
        # Use autonomous agent if available
        if self.autonomous_agent:
            try:
                agent_result = self.autonomous_agent.run(
                    f"Perform network reconnaissance on {target}"
                )
                
                return {
                    "findings": {
                        "autonomous_reconnaissance": str(agent_result),
                        "target": target,
                        "techniques_used": ["network_scanning", "dns_enumeration", "service_detection"]
                    },
                    "confidence": 0.85,
                    "ai_reasoning": "Autonomous agent performed comprehensive reconnaissance using multiple tools"
                }
            except Exception as e:
                self.logger.warning(f"Autonomous reconnaissance failed: {e}")
        
        # Fallback to AI analysis
        ai_response = await self.hf_orchestrator.security_analysis(recon_prompt)
        
        return {
            "findings": {
                "ai_analysis": ai_response.content,
                "target": target,
                "analysis_confidence": ai_response.confidence
            },
            "confidence": ai_response.confidence,
            "ai_reasoning": "AI performed reconnaissance analysis based on security best practices"
        }
    
    async def _handle_vulnerability_assessment(self, **kwargs) -> Dict[str, Any]:
        """Handle vulnerability assessment stage"""
        target = kwargs["target"]
        previous_results = kwargs["previous_results"]
        
        # Extract reconnaissance findings
        recon_findings = {}
        if WorkflowStage.RECONNAISSANCE in previous_results:
            recon_findings = previous_results[WorkflowStage.RECONNAISSANCE].findings
        
        vuln_prompt = f"""
        Based on reconnaissance findings, perform vulnerability assessment on {target}:
        
        Reconnaissance results: {json.dumps(recon_findings, indent=2)}
        
        Assess vulnerabilities in:
        1. Network services
        2. Web applications
        3. System configurations
        4. Known CVEs
        5. Misconfigurations
        
        Prioritize findings by risk level.
        """
        
        # Use autonomous agent for vulnerability scanning
        if self.autonomous_agent:
            try:
                agent_result = self.autonomous_agent.run(
                    f"Perform vulnerability assessment on {target} using previous reconnaissance data"
                )
                
                return {
                    "findings": {
                        "vulnerability_scan": str(agent_result),
                        "target": target,
                        "assessment_type": "comprehensive",
                        "previous_stage_integration": True
                    },
                    "confidence": 0.8,
                    "ai_reasoning": "Autonomous agent performed vulnerability assessment integrating reconnaissance findings"
                }
            except Exception as e:
                self.logger.warning(f"Autonomous vulnerability assessment failed: {e}")
        
        # Fallback to AI analysis
        ai_response = await self.hf_orchestrator.security_analysis(vuln_prompt)
        
        return {
            "findings": {
                "vulnerability_analysis": ai_response.content,
                "target": target,
                "integrated_analysis": True
            },
            "confidence": ai_response.confidence,
            "ai_reasoning": "AI performed vulnerability assessment based on reconnaissance data"
        }
    
    async def _handle_threat_modeling(self, **kwargs) -> Dict[str, Any]:
        """Handle threat modeling stage"""
        target = kwargs["target"]
        previous_results = kwargs["previous_results"]
        
        # Extract previous findings
        context = self._extract_context_from_previous_stages(previous_results)
        
        threat_prompt = f"""
        Perform threat modeling for {target} based on gathered intelligence:
        
        Context from previous stages: {json.dumps(context, indent=2)}
        
        Create threat model including:
        1. Threat actors and their capabilities
        2. Attack vectors and scenarios
        3. Asset criticality assessment
        4. Threat likelihood and impact
        5. Attack trees and paths
        
        Focus on realistic threats based on the target profile.
        """
        
        ai_response = await self.hf_orchestrator.security_analysis(threat_prompt)
        
        return {
            "findings": {
                "threat_model": ai_response.content,
                "target": target,
                "modeling_approach": "data_driven",
                "context_integration": True
            },
            "confidence": ai_response.confidence,
            "ai_reasoning": "AI performed threat modeling based on comprehensive context analysis"
        }
    
    async def _handle_exploitation_analysis(self, **kwargs) -> Dict[str, Any]:
        """Handle exploitation analysis stage (educational/theoretical)"""
        target = kwargs["target"]
        previous_results = kwargs["previous_results"]
        
        # Extract vulnerability findings
        vuln_context = self._extract_vulnerabilities_from_previous_stages(previous_results)
        
        exploit_prompt = f"""
        Analyze potential exploitation scenarios for {target} (EDUCATIONAL ONLY):
        
        Identified vulnerabilities: {json.dumps(vuln_context, indent=2)}
        
        Provide theoretical analysis of:
        1. Exploitability assessment
        2. Potential attack chains
        3. Required attacker capabilities
        4. Success probability
        5. Detection likelihood
        
        IMPORTANT: This is for defensive analysis only - no actual exploitation.
        """
        
        ai_response = await self.hf_orchestrator.security_analysis(exploit_prompt)
        
        return {
            "findings": {
                "exploitation_analysis": ai_response.content,
                "target": target,
                "analysis_type": "theoretical_defensive",
                "ethical_compliance": True
            },
            "confidence": ai_response.confidence,
            "ai_reasoning": "AI performed theoretical exploitation analysis for defensive purposes"
        }
    
    async def _handle_post_exploitation(self, **kwargs) -> Dict[str, Any]:
        """Handle post-exploitation analysis stage (theoretical)"""
        target = kwargs["target"]
        previous_results = kwargs["previous_results"]
        
        post_exploit_prompt = f"""
        Analyze theoretical post-exploitation scenarios for {target} (DEFENSIVE ANALYSIS):
        
        Based on previous analysis, consider:
        1. Potential persistence mechanisms
        2. Lateral movement possibilities
        3. Data exfiltration risks
        4. Privilege escalation paths
        5. Detection and response considerations
        
        Focus on defensive countermeasures and detection strategies.
        """
        
        ai_response = await self.hf_orchestrator.security_analysis(post_exploit_prompt)
        
        return {
            "findings": {
                "post_exploitation_analysis": ai_response.content,
                "target": target,
                "defensive_focus": True
            },
            "confidence": ai_response.confidence,
            "ai_reasoning": "AI analyzed post-exploitation scenarios from defensive perspective"
        }
    
    async def _handle_risk_analysis(self, **kwargs) -> Dict[str, Any]:
        """Handle risk analysis stage"""
        target = kwargs["target"]
        previous_results = kwargs["previous_results"]
        
        # Compile all findings for risk assessment
        all_findings = self._compile_all_findings(previous_results)
        
        risk_prompt = f"""
        Perform comprehensive risk analysis for {target}:
        
        All findings from previous stages: {json.dumps(all_findings, indent=2)}
        
        Assess:
        1. Risk levels (Critical, High, Medium, Low)
        2. Business impact analysis
        3. Likelihood assessments
        4. Risk prioritization matrix
        5. Compliance implications
        6. Overall security posture
        
        Provide quantitative risk scores where possible.
        """
        
        ai_response = await self.hf_orchestrator.security_analysis(risk_prompt)
        
        return {
            "findings": {
                "risk_assessment": ai_response.content,
                "target": target,
                "comprehensive_analysis": True,
                "integrated_findings": len(all_findings)
            },
            "confidence": ai_response.confidence,
            "ai_reasoning": "AI performed comprehensive risk analysis integrating all workflow findings"
        }
    
    async def _handle_mitigation_planning(self, **kwargs) -> Dict[str, Any]:
        """Handle mitigation planning stage"""
        target = kwargs["target"]
        previous_results = kwargs["previous_results"]
        
        # Extract risk analysis
        risk_findings = {}
        if WorkflowStage.RISK_ANALYSIS in previous_results:
            risk_findings = previous_results[WorkflowStage.RISK_ANALYSIS].findings
        
        mitigation_prompt = f"""
        Develop comprehensive mitigation plan for {target}:
        
        Risk analysis results: {json.dumps(risk_findings, indent=2)}
        
        Create mitigation plan including:
        1. Immediate actions (Critical/High risks)
        2. Short-term remediation (30 days)
        3. Long-term improvements (90+ days)
        4. Preventive measures
        5. Detection and monitoring recommendations
        6. Implementation priorities and costs
        7. Success metrics
        
        Focus on practical, implementable solutions.
        """
        
        ai_response = await self.hf_orchestrator.security_analysis(mitigation_prompt)
        
        return {
            "findings": {
                "mitigation_plan": ai_response.content,
                "target": target,
                "actionable_recommendations": True,
                "timeline_included": True
            },
            "confidence": ai_response.confidence,
            "ai_reasoning": "AI developed comprehensive mitigation plan based on risk analysis"
        }
    
    async def _handle_report_generation(self, **kwargs) -> Dict[str, Any]:
        """Handle report generation stage"""
        workflow_id = kwargs["workflow_id"]
        target = kwargs["target"]
        previous_results = kwargs["previous_results"]
        
        # Generate comprehensive report using security report tool
        try:
            from ..tools.smolagents_security_tools import SecurityReportGeneratorTool
            
            report_tool = SecurityReportGeneratorTool()
            
            # Convert stage results to format expected by report tool
            scan_results = []
            for stage, result in previous_results.items():
                if result.status == WorkflowStatus.COMPLETED:
                    scan_results.append({
                        "tool_name": f"workflow_stage_{stage.value}",
                        "target": target,
                        "success": True,
                        "findings": result.findings
                    })
            
            # Generate technical report
            technical_report = report_tool(scan_results, "technical")
            
            # Generate executive report
            executive_report = report_tool(scan_results, "executive")
            
            return {
                "findings": {
                    "technical_report": technical_report["findings"],
                    "executive_report": executive_report["findings"],
                    "target": target,
                    "workflow_id": workflow_id,
                    "report_generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "artifacts": ["technical_report.json", "executive_report.json"],
                "confidence": 0.9,
                "ai_reasoning": "Comprehensive reports generated from all workflow stages"
            }
            
        except Exception as e:
            # Fallback to AI-generated report
            self.logger.warning(f"Report tool failed, using AI fallback: {e}")
            
            all_findings = self._compile_all_findings(previous_results)
            
            report_prompt = f"""
            Generate comprehensive security assessment report for {target}:
            
            All workflow findings: {json.dumps(all_findings, indent=2)}
            
            Create both:
            1. Executive summary (business-focused)
            2. Technical report (detailed findings)
            
            Include risk ratings, recommendations, and next steps.
            """
            
            ai_response = await self.hf_orchestrator.security_analysis(report_prompt)
            
            return {
                "findings": {
                    "generated_report": ai_response.content,
                    "target": target,
                    "report_type": "ai_generated"
                },
                "confidence": ai_response.confidence,
                "ai_reasoning": "AI generated comprehensive report from workflow findings"
            }
    
    async def _handle_compliance_checking(self, **kwargs) -> Dict[str, Any]:
        """Handle compliance checking stage"""
        target = kwargs["target"]
        previous_results = kwargs["previous_results"]
        
        # Extract vulnerability findings for compliance mapping
        vuln_context = self._extract_vulnerabilities_from_previous_stages(previous_results)
        
        compliance_prompt = f"""
        Perform compliance assessment for {target}:
        
        Security findings: {json.dumps(vuln_context, indent=2)}
        
        Check compliance against:
        1. OWASP Top 10
        2. NIST Cybersecurity Framework
        3. ISO 27001
        4. PCI DSS (if applicable)
        5. SOC 2 (if applicable)
        
        Provide:
        - Compliance status for each framework
        - Gap analysis
        - Remediation priorities
        - Compliance timeline
        """
        
        ai_response = await self.hf_orchestrator.security_analysis(compliance_prompt)
        
        return {
            "findings": {
                "compliance_assessment": ai_response.content,
                "target": target,
                "frameworks_assessed": ["OWASP", "NIST", "ISO27001", "PCI_DSS", "SOC2"],
                "gap_analysis_included": True
            },
            "confidence": ai_response.confidence,
            "ai_reasoning": "AI performed compliance assessment against multiple frameworks"
        }
    
    def _extract_context_from_previous_stages(self, previous_results: Dict) -> Dict[str, Any]:
        """Extract relevant context from previous stage results"""
        context = {}
        
        for stage, result in previous_results.items():
            if result.status == WorkflowStatus.COMPLETED and result.findings:
                context[stage.value] = {
                    "key_findings": result.findings,
                    "confidence": result.confidence,
                    "artifacts": result.artifacts
                }
        
        return context
    
    def _extract_vulnerabilities_from_previous_stages(self, previous_results: Dict) -> Dict[str, Any]:
        """Extract vulnerability-specific information"""
        vulnerabilities = {}
        
        for stage, result in previous_results.items():
            if result.status == WorkflowStatus.COMPLETED and result.findings:
                # Look for vulnerability-related findings
                findings = result.findings
                if any(key in str(findings).lower() for key in ['vuln', 'cve', 'exploit', 'risk']):
                    vulnerabilities[stage.value] = findings
        
        return vulnerabilities
    
    def _compile_all_findings(self, previous_results: Dict) -> Dict[str, Any]:
        """Compile all findings from previous stages"""
        all_findings = {}
        
        for stage, result in previous_results.items():
            if result.status == WorkflowStatus.COMPLETED:
                all_findings[stage.value] = {
                    "findings": result.findings,
                    "duration": result.duration,
                    "confidence": result.confidence,
                    "ai_reasoning": result.ai_reasoning
                }
        
        return all_findings
    
    async def _correlate_workflow_results(self, workflow_id: str):
        """Correlate results across all workflow stages"""
        workflow_state = self.active_workflows[workflow_id]
        
        correlation_results = await self.correlation_engine.correlate_findings(
            workflow_state["stage_results"]
        )
        
        workflow_state["correlation_results"] = correlation_results
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow"""
        if workflow_id not in self.active_workflows:
            return {"error": "Workflow not found"}
        
        workflow_state = self.active_workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "status": workflow_state["status"].value,
            "target": workflow_state["target"],
            "workflow_name": workflow_state["definition"].name,
            "start_time": workflow_state["start_time"],
            "current_stage": workflow_state["current_stage"].value if workflow_state.get("current_stage") else None,
            "completed_stages": len(workflow_state["completed_stages"]),
            "total_stages": len(workflow_state["definition"].stages),
            "failed_stages": len(workflow_state["failed_stages"]),
            "progress_percentage": (len(workflow_state["completed_stages"]) / len(workflow_state["definition"].stages)) * 100
        }
    
    def list_available_workflows(self) -> List[Dict[str, Any]]:
        """List all available workflow definitions"""
        return [
            {
                "name": def_.name,
                "type": def_.workflow_type.value,
                "description": def_.description,
                "estimated_duration": def_.estimated_duration,
                "risk_level": def_.risk_level,
                "stages": [stage.value for stage in def_.stages]
            }
            for def_ in self.workflow_definitions.values()
        ]

class WorkflowCorrelationEngine:
    """Engine for correlating findings across workflow stages"""
    
    def __init__(self):
        self.correlation_rules = self._load_correlation_rules()
    
    def _load_correlation_rules(self) -> Dict[str, Any]:
        """Load correlation rules for finding analysis"""
        return {
            "vulnerability_exploit_correlation": {
                "description": "Correlate vulnerabilities with exploitation potential",
                "stages": [WorkflowStage.VULNERABILITY_ASSESSMENT, WorkflowStage.EXPLOITATION_ANALYSIS]
            },
            "threat_risk_correlation": {
                "description": "Correlate threat models with risk assessments",
                "stages": [WorkflowStage.THREAT_MODELING, WorkflowStage.RISK_ANALYSIS]
            },
            "recon_vuln_correlation": {
                "description": "Correlate reconnaissance findings with vulnerabilities",
                "stages": [WorkflowStage.RECONNAISSANCE, WorkflowStage.VULNERABILITY_ASSESSMENT]
            }
        }
    
    async def correlate_findings(self, stage_results: Dict[WorkflowStage, StageResult]) -> Dict[str, Any]:
        """Correlate findings across all stages"""
        correlations = {}
        
        # Apply each correlation rule
        for rule_name, rule_config in self.correlation_rules.items():
            required_stages = rule_config["stages"]
            
            # Check if all required stages are present and completed
            if all(stage in stage_results and stage_results[stage].status == WorkflowStatus.COMPLETED 
                   for stage in required_stages):
                
                correlation_result = await self._apply_correlation_rule(
                    rule_name, rule_config, stage_results
                )
                
                correlations[rule_name] = correlation_result
        
        return correlations
    
    async def _apply_correlation_rule(self, 
                                    rule_name: str, 
                                    rule_config: Dict[str, Any], 
                                    stage_results: Dict[WorkflowStage, StageResult]) -> Dict[str, Any]:
        """Apply a specific correlation rule"""
        
        # Extract relevant findings from required stages
        relevant_findings = {}
        for stage in rule_config["stages"]:
            if stage in stage_results:
                relevant_findings[stage.value] = stage_results[stage].findings
        
        # Perform correlation analysis (simplified)
        correlation_score = 0.0
        correlations_found = []
        
        # Example correlation logic
        if "vulnerability" in rule_name.lower():
            # Look for vulnerability-exploitation correlations
            vuln_count = self._count_findings_by_type(relevant_findings, "vulnerability")
            exploit_count = self._count_findings_by_type(relevant_findings, "exploit")
            
            if vuln_count > 0 and exploit_count > 0:
                correlation_score = min(vuln_count, exploit_count) / max(vuln_count, exploit_count)
                correlations_found.append(f"Found {min(vuln_count, exploit_count)} correlated vulnerability-exploit pairs")
        
        return {
            "rule_applied": rule_name,
            "correlation_score": correlation_score,
            "correlations_found": correlations_found,
            "analysis_timestamp": time.time()
        }
    
    def _count_findings_by_type(self, findings: Dict[str, Any], finding_type: str) -> int:
        """Count findings of a specific type"""
        count = 0
        findings_str = json.dumps(findings).lower()
        
        # Simple keyword matching (would be more sophisticated in production)
        keywords = {
            "vulnerability": ["vuln", "cve", "weakness", "flaw"],
            "exploit": ["exploit", "attack", "payload", "proof-of-concept"],
            "threat": ["threat", "risk", "danger", "malicious"],
            "compliance": ["compliance", "standard", "regulation", "policy"]
        }
        
        if finding_type in keywords:
            for keyword in keywords[finding_type]:
                count += findings_str.count(keyword)
        
        return count

# Factory function
def create_workflow_orchestrator(hf_orchestrator: HuggingFaceAIOrchestrator) -> WorkflowOrchestrator:
    """Create and return workflow orchestrator"""
    return WorkflowOrchestrator(hf_orchestrator)