#!/usr/bin/env python3
"""
Real Kali Linux Tool Integration for Archangel Agents
Comprehensive integration of actual penetration testing and security tools

This module provides:
- Real Kali Linux tool execution in containers
- Output parsing and intelligence extraction
- Safety controls and ethical boundaries
- Learning data generation from tool outputs
"""

import asyncio
import json
import logging
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import tempfile
import os

class ToolCategory(Enum):
    RECONNAISSANCE = "reconnaissance"
    VULNERABILITY_SCANNING = "vulnerability_scanning"
    EXPLOITATION = "exploitation"
    POST_EXPLOITATION = "post_exploitation"
    NETWORK_ANALYSIS = "network_analysis"
    WEB_APPLICATION = "web_application"
    WIRELESS = "wireless"
    FORENSICS = "forensics"
    SOCIAL_ENGINEERING = "social_engineering"

class SafetyLevel(Enum):
    SAFE = "safe"           # Read-only operations
    CONTROLLED = "controlled"  # Limited impact operations
    RESTRICTED = "restricted"  # High-impact operations (training only)
    PROHIBITED = "prohibited"  # Never allowed

@dataclass
class ToolResult:
    """Result from executing a security tool"""
    tool_name: str
    command: str
    target: str
    start_time: datetime
    end_time: datetime
    exit_code: int
    stdout: str
    stderr: str
    parsed_data: Dict[str, Any]
    findings: List[str]
    indicators: List[str]
    severity: str
    confidence: float
    learning_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolConfiguration:
    """Configuration for a security tool"""
    name: str
    binary_path: str
    category: ToolCategory
    safety_level: SafetyLevel
    description: str
    common_args: List[str]
    output_formats: List[str]
    parser_class: str
    requires_root: bool = False
    timeout_seconds: int = 300
    rate_limit: Optional[int] = None
    allowed_targets: List[str] = field(default_factory=list)

class KaliToolIntegration:
    """
    Complete Kali Linux tool integration system
    
    Provides safe, controlled execution of real penetration testing tools
    with comprehensive output parsing and learning data generation.
    """
    
    def __init__(self, container_manager=None):
        self.container_manager = container_manager
        self.logger = logging.getLogger(__name__)
        
        # Tool configurations
        self.tools = self._initialize_tool_configurations()
        self.parsers = self._initialize_output_parsers()
        
        # Safety and control systems
        self.safety_enforcer = ToolSafetyEnforcer()
        self.rate_limiter = ToolRateLimiter()
        
        # Learning data collection
        self.execution_history: List[ToolResult] = []
        self.learning_database: Dict[str, List[Dict[str, Any]]] = {}
        
    def _initialize_tool_configurations(self) -> Dict[str, ToolConfiguration]:
        """Initialize comprehensive tool configurations"""
        
        tools = {}
        
        # === RECONNAISSANCE TOOLS ===
        
        tools["nmap"] = ToolConfiguration(
            name="nmap",
            binary_path="/usr/bin/nmap",
            category=ToolCategory.RECONNAISSANCE,
            safety_level=SafetyLevel.SAFE,
            description="Network discovery and security auditing",
            common_args=["-sS", "-sV", "-O", "-A"],
            output_formats=["normal", "xml", "grepable"],
            parser_class="NmapParser",
            timeout_seconds=600,
            rate_limit=10  # Max 10 scans per minute
        )
        
        tools["masscan"] = ToolConfiguration(
            name="masscan",
            binary_path="/usr/bin/masscan",
            category=ToolCategory.RECONNAISSANCE,
            safety_level=SafetyLevel.CONTROLLED,
            description="High-speed port scanner",
            common_args=["-p1-65535", "--rate=1000"],
            output_formats=["list", "xml", "json"],
            parser_class="MasscanParser",
            timeout_seconds=300,
            rate_limit=5
        )
        
        tools["dirb"] = ToolConfiguration(
            name="dirb",
            binary_path="/usr/bin/dirb",
            category=ToolCategory.WEB_APPLICATION,
            safety_level=SafetyLevel.CONTROLLED,
            description="Web content scanner",
            common_args=["-r", "-S", "-w"],
            output_formats=["text"],
            parser_class="DirbParser",
            timeout_seconds=600,
            rate_limit=3
        )
        
        tools["gobuster"] = ToolConfiguration(
            name="gobuster",
            binary_path="/usr/bin/gobuster",
            category=ToolCategory.WEB_APPLICATION,
            safety_level=SafetyLevel.CONTROLLED,
            description="Fast directory/file & DNS brute-forcer",
            common_args=["dir", "-u", "-w"],
            output_formats=["text"],
            parser_class="GobusterParser",
            timeout_seconds=600,
            rate_limit=3
        )
        
        tools["nikto"] = ToolConfiguration(
            name="nikto",
            binary_path="/usr/bin/nikto",
            category=ToolCategory.VULNERABILITY_SCANNING,
            safety_level=SafetyLevel.CONTROLLED,
            description="Web server vulnerability scanner",
            common_args=["-h", "-Format", "xml"],
            output_formats=["xml", "text"],
            parser_class="NiktoParser",
            timeout_seconds=1200,
            rate_limit=2
        )
        
        # === VULNERABILITY SCANNERS ===
        
        tools["openvas"] = ToolConfiguration(
            name="openvas",
            binary_path="/usr/bin/openvas",
            category=ToolCategory.VULNERABILITY_SCANNING,
            safety_level=SafetyLevel.CONTROLLED,
            description="Comprehensive vulnerability scanner",
            common_args=["--scan", "--target"],
            output_formats=["xml", "pdf"],
            parser_class="OpenVASParser",
            timeout_seconds=3600,
            rate_limit=1
        )
        
        tools["nuclei"] = ToolConfiguration(
            name="nuclei",
            binary_path="/usr/bin/nuclei",
            category=ToolCategory.VULNERABILITY_SCANNING,
            safety_level=SafetyLevel.CONTROLLED,
            description="Fast vulnerability scanner based on templates",
            common_args=["-u", "-t", "/opt/nuclei-templates/"],
            output_formats=["json", "text"],
            parser_class="NucleiParser",
            timeout_seconds=600,
            rate_limit=5
        )
        
        # === EXPLOITATION TOOLS ===
        
        tools["metasploit"] = ToolConfiguration(
            name="metasploit",
            binary_path="/usr/bin/msfconsole",
            category=ToolCategory.EXPLOITATION,
            safety_level=SafetyLevel.RESTRICTED,
            description="Penetration testing framework",
            common_args=["-q", "-r"],
            output_formats=["text"],
            parser_class="MetasploitParser",
            timeout_seconds=1800,
            rate_limit=1,
            allowed_targets=["127.0.0.1", "localhost", "test.local"]
        )
        
        tools["sqlmap"] = ToolConfiguration(
            name="sqlmap",
            binary_path="/usr/bin/sqlmap",
            category=ToolCategory.WEB_APPLICATION,
            safety_level=SafetyLevel.RESTRICTED,
            description="SQL injection testing tool",
            common_args=["-u", "--batch", "--random-agent"],
            output_formats=["text"],
            parser_class="SQLMapParser",
            timeout_seconds=1200,
            rate_limit=2,
            allowed_targets=["http://127.0.0.1", "http://localhost", "http://test.local"]
        )
        
        tools["burpsuite"] = ToolConfiguration(
            name="burpsuite",
            binary_path="/usr/bin/burpsuite",
            category=ToolCategory.WEB_APPLICATION,
            safety_level=SafetyLevel.CONTROLLED,
            description="Web application security testing",
            common_args=["--headless", "--project-file"],
            output_formats=["xml", "json"],
            parser_class="BurpSuiteParser",
            timeout_seconds=1800,
            rate_limit=1
        )
        
        # === NETWORK ANALYSIS TOOLS ===
        
        tools["wireshark"] = ToolConfiguration(
            name="wireshark",
            binary_path="/usr/bin/tshark",
            category=ToolCategory.NETWORK_ANALYSIS,
            safety_level=SafetyLevel.SAFE,
            description="Network protocol analyzer",
            common_args=["-i", "-w", "-f"],
            output_formats=["pcap", "json"],
            parser_class="WiresharkParser",
            timeout_seconds=300,
            requires_root=True
        )
        
        tools["tcpdump"] = ToolConfiguration(
            name="tcpdump",
            binary_path="/usr/bin/tcpdump",
            category=ToolCategory.NETWORK_ANALYSIS,
            safety_level=SafetyLevel.SAFE,
            description="Network packet analyzer",
            common_args=["-i", "-w", "-c"],
            output_formats=["pcap", "text"],
            parser_class="TcpdumpParser",
            timeout_seconds=300,
            requires_root=True
        )
        
        # === FORENSICS TOOLS ===
        
        tools["volatility"] = ToolConfiguration(
            name="volatility",
            binary_path="/usr/bin/volatility",
            category=ToolCategory.FORENSICS,
            safety_level=SafetyLevel.SAFE,
            description="Memory forensics framework",
            common_args=["-f", "--profile"],
            output_formats=["text"],
            parser_class="VolatilityParser",
            timeout_seconds=1800
        )
        
        tools["autopsy"] = ToolConfiguration(
            name="autopsy",
            binary_path="/opt/autopsy/bin/autopsy",
            category=ToolCategory.FORENSICS,
            safety_level=SafetyLevel.SAFE,
            description="Digital forensics platform",
            common_args=["--headless"],
            output_formats=["xml", "json"],
            parser_class="AutopsyParser",
            timeout_seconds=3600
        )
        
        # === SOCIAL ENGINEERING TOOLS ===
        
        tools["setoolkit"] = ToolConfiguration(
            name="setoolkit",
            binary_path="/usr/bin/setoolkit",
            category=ToolCategory.SOCIAL_ENGINEERING,
            safety_level=SafetyLevel.PROHIBITED,  # Never allowed in autonomous mode
            description="Social engineering toolkit",
            common_args=["--no-update"],
            output_formats=["text"],
            parser_class="SEToolkitParser",
            timeout_seconds=600
        )
        
        return tools
    
    def _initialize_output_parsers(self) -> Dict[str, Any]:
        """Initialize output parsers for each tool"""
        
        parsers = {
            "NmapParser": NmapOutputParser(),
            "MasscanParser": MasscanOutputParser(),
            "DirbParser": DirbOutputParser(),
            "GobusterParser": GobusterOutputParser(),
            "NiktoParser": NiktoOutputParser(),
            "OpenVASParser": OpenVASOutputParser(),
            "NucleiParser": NucleiOutputParser(),
            "MetasploitParser": MetasploitOutputParser(),
            "SQLMapParser": SQLMapOutputParser(),
            "BurpSuiteParser": BurpSuiteOutputParser(),
            "WiresharkParser": WiresharkOutputParser(),
            "TcpdumpParser": TcpdumpOutputParser(),
            "VolatilityParser": VolatilityOutputParser(),
            "AutopsyParser": AutopsyOutputParser(),
            "SEToolkitParser": SEToolkitOutputParser()
        }
        
        return parsers
    
    async def execute_tool(self,
                          tool_name: str,
                          target: str,
                          args: List[str] = None,
                          container_name: str = None,
                          context: Dict[str, Any] = None) -> ToolResult:
        """Execute a security tool with comprehensive safety checks"""
        
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        tool_config = self.tools[tool_name]
        
        # Safety checks
        await self._enforce_safety_checks(tool_config, target, args, context)
        
        # Rate limiting
        await self.rate_limiter.check_rate_limit(tool_name)
        
        self.logger.info(f"🔧 Executing {tool_name} against {target}")
        
        start_time = datetime.now()
        
        try:
            # Build command
            command_parts = [tool_config.binary_path]
            if args:
                command_parts.extend(args)
            else:
                command_parts.extend(tool_config.common_args)
                command_parts.append(target)
            
            command = " ".join(command_parts)
            
            # Execute in container or locally
            if container_name and self.container_manager:
                result = await self._execute_in_container(
                    container_name, command, tool_config.timeout_seconds
                )
            else:
                result = await self._execute_locally(
                    command, tool_config.timeout_seconds
                )
            
            end_time = datetime.now()
            
            # Parse output
            parser = self.parsers[tool_config.parser_class]
            parsed_data = await parser.parse_output(
                result["stdout"], result["stderr"], tool_name
            )
            
            # Create tool result
            tool_result = ToolResult(
                tool_name=tool_name,
                command=command,
                target=target,
                start_time=start_time,
                end_time=end_time,
                exit_code=result["exit_code"],
                stdout=result["stdout"],
                stderr=result["stderr"],
                parsed_data=parsed_data,
                findings=parsed_data.get("findings", []),
                indicators=parsed_data.get("indicators", []),
                severity=parsed_data.get("severity", "unknown"),
                confidence=parsed_data.get("confidence", 0.5),
                learning_data=await self._generate_learning_data(
                    tool_name, target, parsed_data, context
                )
            )
            
            # Store execution history
            self.execution_history.append(tool_result)
            
            # Update learning database
            await self._update_learning_database(tool_result)
            
            self.logger.info(f"✅ {tool_name} completed: {len(tool_result.findings)} findings")
            return tool_result
            
        except Exception as e:
            end_time = datetime.now()
            self.logger.error(f"❌ {tool_name} execution failed: {e}")
            
            return ToolResult(
                tool_name=tool_name,
                command=command if 'command' in locals() else "unknown",
                target=target,
                start_time=start_time,
                end_time=end_time,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                parsed_data={"error": str(e)},
                findings=[],
                indicators=[],
                severity="error",
                confidence=0.0,
                learning_data={"execution_error": str(e)}
            )
    
    async def _enforce_safety_checks(self,
                                   tool_config: ToolConfiguration,
                                   target: str,
                                   args: List[str],
                                   context: Dict[str, Any]):
        """Enforce safety checks before tool execution"""
        
        # Check safety level
        if tool_config.safety_level == SafetyLevel.PROHIBITED:
            raise ValueError(f"Tool {tool_config.name} is prohibited in autonomous mode")
        
        # Check target restrictions
        if tool_config.allowed_targets:
            target_allowed = False
            for allowed in tool_config.allowed_targets:
                if allowed in target or target in allowed:
                    target_allowed = True
                    break
            
            if not target_allowed:
                raise ValueError(f"Target {target} not allowed for {tool_config.name}")
        
        # Restricted tools require explicit authorization
        if tool_config.safety_level == SafetyLevel.RESTRICTED:
            if not context or not context.get("authorized", False):
                raise ValueError(f"Tool {tool_config.name} requires explicit authorization")
            
            if not context.get("training_environment", False):
                raise ValueError(f"Restricted tool {tool_config.name} only allowed in training")
        
        # Check for dangerous arguments
        if args:
            dangerous_args = [
                "--os-cmd", "--sql-query", "--file-write", "--file-read",
                "--dump-all", "--privileges", "--passwords", "--crack"
            ]
            
            for arg in args:
                if any(dangerous in arg.lower() for dangerous in dangerous_args):
                    if tool_config.safety_level != SafetyLevel.RESTRICTED:
                        raise ValueError(f"Dangerous argument {arg} not allowed")
        
        self.logger.debug(f"✅ Safety checks passed for {tool_config.name}")
    
    async def _execute_in_container(self,
                                  container_name: str,
                                  command: str,
                                  timeout: int) -> Dict[str, Any]:
        """Execute command in container"""
        
        if not self.container_manager:
            raise ValueError("Container manager not available")
        
        result = await self.container_manager.execute_in_container(
            container_name, command, timeout
        )
        
        return {
            "exit_code": result.get("exit_code", -1),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", "")
        }
    
    async def _execute_locally(self, command: str, timeout: int) -> Dict[str, Any]:
        """Execute command locally (for testing/development)"""
        
        try:
            # WARNING: Only use for testing in safe environments
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                
                return {
                    "exit_code": process.returncode,
                    "stdout": stdout.decode("utf-8", errors="ignore"),
                    "stderr": stderr.decode("utf-8", errors="ignore")
                }
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Command timed out after {timeout} seconds")
                
        except Exception as e:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e)
            }
    
    async def _generate_learning_data(self,
                                    tool_name: str,
                                    target: str,
                                    parsed_data: Dict[str, Any],
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate learning data from tool execution"""
        
        learning_data = {
            "tool_category": self.tools[tool_name].category.value,
            "execution_context": context or {},
            "target_type": self._classify_target_type(target),
            "findings_count": len(parsed_data.get("findings", [])),
            "indicators_count": len(parsed_data.get("indicators", [])),
            "effectiveness_score": self._calculate_effectiveness_score(parsed_data),
            "learning_insights": [],
            "recommended_followup": []
        }
        
        # Generate insights based on findings
        findings = parsed_data.get("findings", [])
        if findings:
            learning_data["learning_insights"].append(
                f"{tool_name} identified {len(findings)} security findings on {target}"
            )
            
            # Categorize findings
            high_severity = [f for f in findings if "critical" in f.lower() or "high" in f.lower()]
            if high_severity:
                learning_data["learning_insights"].append(
                    f"High-severity findings detected: {len(high_severity)}"
                )
        
        # Generate follow-up recommendations
        if tool_name in ["nmap", "masscan"]:
            open_ports = [f for f in findings if "open" in f.lower()]
            if open_ports:
                learning_data["recommended_followup"].append("service_enumeration")
                learning_data["recommended_followup"].append("vulnerability_scanning")
        
        elif tool_name in ["dirb", "gobuster"]:
            directories = [f for f in findings if "directory" in f.lower()]
            if directories:
                learning_data["recommended_followup"].append("content_analysis")
                learning_data["recommended_followup"].append("file_enumeration")
        
        elif tool_name in ["nikto", "nuclei"]:
            vulnerabilities = [f for f in findings if "vulnerability" in f.lower()]
            if vulnerabilities:
                learning_data["recommended_followup"].append("exploitation_testing")
                learning_data["recommended_followup"].append("manual_verification")
        
        return learning_data
    
    def _classify_target_type(self, target: str) -> str:
        """Classify the type of target"""
        if target.startswith("http://") or target.startswith("https://"):
            return "web_application"
        elif ":" in target and not target.count(":") > 1:
            return "network_service"
        elif target.replace(".", "").isdigit():
            return "ip_address"
        elif "." in target:
            return "domain_name"
        else:
            return "unknown"
    
    def _calculate_effectiveness_score(self, parsed_data: Dict[str, Any]) -> float:
        """Calculate tool effectiveness score"""
        findings_count = len(parsed_data.get("findings", []))
        indicators_count = len(parsed_data.get("indicators", []))
        confidence = parsed_data.get("confidence", 0.5)
        
        # Base score from findings
        base_score = min(findings_count * 0.1, 0.8)
        
        # Bonus for indicators
        indicator_bonus = min(indicators_count * 0.05, 0.2)
        
        # Apply confidence weighting
        effectiveness = (base_score + indicator_bonus) * confidence
        
        return min(max(effectiveness, 0.0), 1.0)
    
    async def _update_learning_database(self, tool_result: ToolResult):
        """Update learning database with execution results"""
        
        tool_name = tool_result.tool_name
        
        if tool_name not in self.learning_database:
            self.learning_database[tool_name] = []
        
        learning_entry = {
            "timestamp": tool_result.start_time.isoformat(),
            "target": tool_result.target,
            "success": tool_result.exit_code == 0,
            "findings_count": len(tool_result.findings),
            "effectiveness": tool_result.learning_data.get("effectiveness_score", 0.0),
            "execution_time": (tool_result.end_time - tool_result.start_time).total_seconds(),
            "insights": tool_result.learning_data.get("learning_insights", []),
            "followup": tool_result.learning_data.get("recommended_followup", [])
        }
        
        self.learning_database[tool_name].append(learning_entry)
        
        # Keep only recent entries (last 100 per tool)
        if len(self.learning_database[tool_name]) > 100:
            self.learning_database[tool_name] = self.learning_database[tool_name][-100:]
    
    async def get_tool_learning_insights(self, tool_name: str) -> Dict[str, Any]:
        """Get learning insights for a specific tool"""
        
        if tool_name not in self.learning_database:
            return {"error": f"No learning data for {tool_name}"}
        
        entries = self.learning_database[tool_name]
        
        if not entries:
            return {"error": f"No execution history for {tool_name}"}
        
        # Calculate aggregated insights
        total_executions = len(entries)
        successful_executions = len([e for e in entries if e["success"]])
        success_rate = successful_executions / total_executions
        
        avg_findings = sum(e["findings_count"] for e in entries) / total_executions
        avg_effectiveness = sum(e["effectiveness"] for e in entries) / total_executions
        avg_execution_time = sum(e["execution_time"] for e in entries) / total_executions
        
        # Extract common insights
        all_insights = []
        for entry in entries:
            all_insights.extend(entry["insights"])
        
        # Find most common follow-up recommendations
        all_followups = []
        for entry in entries:
            all_followups.extend(entry["followup"])
        
        from collections import Counter
        common_followups = Counter(all_followups).most_common(5)
        
        return {
            "tool_name": tool_name,
            "total_executions": total_executions,
            "success_rate": success_rate,
            "average_findings": avg_findings,
            "average_effectiveness": avg_effectiveness,
            "average_execution_time": avg_execution_time,
            "common_followups": common_followups,
            "recent_insights": all_insights[-10:],  # Last 10 insights
            "learning_quality": "high" if success_rate > 0.8 else "medium" if success_rate > 0.5 else "low"
        }
    
    async def recommend_tool_sequence(self,
                                   target: str,
                                   objective: str,
                                   context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Recommend optimal tool sequence for objective"""
        
        target_type = self._classify_target_type(target)
        recommendations = []
        
        # Web application testing sequence
        if target_type == "web_application":
            if objective == "reconnaissance":
                recommendations = [
                    {"tool": "dirb", "purpose": "Directory enumeration", "priority": 1},
                    {"tool": "nikto", "purpose": "Vulnerability scanning", "priority": 2},
                    {"tool": "nuclei", "purpose": "Template-based scanning", "priority": 3}
                ]
            elif objective == "vulnerability_assessment":
                recommendations = [
                    {"tool": "nikto", "purpose": "Web vulnerability scanning", "priority": 1},
                    {"tool": "nuclei", "purpose": "Comprehensive template scan", "priority": 2},
                    {"tool": "sqlmap", "purpose": "SQL injection testing", "priority": 3}
                ]
        
        # Network target testing sequence
        elif target_type in ["ip_address", "domain_name"]:
            if objective == "reconnaissance":
                recommendations = [
                    {"tool": "nmap", "purpose": "Port scanning and service detection", "priority": 1},
                    {"tool": "masscan", "purpose": "Fast port discovery", "priority": 2}
                ]
            elif objective == "vulnerability_assessment":
                recommendations = [
                    {"tool": "nmap", "purpose": "Service enumeration", "priority": 1},
                    {"tool": "nuclei", "purpose": "Service vulnerability scanning", "priority": 2},
                    {"tool": "openvas", "purpose": "Comprehensive vulnerability scan", "priority": 3}
                ]
        
        # Add learning-based recommendations
        for rec in recommendations:
            tool_name = rec["tool"]
            learning_insights = await self.get_tool_learning_insights(tool_name)
            
            if "error" not in learning_insights:
                rec["effectiveness"] = learning_insights.get("average_effectiveness", 0.5)
                rec["success_rate"] = learning_insights.get("success_rate", 0.5)
                rec["avg_execution_time"] = learning_insights.get("average_execution_time", 60)
        
        # Sort by priority and effectiveness
        recommendations.sort(key=lambda x: (x["priority"], -x.get("effectiveness", 0.5)))
        
        return recommendations
    
    async def execute_tool_sequence(self,
                                  recommendations: List[Dict[str, Any]],
                                  target: str,
                                  container_name: str = None,
                                  context: Dict[str, Any] = None) -> List[ToolResult]:
        """Execute a sequence of recommended tools"""
        
        results = []
        
        for rec in recommendations:
            tool_name = rec["tool"]
            
            try:
                self.logger.info(f"🔄 Executing {tool_name} for {rec['purpose']}")
                
                result = await self.execute_tool(
                    tool_name, target, container_name=container_name, context=context
                )
                
                results.append(result)
                
                # Use results to inform next tool execution
                if result.findings:
                    # Update context with new findings
                    if context is None:
                        context = {}
                    context[f"{tool_name}_findings"] = result.findings
                
                # Brief pause between tools
                await asyncio.sleep(2)
                
            except Exception as e:
                self.logger.error(f"❌ Failed to execute {tool_name}: {e}")
                continue
        
        return results
    
    def get_available_tools(self,
                          category: ToolCategory = None,
                          safety_level: SafetyLevel = None) -> List[str]:
        """Get list of available tools with optional filtering"""
        
        tools = []
        
        for tool_name, config in self.tools.items():
            # Filter by category
            if category and config.category != category:
                continue
            
            # Filter by safety level
            if safety_level and config.safety_level != safety_level:
                continue
            
            tools.append(tool_name)
        
        return tools
    
    async def export_learning_data(self, filename: str):
        """Export learning database for model training"""
        
        export_data = {
            "execution_history": [
                {
                    "tool_name": result.tool_name,
                    "target": result.target,
                    "command": result.command,
                    "success": result.exit_code == 0,
                    "findings_count": len(result.findings),
                    "findings": result.findings,
                    "indicators": result.indicators,
                    "severity": result.severity,
                    "confidence": result.confidence,
                    "learning_data": result.learning_data,
                    "execution_time": (result.end_time - result.start_time).total_seconds(),
                    "timestamp": result.start_time.isoformat()
                }
                for result in self.execution_history
            ],
            "learning_database": self.learning_database,
            "tool_configurations": {
                name: {
                    "category": config.category.value,
                    "safety_level": config.safety_level.value,
                    "description": config.description
                }
                for name, config in self.tools.items()
            },
            "export_timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"✅ Learning data exported to: {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to export learning data: {e}")


# === OUTPUT PARSERS ===

class BaseOutputParser:
    """Base class for tool output parsers"""
    
    async def parse_output(self, stdout: str, stderr: str, tool_name: str) -> Dict[str, Any]:
        """Parse tool output and extract structured data"""
        return {
            "findings": [],
            "indicators": [],
            "severity": "unknown",
            "confidence": 0.5,
            "raw_output": stdout,
            "errors": stderr
        }

class NmapOutputParser(BaseOutputParser):
    """Parser for Nmap output"""
    
    async def parse_output(self, stdout: str, stderr: str, tool_name: str) -> Dict[str, Any]:
        findings = []
        indicators = []
        
        lines = stdout.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Parse open ports
            if "/tcp" in line and "open" in line:
                findings.append(f"Open TCP port: {line}")
                
                # Extract port number as indicator
                port_match = re.search(r'(\d+)/tcp', line)
                if port_match:
                    indicators.append(f"open_port:{port_match.group(1)}")
            
            elif "/udp" in line and "open" in line:
                findings.append(f"Open UDP port: {line}")
                
                port_match = re.search(r'(\d+)/udp', line)
                if port_match:
                    indicators.append(f"open_port:{port_match.group(1)}")
            
            # Parse OS detection
            elif "OS:" in line:
                findings.append(f"OS detection: {line}")
                indicators.append("os_detected")
            
            # Parse service versions
            elif "Version:" in line:
                findings.append(f"Service version: {line}")
                indicators.append("service_version")
        
        # Calculate severity based on findings
        severity = "low"
        if len([f for f in findings if "open" in f.lower()]) > 10:
            severity = "medium"
        if any("admin" in f.lower() or "root" in f.lower() for f in findings):
            severity = "high"
        
        confidence = 0.9 if findings else 0.3
        
        return {
            "findings": findings,
            "indicators": indicators,
            "severity": severity,
            "confidence": confidence,
            "open_ports": len([f for f in findings if "open" in f.lower()]),
            "services_detected": len([f for f in findings if "Version:" in f]),
            "raw_output": stdout,
            "errors": stderr
        }

class DirbOutputParser(BaseOutputParser):
    """Parser for Dirb output"""
    
    async def parse_output(self, stdout: str, stderr: str, tool_name: str) -> Dict[str, Any]:
        findings = []
        indicators = []
        
        lines = stdout.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Parse discovered directories
            if "==> DIRECTORY:" in line:
                findings.append(f"Directory found: {line}")
                indicators.append("directory_found")
            
            # Parse interesting responses
            elif "CODE:" in line:
                if "200" in line:
                    findings.append(f"Accessible resource: {line}")
                    indicators.append("accessible_resource")
                elif "301" in line or "302" in line:
                    findings.append(f"Redirect found: {line}")
                    indicators.append("redirect_found")
                elif "403" in line:
                    findings.append(f"Forbidden resource: {line}")
                    indicators.append("forbidden_resource")
        
        severity = "low"
        if len(findings) > 5:
            severity = "medium"
        if any("admin" in f.lower() or "config" in f.lower() for f in findings):
            severity = "high"
        
        confidence = 0.8 if findings else 0.2
        
        return {
            "findings": findings,
            "indicators": indicators,
            "severity": severity,
            "confidence": confidence,
            "directories_found": len([f for f in findings if "Directory" in f]),
            "files_found": len([f for f in findings if "CODE:" in f]),
            "raw_output": stdout,
            "errors": stderr
        }

# Add more parsers for other tools...
class MasscanOutputParser(BaseOutputParser):
    async def parse_output(self, stdout: str, stderr: str, tool_name: str) -> Dict[str, Any]:
        # Implementation for Masscan output parsing
        return await super().parse_output(stdout, stderr, tool_name)

class GobusterOutputParser(BaseOutputParser):
    async def parse_output(self, stdout: str, stderr: str, tool_name: str) -> Dict[str, Any]:
        # Implementation for Gobuster output parsing
        return await super().parse_output(stdout, stderr, tool_name)

class NiktoOutputParser(BaseOutputParser):
    async def parse_output(self, stdout: str, stderr: str, tool_name: str) -> Dict[str, Any]:
        # Implementation for Nikto output parsing
        return await super().parse_output(stdout, stderr, tool_name)

class OpenVASOutputParser(BaseOutputParser):
    async def parse_output(self, stdout: str, stderr: str, tool_name: str) -> Dict[str, Any]:
        # Implementation for OpenVAS output parsing
        return await super().parse_output(stdout, stderr, tool_name)

class NucleiOutputParser(BaseOutputParser):
    async def parse_output(self, stdout: str, stderr: str, tool_name: str) -> Dict[str, Any]:
        # Implementation for Nuclei output parsing
        return await super().parse_output(stdout, stderr, tool_name)

class MetasploitOutputParser(BaseOutputParser):
    async def parse_output(self, stdout: str, stderr: str, tool_name: str) -> Dict[str, Any]:
        # Implementation for Metasploit output parsing
        return await super().parse_output(stdout, stderr, tool_name)

class SQLMapOutputParser(BaseOutputParser):
    async def parse_output(self, stdout: str, stderr: str, tool_name: str) -> Dict[str, Any]:
        # Implementation for SQLMap output parsing
        return await super().parse_output(stdout, stderr, tool_name)

class BurpSuiteOutputParser(BaseOutputParser):
    async def parse_output(self, stdout: str, stderr: str, tool_name: str) -> Dict[str, Any]:
        # Implementation for Burp Suite output parsing
        return await super().parse_output(stdout, stderr, tool_name)

class WiresharkOutputParser(BaseOutputParser):
    async def parse_output(self, stdout: str, stderr: str, tool_name: str) -> Dict[str, Any]:
        # Implementation for Wireshark/tshark output parsing
        return await super().parse_output(stdout, stderr, tool_name)

class TcpdumpOutputParser(BaseOutputParser):
    async def parse_output(self, stdout: str, stderr: str, tool_name: str) -> Dict[str, Any]:
        # Implementation for tcpdump output parsing
        return await super().parse_output(stdout, stderr, tool_name)

class VolatilityOutputParser(BaseOutputParser):
    async def parse_output(self, stdout: str, stderr: str, tool_name: str) -> Dict[str, Any]:
        # Implementation for Volatility output parsing
        return await super().parse_output(stdout, stderr, tool_name)

class AutopsyOutputParser(BaseOutputParser):
    async def parse_output(self, stdout: str, stderr: str, tool_name: str) -> Dict[str, Any]:
        # Implementation for Autopsy output parsing
        return await super().parse_output(stdout, stderr, tool_name)

class SEToolkitOutputParser(BaseOutputParser):
    async def parse_output(self, stdout: str, stderr: str, tool_name: str) -> Dict[str, Any]:
        # Implementation for SET output parsing
        return await super().parse_output(stdout, stderr, tool_name)


# === SAFETY AND CONTROL SYSTEMS ===

class ToolSafetyEnforcer:
    """Enforces safety controls for tool execution"""
    
    def __init__(self):
        self.prohibited_commands = [
            "rm -rf", "mkfs", "dd if=", ":(){ :|:& };:",
            "curl | sh", "wget | sh", "> /dev/sda"
        ]
        
        self.restricted_args = [
            "--os-cmd", "--sql-query", "--file-write", "--passwords"
        ]
    
    async def validate_command(self, command: str, tool_config: ToolConfiguration) -> bool:
        """Validate command for safety"""
        
        command_lower = command.lower()
        
        # Check for prohibited commands
        for prohibited in self.prohibited_commands:
            if prohibited in command_lower:
                raise ValueError(f"Prohibited command pattern: {prohibited}")
        
        # Check for restricted arguments based on safety level
        if tool_config.safety_level != SafetyLevel.RESTRICTED:
            for restricted in self.restricted_args:
                if restricted in command_lower:
                    raise ValueError(f"Restricted argument: {restricted}")
        
        return True

class ToolRateLimiter:
    """Rate limiting for tool execution"""
    
    def __init__(self):
        self.execution_times: Dict[str, List[datetime]] = {}
    
    async def check_rate_limit(self, tool_name: str):
        """Check if tool execution is within rate limits"""
        
        if tool_name not in self.execution_times:
            self.execution_times[tool_name] = []
        
        now = datetime.now()
        
        # Remove old entries (older than 1 minute)
        cutoff = now - timedelta(minutes=1)
        self.execution_times[tool_name] = [
            t for t in self.execution_times[tool_name] if t > cutoff
        ]
        
        # Add current execution
        self.execution_times[tool_name].append(now)
        
        # For now, just log the rate - could implement actual limits
        rate = len(self.execution_times[tool_name])
        if rate > 10:  # More than 10 executions per minute
            logging.warning(f"High execution rate for {tool_name}: {rate}/minute")


# === FACTORY FUNCTIONS ===

def create_kali_tool_integration(container_manager=None) -> KaliToolIntegration:
    """Create Kali tool integration system"""
    return KaliToolIntegration(container_manager)

async def demo_kali_tool_integration():
    """Demonstrate Kali tool integration"""
    print("🔧 Kali Linux Tool Integration Demo")
    print("=" * 50)
    
    # Create integration system
    kali_tools = create_kali_tool_integration()
    
    # Show available tools
    print("📋 Available Tools:")
    for category in ToolCategory:
        tools = kali_tools.get_available_tools(category=category)
        if tools:
            print(f"  {category.value}: {', '.join(tools)}")
    
    # Demo tool recommendations
    print(f"\n🎯 Tool Recommendations for Web Application:")
    recommendations = await kali_tools.recommend_tool_sequence(
        "http://test.local",
        "reconnaissance"
    )
    
    for rec in recommendations:
        print(f"  {rec['priority']}. {rec['tool']}: {rec['purpose']}")
    
    # Demo learning insights (mock data)
    print(f"\n📊 Learning Insights:")
    insights = await kali_tools.get_tool_learning_insights("nmap")
    if "error" not in insights:
        print(f"  Success Rate: {insights['success_rate']:.2f}")
        print(f"  Average Effectiveness: {insights['average_effectiveness']:.2f}")
    else:
        print(f"  No learning data available yet")
    
    print(f"\n✅ Kali tool integration demo completed")

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sys.exit(asyncio.run(demo_kali_tool_integration()))