"""
Archangel MCP (Model Context Protocol) Integration Architecture
Secure MCP connections for autonomous red and blue team agents

This system provides:
- Isolated MCP servers for red/blue teams
- Secure external resource access
- Professional-grade SDK integrations
- Threat intelligence and vulnerability database connections
- Elite-level security tool integration
"""

import asyncio
import json
import logging
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import jwt
import aiohttp
import ssl
from contextlib import asynccontextmanager
import hashlib
import hmac

# MCP Protocol imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import Resource, Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Mock classes for development
    class ClientSession: pass
    class StdioServerParameters: pass
    class Resource: pass
    class Tool: pass
    class TextContent: pass

class TeamType(Enum):
    RED_TEAM = "red_team"
    BLUE_TEAM = "blue_team"
    NEUTRAL = "neutral"

class ResourceType(Enum):
    THREAT_INTELLIGENCE = "threat_intel"
    VULNERABILITY_DATABASE = "vuln_db"
    SECURITY_TOOLS = "security_tools"
    ATTACK_FRAMEWORKS = "attack_frameworks"
    DEFENSE_PLATFORMS = "defense_platforms"
    OSINT_SOURCES = "osint"
    MALWARE_ANALYSIS = "malware_analysis"
    FORENSICS_TOOLS = "forensics"

@dataclass
class MCPCredentials:
    """Secure credentials for external services"""
    service_name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    custom_headers: Dict[str, str] = field(default_factory=dict)
    expires_at: Optional[datetime] = None
    team_restricted: Optional[TeamType] = None

@dataclass
class ExternalResource:
    """External resource definition"""
    resource_id: str
    name: str
    resource_type: ResourceType
    endpoint_url: str
    credentials_id: str
    allowed_teams: List[TeamType]
    rate_limits: Dict[str, int] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    security_level: str = "standard"  # standard, restricted, classified
    description: str = ""

@dataclass
class SDKIntegration:
    """SDK integration configuration"""
    sdk_name: str
    sdk_version: str
    installation_method: str  # pip, docker, binary
    docker_image: Optional[str] = None
    binary_path: Optional[str] = None
    config_template: Dict[str, Any] = field(default_factory=dict)
    team_restrictions: List[TeamType] = field(default_factory=list)
    required_credentials: List[str] = field(default_factory=list)

class MCPSecurityManager:
    """Manages authentication, authorization, and security for MCP connections"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.credentials_store: Dict[str, MCPCredentials] = {}
        self.access_tokens: Dict[str, Dict[str, Any]] = {}
        self.rate_limiters: Dict[str, Dict[str, float]] = {}
        self.logger = logging.getLogger("mcp_security")
    
    def add_credentials(self, credentials: MCPCredentials) -> str:
        """Securely store credentials"""
        cred_id = f"cred_{uuid.uuid4().hex[:16]}"
        
        # Encrypt sensitive data
        encrypted_creds = self._encrypt_credentials(credentials)
        self.credentials_store[cred_id] = encrypted_creds
        
        self.logger.info(f"🔐 Stored credentials for {credentials.service_name}")
        return cred_id
    
    def _encrypt_credentials(self, credentials: MCPCredentials) -> MCPCredentials:
        """Encrypt sensitive credential fields"""
        # In production, use proper encryption (Fernet, etc.)
        # For demo, we'll hash sensitive values
        encrypted = MCPCredentials(
            service_name=credentials.service_name,
            team_restricted=credentials.team_restricted,
            custom_headers=credentials.custom_headers,
            expires_at=credentials.expires_at
        )
        
        if credentials.api_key:
            encrypted.api_key = self._hash_secret(credentials.api_key)
        if credentials.api_secret:
            encrypted.api_secret = self._hash_secret(credentials.api_secret)
        if credentials.token:
            encrypted.token = self._hash_secret(credentials.token)
        if credentials.password:
            encrypted.password = self._hash_secret(credentials.password)
        
        encrypted.username = credentials.username  # Username not encrypted
        
        return encrypted
    
    def _hash_secret(self, secret: str) -> str:
        """Hash secret for storage"""
        return hmac.new(
            self.secret_key.encode(),
            secret.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def generate_access_token(self, agent_id: str, team: TeamType, 
                            resource_ids: List[str], ttl_hours: int = 24) -> str:
        """Generate JWT access token for agent"""
        payload = {
            "agent_id": agent_id,
            "team": team.value,
            "resource_ids": resource_ids,
            "issued_at": time.time(),
            "expires_at": time.time() + (ttl_hours * 3600),
            "permissions": self._get_team_permissions(team)
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        self.access_tokens[token] = payload
        
        self.logger.info(f"🎫 Generated access token for {agent_id} ({team.value})")
        return token
    
    def _get_team_permissions(self, team: TeamType) -> List[str]:
        """Get permissions based on team type"""
        if team == TeamType.RED_TEAM:
            return [
                "read_attack_frameworks",
                "access_penetration_tools",
                "query_vulnerability_databases",
                "use_exploitation_frameworks",
                "access_osint_sources",
                "read_malware_samples"
            ]
        elif team == TeamType.BLUE_TEAM:
            return [
                "read_threat_intelligence",
                "access_defense_platforms",
                "query_security_tools",
                "access_monitoring_apis",
                "read_incident_databases",
                "access_forensics_tools",
                "query_reputation_services"
            ]
        else:
            return ["read_public_resources"]
    
    def validate_access(self, token: str, resource_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate agent access to resource"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            
            # Check token expiration
            if time.time() > payload.get("expires_at", 0):
                return False, {"error": "Token expired"}
            
            # Check resource access
            if resource_id not in payload.get("resource_ids", []):
                return False, {"error": "Resource not authorized"}
            
            return True, payload
            
        except jwt.InvalidTokenError as e:
            return False, {"error": f"Invalid token: {e}"}
    
    def check_rate_limit(self, agent_id: str, resource_id: str, 
                        limit_per_hour: int = 1000) -> bool:
        """Check if agent is within rate limits"""
        key = f"{agent_id}:{resource_id}"
        current_time = time.time()
        
        if key not in self.rate_limiters:
            self.rate_limiters[key] = {"count": 0, "window_start": current_time}
        
        limiter = self.rate_limiters[key]
        
        # Reset window if hour has passed
        if current_time - limiter["window_start"] > 3600:
            limiter["count"] = 0
            limiter["window_start"] = current_time
        
        if limiter["count"] >= limit_per_hour:
            return False
        
        limiter["count"] += 1
        return True

class MCPServer:
    """Base MCP server for team-specific operations"""
    
    def __init__(self, team: TeamType, security_manager: MCPSecurityManager):
        self.team = team
        self.security_manager = security_manager
        self.server_id = f"mcp_{team.value}_{uuid.uuid4().hex[:8]}"
        self.logger = logging.getLogger(f"mcp_server_{team.value}")
        
        self.resources: Dict[str, ExternalResource] = {}
        self.sdk_integrations: Dict[str, SDKIntegration] = {}
        self.active_connections: Dict[str, Any] = {}
        
        self.server_running = False
        self.client_sessions: Dict[str, ClientSession] = {}
    
    async def start_server(self) -> bool:
        """Start the MCP server"""
        try:
            self.logger.info(f"🚀 Starting {self.team.value} MCP server: {self.server_id}")
            
            # Initialize team-specific resources
            await self._initialize_team_resources()
            
            # Setup SDK integrations
            await self._setup_sdk_integrations()
            
            # Start MCP protocol server
            if MCP_AVAILABLE:
                await self._start_mcp_protocol()
            else:
                self.logger.warning("⚠️ MCP not available - using mock server")
                await self._start_mock_server()
            
            self.server_running = True
            self.logger.info(f"✅ {self.team.value} MCP server ready")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start MCP server: {e}")
            return False
    
    async def _initialize_team_resources(self):
        """Initialize team-specific external resources"""
        if self.team == TeamType.RED_TEAM:
            await self._setup_red_team_resources()
        elif self.team == TeamType.BLUE_TEAM:
            await self._setup_blue_team_resources()
    
    async def _setup_red_team_resources(self):
        """Setup red team specific resources"""
        red_team_resources = [
            ExternalResource(
                resource_id="metasploit_api",
                name="Metasploit Framework API",
                resource_type=ResourceType.ATTACK_FRAMEWORKS,
                endpoint_url="https://localhost:55553",
                credentials_id="metasploit_creds",
                allowed_teams=[TeamType.RED_TEAM],
                capabilities=["exploit_search", "payload_generation", "session_management"],
                security_level="restricted",
                description="Metasploit Framework for penetration testing"
            ),
            ExternalResource(
                resource_id="shodan_api",
                name="Shodan Internet Search",
                resource_type=ResourceType.OSINT_SOURCES,
                endpoint_url="https://api.shodan.io",
                credentials_id="shodan_creds",
                allowed_teams=[TeamType.RED_TEAM, TeamType.BLUE_TEAM],
                rate_limits={"requests_per_hour": 1000},
                capabilities=["host_search", "vulnerability_search", "banner_grabbing"],
                description="Shodan search engine for internet-connected devices"
            ),
            ExternalResource(
                resource_id="nuclei_templates",
                name="Nuclei Vulnerability Scanner",
                resource_type=ResourceType.SECURITY_TOOLS,
                endpoint_url="https://api.nuclei.org",
                credentials_id="nuclei_creds",
                allowed_teams=[TeamType.RED_TEAM],
                capabilities=["vulnerability_scanning", "template_execution"],
                description="Fast vulnerability scanner with templates"
            ),
            ExternalResource(
                resource_id="exploit_db",
                name="Exploit Database",
                resource_type=ResourceType.VULNERABILITY_DATABASE,
                endpoint_url="https://www.exploit-db.com/api",
                credentials_id="public_access",
                allowed_teams=[TeamType.RED_TEAM, TeamType.BLUE_TEAM],
                capabilities=["exploit_search", "poc_download"],
                description="Archive of public exploits and vulnerability data"
            )
        ]
        
        for resource in red_team_resources:
            self.resources[resource.resource_id] = resource
        
        self.logger.info(f"🔴 Red team resources initialized: {len(red_team_resources)} resources")
    
    async def _setup_blue_team_resources(self):
        """Setup blue team specific resources"""
        blue_team_resources = [
            ExternalResource(
                resource_id="virustotal_api",
                name="VirusTotal API",
                resource_type=ResourceType.THREAT_INTELLIGENCE,
                endpoint_url="https://www.virustotal.com/vtapi/v2",
                credentials_id="virustotal_creds",
                allowed_teams=[TeamType.BLUE_TEAM],
                rate_limits={"requests_per_minute": 4},
                capabilities=["file_analysis", "url_analysis", "ip_reputation"],
                description="Multi-engine malware analysis service"
            ),
            ExternalResource(
                resource_id="misp_api",
                name="MISP Threat Intelligence",
                resource_type=ResourceType.THREAT_INTELLIGENCE,
                endpoint_url="https://misp.example.com",
                credentials_id="misp_creds",
                allowed_teams=[TeamType.BLUE_TEAM],
                capabilities=["ioc_search", "event_correlation", "attribute_search"],
                security_level="classified",
                description="Malware Information Sharing Platform"
            ),
            ExternalResource(
                resource_id="osquery_api",
                name="OSQuery Endpoint Visibility",
                resource_type=ResourceType.FORENSICS_TOOLS,
                endpoint_url="https://osquery.example.com:8443",
                credentials_id="osquery_creds",
                allowed_teams=[TeamType.BLUE_TEAM],
                capabilities=["endpoint_query", "live_investigation", "artifact_collection"],
                description="SQL-based operating system instrumentation"
            ),
            ExternalResource(
                resource_id="yara_rules",
                name="YARA Malware Detection",
                resource_type=ResourceType.MALWARE_ANALYSIS,
                endpoint_url="https://yara-rules.example.com",
                credentials_id="yara_creds",
                allowed_teams=[TeamType.BLUE_TEAM],
                capabilities=["rule_matching", "pattern_detection", "signature_creation"],
                description="Pattern matching engine for malware research"
            ),
            ExternalResource(
                resource_id="elastic_siem",
                name="Elastic Security SIEM",
                resource_type=ResourceType.DEFENSE_PLATFORMS,
                endpoint_url="https://elastic.example.com:9200",
                credentials_id="elastic_creds",
                allowed_teams=[TeamType.BLUE_TEAM],
                capabilities=["log_analysis", "threat_hunting", "alert_management"],
                description="Security Information and Event Management platform"
            )
        ]
        
        for resource in blue_team_resources:
            self.resources[resource.resource_id] = resource
        
        self.logger.info(f"🔵 Blue team resources initialized: {len(blue_team_resources)} resources")
    
    async def _setup_sdk_integrations(self):
        """Setup SDK integrations for the team"""
        if self.team == TeamType.RED_TEAM:
            await self._setup_red_team_sdks()
        elif self.team == TeamType.BLUE_TEAM:
            await self._setup_blue_team_sdks()
    
    async def _setup_red_team_sdks(self):
        """Setup red team SDK integrations"""
        red_team_sdks = [
            SDKIntegration(
                sdk_name="metasploit-framework",
                sdk_version="6.3.0",
                installation_method="docker",
                docker_image="metasploitframework/metasploit-framework:latest",
                config_template={
                    "msfrpcd_port": 55553,
                    "msfrpcd_user": "msf",
                    "enable_modules": ["exploits", "payloads", "auxiliary"]
                },
                team_restrictions=[TeamType.RED_TEAM],
                required_credentials=["metasploit_creds"]
            ),
            SDKIntegration(
                sdk_name="nmap",
                sdk_version="7.94",
                installation_method="binary",
                binary_path="/usr/bin/nmap",
                config_template={
                    "scan_techniques": ["TCP_SYN", "UDP", "TCP_CONNECT"],
                    "timing_template": "T3",
                    "max_parallelism": 50
                },
                team_restrictions=[TeamType.RED_TEAM, TeamType.BLUE_TEAM]
            ),
            SDKIntegration(
                sdk_name="sqlmap",
                sdk_version="1.7.2",
                installation_method="docker",
                docker_image="sqlmapproject/sqlmap:latest",
                config_template={
                    "api_port": 8775,
                    "enable_tampers": True,
                    "risk_level": 2
                },
                team_restrictions=[TeamType.RED_TEAM]
            )
        ]
        
        for sdk in red_team_sdks:
            self.sdk_integrations[sdk.sdk_name] = sdk
        
        self.logger.info(f"🔴 Red team SDKs configured: {len(red_team_sdks)} SDKs")
    
    async def _setup_blue_team_sdks(self):
        """Setup blue team SDK integrations"""
        blue_team_sdks = [
            SDKIntegration(
                sdk_name="volatility3",
                sdk_version="2.5.0",
                installation_method="pip",
                config_template={
                    "plugin_path": "/opt/volatility3/plugins",
                    "symbol_path": "/opt/volatility3/symbols",
                    "enable_experimental": False
                },
                team_restrictions=[TeamType.BLUE_TEAM],
                required_credentials=[]
            ),
            SDKIntegration(
                sdk_name="suricata",
                sdk_version="7.0.0",
                installation_method="docker",
                docker_image="jasonish/suricata:latest",
                config_template={
                    "interface": "eth0",
                    "ruleset": "emerging-threats",
                    "eve_log": True
                },
                team_restrictions=[TeamType.BLUE_TEAM]
            ),
            SDKIntegration(
                sdk_name="yara-python",
                sdk_version="4.3.1",
                installation_method="pip",
                config_template={
                    "max_strings_per_rule": 10000,
                    "enable_profiling": True,
                    "timeout": 60
                },
                team_restrictions=[TeamType.BLUE_TEAM]
            )
        ]
        
        for sdk in blue_team_sdks:
            self.sdk_integrations[sdk.sdk_name] = sdk
        
        self.logger.info(f"🔵 Blue team SDKs configured: {len(blue_team_sdks)} SDKs")
    
    async def _start_mcp_protocol(self):
        """Start actual MCP protocol server"""
        # This would start the real MCP server
        # For now, we'll create a mock implementation
        await self._start_mock_server()
    
    async def _start_mock_server(self):
        """Start mock MCP server for development"""
        self.logger.info(f"🔧 Starting mock MCP server for {self.team.value}")
        
        # Simulate server startup
        await asyncio.sleep(0.1)
        
        # Create mock client sessions for each resource
        for resource_id, resource in self.resources.items():
            mock_session = f"session_{resource_id}_{uuid.uuid4().hex[:8]}"
            self.client_sessions[resource_id] = mock_session
            self.logger.debug(f"📡 Mock connection established: {resource.name}")
    
    async def execute_tool(self, agent_token: str, tool_name: str, 
                          parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool through MCP connection"""
        # Validate access
        is_valid, token_data = self.security_manager.validate_access(agent_token, tool_name)
        if not is_valid:
            return {"error": "Access denied", "details": token_data}
        
        agent_id = token_data.get("agent_id", "unknown")
        
        # Check rate limits
        if not self.security_manager.check_rate_limit(agent_id, tool_name):
            return {"error": "Rate limit exceeded"}
        
        # Execute tool
        try:
            result = await self._execute_tool_internal(tool_name, parameters, token_data)
            
            self.logger.info(f"🔧 Tool executed: {tool_name} by {agent_id}")
            return {
                "success": True,
                "tool": tool_name,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Tool execution failed: {tool_name} - {e}")
            return {"error": f"Tool execution failed: {e}"}
    
    async def _execute_tool_internal(self, tool_name: str, parameters: Dict[str, Any], 
                                   token_data: Dict[str, Any]) -> Dict[str, Any]:
        """Internal tool execution logic"""
        # Find matching resource
        resource = None
        for res in self.resources.values():
            if tool_name in res.capabilities:
                resource = res
                break
        
        if not resource:
            raise ValueError(f"Tool {tool_name} not found")
        
        # Team-specific tool execution
        if self.team == TeamType.RED_TEAM:
            return await self._execute_red_team_tool(tool_name, parameters, resource)
        elif self.team == TeamType.BLUE_TEAM:
            return await self._execute_blue_team_tool(tool_name, parameters, resource)
        else:
            raise ValueError("Unsupported team type")
    
    async def _execute_red_team_tool(self, tool_name: str, parameters: Dict[str, Any], 
                                   resource: ExternalResource) -> Dict[str, Any]:
        """Execute red team specific tools"""
        if tool_name == "exploit_search":
            return {
                "exploits_found": [
                    {"cve": "CVE-2023-1234", "title": "Test Exploit", "rank": "excellent"},
                    {"cve": "CVE-2023-5678", "title": "Another Exploit", "rank": "good"}
                ],
                "search_query": parameters.get("query", ""),
                "resource": resource.name
            }
        
        elif tool_name == "vulnerability_scanning":
            return {
                "vulnerabilities": [
                    {"severity": "high", "cve": "CVE-2023-1111", "port": 22},
                    {"severity": "medium", "cve": "CVE-2023-2222", "port": 80}
                ],
                "target": parameters.get("target", "unknown"),
                "scan_duration": 30.5
            }
        
        elif tool_name == "host_search":
            return {
                "hosts_found": [
                    {"ip": "192.168.1.100", "port": 22, "service": "SSH"},
                    {"ip": "192.168.1.101", "port": 80, "service": "HTTP"}
                ],
                "query": parameters.get("query", ""),
                "total_results": 2
            }
        
        else:
            return {"message": f"Executed {tool_name}", "parameters": parameters}
    
    async def _execute_blue_team_tool(self, tool_name: str, parameters: Dict[str, Any], 
                                    resource: ExternalResource) -> Dict[str, Any]:
        """Execute blue team specific tools"""
        if tool_name == "file_analysis":
            return {
                "analysis_result": {
                    "malicious": False,
                    "detections": 2,
                    "total_engines": 70,
                    "scan_date": datetime.now().isoformat()
                },
                "file_hash": parameters.get("hash", "unknown"),
                "resource": resource.name
            }
        
        elif tool_name == "threat_hunting":
            return {
                "threats_found": [
                    {"type": "suspicious_process", "confidence": 0.8, "host": "workstation-01"},
                    {"type": "unusual_network", "confidence": 0.6, "host": "server-02"}
                ],
                "hunt_query": parameters.get("query", ""),
                "time_range": parameters.get("time_range", "24h")
            }
        
        elif tool_name == "ioc_search":
            return {
                "indicators": [
                    {"type": "ip", "value": "192.168.1.1", "threat_type": "c2"},
                    {"type": "domain", "value": "evil.com", "threat_type": "phishing"}
                ],
                "search_terms": parameters.get("terms", []),
                "confidence_threshold": parameters.get("confidence", 0.7)
            }
        
        else:
            return {"message": f"Executed {tool_name}", "parameters": parameters}
    
    async def get_available_tools(self, agent_token: str) -> Dict[str, Any]:
        """Get list of available tools for agent"""
        # Validate access
        is_valid, token_data = self.security_manager.validate_access(agent_token, "list_tools")
        if not is_valid:
            return {"error": "Access denied"}
        
        tools = []
        for resource in self.resources.values():
            if self.team in resource.allowed_teams:
                for capability in resource.capabilities:
                    tools.append({
                        "name": capability,
                        "resource": resource.name,
                        "description": f"{capability} via {resource.name}",
                        "security_level": resource.security_level
                    })
        
        return {
            "team": self.team.value,
            "available_tools": tools,
            "total_tools": len(tools)
        }
    
    async def stop_server(self):
        """Stop the MCP server"""
        self.logger.info(f"🛑 Stopping {self.team.value} MCP server")
        
        # Close all client sessions
        for session_id in self.client_sessions:
            # In real implementation, would properly close MCP sessions
            pass
        
        self.client_sessions.clear()
        self.server_running = False
        
        self.logger.info(f"✅ {self.team.value} MCP server stopped")

class MCPOrchestrator:
    """Main orchestrator for MCP integration architecture"""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or f"archangel_mcp_{uuid.uuid4().hex}"
        self.security_manager = MCPSecurityManager(self.secret_key)
        
        self.red_team_server: Optional[MCPServer] = None
        self.blue_team_server: Optional[MCPServer] = None
        
        self.logger = logging.getLogger("mcp_orchestrator")
        
        # Configuration management
        self.config = self._load_default_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default MCP configuration"""
        return {
            "isolation": {
                "enable_team_isolation": True,
                "shared_resources": ["public_osint", "public_databases"],
                "cross_team_communication": False
            },
            "security": {
                "token_ttl_hours": 24,
                "rate_limit_per_hour": 1000,
                "require_mfa": False,
                "audit_all_requests": True
            },
            "performance": {
                "max_concurrent_connections": 10,
                "connection_timeout_seconds": 30,
                "response_cache_ttl": 300
            }
        }
    
    async def initialize_mcp_architecture(self) -> bool:
        """Initialize the complete MCP architecture"""
        self.logger.info("🏗️ Initializing MCP integration architecture...")
        
        try:
            # Setup credentials for external services
            await self._setup_external_credentials()
            
            # Initialize red team MCP server
            self.red_team_server = MCPServer(TeamType.RED_TEAM, self.security_manager)
            red_started = await self.red_team_server.start_server()
            
            # Initialize blue team MCP server
            self.blue_team_server = MCPServer(TeamType.BLUE_TEAM, self.security_manager)
            blue_started = await self.blue_team_server.start_server()
            
            if red_started and blue_started:
                self.logger.info("✅ MCP architecture initialized successfully")
                return True
            else:
                self.logger.error("❌ Failed to initialize MCP servers")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ MCP architecture initialization failed: {e}")
            return False
    
    async def _setup_external_credentials(self):
        """Setup credentials for external services"""
        # Example credentials (in production, load from secure vault)
        credentials_to_add = [
            MCPCredentials(
                service_name="shodan",
                api_key="YOUR_SHODAN_API_KEY",
                team_restricted=None  # Available to both teams
            ),
            MCPCredentials(
                service_name="virustotal",
                api_key="YOUR_VIRUSTOTAL_API_KEY",
                team_restricted=TeamType.BLUE_TEAM
            ),
            MCPCredentials(
                service_name="metasploit",
                username="msf",
                password="msf_password",
                team_restricted=TeamType.RED_TEAM
            ),
            MCPCredentials(
                service_name="misp",
                token="YOUR_MISP_API_TOKEN",
                custom_headers={"Accept": "application/json"},
                team_restricted=TeamType.BLUE_TEAM
            )
        ]
        
        for creds in credentials_to_add:
            cred_id = self.security_manager.add_credentials(creds)
            self.logger.debug(f"🔐 Added credentials: {creds.service_name} -> {cred_id}")
    
    def provision_agent_access(self, agent_id: str, team: TeamType, 
                             resource_ids: List[str] = None) -> str:
        """Provision MCP access for an agent"""
        if resource_ids is None:
            # Get all resources for team
            server = self.red_team_server if team == TeamType.RED_TEAM else self.blue_team_server
            resource_ids = list(server.resources.keys()) if server else []
        
        # Generate access token
        token = self.security_manager.generate_access_token(
            agent_id, team, resource_ids, ttl_hours=24
        )
        
        self.logger.info(f"🎫 Provisioned {team.value} access for agent {agent_id}")
        return token
    
    async def execute_agent_tool(self, agent_token: str, tool_name: str, 
                                parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool for agent through appropriate MCP server"""
        # Validate token to determine team
        is_valid, token_data = self.security_manager.validate_access(agent_token, tool_name)
        if not is_valid:
            return {"error": "Invalid access token"}
        
        team_str = token_data.get("team")
        if team_str == "red_team" and self.red_team_server:
            return await self.red_team_server.execute_tool(agent_token, tool_name, parameters)
        elif team_str == "blue_team" and self.blue_team_server:
            return await self.blue_team_server.execute_tool(agent_token, tool_name, parameters)
        else:
            return {"error": "No appropriate MCP server available"}
    
    async def get_agent_capabilities(self, agent_token: str) -> Dict[str, Any]:
        """Get available capabilities for agent"""
        # Validate token to determine team
        is_valid, token_data = self.security_manager.validate_access(agent_token, "list_tools")
        if not is_valid:
            return {"error": "Invalid access token"}
        
        team_str = token_data.get("team")
        if team_str == "red_team" and self.red_team_server:
            return await self.red_team_server.get_available_tools(agent_token)
        elif team_str == "blue_team" and self.blue_team_server:
            return await self.blue_team_server.get_available_tools(agent_token)
        else:
            return {"error": "No appropriate MCP server available"}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive MCP system status"""
        status = {
            "orchestrator_status": "operational",
            "servers": {},
            "security": {
                "credentials_stored": len(self.security_manager.credentials_store),
                "active_tokens": len(self.security_manager.access_tokens),
                "rate_limiters": len(self.security_manager.rate_limiters)
            },
            "configuration": self.config
        }
        
        # Red team server status
        if self.red_team_server:
            status["servers"]["red_team"] = {
                "running": self.red_team_server.server_running,
                "resources": len(self.red_team_server.resources),
                "sdk_integrations": len(self.red_team_server.sdk_integrations),
                "active_connections": len(self.red_team_server.active_connections)
            }
        
        # Blue team server status
        if self.blue_team_server:
            status["servers"]["blue_team"] = {
                "running": self.blue_team_server.server_running,
                "resources": len(self.blue_team_server.resources),
                "sdk_integrations": len(self.blue_team_server.sdk_integrations),
                "active_connections": len(self.blue_team_server.active_connections)
            }
        
        return status
    
    async def shutdown_mcp_architecture(self):
        """Shutdown the MCP architecture gracefully"""
        self.logger.info("🛑 Shutting down MCP architecture...")
        
        # Stop red team server
        if self.red_team_server:
            await self.red_team_server.stop_server()
        
        # Stop blue team server
        if self.blue_team_server:
            await self.blue_team_server.stop_server()
        
        # Clear security manager data
        self.security_manager.access_tokens.clear()
        self.security_manager.rate_limiters.clear()
        
        self.logger.info("✅ MCP architecture shutdown complete")

# Factory functions and demo

def create_mcp_orchestrator(secret_key: str = None) -> MCPOrchestrator:
    """Create MCP orchestrator instance"""
    return MCPOrchestrator(secret_key)

async def demo_mcp_integration() -> Dict[str, Any]:
    """Demonstrate MCP integration capabilities"""
    orchestrator = create_mcp_orchestrator()
    
    # Initialize architecture
    if not await orchestrator.initialize_mcp_architecture():
        return {"error": "Failed to initialize MCP architecture"}
    
    # Provision access for demo agents
    red_token = orchestrator.provision_agent_access("red_agent_001", TeamType.RED_TEAM)
    blue_token = orchestrator.provision_agent_access("blue_agent_001", TeamType.BLUE_TEAM)
    
    # Test red team capabilities
    red_capabilities = await orchestrator.get_agent_capabilities(red_token)
    
    # Test blue team capabilities
    blue_capabilities = await orchestrator.get_agent_capabilities(blue_token)
    
    # Execute sample tools
    red_tool_result = await orchestrator.execute_agent_tool(
        red_token, "exploit_search", {"query": "apache"}
    )
    
    blue_tool_result = await orchestrator.execute_agent_tool(
        blue_token, "file_analysis", {"hash": "test_hash_123"}
    )
    
    # Get system status
    system_status = await orchestrator.get_system_status()
    
    # Cleanup
    await orchestrator.shutdown_mcp_architecture()
    
    return {
        "demo_completed": True,
        "red_team": {
            "capabilities": red_capabilities,
            "tool_execution": red_tool_result
        },
        "blue_team": {
            "capabilities": blue_capabilities,
            "tool_execution": blue_tool_result
        },
        "system_status": system_status
    }

# Integration with existing Archangel agents

class ArchangelMCPIntegration:
    """Integration layer between Archangel agents and MCP architecture"""
    
    def __init__(self, mcp_orchestrator: MCPOrchestrator):
        self.mcp_orchestrator = mcp_orchestrator
        self.agent_tokens: Dict[str, str] = {}
        self.logger = logging.getLogger("archangel_mcp")
    
    async def register_agent(self, agent_id: str, team: TeamType) -> str:
        """Register Archangel agent with MCP system"""
        token = self.mcp_orchestrator.provision_agent_access(agent_id, team)
        self.agent_tokens[agent_id] = token
        
        self.logger.info(f"🤖 Registered agent {agent_id} with {team.value} MCP access")
        return token
    
    async def enhance_agent_with_mcp(self, agent, team: TeamType):
        """Enhance existing Archangel agent with MCP capabilities"""
        # Register agent
        token = await self.register_agent(agent.agent_id, team)
        
        # Add MCP methods to agent
        agent.mcp_token = token
        agent.execute_mcp_tool = lambda tool, params: self.mcp_orchestrator.execute_agent_tool(token, tool, params)
        agent.get_mcp_capabilities = lambda: self.mcp_orchestrator.get_agent_capabilities(token)
        
        self.logger.info(f"✨ Enhanced agent {agent.agent_id} with MCP capabilities")
        return agent

if __name__ == "__main__":
    # Run demo
    async def main():
        result = await demo_mcp_integration()
        print(json.dumps(result, indent=2))
    
    asyncio.run(main())