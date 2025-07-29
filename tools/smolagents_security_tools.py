"""
Archangel SmolAgents Security Tools
Custom security tools for autonomous AI operations
"""

import asyncio
import subprocess
import json
import socket
import dns.resolver
import requests
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import time
import logging
from pathlib import Path

try:
    from smolagents.tools import Tool
    from smolagents import CodeAgent
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    # Fallback base class if SmolAgents not available
    class Tool:
        def __init__(self, name: str, description: str):
            self.name = name
            self.description = description
        
        def __call__(self, *args, **kwargs):
            raise NotImplementedError("SmolAgents not available")
    
    SMOLAGENTS_AVAILABLE = False

@dataclass
class SecurityScanResult:
    """Standardized security scan result"""
    tool_name: str
    target: str
    success: bool
    findings: Dict[str, Any]
    raw_output: str
    execution_time: float
    timestamp: float

class NetworkReconnaissanceTool(Tool):
    """SmolAgents tool for network reconnaissance"""
    
    def __init__(self):
        super().__init__(
            name="network_reconnaissance",
            description="""Perform network reconnaissance on a target.
            Args:
                target (str): Target IP address or domain name
                scan_type (str): Type of scan - 'ping', 'port', 'service', 'full'
            Returns:
                Dict with scan results including open ports, services, and host information
            """
        )
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, target: str, scan_type: str = "ping") -> Dict[str, Any]:
        """Execute network reconnaissance"""
        start_time = time.time()
        
        try:
            if scan_type == "ping":
                return self._ping_scan(target, start_time)
            elif scan_type == "port":
                return self._port_scan(target, start_time)
            elif scan_type == "service":
                return self._service_scan(target, start_time)
            elif scan_type == "full":
                return self._full_scan(target, start_time)
            else:
                return self._create_error_result(target, f"Unknown scan type: {scan_type}", start_time)
                
        except Exception as e:
            return self._create_error_result(target, str(e), start_time)
    
    def _ping_scan(self, target: str, start_time: float) -> Dict[str, Any]:
        """Perform ping scan to check if host is alive"""
        try:
            # Try to resolve hostname first
            try:
                socket.gethostbyname(target)
                host_resolvable = True
            except socket.gaierror:
                host_resolvable = False
            
            # Perform ping
            if self._is_windows():
                cmd = ["ping", "-n", "1", target]
            else:
                cmd = ["ping", "-c", "1", target]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            findings = {
                "host_alive": result.returncode == 0,
                "host_resolvable": host_resolvable,
                "response_time": self._extract_ping_time(result.stdout),
                "target_resolved": target
            }
            
            return {
                "tool_name": "network_reconnaissance",
                "target": target,
                "scan_type": "ping",
                "success": True,
                "findings": findings,
                "raw_output": result.stdout,
                "execution_time": time.time() - start_time,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return self._create_error_result(target, str(e), start_time)
    
    def _port_scan(self, target: str, start_time: float) -> Dict[str, Any]:
        """Perform basic port scan"""
        common_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 3389, 5432, 3306]
        open_ports = []
        
        try:
            for port in common_ports:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((target, port))
                if result == 0:
                    open_ports.append({
                        "port": port,
                        "state": "open",
                        "service": self._guess_service(port)
                    })
                sock.close()
            
            findings = {
                "open_ports": open_ports,
                "total_ports_scanned": len(common_ports),
                "open_port_count": len(open_ports)
            }
            
            return {
                "tool_name": "network_reconnaissance",
                "target": target,
                "scan_type": "port",
                "success": True,
                "findings": findings,
                "raw_output": f"Scanned {len(common_ports)} common ports, found {len(open_ports)} open",
                "execution_time": time.time() - start_time,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return self._create_error_result(target, str(e), start_time)
    
    def _service_scan(self, target: str, start_time: float) -> Dict[str, Any]:
        """Perform service detection on open ports"""
        # First do port scan
        port_result = self._port_scan(target, start_time)
        if not port_result["success"]:
            return port_result
        
        services = []
        for port_info in port_result["findings"]["open_ports"]:
            port = port_info["port"]
            service_info = {
                "port": port,
                "service": port_info["service"],
                "version": "unknown",
                "banner": self._grab_banner(target, port)
            }
            services.append(service_info)
        
        findings = {
            "services": services,
            "service_count": len(services),
            "web_services": [s for s in services if s["service"] in ["http", "https"]],
            "ssh_services": [s for s in services if s["service"] == "ssh"]
        }
        
        return {
            "tool_name": "network_reconnaissance",
            "target": target,
            "scan_type": "service",
            "success": True,
            "findings": findings,
            "raw_output": f"Identified {len(services)} services",
            "execution_time": time.time() - start_time,
            "timestamp": time.time()
        }
    
    def _full_scan(self, target: str, start_time: float) -> Dict[str, Any]:
        """Perform comprehensive scan"""
        results = {}
        
        # Ping scan
        ping_result = self._ping_scan(target, start_time)
        results["ping"] = ping_result["findings"]
        
        if not ping_result["findings"]["host_alive"]:
            return {
                "tool_name": "network_reconnaissance",
                "target": target,
                "scan_type": "full",
                "success": True,
                "findings": {"host_not_alive": True, "ping_result": results["ping"]},
                "raw_output": "Host not responding to ping",
                "execution_time": time.time() - start_time,
                "timestamp": time.time()
            }
        
        # Service scan
        service_result = self._service_scan(target, start_time)
        results["services"] = service_result["findings"]
        
        # DNS lookup
        results["dns"] = self._dns_lookup(target)
        
        return {
            "tool_name": "network_reconnaissance",
            "target": target,
            "scan_type": "full",
            "success": True,
            "findings": results,
            "raw_output": f"Full scan completed with {len(results)} components",
            "execution_time": time.time() - start_time,
            "timestamp": time.time()
        }
    
    def _dns_lookup(self, target: str) -> Dict[str, Any]:
        """Perform DNS lookup"""
        dns_info = {}
        
        try:
            # A record
            answers = dns.resolver.resolve(target, 'A')
            dns_info["A"] = [str(answer) for answer in answers]
        except:
            dns_info["A"] = []
        
        try:
            # MX record
            answers = dns.resolver.resolve(target, 'MX')
            dns_info["MX"] = [str(answer) for answer in answers]
        except:
            dns_info["MX"] = []
        
        try:
            # TXT record
            answers = dns.resolver.resolve(target, 'TXT')
            dns_info["TXT"] = [str(answer) for answer in answers]
        except:
            dns_info["TXT"] = []
        
        return dns_info
    
    def _grab_banner(self, target: str, port: int) -> str:
        """Attempt to grab service banner"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect((target, port))
            
            # Send appropriate probe based on port
            if port == 80:
                sock.send(b"GET / HTTP/1.0\r\n\r\n")
            elif port == 22:
                pass  # SSH will send banner automatically
            elif port == 21:
                pass  # FTP will send banner automatically
            else:
                sock.send(b"\r\n")
            
            banner = sock.recv(1024).decode('utf-8', errors='ignore').strip()
            sock.close()
            return banner[:200]  # Limit banner length
            
        except:
            return ""
    
    def _guess_service(self, port: int) -> str:
        """Guess service based on port number"""
        port_map = {
            21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns",
            80: "http", 110: "pop3", 143: "imap", 443: "https",
            993: "imaps", 995: "pop3s", 3389: "rdp", 5432: "postgresql", 3306: "mysql"
        }
        return port_map.get(port, "unknown")
    
    def _extract_ping_time(self, ping_output: str) -> Optional[float]:
        """Extract ping response time from output"""
        import re
        pattern = r"time[<=](\d+\.?\d*)\s*ms"
        match = re.search(pattern, ping_output)
        if match:
            return float(match.group(1))
        return None
    
    def _is_windows(self) -> bool:
        """Check if running on Windows"""
        import platform
        return platform.system().lower() == "windows"
    
    def _create_error_result(self, target: str, error: str, start_time: float) -> Dict[str, Any]:
        """Create error result"""
        return {
            "tool_name": "network_reconnaissance",
            "target": target,
            "success": False,
            "error": error,
            "findings": {},
            "raw_output": f"Error: {error}",
            "execution_time": time.time() - start_time,
            "timestamp": time.time()
        }

class WebApplicationScannerTool(Tool):
    """SmolAgents tool for web application security scanning"""
    
    def __init__(self):
        super().__init__(
            name="web_app_scanner",
            description="""Perform web application security scanning.
            Args:
                target_url (str): Target web application URL
                scan_type (str): Type of scan - 'headers', 'directories', 'forms', 'full'
            Returns:
                Dict with web security findings including headers, directories, and vulnerabilities
            """
        )
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, target_url: str, scan_type: str = "headers") -> Dict[str, Any]:
        """Execute web application scan"""
        start_time = time.time()
        
        try:
            if scan_type == "headers":
                return self._scan_headers(target_url, start_time)
            elif scan_type == "directories":
                return self._scan_directories(target_url, start_time)
            elif scan_type == "forms":
                return self._scan_forms(target_url, start_time)
            elif scan_type == "full":
                return self._full_web_scan(target_url, start_time)
            else:
                return self._create_error_result(target_url, f"Unknown scan type: {scan_type}", start_time)
                
        except Exception as e:
            return self._create_error_result(target_url, str(e), start_time)
    
    def _scan_headers(self, target_url: str, start_time: float) -> Dict[str, Any]:
        """Scan HTTP headers for security issues"""
        try:
            response = requests.get(target_url, timeout=10, verify=False)
            headers = dict(response.headers)
            
            security_headers = {
                "Content-Security-Policy": headers.get("Content-Security-Policy"),
                "X-Frame-Options": headers.get("X-Frame-Options"),
                "X-XSS-Protection": headers.get("X-XSS-Protection"),
                "X-Content-Type-Options": headers.get("X-Content-Type-Options"),
                "Strict-Transport-Security": headers.get("Strict-Transport-Security"),
                "Referrer-Policy": headers.get("Referrer-Policy")
            }
            
            # Analyze security headers
            missing_headers = [k for k, v in security_headers.items() if v is None]
            security_score = (len(security_headers) - len(missing_headers)) / len(security_headers) * 100
            
            findings = {
                "all_headers": headers,
                "security_headers": security_headers,
                "missing_security_headers": missing_headers,
                "security_score": security_score,
                "server": headers.get("Server", "unknown"),
                "powered_by": headers.get("X-Powered-By"),
                "status_code": response.status_code
            }
            
            return {
                "tool_name": "web_app_scanner",
                "target": target_url,
                "scan_type": "headers",
                "success": True,
                "findings": findings,
                "raw_output": f"Analyzed {len(headers)} headers, security score: {security_score:.1f}%",
                "execution_time": time.time() - start_time,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return self._create_error_result(target_url, str(e), start_time)
    
    def _scan_directories(self, target_url: str, start_time: float) -> Dict[str, Any]:
        """Scan for common directories and files"""
        common_paths = [
            "/admin", "/login", "/wp-admin", "/phpmyadmin", "/administrator",
            "/backup", "/test", "/dev", "/robots.txt", "/sitemap.xml",
            "/.git", "/.env", "/config", "/database", "/db"
        ]
        
        found_paths = []
        
        try:
            for path in common_paths:
                full_url = target_url.rstrip('/') + path
                try:
                    response = requests.get(full_url, timeout=5, verify=False)
                    if response.status_code in [200, 301, 302, 403]:
                        found_paths.append({
                            "path": path,
                            "status_code": response.status_code,
                            "size": len(response.content),
                            "title": self._extract_title(response.text)
                        })
                except:
                    continue
            
            findings = {
                "found_paths": found_paths,
                "total_paths_checked": len(common_paths),
                "interesting_files": [p for p in found_paths if p["path"] in ["/.git", "/.env", "/backup"]],
                "admin_panels": [p for p in found_paths if any(admin in p["path"] for admin in ["admin", "login"])]
            }
            
            return {
                "tool_name": "web_app_scanner",
                "target": target_url,
                "scan_type": "directories",
                "success": True,
                "findings": findings,
                "raw_output": f"Found {len(found_paths)} accessible paths out of {len(common_paths)} checked",
                "execution_time": time.time() - start_time,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return self._create_error_result(target_url, str(e), start_time)
    
    def _scan_forms(self, target_url: str, start_time: float) -> Dict[str, Any]:
        """Scan for forms and input fields"""
        try:
            response = requests.get(target_url, timeout=10, verify=False)
            
            # Simple form detection (would use BeautifulSoup in production)
            forms_found = response.text.count('<form')
            input_fields = response.text.count('<input')
            
            # Look for specific form characteristics
            has_login_form = any(keyword in response.text.lower() for keyword in ['password', 'username', 'login'])
            has_search_form = 'type="search"' in response.text.lower() or 'name="search"' in response.text.lower()
            has_upload_form = 'type="file"' in response.text.lower()
            
            findings = {
                "forms_count": forms_found,
                "input_fields_count": input_fields,
                "has_login_form": has_login_form,
                "has_search_form": has_search_form,
                "has_upload_form": has_upload_form,
                "potential_sql_injection_points": input_fields if has_search_form else 0,
                "potential_xss_points": input_fields
            }
            
            return {
                "tool_name": "web_app_scanner",
                "target": target_url,
                "scan_type": "forms",
                "success": True,
                "findings": findings,
                "raw_output": f"Found {forms_found} forms with {input_fields} input fields",
                "execution_time": time.time() - start_time,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return self._create_error_result(target_url, str(e), start_time)
    
    def _full_web_scan(self, target_url: str, start_time: float) -> Dict[str, Any]:
        """Perform comprehensive web application scan"""
        results = {}
        
        # Headers scan
        headers_result = self._scan_headers(target_url, start_time)
        results["headers"] = headers_result["findings"]
        
        # Directory scan
        dir_result = self._scan_directories(target_url, start_time)
        results["directories"] = dir_result["findings"]
        
        # Forms scan
        forms_result = self._scan_forms(target_url, start_time)
        results["forms"] = forms_result["findings"]
        
        # Overall risk assessment
        risk_factors = []
        if results["headers"]["security_score"] < 50:
            risk_factors.append("Poor security headers")
        if results["directories"]["interesting_files"]:
            risk_factors.append("Sensitive files exposed")
        if results["forms"]["has_login_form"] and results["headers"]["security_score"] < 70:
            risk_factors.append("Login form with poor security headers")
        
        results["risk_assessment"] = {
            "risk_factors": risk_factors,
            "risk_level": "HIGH" if len(risk_factors) > 2 else "MEDIUM" if len(risk_factors) > 0 else "LOW"
        }
        
        return {
            "tool_name": "web_app_scanner",
            "target": target_url,
            "scan_type": "full",
            "success": True,
            "findings": results,
            "raw_output": f"Full web scan completed with {len(risk_factors)} risk factors identified",
            "execution_time": time.time() - start_time,
            "timestamp": time.time()
        }
    
    def _extract_title(self, html: str) -> str:
        """Extract page title from HTML"""
        import re
        match = re.search(r'<title.*?>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()[:100]
        return ""
    
    def _create_error_result(self, target_url: str, error: str, start_time: float) -> Dict[str, Any]:
        """Create error result"""
        return {
            "tool_name": "web_app_scanner",
            "target": target_url,
            "success": False,
            "error": error,
            "findings": {},
            "raw_output": f"Error: {error}",
            "execution_time": time.time() - start_time,
            "timestamp": time.time()
        }

class ThreatIntelligenceTool(Tool):
    """SmolAgents tool for threat intelligence gathering"""
    
    def __init__(self):
        super().__init__(
            name="threat_intelligence",
            description="""Gather threat intelligence on targets.
            Args:
                target (str): Target IP, domain, or hash
                intel_type (str): Type of intelligence - 'reputation', 'whois', 'geolocation', 'all'
            Returns:
                Dict with threat intelligence including reputation, location, and historical data
            """
        )
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, target: str, intel_type: str = "reputation") -> Dict[str, Any]:
        """Execute threat intelligence gathering"""
        start_time = time.time()
        
        try:
            if intel_type == "reputation":
                return self._check_reputation(target, start_time)
            elif intel_type == "whois":
                return self._whois_lookup(target, start_time)
            elif intel_type == "geolocation":
                return self._geolocate(target, start_time)
            elif intel_type == "all":
                return self._full_intel(target, start_time)
            else:
                return self._create_error_result(target, f"Unknown intel type: {intel_type}", start_time)
                
        except Exception as e:
            return self._create_error_result(target, str(e), start_time)
    
    def _check_reputation(self, target: str, start_time: float) -> Dict[str, Any]:
        """Check target reputation (mock implementation)"""
        # In production, this would integrate with threat intel APIs
        # For now, provide mock assessment
        
        findings = {
            "target": target,
            "reputation_score": 85,  # Mock score
            "threat_categories": [],
            "last_seen_malicious": None,
            "known_malware_families": [],
            "reputation_sources": ["mock_database"],
            "assessment": "Clean - no malicious activity detected"
        }
        
        # Simple heuristics for demo
        if any(suspicious in target.lower() for suspicious in ['malware', 'phish', 'hack', 'exploit']):
            findings["reputation_score"] = 15
            findings["threat_categories"] = ["suspicious_domain"]
            findings["assessment"] = "Suspicious - domain name contains threat indicators"
        
        return {
            "tool_name": "threat_intelligence",
            "target": target,
            "intel_type": "reputation",
            "success": True,
            "findings": findings,
            "raw_output": f"Reputation check completed - score: {findings['reputation_score']}/100",
            "execution_time": time.time() - start_time,
            "timestamp": time.time()
        }
    
    def _whois_lookup(self, target: str, start_time: float) -> Dict[str, Any]:
        """Perform WHOIS lookup"""
        try:
            # Simple WHOIS implementation
            import whois
            
            whois_data = whois.whois(target)
            
            findings = {
                "domain": target,
                "registrar": whois_data.get("registrar"),
                "creation_date": str(whois_data.get("creation_date")),
                "expiration_date": str(whois_data.get("expiration_date")),
                "name_servers": whois_data.get("name_servers", []),
                "status": whois_data.get("status", []),
                "country": whois_data.get("country"),
                "org": whois_data.get("org")
            }
            
            return {
                "tool_name": "threat_intelligence",
                "target": target,
                "intel_type": "whois",
                "success": True,
                "findings": findings,
                "raw_output": f"WHOIS data retrieved for {target}",
                "execution_time": time.time() - start_time,
                "timestamp": time.time()
            }
            
        except Exception as e:
            # Fallback mock data if whois fails
            findings = {
                "domain": target,
                "error": "WHOIS lookup failed - using mock data",
                "registrar": "Unknown",
                "creation_date": "Unknown",
                "expiration_date": "Unknown"
            }
            
            return {
                "tool_name": "threat_intelligence",
                "target": target,
                "intel_type": "whois",
                "success": False,
                "findings": findings,
                "raw_output": f"WHOIS lookup failed: {str(e)}",
                "execution_time": time.time() - start_time,
                "timestamp": time.time()
            }
    
    def _geolocate(self, target: str, start_time: float) -> Dict[str, Any]:
        """Geolocate IP address or domain"""
        try:
            # Resolve domain to IP if needed
            if not target.replace('.', '').isdigit():
                ip = socket.gethostbyname(target)
            else:
                ip = target
            
            # Mock geolocation (in production would use MaxMind or similar)
            findings = {
                "ip": ip,
                "country": "Unknown",
                "city": "Unknown",
                "latitude": 0.0,
                "longitude": 0.0,
                "isp": "Unknown",
                "organization": "Unknown",
                "timezone": "Unknown"
            }
            
            # Simple heuristics for demo
            if ip.startswith("192.168.") or ip.startswith("10.") or ip.startswith("172."):
                findings["country"] = "Private Network"
                findings["city"] = "RFC 1918 Address"
            
            return {
                "tool_name": "threat_intelligence",
                "target": target,
                "intel_type": "geolocation",
                "success": True,
                "findings": findings,
                "raw_output": f"Geolocation completed for {ip}",
                "execution_time": time.time() - start_time,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return self._create_error_result(target, str(e), start_time)
    
    def _full_intel(self, target: str, start_time: float) -> Dict[str, Any]:
        """Perform comprehensive threat intelligence gathering"""
        results = {}
        
        # Reputation check
        rep_result = self._check_reputation(target, start_time)
        results["reputation"] = rep_result["findings"]
        
        # WHOIS lookup
        whois_result = self._whois_lookup(target, start_time)
        results["whois"] = whois_result["findings"]
        
        # Geolocation
        geo_result = self._geolocate(target, start_time)
        results["geolocation"] = geo_result["findings"]
        
        # Overall threat assessment
        threat_score = results["reputation"].get("reputation_score", 50)
        if threat_score < 30:
            threat_level = "HIGH"
        elif threat_score < 60:
            threat_level = "MEDIUM"
        else:
            threat_level = "LOW"
        
        results["overall_assessment"] = {
            "threat_level": threat_level,
            "confidence": "Medium",
            "recommendation": f"Threat level assessed as {threat_level} based on available intelligence"
        }
        
        return {
            "tool_name": "threat_intelligence",
            "target": target,
            "intel_type": "full",
            "success": True,
            "findings": results,
            "raw_output": f"Full threat intelligence gathered - threat level: {threat_level}",
            "execution_time": time.time() - start_time,
            "timestamp": time.time()
        }
    
    def _create_error_result(self, target: str, error: str, start_time: float) -> Dict[str, Any]:
        """Create error result"""
        return {
            "tool_name": "threat_intelligence",
            "target": target,
            "success": False,
            "error": error,
            "findings": {},
            "raw_output": f"Error: {error}",
            "execution_time": time.time() - start_time,
            "timestamp": time.time()
        }

class SecurityReportGeneratorTool(Tool):
    """SmolAgents tool for generating security reports"""
    
    def __init__(self):
        super().__init__(
            name="security_report_generator",
            description="""Generate comprehensive security reports from scan results.
            Args:
                scan_results (list): List of scan results from other tools
                report_type (str): Type of report - 'executive', 'technical', 'compliance'
            Returns:
                Dict with formatted security report including findings, risks, and recommendations
            """
        )
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, scan_results: List[Dict[str, Any]], report_type: str = "technical") -> Dict[str, Any]:
        """Generate security report"""
        start_time = time.time()
        
        try:
            if report_type == "executive":
                return self._generate_executive_report(scan_results, start_time)
            elif report_type == "technical":
                return self._generate_technical_report(scan_results, start_time)
            elif report_type == "compliance":
                return self._generate_compliance_report(scan_results, start_time)
            else:
                return self._create_error_result(report_type, f"Unknown report type: {report_type}", start_time)
                
        except Exception as e:
            return self._create_error_result(report_type, str(e), start_time)
    
    def _generate_technical_report(self, scan_results: List[Dict[str, Any]], start_time: float) -> Dict[str, Any]:
        """Generate detailed technical report"""
        
        # Analyze results
        total_scans = len(scan_results)
        successful_scans = len([r for r in scan_results if r.get("success", False)])
        
        # Extract key findings
        all_findings = []
        high_risk_items = []
        recommendations = []
        
        for result in scan_results:
            if result.get("success") and result.get("findings"):
                all_findings.append(result["findings"])
                
                # Extract high-risk items
                findings = result["findings"]
                if isinstance(findings, dict):
                    if findings.get("risk_level") == "HIGH":
                        high_risk_items.append(f"{result.get('tool_name', 'Unknown')}: High risk detected")
                    
                    if "missing_security_headers" in findings and findings["missing_security_headers"]:
                        high_risk_items.append(f"Missing security headers: {', '.join(findings['missing_security_headers'])}")
                    
                    if "open_ports" in findings and len(findings.get("open_ports", [])) > 5:
                        high_risk_items.append(f"Multiple open ports detected: {len(findings['open_ports'])}")
        
        # Generate recommendations
        if high_risk_items:
            recommendations.extend([
                "Implement missing security headers",
                "Review and close unnecessary open ports",
                "Conduct regular security assessments",
                "Implement monitoring and alerting"
            ])
        else:
            recommendations.extend([
                "Maintain current security posture",
                "Continue regular security monitoring",
                "Consider periodic penetration testing"
            ])
        
        # Create report
        report = {
            "report_type": "Technical Security Assessment",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "executive_summary": {
                "total_scans_performed": total_scans,
                "successful_scans": successful_scans,
                "high_risk_findings": len(high_risk_items),
                "overall_risk_level": "HIGH" if len(high_risk_items) > 3 else "MEDIUM" if len(high_risk_items) > 0 else "LOW"
            },
            "detailed_findings": all_findings,
            "high_risk_items": high_risk_items,
            "recommendations": recommendations,
            "next_steps": [
                "Address high-risk findings immediately",
                "Implement recommended security controls",
                "Schedule follow-up assessment in 30 days"
            ]
        }
        
        return {
            "tool_name": "security_report_generator",
            "report_type": "technical",
            "success": True,
            "findings": report,
            "raw_output": f"Technical report generated with {len(high_risk_items)} high-risk findings",
            "execution_time": time.time() - start_time,
            "timestamp": time.time()
        }
    
    def _generate_executive_report(self, scan_results: List[Dict[str, Any]], start_time: float) -> Dict[str, Any]:
        """Generate executive summary report"""
        
        # High-level analysis
        total_scans = len(scan_results)
        issues_found = sum(1 for r in scan_results if r.get("success") and 
                          (r.get("findings", {}).get("risk_level") in ["HIGH", "MEDIUM"]))
        
        risk_level = "HIGH" if issues_found > total_scans * 0.5 else "MEDIUM" if issues_found > 0 else "LOW"
        
        report = {
            "report_type": "Executive Security Summary",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "key_metrics": {
                "security_posture": f"{risk_level} RISK",
                "assessments_completed": total_scans,
                "critical_issues": issues_found,
                "recommendation_priority": "Immediate action required" if risk_level == "HIGH" else "Address within 30 days"
            },
            "business_impact": {
                "risk_to_operations": risk_level,
                "compliance_status": "Under Review",
                "estimated_remediation_time": "2-4 weeks" if risk_level == "HIGH" else "1-2 weeks"
            },
            "executive_recommendations": [
                "Allocate budget for immediate security improvements",
                "Assign dedicated security personnel",
                "Implement ongoing security monitoring",
                "Schedule quarterly security assessments"
            ]
        }
        
        return {
            "tool_name": "security_report_generator",
            "report_type": "executive",
            "success": True,
            "findings": report,
            "raw_output": f"Executive report generated - {risk_level} risk level",
            "execution_time": time.time() - start_time,
            "timestamp": time.time()
        }
    
    def _generate_compliance_report(self, scan_results: List[Dict[str, Any]], start_time: float) -> Dict[str, Any]:
        """Generate compliance-focused report"""
        
        # Map findings to compliance frameworks
        compliance_findings = {
            "OWASP_Top_10": [],
            "NIST_Framework": [],
            "ISO_27001": [],
            "PCI_DSS": []
        }
        
        for result in scan_results:
            if result.get("success") and result.get("findings"):
                findings = result["findings"]
                
                # Map web findings to OWASP
                if "missing_security_headers" in findings:
                    compliance_findings["OWASP_Top_10"].append("A6: Security Misconfiguration")
                
                if "forms" in findings and findings.get("forms", {}).get("potential_xss_points", 0) > 0:
                    compliance_findings["OWASP_Top_10"].append("A7: Cross-Site Scripting (XSS)")
                
                # Map to other frameworks
                if "open_ports" in findings:
                    compliance_findings["NIST_Framework"].append("PR.AC: Access Control")
                    compliance_findings["ISO_27001"].append("A.13.1: Network Security Management")
        
        report = {
            "report_type": "Compliance Security Assessment",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "compliance_frameworks": compliance_findings,
            "compliance_score": {
                "OWASP_Top_10": f"{max(0, 100 - len(compliance_findings['OWASP_Top_10']) * 10)}%",
                "NIST_Framework": f"{max(0, 100 - len(compliance_findings['NIST_Framework']) * 15)}%",
                "ISO_27001": f"{max(0, 100 - len(compliance_findings['ISO_27001']) * 15)}%"
            },
            "remediation_priorities": [
                "Address OWASP Top 10 vulnerabilities",
                "Implement NIST Framework controls",
                "Ensure ISO 27001 compliance"
            ]
        }
        
        return {
            "tool_name": "security_report_generator",
            "report_type": "compliance",
            "success": True,
            "findings": report,
            "raw_output": f"Compliance report generated for {len(compliance_findings)} frameworks",
            "execution_time": time.time() - start_time,
            "timestamp": time.time()
        }
    
    def _create_error_result(self, target: str, error: str, start_time: float) -> Dict[str, Any]:
        """Create error result"""
        return {
            "tool_name": "security_report_generator",
            "target": target,
            "success": False,
            "error": error,
            "findings": {},
            "raw_output": f"Error: {error}",
            "execution_time": time.time() - start_time,
            "timestamp": time.time()
        }

# Factory functions for easy tool creation
def create_security_tools() -> List[Tool]:
    """Create all security tools for SmolAgents"""
    tools = []
    
    if SMOLAGENTS_AVAILABLE:
        tools.extend([
            NetworkReconnaissanceTool(),
            WebApplicationScannerTool(),
            ThreatIntelligenceTool(),
            SecurityReportGeneratorTool()
        ])
    
    return tools

def create_autonomous_security_agent(hf_model, max_iterations: int = 10) -> Optional['CodeAgent']:
    """Create autonomous security agent with all security tools"""
    if not SMOLAGENTS_AVAILABLE:
        return None
    
    security_tools = create_security_tools()
    
    system_prompt = """You are Archangel, an autonomous AI security expert. You perform comprehensive security assessments using available tools.

Your capabilities:
- Network reconnaissance and port scanning
- Web application security testing  
- Threat intelligence gathering
- Security report generation

Always:
1. Follow ethical hacking principles
2. Stay within authorized scope
3. Explain your reasoning clearly
4. Focus on defensive security
5. Generate actionable recommendations

Use tools systematically and correlate findings across multiple sources."""
    
    agent = CodeAgent(
        tools=security_tools,
        model=hf_model,
        max_iterations=max_iterations,
        system_prompt=system_prompt
    )
    
    return agent