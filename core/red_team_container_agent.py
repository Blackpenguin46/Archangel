#!/usr/bin/env python3
"""
AI-Controlled Red Team Agent running inside Kali Linux container
Executes real penetration testing tools based on AI decisions
"""

import asyncio
import subprocess
import json
import requests
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
import nmap
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - RED TEAM - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/logs/red_team_container.log')
    ]
)
logger = logging.getLogger(__name__)

class KaliRedTeamAgent:
    """AI-controlled red team agent with real Kali Linux tools"""
    
    def __init__(self):
        self.agent_id = "kali_red_team_001"
        self.target_network = "172.20.0.0/24"
        self.discovered_hosts = []
        self.active_sessions = []
        self.tools_available = self._detect_available_tools()
        
        logger.info(f"🔴 Red Team Agent {self.agent_id} initialized in Kali Linux")
        logger.info(f"Available tools: {', '.join(self.tools_available)}")
    
    def _detect_available_tools(self) -> List[str]:
        """Detect which penetration testing tools are available"""
        tools = []
        tool_commands = {
            'nmap': 'nmap --version',
            'metasploit': 'msfconsole --version',
            'sqlmap': 'sqlmap --version',
            'hydra': 'hydra -h',
            'nikto': 'nikto -Version',
            'gobuster': 'gobuster version',
            'john': 'john --version'
        }
        
        for tool, cmd in tool_commands.items():
            try:
                subprocess.run(cmd.split(), capture_output=True, check=True, timeout=5)
                tools.append(tool)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        return tools
    
    async def run_autonomous_operations(self):
        """Main loop for autonomous red team operations"""
        logger.info("🚀 Starting autonomous red team operations...")
        
        while True:
            try:
                # Get AI decision from orchestrator
                ai_decision = await self._get_ai_decision()
                
                if ai_decision:
                    # Execute the AI-decided action with real tools
                    result = await self._execute_action(ai_decision)
                    
                    # Report results back to orchestrator
                    await self._report_results(ai_decision, result)
                
                # Wait before next action
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in autonomous operations: {e}")
                await asyncio.sleep(30)
    
    async def _get_ai_decision(self) -> Dict[str, Any]:
        """Get decision from AI orchestrator"""
        try:
            # In a real implementation, this would connect to the orchestrator
            # For now, simulate AI decisions based on current state
            return await self._simulate_ai_decision()
        except Exception as e:
            logger.error(f"Failed to get AI decision: {e}")
            return None
    
    async def _simulate_ai_decision(self) -> Dict[str, Any]:
        """Simulate AI decision-making for demonstration"""
        if not self.discovered_hosts:
            return {
                'action': 'reconnaissance',
                'target': self.target_network,
                'tool': 'nmap',
                'parameters': {
                    'scan_type': 'discovery',
                    'intensity': 'stealth'
                }
            }
        else:
            target = self.discovered_hosts[0] if self.discovered_hosts else self.target_network
            return {
                'action': 'vulnerability_scan',
                'target': target,
                'tool': 'nmap',
                'parameters': {
                    'scan_type': 'vulnerability',
                    'scripts': 'vuln'
                }
            }
    
    async def _execute_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the AI decision using real penetration testing tools"""
        action = decision['action']
        tool = decision['tool']
        target = decision['target']
        params = decision.get('parameters', {})
        
        logger.info(f"🎯 Executing {action} against {target} using {tool}")
        
        try:
            if action == 'reconnaissance' and tool == 'nmap':
                return await self._run_nmap_discovery(target, params)
            elif action == 'vulnerability_scan' and tool == 'nmap':
                return await self._run_nmap_vuln_scan(target, params)
            elif action == 'web_scan' and tool == 'nikto':
                return await self._run_nikto_scan(target, params)
            elif action == 'brute_force' and tool == 'hydra':
                return await self._run_hydra_attack(target, params)
            elif action == 'sql_injection' and tool == 'sqlmap':
                return await self._run_sqlmap_attack(target, params)
            else:
                logger.warning(f"Unknown action/tool combination: {action}/{tool}")
                return {'success': False, 'error': 'Unknown action'}
                
        except Exception as e:
            logger.error(f"Error executing {action}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _run_nmap_discovery(self, target: str, params: Dict) -> Dict[str, Any]:
        """Run nmap host discovery scan"""
        try:
            logger.info(f"🔍 Running nmap discovery scan on {target}")
            
            # Use python-nmap for better integration
            nm = nmap.PortScanner()
            
            # Ping scan to discover live hosts
            scan_args = '-sn'  # Ping scan
            if params.get('intensity') == 'stealth':
                scan_args += ' -T2'  # Slow timing
            
            nm.scan(hosts=target, arguments=scan_args)
            
            discovered = []
            for host in nm.all_hosts():
                if nm[host]['status']['state'] == 'up':
                    discovered.append(host)
                    if host not in self.discovered_hosts:
                        self.discovered_hosts.append(host)
            
            result = {
                'success': True,
                'tool': 'nmap',
                'action': 'discovery',
                'hosts_discovered': discovered,
                'total_discovered': len(discovered),
                'raw_output': str(nm._scan_result)
            }
            
            logger.info(f"✅ Discovery complete: {len(discovered)} hosts found")
            return result
            
        except Exception as e:
            logger.error(f"Nmap discovery failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _run_nmap_vuln_scan(self, target: str, params: Dict) -> Dict[str, Any]:
        """Run nmap vulnerability scan"""
        try:
            logger.info(f"🔍 Running nmap vulnerability scan on {target}")
            
            nm = nmap.PortScanner()
            
            # Vulnerability scan with NSE scripts
            scan_args = '-sV --script=vuln'
            if params.get('intensity') == 'aggressive':
                scan_args += ' -T4'
            else:
                scan_args += ' -T2'
            
            nm.scan(hosts=target, arguments=scan_args)
            
            vulnerabilities = []
            for host in nm.all_hosts():
                for protocol in nm[host].all_protocols():
                    ports = nm[host][protocol].keys()
                    for port in ports:
                        port_info = nm[host][protocol][port]
                        if 'script' in port_info:
                            for script, output in port_info['script'].items():
                                if 'VULNERABLE' in output.upper():
                                    vulnerabilities.append({
                                        'host': host,
                                        'port': port,
                                        'service': port_info.get('name', 'unknown'),
                                        'vulnerability': script,
                                        'details': output
                                    })
            
            result = {
                'success': True,
                'tool': 'nmap',
                'action': 'vulnerability_scan',
                'target': target,
                'vulnerabilities_found': vulnerabilities,
                'vulnerability_count': len(vulnerabilities),
                'raw_output': str(nm._scan_result)
            }
            
            logger.info(f"✅ Vulnerability scan complete: {len(vulnerabilities)} vulnerabilities found")
            return result
            
        except Exception as e:
            logger.error(f"Nmap vulnerability scan failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _run_nikto_scan(self, target: str, params: Dict) -> Dict[str, Any]:
        """Run nikto web vulnerability scan"""
        try:
            logger.info(f"🌐 Running nikto web scan on {target}")
            
            cmd = ['nikto', '-h', target, '-Format', 'json']
            if params.get('ssl'):
                cmd.append('-ssl')
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            result = {
                'success': process.returncode == 0,
                'tool': 'nikto',
                'action': 'web_scan',
                'target': target,
                'raw_output': stdout.decode(),
                'errors': stderr.decode() if stderr else None
            }
            
            logger.info(f"✅ Nikto scan complete")
            return result
            
        except Exception as e:
            logger.error(f"Nikto scan failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _run_hydra_attack(self, target: str, params: Dict) -> Dict[str, Any]:
        """Run hydra brute force attack"""
        try:
            service = params.get('service', 'ssh')
            username_list = params.get('usernames', ['admin', 'root', 'user'])
            password_list = params.get('passwords', ['password', '123456', 'admin'])
            
            logger.info(f"💥 Running hydra brute force on {target}:{service}")
            
            # Create temporary wordlist files
            with open('/tmp/users.txt', 'w') as f:
                f.write('\n'.join(username_list))
            
            with open('/tmp/passwords.txt', 'w') as f:
                f.write('\n'.join(password_list))
            
            cmd = [
                'hydra', '-L', '/tmp/users.txt', '-P', '/tmp/passwords.txt',
                '-t', '4', '-f', target, service
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            # Clean up temp files
            os.remove('/tmp/users.txt')
            os.remove('/tmp/passwords.txt')
            
            result = {
                'success': True,
                'tool': 'hydra',
                'action': 'brute_force',
                'target': target,
                'service': service,
                'raw_output': stdout.decode(),
                'found_credentials': 'valid password' in stdout.decode().lower()
            }
            
            logger.info(f"✅ Hydra attack complete")
            return result
            
        except Exception as e:
            logger.error(f"Hydra attack failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _run_sqlmap_attack(self, target: str, params: Dict) -> Dict[str, Any]:
        """Run sqlmap SQL injection attack"""
        try:
            url = params.get('url', f"http://{target}")
            logger.info(f"💉 Running sqlmap attack on {url}")
            
            cmd = [
                'sqlmap', '-u', url, '--batch', '--risk=1', '--level=1',
                '--dbs', '--no-cast', '--disable-coloring'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            result = {
                'success': True,
                'tool': 'sqlmap',
                'action': 'sql_injection',
                'target': url,
                'raw_output': stdout.decode(),
                'injection_found': 'is vulnerable' in stdout.decode().lower()
            }
            
            logger.info(f"✅ SQLMap attack complete")
            return result
            
        except Exception as e:
            logger.error(f"SQLMap attack failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _report_results(self, decision: Dict[str, Any], result: Dict[str, Any]):
        """Report results back to orchestrator"""
        try:
            report = {
                'agent_id': self.agent_id,
                'timestamp': datetime.now().isoformat(),
                'decision': decision,
                'result': result,
                'discovered_hosts': self.discovered_hosts,
                'active_sessions': len(self.active_sessions)
            }
            
            # Log the results
            logger.info(f"📊 Action complete: {decision['action']} - Success: {result.get('success', False)}")
            
            # In real implementation, would send to orchestrator via API
            # For now, just log to file
            with open('/app/logs/red_team_results.json', 'a') as f:
                f.write(json.dumps(report) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to report results: {e}")

async def main():
    """Main entry point for red team container agent"""
    logger.info("🔴 Starting Archangel Red Team Container Agent")
    
    agent = KaliRedTeamAgent()
    
    try:
        await agent.run_autonomous_operations()
    except KeyboardInterrupt:
        logger.info("🛑 Red team agent shutting down")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())