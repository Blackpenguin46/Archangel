#!/usr/bin/env python3
"""
Honeypot Monitoring and Alert Generation System
Monitors honeypot activities and generates alerts for Blue Team agents
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import sqlite3
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import yaml

@dataclass
class HoneypotAlert:
    """Represents a honeypot alert"""
    alert_id: str
    timestamp: str
    honeypot_type: str
    source_ip: str
    target_port: int
    attack_type: str
    severity: str
    details: Dict[str, Any]
    raw_data: str
    geolocation: Optional[Dict[str, str]] = None
    threat_intel: Optional[Dict[str, Any]] = None

@dataclass
class AttackPattern:
    """Represents an identified attack pattern"""
    pattern_id: str
    attack_type: str
    source_ips: List[str]
    target_services: List[str]
    frequency: int
    first_seen: str
    last_seen: str
    confidence: float
    mitre_techniques: List[str]

class HoneypotMonitor:
    """Main honeypot monitoring system"""
    
    def __init__(self, config_path: str = "/opt/honeypots/monitor_config.yaml"):
        self.config = self._load_config(config_path)
        self.db_path = self.config.get("database", {}).get("path", "/var/lib/honeypots/monitor.db")
        self.logger = self._setup_logging()
        self.running = False
        self.alerts = []
        self.attack_patterns = {}
        
        # Initialize database
        self._init_database()
        
        # Load threat intelligence
        self.threat_intel = self._load_threat_intel()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            "monitoring": {
                "poll_interval": 30,
                "log_paths": {
                    "cowrie": "/cowrie/log/cowrie.json",
                    "dionaea": "/opt/dionaea/var/log/dionaea/dionaea.json",
                    "glastopf": "/opt/glastopf/log/glastopf.log",
                    "decoy_services": "/var/log/honeypots/"
                },
                "alert_thresholds": {
                    "high_frequency": 10,  # alerts per minute
                    "brute_force": 5,      # failed attempts
                    "port_scan": 20        # ports scanned
                }
            },
            "alerting": {
                "enabled": True,
                "channels": ["syslog", "webhook", "database"],
                "webhook_url": "http://blue-team-coordinator:8080/alerts",
                "email": {
                    "enabled": False,
                    "smtp_server": "localhost",
                    "smtp_port": 587,
                    "from_address": "honeypot@corporate.local",
                    "to_addresses": ["soc@corporate.local"]
                }
            },
            "threat_intel": {
                "enabled": True,
                "sources": ["abuseipdb", "virustotal", "otx"],
                "cache_duration": 3600
            },
            "pattern_detection": {
                "enabled": True,
                "min_events": 3,
                "time_window": 300,  # 5 minutes
                "confidence_threshold": 0.7
            }
        }
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {config_path}, using defaults")
            return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the monitor"""
        logger = logging.getLogger("honeypot_monitor")
        logger.setLevel(logging.INFO)
        
        # File handler
        os.makedirs("/var/log/honeypots", exist_ok=True)
        file_handler = logging.FileHandler("/var/log/honeypots/monitor.log")
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(file_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _init_database(self):
        """Initialize SQLite database for storing alerts and patterns"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    honeypot_type TEXT NOT NULL,
                    source_ip TEXT NOT NULL,
                    target_port INTEGER,
                    attack_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    details TEXT,
                    raw_data TEXT,
                    geolocation TEXT,
                    threat_intel TEXT
                )
            ''')
            
            # Create attack patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attack_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    attack_type TEXT NOT NULL,
                    source_ips TEXT NOT NULL,
                    target_services TEXT NOT NULL,
                    frequency INTEGER NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    mitre_techniques TEXT
                )
            ''')
            
            # Create indices for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_source_ip ON alerts(source_ip)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_attack_type ON attack_patterns(attack_type)')
            
            conn.commit()
    
    def _load_threat_intel(self) -> Dict[str, Any]:
        """Load threat intelligence data"""
        threat_intel = {
            "malicious_ips": set(),
            "known_scanners": set(),
            "tor_exits": set(),
            "malware_hashes": set()
        }
        
        # Load from local files if available
        intel_dir = "/opt/honeypots/threat_intel"
        if os.path.exists(intel_dir):
            try:
                # Load malicious IPs
                malicious_ips_file = os.path.join(intel_dir, "malicious_ips.txt")
                if os.path.exists(malicious_ips_file):
                    with open(malicious_ips_file, 'r') as f:
                        threat_intel["malicious_ips"].update(
                            line.strip() for line in f if line.strip()
                        )
                
                # Load known scanners
                scanners_file = os.path.join(intel_dir, "scanners.txt")
                if os.path.exists(scanners_file):
                    with open(scanners_file, 'r') as f:
                        threat_intel["known_scanners"].update(
                            line.strip() for line in f if line.strip()
                        )
                        
            except Exception as e:
                self.logger.error(f"Error loading threat intelligence: {e}")
        
        return threat_intel
    
    def start_monitoring(self):
        """Start the honeypot monitoring system"""
        self.running = True
        self.logger.info("Starting honeypot monitoring system")
        
        # Start monitoring threads
        threading.Thread(target=self._monitor_cowrie, daemon=True).start()
        threading.Thread(target=self._monitor_dionaea, daemon=True).start()
        threading.Thread(target=self._monitor_glastopf, daemon=True).start()
        threading.Thread(target=self._monitor_decoy_services, daemon=True).start()
        threading.Thread(target=self._pattern_detection, daemon=True).start()
        
        # Main monitoring loop
        while self.running:
            try:
                time.sleep(self.config["monitoring"]["poll_interval"])
                self._process_alerts()
                self._cleanup_old_data()
            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal, stopping...")
                self.stop_monitoring()
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def stop_monitoring(self):
        """Stop the honeypot monitoring system"""
        self.running = False
        self.logger.info("Stopping honeypot monitoring system")
    
    def _monitor_cowrie(self):
        """Monitor Cowrie SSH honeypot logs"""
        log_path = self.config["monitoring"]["log_paths"]["cowrie"]
        last_position = 0
        
        while self.running:
            try:
                if os.path.exists(log_path):
                    with open(log_path, 'r') as f:
                        f.seek(last_position)
                        
                        for line in f:
                            if not line.strip():
                                continue
                                
                            try:
                                log_entry = json.loads(line.strip())
                                alert = self._process_cowrie_log(log_entry)
                                if alert:
                                    self._add_alert(alert)
                            except json.JSONDecodeError:
                                continue
                        
                        last_position = f.tell()
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring Cowrie: {e}")
                time.sleep(10)
    
    def _process_cowrie_log(self, log_entry: Dict[str, Any]) -> Optional[HoneypotAlert]:
        """Process a Cowrie log entry and create alert if needed"""
        event_type = log_entry.get("eventid", "")
        
        if event_type in ["cowrie.login.success", "cowrie.login.failed"]:
            source_ip = log_entry.get("src_ip", "unknown")
            username = log_entry.get("username", "")
            password = log_entry.get("password", "")
            
            # Determine severity
            severity = "medium"
            if event_type == "cowrie.login.success":
                severity = "high"
            elif source_ip in self.threat_intel["malicious_ips"]:
                severity = "high"
            elif source_ip in self.threat_intel["known_scanners"]:
                severity = "medium"
            
            alert = HoneypotAlert(
                alert_id=f"cowrie_{int(time.time())}_{hash(str(log_entry)) % 10000}",
                timestamp=log_entry.get("timestamp", datetime.now().isoformat()),
                honeypot_type="cowrie_ssh",
                source_ip=source_ip,
                target_port=log_entry.get("dst_port", 22),
                attack_type="ssh_brute_force" if event_type == "cowrie.login.failed" else "ssh_compromise",
                severity=severity,
                details={
                    "username": username,
                    "password": password,
                    "success": event_type == "cowrie.login.success",
                    "session": log_entry.get("session", "")
                },
                raw_data=json.dumps(log_entry)
            )
            
            return alert
        
        return None
    
    def _monitor_dionaea(self):
        """Monitor Dionaea malware honeypot logs"""
        log_path = self.config["monitoring"]["log_paths"]["dionaea"]
        last_position = 0
        
        while self.running:
            try:
                if os.path.exists(log_path):
                    with open(log_path, 'r') as f:
                        f.seek(last_position)
                        
                        for line in f:
                            if not line.strip():
                                continue
                                
                            try:
                                log_entry = json.loads(line.strip())
                                alert = self._process_dionaea_log(log_entry)
                                if alert:
                                    self._add_alert(alert)
                            except json.JSONDecodeError:
                                continue
                        
                        last_position = f.tell()
                
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error monitoring Dionaea: {e}")
                time.sleep(10)
    
    def _process_dionaea_log(self, log_entry: Dict[str, Any]) -> Optional[HoneypotAlert]:
        """Process a Dionaea log entry and create alert if needed"""
        connection_type = log_entry.get("connection_type", "")
        
        if connection_type in ["accept", "connect"]:
            source_ip = log_entry.get("remote_host", "unknown")
            target_port = log_entry.get("local_port", 0)
            
            # Map port to service type
            service_map = {
                21: "ftp", 80: "http", 443: "https", 135: "rpc",
                445: "smb", 1433: "mssql", 3306: "mysql", 5060: "sip"
            }
            
            service = service_map.get(target_port, f"port_{target_port}")
            
            alert = HoneypotAlert(
                alert_id=f"dionaea_{int(time.time())}_{hash(str(log_entry)) % 10000}",
                timestamp=log_entry.get("timestamp", datetime.now().isoformat()),
                honeypot_type="dionaea_malware",
                source_ip=source_ip,
                target_port=target_port,
                attack_type=f"{service}_probe",
                severity="medium",
                details={
                    "service": service,
                    "connection_type": connection_type,
                    "protocol": log_entry.get("connection_protocol", "tcp")
                },
                raw_data=json.dumps(log_entry)
            )
            
            return alert
        
        return None
    
    def _monitor_glastopf(self):
        """Monitor Glastopf web honeypot logs"""
        log_path = self.config["monitoring"]["log_paths"]["glastopf"]
        
        while self.running:
            try:
                if os.path.exists(log_path):
                    # Glastopf uses text logs, parse them
                    with open(log_path, 'r') as f:
                        f.seek(0, 2)  # Go to end of file
                        
                        while self.running:
                            line = f.readline()
                            if not line:
                                time.sleep(1)
                                continue
                            
                            alert = self._process_glastopf_log(line.strip())
                            if alert:
                                self._add_alert(alert)
                
            except Exception as e:
                self.logger.error(f"Error monitoring Glastopf: {e}")
                time.sleep(10)
    
    def _process_glastopf_log(self, log_line: str) -> Optional[HoneypotAlert]:
        """Process a Glastopf log line and create alert if needed"""
        # Simple parsing of Glastopf log format
        if "attack" in log_line.lower() or "exploit" in log_line.lower():
            # Extract basic information (this would need more sophisticated parsing)
            parts = log_line.split()
            if len(parts) >= 3:
                timestamp = f"{parts[0]} {parts[1]}"
                
                alert = HoneypotAlert(
                    alert_id=f"glastopf_{int(time.time())}_{hash(log_line) % 10000}",
                    timestamp=timestamp,
                    honeypot_type="glastopf_web",
                    source_ip="unknown",  # Would need better parsing
                    target_port=80,
                    attack_type="web_exploit",
                    severity="medium",
                    details={"log_line": log_line},
                    raw_data=log_line
                )
                
                return alert
        
        return None
    
    def _monitor_decoy_services(self):
        """Monitor decoy services logs"""
        log_dir = self.config["monitoring"]["log_paths"]["decoy_services"]
        
        while self.running:
            try:
                if os.path.exists(log_dir):
                    for log_file in os.listdir(log_dir):
                        if log_file.startswith("decoy_") and log_file.endswith(".log"):
                            log_path = os.path.join(log_dir, log_file)
                            self._process_decoy_log_file(log_path)
                
                time.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error monitoring decoy services: {e}")
                time.sleep(10)
    
    def _process_decoy_log_file(self, log_path: str):
        """Process a decoy service log file"""
        try:
            with open(log_path, 'r') as f:
                for line in f:
                    if "Connection attempt" in line or "LOGIN ATTEMPT" in line:
                        alert = self._process_decoy_log_line(line.strip(), log_path)
                        if alert:
                            self._add_alert(alert)
        except Exception as e:
            self.logger.error(f"Error processing decoy log {log_path}: {e}")
    
    def _process_decoy_log_line(self, log_line: str, log_path: str) -> Optional[HoneypotAlert]:
        """Process a decoy service log line"""
        service_name = os.path.basename(log_path).replace("decoy_", "").replace(".log", "")
        
        # Extract IP address from log line
        import re
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        ip_match = re.search(ip_pattern, log_line)
        source_ip = ip_match.group() if ip_match else "unknown"
        
        alert = HoneypotAlert(
            alert_id=f"decoy_{service_name}_{int(time.time())}_{hash(log_line) % 10000}",
            timestamp=datetime.now().isoformat(),
            honeypot_type=f"decoy_{service_name}",
            source_ip=source_ip,
            target_port=0,  # Would need to extract from service config
            attack_type=f"{service_name}_probe",
            severity="low",
            details={"service": service_name, "log_line": log_line},
            raw_data=log_line
        )
        
        return alert
    
    def _add_alert(self, alert: HoneypotAlert):
        """Add an alert to the system"""
        self.alerts.append(alert)
        
        # Store in database
        self._store_alert_in_db(alert)
        
        # Send alert to configured channels
        self._send_alert(alert)
        
        self.logger.info(f"Generated alert: {alert.alert_id} - {alert.attack_type} from {alert.source_ip}")
    
    def _store_alert_in_db(self, alert: HoneypotAlert):
        """Store alert in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO alerts 
                    (alert_id, timestamp, honeypot_type, source_ip, target_port, 
                     attack_type, severity, details, raw_data, geolocation, threat_intel)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.alert_id, alert.timestamp, alert.honeypot_type,
                    alert.source_ip, alert.target_port, alert.attack_type,
                    alert.severity, json.dumps(alert.details), alert.raw_data,
                    json.dumps(alert.geolocation) if alert.geolocation else None,
                    json.dumps(alert.threat_intel) if alert.threat_intel else None
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error storing alert in database: {e}")
    
    def _send_alert(self, alert: HoneypotAlert):
        """Send alert to configured channels"""
        if not self.config["alerting"]["enabled"]:
            return
        
        channels = self.config["alerting"]["channels"]
        
        if "webhook" in channels:
            self._send_webhook_alert(alert)
        
        if "syslog" in channels:
            self._send_syslog_alert(alert)
        
        if "email" in channels and self.config["alerting"]["email"]["enabled"]:
            self._send_email_alert(alert)
    
    def _send_webhook_alert(self, alert: HoneypotAlert):
        """Send alert via webhook to Blue Team coordinator"""
        try:
            webhook_url = self.config["alerting"]["webhook_url"]
            payload = {
                "alert_type": "honeypot_activity",
                "alert": asdict(alert),
                "timestamp": datetime.now().isoformat()
            }
            
            response = requests.post(
                webhook_url,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                self.logger.debug(f"Webhook alert sent successfully for {alert.alert_id}")
            else:
                self.logger.warning(f"Webhook alert failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error sending webhook alert: {e}")
    
    def _send_syslog_alert(self, alert: HoneypotAlert):
        """Send alert to syslog"""
        try:
            import syslog
            syslog.openlog("honeypot_monitor")
            
            message = (f"HONEYPOT ALERT: {alert.attack_type} from {alert.source_ip} "
                      f"on {alert.honeypot_type} (severity: {alert.severity})")
            
            # Map severity to syslog priority
            priority_map = {
                "low": syslog.LOG_INFO,
                "medium": syslog.LOG_WARNING,
                "high": syslog.LOG_ERR,
                "critical": syslog.LOG_CRIT
            }
            
            priority = priority_map.get(alert.severity, syslog.LOG_WARNING)
            syslog.syslog(priority, message)
            
        except Exception as e:
            self.logger.error(f"Error sending syslog alert: {e}")
    
    def _send_email_alert(self, alert: HoneypotAlert):
        """Send alert via email"""
        try:
            email_config = self.config["alerting"]["email"]
            
            msg = MimeMultipart()
            msg['From'] = email_config["from_address"]
            msg['To'] = ", ".join(email_config["to_addresses"])
            msg['Subject'] = f"Honeypot Alert: {alert.attack_type} from {alert.source_ip}"
            
            body = f"""
Honeypot Alert Details:

Alert ID: {alert.alert_id}
Timestamp: {alert.timestamp}
Honeypot Type: {alert.honeypot_type}
Source IP: {alert.source_ip}
Target Port: {alert.target_port}
Attack Type: {alert.attack_type}
Severity: {alert.severity}

Details:
{json.dumps(alert.details, indent=2)}

Raw Data:
{alert.raw_data}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            self.logger.error(f"Error sending email alert: {e}")
    
    def _pattern_detection(self):
        """Detect attack patterns from alerts"""
        while self.running:
            try:
                if self.config["pattern_detection"]["enabled"]:
                    self._analyze_attack_patterns()
                time.sleep(60)  # Run pattern detection every minute
            except Exception as e:
                self.logger.error(f"Error in pattern detection: {e}")
    
    def _analyze_attack_patterns(self):
        """Analyze recent alerts for attack patterns"""
        # Get recent alerts from database
        time_window = self.config["pattern_detection"]["time_window"]
        cutoff_time = datetime.now() - timedelta(seconds=time_window)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM alerts 
                    WHERE timestamp > ? 
                    ORDER BY timestamp DESC
                ''', (cutoff_time.isoformat(),))
                
                recent_alerts = cursor.fetchall()
                
                if len(recent_alerts) >= self.config["pattern_detection"]["min_events"]:
                    patterns = self._identify_patterns(recent_alerts)
                    
                    for pattern in patterns:
                        if pattern.confidence >= self.config["pattern_detection"]["confidence_threshold"]:
                            self._store_attack_pattern(pattern)
                            self._alert_on_pattern(pattern)
                            
        except Exception as e:
            self.logger.error(f"Error analyzing attack patterns: {e}")
    
    def _identify_patterns(self, alerts: List[tuple]) -> List[AttackPattern]:
        """Identify attack patterns from alerts"""
        patterns = []
        
        # Group alerts by source IP
        ip_groups = {}
        for alert in alerts:
            source_ip = alert[3]  # source_ip column
            if source_ip not in ip_groups:
                ip_groups[source_ip] = []
            ip_groups[source_ip].append(alert)
        
        # Analyze each IP group for patterns
        for source_ip, ip_alerts in ip_groups.items():
            if len(ip_alerts) >= 3:  # Minimum events for pattern
                # Analyze for brute force pattern
                attack_types = [alert[5] for alert in ip_alerts]  # attack_type column
                target_services = [alert[2] for alert in ip_alerts]  # honeypot_type column
                
                if len(set(attack_types)) == 1 and len(ip_alerts) >= 5:
                    # Potential brute force pattern
                    pattern = AttackPattern(
                        pattern_id=f"pattern_{source_ip}_{int(time.time())}",
                        attack_type="brute_force",
                        source_ips=[source_ip],
                        target_services=list(set(target_services)),
                        frequency=len(ip_alerts),
                        first_seen=min(alert[1] for alert in ip_alerts),  # timestamp column
                        last_seen=max(alert[1] for alert in ip_alerts),
                        confidence=min(1.0, len(ip_alerts) / 10.0),
                        mitre_techniques=["T1110"]  # Brute Force
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _store_attack_pattern(self, pattern: AttackPattern):
        """Store attack pattern in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO attack_patterns
                    (pattern_id, attack_type, source_ips, target_services, 
                     frequency, first_seen, last_seen, confidence, mitre_techniques)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern.pattern_id, pattern.attack_type,
                    json.dumps(pattern.source_ips), json.dumps(pattern.target_services),
                    pattern.frequency, pattern.first_seen, pattern.last_seen,
                    pattern.confidence, json.dumps(pattern.mitre_techniques)
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error storing attack pattern: {e}")
    
    def _alert_on_pattern(self, pattern: AttackPattern):
        """Generate alert for detected attack pattern"""
        pattern_alert = HoneypotAlert(
            alert_id=f"pattern_{pattern.pattern_id}",
            timestamp=datetime.now().isoformat(),
            honeypot_type="pattern_detection",
            source_ip=pattern.source_ips[0] if pattern.source_ips else "multiple",
            target_port=0,
            attack_type=f"pattern_{pattern.attack_type}",
            severity="high",
            details={
                "pattern_id": pattern.pattern_id,
                "attack_type": pattern.attack_type,
                "source_ips": pattern.source_ips,
                "target_services": pattern.target_services,
                "frequency": pattern.frequency,
                "confidence": pattern.confidence,
                "mitre_techniques": pattern.mitre_techniques
            },
            raw_data=json.dumps(asdict(pattern))
        )
        
        self._send_alert(pattern_alert)
        self.logger.warning(f"Attack pattern detected: {pattern.attack_type} from {pattern.source_ips}")
    
    def _process_alerts(self):
        """Process and correlate alerts"""
        # This could include additional alert processing logic
        pass
    
    def _cleanup_old_data(self):
        """Clean up old alerts and patterns from database"""
        try:
            # Keep data for 30 days
            cutoff_time = datetime.now() - timedelta(days=30)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clean up old alerts
                cursor.execute('DELETE FROM alerts WHERE timestamp < ?', (cutoff_time.isoformat(),))
                
                # Clean up old patterns
                cursor.execute('DELETE FROM attack_patterns WHERE last_seen < ?', (cutoff_time.isoformat(),))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of recent alerts"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get alert counts by type
                cursor.execute('''
                    SELECT attack_type, COUNT(*) as count
                    FROM alerts 
                    WHERE timestamp > datetime('now', '-24 hours')
                    GROUP BY attack_type
                    ORDER BY count DESC
                ''')
                attack_types = dict(cursor.fetchall())
                
                # Get top source IPs
                cursor.execute('''
                    SELECT source_ip, COUNT(*) as count
                    FROM alerts 
                    WHERE timestamp > datetime('now', '-24 hours')
                    GROUP BY source_ip
                    ORDER BY count DESC
                    LIMIT 10
                ''')
                top_ips = dict(cursor.fetchall())
                
                # Get severity distribution
                cursor.execute('''
                    SELECT severity, COUNT(*) as count
                    FROM alerts 
                    WHERE timestamp > datetime('now', '-24 hours')
                    GROUP BY severity
                ''')
                severity_dist = dict(cursor.fetchall())
                
                return {
                    "attack_types": attack_types,
                    "top_source_ips": top_ips,
                    "severity_distribution": severity_dist,
                    "total_alerts_24h": sum(attack_types.values())
                }
                
        except Exception as e:
            self.logger.error(f"Error getting alert summary: {e}")
            return {}

def main():
    """Main function to run honeypot monitor"""
    monitor = HoneypotMonitor()
    
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("Shutting down honeypot monitor...")
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()