#!/usr/bin/env python3
"""
Decoy Services Generator
Creates fake services and endpoints to mislead attackers
"""

import socket
import threading
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

class DecoyService:
    """Base class for decoy services"""
    
    def __init__(self, name: str, port: int, protocol: str = "tcp"):
        self.name = name
        self.port = port
        self.protocol = protocol
        self.running = False
        self.connections = []
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the decoy service"""
        logger = logging.getLogger(f"decoy_{self.name}")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(f"/var/log/honeypots/decoy_{self.name}.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def start(self):
        """Start the decoy service"""
        self.running = True
        self.logger.info(f"Starting decoy service {self.name} on port {self.port}")
        
    def stop(self):
        """Stop the decoy service"""
        self.running = False
        self.logger.info(f"Stopping decoy service {self.name}")
        
    def log_connection(self, client_ip: str, data: str = ""):
        """Log connection attempt"""
        connection_info = {
            "timestamp": datetime.now().isoformat(),
            "service": self.name,
            "client_ip": client_ip,
            "port": self.port,
            "data": data
        }
        self.connections.append(connection_info)
        self.logger.warning(f"Connection attempt from {client_ip}: {data}")

class FakeSSHService(DecoyService):
    """Fake SSH service that logs login attempts"""
    
    def __init__(self, port: int = 22):
        super().__init__("ssh", port)
        self.banner = "SSH-2.0-OpenSSH_7.4"
        
    def start(self):
        super().start()
        threading.Thread(target=self._run_server, daemon=True).start()
        
    def _run_server(self):
        """Run the fake SSH server"""
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('0.0.0.0', self.port))
            server_socket.listen(5)
            
            while self.running:
                try:
                    client_socket, client_address = server_socket.accept()
                    threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, client_address),
                        daemon=True
                    ).start()
                except Exception as e:
                    if self.running:
                        self.logger.error(f"Error accepting connection: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error starting SSH server: {e}")
            
    def _handle_client(self, client_socket: socket.socket, client_address: tuple):
        """Handle SSH client connection"""
        try:
            client_ip = client_address[0]
            
            # Send SSH banner
            client_socket.send(f"{self.banner}\r\n".encode())
            
            # Receive and log data
            data = client_socket.recv(1024).decode('utf-8', errors='ignore')
            self.log_connection(client_ip, f"SSH attempt: {data.strip()}")
            
            # Simulate authentication failure after delay
            time.sleep(2)
            client_socket.send(b"Permission denied (publickey,password).\r\n")
            
        except Exception as e:
            self.logger.error(f"Error handling SSH client {client_address}: {e}")
        finally:
            client_socket.close()

class FakeFTPService(DecoyService):
    """Fake FTP service that logs login attempts"""
    
    def __init__(self, port: int = 21):
        super().__init__("ftp", port)
        
    def start(self):
        super().start()
        threading.Thread(target=self._run_server, daemon=True).start()
        
    def _run_server(self):
        """Run the fake FTP server"""
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('0.0.0.0', self.port))
            server_socket.listen(5)
            
            while self.running:
                try:
                    client_socket, client_address = server_socket.accept()
                    threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, client_address),
                        daemon=True
                    ).start()
                except Exception as e:
                    if self.running:
                        self.logger.error(f"Error accepting FTP connection: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error starting FTP server: {e}")
            
    def _handle_client(self, client_socket: socket.socket, client_address: tuple):
        """Handle FTP client connection"""
        try:
            client_ip = client_address[0]
            
            # Send FTP welcome banner
            client_socket.send(b"220 Corporate FTP Server Ready\r\n")
            
            while True:
                data = client_socket.recv(1024).decode('utf-8', errors='ignore').strip()
                if not data:
                    break
                    
                self.log_connection(client_ip, f"FTP command: {data}")
                
                # Handle common FTP commands
                if data.upper().startswith('USER'):
                    client_socket.send(b"331 Password required\r\n")
                elif data.upper().startswith('PASS'):
                    client_socket.send(b"530 Login incorrect\r\n")
                elif data.upper().startswith('QUIT'):
                    client_socket.send(b"221 Goodbye\r\n")
                    break
                else:
                    client_socket.send(b"500 Unknown command\r\n")
                    
        except Exception as e:
            self.logger.error(f"Error handling FTP client {client_address}: {e}")
        finally:
            client_socket.close()

class FakeWebAdminPanel(DecoyService):
    """Fake web admin panel that captures login attempts"""
    
    def __init__(self, port: int = 8080):
        super().__init__("web_admin", port, "http")
        
    def start(self):
        super().start()
        threading.Thread(target=self._run_server, daemon=True).start()
        
    def _run_server(self):
        """Run the fake web admin server"""
        handler = self._create_handler()
        
        try:
            with socketserver.TCPServer(("", self.port), handler) as httpd:
                self.logger.info(f"Web admin panel running on port {self.port}")
                while self.running:
                    httpd.handle_request()
        except Exception as e:
            self.logger.error(f"Error running web admin server: {e}")
            
    def _create_handler(self):
        """Create HTTP request handler"""
        service = self
        
        class AdminPanelHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                client_ip = self.client_address[0]
                service.log_connection(client_ip, f"GET {self.path}")
                
                if self.path == "/" or self.path == "/admin":
                    self._serve_admin_panel()
                elif self.path == "/admin/forgot-password":
                    self._serve_forgot_password()
                else:
                    self._serve_404()
                    
            def do_POST(self):
                client_ip = self.client_address[0]
                
                if self.path == "/admin/login":
                    content_length = int(self.headers.get('Content-Length', 0))
                    post_data = self.rfile.read(content_length).decode('utf-8')
                    
                    # Parse login credentials
                    params = parse_qs(post_data)
                    username = params.get('username', [''])[0]
                    password = params.get('password', [''])[0]
                    
                    service.log_connection(
                        client_ip, 
                        f"LOGIN ATTEMPT - Username: {username}, Password: {password}"
                    )
                    
                    # Always return login failure
                    self._serve_login_failure()
                else:
                    self._serve_404()
                    
            def _serve_admin_panel(self):
                """Serve the fake admin panel"""
                try:
                    with open('/opt/honeypots/fake_admin_panel.html', 'r') as f:
                        content = f.read()
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(content.encode())
                except FileNotFoundError:
                    self._serve_basic_admin_panel()
                    
            def _serve_basic_admin_panel(self):
                """Serve a basic admin panel if file not found"""
                content = """
                <html>
                <head><title>Admin Panel</title></head>
                <body>
                    <h1>Corporate Admin Panel</h1>
                    <form method="post" action="/admin/login">
                        Username: <input type="text" name="username"><br>
                        Password: <input type="password" name="password"><br>
                        <input type="submit" value="Login">
                    </form>
                </body>
                </html>
                """
                
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(content.encode())
                
            def _serve_forgot_password(self):
                """Serve forgot password page"""
                content = """
                <html>
                <head><title>Password Recovery</title></head>
                <body>
                    <h1>Password Recovery</h1>
                    <p>Please contact your system administrator for password reset.</p>
                    <a href="/admin">Back to Login</a>
                </body>
                </html>
                """
                
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(content.encode())
                
            def _serve_login_failure(self):
                """Serve login failure response"""
                content = """
                <html>
                <head><title>Login Failed</title></head>
                <body>
                    <h1>Login Failed</h1>
                    <p>Invalid username or password.</p>
                    <a href="/admin">Try Again</a>
                </body>
                </html>
                """
                
                self.send_response(401)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(content.encode())
                
            def _serve_404(self):
                """Serve 404 page"""
                self.send_response(404)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"<html><body><h1>404 Not Found</h1></body></html>")
                
            def log_message(self, format, *args):
                """Override to prevent default logging"""
                pass
                
        return AdminPanelHandler

class FakeDatabaseService(DecoyService):
    """Fake database service that logs connection attempts"""
    
    def __init__(self, port: int = 3306, db_type: str = "mysql"):
        super().__init__(f"fake_{db_type}", port)
        self.db_type = db_type
        
    def start(self):
        super().start()
        threading.Thread(target=self._run_server, daemon=True).start()
        
    def _run_server(self):
        """Run the fake database server"""
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('0.0.0.0', self.port))
            server_socket.listen(5)
            
            while self.running:
                try:
                    client_socket, client_address = server_socket.accept()
                    threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, client_address),
                        daemon=True
                    ).start()
                except Exception as e:
                    if self.running:
                        self.logger.error(f"Error accepting DB connection: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error starting DB server: {e}")
            
    def _handle_client(self, client_socket: socket.socket, client_address: tuple):
        """Handle database client connection"""
        try:
            client_ip = client_address[0]
            
            if self.db_type == "mysql":
                # Send MySQL handshake
                handshake = b"\x0a5.7.0-fake\x00\x01\x00\x00\x00"
                client_socket.send(handshake)
            elif self.db_type == "postgresql":
                # PostgreSQL would handle differently
                pass
                
            # Log the connection attempt
            data = client_socket.recv(1024)
            self.log_connection(client_ip, f"DB connection attempt: {len(data)} bytes")
            
            # Simulate authentication failure
            time.sleep(1)
            client_socket.send(b"Access denied for user")
            
        except Exception as e:
            self.logger.error(f"Error handling DB client {client_address}: {e}")
        finally:
            client_socket.close()

class DecoyServiceManager:
    """Manages multiple decoy services"""
    
    def __init__(self):
        self.services: List[DecoyService] = []
        self.logger = logging.getLogger("decoy_manager")
        
    def add_service(self, service: DecoyService):
        """Add a decoy service to the manager"""
        self.services.append(service)
        
    def start_all(self):
        """Start all decoy services"""
        for service in self.services:
            try:
                service.start()
                self.logger.info(f"Started {service.name} on port {service.port}")
            except Exception as e:
                self.logger.error(f"Failed to start {service.name}: {e}")
                
    def stop_all(self):
        """Stop all decoy services"""
        for service in self.services:
            try:
                service.stop()
                self.logger.info(f"Stopped {service.name}")
            except Exception as e:
                self.logger.error(f"Failed to stop {service.name}: {e}")
                
    def get_connection_summary(self) -> Dict[str, Any]:
        """Get summary of all connections across services"""
        summary = {
            "total_connections": 0,
            "services": {},
            "top_attackers": {}
        }
        
        attacker_counts = {}
        
        for service in self.services:
            service_connections = len(service.connections)
            summary["total_connections"] += service_connections
            summary["services"][service.name] = {
                "port": service.port,
                "connections": service_connections,
                "recent_connections": service.connections[-5:] if service.connections else []
            }
            
            # Count attackers
            for conn in service.connections:
                ip = conn["client_ip"]
                attacker_counts[ip] = attacker_counts.get(ip, 0) + 1
                
        # Sort attackers by connection count
        summary["top_attackers"] = dict(
            sorted(attacker_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        return summary

def main():
    """Main function to run decoy services"""
    # Setup logging directory
    import os
    os.makedirs("/var/log/honeypots", exist_ok=True)
    
    # Create decoy service manager
    manager = DecoyServiceManager()
    
    # Add various decoy services
    manager.add_service(FakeSSHService(2222))  # Fake SSH on non-standard port
    manager.add_service(FakeFTPService(2121))  # Fake FTP on non-standard port
    manager.add_service(FakeWebAdminPanel(8080))  # Fake web admin
    manager.add_service(FakeDatabaseService(3307, "mysql"))  # Fake MySQL
    
    try:
        print("Starting decoy services...")
        manager.start_all()
        
        # Keep running and periodically show stats
        while True:
            time.sleep(60)  # Wait 1 minute
            summary = manager.get_connection_summary()
            print(f"Total connections: {summary['total_connections']}")
            
            if summary['top_attackers']:
                print("Top attackers:")
                for ip, count in list(summary['top_attackers'].items())[:3]:
                    print(f"  {ip}: {count} attempts")
                    
    except KeyboardInterrupt:
        print("Shutting down decoy services...")
        manager.stop_all()

if __name__ == "__main__":
    main()