#!/usr/bin/env python3
"""
Honeytoken Distribution System
Generates and distributes fake credentials, documents, and other deceptive assets
"""

import os
import json
import random
import string
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any
import yaml

class HoneytokenGenerator:
    """Generates various types of honeytokens for deception"""
    
    def __init__(self, config_path: str = "/opt/honeytokens/config.yaml"):
        self.config = self._load_config(config_path)
        self.generated_tokens = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load honeytoken configuration"""
        default_config = {
            "credentials": {
                "usernames": ["admin", "administrator", "root", "service", "backup", "test"],
                "domains": ["corporate.local", "company.com", "internal.net"],
                "password_patterns": ["Password123!", "Admin2024", "Service@123", "Backup!@#"]
            },
            "documents": {
                "types": ["passwords.txt", "config.ini", "database.conf", "api_keys.json"],
                "locations": ["/tmp", "/var/tmp", "/home/admin", "/opt/backup"]
            },
            "api_keys": {
                "services": ["aws", "azure", "gcp", "github", "slack", "jira"],
                "key_length": 32
            },
            "certificates": {
                "subjects": ["CN=admin", "CN=service", "CN=backup"],
                "key_sizes": [2048, 4096]
            }
        }
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            return default_config
    
    def generate_fake_credentials(self, count: int = 10) -> List[Dict[str, str]]:
        """Generate fake username/password combinations"""
        credentials = []
        
        for _ in range(count):
            username = random.choice(self.config["credentials"]["usernames"])
            domain = random.choice(self.config["credentials"]["domains"])
            password = random.choice(self.config["credentials"]["password_patterns"])
            
            # Add some variation to usernames
            if random.random() < 0.3:
                username += str(random.randint(1, 99))
            
            credential = {
                "username": username,
                "domain": domain,
                "password": password,
                "full_username": f"{username}@{domain}",
                "created": datetime.now().isoformat(),
                "type": "credential_honeytoken"
            }
            
            credentials.append(credential)
            self.generated_tokens.append(credential)
        
        return credentials
    
    def generate_fake_api_keys(self, count: int = 5) -> List[Dict[str, str]]:
        """Generate fake API keys for various services"""
        api_keys = []
        
        for _ in range(count):
            service = random.choice(self.config["api_keys"]["services"])
            key_length = self.config["api_keys"]["key_length"]
            
            # Generate realistic-looking API key
            if service == "aws":
                key = "AKIA" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
                secret = ''.join(random.choices(string.ascii_letters + string.digits + "+/", k=40))
            elif service == "github":
                key = "ghp_" + ''.join(random.choices(string.ascii_letters + string.digits, k=36))
                secret = None
            else:
                key = ''.join(random.choices(string.ascii_letters + string.digits, k=key_length))
                secret = ''.join(random.choices(string.ascii_letters + string.digits, k=key_length))
            
            api_key = {
                "service": service,
                "api_key": key,
                "api_secret": secret,
                "created": datetime.now().isoformat(),
                "type": "api_key_honeytoken"
            }
            
            api_keys.append(api_key)
            self.generated_tokens.append(api_key)
        
        return api_keys
    
    def generate_fake_documents(self, count: int = 8) -> List[Dict[str, str]]:
        """Generate fake sensitive documents"""
        documents = []
        
        for _ in range(count):
            doc_type = random.choice(self.config["documents"]["types"])
            location = random.choice(self.config["documents"]["locations"])
            
            # Generate document content based on type
            if doc_type == "passwords.txt":
                content = self._generate_password_file()
            elif doc_type == "config.ini":
                content = self._generate_config_file()
            elif doc_type == "database.conf":
                content = self._generate_database_config()
            elif doc_type == "api_keys.json":
                content = self._generate_api_keys_file()
            else:
                content = "# Sensitive configuration file\n# DO NOT SHARE\n"
            
            document = {
                "filename": doc_type,
                "location": location,
                "full_path": os.path.join(location, doc_type),
                "content": content,
                "size": len(content),
                "created": datetime.now().isoformat(),
                "type": "document_honeytoken"
            }
            
            documents.append(document)
            self.generated_tokens.append(document)
        
        return documents
    
    def _generate_password_file(self) -> str:
        """Generate fake password file content"""
        passwords = [
            "admin:Password123!",
            "root:RootPass2024",
            "service:ServiceAccount@123",
            "backup:BackupUser!@#",
            "database:DbAdmin2024!",
            "api:ApiKey123456",
            "ftp:FtpUser@789",
            "mail:MailServer123"
        ]
        return "\n".join(passwords) + "\n"
    
    def _generate_config_file(self) -> str:
        """Generate fake configuration file content"""
        config = """[database]
host=192.168.20.10
username=dbadmin
password=DbPassword123!
database=corporate

[api]
endpoint=https://api.corporate.com
key=ak_live_1234567890abcdef
secret=sk_live_abcdef1234567890

[smtp]
server=mail.corporate.com
username=noreply@corporate.com
password=MailPass2024!
"""
        return config
    
    def _generate_database_config(self) -> str:
        """Generate fake database configuration"""
        config = """# Database Configuration
DB_HOST=192.168.20.10
DB_PORT=3306
DB_NAME=corporate_db
DB_USER=admin
DB_PASS=AdminPass123!
DB_ROOT_PASS=RootPassword2024!

# Backup Configuration
BACKUP_HOST=192.168.20.50
BACKUP_USER=backup
BACKUP_PASS=BackupUser@123
"""
        return config
    
    def _generate_api_keys_file(self) -> str:
        """Generate fake API keys JSON file"""
        keys = {
            "aws": {
                "access_key": "AKIA1234567890ABCDEF",
                "secret_key": "abcdef1234567890abcdef1234567890abcdef12"
            },
            "github": {
                "token": "ghp_1234567890abcdef1234567890abcdef123456"
            },
            "slack": {
                "webhook": "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
            }
        }
        return json.dumps(keys, indent=2)
    
    def distribute_honeytokens(self, target_paths: List[str]) -> Dict[str, List[str]]:
        """Distribute honeytokens to specified locations"""
        distribution_log = {"success": [], "failed": []}
        
        # Generate all types of honeytokens
        credentials = self.generate_fake_credentials()
        api_keys = self.generate_fake_api_keys()
        documents = self.generate_fake_documents()
        
        # Save credentials to files
        for path in target_paths:
            try:
                os.makedirs(path, exist_ok=True)
                
                # Save credentials file
                cred_file = os.path.join(path, "credentials.json")
                with open(cred_file, 'w') as f:
                    json.dump(credentials, f, indent=2)
                distribution_log["success"].append(cred_file)
                
                # Save API keys file
                api_file = os.path.join(path, "api_keys.json")
                with open(api_file, 'w') as f:
                    json.dump(api_keys, f, indent=2)
                distribution_log["success"].append(api_file)
                
                # Save document files
                for doc in documents:
                    doc_path = os.path.join(path, doc["filename"])
                    with open(doc_path, 'w') as f:
                        f.write(doc["content"])
                    distribution_log["success"].append(doc_path)
                    
            except Exception as e:
                distribution_log["failed"].append(f"{path}: {str(e)}")
        
        return distribution_log
    
    def generate_monitoring_config(self) -> Dict[str, Any]:
        """Generate configuration for monitoring honeytoken access"""
        config = {
            "monitoring": {
                "enabled": True,
                "log_file": "/var/log/honeytokens/access.log",
                "alert_threshold": 1,  # Alert on any access
                "notification_channels": ["syslog", "webhook"]
            },
            "tokens": [
                {
                    "id": token.get("id", hashlib.md5(str(token).encode()).hexdigest()[:8]),
                    "type": token["type"],
                    "created": token["created"],
                    "monitored": True
                }
                for token in self.generated_tokens
            ]
        }
        return config

def main():
    """Main function for testing honeytoken generation"""
    generator = HoneytokenGenerator()
    
    # Generate honeytokens
    print("Generating honeytokens...")
    
    credentials = generator.generate_fake_credentials(5)
    print(f"Generated {len(credentials)} fake credentials")
    
    api_keys = generator.generate_fake_api_keys(3)
    print(f"Generated {len(api_keys)} fake API keys")
    
    documents = generator.generate_fake_documents(4)
    print(f"Generated {len(documents)} fake documents")
    
    # Distribute to test locations
    test_paths = ["/tmp/honeytokens", "/var/tmp/honeytokens"]
    distribution_log = generator.distribute_honeytokens(test_paths)
    
    print(f"Successfully distributed to: {distribution_log['success']}")
    if distribution_log['failed']:
        print(f"Failed distributions: {distribution_log['failed']}")
    
    # Generate monitoring config
    monitoring_config = generator.generate_monitoring_config()
    print(f"Generated monitoring config for {len(monitoring_config['tokens'])} tokens")

if __name__ == "__main__":
    main()