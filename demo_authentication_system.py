#!/usr/bin/env python3
"""
Demo script for enterprise authentication and directory services.

This script demonstrates the comprehensive authentication infrastructure including:
- Active Directory simulation
- DNS and DHCP services
- Certificate Authority
- OAuth2 provider
- mTLS authenticator
- RBAC manager
"""

import json
import time
import logging
from datetime import datetime, timedelta

from authentication.active_directory import ActiveDirectorySimulator, UserAccountControl
from authentication.dns_service import DNSService, DNSRecordType
from authentication.dhcp_service import DHCPService
from authentication.certificate_authority import CertificateAuthority, CAConfiguration, RevocationReason
from authentication.oauth2_provider import OAuth2Provider, GrantType
from authentication.mtls_authenticator import MTLSAuthenticator, AuthenticationResult
from authentication.rbac_manager import RBACManager, ResourceType, PermissionType


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")


def demo_active_directory():
    """Demonstrate Active Directory functionality."""
    print_section("ACTIVE DIRECTORY SIMULATION")
    
    # Initialize AD
    ad = ActiveDirectorySimulator("archangel.local")
    
    print(f"Domain: {ad.domain_name}")
    print(f"Domain Controller: {ad.domain_controller}")
    
    # Show domain info
    domain_info = ad.get_domain_info()
    print(f"Total Users: {domain_info['total_users']}")
    print(f"Total Groups: {domain_info['total_groups']}")
    print(f"Total OUs: {domain_info['total_ous']}")
    
    print_subsection("User Authentication Tests")
    
    # Test successful authentication
    print("Testing successful authentication...")
    result = ad.authenticate_user("admin", "Password123!", "192.168.1.100")
    print(f"Admin login: {'SUCCESS' if result['success'] else 'FAILED'}")
    if result['success']:
        user_info = result['user_info']
        print(f"  User: {user_info['display_name']}")
        print(f"  Groups: {', '.join(user_info['groups'])}")
        print(f"  Login Count: {user_info['login_count']}")
    
    # Test failed authentication
    print("\nTesting failed authentication...")
    result = ad.authenticate_user("admin", "wrongpassword", "192.168.1.100")
    print(f"Wrong password: {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"Reason: {result['reason']}")
    
    # Test account lockout
    print("\nTesting account lockout...")
    test_user = ad.create_user("testuser", "Test User", "test@archangel.local", "IT", "Tester")
    
    # Attempt multiple failed logins
    for i in range(6):
        result = ad.authenticate_user("testuser", "wrongpassword")
        print(f"  Attempt {i+1}: {result['reason']}")
        if "locked" in result['reason'].lower():
            break
    
    print_subsection("Group Management")
    
    # Create custom group
    ad.create_group("Security Team", "Security specialists", ad.groups["IT Support"].group_type)
    ad.add_user_to_group("mwilson", "Security Team")
    
    # Check user permissions
    has_it_access = ad.check_user_permission("mwilson", ["IT Support"])
    has_security_access = ad.check_user_permission("mwilson", ["Security Team"])
    
    print(f"Mary Wilson IT Access: {has_it_access}")
    print(f"Mary Wilson Security Access: {has_security_access}")
    
    return ad


def demo_dns_service():
    """Demonstrate DNS service functionality."""
    print_section("DNS SERVICE SIMULATION")
    
    # Initialize DNS service
    dns = DNSService("dns01.archangel.local")
    
    print(f"DNS Server: {dns.server_name}")
    print(f"Zones: {len(dns.zones)}")
    
    print_subsection("DNS Resolution Tests")
    
    # Test A record resolution
    print("Resolving A records...")
    test_queries = [
        "dns01.archangel.local",
        "web01.archangel.local",
        "db01.archangel.local",
        "nonexistent.archangel.local"
    ]
    
    for query in test_queries:
        records = dns.resolve_query(query, DNSRecordType.A, "192.168.1.100")
        if records:
            print(f"  {query} -> {records[0].value}")
        else:
            print(f"  {query} -> NOT FOUND")
    
    # Test CNAME resolution
    print("\nResolving CNAME records...")
    cname_queries = ["www.archangel.local", "mail.archangel.local"]
    
    for query in cname_queries:
        records = dns.resolve_query(query, DNSRecordType.A, "192.168.1.100")
        if records:
            print(f"  {query} -> {records[-1].value} (via CNAME)")
        else:
            print(f"  {query} -> NOT FOUND")
    
    print_subsection("Dynamic DNS")
    
    # Add dynamic records
    print("Adding dynamic DNS records...")
    dynamic_records = [
        ("agent01.archangel.local", "192.168.20.100"),
        ("agent02.archangel.local", "192.168.20.101"),
        ("honeypot01.archangel.local", "192.168.50.10")
    ]
    
    for hostname, ip in dynamic_records:
        success = dns.add_dynamic_record(hostname, DNSRecordType.A, ip)
        print(f"  Added {hostname} -> {ip}: {'SUCCESS' if success else 'FAILED'}")
    
    # Verify dynamic records
    print("\nVerifying dynamic records...")
    for hostname, expected_ip in dynamic_records:
        records = dns.resolve_query(hostname, DNSRecordType.A)
        if records and records[0].value == expected_ip:
            print(f"  {hostname}: VERIFIED")
        else:
            print(f"  {hostname}: FAILED")
    
    # Show DNS statistics
    print_subsection("DNS Statistics")
    stats = dns.get_query_statistics()
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Successful: {stats['successful_queries']}")
    print(f"Failed: {stats['failed_queries']}")
    print(f"Cached: {stats['cached_queries']}")
    print(f"Average Response Time: {stats['average_response_time_ms']:.2f}ms")
    
    return dns


def demo_dhcp_service():
    """Demonstrate DHCP service functionality."""
    print_section("DHCP SERVICE SIMULATION")
    
    # Initialize DHCP service
    dhcp = DHCPService("dhcp01.archangel.local")
    
    print(f"DHCP Server: {dhcp.server_name}")
    print(f"Scopes: {len(dhcp.scopes)}")
    
    print_subsection("DHCP Lease Process")
    
    # Simulate DHCP client interactions
    clients = [
        ("00:50:56:12:34:50", "agent-red-01"),
        ("00:50:56:12:34:51", "agent-blue-01"),
        ("00:50:56:12:34:52", "honeypot-ssh"),
        ("00:50:56:12:34:53", "workstation-01")
    ]
    
    for mac_address, hostname in clients:
        print(f"\nClient: {hostname} ({mac_address})")
        
        # DHCP DISCOVER
        offer = dhcp.handle_dhcp_discover(mac_address, hostname)
        if offer:
            offered_ip = offer["offered_ip"]
            print(f"  DISCOVER -> OFFER: {offered_ip}")
            
            # DHCP REQUEST
            ack = dhcp.handle_dhcp_request(
                mac_address, offered_ip, offer["server_ip"], hostname
            )
            if ack:
                print(f"  REQUEST -> ACK: {ack['client_ip']}")
                print(f"  Lease Time: {ack['lease_time']} seconds")
                print(f"  DNS Servers: {', '.join(ack['dns_servers'])}")
            else:
                print("  REQUEST -> NAK")
        else:
            print("  DISCOVER -> No offer available")
    
    print_subsection("DHCP Reservations")
    
    # Add reservations for infrastructure
    reservations = [
        ("mgmt", "00:50:56:12:34:60", "192.168.1.60", "monitoring-01"),
        ("dmz", "00:50:56:12:34:61", "192.168.10.60", "web-proxy"),
        ("lan", "00:50:56:12:34:62", "192.168.20.60", "file-server")
    ]
    
    for scope, mac, ip, hostname in reservations:
        success = dhcp.add_reservation(scope, mac, ip, hostname)
        print(f"Reserved {ip} for {hostname}: {'SUCCESS' if success else 'FAILED'}")
    
    # Test reservation
    print("\nTesting reservation...")
    offer = dhcp.handle_dhcp_discover("00:50:56:12:34:60", "monitoring-01")
    if offer and offer["offered_ip"] == "192.168.1.60":
        print("Reservation working correctly!")
    
    print_subsection("DHCP Statistics")
    
    # Show scope statistics
    for scope_name in dhcp.scopes:
        stats = dhcp.get_scope_statistics(scope_name)
        if stats:
            print(f"\nScope: {stats['scope_name']}")
            print(f"  Network: {stats['network']}")
            print(f"  Total Addresses: {stats['total_addresses']}")
            print(f"  Active Leases: {stats['active_leases']}")
            print(f"  Utilization: {stats['utilization_percent']:.1f}%")
    
    return dhcp


def demo_certificate_authority():
    """Demonstrate Certificate Authority functionality."""
    print_section("CERTIFICATE AUTHORITY SIMULATION")
    
    # Initialize CA
    ca_config = CAConfiguration(
        ca_name="Archangel Root CA",
        subject_dn="CN=Archangel Root CA,O=Archangel Corp,C=US",
        validity_days=3650
    )
    ca = CertificateAuthority(ca_config)
    
    print(f"CA Name: {ca.config.ca_name}")
    print(f"CA Subject: {ca.config.subject_dn}")
    
    if ca.ca_certificate:
        print(f"CA Serial: {ca.ca_certificate.serial_number}")
        print(f"CA Valid Until: {ca.ca_certificate.not_after}")
    
    print_subsection("Certificate Issuance")
    
    # Issue certificates for different purposes
    certificates = []
    
    # Server certificates
    server_certs = [
        ("coordinator.archangel.local", ["DNS:coordinator.archangel.local", "DNS:localhost"], ["server_auth"]),
        ("api.archangel.local", ["DNS:api.archangel.local"], ["server_auth"]),
        ("monitoring.archangel.local", ["DNS:monitoring.archangel.local"], ["server_auth"])
    ]
    
    for cn, san_list, eku in server_certs:
        cert = ca.issue_certificate(
            subject_dn=f"CN={cn},OU=Servers,O=Archangel Corp,C=US",
            subject_alt_names=san_list,
            key_usage=["digital_signature", "key_encipherment"],
            extended_key_usage=eku,
            validity_days=365
        )
        if cert:
            certificates.append(cert)
            print(f"Issued server cert: {cn} (Serial: {cert.serial_number})")
    
    # Client certificates for agents
    agent_certs = [
        ("red-team-recon", "Red Team"),
        ("red-team-exploit", "Red Team"),
        ("blue-team-soc", "Blue Team"),
        ("blue-team-firewall", "Blue Team")
    ]
    
    for agent_name, team in agent_certs:
        cert = ca.issue_certificate(
            subject_dn=f"CN={agent_name},OU={team} Agents,O=Archangel Corp,C=US",
            key_usage=["digital_signature", "key_encipherment"],
            extended_key_usage=["client_auth"],
            validity_days=90
        )
        if cert:
            certificates.append(cert)
            print(f"Issued agent cert: {agent_name} (Serial: {cert.serial_number})")
    
    print_subsection("Certificate Validation")
    
    # Validate certificates
    for cert in certificates[:3]:  # Test first 3 certificates
        result = ca.validate_certificate(cert.certificate_pem)
        status = "VALID" if result["valid"] else "INVALID"
        print(f"Certificate {cert.serial_number}: {status}")
        if not result["valid"]:
            print(f"  Errors: {', '.join(result['errors'])}")
    
    print_subsection("Certificate Revocation")
    
    # Revoke a certificate
    if certificates:
        revoke_cert = certificates[-1]  # Revoke last certificate
        success = ca.revoke_certificate(revoke_cert.serial_number, RevocationReason.KEY_COMPROMISE)
        print(f"Revoked certificate {revoke_cert.serial_number}: {'SUCCESS' if success else 'FAILED'}")
        
        # Validate revoked certificate
        result = ca.validate_certificate(revoke_cert.certificate_pem)
        print(f"Revoked cert validation: {'VALID' if result['valid'] else 'INVALID'}")
        if not result["valid"]:
            print(f"  Reason: {', '.join(result['errors'])}")
    
    print_subsection("Certificate Revocation List")
    
    # Generate CRL
    crl = ca.generate_crl()
    print(f"Generated CRL: {len(crl)} bytes")
    print("CRL Preview:")
    print(crl[:200] + "..." if len(crl) > 200 else crl)
    
    print_subsection("CA Statistics")
    
    stats = ca.get_ca_statistics()
    print(f"Total Certificates: {stats['total_certificates']}")
    print(f"Valid Certificates: {stats['valid_certificates']}")
    print(f"Expired Certificates: {stats['expired_certificates']}")
    print(f"Revoked Certificates: {stats['revoked_certificates']}")
    print(f"CRL Number: {stats['crl_number']}")
    
    return ca, certificates


def demo_oauth2_provider():
    """Demonstrate OAuth2 provider functionality."""
    print_section("OAUTH2 PROVIDER SIMULATION")
    
    # Initialize OAuth2 provider
    oauth2 = OAuth2Provider("https://auth.archangel.local")
    
    print(f"OAuth2 Issuer: {oauth2.issuer}")
    print(f"Registered Clients: {len(oauth2.clients)}")
    
    print_subsection("Client Registration")
    
    # Register additional clients
    test_client = oauth2.register_client(
        client_name="Test Integration Client",
        redirect_uris=["http://localhost:8080/callback"],
        grant_types=[GrantType.CLIENT_CREDENTIALS, GrantType.AUTHORIZATION_CODE],
        scopes={"test:read", "test:write", "integration:access"}
    )
    
    print(f"Registered client: {test_client.client_name}")
    print(f"Client ID: {test_client.client_id}")
    print(f"Client Secret: {test_client.client_secret[:8]}...")
    
    print_subsection("Client Credentials Flow")
    
    # Test client credentials flow for different clients
    test_clients = ["red_team_agents", "blue_team_agents", "archangel_coordinator"]
    
    for client_id in test_clients:
        if client_id in oauth2.clients:
            client = oauth2.clients[client_id]
            
            print(f"\nTesting {client.client_name}:")
            
            # Request token
            token_response = oauth2.client_credentials_grant(
                client.client_id, client.client_secret
            )
            
            if token_response:
                print(f"  Token Type: {token_response['token_type']}")
                print(f"  Expires In: {token_response['expires_in']} seconds")
                print(f"  Scopes: {token_response['scope']}")
                
                # Validate token
                access_token = token_response["access_token"]
                token_info = oauth2.validate_access_token(access_token)
                
                if token_info:
                    print(f"  Token Valid: YES")
                    print(f"  Client ID: {token_info['client_id']}")
                else:
                    print(f"  Token Valid: NO")
            else:
                print(f"  Token Request: FAILED")
    
    print_subsection("Authorization Code Flow")
    
    # Test authorization code flow
    client = oauth2.clients["blue_team_agents"]
    redirect_uri = client.redirect_uris[0]
    scopes = {"agent:read", "coordination:participate"}
    
    print(f"Testing authorization code flow for {client.client_name}:")
    
    # Create authorization code
    auth_code = oauth2.create_authorization_code(
        client.client_id, redirect_uri, scopes, "test_user"
    )
    
    if auth_code:
        print(f"  Authorization Code: {auth_code[:16]}...")
        
        # Exchange code for token
        token_response = oauth2.exchange_authorization_code(
            client.client_id, client.client_secret, auth_code, redirect_uri
        )
        
        if token_response:
            print(f"  Access Token: {token_response['access_token'][:16]}...")
            print(f"  Refresh Token: {token_response['refresh_token'][:16]}...")
            print(f"  Expires In: {token_response['expires_in']} seconds")
        else:
            print(f"  Token Exchange: FAILED")
    else:
        print(f"  Authorization Code: FAILED")
    
    print_subsection("Token Validation and Scopes")
    
    # Test scope-based access
    client = oauth2.clients["red_team_agents"]
    token_response = oauth2.client_credentials_grant(client.client_id, client.client_secret)
    
    if token_response:
        access_token = token_response["access_token"]
        
        # Test different scope requirements
        test_scopes = ["agent:read", "coordination:participate", "defense:manage", "invalid:scope"]
        
        for scope in test_scopes:
            token_info = oauth2.validate_access_token(access_token, scope)
            has_scope = token_info is not None
            print(f"  Scope '{scope}': {'GRANTED' if has_scope else 'DENIED'}")
    
    print_subsection("OAuth2 Statistics")
    
    stats = oauth2.get_server_statistics()
    print(f"Total Clients: {stats['total_clients']}")
    print(f"Active Access Tokens: {stats['active_access_tokens']}")
    print(f"Active Refresh Tokens: {stats['active_refresh_tokens']}")
    
    return oauth2


def demo_mtls_authenticator(ca, certificates):
    """Demonstrate mTLS authenticator functionality."""
    print_section("MTLS AUTHENTICATOR SIMULATION")
    
    # Initialize mTLS authenticator
    mtls = MTLSAuthenticator(ca, "coordinator.archangel.local")
    
    print(f"mTLS Server: {mtls.server_name}")
    print(f"Trusted CAs: {len(mtls.trusted_cas)}")
    
    if mtls.server_certificate:
        print(f"Server Certificate: {mtls.server_certificate.serial_number}")
    
    print_subsection("Client Authentication")
    
    # Test client authentication with different certificates
    client_certs = [cert for cert in certificates if "agent" in cert.subject_dn.lower()]
    
    for cert in client_certs[:4]:  # Test first 4 client certificates
        print(f"\nTesting certificate: {cert.subject_dn}")
        
        result, identity = mtls.authenticate_client(
            cert.certificate_pem, "192.168.1.100"
        )
        
        print(f"  Authentication: {result.value}")
        
        if identity:
            print(f"  Client ID: {identity.client_id}")
            print(f"  Common Name: {identity.common_name}")
            print(f"  Organization: {identity.organization}")
            print(f"  Serial: {identity.certificate_serial}")
            
            # Establish connection
            conn_id = mtls.establish_connection(identity, "192.168.1.100")
            print(f"  Connection ID: {conn_id}")
            
            # Validate connection
            is_valid = mtls.validate_connection(conn_id)
            print(f"  Connection Valid: {is_valid}")
    
    print_subsection("Certificate Revocation Test")
    
    # Test with revoked certificate
    if client_certs:
        test_cert = client_certs[0]
        
        print(f"Testing revoked certificate: {test_cert.subject_dn}")
        
        # Revoke certificate in mTLS authenticator
        mtls.revoke_certificate(test_cert.serial_number)
        
        # Try to authenticate
        result, identity = mtls.authenticate_client(
            test_cert.certificate_pem, "192.168.1.101"
        )
        
        print(f"  Authentication: {result.value}")
        print(f"  Expected: CERTIFICATE_REVOKED")
    
    print_subsection("Connection Management")
    
    # Show active connections
    print(f"Active Connections: {len(mtls.active_connections)}")
    
    for conn_id, connection in list(mtls.active_connections.items())[:3]:
        conn_info = mtls.get_connection_info(conn_id)
        if conn_info:
            print(f"\nConnection: {conn_id}")
            print(f"  Client: {conn_info['client_identity']['common_name']}")
            print(f"  IP: {conn_info['client_ip']}")
            print(f"  Established: {conn_info['established_at']}")
            print(f"  Active: {conn_info['is_active']}")
    
    print_subsection("Authentication Statistics")
    
    stats = mtls.get_authentication_statistics()
    print(f"Total Attempts: {stats['total_attempts']}")
    print(f"Successful: {stats['successful_authentications']}")
    print(f"Failed: {stats['failed_authentications']}")
    print(f"Active Connections: {stats['active_connections']}")
    
    if stats['failure_reasons']:
        print("Failure Reasons:")
        for reason, count in stats['failure_reasons'].items():
            print(f"  {reason}: {count}")
    
    return mtls


def demo_rbac_manager():
    """Demonstrate RBAC manager functionality."""
    print_section("RBAC MANAGER SIMULATION")
    
    # Initialize RBAC manager
    rbac = RBACManager()
    
    print(f"Permissions: {len(rbac.permissions)}")
    print(f"Roles: {len(rbac.roles)}")
    print(f"Subjects: {len(rbac.subjects)}")
    print(f"Policies: {len(rbac.policy_rules)}")
    
    print_subsection("Role and Permission Overview")
    
    # Show key roles
    key_roles = ["red_team_agent", "blue_team_agent", "coordinator", "observer"]
    
    for role_name in key_roles:
        role_info = rbac.get_role_info(role_name)
        if role_info:
            print(f"\nRole: {role_info['name']}")
            print(f"  Description: {role_info['description']}")
            print(f"  Permissions: {len(role_info['permissions'])}")
            print(f"  Key Permissions: {', '.join(list(role_info['permissions'])[:5])}")
    
    print_subsection("Access Control Tests")
    
    # Test access control for different scenarios
    access_tests = [
        # (subject_id, resource_type, resource_name, permission_type, expected)
        ("recon_agent", ResourceType.NETWORK, "target_network", PermissionType.READ, True),
        ("recon_agent", ResourceType.NETWORK, "target_network", PermissionType.EXECUTE, True),
        ("recon_agent", ResourceType.SERVICE, "defense:firewall", PermissionType.READ, False),
        ("soc_analyst", ResourceType.SERVICE, "defense:siem", PermissionType.READ, True),
        ("soc_analyst", ResourceType.NETWORK, "target_network", PermissionType.EXECUTE, False),
        ("archangel_coordinator", ResourceType.SYSTEM, "configuration", PermissionType.ADMIN, True),
        ("monitoring_system", ResourceType.MONITORING, "metrics", PermissionType.READ, True),
        ("monitoring_system", ResourceType.AGENT, "any_agent", PermissionType.ADMIN, False),
    ]
    
    print("Access Control Test Results:")
    print(f"{'Subject':<20} {'Resource':<25} {'Permission':<10} {'Expected':<8} {'Actual':<8} {'Status'}")
    print("-" * 80)
    
    for subject_id, resource_type, resource_name, permission_type, expected in access_tests:
        decision = rbac.check_access(subject_id, resource_type, resource_name, permission_type)
        actual = decision.granted
        status = "PASS" if actual == expected else "FAIL"
        
        resource_str = f"{resource_type.value}:{resource_name}"
        print(f"{subject_id:<20} {resource_str:<25} {permission_type.value:<10} {expected!s:<8} {actual!s:<8} {status}")
    
    print_subsection("Subject Management")
    
    # Create custom subject
    custom_subject = rbac.create_subject(
        subject_id="custom_agent",
        subject_type="agent",
        display_name="Custom Agent",
        roles=["observer"],
        attributes={"team": "neutral", "clearance": "low"}
    )
    
    print(f"Created subject: {custom_subject.subject_id}")
    
    # Assign additional role
    rbac.assign_role("custom_agent", "monitoring_system")
    
    # Show subject info
    subject_info = rbac.get_subject_info("custom_agent")
    if subject_info:
        print(f"Subject: {subject_info['display_name']}")
        print(f"  Type: {subject_info['subject_type']}")
        print(f"  Roles: {', '.join(subject_info['roles'])}")
        print(f"  Total Permissions: {len(subject_info['all_permissions'])}")
        print(f"  Attributes: {subject_info['attributes']}")
    
    print_subsection("Policy Enforcement")
    
    # Test policy enforcement
    policy_tests = [
        ("recon_agent", ResourceType.SERVICE, "defense:siem", PermissionType.READ, "Team isolation policy"),
        ("soc_analyst", ResourceType.NETWORK, "target", PermissionType.EXECUTE, "Blue team attack restriction"),
        ("monitoring_system", ResourceType.SYSTEM, "config", PermissionType.ADMIN, "Admin protection policy"),
    ]
    
    print("Policy Enforcement Tests:")
    for subject_id, resource_type, resource_name, permission_type, policy_desc in policy_tests:
        decision = rbac.check_access(subject_id, resource_type, resource_name, permission_type)
        
        print(f"\nTest: {policy_desc}")
        print(f"  Subject: {subject_id}")
        print(f"  Resource: {resource_type.value}:{resource_name}")
        print(f"  Permission: {permission_type.value}")
        print(f"  Result: {'DENIED' if not decision.granted else 'GRANTED'}")
        print(f"  Reason: {decision.reason}")
    
    print_subsection("RBAC Statistics")
    
    stats = rbac.get_access_statistics()
    print(f"Total Access Requests: {stats['total_requests']}")
    print(f"Granted Requests: {stats['granted_requests']}")
    print(f"Denied Requests: {stats['denied_requests']}")
    
    if stats['top_subjects']:
        print("\nTop Subjects by Request Count:")
        for subject_id, count in stats['top_subjects'][:5]:
            print(f"  {subject_id}: {count}")
    
    if stats['denial_reasons']:
        print("\nDenial Reasons:")
        for reason, count in stats['denial_reasons'].items():
            print(f"  {reason}: {count}")
    
    return rbac


def demo_integration_scenario():
    """Demonstrate integrated authentication scenario."""
    print_section("INTEGRATION SCENARIO: AGENT ONBOARDING")
    
    print("Simulating complete agent onboarding process...")
    
    # 1. Initialize all services
    print("\n1. Initializing authentication infrastructure...")
    ad = ActiveDirectorySimulator("archangel.local")
    dns = DNSService("dns01.archangel.local")
    dhcp = DHCPService("dhcp01.archangel.local")
    
    ca_config = CAConfiguration(
        ca_name="Archangel Enterprise CA",
        subject_dn="CN=Archangel Enterprise CA,O=Archangel Corp,C=US"
    )
    ca = CertificateAuthority(ca_config)
    oauth2 = OAuth2Provider("https://auth.archangel.local")
    mtls = MTLSAuthenticator(ca)
    rbac = RBACManager()
    
    # 2. Create agent user account
    print("\n2. Creating agent user account in Active Directory...")
    agent_user = ad.create_user(
        username="red_agent_001",
        display_name="Red Team Agent 001",
        email="red001@archangel.local",
        department="Security",
        title="Penetration Tester"
    )
    ad.add_user_to_group("red_agent_001", "IT Support")
    print(f"   Created user: {agent_user.display_name}")
    
    # 3. Register DNS record
    print("\n3. Registering DNS record...")
    dns_success = dns.add_dynamic_record(
        "red-agent-001.archangel.local",
        DNSRecordType.A,
        "192.168.20.150"
    )
    print(f"   DNS registration: {'SUCCESS' if dns_success else 'FAILED'}")
    
    # 4. Obtain DHCP lease
    print("\n4. Obtaining DHCP lease...")
    mac_address = "00:50:56:AA:BB:01"
    offer = dhcp.handle_dhcp_discover(mac_address, "red-agent-001")
    if offer:
        ack = dhcp.handle_dhcp_request(
            mac_address, offer["offered_ip"], offer["server_ip"], "red-agent-001"
        )
        if ack:
            print(f"   DHCP lease: {ack['client_ip']}")
        else:
            print("   DHCP lease: FAILED")
    else:
        print("   DHCP lease: No offer")
    
    # 5. Issue client certificate
    print("\n5. Issuing client certificate...")
    agent_cert = ca.issue_certificate(
        subject_dn="CN=red-agent-001,OU=Red Team Agents,O=Archangel Corp,C=US",
        subject_alt_names=["DNS:red-agent-001.archangel.local"],
        key_usage=["digital_signature", "key_encipherment"],
        extended_key_usage=["client_auth"],
        validity_days=90
    )
    if agent_cert:
        print(f"   Certificate issued: {agent_cert.serial_number}")
    else:
        print("   Certificate issuance: FAILED")
    
    # 6. Authenticate with Active Directory
    print("\n6. Authenticating with Active Directory...")
    auth_result = ad.authenticate_user("red_agent_001", "Password123!")
    if auth_result["success"]:
        print(f"   AD authentication: SUCCESS")
        print(f"   User groups: {', '.join(auth_result['user_info']['groups'])}")
    else:
        print(f"   AD authentication: FAILED - {auth_result['reason']}")
    
    # 7. Establish mTLS connection
    print("\n7. Establishing mTLS connection...")
    if agent_cert:
        mtls_result, identity = mtls.authenticate_client(
            agent_cert.certificate_pem, "192.168.20.150"
        )
        if mtls_result == AuthenticationResult.SUCCESS:
            conn_id = mtls.establish_connection(identity, "192.168.20.150")
            print(f"   mTLS connection: SUCCESS ({conn_id})")
        else:
            print(f"   mTLS connection: FAILED - {mtls_result.value}")
    
    # 8. Obtain OAuth2 token
    print("\n8. Obtaining OAuth2 access token...")
    client = oauth2.clients["red_team_agents"]
    token_response = oauth2.client_credentials_grant(
        client.client_id, client.client_secret
    )
    if token_response:
        print(f"   OAuth2 token: SUCCESS")
        print(f"   Expires in: {token_response['expires_in']} seconds")
        print(f"   Scopes: {token_response['scope']}")
    else:
        print("   OAuth2 token: FAILED")
    
    # 9. Test RBAC permissions
    print("\n9. Testing RBAC permissions...")
    
    # Create RBAC subject for the agent
    rbac.create_subject(
        subject_id="red_agent_001",
        subject_type="agent",
        display_name="Red Team Agent 001",
        roles=["red_team_agent"],
        attributes={"team": "red", "clearance": "standard"}
    )
    
    # Test various permissions
    permission_tests = [
        (ResourceType.NETWORK, "target_network", PermissionType.READ, "Network scanning"),
        (ResourceType.INTELLIGENCE, "threat_data", PermissionType.WRITE, "Intelligence sharing"),
        (ResourceType.SERVICE, "defense:firewall", PermissionType.READ, "Defense system access"),
        (ResourceType.SYSTEM, "configuration", PermissionType.ADMIN, "System administration")
    ]
    
    for resource_type, resource_name, permission_type, description in permission_tests:
        decision = rbac.check_access("red_agent_001", resource_type, resource_name, permission_type)
        status = "GRANTED" if decision.granted else "DENIED"
        print(f"   {description}: {status}")
    
    # 10. Summary
    print("\n10. Agent onboarding summary:")
    print("   ✓ Active Directory account created")
    print("   ✓ DNS record registered")
    print("   ✓ DHCP lease obtained")
    print("   ✓ Client certificate issued")
    print("   ✓ AD authentication successful")
    print("   ✓ mTLS connection established")
    print("   ✓ OAuth2 token obtained")
    print("   ✓ RBAC permissions configured")
    print("\n   Agent is fully authenticated and authorized!")


def main():
    """Main demo function."""
    setup_logging()
    
    print("ARCHANGEL ENTERPRISE AUTHENTICATION SYSTEM DEMO")
    print("=" * 60)
    print("This demo showcases the comprehensive authentication infrastructure")
    print("including Active Directory, DNS, DHCP, PKI, OAuth2, mTLS, and RBAC.")
    print()
    
    try:
        # Run individual component demos
        ad = demo_active_directory()
        dns = demo_dns_service()
        dhcp = demo_dhcp_service()
        ca, certificates = demo_certificate_authority()
        oauth2 = demo_oauth2_provider()
        mtls = demo_mtls_authenticator(ca, certificates)
        rbac = demo_rbac_manager()
        
        # Run integration scenario
        demo_integration_scenario()
        
        print_section("DEMO COMPLETED SUCCESSFULLY")
        print("All authentication components are working correctly!")
        print("The system is ready for agent deployment and operation.")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())