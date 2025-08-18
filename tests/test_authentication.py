"""
Comprehensive tests for enterprise authentication and directory services.

Tests cover:
- Active Directory simulation
- DNS and DHCP services
- Certificate Authority functionality
- OAuth2 provider
- mTLS authenticator
- RBAC manager
"""

import pytest
import json
import secrets
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from authentication.active_directory import (
    ActiveDirectorySimulator, ADUser, ADGroup, OrganizationalUnit,
    UserAccountControl, GroupType
)
from authentication.dns_service import DNSService, DNSRecordType, DNSZone
from authentication.dhcp_service import DHCPService, DHCPScope, LeaseState
from authentication.certificate_authority import (
    CertificateAuthority, CAConfiguration, CertificateStatus, RevocationReason
)
from authentication.oauth2_provider import OAuth2Provider, GrantType, TokenType
from authentication.mtls_authenticator import MTLSAuthenticator, AuthenticationResult
from authentication.rbac_manager import (
    RBACManager, Permission, Role, Subject, ResourceType, PermissionType
)


class TestActiveDirectorySimulator:
    """Test Active Directory simulation functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.ad = ActiveDirectorySimulator("test.local")
    
    def test_initialization(self):
        """Test AD initialization with default structure."""
        assert self.ad.domain_name == "test.local"
        assert len(self.ad.users) > 0
        assert len(self.ad.groups) > 0
        assert len(self.ad.organizational_units) > 0
        
        # Check default admin user exists
        assert "admin" in self.ad.users
        admin_user = self.ad.users["admin"]
        assert "Domain Admins" in admin_user.groups
    
    def test_user_creation(self):
        """Test creating new users."""
        user = self.ad.create_user(
            username="testuser",
            display_name="Test User",
            email="testuser@test.local",
            department="IT",
            title="Test Engineer"
        )
        
        assert user.username == "testuser"
        assert user.email == "testuser@test.local"
        assert "testuser" in self.ad.users
        
        # Test duplicate user creation fails
        with pytest.raises(ValueError):
            self.ad.create_user("testuser", "Duplicate", "dup@test.local", "IT", "Tester")
    
    def test_group_management(self):
        """Test group creation and membership."""
        # Create group
        group = self.ad.create_group("Test Group", "Test Description", GroupType.SECURITY_GLOBAL)
        assert group.name == "Test Group"
        assert "Test Group" in self.ad.groups
        
        # Add user to group
        success = self.ad.add_user_to_group("admin", "Test Group")
        assert success
        assert "Test Group" in self.ad.users["admin"].groups
        assert "admin" in self.ad.groups["Test Group"].members
    
    def test_authentication(self):
        """Test user authentication."""
        # Test successful authentication
        result = self.ad.authenticate_user("admin", "Password123!")
        assert result["success"]
        assert result["user_info"]["username"] == "admin"
        
        # Test failed authentication
        result = self.ad.authenticate_user("admin", "wrongpassword")
        assert not result["success"]
        assert "Invalid password" in result["reason"]
        
        # Test unknown user
        result = self.ad.authenticate_user("unknown", "password")
        assert not result["success"]
        assert "User not found" in result["reason"]
    
    def test_account_lockout(self):
        """Test account lockout functionality."""
        # Attempt multiple failed logins
        for _ in range(6):  # Default lockout threshold is 5
            self.ad.authenticate_user("admin", "wrongpassword")
        
        # Account should be locked
        result = self.ad.authenticate_user("admin", "Password123!")
        assert not result["success"]
        assert "locked" in result["reason"].lower()
    
    def test_user_permissions(self):
        """Test user permission checking."""
        # Test admin permissions
        has_permission = self.ad.check_user_permission("admin", ["Domain Admins"])
        assert has_permission
        
        # Test regular user permissions
        self.ad.create_user("regular", "Regular User", "regular@test.local", "Users", "User")
        has_permission = self.ad.check_user_permission("regular", ["Domain Admins"])
        assert not has_permission


class TestDNSService:
    """Test DNS service functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.dns = DNSService("testdns.local")
    
    def test_initialization(self):
        """Test DNS service initialization."""
        assert self.dns.server_name == "testdns.local"
        assert len(self.dns.zones) > 0
        
        # Check default zone exists
        assert "archangel.local" in self.dns.zones
    
    def test_dns_resolution(self):
        """Test DNS query resolution."""
        # Test A record resolution
        records = self.dns.resolve_query("dns01.archangel.local", DNSRecordType.A)
        assert len(records) > 0
        assert records[0].record_type == DNSRecordType.A
        
        # Test CNAME resolution
        records = self.dns.resolve_query("www.archangel.local", DNSRecordType.A)
        assert len(records) > 0  # Should resolve CNAME and follow to A record
        
        # Test non-existent record
        records = self.dns.resolve_query("nonexistent.archangel.local", DNSRecordType.A)
        assert len(records) == 0
    
    def test_dynamic_dns(self):
        """Test dynamic DNS record management."""
        # Add dynamic record
        success = self.dns.add_dynamic_record(
            "dynamic.archangel.local",
            DNSRecordType.A,
            "192.168.1.100"
        )
        assert success
        
        # Verify record exists
        records = self.dns.resolve_query("dynamic.archangel.local", DNSRecordType.A)
        assert len(records) > 0
        assert records[0].value == "192.168.1.100"
        
        # Remove dynamic record
        success = self.dns.remove_dynamic_record("dynamic.archangel.local", DNSRecordType.A)
        assert success
        
        # Verify record removed
        records = self.dns.resolve_query("dynamic.archangel.local", DNSRecordType.A)
        assert len(records) == 0
    
    def test_zone_management(self):
        """Test DNS zone information."""
        zone_info = self.dns.get_zone_info("archangel.local")
        assert zone_info is not None
        assert zone_info["name"] == "archangel.local"
        assert zone_info["type"] == "primary"
        assert zone_info["record_count"] > 0
    
    def test_query_statistics(self):
        """Test DNS query statistics."""
        # Perform some queries
        self.dns.resolve_query("dns01.archangel.local", DNSRecordType.A)
        self.dns.resolve_query("web01.archangel.local", DNSRecordType.A)
        
        stats = self.dns.get_query_statistics()
        assert stats["total_queries"] >= 2
        assert stats["successful_queries"] >= 2


class TestDHCPService:
    """Test DHCP service functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.dhcp = DHCPService("testdhcp.local")
    
    def test_initialization(self):
        """Test DHCP service initialization."""
        assert self.dhcp.server_name == "testdhcp.local"
        assert len(self.dhcp.scopes) > 0
        
        # Check default scopes exist
        assert "mgmt" in self.dhcp.scopes
        assert "dmz" in self.dhcp.scopes
        assert "lan" in self.dhcp.scopes
    
    def test_dhcp_discover_offer(self):
        """Test DHCP DISCOVER/OFFER flow."""
        client_mac = "00:11:22:33:44:55"
        
        # Send DISCOVER
        offer = self.dhcp.handle_dhcp_discover(client_mac, "testclient")
        assert offer is not None
        assert offer["message_type"] == "OFFER"
        assert "offered_ip" in offer
        
        # Verify lease created
        offered_ip = offer["offered_ip"]
        assert offered_ip in self.dhcp.leases
        lease = self.dhcp.leases[offered_ip]
        assert lease.mac_address == client_mac
        assert lease.state == LeaseState.OFFERED
    
    def test_dhcp_request_ack(self):
        """Test DHCP REQUEST/ACK flow."""
        client_mac = "00:11:22:33:44:66"
        
        # Get offer first
        offer = self.dhcp.handle_dhcp_discover(client_mac, "testclient2")
        offered_ip = offer["offered_ip"]
        
        # Send REQUEST
        ack = self.dhcp.handle_dhcp_request(
            client_mac, offered_ip, offer["server_ip"], "testclient2"
        )
        assert ack is not None
        assert ack["message_type"] == "ACK"
        assert ack["client_ip"] == offered_ip
        
        # Verify lease bound
        lease = self.dhcp.leases[offered_ip]
        assert lease.state == LeaseState.BOUND
    
    def test_dhcp_reservations(self):
        """Test DHCP reservations."""
        # Add reservation
        success = self.dhcp.add_reservation(
            "mgmt", "00:50:56:12:34:99", "192.168.1.199", "reserved-host"
        )
        assert success
        
        # Test DISCOVER with reserved MAC
        offer = self.dhcp.handle_dhcp_discover("00:50:56:12:34:99", "reserved-host")
        assert offer is not None
        assert offer["offered_ip"] == "192.168.1.199"
    
    def test_dhcp_release(self):
        """Test DHCP RELEASE."""
        client_mac = "00:11:22:33:44:77"
        
        # Get lease
        offer = self.dhcp.handle_dhcp_discover(client_mac)
        offered_ip = offer["offered_ip"]
        self.dhcp.handle_dhcp_request(client_mac, offered_ip, offer["server_ip"])
        
        # Release lease
        success = self.dhcp.handle_dhcp_release(client_mac, offered_ip)
        assert success
        assert offered_ip not in self.dhcp.leases
    
    def test_scope_statistics(self):
        """Test DHCP scope statistics."""
        stats = self.dhcp.get_scope_statistics("mgmt")
        assert stats is not None
        assert stats["scope_name"] == "Management Network"
        assert stats["total_addresses"] > 0
        assert stats["utilization_percent"] >= 0


class TestCertificateAuthority:
    """Test Certificate Authority functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        config = CAConfiguration(
            ca_name="Test CA",
            subject_dn="CN=Test CA,O=Test Corp,C=US"
        )
        self.ca = CertificateAuthority(config)
    
    def test_ca_initialization(self):
        """Test CA initialization."""
        assert self.ca.config.ca_name == "Test CA"
        assert self.ca.ca_certificate is not None
        assert self.ca.ca_certificate.subject_dn == "CN=Test CA,O=Test Corp,C=US"
    
    def test_certificate_issuance(self):
        """Test certificate issuance."""
        cert = self.ca.issue_certificate(
            subject_dn="CN=test.example.com,O=Test Corp,C=US",
            subject_alt_names=["DNS:test.example.com", "DNS:www.test.example.com"],
            key_usage=["digital_signature", "key_encipherment"],
            extended_key_usage=["server_auth"]
        )
        
        assert cert is not None
        assert cert.subject_dn == "CN=test.example.com,O=Test Corp,C=US"
        assert cert.status == CertificateStatus.VALID
        assert "DNS:test.example.com" in cert.subject_alt_names
        assert cert.serial_number in self.ca.certificates
    
    def test_certificate_revocation(self):
        """Test certificate revocation."""
        # Issue certificate
        cert = self.ca.issue_certificate("CN=revoke-test.example.com,O=Test Corp,C=US")
        serial_number = cert.serial_number
        
        # Revoke certificate
        success = self.ca.revoke_certificate(serial_number, RevocationReason.KEY_COMPROMISE)
        assert success
        
        # Verify revocation
        assert serial_number in self.ca.revoked_certificates
        revoked_cert = self.ca.revoked_certificates[serial_number]
        assert revoked_cert.status == CertificateStatus.REVOKED
        assert revoked_cert.revocation_reason == RevocationReason.KEY_COMPROMISE
    
    def test_certificate_validation(self):
        """Test certificate validation."""
        # Issue certificate
        cert = self.ca.issue_certificate("CN=validate-test.example.com,O=Test Corp,C=US")
        
        # Validate certificate
        result = self.ca.validate_certificate(cert.certificate_pem)
        assert result["valid"]
        assert result["serial_number"] == cert.serial_number
        
        # Test invalid certificate
        invalid_result = self.ca.validate_certificate("invalid certificate")
        assert not invalid_result["valid"]
        assert len(invalid_result["errors"]) > 0
    
    def test_crl_generation(self):
        """Test CRL generation."""
        # Issue and revoke some certificates
        cert1 = self.ca.issue_certificate("CN=crl-test1.example.com,O=Test Corp,C=US")
        cert2 = self.ca.issue_certificate("CN=crl-test2.example.com,O=Test Corp,C=US")
        
        self.ca.revoke_certificate(cert1.serial_number)
        self.ca.revoke_certificate(cert2.serial_number)
        
        # Generate CRL
        crl = self.ca.generate_crl()
        assert crl is not None
        assert "BEGIN X509 CRL" in crl
        assert "END X509 CRL" in crl
    
    def test_ca_statistics(self):
        """Test CA statistics."""
        stats = self.ca.get_ca_statistics()
        assert stats["ca_name"] == "Test CA"
        assert stats["total_certificates"] >= 1  # At least CA cert
        assert stats["valid_certificates"] >= 1


class TestOAuth2Provider:
    """Test OAuth2 provider functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.oauth2 = OAuth2Provider("https://test.auth.local")
    
    def test_initialization(self):
        """Test OAuth2 provider initialization."""
        assert self.oauth2.issuer == "https://test.auth.local"
        assert len(self.oauth2.clients) > 0
        
        # Check default clients exist
        assert "red_team_agents" in self.oauth2.clients
        assert "blue_team_agents" in self.oauth2.clients
    
    def test_client_registration(self):
        """Test OAuth2 client registration."""
        client = self.oauth2.register_client(
            client_name="Test Client",
            redirect_uris=["http://localhost:8080/callback"],
            grant_types=[GrantType.CLIENT_CREDENTIALS],
            scopes={"test:read", "test:write"}
        )
        
        assert client.client_name == "Test Client"
        assert client.client_id in self.oauth2.clients
        assert len(client.client_secret) > 0
    
    def test_client_credentials_flow(self):
        """Test client credentials grant flow."""
        client = self.oauth2.clients["red_team_agents"]
        
        # Request token
        token_response = self.oauth2.client_credentials_grant(
            client.client_id, client.client_secret
        )
        
        assert token_response is not None
        assert token_response["token_type"] == "Bearer"
        assert "access_token" in token_response
        assert token_response["expires_in"] > 0
        
        # Validate token
        token_info = self.oauth2.validate_access_token(token_response["access_token"])
        assert token_info is not None
        assert token_info["client_id"] == client.client_id
    
    def test_authorization_code_flow(self):
        """Test authorization code grant flow."""
        client = self.oauth2.clients["blue_team_agents"]
        redirect_uri = client.redirect_uris[0]
        scopes = {"agent:read", "coordination:participate"}
        
        # Create authorization code
        auth_code = self.oauth2.create_authorization_code(
            client.client_id, redirect_uri, scopes
        )
        assert auth_code is not None
        
        # Exchange code for token
        token_response = self.oauth2.exchange_authorization_code(
            client.client_id, client.client_secret, auth_code, redirect_uri
        )
        
        assert token_response is not None
        assert "access_token" in token_response
        assert "refresh_token" in token_response
    
    def test_token_validation(self):
        """Test access token validation."""
        client = self.oauth2.clients["red_team_agents"]
        
        # Get token
        token_response = self.oauth2.client_credentials_grant(
            client.client_id, client.client_secret
        )
        access_token = token_response["access_token"]
        
        # Validate with correct scope
        token_info = self.oauth2.validate_access_token(access_token, "agent:read")
        assert token_info is not None
        
        # Validate with incorrect scope
        token_info = self.oauth2.validate_access_token(access_token, "invalid:scope")
        assert token_info is None
    
    def test_token_revocation(self):
        """Test token revocation."""
        client = self.oauth2.clients["red_team_agents"]
        
        # Get token
        token_response = self.oauth2.client_credentials_grant(
            client.client_id, client.client_secret
        )
        access_token = token_response["access_token"]
        
        # Revoke token
        success = self.oauth2.revoke_token(access_token, client.client_id, client.client_secret)
        assert success
        
        # Verify token is invalid
        token_info = self.oauth2.validate_access_token(access_token)
        assert token_info is None


class TestMTLSAuthenticator:
    """Test mTLS authenticator functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        ca_config = CAConfiguration(
            ca_name="Test mTLS CA",
            subject_dn="CN=Test mTLS CA,O=Test Corp,C=US"
        )
        self.ca = CertificateAuthority(ca_config)
        self.mtls = MTLSAuthenticator(self.ca, "test-server.local")
    
    def test_initialization(self):
        """Test mTLS authenticator initialization."""
        assert self.mtls.server_name == "test-server.local"
        assert len(self.mtls.trusted_cas) > 0
        assert self.mtls.server_certificate is not None
    
    def test_client_authentication(self):
        """Test client certificate authentication."""
        # Issue client certificate
        client_cert = self.ca.issue_certificate(
            subject_dn="CN=test-client,OU=Agents,O=Test Corp,C=US",
            key_usage=["digital_signature", "key_encipherment"],
            extended_key_usage=["client_auth"]
        )
        
        # Authenticate client
        result, identity = self.mtls.authenticate_client(
            client_cert.certificate_pem, "192.168.1.100"
        )
        
        assert result == AuthenticationResult.SUCCESS
        assert identity is not None
        assert identity.common_name == "test-client"
        assert identity.organization == "Test Corp"
    
    def test_certificate_revocation_check(self):
        """Test certificate revocation checking."""
        # Issue client certificate
        client_cert = self.ca.issue_certificate(
            subject_dn="CN=revoked-client,OU=Agents,O=Test Corp,C=US"
        )
        
        # Revoke certificate
        self.mtls.revoke_certificate(client_cert.serial_number)
        
        # Try to authenticate with revoked certificate
        result, identity = self.mtls.authenticate_client(
            client_cert.certificate_pem, "192.168.1.100"
        )
        
        assert result == AuthenticationResult.CERTIFICATE_REVOKED
        assert identity is None
    
    def test_connection_management(self):
        """Test mTLS connection management."""
        # Issue and authenticate client
        client_cert = self.ca.issue_certificate(
            subject_dn="CN=connection-test,OU=Agents,O=Test Corp,C=US"
        )
        
        result, identity = self.mtls.authenticate_client(
            client_cert.certificate_pem, "192.168.1.100"
        )
        
        # Establish connection
        conn_id = self.mtls.establish_connection(identity, "192.168.1.100")
        assert conn_id in self.mtls.active_connections
        
        # Validate connection
        is_valid = self.mtls.validate_connection(conn_id)
        assert is_valid
        
        # Get connection info
        conn_info = self.mtls.get_connection_info(conn_id)
        assert conn_info is not None
        assert conn_info["client_identity"]["common_name"] == "connection-test"
    
    def test_authentication_statistics(self):
        """Test authentication statistics."""
        # Perform some authentications
        client_cert = self.ca.issue_certificate(
            subject_dn="CN=stats-test,OU=Agents,O=Test Corp,C=US"
        )
        
        self.mtls.authenticate_client(client_cert.certificate_pem, "192.168.1.100")
        self.mtls.authenticate_client("invalid certificate", "192.168.1.101")
        
        stats = self.mtls.get_authentication_statistics()
        assert stats["total_attempts"] >= 2
        assert stats["successful_authentications"] >= 1
        assert stats["failed_authentications"] >= 1


class TestRBACManager:
    """Test RBAC manager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.rbac = RBACManager()
    
    def test_initialization(self):
        """Test RBAC manager initialization."""
        assert len(self.rbac.permissions) > 0
        assert len(self.rbac.roles) > 0
        assert len(self.rbac.subjects) > 0
        
        # Check default roles exist
        assert "red_team_agent" in self.rbac.roles
        assert "blue_team_agent" in self.rbac.roles
        assert "coordinator" in self.rbac.roles
    
    def test_permission_creation(self):
        """Test permission creation."""
        permission = self.rbac.create_permission(
            name="test:custom",
            resource_type=ResourceType.SYSTEM,
            permission_type=PermissionType.READ,
            resource_pattern="test:*",
            description="Test permission"
        )
        
        assert permission.name == "test:custom"
        assert "test:custom" in self.rbac.permissions
    
    def test_role_creation(self):
        """Test role creation."""
        role = self.rbac.create_role(
            name="test_role",
            description="Test role",
            permissions=["agent:read", "system:read"]
        )
        
        assert role.name == "test_role"
        assert "test_role" in self.rbac.roles
        assert "agent:read" in role.permissions
    
    def test_subject_creation(self):
        """Test subject creation."""
        subject = self.rbac.create_subject(
            subject_id="test_agent",
            subject_type="agent",
            display_name="Test Agent",
            roles=["red_team_agent"],
            attributes={"team": "red"}
        )
        
        assert subject.subject_id == "test_agent"
        assert "test_agent" in self.rbac.subjects
        assert "red_team_agent" in subject.roles
    
    def test_access_control(self):
        """Test access control decisions."""
        # Test red team agent access
        decision = self.rbac.check_access(
            subject_id="recon_agent",
            resource_type=ResourceType.NETWORK,
            resource_name="target_network",
            permission_type=PermissionType.READ
        )
        assert decision.granted  # Red team should have network scan access
        
        # Test blue team agent access to defense resources
        decision = self.rbac.check_access(
            subject_id="soc_analyst",
            resource_type=ResourceType.SERVICE,
            resource_name="defense:firewall",
            permission_type=PermissionType.READ
        )
        assert decision.granted  # Blue team should have defense access
        
        # Test unauthorized access
        decision = self.rbac.check_access(
            subject_id="recon_agent",
            resource_type=ResourceType.SERVICE,
            resource_name="defense:firewall",
            permission_type=PermissionType.ADMIN
        )
        assert not decision.granted  # Red team should not have defense admin
    
    def test_role_assignment(self):
        """Test role assignment and revocation."""
        # Create test subject
        self.rbac.create_subject("test_user", "user", "Test User")
        
        # Assign role
        success = self.rbac.assign_role("test_user", "observer")
        assert success
        assert "observer" in self.rbac.subjects["test_user"].roles
        
        # Revoke role
        success = self.rbac.revoke_role("test_user", "observer")
        assert success
        assert "observer" not in self.rbac.subjects["test_user"].roles
    
    def test_policy_enforcement(self):
        """Test policy rule enforcement."""
        # Test that red team agents cannot access defense resources (policy rule)
        decision = self.rbac.check_access(
            subject_id="recon_agent",
            resource_type=ResourceType.SERVICE,
            resource_name="defense:siem",
            permission_type=PermissionType.READ
        )
        assert not decision.granted
        assert "policy" in decision.reason.lower()
    
    def test_subject_info(self):
        """Test subject information retrieval."""
        info = self.rbac.get_subject_info("recon_agent")
        assert info is not None
        assert info["subject_id"] == "recon_agent"
        assert info["subject_type"] == "agent"
        assert len(info["all_permissions"]) > 0
    
    def test_access_statistics(self):
        """Test access statistics."""
        # Perform some access checks
        self.rbac.check_access("recon_agent", ResourceType.NETWORK, "test", PermissionType.READ)
        self.rbac.check_access("soc_analyst", ResourceType.MONITORING, "test", PermissionType.READ)
        
        stats = self.rbac.get_access_statistics()
        assert stats["total_requests"] >= 2
        assert stats["total_subjects"] > 0
        assert stats["total_roles"] > 0


class TestIntegration:
    """Integration tests for authentication components."""
    
    def setup_method(self):
        """Set up integrated test environment."""
        # Initialize all components
        self.ad = ActiveDirectorySimulator("integration.local")
        self.dns = DNSService("dns.integration.local")
        self.dhcp = DHCPService("dhcp.integration.local")
        
        ca_config = CAConfiguration(
            ca_name="Integration CA",
            subject_dn="CN=Integration CA,O=Integration Corp,C=US"
        )
        self.ca = CertificateAuthority(ca_config)
        self.oauth2 = OAuth2Provider("https://auth.integration.local")
        self.mtls = MTLSAuthenticator(self.ca)
        self.rbac = RBACManager()
    
    def test_end_to_end_authentication(self):
        """Test end-to-end authentication flow."""
        # 1. Create user in AD
        user = self.ad.create_user(
            username="integration_agent",
            display_name="Integration Agent",
            email="agent@integration.local",
            department="IT",
            title="Test Agent"
        )
        
        # 2. Authenticate user in AD
        auth_result = self.ad.authenticate_user("integration_agent", "Password123!")
        assert auth_result["success"]
        
        # 3. Register DNS record for agent
        success = self.dns.add_dynamic_record(
            "integration-agent.integration.local",
            DNSRecordType.A,
            "192.168.1.150"
        )
        assert success
        
        # 4. Get DHCP lease for agent
        offer = self.dhcp.handle_dhcp_discover("00:11:22:33:44:88", "integration-agent")
        assert offer is not None
        
        # 5. Issue certificate for agent
        agent_cert = self.ca.issue_certificate(
            subject_dn="CN=integration-agent,OU=Agents,O=Integration Corp,C=US",
            subject_alt_names=["DNS:integration-agent.integration.local"],
            extended_key_usage=["client_auth"]
        )
        assert agent_cert is not None
        
        # 6. Authenticate with mTLS
        result, identity = self.mtls.authenticate_client(
            agent_cert.certificate_pem, "192.168.1.150"
        )
        assert result == AuthenticationResult.SUCCESS
        
        # 7. Get OAuth2 token
        client = self.oauth2.clients["red_team_agents"]
        token_response = self.oauth2.client_credentials_grant(
            client.client_id, client.client_secret
        )
        assert token_response is not None
        
        # 8. Check RBAC permissions
        decision = self.rbac.check_access(
            subject_id="recon_agent",
            resource_type=ResourceType.NETWORK,
            resource_name="target_network",
            permission_type=PermissionType.READ
        )
        assert decision.granted
    
    def test_security_integration(self):
        """Test security features integration."""
        # Issue certificate
        cert = self.ca.issue_certificate(
            subject_dn="CN=security-test,OU=Agents,O=Integration Corp,C=US"
        )
        
        # Authenticate successfully
        result, identity = self.mtls.authenticate_client(cert.certificate_pem)
        assert result == AuthenticationResult.SUCCESS
        
        # Revoke certificate
        self.ca.revoke_certificate(cert.serial_number)
        self.mtls.revoke_certificate(cert.serial_number)
        
        # Authentication should now fail
        result, identity = self.mtls.authenticate_client(cert.certificate_pem)
        assert result == AuthenticationResult.CERTIFICATE_REVOKED
    
    def test_monitoring_integration(self):
        """Test monitoring and logging integration."""
        # Perform various operations
        self.ad.authenticate_user("admin", "Password123!")
        self.dns.resolve_query("dns01.archangel.local", DNSRecordType.A)
        self.dhcp.handle_dhcp_discover("00:11:22:33:44:99")
        
        # Check that events were logged
        assert len(self.ad.authentication_log) > 0
        assert len(self.dns.query_log) > 0
        assert len(self.dhcp.transactions) > 0
        
        # Get statistics
        ad_info = self.ad.get_domain_info()
        dns_stats = self.dns.get_query_statistics()
        dhcp_stats = self.dhcp.get_server_statistics()
        
        assert ad_info["total_users"] > 0
        assert dns_stats["total_queries"] > 0
        assert dhcp_stats["total_transactions"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])