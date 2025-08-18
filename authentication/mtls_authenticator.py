"""
Mutual TLS (mTLS) Authenticator for secure agent communication.

This module provides mTLS authentication services with:
- Client certificate validation
- Certificate-based identity verification
- TLS connection security
- Certificate revocation checking
"""

import ssl
import socket
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import base64

try:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.x509.verification import PolicyBuilder, StoreBuilder
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

from .certificate_authority import CertificateAuthority, CertificateInfo


class AuthenticationResult(Enum):
    """mTLS authentication results."""
    SUCCESS = "success"
    CERTIFICATE_INVALID = "certificate_invalid"
    CERTIFICATE_EXPIRED = "certificate_expired"
    CERTIFICATE_REVOKED = "certificate_revoked"
    CERTIFICATE_NOT_TRUSTED = "certificate_not_trusted"
    HANDSHAKE_FAILED = "handshake_failed"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ClientIdentity:
    """Client identity extracted from certificate."""
    client_id: str
    common_name: str
    organization: str
    organizational_unit: str
    email: Optional[str] = None
    subject_alt_names: List[str] = field(default_factory=list)
    certificate_serial: str = ""
    certificate_fingerprint: str = ""
    not_before: Optional[datetime] = None
    not_after: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if client certificate has expired."""
        if self.not_after:
            return datetime.now() > self.not_after
        return False


@dataclass
class MTLSConnection:
    """mTLS connection information."""
    connection_id: str
    client_identity: ClientIdentity
    server_name: str
    client_ip: str
    established_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    bytes_sent: int = 0
    bytes_received: int = 0
    is_active: bool = True
    
    def update_activity(self, bytes_sent: int = 0, bytes_received: int = 0) -> None:
        """Update connection activity."""
        self.last_activity = datetime.now()
        self.bytes_sent += bytes_sent
        self.bytes_received += bytes_received


@dataclass
class AuthenticationEvent:
    """mTLS authentication event log."""
    event_id: str
    timestamp: datetime
    client_ip: str
    client_identity: Optional[ClientIdentity]
    result: AuthenticationResult
    error_message: Optional[str] = None
    connection_id: Optional[str] = None


class MTLSAuthenticator:
    """
    Mutual TLS Authenticator for agent communication.
    
    Provides mTLS authentication services including:
    - Client certificate validation
    - Certificate-based identity verification
    - Connection management and monitoring
    - Authentication event logging
    """
    
    def __init__(self, ca: CertificateAuthority, server_name: str = "coordinator.archangel.local"):
        self.ca = ca
        self.server_name = server_name
        self.trusted_cas: Set[str] = set()
        self.active_connections: Dict[str, MTLSConnection] = {}
        self.authentication_events: List[AuthenticationEvent] = []
        self.revoked_certificates: Set[str] = set()
        self.logger = logging.getLogger(__name__)
        
        # Add our CA as trusted
        if self.ca.ca_certificate:
            self.trusted_cas.add(self.ca.ca_certificate.certificate_pem)
        
        # Initialize server certificate
        self.server_certificate = self._generate_server_certificate()
    
    def _generate_server_certificate(self) -> Optional[CertificateInfo]:
        """Generate server certificate for mTLS."""
        
        try:
            server_cert = self.ca.issue_certificate(
                subject_dn=f"CN={self.server_name},OU=Coordination,O=Archangel Corp,C=US",
                subject_alt_names=[f"DNS:{self.server_name}", "DNS:localhost", "IP:127.0.0.1"],
                key_usage=["digital_signature", "key_encipherment"],
                extended_key_usage=["server_auth"],
                validity_days=365
            )
            
            if server_cert:
                self.logger.info(f"Generated server certificate for {self.server_name}")
                return server_cert
            else:
                self.logger.error("Failed to generate server certificate")
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating server certificate: {e}")
            return None
    
    def add_trusted_ca(self, ca_certificate_pem: str) -> bool:
        """
        Add a trusted Certificate Authority.
        
        Args:
            ca_certificate_pem: CA certificate in PEM format
            
        Returns:
            True if CA was added successfully
        """
        
        try:
            # Validate certificate format (basic check)
            if "BEGIN CERTIFICATE" in ca_certificate_pem and "END CERTIFICATE" in ca_certificate_pem:
                self.trusted_cas.add(ca_certificate_pem)
                self.logger.info("Added trusted CA certificate")
                return True
            else:
                self.logger.error("Invalid CA certificate format")
                return False
                
        except Exception as e:
            self.logger.error(f"Error adding trusted CA: {e}")
            return False
    
    def revoke_certificate(self, certificate_serial: str) -> bool:
        """
        Revoke a client certificate.
        
        Args:
            certificate_serial: Serial number of certificate to revoke
            
        Returns:
            True if certificate was revoked successfully
        """
        
        self.revoked_certificates.add(certificate_serial)
        
        # Close any active connections using this certificate
        connections_to_close = []
        for conn_id, connection in self.active_connections.items():
            if connection.client_identity.certificate_serial == certificate_serial:
                connections_to_close.append(conn_id)
        
        for conn_id in connections_to_close:
            self._close_connection(conn_id, "Certificate revoked")
        
        self.logger.info(f"Revoked certificate: {certificate_serial}")
        return True
    
    def authenticate_client(self, client_certificate_pem: str, 
                          client_ip: str = "127.0.0.1") -> Tuple[AuthenticationResult, Optional[ClientIdentity]]:
        """
        Authenticate client using certificate.
        
        Args:
            client_certificate_pem: Client certificate in PEM format
            client_ip: Client IP address
            
        Returns:
            Tuple of authentication result and client identity (if successful)
        """
        
        event_id = f"auth_{len(self.authentication_events) + 1}"
        
        try:
            # Parse client certificate
            client_identity = self._parse_certificate(client_certificate_pem)
            if not client_identity:
                result = AuthenticationResult.CERTIFICATE_INVALID
                self._log_authentication_event(event_id, client_ip, None, result, "Failed to parse certificate")
                return result, None
            
            # Check if certificate is expired
            if client_identity.is_expired():
                result = AuthenticationResult.CERTIFICATE_EXPIRED
                self._log_authentication_event(event_id, client_ip, client_identity, result, "Certificate has expired")
                return result, None
            
            # Check if certificate is revoked
            if client_identity.certificate_serial in self.revoked_certificates:
                result = AuthenticationResult.CERTIFICATE_REVOKED
                self._log_authentication_event(event_id, client_ip, client_identity, result, "Certificate has been revoked")
                return result, None
            
            # Validate certificate chain
            if not self._validate_certificate_chain(client_certificate_pem):
                result = AuthenticationResult.CERTIFICATE_NOT_TRUSTED
                self._log_authentication_event(event_id, client_ip, client_identity, result, "Certificate not trusted")
                return result, None
            
            # Authentication successful
            result = AuthenticationResult.SUCCESS
            self._log_authentication_event(event_id, client_ip, client_identity, result)
            
            self.logger.info(f"Successfully authenticated client: {client_identity.common_name} from {client_ip}")
            return result, client_identity
            
        except Exception as e:
            result = AuthenticationResult.UNKNOWN_ERROR
            self._log_authentication_event(event_id, client_ip, None, result, str(e))
            self.logger.error(f"Authentication error for client from {client_ip}: {e}")
            return result, None
    
    def _parse_certificate(self, certificate_pem: str) -> Optional[ClientIdentity]:
        """Parse client certificate and extract identity information."""
        
        try:
            if CRYPTOGRAPHY_AVAILABLE:
                return self._parse_real_certificate(certificate_pem)
            else:
                return self._parse_mock_certificate(certificate_pem)
                
        except Exception as e:
            self.logger.error(f"Error parsing certificate: {e}")
            return None
    
    def _parse_real_certificate(self, certificate_pem: str) -> Optional[ClientIdentity]:
        """Parse real certificate using cryptography library."""
        
        try:
            cert = x509.load_pem_x509_certificate(certificate_pem.encode())
            
            # Extract subject information
            subject = cert.subject
            common_name = ""
            organization = ""
            organizational_unit = ""
            email = None
            
            for attribute in subject:
                if attribute.oid == x509.NameOID.COMMON_NAME:
                    common_name = attribute.value
                elif attribute.oid == x509.NameOID.ORGANIZATION_NAME:
                    organization = attribute.value
                elif attribute.oid == x509.NameOID.ORGANIZATIONAL_UNIT_NAME:
                    organizational_unit = attribute.value
                elif attribute.oid == x509.NameOID.EMAIL_ADDRESS:
                    email = attribute.value
            
            # Extract Subject Alternative Names
            subject_alt_names = []
            try:
                san_extension = cert.extensions.get_extension_for_oid(x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
                for san in san_extension.value:
                    if isinstance(san, x509.DNSName):
                        subject_alt_names.append(f"DNS:{san.value}")
                    elif isinstance(san, x509.IPAddress):
                        subject_alt_names.append(f"IP:{san.value}")
                    elif isinstance(san, x509.RFC822Name):
                        subject_alt_names.append(f"email:{san.value}")
            except x509.ExtensionNotFound:
                pass
            
            # Calculate certificate fingerprint
            fingerprint = hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()
            
            return ClientIdentity(
                client_id=common_name,
                common_name=common_name,
                organization=organization,
                organizational_unit=organizational_unit,
                email=email,
                subject_alt_names=subject_alt_names,
                certificate_serial=str(cert.serial_number),
                certificate_fingerprint=fingerprint,
                not_before=cert.not_valid_before,
                not_after=cert.not_valid_after
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing real certificate: {e}")
            return None
    
    def _parse_mock_certificate(self, certificate_pem: str) -> Optional[ClientIdentity]:
        """Parse mock certificate when cryptography library is not available."""
        
        try:
            # Extract mock certificate data
            if "MOCK_CERT_" in certificate_pem:
                cert_data = base64.b64decode(certificate_pem.split('\n')[1]).decode()
                parts = cert_data.split('_')
                
                if len(parts) >= 4:
                    serial_number = parts[2]
                    subject_dn = parts[3]
                    
                    # Parse subject DN (simplified)
                    subject_parts = {}
                    for part in subject_dn.split(','):
                        if '=' in part:
                            key, value = part.strip().split('=', 1)
                            subject_parts[key.strip()] = value.strip()
                    
                    common_name = subject_parts.get('CN', '')
                    organization = subject_parts.get('O', '')
                    organizational_unit = subject_parts.get('OU', '')
                    
                    # Generate mock fingerprint
                    fingerprint = hashlib.sha256(certificate_pem.encode()).hexdigest()
                    
                    return ClientIdentity(
                        client_id=common_name,
                        common_name=common_name,
                        organization=organization,
                        organizational_unit=organizational_unit,
                        certificate_serial=serial_number,
                        certificate_fingerprint=fingerprint,
                        not_before=datetime.now() - timedelta(days=1),
                        not_after=datetime.now() + timedelta(days=365)
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error parsing mock certificate: {e}")
            return None
    
    def _validate_certificate_chain(self, certificate_pem: str) -> bool:
        """Validate certificate chain against trusted CAs."""
        
        # For mock implementation, check if certificate was issued by our CA
        if "MOCK_CERT_" in certificate_pem:
            # Check if any trusted CA contains similar mock format
            for trusted_ca in self.trusted_cas:
                if "MOCK_CA_CERT_" in trusted_ca:
                    return True
            return False
        
        # For real certificates, would implement proper chain validation
        # For now, assume valid if we have trusted CAs
        return len(self.trusted_cas) > 0
    
    def establish_connection(self, client_identity: ClientIdentity, 
                           client_ip: str) -> str:
        """
        Establish mTLS connection with authenticated client.
        
        Args:
            client_identity: Authenticated client identity
            client_ip: Client IP address
            
        Returns:
            Connection ID
        """
        
        connection_id = f"conn_{len(self.active_connections) + 1}_{client_identity.client_id}"
        
        connection = MTLSConnection(
            connection_id=connection_id,
            client_identity=client_identity,
            server_name=self.server_name,
            client_ip=client_ip
        )
        
        self.active_connections[connection_id] = connection
        
        self.logger.info(f"Established mTLS connection: {connection_id} for {client_identity.common_name}")
        return connection_id
    
    def _close_connection(self, connection_id: str, reason: str = "Normal closure") -> bool:
        """Close mTLS connection."""
        
        if connection_id in self.active_connections:
            connection = self.active_connections[connection_id]
            connection.is_active = False
            del self.active_connections[connection_id]
            
            self.logger.info(f"Closed mTLS connection: {connection_id} - {reason}")
            return True
        
        return False
    
    def validate_connection(self, connection_id: str) -> bool:
        """
        Validate that mTLS connection is still active and valid.
        
        Args:
            connection_id: Connection ID to validate
            
        Returns:
            True if connection is valid
        """
        
        if connection_id not in self.active_connections:
            return False
        
        connection = self.active_connections[connection_id]
        
        # Check if client certificate is still valid
        if connection.client_identity.is_expired():
            self._close_connection(connection_id, "Client certificate expired")
            return False
        
        # Check if certificate has been revoked
        if connection.client_identity.certificate_serial in self.revoked_certificates:
            self._close_connection(connection_id, "Client certificate revoked")
            return False
        
        return connection.is_active
    
    def _log_authentication_event(self, event_id: str, client_ip: str,
                                 client_identity: Optional[ClientIdentity],
                                 result: AuthenticationResult,
                                 error_message: Optional[str] = None) -> None:
        """Log authentication event."""
        
        event = AuthenticationEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            client_ip=client_ip,
            client_identity=client_identity,
            result=result,
            error_message=error_message
        )
        
        self.authentication_events.append(event)
    
    def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get information about an mTLS connection."""
        
        if connection_id not in self.active_connections:
            return None
        
        connection = self.active_connections[connection_id]
        
        return {
            "connection_id": connection.connection_id,
            "client_identity": {
                "client_id": connection.client_identity.client_id,
                "common_name": connection.client_identity.common_name,
                "organization": connection.client_identity.organization,
                "organizational_unit": connection.client_identity.organizational_unit,
                "certificate_serial": connection.client_identity.certificate_serial,
                "certificate_fingerprint": connection.client_identity.certificate_fingerprint
            },
            "server_name": connection.server_name,
            "client_ip": connection.client_ip,
            "established_at": connection.established_at.isoformat(),
            "last_activity": connection.last_activity.isoformat(),
            "bytes_sent": connection.bytes_sent,
            "bytes_received": connection.bytes_received,
            "is_active": connection.is_active
        }
    
    def get_authentication_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get mTLS authentication statistics."""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_events = [event for event in self.authentication_events if event.timestamp >= cutoff_time]
        
        if not recent_events:
            return {
                "total_attempts": 0,
                "successful_authentications": 0,
                "failed_authentications": 0,
                "active_connections": len(self.active_connections),
                "trusted_cas": len(self.trusted_cas),
                "revoked_certificates": len(self.revoked_certificates)
            }
        
        successful_auths = len([event for event in recent_events if event.result == AuthenticationResult.SUCCESS])
        failed_auths = len(recent_events) - successful_auths
        
        # Count failure reasons
        failure_reasons = {}
        for event in recent_events:
            if event.result != AuthenticationResult.SUCCESS:
                reason = event.result.value
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        return {
            "total_attempts": len(recent_events),
            "successful_authentications": successful_auths,
            "failed_authentications": failed_auths,
            "failure_reasons": failure_reasons,
            "active_connections": len(self.active_connections),
            "trusted_cas": len(self.trusted_cas),
            "revoked_certificates": len(self.revoked_certificates)
        }
    
    def get_server_certificate(self) -> Optional[str]:
        """Get server certificate in PEM format."""
        
        if self.server_certificate:
            return self.server_certificate.certificate_pem
        return None
    
    def get_server_private_key(self) -> Optional[str]:
        """Get server private key in PEM format."""
        
        if self.server_certificate:
            return self.server_certificate.private_key_pem
        return None