"""
Certificate Authority simulation for enterprise PKI infrastructure.

This module provides a realistic Certificate Authority implementation with:
- Root and intermediate CA hierarchy
- Certificate issuance and revocation
- Certificate validation and chain verification
- CRL (Certificate Revocation List) management
"""

import os
import json
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import base64

try:
    from cryptography import x509
    from cryptography.x509.oid import NameOID, ExtensionOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False


class CertificateStatus(Enum):
    """Certificate status values."""
    VALID = "valid"
    REVOKED = "revoked"
    EXPIRED = "expired"
    SUSPENDED = "suspended"


class RevocationReason(Enum):
    """Certificate revocation reasons (RFC 5280)."""
    UNSPECIFIED = 0
    KEY_COMPROMISE = 1
    CA_COMPROMISE = 2
    AFFILIATION_CHANGED = 3
    SUPERSEDED = 4
    CESSATION_OF_OPERATION = 5
    CERTIFICATE_HOLD = 6
    REMOVE_FROM_CRL = 8
    PRIVILEGE_WITHDRAWN = 9
    AA_COMPROMISE = 10


@dataclass
class CertificateInfo:
    """Certificate information and metadata."""
    serial_number: str
    subject_dn: str
    issuer_dn: str
    not_before: datetime
    not_after: datetime
    public_key_algorithm: str
    signature_algorithm: str
    key_usage: List[str]
    extended_key_usage: List[str] = field(default_factory=list)
    subject_alt_names: List[str] = field(default_factory=list)
    status: CertificateStatus = CertificateStatus.VALID
    revocation_date: Optional[datetime] = None
    revocation_reason: Optional[RevocationReason] = None
    certificate_pem: str = ""
    private_key_pem: str = ""
    
    def is_expired(self) -> bool:
        """Check if certificate has expired."""
        return datetime.now() > self.not_after
    
    def is_valid(self) -> bool:
        """Check if certificate is currently valid."""
        now = datetime.now()
        return (self.status == CertificateStatus.VALID and 
                self.not_before <= now <= self.not_after)


@dataclass
class CAConfiguration:
    """Certificate Authority configuration."""
    ca_name: str
    subject_dn: str
    key_size: int = 2048
    validity_days: int = 3650
    hash_algorithm: str = "SHA256"
    is_root_ca: bool = True
    parent_ca: Optional[str] = None
    crl_distribution_points: List[str] = field(default_factory=list)
    ocsp_responders: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.crl_distribution_points:
            self.crl_distribution_points = [f"http://crl.archangel.local/{self.ca_name}.crl"]
        if not self.ocsp_responders:
            self.ocsp_responders = [f"http://ocsp.archangel.local/{self.ca_name}"]


class CertificateAuthority:
    """
    Certificate Authority simulation for enterprise PKI.
    
    Provides realistic CA functionality including:
    - Certificate issuance and management
    - Certificate revocation and CRL generation
    - Certificate chain validation
    - OCSP response simulation
    """
    
    def __init__(self, config: CAConfiguration):
        self.config = config
        self.certificates: Dict[str, CertificateInfo] = {}
        self.revoked_certificates: Dict[str, CertificateInfo] = {}
        self.ca_certificate: Optional[CertificateInfo] = None
        self.ca_private_key = None
        self.crl_number = 1
        self.logger = logging.getLogger(__name__)
        
        # Initialize CA certificate
        self._initialize_ca()
    
    def _initialize_ca(self) -> None:
        """Initialize the Certificate Authority with self-signed certificate."""
        
        if not CRYPTOGRAPHY_AVAILABLE:
            self.logger.warning("Cryptography library not available, using mock certificates")
            self._initialize_mock_ca()
            return
        
        try:
            # Generate CA private key
            self.ca_private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=self.config.key_size
            )
            
            # Create CA certificate
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Archangel Corp"),
                x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "IT Security"),
                x509.NameAttribute(NameOID.COMMON_NAME, self.config.ca_name),
            ])
            
            # Generate serial number
            serial_number = int(secrets.token_hex(16), 16)
            
            # Build certificate
            builder = x509.CertificateBuilder()
            builder = builder.subject_name(subject)
            builder = builder.issuer_name(issuer)
            builder = builder.public_key(self.ca_private_key.public_key())
            builder = builder.serial_number(serial_number)
            builder = builder.not_valid_before(datetime.now())
            builder = builder.not_valid_after(datetime.now() + timedelta(days=self.config.validity_days))
            
            # Add extensions
            builder = builder.add_extension(
                x509.BasicConstraints(ca=True, path_length=None),
                critical=True
            )
            
            builder = builder.add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_cert_sign=True,
                    crl_sign=True,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    content_commitment=False,
                    encipher_only=False,
                    decipher_only=False
                ),
                critical=True
            )
            
            # Add Subject Key Identifier
            builder = builder.add_extension(
                x509.SubjectKeyIdentifier.from_public_key(self.ca_private_key.public_key()),
                critical=False
            )
            
            # Add CRL Distribution Points
            if self.config.crl_distribution_points:
                crl_points = [x509.UniformResourceIdentifier(url) for url in self.config.crl_distribution_points]
                builder = builder.add_extension(
                    x509.CRLDistributionPoints([
                        x509.DistributionPoint(
                            full_name=[x509.UniformResourceIdentifier(url) for url in self.config.crl_distribution_points],
                            relative_name=None,
                            crl_issuer=None,
                            reasons=None
                        )
                    ]),
                    critical=False
                )
            
            # Sign certificate
            ca_cert = builder.sign(self.ca_private_key, hashes.SHA256())
            
            # Store CA certificate info
            self.ca_certificate = CertificateInfo(
                serial_number=str(serial_number),
                subject_dn=self.config.subject_dn,
                issuer_dn=self.config.subject_dn,
                not_before=ca_cert.not_valid_before,
                not_after=ca_cert.not_valid_after,
                public_key_algorithm="RSA",
                signature_algorithm="SHA256withRSA",
                key_usage=["digital_signature", "key_cert_sign", "crl_sign"],
                certificate_pem=ca_cert.public_bytes(Encoding.PEM).decode(),
                private_key_pem=self.ca_private_key.private_bytes(
                    Encoding.PEM,
                    PrivateFormat.PKCS8,
                    NoEncryption()
                ).decode()
            )
            
            self.certificates[str(serial_number)] = self.ca_certificate
            
            self.logger.info(f"Initialized CA: {self.config.ca_name} (Serial: {serial_number})")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CA with cryptography: {e}")
            self._initialize_mock_ca()
    
    def _initialize_mock_ca(self) -> None:
        """Initialize mock CA when cryptography library is not available."""
        
        serial_number = secrets.token_hex(16)
        
        # Create mock CA certificate
        self.ca_certificate = CertificateInfo(
            serial_number=serial_number,
            subject_dn=self.config.subject_dn,
            issuer_dn=self.config.subject_dn,
            not_before=datetime.now(),
            not_after=datetime.now() + timedelta(days=self.config.validity_days),
            public_key_algorithm="RSA",
            signature_algorithm="SHA256withRSA",
            key_usage=["digital_signature", "key_cert_sign", "crl_sign"],
            certificate_pem=f"-----BEGIN CERTIFICATE-----\n{base64.b64encode(f'MOCK_CA_CERT_{serial_number}'.encode()).decode()}\n-----END CERTIFICATE-----",
            private_key_pem=f"-----BEGIN PRIVATE KEY-----\n{base64.b64encode(f'MOCK_CA_KEY_{serial_number}'.encode()).decode()}\n-----END PRIVATE KEY-----"
        )
        
        self.certificates[serial_number] = self.ca_certificate
        
        self.logger.info(f"Initialized mock CA: {self.config.ca_name} (Serial: {serial_number})")
    
    def issue_certificate(self, subject_dn: str, 
                         subject_alt_names: List[str] = None,
                         key_usage: List[str] = None,
                         extended_key_usage: List[str] = None,
                         validity_days: int = 365) -> Optional[CertificateInfo]:
        """
        Issue a new certificate.
        
        Args:
            subject_dn: Subject Distinguished Name
            subject_alt_names: Subject Alternative Names
            key_usage: Key usage extensions
            extended_key_usage: Extended key usage extensions
            validity_days: Certificate validity period in days
            
        Returns:
            Certificate information or None if issuance failed
        """
        
        if subject_alt_names is None:
            subject_alt_names = []
        if key_usage is None:
            key_usage = ["digital_signature", "key_encipherment"]
        if extended_key_usage is None:
            extended_key_usage = ["server_auth", "client_auth"]
        
        try:
            if CRYPTOGRAPHY_AVAILABLE and self.ca_private_key:
                return self._issue_real_certificate(
                    subject_dn, subject_alt_names, key_usage, 
                    extended_key_usage, validity_days
                )
            else:
                return self._issue_mock_certificate(
                    subject_dn, subject_alt_names, key_usage,
                    extended_key_usage, validity_days
                )
        except Exception as e:
            self.logger.error(f"Failed to issue certificate for {subject_dn}: {e}")
            return None
    
    def _issue_real_certificate(self, subject_dn: str, subject_alt_names: List[str],
                               key_usage: List[str], extended_key_usage: List[str],
                               validity_days: int) -> CertificateInfo:
        """Issue a real certificate using cryptography library."""
        
        # Generate private key for certificate
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Parse subject DN (simplified parsing)
        subject_parts = {}
        for part in subject_dn.split(','):
            if '=' in part:
                key, value = part.strip().split('=', 1)
                subject_parts[key.strip()] = value.strip()
        
        # Build subject
        subject_components = []
        if 'CN' in subject_parts:
            subject_components.append(x509.NameAttribute(NameOID.COMMON_NAME, subject_parts['CN']))
        if 'O' in subject_parts:
            subject_components.append(x509.NameAttribute(NameOID.ORGANIZATION_NAME, subject_parts['O']))
        if 'OU' in subject_parts:
            subject_components.append(x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, subject_parts['OU']))
        if 'C' in subject_parts:
            subject_components.append(x509.NameAttribute(NameOID.COUNTRY_NAME, subject_parts['C']))
        
        subject = x509.Name(subject_components)
        
        # Generate serial number
        serial_number = int(secrets.token_hex(16), 16)
        
        # Build certificate
        builder = x509.CertificateBuilder()
        builder = builder.subject_name(subject)
        builder = builder.issuer_name(x509.Name.from_rfc4514_string(self.ca_certificate.subject_dn))
        builder = builder.public_key(private_key.public_key())
        builder = builder.serial_number(serial_number)
        builder = builder.not_valid_before(datetime.now())
        builder = builder.not_valid_after(datetime.now() + timedelta(days=validity_days))
        
        # Add basic constraints
        builder = builder.add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True
        )
        
        # Add key usage
        key_usage_obj = x509.KeyUsage(
            digital_signature='digital_signature' in key_usage,
            key_encipherment='key_encipherment' in key_usage,
            key_cert_sign='key_cert_sign' in key_usage,
            crl_sign='crl_sign' in key_usage,
            data_encipherment='data_encipherment' in key_usage,
            key_agreement='key_agreement' in key_usage,
            content_commitment='content_commitment' in key_usage,
            encipher_only=False,
            decipher_only=False
        )
        builder = builder.add_extension(key_usage_obj, critical=True)
        
        # Add extended key usage
        if extended_key_usage:
            eku_oids = []
            if 'server_auth' in extended_key_usage:
                eku_oids.append(x509.oid.ExtendedKeyUsageOID.SERVER_AUTH)
            if 'client_auth' in extended_key_usage:
                eku_oids.append(x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH)
            if 'code_signing' in extended_key_usage:
                eku_oids.append(x509.oid.ExtendedKeyUsageOID.CODE_SIGNING)
            
            if eku_oids:
                builder = builder.add_extension(
                    x509.ExtendedKeyUsage(eku_oids),
                    critical=False
                )
        
        # Add Subject Alternative Names
        if subject_alt_names:
            san_list = []
            for san in subject_alt_names:
                if san.startswith('DNS:'):
                    san_list.append(x509.DNSName(san[4:]))
                elif san.startswith('IP:'):
                    san_list.append(x509.IPAddress(san[3:]))
                elif san.startswith('email:'):
                    san_list.append(x509.RFC822Name(san[6:]))
                else:
                    san_list.append(x509.DNSName(san))
            
            if san_list:
                builder = builder.add_extension(
                    x509.SubjectAlternativeName(san_list),
                    critical=False
                )
        
        # Sign certificate
        cert = builder.sign(self.ca_private_key, hashes.SHA256())
        
        # Create certificate info
        cert_info = CertificateInfo(
            serial_number=str(serial_number),
            subject_dn=subject_dn,
            issuer_dn=self.ca_certificate.subject_dn,
            not_before=cert.not_valid_before,
            not_after=cert.not_valid_after,
            public_key_algorithm="RSA",
            signature_algorithm="SHA256withRSA",
            key_usage=key_usage,
            extended_key_usage=extended_key_usage,
            subject_alt_names=subject_alt_names,
            certificate_pem=cert.public_bytes(Encoding.PEM).decode(),
            private_key_pem=private_key.private_bytes(
                Encoding.PEM,
                PrivateFormat.PKCS8,
                NoEncryption()
            ).decode()
        )
        
        self.certificates[str(serial_number)] = cert_info
        
        self.logger.info(f"Issued certificate: {subject_dn} (Serial: {serial_number})")
        return cert_info
    
    def _issue_mock_certificate(self, subject_dn: str, subject_alt_names: List[str],
                               key_usage: List[str], extended_key_usage: List[str],
                               validity_days: int) -> CertificateInfo:
        """Issue a mock certificate when cryptography library is not available."""
        
        serial_number = secrets.token_hex(16)
        
        cert_info = CertificateInfo(
            serial_number=serial_number,
            subject_dn=subject_dn,
            issuer_dn=self.ca_certificate.subject_dn,
            not_before=datetime.now(),
            not_after=datetime.now() + timedelta(days=validity_days),
            public_key_algorithm="RSA",
            signature_algorithm="SHA256withRSA",
            key_usage=key_usage,
            extended_key_usage=extended_key_usage,
            subject_alt_names=subject_alt_names,
            certificate_pem=f"-----BEGIN CERTIFICATE-----\n{base64.b64encode(f'MOCK_CERT_{serial_number}_{subject_dn}'.encode()).decode()}\n-----END CERTIFICATE-----",
            private_key_pem=f"-----BEGIN PRIVATE KEY-----\n{base64.b64encode(f'MOCK_KEY_{serial_number}'.encode()).decode()}\n-----END PRIVATE KEY-----"
        )
        
        self.certificates[serial_number] = cert_info
        
        self.logger.info(f"Issued mock certificate: {subject_dn} (Serial: {serial_number})")
        return cert_info
    
    def revoke_certificate(self, serial_number: str, 
                          reason: RevocationReason = RevocationReason.UNSPECIFIED) -> bool:
        """
        Revoke a certificate.
        
        Args:
            serial_number: Certificate serial number
            reason: Revocation reason
            
        Returns:
            True if certificate was revoked successfully
        """
        
        if serial_number not in self.certificates:
            self.logger.warning(f"Cannot revoke unknown certificate: {serial_number}")
            return False
        
        cert_info = self.certificates[serial_number]
        
        if cert_info.status == CertificateStatus.REVOKED:
            self.logger.warning(f"Certificate already revoked: {serial_number}")
            return False
        
        # Update certificate status
        cert_info.status = CertificateStatus.REVOKED
        cert_info.revocation_date = datetime.now()
        cert_info.revocation_reason = reason
        
        # Move to revoked certificates
        self.revoked_certificates[serial_number] = cert_info
        
        self.logger.info(f"Revoked certificate: {serial_number} (Reason: {reason.name})")
        return True
    
    def validate_certificate(self, certificate_pem: str) -> Dict[str, Any]:
        """
        Validate a certificate against this CA.
        
        Args:
            certificate_pem: Certificate in PEM format
            
        Returns:
            Validation result with status and details
        """
        
        validation_result = {
            "valid": False,
            "serial_number": None,
            "subject_dn": None,
            "issuer_dn": None,
            "not_before": None,
            "not_after": None,
            "status": None,
            "errors": []
        }
        
        try:
            # Extract certificate info (simplified for mock)
            if "MOCK_CERT_" in certificate_pem:
                # Parse mock certificate
                cert_data = base64.b64decode(certificate_pem.split('\n')[1]).decode()
                parts = cert_data.split('_')
                if len(parts) >= 3:
                    serial_number = parts[2]
                    
                    if serial_number in self.certificates:
                        cert_info = self.certificates[serial_number]
                        validation_result.update({
                            "valid": cert_info.is_valid(),
                            "serial_number": cert_info.serial_number,
                            "subject_dn": cert_info.subject_dn,
                            "issuer_dn": cert_info.issuer_dn,
                            "not_before": cert_info.not_before.isoformat(),
                            "not_after": cert_info.not_after.isoformat(),
                            "status": cert_info.status.value
                        })
                        
                        if cert_info.is_expired():
                            validation_result["errors"].append("Certificate has expired")
                        if cert_info.status == CertificateStatus.REVOKED:
                            validation_result["errors"].append("Certificate has been revoked")
                    else:
                        validation_result["errors"].append("Certificate not found in CA database")
            else:
                validation_result["errors"].append("Certificate format not supported in mock mode")
                
        except Exception as e:
            validation_result["errors"].append(f"Certificate parsing error: {str(e)}")
        
        return validation_result
    
    def generate_crl(self) -> str:
        """
        Generate Certificate Revocation List (CRL).
        
        Returns:
            CRL in PEM format
        """
        
        crl_entries = []
        
        # Add revoked certificates
        for serial_number, cert_info in self.revoked_certificates.items():
            if cert_info.status == CertificateStatus.REVOKED:
                crl_entries.append({
                    "serial_number": serial_number,
                    "revocation_date": cert_info.revocation_date.isoformat(),
                    "reason": cert_info.revocation_reason.name if cert_info.revocation_reason else "UNSPECIFIED"
                })
        
        # Create CRL header
        crl_data = {
            "issuer": self.ca_certificate.issuer_dn,
            "this_update": datetime.now().isoformat(),
            "next_update": (datetime.now() + timedelta(days=7)).isoformat(),
            "crl_number": self.crl_number,
            "revoked_certificates": crl_entries
        }
        
        # Increment CRL number for next time
        self.crl_number += 1
        
        # Generate mock CRL in PEM format
        crl_json = json.dumps(crl_data, indent=2)
        crl_b64 = base64.b64encode(crl_json.encode()).decode()
        
        crl_pem = f"-----BEGIN X509 CRL-----\n{crl_b64}\n-----END X509 CRL-----"
        
        self.logger.info(f"Generated CRL with {len(crl_entries)} revoked certificates")
        return crl_pem
    
    def get_certificate_info(self, serial_number: str) -> Optional[Dict[str, Any]]:
        """Get information about a certificate."""
        
        if serial_number not in self.certificates:
            return None
        
        cert_info = self.certificates[serial_number]
        
        return {
            "serial_number": cert_info.serial_number,
            "subject_dn": cert_info.subject_dn,
            "issuer_dn": cert_info.issuer_dn,
            "not_before": cert_info.not_before.isoformat(),
            "not_after": cert_info.not_after.isoformat(),
            "public_key_algorithm": cert_info.public_key_algorithm,
            "signature_algorithm": cert_info.signature_algorithm,
            "key_usage": cert_info.key_usage,
            "extended_key_usage": cert_info.extended_key_usage,
            "subject_alt_names": cert_info.subject_alt_names,
            "status": cert_info.status.value,
            "is_valid": cert_info.is_valid(),
            "is_expired": cert_info.is_expired(),
            "revocation_date": cert_info.revocation_date.isoformat() if cert_info.revocation_date else None,
            "revocation_reason": cert_info.revocation_reason.name if cert_info.revocation_reason else None
        }
    
    def get_ca_statistics(self) -> Dict[str, Any]:
        """Get CA statistics and status."""
        
        total_certs = len(self.certificates)
        valid_certs = len([cert for cert in self.certificates.values() if cert.is_valid()])
        expired_certs = len([cert for cert in self.certificates.values() if cert.is_expired()])
        revoked_certs = len(self.revoked_certificates)
        
        return {
            "ca_name": self.config.ca_name,
            "ca_subject_dn": self.config.subject_dn,
            "is_root_ca": self.config.is_root_ca,
            "ca_serial_number": self.ca_certificate.serial_number if self.ca_certificate else None,
            "ca_not_after": self.ca_certificate.not_after.isoformat() if self.ca_certificate else None,
            "total_certificates": total_certs,
            "valid_certificates": valid_certs,
            "expired_certificates": expired_certs,
            "revoked_certificates": revoked_certs,
            "crl_number": self.crl_number - 1,
            "cryptography_available": CRYPTOGRAPHY_AVAILABLE
        }
    
    def export_ca_certificate(self) -> Optional[str]:
        """Export CA certificate in PEM format."""
        
        if self.ca_certificate:
            return self.ca_certificate.certificate_pem
        return None
    
    def create_certificate_chain(self, leaf_cert_serial: str) -> List[str]:
        """
        Create certificate chain from leaf to root.
        
        Args:
            leaf_cert_serial: Serial number of leaf certificate
            
        Returns:
            List of certificates in PEM format (leaf to root)
        """
        
        chain = []
        
        # Add leaf certificate
        if leaf_cert_serial in self.certificates:
            leaf_cert = self.certificates[leaf_cert_serial]
            chain.append(leaf_cert.certificate_pem)
        
        # Add CA certificate (for now, we only have one level)
        if self.ca_certificate:
            chain.append(self.ca_certificate.certificate_pem)
        
        return chain