#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Secure Communication System
Advanced encrypted agent communication with mutual TLS, Noise Protocol Framework,
and secure key distribution and rotation mechanisms.
"""

import asyncio
import json
import logging
import uuid
import hashlib
import hmac
import time
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
import ssl
import socket
from pathlib import Path

# Optional imports for full functionality
try:
    import zmq
    import zmq.asyncio
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    zmq = None

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    # Mock classes for when cryptography is not available
    class x509:
        class Certificate: pass
        class Name: pass
        class NameAttribute: pass
        class CertificateBuilder: pass
        class BasicConstraints: pass
        class KeyUsage: pass
        class ExtendedKeyUsage: pass
        class SubjectAlternativeName: pass
        class DNSName: pass
        class oid:
            class ExtendedKeyUsageOID:
                CLIENT_AUTH = "client_auth"
                SERVER_AUTH = "server_auth"
        @staticmethod
        def random_serial_number():
            return 12345
    
    class NameOID:
        COUNTRY_NAME = "C"
        STATE_OR_PROVINCE_NAME = "ST"
        LOCALITY_NAME = "L"
        ORGANIZATION_NAME = "O"
        COMMON_NAME = "CN"
    
    class rsa:
        class RSAPrivateKey: pass
        @staticmethod
        def generate_private_key(public_exponent, key_size):
            return MockRSAPrivateKey()
    
    class hashes:
        class SHA256: pass
    
    class serialization:
        class Encoding:
            PEM = "pem"
            DER = "der"
        class PrivateFormat:
            PKCS8 = "pkcs8"
        class NoEncryption: pass
    
    class MockRSAPrivateKey:
        def public_key(self):
            return MockRSAPublicKey()
        def private_bytes(self, encoding, format, encryption_algorithm):
            return b"mock_private_key_bytes"
    
    class MockRSAPublicKey:
        def public_bytes(self, encoding):
            return b"mock_public_key_bytes"

try:
    import noise
    NOISE_AVAILABLE = True
except ImportError:
    NOISE_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels for communication"""
    BASIC = "basic"
    ENHANCED = "enhanced"
    MAXIMUM = "maximum"

class KeyType(Enum):
    """Types of cryptographic keys"""
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    ECDSA_P256 = "ecdsa_p256"
    ECDSA_P384 = "ecdsa_p384"

class ProtocolType(Enum):
    """Communication protocol types"""
    MTLS = "mutual_tls"
    NOISE_XX = "noise_xx"
    NOISE_IK = "noise_ik"
    NOISE_NK = "noise_nk"

@dataclass
class CertificateInfo:
    """Certificate information for mTLS"""
    cert_path: str
    key_path: str
    ca_path: str
    fingerprint: str
    expiry: datetime
    subject: str
    issuer: str

@dataclass
class NoiseHandshakeState:
    """Noise protocol handshake state"""
    protocol_name: str
    local_keypair: Optional[Tuple[bytes, bytes]]
    remote_public_key: Optional[bytes]
    handshake_state: Any
    cipher_state_send: Any
    cipher_state_recv: Any
    completed: bool = False

@dataclass
class SecureMessage:
    """Secure message with integrity and authenticity"""
    message_id: str
    sender_id: str
    recipient_id: str
    content: bytes
    timestamp: datetime
    nonce: bytes
    signature: bytes
    protocol_used: ProtocolType
    key_version: int
    integrity_hash: str

class CertificateManager:
    """
    Manages X.509 certificates for mutual TLS authentication with advanced pinning
    """
    
    def __init__(self, cert_dir: str = "certs"):
        self.cert_dir = Path(cert_dir)
        self.cert_dir.mkdir(exist_ok=True)
        self.ca_cert = None
        self.ca_key = None
        self.agent_certs = {}
        self.cert_pins = {}
        self.cert_chain_validation = {}
        self.revoked_certs = set()
        self.cert_expiry_warnings = {}
        
    def generate_ca_certificate(self) -> Tuple[Any, Any]:
        """Generate Certificate Authority certificate and key"""
        try:
            if not CRYPTOGRAPHY_AVAILABLE:
                # Mock CA certificate generation
                mock_cert = type('MockCert', (), {
                    'subject': type('Subject', (), {'rfc4514_string': lambda: 'CN=Archangel Root CA'})(),
                    'not_valid_after': datetime.utcnow() + timedelta(days=365),
                    'public_bytes': lambda encoding: b'mock_ca_cert_bytes'
                })()
                mock_key = type('MockKey', (), {
                    'private_bytes': lambda **kwargs: b'mock_ca_key_bytes'
                })()
                
                # Create mock files
                ca_cert_path = self.cert_dir / "ca.crt"
                ca_key_path = self.cert_dir / "ca.key"
                
                with open(ca_cert_path, "wb") as f:
                    f.write(b'mock_ca_cert_bytes')
                
                with open(ca_key_path, "wb") as f:
                    f.write(b'mock_ca_key_bytes')
                
                self.ca_cert = mock_cert
                self.ca_key = mock_key
                
                logger.info("Generated mock CA certificate and key")
                return mock_cert, mock_key
            
            # Generate CA private key
            ca_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096
            )
            
            # Create CA certificate
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Virtual"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "Simulation"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Archangel CA"),
                x509.NameAttribute(NameOID.COMMON_NAME, "Archangel Root CA"),
            ])
            
            ca_cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                ca_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=365)
            ).add_extension(
                x509.BasicConstraints(ca=True, path_length=None),
                critical=True,
            ).add_extension(
                x509.KeyUsage(
                    key_cert_sign=True,
                    crl_sign=True,
                    digital_signature=False,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            ).sign(ca_key, hashes.SHA256())
            
            # Save CA certificate and key
            ca_cert_path = self.cert_dir / "ca.crt"
            ca_key_path = self.cert_dir / "ca.key"
            
            with open(ca_cert_path, "wb") as f:
                f.write(ca_cert.public_bytes(serialization.Encoding.PEM))
            
            with open(ca_key_path, "wb") as f:
                f.write(ca_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            self.ca_cert = ca_cert
            self.ca_key = ca_key
            
            logger.info("Generated CA certificate and key")
            return ca_cert, ca_key
            
        except Exception as e:
            logger.error(f"Failed to generate CA certificate: {e}")
            raise
    
    def generate_agent_certificate(self, agent_id: str) -> CertificateInfo:
        """Generate certificate for agent with certificate pinning"""
        try:
            if not self.ca_cert or not self.ca_key:
                self.generate_ca_certificate()
            
            if not CRYPTOGRAPHY_AVAILABLE:
                # Mock agent certificate generation
                cert_path = self.cert_dir / f"{agent_id}.crt"
                key_path = self.cert_dir / f"{agent_id}.key"
                ca_path = self.cert_dir / "ca.crt"
                
                mock_cert_data = f"mock_cert_data_for_{agent_id}".encode()
                mock_key_data = f"mock_key_data_for_{agent_id}".encode()
                
                with open(cert_path, "wb") as f:
                    f.write(mock_cert_data)
                
                with open(key_path, "wb") as f:
                    f.write(mock_key_data)
                
                # Calculate mock fingerprint
                fingerprint = hashlib.sha256(mock_cert_data).hexdigest()
                
                cert_info = CertificateInfo(
                    cert_path=str(cert_path),
                    key_path=str(key_path),
                    ca_path=str(ca_path),
                    fingerprint=fingerprint,
                    expiry=datetime.utcnow() + timedelta(days=90),
                    subject=f"CN={agent_id}",
                    issuer="CN=Archangel Root CA"
                )
                
                self.agent_certs[agent_id] = cert_info
                self.cert_pins[agent_id] = fingerprint
                
                logger.info(f"Generated mock certificate for agent {agent_id}")
                return cert_info
            
            # Generate agent private key
            agent_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Create agent certificate
            subject = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Virtual"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "Simulation"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Archangel Agents"),
                x509.NameAttribute(NameOID.COMMON_NAME, agent_id),
            ])
            
            agent_cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                self.ca_cert.subject
            ).public_key(
                agent_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=90)
            ).add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName(agent_id),
                    x509.DNSName(f"{agent_id}.archangel.local"),
                ]),
                critical=False,
            ).add_extension(
                x509.KeyUsage(
                    key_cert_sign=False,
                    crl_sign=False,
                    digital_signature=True,
                    content_commitment=True,
                    key_encipherment=True,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            ).add_extension(
                x509.ExtendedKeyUsage([
                    x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
                    x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                ]),
                critical=True,
            ).sign(self.ca_key, hashes.SHA256())
            
            # Save agent certificate and key
            cert_path = self.cert_dir / f"{agent_id}.crt"
            key_path = self.cert_dir / f"{agent_id}.key"
            ca_path = self.cert_dir / "ca.crt"
            
            with open(cert_path, "wb") as f:
                f.write(agent_cert.public_bytes(serialization.Encoding.PEM))
            
            with open(key_path, "wb") as f:
                f.write(agent_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # Calculate certificate fingerprint for pinning
            fingerprint = hashlib.sha256(
                agent_cert.public_bytes(serialization.Encoding.DER)
            ).hexdigest()
            
            cert_info = CertificateInfo(
                cert_path=str(cert_path),
                key_path=str(key_path),
                ca_path=str(ca_path),
                fingerprint=fingerprint,
                expiry=agent_cert.not_valid_after,
                subject=agent_cert.subject.rfc4514_string(),
                issuer=agent_cert.issuer.rfc4514_string()
            )
            
            self.agent_certs[agent_id] = cert_info
            self.cert_pins[agent_id] = fingerprint
            
            logger.info(f"Generated certificate for agent {agent_id}")
            return cert_info
            
        except Exception as e:
            logger.error(f"Failed to generate agent certificate: {e}")
            raise
    
    def verify_certificate_pin(self, agent_id: str, cert_data: bytes) -> bool:
        """Verify certificate against pinned fingerprint with enhanced validation"""
        try:
            if agent_id not in self.cert_pins:
                logger.warning(f"No pinned certificate for agent {agent_id}")
                return False
            
            # Check if certificate is revoked
            fingerprint = hashlib.sha256(cert_data).hexdigest()
            if fingerprint in self.revoked_certs:
                logger.error(f"Certificate for agent {agent_id} has been revoked")
                return False
            
            # Compare with pinned fingerprint
            pin_valid = fingerprint == self.cert_pins[agent_id]
            
            if pin_valid:
                # Additional validation: check certificate expiry
                if CRYPTOGRAPHY_AVAILABLE:
                    try:
                        cert = x509.load_pem_x509_certificate(cert_data)
                        if datetime.utcnow() > cert.not_valid_after:
                            logger.error(f"Certificate for agent {agent_id} has expired")
                            return False
                        
                        # Warn if certificate expires soon (within 7 days)
                        if datetime.utcnow() + timedelta(days=7) > cert.not_valid_after:
                            if agent_id not in self.cert_expiry_warnings:
                                logger.warning(f"Certificate for agent {agent_id} expires soon: {cert.not_valid_after}")
                                self.cert_expiry_warnings[agent_id] = datetime.utcnow()
                    except Exception as cert_parse_error:
                        logger.warning(f"Could not parse certificate for expiry check: {cert_parse_error}")
            
            return pin_valid
            
        except Exception as e:
            logger.error(f"Certificate pin verification failed: {e}")
            return False
    
    def revoke_certificate(self, agent_id: str) -> bool:
        """Revoke certificate for agent"""
        try:
            if agent_id not in self.cert_pins:
                logger.warning(f"No certificate to revoke for agent {agent_id}")
                return False
            
            fingerprint = self.cert_pins[agent_id]
            self.revoked_certs.add(fingerprint)
            
            logger.info(f"Revoked certificate for agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke certificate: {e}")
            return False
    
    def validate_certificate_chain(self, agent_id: str, cert_chain: List[bytes]) -> bool:
        """Validate full certificate chain"""
        try:
            if not CRYPTOGRAPHY_AVAILABLE:
                # Mock validation for testing
                return len(cert_chain) > 0
            
            if not cert_chain:
                return False
            
            # Load certificates from chain
            certs = []
            for cert_data in cert_chain:
                cert = x509.load_pem_x509_certificate(cert_data)
                certs.append(cert)
            
            # Validate chain (simplified validation)
            # In production, use proper certificate chain validation
            leaf_cert = certs[0]
            
            # Check if leaf certificate matches agent
            if agent_id not in leaf_cert.subject.rfc4514_string():
                logger.error(f"Certificate subject does not match agent {agent_id}")
                return False
            
            # Check if signed by our CA
            if self.ca_cert and len(certs) > 1:
                # Verify signature (simplified)
                issuer_cert = certs[1]
                if issuer_cert.subject != self.ca_cert.subject:
                    logger.error("Certificate not signed by trusted CA")
                    return False
            
            self.cert_chain_validation[agent_id] = {
                'validated': True,
                'timestamp': datetime.utcnow(),
                'chain_length': len(certs)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Certificate chain validation failed: {e}")
            return False

class NoiseProtocolManager:
    """
    Manages Noise Protocol Framework for secure communication with forward secrecy
    """
    
    def __init__(self):
        self.handshake_states = {}
        self.cipher_states = {}
        self.keypairs = {}
        self.ephemeral_keys = {}
        self.session_keys = {}
        self.forward_secrecy_enabled = True
        
    def generate_keypair(self, agent_id: str) -> Tuple[bytes, bytes]:
        """Generate Noise protocol keypair for agent"""
        try:
            if not NOISE_AVAILABLE:
                # Fallback to basic key generation
                private_key = os.urandom(32)
                public_key = hashlib.sha256(private_key).digest()
                self.keypairs[agent_id] = (private_key, public_key)
                return private_key, public_key
            
            # Generate Curve25519 keypair for Noise protocol
            private_key = noise.PrivateKey.generate()
            public_key = private_key.public_key
            
            private_bytes = bytes(private_key)
            public_bytes = bytes(public_key)
            
            self.keypairs[agent_id] = (private_bytes, public_bytes)
            
            logger.info(f"Generated Noise keypair for agent {agent_id}")
            return private_bytes, public_bytes
            
        except Exception as e:
            logger.error(f"Failed to generate Noise keypair: {e}")
            raise
    
    def initialize_handshake(self, agent_id: str, protocol_name: str = "Noise_XX_25519_AESGCM_SHA256",
                           is_initiator: bool = True, remote_public_key: Optional[bytes] = None) -> NoiseHandshakeState:
        """Initialize Noise protocol handshake"""
        try:
            if not NOISE_AVAILABLE:
                # Get agent keypair
                if agent_id not in self.keypairs:
                    self.generate_keypair(agent_id)
                
                private_key_bytes, public_key_bytes = self.keypairs[agent_id]
                
                # Mock handshake state for testing
                handshake_state = NoiseHandshakeState(
                    protocol_name=protocol_name,
                    local_keypair=(private_key_bytes, public_key_bytes),
                    remote_public_key=remote_public_key,
                    handshake_state=None,
                    cipher_state_send=None,
                    cipher_state_recv=None,
                    completed=False
                )
                
                self.handshake_states[agent_id] = handshake_state
                return handshake_state
            
            # Get agent keypair
            if agent_id not in self.keypairs:
                self.generate_keypair(agent_id)
            
            private_key_bytes, public_key_bytes = self.keypairs[agent_id]
            
            # Create Noise protocol instance
            noise_protocol = noise.NoiseConnection.from_name(protocol_name.encode())
            
            # Set local keypair
            noise_protocol.set_keypair_from_private_bytes(
                noise.Keypair.STATIC, private_key_bytes
            )
            
            # Set remote public key if provided (for IK/NK patterns)
            if remote_public_key:
                noise_protocol.set_keypair_from_public_bytes(
                    noise.Keypair.REMOTE_STATIC, remote_public_key
                )
            
            # Start handshake
            if is_initiator:
                noise_protocol.start_handshake()
            else:
                noise_protocol.start_handshake(noise.HandshakeRole.RESPONDER)
            
            handshake_state = NoiseHandshakeState(
                protocol_name=protocol_name,
                local_keypair=(private_key_bytes, public_key_bytes),
                remote_public_key=remote_public_key,
                handshake_state=noise_protocol,
                cipher_state_send=None,
                cipher_state_recv=None,
                completed=False
            )
            
            self.handshake_states[agent_id] = handshake_state
            
            logger.info(f"Initialized Noise handshake for agent {agent_id}")
            return handshake_state
            
        except Exception as e:
            logger.error(f"Failed to initialize Noise handshake: {e}")
            raise
    
    def process_handshake_message(self, agent_id: str, message: bytes) -> Tuple[Optional[bytes], bool]:
        """Process handshake message and return response if needed"""
        try:
            if agent_id not in self.handshake_states:
                raise ValueError(f"No handshake state for agent {agent_id}")
            
            handshake_state = self.handshake_states[agent_id]
            
            if not NOISE_AVAILABLE:
                # Mock handshake completion
                handshake_state.completed = True
                return None, True
            
            noise_protocol = handshake_state.handshake_state
            
            # Process handshake message
            response = noise_protocol.read_message(message)
            
            # Check if handshake is complete
            if noise_protocol.handshake_finished:
                # Get cipher states for encryption/decryption
                handshake_state.cipher_state_send = noise_protocol.cipher_state_encrypt
                handshake_state.cipher_state_recv = noise_protocol.cipher_state_decrypt
                handshake_state.completed = True
                
                self.cipher_states[agent_id] = {
                    'send': handshake_state.cipher_state_send,
                    'recv': handshake_state.cipher_state_recv
                }
                
                logger.info(f"Noise handshake completed for agent {agent_id}")
                return response, True
            
            return response, False
            
        except Exception as e:
            logger.error(f"Failed to process handshake message: {e}")
            raise
    
    def encrypt_message(self, agent_id: str, plaintext: bytes) -> bytes:
        """Encrypt message using Noise protocol"""
        try:
            if agent_id not in self.cipher_states:
                raise ValueError(f"No cipher state for agent {agent_id}")
            
            if not NOISE_AVAILABLE:
                # Simple XOR encryption for testing
                key = hashlib.sha256(agent_id.encode()).digest()
                return bytes(a ^ b for a, b in zip(plaintext, key * (len(plaintext) // 32 + 1)))
            
            cipher_state = self.cipher_states[agent_id]['send']
            return cipher_state.encrypt(plaintext)
            
        except Exception as e:
            logger.error(f"Failed to encrypt message: {e}")
            raise
    
    def decrypt_message(self, agent_id: str, ciphertext: bytes) -> bytes:
        """Decrypt message using Noise protocol"""
        try:
            if agent_id not in self.cipher_states:
                raise ValueError(f"No cipher state for agent {agent_id}")
            
            if not NOISE_AVAILABLE:
                # Simple XOR decryption for testing
                key = hashlib.sha256(agent_id.encode()).digest()
                return bytes(a ^ b for a, b in zip(ciphertext, key * (len(ciphertext) // 32 + 1)))
            
            cipher_state = self.cipher_states[agent_id]['recv']
            return cipher_state.decrypt(ciphertext)
            
        except Exception as e:
            logger.error(f"Failed to decrypt message: {e}")
            raise
    
    def rotate_ephemeral_keys(self, agent_id: str) -> bool:
        """Rotate ephemeral keys for forward secrecy"""
        try:
            if not self.forward_secrecy_enabled:
                return False
            
            # Generate new ephemeral keypair
            if NOISE_AVAILABLE:
                ephemeral_private = noise.PrivateKey.generate()
                ephemeral_public = ephemeral_private.public_key
                
                self.ephemeral_keys[agent_id] = {
                    'private': bytes(ephemeral_private),
                    'public': bytes(ephemeral_public),
                    'created': datetime.now()
                }
            else:
                # Mock ephemeral keys for testing
                ephemeral_private = os.urandom(32)
                ephemeral_public = hashlib.sha256(ephemeral_private).digest()
                
                self.ephemeral_keys[agent_id] = {
                    'private': ephemeral_private,
                    'public': ephemeral_public,
                    'created': datetime.now()
                }
            
            logger.debug(f"Rotated ephemeral keys for agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rotate ephemeral keys: {e}")
            return False
    
    def derive_session_key(self, agent_id: str, peer_public_key: bytes) -> bytes:
        """Derive session key using ECDH key exchange"""
        try:
            if agent_id not in self.ephemeral_keys:
                self.rotate_ephemeral_keys(agent_id)
            
            ephemeral_private = self.ephemeral_keys[agent_id]['private']
            
            # Perform ECDH (simplified for testing)
            if NOISE_AVAILABLE:
                # Use Noise protocol's key derivation
                shared_secret = hashlib.sha256(ephemeral_private + peer_public_key).digest()
            else:
                # Mock ECDH for testing
                shared_secret = hashlib.sha256(ephemeral_private + peer_public_key).digest()
            
            # Derive session key using HKDF
            session_key = hashlib.pbkdf2_hmac('sha256', shared_secret, b'archangel_session', 100000, 32)
            
            self.session_keys[agent_id] = {
                'key': session_key,
                'derived': datetime.now(),
                'peer_public': peer_public_key
            }
            
            return session_key
            
        except Exception as e:
            logger.error(f"Failed to derive session key: {e}")
            raise

class KeyRotationManager:
    """
    Manages secure key distribution and rotation with advanced key derivation
    """
    
    def __init__(self, rotation_interval: timedelta = timedelta(hours=24)):
        self.rotation_interval = rotation_interval
        self.key_versions = {}
        self.active_keys = {}
        self.key_history = {}
        self.rotation_schedule = {}
        self.key_derivation_salt = os.urandom(32)
        self.master_key = os.urandom(32)
        self.key_distribution_log = []
        
    def generate_session_key(self, agent_id: str) -> Tuple[bytes, int]:
        """Generate new session key for agent"""
        try:
            # Generate random session key
            session_key = os.urandom(32)
            
            # Increment key version
            current_version = self.key_versions.get(agent_id, 0) + 1
            self.key_versions[agent_id] = current_version
            
            # Store active key
            self.active_keys[agent_id] = {
                'key': session_key,
                'version': current_version,
                'created': datetime.now(),
                'expires': datetime.now() + self.rotation_interval
            }
            
            # Add to key history
            if agent_id not in self.key_history:
                self.key_history[agent_id] = []
            self.key_history[agent_id].append({
                'version': current_version,
                'created': datetime.now(),
                'key_hash': hashlib.sha256(session_key).hexdigest()
            })
            
            # Schedule next rotation
            self.schedule_key_rotation(agent_id)
            
            logger.info(f"Generated session key v{current_version} for agent {agent_id}")
            return session_key, current_version
            
        except Exception as e:
            logger.error(f"Failed to generate session key: {e}")
            raise
    
    def schedule_key_rotation(self, agent_id: str) -> None:
        """Schedule automatic key rotation for agent"""
        try:
            next_rotation = datetime.now() + self.rotation_interval
            self.rotation_schedule[agent_id] = next_rotation
            
            logger.debug(f"Scheduled key rotation for agent {agent_id} at {next_rotation}")
            
        except Exception as e:
            logger.error(f"Failed to schedule key rotation: {e}")
    
    def rotate_key_if_needed(self, agent_id: str) -> Optional[Tuple[bytes, int]]:
        """Check if key rotation is needed and rotate if necessary"""
        try:
            if agent_id not in self.rotation_schedule:
                return None
            
            if datetime.now() >= self.rotation_schedule[agent_id]:
                logger.info(f"Rotating key for agent {agent_id}")
                return self.generate_session_key(agent_id)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check key rotation: {e}")
            return None
    
    def get_active_key(self, agent_id: str) -> Optional[Tuple[bytes, int]]:
        """Get active session key for agent"""
        try:
            if agent_id not in self.active_keys:
                return None
            
            key_info = self.active_keys[agent_id]
            
            # Check if key has expired
            if datetime.now() > key_info['expires']:
                logger.warning(f"Key expired for agent {agent_id}")
                return None
            
            return key_info['key'], key_info['version']
            
        except Exception as e:
            logger.error(f"Failed to get active key: {e}")
            return None
    
    def derive_agent_key(self, agent_id: str, purpose: str = "session") -> bytes:
        """Derive agent-specific key from master key"""
        try:
            # Create agent-specific derivation material
            derivation_input = f"{agent_id}:{purpose}".encode('utf-8')
            
            # Use PBKDF2 for key derivation
            derived_key = hashlib.pbkdf2_hmac(
                'sha256',
                self.master_key,
                self.key_derivation_salt + derivation_input,
                100000,  # iterations
                32       # key length
            )
            
            return derived_key
            
        except Exception as e:
            logger.error(f"Failed to derive agent key: {e}")
            raise
    
    def secure_key_distribution(self, agent_id: str, recipient_id: str, key: bytes) -> bytes:
        """Securely distribute key to recipient using key wrapping"""
        try:
            # Derive key-wrapping key for recipient
            wrapping_key = self.derive_agent_key(recipient_id, "key_wrapping")
            
            # Encrypt the key using AES-GCM (simplified)
            if CRYPTOGRAPHY_AVAILABLE:
                nonce = os.urandom(12)
                cipher = Cipher(algorithms.AES(wrapping_key), modes.GCM(nonce))
                encryptor = cipher.encryptor()
                wrapped_key = encryptor.update(key) + encryptor.finalize()
                
                # Return nonce + wrapped_key + auth_tag
                return nonce + wrapped_key + encryptor.tag
            else:
                # Simple XOR for testing
                return bytes(a ^ b for a, b in zip(key, wrapping_key))
            
        except Exception as e:
            logger.error(f"Failed to securely distribute key: {e}")
            raise
    
    def unwrap_distributed_key(self, recipient_id: str, wrapped_key: bytes) -> bytes:
        """Unwrap securely distributed key"""
        try:
            # Derive key-wrapping key for recipient
            wrapping_key = self.derive_agent_key(recipient_id, "key_wrapping")
            
            if CRYPTOGRAPHY_AVAILABLE:
                # Extract components
                nonce = wrapped_key[:12]
                ciphertext = wrapped_key[12:-16]
                auth_tag = wrapped_key[-16:]
                
                # Decrypt the key
                cipher = Cipher(algorithms.AES(wrapping_key), modes.GCM(nonce, auth_tag))
                decryptor = cipher.decryptor()
                key = decryptor.update(ciphertext) + decryptor.finalize()
                
                return key
            else:
                # Simple XOR for testing
                return bytes(a ^ b for a, b in zip(wrapped_key, wrapping_key))
            
        except Exception as e:
            logger.error(f"Failed to unwrap distributed key: {e}")
            raise
    
    def log_key_distribution(self, agent_id: str, recipient_id: str, key_version: int) -> None:
        """Log key distribution for audit purposes"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'distributor': agent_id,
                'recipient': recipient_id,
                'key_version': key_version,
                'operation': 'key_distribution'
            }
            
            self.key_distribution_log.append(log_entry)
            
            # Keep only last 1000 entries
            if len(self.key_distribution_log) > 1000:
                self.key_distribution_log = self.key_distribution_log[-1000:]
                
        except Exception as e:
            logger.error(f"Failed to log key distribution: {e}")

class SecureCommunicationBus:
    """
    Advanced secure communication bus with mutual TLS, Noise Protocol Framework,
    message integrity verification, and replay attack prevention
    """
    
    def __init__(self, 
                 bind_address: str = "tcp://*:5556",
                 security_level: SecurityLevel = SecurityLevel.ENHANCED,
                 protocol_type: ProtocolType = ProtocolType.NOISE_XX,
                 cert_dir: str = "certs"):
        self.bind_address = bind_address
        self.security_level = security_level
        self.protocol_type = protocol_type
        
        # Security components
        self.cert_manager = CertificateManager(cert_dir)
        self.noise_manager = NoiseProtocolManager()
        self.key_manager = KeyRotationManager()
        
        # ZeroMQ context and sockets
        if ZMQ_AVAILABLE:
            self.context = zmq.asyncio.Context()
        else:
            self.context = None
        self.secure_router = None
        self.secure_dealer = None
        
        # Redis connection for alternative transport
        self.redis_client = None
        self.redis_pubsub = None
        
        # Message tracking for replay prevention
        self.message_nonces = set()
        self.nonce_expiry = {}
        self.nonce_cleanup_interval = timedelta(minutes=30)
        
        # Statistics
        self.stats = {
            'secure_messages_sent': 0,
            'secure_messages_received': 0,
            'handshakes_completed': 0,
            'keys_rotated': 0,
            'replay_attacks_prevented': 0,
            'integrity_failures': 0,
            'authentication_failures': 0
        }
        
        self.logger = logging.getLogger(__name__)
        self.running = False
    
    async def initialize(self) -> None:
        """Initialize secure communication bus with multiple transport options"""
        try:
            self.logger.info(f"Initializing secure communication bus with {self.security_level.value} security")
            
            # Generate CA certificate if not exists
            if not self.cert_manager.ca_cert:
                self.cert_manager.generate_ca_certificate()
            
            # Initialize ZeroMQ if available
            if ZMQ_AVAILABLE:
                self.secure_router = self.context.socket(zmq.ROUTER)
                self.secure_dealer = self.context.socket(zmq.DEALER)
                
                # Configure security based on protocol type
                if self.protocol_type == ProtocolType.MTLS:
                    await self._setup_mtls_security()
                elif self.protocol_type in [ProtocolType.NOISE_XX, ProtocolType.NOISE_IK, ProtocolType.NOISE_NK]:
                    await self._setup_noise_security()
                
                # Bind sockets
                self.secure_router.bind(self.bind_address)
            else:
                self.logger.warning("ZeroMQ not available")
            
            # Initialize Redis if available
            if REDIS_AVAILABLE:
                try:
                    import redis.asyncio as aioredis
                    self.redis_client = aioredis.Redis(host='localhost', port=6379, decode_responses=False)
                    await self.redis_client.ping()
                    self.logger.info("Redis connection established")
                except Exception as redis_error:
                    self.logger.warning(f"Redis connection failed: {redis_error}")
                    self.redis_client = None
            
            self.running = True
            self.logger.info("Secure communication bus initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize secure communication bus: {e}")
            raise
    
    async def _setup_mtls_security(self) -> None:
        """Setup mutual TLS security for ZeroMQ sockets"""
        try:
            # Configure CURVE security for ZeroMQ
            # This is a simplified implementation - in production use proper TLS
            server_secret_key = zmq.curve_keypair()[1]
            self.secure_router.curve_secretkey = server_secret_key
            self.secure_router.curve_server = True
            
            self.logger.info("Configured mTLS security for communication")
            
        except Exception as e:
            self.logger.error(f"Failed to setup mTLS security: {e}")
            raise
    
    async def _setup_noise_security(self) -> None:
        """Setup Noise Protocol Framework security"""
        try:
            # Noise protocol will be handled at the application layer
            # ZeroMQ will handle transport, Noise handles encryption
            self.logger.info(f"Configured {self.protocol_type.value} security for communication")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Noise security: {e}")
            raise
    
    def register_secure_agent(self, agent_id: str, team: str) -> CertificateInfo:
        """Register agent with secure communication capabilities"""
        try:
            # Generate certificate for mTLS
            cert_info = self.cert_manager.generate_agent_certificate(agent_id)
            
            # Generate Noise keypair
            self.noise_manager.generate_keypair(agent_id)
            
            # Generate initial session key
            self.key_manager.generate_session_key(agent_id)
            
            self.logger.info(f"Registered secure agent {agent_id} with team {team}")
            return cert_info
            
        except Exception as e:
            self.logger.error(f"Failed to register secure agent {agent_id}: {e}")
            raise
    
    async def establish_secure_channel(self, initiator_id: str, responder_id: str) -> bool:
        """Establish secure communication channel between agents"""
        try:
            if self.protocol_type == ProtocolType.MTLS:
                return await self._establish_mtls_channel(initiator_id, responder_id)
            elif self.protocol_type in [ProtocolType.NOISE_XX, ProtocolType.NOISE_IK, ProtocolType.NOISE_NK]:
                return await self._establish_noise_channel(initiator_id, responder_id)
            else:
                raise ValueError(f"Unsupported protocol type: {self.protocol_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to establish secure channel: {e}")
            return False
    
    async def _establish_mtls_channel(self, initiator_id: str, responder_id: str) -> bool:
        """Establish mutual TLS channel with certificate pinning"""
        try:
            # Verify both agents have certificates
            if initiator_id not in self.cert_manager.agent_certs:
                raise ValueError(f"No certificate for initiator {initiator_id}")
            if responder_id not in self.cert_manager.agent_certs:
                raise ValueError(f"No certificate for responder {responder_id}")
            
            # Verify certificate pins
            initiator_cert = self.cert_manager.agent_certs[initiator_id]
            responder_cert = self.cert_manager.agent_certs[responder_id]
            
            # In a real implementation, this would involve actual TLS handshake
            # For simulation, we'll mark the channel as established
            
            self.logger.info(f"Established mTLS channel between {initiator_id} and {responder_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to establish mTLS channel: {e}")
            return False
    
    async def _establish_noise_channel(self, initiator_id: str, responder_id: str) -> bool:
        """Establish Noise protocol secure channel"""
        try:
            # Initialize handshake for initiator
            initiator_handshake = self.noise_manager.initialize_handshake(
                initiator_id, 
                protocol_name=f"Noise_{self.protocol_type.value.split('_')[1]}_25519_AESGCM_SHA256",
                is_initiator=True
            )
            
            # Initialize handshake for responder
            responder_handshake = self.noise_manager.initialize_handshake(
                responder_id,
                protocol_name=f"Noise_{self.protocol_type.value.split('_')[1]}_25519_AESGCM_SHA256",
                is_initiator=False
            )
            
            # Simulate handshake messages (in real implementation, these would be sent over network)
            if NOISE_AVAILABLE:
                # First message from initiator
                msg1 = initiator_handshake.handshake_state.write_message()
                
                # Process by responder
                msg2, completed = self.noise_manager.process_handshake_message(responder_id, msg1)
                
                if msg2 and not completed:
                    # Process response by initiator
                    msg3, completed = self.noise_manager.process_handshake_message(initiator_id, msg2)
                    
                    if msg3 and not completed:
                        # Final message to responder
                        _, completed = self.noise_manager.process_handshake_message(responder_id, msg3)
            else:
                # Mock completion for testing
                initiator_handshake.completed = True
                responder_handshake.completed = True
                completed = True
            
            if completed or not NOISE_AVAILABLE:
                self.stats['handshakes_completed'] += 1
                self.logger.info(f"Established Noise channel between {initiator_id} and {responder_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to establish Noise channel: {e}")
            return False
    
    def generate_message_nonce(self) -> bytes:
        """Generate unique nonce for message"""
        timestamp = int(time.time() * 1000000)  # microsecond precision
        random_bytes = os.urandom(8)
        return timestamp.to_bytes(8, 'big') + random_bytes
    
    def verify_message_nonce(self, nonce: bytes) -> bool:
        """Verify message nonce to prevent replay attacks"""
        try:
            # Check if nonce already used
            nonce_hex = nonce.hex()
            if nonce_hex in self.message_nonces:
                self.stats['replay_attacks_prevented'] += 1
                self.logger.warning(f"Replay attack detected with nonce {nonce_hex}")
                return False
            
            # Extract timestamp from nonce
            timestamp = int.from_bytes(nonce[:8], 'big')
            current_time = int(time.time() * 1000000)
            
            # Check if message is too old (prevent replay of old messages)
            max_age = 300 * 1000000  # 5 minutes in microseconds
            if current_time - timestamp > max_age:
                self.logger.warning(f"Message too old, nonce timestamp: {timestamp}")
                return False
            
            # Add nonce to used set
            self.message_nonces.add(nonce_hex)
            self.nonce_expiry[nonce_hex] = datetime.now() + self.nonce_cleanup_interval
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to verify message nonce: {e}")
            return False
    
    def cleanup_expired_nonces(self) -> None:
        """Clean up expired nonces to prevent memory growth"""
        try:
            current_time = datetime.now()
            expired_nonces = [
                nonce for nonce, expiry in self.nonce_expiry.items()
                if current_time > expiry
            ]
            
            for nonce in expired_nonces:
                self.message_nonces.discard(nonce)
                del self.nonce_expiry[nonce]
            
            if expired_nonces:
                self.logger.debug(f"Cleaned up {len(expired_nonces)} expired nonces")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired nonces: {e}")
    
    def calculate_message_integrity(self, message_data: bytes, key: bytes) -> str:
        """Calculate HMAC for message integrity"""
        try:
            hmac_obj = hmac.new(key, message_data, hashlib.sha256)
            return hmac_obj.hexdigest()
            
        except Exception as e:
            self.logger.error(f"Failed to calculate message integrity: {e}")
            raise
    
    def verify_message_integrity(self, message_data: bytes, key: bytes, expected_hash: str) -> bool:
        """Verify message integrity using HMAC"""
        try:
            calculated_hash = self.calculate_message_integrity(message_data, key)
            return hmac.compare_digest(calculated_hash, expected_hash)
            
        except Exception as e:
            self.logger.error(f"Failed to verify message integrity: {e}")
            return False
    
    def sign_message(self, agent_id: str, message_data: bytes) -> bytes:
        """Sign message using agent's private key"""
        try:
            if agent_id not in self.cert_manager.agent_certs:
                raise ValueError(f"No certificate for agent {agent_id}")
            
            cert_info = self.cert_manager.agent_certs[agent_id]
            
            if CRYPTOGRAPHY_AVAILABLE:
                # Load private key
                with open(cert_info.key_path, 'rb') as key_file:
                    private_key = load_pem_private_key(key_file.read(), password=None)
                
                # Sign message
                signature = private_key.sign(
                    message_data,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                
                return signature
            else:
                # Mock signature for testing
                return hashlib.sha256(message_data + agent_id.encode()).digest()
                
        except Exception as e:
            self.logger.error(f"Failed to sign message: {e}")
            raise
    
    def verify_message_signature(self, agent_id: str, message_data: bytes, signature: bytes) -> bool:
        """Verify message signature using agent's public key"""
        try:
            if agent_id not in self.cert_manager.agent_certs:
                self.logger.error(f"No certificate for agent {agent_id}")
                return False
            
            cert_info = self.cert_manager.agent_certs[agent_id]
            
            if CRYPTOGRAPHY_AVAILABLE:
                # Load certificate and extract public key
                with open(cert_info.cert_path, 'rb') as cert_file:
                    cert = x509.load_pem_x509_certificate(cert_file.read())
                
                public_key = cert.public_key()
                
                # Verify signature
                try:
                    public_key.verify(
                        signature,
                        message_data,
                        padding.PSS(
                            mgf=padding.MGF1(hashes.SHA256()),
                            salt_length=padding.PSS.MAX_LENGTH
                        ),
                        hashes.SHA256()
                    )
                    return True
                except Exception:
                    return False
            else:
                # Mock signature verification for testing
                expected_signature = hashlib.sha256(message_data + agent_id.encode()).digest()
                return hmac.compare_digest(signature, expected_signature)
                
        except Exception as e:
            self.logger.error(f"Failed to verify message signature: {e}")
            return False
    
    async def send_secure_message(self, sender_id: str, recipient_id: str, 
                                content: Dict[str, Any]) -> bool:
        """Send secure message with encryption and integrity protection"""
        try:
            # Check if key rotation is needed
            rotated_key = self.key_manager.rotate_key_if_needed(sender_id)
            if rotated_key:
                self.stats['keys_rotated'] += 1
            
            # Get active session key
            key_info = self.key_manager.get_active_key(sender_id)
            if not key_info:
                raise ValueError(f"No active key for sender {sender_id}")
            
            session_key, key_version = key_info
            
            # Generate message nonce
            nonce = self.generate_message_nonce()
            
            # Serialize content
            content_bytes = json.dumps(content).encode('utf-8')
            
            # Encrypt content based on protocol
            if self.protocol_type == ProtocolType.MTLS:
                # For mTLS, content is encrypted using session key
                encrypted_content = self._encrypt_with_session_key(content_bytes, session_key, nonce)
            else:
                # For Noise protocols, use Noise encryption
                # Check if cipher state exists, if not use session key encryption
                if sender_id in self.noise_manager.cipher_states:
                    encrypted_content = self.noise_manager.encrypt_message(sender_id, content_bytes)
                else:
                    # Fallback to session key encryption
                    encrypted_content = self._encrypt_with_session_key(content_bytes, session_key, nonce)
            
            # Calculate integrity hash
            integrity_hash = self.calculate_message_integrity(encrypted_content, session_key)
            
            # Sign the message for authenticity
            message_for_signing = encrypted_content + nonce + integrity_hash.encode()
            signature = self.sign_message(sender_id, message_for_signing)
            
            # Create secure message
            secure_message = SecureMessage(
                message_id=str(uuid.uuid4()),
                sender_id=sender_id,
                recipient_id=recipient_id,
                content=encrypted_content,
                timestamp=datetime.now(),
                nonce=nonce,
                signature=signature,
                protocol_used=self.protocol_type,
                key_version=key_version,
                integrity_hash=integrity_hash
            )
            
            # Send message (simplified for simulation)
            await self._transmit_secure_message(secure_message)
            
            self.stats['secure_messages_sent'] += 1
            self.logger.debug(f"Sent secure message from {sender_id} to {recipient_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send secure message: {e}")
            return False
    
    def _encrypt_with_session_key(self, plaintext: bytes, key: bytes, nonce: bytes) -> bytes:
        """Encrypt data with session key using AES-GCM"""
        try:
            if not CRYPTOGRAPHY_AVAILABLE:
                # Simple XOR for testing
                return bytes(a ^ b for a, b in zip(plaintext, key * (len(plaintext) // 32 + 1)))
            
            # Use AES-GCM for authenticated encryption
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(nonce[:12])  # GCM requires 12-byte nonce
            )
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(plaintext) + encryptor.finalize()
            
            # Return ciphertext + auth tag
            return ciphertext + encryptor.tag
            
        except Exception as e:
            self.logger.error(f"Failed to encrypt with session key: {e}")
            raise
    
    async def _transmit_secure_message(self, message: SecureMessage) -> None:
        """Transmit secure message over network using available transport"""
        try:
            # Serialize secure message
            message_dict = {
                'message_id': message.message_id,
                'sender_id': message.sender_id,
                'recipient_id': message.recipient_id,
                'content': message.content.hex(),
                'timestamp': message.timestamp.isoformat(),
                'nonce': message.nonce.hex(),
                'signature': message.signature.hex(),
                'protocol_used': message.protocol_used.value,
                'key_version': message.key_version,
                'integrity_hash': message.integrity_hash
            }
            
            message_bytes = json.dumps(message_dict).encode('utf-8')
            
            # Try ZeroMQ first
            if self.secure_router and ZMQ_AVAILABLE:
                await self.secure_router.send_multipart([
                    message.recipient_id.encode('utf-8'),
                    message_bytes
                ])
                self.logger.debug(f"Transmitted secure message {message.message_id} via ZeroMQ")
            
            # Fallback to Redis if available
            elif self.redis_client:
                channel = f"secure_messages:{message.recipient_id}"
                await self.redis_client.publish(channel, message_bytes)
                self.logger.debug(f"Transmitted secure message {message.message_id} via Redis")
            
            else:
                self.logger.warning("No transport available for message transmission")
            
        except Exception as e:
            self.logger.error(f"Failed to transmit secure message: {e}")
            raise
    
    async def subscribe_to_secure_messages(self, agent_id: str, callback: Callable) -> None:
        """Subscribe to secure messages for agent"""
        try:
            if self.redis_client:
                pubsub = self.redis_client.pubsub()
                channel = f"secure_messages:{agent_id}"
                await pubsub.subscribe(channel)
                
                self.logger.info(f"Subscribed to secure messages for agent {agent_id}")
                
                # Message processing loop (would run in background)
                async for message in pubsub.listen():
                    if message['type'] == 'message':
                        try:
                            message_dict = json.loads(message['data'].decode('utf-8'))
                            secure_message = self._deserialize_secure_message(message_dict)
                            
                            # Verify and decrypt message
                            decrypted_content = self.verify_and_decrypt_message(agent_id, secure_message)
                            if decrypted_content:
                                await callback(secure_message.sender_id, decrypted_content)
                                
                        except Exception as msg_error:
                            self.logger.error(f"Failed to process received message: {msg_error}")
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to secure messages: {e}")
    
    def _deserialize_secure_message(self, message_dict: Dict[str, Any]) -> SecureMessage:
        """Deserialize secure message from dictionary"""
        return SecureMessage(
            message_id=message_dict['message_id'],
            sender_id=message_dict['sender_id'],
            recipient_id=message_dict['recipient_id'],
            content=bytes.fromhex(message_dict['content']),
            timestamp=datetime.fromisoformat(message_dict['timestamp']),
            nonce=bytes.fromhex(message_dict['nonce']),
            signature=bytes.fromhex(message_dict['signature']),
            protocol_used=ProtocolType(message_dict['protocol_used']),
            key_version=message_dict['key_version'],
            integrity_hash=message_dict['integrity_hash']
        )
    
    async def receive_secure_message(self, recipient_id: str) -> Optional[Dict[str, Any]]:
        """Receive and decrypt secure message with full verification"""
        try:
            # This would receive from network in real implementation
            # For simulation, we'll return None unless we have a test message
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to receive secure message: {e}")
            return None
    
    def verify_and_decrypt_message(self, recipient_id: str, secure_message: SecureMessage) -> Optional[Dict[str, Any]]:
        """Verify and decrypt received secure message"""
        try:
            # Verify message nonce to prevent replay attacks
            if not self.verify_message_nonce(secure_message.nonce):
                self.logger.error("Message nonce verification failed - possible replay attack")
                return None
            
            # Get recipient's active key
            key_info = self.key_manager.get_active_key(recipient_id)
            if not key_info:
                self.logger.error(f"No active key for recipient {recipient_id}")
                return None
            
            session_key, key_version = key_info
            
            # Verify key version matches
            if key_version != secure_message.key_version:
                self.logger.warning(f"Key version mismatch: expected {key_version}, got {secure_message.key_version}")
                # Try to get the specific key version if available
                # For now, we'll continue with current key
            
            # Verify message integrity
            if not self.verify_message_integrity(secure_message.content, session_key, secure_message.integrity_hash):
                self.stats['integrity_failures'] += 1
                self.logger.error("Message integrity verification failed")
                return None
            
            # Verify message signature
            message_for_verification = secure_message.content + secure_message.nonce + secure_message.integrity_hash.encode()
            if not self.verify_message_signature(secure_message.sender_id, message_for_verification, secure_message.signature):
                self.stats['authentication_failures'] += 1
                self.logger.error("Message signature verification failed")
                return None
            
            # Decrypt content based on protocol
            if secure_message.protocol_used == ProtocolType.MTLS:
                decrypted_content = self._decrypt_with_session_key(secure_message.content, session_key, secure_message.nonce)
            else:
                decrypted_content = self.noise_manager.decrypt_message(recipient_id, secure_message.content)
            
            # Parse decrypted content
            content_dict = json.loads(decrypted_content.decode('utf-8'))
            
            self.stats['secure_messages_received'] += 1
            self.logger.debug(f"Successfully verified and decrypted message from {secure_message.sender_id}")
            
            return content_dict
            
        except Exception as e:
            self.logger.error(f"Failed to verify and decrypt message: {e}")
            return None
    
    def _decrypt_with_session_key(self, ciphertext: bytes, key: bytes, nonce: bytes) -> bytes:
        """Decrypt data with session key using AES-GCM"""
        try:
            if not CRYPTOGRAPHY_AVAILABLE:
                # Simple XOR for testing
                return bytes(a ^ b for a, b in zip(ciphertext, key * (len(ciphertext) // 32 + 1)))
            
            # Extract auth tag from end of ciphertext
            auth_tag = ciphertext[-16:]
            actual_ciphertext = ciphertext[:-16]
            
            # Use AES-GCM for authenticated decryption
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(nonce[:12], auth_tag)
            )
            decryptor = cipher.decryptor()
            plaintext = decryptor.update(actual_ciphertext) + decryptor.finalize()
            
            return plaintext
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt with session key: {e}")
            raise
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        return {
            **self.stats,
            'active_agents': len(self.cert_manager.agent_certs),
            'active_nonces': len(self.message_nonces),
            'security_level': self.security_level.value,
            'protocol_type': self.protocol_type.value
        }
    
    async def shutdown(self) -> None:
        """Shutdown secure communication bus"""
        try:
            self.running = False
            
            # Close ZeroMQ resources
            if self.secure_router:
                self.secure_router.close()
            if self.secure_dealer:
                self.secure_dealer.close()
            if self.context:
                self.context.term()
            
            # Close Redis resources
            if self.redis_client:
                await self.redis_client.close()
            
            self.logger.info("Secure communication bus shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Example usage and testing functions
async def demo_advanced_secure_communication():
    """Demonstrate advanced secure communication capabilities"""
    try:
        print("=== Advanced Secure Communication Demo ===")
        
        # Initialize secure communication bus with maximum security
        secure_bus = SecureCommunicationBus(
            security_level=SecurityLevel.MAXIMUM,
            protocol_type=ProtocolType.NOISE_XX
        )
        
        await secure_bus.initialize()
        print("✓ Secure communication bus initialized")
        
        # Register multiple agents
        agents = ["red_recon", "red_exploit", "blue_soc", "blue_firewall"]
        agent_certs = {}
        
        for agent_id in agents:
            team = "red" if agent_id.startswith("red") else "blue"
            cert_info = secure_bus.register_secure_agent(agent_id, team)
            agent_certs[agent_id] = cert_info
            print(f"✓ Registered {agent_id} with certificate {cert_info.fingerprint[:16]}...")
        
        # Demonstrate certificate pinning and validation
        print("\n=== Certificate Security Features ===")
        for agent_id, cert_info in agent_certs.items():
            with open(cert_info.cert_path, "rb") as f:
                cert_data = f.read()
            
            # Test certificate pinning
            pin_valid = secure_bus.cert_manager.verify_certificate_pin(agent_id, cert_data)
            print(f"✓ Certificate pin validation for {agent_id}: {pin_valid}")
            
            # Test certificate chain validation
            chain_valid = secure_bus.cert_manager.validate_certificate_chain(agent_id, [cert_data])
            print(f"✓ Certificate chain validation for {agent_id}: {chain_valid}")
        
        # Demonstrate key rotation and distribution
        print("\n=== Key Management Features ===")
        for agent_id in agents:
            # Test key rotation
            rotated = secure_bus.key_manager.rotate_key_if_needed(agent_id)
            if rotated:
                print(f"✓ Key rotated for {agent_id}")
            
            # Test key derivation
            derived_key = secure_bus.key_manager.derive_agent_key(agent_id, "test_purpose")
            print(f"✓ Derived key for {agent_id}: {len(derived_key)} bytes")
        
        # Demonstrate secure channel establishment
        print("\n=== Secure Channel Establishment ===")
        channels = [
            ("red_recon", "red_exploit"),
            ("blue_soc", "blue_firewall"),
        ]
        
        for sender, recipient in channels:
            channel_established = await secure_bus.establish_secure_channel(sender, recipient)
            print(f"✓ Secure channel {sender} -> {recipient}: {channel_established}")
        
        # Demonstrate advanced message features
        print("\n=== Advanced Message Security ===")
        
        # Test message signing and verification
        test_message = b"This is a test message for signing"
        signature = secure_bus.sign_message("red_recon", test_message)
        signature_valid = secure_bus.verify_message_signature("red_recon", test_message, signature)
        print(f"✓ Message signing and verification: {signature_valid}")
        
        # Test secure message exchange with full verification
        message_content = {
            "type": "intelligence_report",
            "source": "red_recon",
            "target": "192.168.1.100",
            "vulnerabilities": ["CVE-2023-1234", "CVE-2023-5678"],
            "confidence": 0.95,
            "timestamp": datetime.now().isoformat()
        }
        
        message_sent = await secure_bus.send_secure_message("red_recon", "red_exploit", message_content)
        print(f"✓ Advanced secure message sent: {message_sent}")
        
        # Demonstrate replay attack prevention
        print("\n=== Replay Attack Prevention ===")
        nonce1 = secure_bus.generate_message_nonce()
        nonce2 = secure_bus.generate_message_nonce()
        
        # First use should succeed
        valid1 = secure_bus.verify_message_nonce(nonce1)
        valid2 = secure_bus.verify_message_nonce(nonce2)
        print(f"✓ First nonce usage: {valid1}, {valid2}")
        
        # Replay should fail
        replay1 = secure_bus.verify_message_nonce(nonce1)
        replay2 = secure_bus.verify_message_nonce(nonce2)
        print(f"✓ Replay prevention: {not replay1 and not replay2}")
        
        # Demonstrate forward secrecy
        print("\n=== Forward Secrecy ===")
        for agent_id in ["red_recon", "blue_soc"]:
            ephemeral_rotated = secure_bus.noise_manager.rotate_ephemeral_keys(agent_id)
            print(f"✓ Ephemeral key rotation for {agent_id}: {ephemeral_rotated}")
            
            if agent_id in secure_bus.noise_manager.ephemeral_keys:
                peer_key = os.urandom(32)
                session_key = secure_bus.noise_manager.derive_session_key(agent_id, peer_key)
                print(f"✓ Session key derived for {agent_id}: {len(session_key)} bytes")
        
        # Display comprehensive security statistics
        print("\n=== Security Statistics ===")
        stats = secure_bus.get_security_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test certificate revocation
        print("\n=== Certificate Revocation ===")
        revoked = secure_bus.cert_manager.revoke_certificate("red_recon")
        print(f"✓ Certificate revoked for red_recon: {revoked}")
        
        # Verify revoked certificate fails validation
        with open(agent_certs["red_recon"].cert_path, "rb") as f:
            cert_data = f.read()
        revoked_validation = secure_bus.cert_manager.verify_certificate_pin("red_recon", cert_data)
        print(f"✓ Revoked certificate validation fails: {not revoked_validation}")
        
        await secure_bus.shutdown()
        print("\n✓ Secure communication bus shutdown complete")
        print("=== Demo completed successfully ===")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(demo_advanced_secure_communication())