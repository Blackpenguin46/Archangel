"""
Enterprise-grade authentication and directory services module.

This module provides comprehensive authentication infrastructure including:
- Active Directory simulation with realistic domain structure
- DNS, DHCP, and Certificate Authority services
- OAuth2 and mTLS authentication for agent communication
- Role-Based Access Control (RBAC) policies
"""

__version__ = "1.0.0"
__author__ = "Archangel Team"

from .active_directory import ActiveDirectorySimulator
from .dns_service import DNSService
from .dhcp_service import DHCPService
from .certificate_authority import CertificateAuthority

# Optional imports that may not be available
__all__ = [
    "ActiveDirectorySimulator",
    "DNSService", 
    "DHCPService",
    "CertificateAuthority"
]

try:
    from .oauth2_provider import OAuth2Provider
    __all__.append("OAuth2Provider")
except ImportError:
    pass

try:
    from .mtls_authenticator import MTLSAuthenticator
    __all__.append("MTLSAuthenticator")
except ImportError:
    pass

try:
    from .rbac_manager import RBACManager
    __all__.append("RBACManager")
except ImportError:
    pass