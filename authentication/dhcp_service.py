"""
DHCP Service simulation for enterprise network realism.

This module provides a realistic DHCP service implementation with:
- IP address allocation and lease management
- DHCP options and vendor-specific information
- Reservation management
- DHCP relay and failover simulation
"""

import json
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import ipaddress
import secrets


class DHCPMessageType(Enum):
    """DHCP message types."""
    DISCOVER = 1
    OFFER = 2
    REQUEST = 3
    DECLINE = 4
    ACK = 5
    NAK = 6
    RELEASE = 7
    INFORM = 8


class LeaseState(Enum):
    """DHCP lease states."""
    AVAILABLE = "available"
    OFFERED = "offered"
    BOUND = "bound"
    EXPIRED = "expired"
    RESERVED = "reserved"


@dataclass
class DHCPLease:
    """DHCP lease information."""
    ip_address: str
    mac_address: str
    hostname: Optional[str] = None
    client_id: Optional[str] = None
    lease_start: datetime = field(default_factory=datetime.now)
    lease_end: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=24))
    state: LeaseState = LeaseState.BOUND
    vendor_class: Optional[str] = None
    user_class: Optional[str] = None
    renewal_time: Optional[datetime] = None
    rebinding_time: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if lease has expired."""
        return datetime.now() > self.lease_end
    
    def time_remaining(self) -> timedelta:
        """Get remaining lease time."""
        if self.is_expired():
            return timedelta(0)
        return self.lease_end - datetime.now()
    
    def renew_lease(self, duration: timedelta) -> None:
        """Renew the lease for specified duration."""
        self.lease_end = datetime.now() + duration
        self.renewal_time = datetime.now() + (duration * 0.5)  # T1 = 50% of lease time
        self.rebinding_time = datetime.now() + (duration * 0.875)  # T2 = 87.5% of lease time


@dataclass
class DHCPReservation:
    """DHCP reservation for specific MAC addresses."""
    mac_address: str
    ip_address: str
    hostname: Optional[str] = None
    description: Optional[str] = None
    options: Dict[int, str] = field(default_factory=dict)


@dataclass
class DHCPScope:
    """DHCP scope configuration."""
    name: str
    network: str  # Network in CIDR notation
    start_ip: str
    end_ip: str
    subnet_mask: str
    default_gateway: str
    dns_servers: List[str] = field(default_factory=list)
    domain_name: Optional[str] = None
    lease_duration: timedelta = field(default_factory=lambda: timedelta(hours=24))
    reservations: Dict[str, DHCPReservation] = field(default_factory=dict)
    options: Dict[int, str] = field(default_factory=dict)
    enabled: bool = True
    
    def __post_init__(self):
        # Validate IP range
        start = ipaddress.IPv4Address(self.start_ip)
        end = ipaddress.IPv4Address(self.end_ip)
        if start >= end:
            raise ValueError("Start IP must be less than end IP")
        
        # Set default options
        if not self.options:
            self.options = {
                1: self.subnet_mask,  # Subnet mask
                3: self.default_gateway,  # Router
                6: ','.join(self.dns_servers) if self.dns_servers else "",  # DNS servers
                15: self.domain_name or "",  # Domain name
                51: str(int(self.lease_duration.total_seconds())),  # Lease time
            }
    
    def is_ip_in_range(self, ip_address: str) -> bool:
        """Check if IP address is within the scope range."""
        ip = ipaddress.IPv4Address(ip_address)
        start = ipaddress.IPv4Address(self.start_ip)
        end = ipaddress.IPv4Address(self.end_ip)
        return start <= ip <= end
    
    def get_available_ips(self, exclude: Set[str] = None) -> List[str]:
        """Get list of available IP addresses in the scope."""
        if exclude is None:
            exclude = set()
        
        available_ips = []
        start = ipaddress.IPv4Address(self.start_ip)
        end = ipaddress.IPv4Address(self.end_ip)
        
        current = start
        while current <= end:
            ip_str = str(current)
            if ip_str not in exclude and ip_str not in self.reservations:
                available_ips.append(ip_str)
            current += 1
        
        return available_ips


@dataclass
class DHCPTransaction:
    """DHCP transaction log entry."""
    transaction_id: str
    client_mac: str
    client_ip: Optional[str] = None
    message_type: DHCPMessageType = DHCPMessageType.DISCOVER
    timestamp: datetime = field(default_factory=datetime.now)
    server_response: Optional[str] = None
    lease_duration: Optional[int] = None
    options_sent: Dict[int, str] = field(default_factory=dict)


class DHCPService:
    """
    DHCP Service simulation for enterprise environments.
    
    Provides realistic DHCP functionality including:
    - IP address allocation and lease management
    - DHCP options and vendor-specific information
    - Reservation management
    - Transaction logging and statistics
    """
    
    def __init__(self, server_name: str = "dhcp01.archangel.local", 
                 listen_port: int = 67):
        self.server_name = server_name
        self.listen_port = listen_port
        self.scopes: Dict[str, DHCPScope] = {}
        self.leases: Dict[str, DHCPLease] = {}  # IP -> Lease mapping
        self.mac_to_ip: Dict[str, str] = {}  # MAC -> IP mapping
        self.transactions: List[DHCPTransaction] = []
        self.running = False
        self.logger = logging.getLogger(__name__)
        
        # Initialize default scopes
        self._initialize_default_scopes()
        
        # Start lease cleanup thread
        self._start_lease_cleanup()
    
    def _initialize_default_scopes(self) -> None:
        """Initialize default DHCP scopes for enterprise networks."""
        
        # Management network scope
        mgmt_scope = DHCPScope(
            name="Management Network",
            network="192.168.1.0/24",
            start_ip="192.168.1.100",
            end_ip="192.168.1.200",
            subnet_mask="255.255.255.0",
            default_gateway="192.168.1.1",
            dns_servers=["192.168.1.10", "192.168.1.11"],
            domain_name="archangel.local",
            lease_duration=timedelta(hours=24)
        )
        
        # Add reservations for infrastructure servers
        mgmt_reservations = [
            ("00:50:56:12:34:10", "192.168.1.10", "dns01"),
            ("00:50:56:12:34:11", "192.168.1.11", "dns02"),
            ("00:50:56:12:34:20", "192.168.1.20", "dc01"),
            ("00:50:56:12:34:21", "192.168.1.21", "dc02"),
        ]
        
        for mac, ip, hostname in mgmt_reservations:
            reservation = DHCPReservation(
                mac_address=mac,
                ip_address=ip,
                hostname=hostname,
                description=f"Reserved for {hostname}"
            )
            mgmt_scope.reservations[mac] = reservation
        
        self.scopes["mgmt"] = mgmt_scope
        
        # DMZ network scope
        dmz_scope = DHCPScope(
            name="DMZ Network",
            network="192.168.10.0/24",
            start_ip="192.168.10.50",
            end_ip="192.168.10.100",
            subnet_mask="255.255.255.0",
            default_gateway="192.168.10.1",
            dns_servers=["192.168.1.10", "192.168.1.11"],
            domain_name="archangel.local",
            lease_duration=timedelta(hours=12)
        )
        
        # DMZ reservations
        dmz_reservations = [
            ("00:50:56:12:35:10", "192.168.10.10", "web01"),
            ("00:50:56:12:35:11", "192.168.10.11", "web02"),
            ("00:50:56:12:35:20", "192.168.10.20", "mail01"),
        ]
        
        for mac, ip, hostname in dmz_reservations:
            reservation = DHCPReservation(
                mac_address=mac,
                ip_address=ip,
                hostname=hostname,
                description=f"Reserved for {hostname}"
            )
            dmz_scope.reservations[mac] = reservation
        
        self.scopes["dmz"] = dmz_scope
        
        # Internal LAN scope
        lan_scope = DHCPScope(
            name="Internal LAN",
            network="192.168.20.0/24",
            start_ip="192.168.20.100",
            end_ip="192.168.20.200",
            subnet_mask="255.255.255.0",
            default_gateway="192.168.20.1",
            dns_servers=["192.168.1.10", "192.168.1.11"],
            domain_name="archangel.local",
            lease_duration=timedelta(hours=8)
        )
        
        # LAN reservations for servers
        lan_reservations = [
            ("00:50:56:12:36:10", "192.168.20.10", "db01"),
            ("00:50:56:12:36:11", "192.168.20.11", "db02"),
            ("00:50:56:12:36:20", "192.168.20.20", "file01"),
        ]
        
        for mac, ip, hostname in lan_reservations:
            reservation = DHCPReservation(
                mac_address=mac,
                ip_address=ip,
                hostname=hostname,
                description=f"Reserved for {hostname}"
            )
            lan_scope.reservations[mac] = reservation
        
        self.scopes["lan"] = lan_scope
    
    def _start_lease_cleanup(self) -> None:
        """Start background thread to clean up expired leases."""
        
        def cleanup_expired_leases():
            while True:
                try:
                    current_time = datetime.now()
                    expired_ips = []
                    
                    for ip, lease in self.leases.items():
                        if lease.is_expired() and lease.state == LeaseState.BOUND:
                            lease.state = LeaseState.EXPIRED
                            expired_ips.append(ip)
                    
                    # Remove expired leases
                    for ip in expired_ips:
                        lease = self.leases[ip]
                        if lease.mac_address in self.mac_to_ip:
                            del self.mac_to_ip[lease.mac_address]
                        del self.leases[ip]
                        self.logger.info(f"Expired lease for {ip} (MAC: {lease.mac_address})")
                    
                    time.sleep(300)  # Check every 5 minutes
                except Exception as e:
                    self.logger.error(f"Error in lease cleanup: {e}")
                    time.sleep(60)
        
        cleanup_thread = threading.Thread(target=cleanup_expired_leases, daemon=True)
        cleanup_thread.start()
    
    def _find_scope_for_client(self, client_ip: Optional[str] = None, 
                              relay_agent_ip: Optional[str] = None) -> Optional[DHCPScope]:
        """Find appropriate DHCP scope for client."""
        
        # If relay agent IP is provided, use it to determine scope
        if relay_agent_ip:
            for scope in self.scopes.values():
                network = ipaddress.IPv4Network(scope.network)
                if ipaddress.IPv4Address(relay_agent_ip) in network:
                    return scope
        
        # If client IP is provided, find matching scope
        if client_ip and client_ip != "0.0.0.0":
            for scope in self.scopes.values():
                if scope.is_ip_in_range(client_ip):
                    return scope
        
        # Default to first enabled scope
        for scope in self.scopes.values():
            if scope.enabled:
                return scope
        
        return None
    
    def _get_next_available_ip(self, scope: DHCPScope, 
                              exclude_ips: Set[str] = None) -> Optional[str]:
        """Get next available IP address from scope."""
        
        if exclude_ips is None:
            exclude_ips = set()
        
        # Add currently leased IPs to exclusion list
        leased_ips = {ip for ip, lease in self.leases.items() 
                     if lease.state in [LeaseState.BOUND, LeaseState.OFFERED]}
        exclude_ips.update(leased_ips)
        
        # Add reserved IPs
        reserved_ips = {res.ip_address for res in scope.reservations.values()}
        exclude_ips.update(reserved_ips)
        
        available_ips = scope.get_available_ips(exclude_ips)
        
        return available_ips[0] if available_ips else None
    
    def handle_dhcp_discover(self, client_mac: str, hostname: Optional[str] = None,
                           client_ip: Optional[str] = None,
                           relay_agent_ip: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Handle DHCP DISCOVER message.
        
        Args:
            client_mac: Client MAC address
            hostname: Requested hostname
            client_ip: Client's current IP (if any)
            relay_agent_ip: DHCP relay agent IP
            
        Returns:
            DHCP OFFER response or None if no IP available
        """
        
        transaction_id = secrets.token_hex(4)
        
        # Log transaction
        transaction = DHCPTransaction(
            transaction_id=transaction_id,
            client_mac=client_mac,
            client_ip=client_ip,
            message_type=DHCPMessageType.DISCOVER
        )
        self.transactions.append(transaction)
        
        # Find appropriate scope
        scope = self._find_scope_for_client(client_ip, relay_agent_ip)
        if not scope or not scope.enabled:
            self.logger.warning(f"No suitable scope found for client {client_mac}")
            return None
        
        # Check for existing reservation
        offered_ip = None
        if client_mac in scope.reservations:
            reservation = scope.reservations[client_mac]
            offered_ip = reservation.ip_address
            self.logger.info(f"Using reservation for {client_mac}: {offered_ip}")
        else:
            # Check for existing lease
            if client_mac in self.mac_to_ip:
                existing_ip = self.mac_to_ip[client_mac]
                if existing_ip in self.leases:
                    lease = self.leases[existing_ip]
                    if not lease.is_expired():
                        offered_ip = existing_ip
                        self.logger.info(f"Renewing existing lease for {client_mac}: {offered_ip}")
            
            # Get new IP if no existing lease
            if not offered_ip:
                offered_ip = self._get_next_available_ip(scope)
                if not offered_ip:
                    self.logger.warning(f"No available IP addresses in scope {scope.name}")
                    return None
        
        # Create or update lease
        lease = DHCPLease(
            ip_address=offered_ip,
            mac_address=client_mac,
            hostname=hostname,
            lease_start=datetime.now(),
            lease_end=datetime.now() + scope.lease_duration,
            state=LeaseState.OFFERED
        )
        
        self.leases[offered_ip] = lease
        self.mac_to_ip[client_mac] = offered_ip
        
        # Prepare DHCP options
        options = scope.options.copy()
        if client_mac in scope.reservations:
            options.update(scope.reservations[client_mac].options)
        
        # Update transaction
        transaction.client_ip = offered_ip
        transaction.server_response = "OFFER"
        transaction.lease_duration = int(scope.lease_duration.total_seconds())
        transaction.options_sent = options
        
        offer_response = {
            "message_type": "OFFER",
            "transaction_id": transaction_id,
            "offered_ip": offered_ip,
            "server_ip": self._get_server_ip(scope),
            "lease_time": int(scope.lease_duration.total_seconds()),
            "renewal_time": int(scope.lease_duration.total_seconds() * 0.5),
            "rebinding_time": int(scope.lease_duration.total_seconds() * 0.875),
            "subnet_mask": scope.subnet_mask,
            "router": scope.default_gateway,
            "dns_servers": scope.dns_servers,
            "domain_name": scope.domain_name,
            "options": options
        }
        
        self.logger.info(f"DHCP OFFER: {offered_ip} to {client_mac} (scope: {scope.name})")
        return offer_response
    
    def handle_dhcp_request(self, client_mac: str, requested_ip: str,
                           server_ip: Optional[str] = None,
                           hostname: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Handle DHCP REQUEST message.
        
        Args:
            client_mac: Client MAC address
            requested_ip: IP address requested by client
            server_ip: DHCP server IP from client request
            hostname: Client hostname
            
        Returns:
            DHCP ACK or NAK response
        """
        
        transaction_id = secrets.token_hex(4)
        
        # Log transaction
        transaction = DHCPTransaction(
            transaction_id=transaction_id,
            client_mac=client_mac,
            client_ip=requested_ip,
            message_type=DHCPMessageType.REQUEST
        )
        self.transactions.append(transaction)
        
        # Check if we have an offered lease for this IP and MAC
        if requested_ip not in self.leases:
            self.logger.warning(f"DHCP REQUEST for unknown IP {requested_ip} from {client_mac}")
            transaction.server_response = "NAK"
            return {
                "message_type": "NAK",
                "transaction_id": transaction_id,
                "reason": "Unknown IP address"
            }
        
        lease = self.leases[requested_ip]
        
        # Verify MAC address matches
        if lease.mac_address != client_mac:
            self.logger.warning(f"DHCP REQUEST MAC mismatch for {requested_ip}: expected {lease.mac_address}, got {client_mac}")
            transaction.server_response = "NAK"
            return {
                "message_type": "NAK",
                "transaction_id": transaction_id,
                "reason": "MAC address mismatch"
            }
        
        # Find scope for this IP
        scope = None
        for s in self.scopes.values():
            if s.is_ip_in_range(requested_ip):
                scope = s
                break
        
        if not scope:
            self.logger.error(f"No scope found for IP {requested_ip}")
            transaction.server_response = "NAK"
            return {
                "message_type": "NAK",
                "transaction_id": transaction_id,
                "reason": "IP not in any scope"
            }
        
        # Update lease to bound state
        lease.state = LeaseState.BOUND
        lease.hostname = hostname
        lease.lease_start = datetime.now()
        lease.lease_end = datetime.now() + scope.lease_duration
        lease.renew_lease(scope.lease_duration)
        
        # Prepare options
        options = scope.options.copy()
        if client_mac in scope.reservations:
            options.update(scope.reservations[client_mac].options)
        
        # Update transaction
        transaction.server_response = "ACK"
        transaction.lease_duration = int(scope.lease_duration.total_seconds())
        transaction.options_sent = options
        
        ack_response = {
            "message_type": "ACK",
            "transaction_id": transaction_id,
            "client_ip": requested_ip,
            "server_ip": self._get_server_ip(scope),
            "lease_time": int(scope.lease_duration.total_seconds()),
            "renewal_time": int(scope.lease_duration.total_seconds() * 0.5),
            "rebinding_time": int(scope.lease_duration.total_seconds() * 0.875),
            "subnet_mask": scope.subnet_mask,
            "router": scope.default_gateway,
            "dns_servers": scope.dns_servers,
            "domain_name": scope.domain_name,
            "options": options
        }
        
        self.logger.info(f"DHCP ACK: {requested_ip} to {client_mac} (lease expires: {lease.lease_end})")
        return ack_response
    
    def handle_dhcp_release(self, client_mac: str, client_ip: str) -> bool:
        """
        Handle DHCP RELEASE message.
        
        Args:
            client_mac: Client MAC address
            client_ip: IP address being released
            
        Returns:
            True if release was successful
        """
        
        # Log transaction
        transaction = DHCPTransaction(
            transaction_id=secrets.token_hex(4),
            client_mac=client_mac,
            client_ip=client_ip,
            message_type=DHCPMessageType.RELEASE
        )
        self.transactions.append(transaction)
        
        if client_ip not in self.leases:
            self.logger.warning(f"DHCP RELEASE for unknown IP {client_ip}")
            return False
        
        lease = self.leases[client_ip]
        
        if lease.mac_address != client_mac:
            self.logger.warning(f"DHCP RELEASE MAC mismatch for {client_ip}")
            return False
        
        # Remove lease
        del self.leases[client_ip]
        if client_mac in self.mac_to_ip:
            del self.mac_to_ip[client_mac]
        
        transaction.server_response = "RELEASED"
        
        self.logger.info(f"DHCP RELEASE: {client_ip} from {client_mac}")
        return True
    
    def _get_server_ip(self, scope: DHCPScope) -> str:
        """Get server IP for the given scope."""
        # Return the gateway IP as server IP for simplicity
        return scope.default_gateway
    
    def add_reservation(self, scope_name: str, mac_address: str, ip_address: str,
                       hostname: Optional[str] = None, 
                       description: Optional[str] = None) -> bool:
        """Add a DHCP reservation."""
        
        if scope_name not in self.scopes:
            return False
        
        scope = self.scopes[scope_name]
        
        if not scope.is_ip_in_range(ip_address):
            return False
        
        reservation = DHCPReservation(
            mac_address=mac_address,
            ip_address=ip_address,
            hostname=hostname,
            description=description
        )
        
        scope.reservations[mac_address] = reservation
        
        self.logger.info(f"Added DHCP reservation: {mac_address} -> {ip_address} in scope {scope_name}")
        return True
    
    def remove_reservation(self, scope_name: str, mac_address: str) -> bool:
        """Remove a DHCP reservation."""
        
        if scope_name not in self.scopes:
            return False
        
        scope = self.scopes[scope_name]
        
        if mac_address in scope.reservations:
            del scope.reservations[mac_address]
            self.logger.info(f"Removed DHCP reservation for {mac_address} in scope {scope_name}")
            return True
        
        return False
    
    def get_lease_info(self, ip_address: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific lease."""
        
        if ip_address not in self.leases:
            return None
        
        lease = self.leases[ip_address]
        
        return {
            "ip_address": lease.ip_address,
            "mac_address": lease.mac_address,
            "hostname": lease.hostname,
            "client_id": lease.client_id,
            "lease_start": lease.lease_start.isoformat(),
            "lease_end": lease.lease_end.isoformat(),
            "state": lease.state.value,
            "time_remaining": str(lease.time_remaining()),
            "vendor_class": lease.vendor_class,
            "user_class": lease.user_class
        }
    
    def get_scope_statistics(self, scope_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a DHCP scope."""
        
        if scope_name not in self.scopes:
            return None
        
        scope = self.scopes[scope_name]
        
        # Count leases in this scope
        scope_leases = [lease for lease in self.leases.values() 
                       if scope.is_ip_in_range(lease.ip_address)]
        
        active_leases = [lease for lease in scope_leases 
                        if lease.state == LeaseState.BOUND and not lease.is_expired()]
        
        # Calculate utilization
        total_ips = len(scope.get_available_ips())
        reserved_ips = len(scope.reservations)
        available_ips = total_ips - len(active_leases) - reserved_ips
        utilization = (len(active_leases) / total_ips * 100) if total_ips > 0 else 0
        
        return {
            "scope_name": scope.name,
            "network": scope.network,
            "total_addresses": total_ips,
            "active_leases": len(active_leases),
            "reservations": reserved_ips,
            "available_addresses": available_ips,
            "utilization_percent": round(utilization, 2),
            "lease_duration_hours": scope.lease_duration.total_seconds() / 3600,
            "enabled": scope.enabled
        }
    
    def get_server_statistics(self) -> Dict[str, Any]:
        """Get overall DHCP server statistics."""
        
        total_leases = len(self.leases)
        active_leases = len([lease for lease in self.leases.values() 
                           if lease.state == LeaseState.BOUND and not lease.is_expired()])
        
        # Recent transactions (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_transactions = [t for t in self.transactions if t.timestamp >= cutoff_time]
        
        return {
            "server_name": self.server_name,
            "total_scopes": len(self.scopes),
            "total_leases": total_leases,
            "active_leases": active_leases,
            "total_transactions": len(self.transactions),
            "recent_transactions_24h": len(recent_transactions),
            "running": self.running
        }