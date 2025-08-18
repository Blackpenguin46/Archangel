"""
DNS Service simulation for enterprise network realism.

This module provides a realistic DNS service implementation with:
- Forward and reverse DNS resolution
- DNS zone management
- Dynamic DNS updates
- DNS security features (DNSSEC simulation)
"""

import json
import socket
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import ipaddress


class DNSRecordType(Enum):
    """DNS record types."""
    A = "A"
    AAAA = "AAAA"
    CNAME = "CNAME"
    MX = "MX"
    NS = "NS"
    PTR = "PTR"
    SOA = "SOA"
    SRV = "SRV"
    TXT = "TXT"


@dataclass
class DNSRecord:
    """DNS resource record."""
    name: str
    record_type: DNSRecordType
    value: str
    ttl: int = 3600
    priority: Optional[int] = None
    weight: Optional[int] = None
    port: Optional[int] = None
    created: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class DNSZone:
    """DNS zone configuration."""
    name: str
    zone_type: str  # "primary", "secondary", "stub"
    records: Dict[str, List[DNSRecord]] = field(default_factory=dict)
    serial: int = field(default_factory=lambda: int(datetime.now().timestamp()))
    refresh: int = 3600
    retry: int = 1800
    expire: int = 604800
    minimum_ttl: int = 86400
    authoritative_servers: List[str] = field(default_factory=list)
    
    def add_record(self, record: DNSRecord) -> None:
        """Add a DNS record to the zone."""
        if record.name not in self.records:
            self.records[record.name] = []
        self.records[record.name].append(record)
        self.serial += 1
        record.last_updated = datetime.now()
    
    def remove_record(self, name: str, record_type: DNSRecordType) -> bool:
        """Remove DNS records by name and type."""
        if name in self.records:
            original_count = len(self.records[name])
            self.records[name] = [r for r in self.records[name] if r.record_type != record_type]
            if len(self.records[name]) < original_count:
                self.serial += 1
                return True
        return False
    
    def get_records(self, name: str, record_type: Optional[DNSRecordType] = None) -> List[DNSRecord]:
        """Get DNS records by name and optionally by type."""
        if name not in self.records:
            return []
        
        records = self.records[name]
        if record_type:
            records = [r for r in records if r.record_type == record_type]
        
        return records


@dataclass
class DNSQuery:
    """DNS query information."""
    query_id: int
    client_ip: str
    query_name: str
    query_type: DNSRecordType
    timestamp: datetime = field(default_factory=datetime.now)
    response_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    cached: bool = False


class DNSService:
    """
    DNS Service simulation for enterprise environments.
    
    Provides realistic DNS functionality including:
    - Forward and reverse DNS resolution
    - Zone management and transfers
    - Dynamic DNS updates
    - Query logging and statistics
    """
    
    def __init__(self, server_name: str = "dns01.archangel.local", 
                 listen_port: int = 53):
        self.server_name = server_name
        self.listen_port = listen_port
        self.zones: Dict[str, DNSZone] = {}
        self.query_log: List[DNSQuery] = []
        self.cache: Dict[str, Tuple[List[DNSRecord], datetime]] = {}
        self.cache_ttl = 300  # 5 minutes
        self.running = False
        self.logger = logging.getLogger(__name__)
        
        # Initialize default zones
        self._initialize_default_zones()
    
    def _initialize_default_zones(self) -> None:
        """Initialize default DNS zones for the enterprise environment."""
        
        # Forward zone for archangel.local
        forward_zone = DNSZone(
            name="archangel.local",
            zone_type="primary",
            authoritative_servers=["dns01.archangel.local", "dns02.archangel.local"]
        )
        
        # SOA record
        soa_record = DNSRecord(
            name="archangel.local",
            record_type=DNSRecordType.SOA,
            value=f"dns01.archangel.local. admin.archangel.local. {forward_zone.serial} {forward_zone.refresh} {forward_zone.retry} {forward_zone.expire} {forward_zone.minimum_ttl}"
        )
        forward_zone.add_record(soa_record)
        
        # NS records
        for ns_server in forward_zone.authoritative_servers:
            ns_record = DNSRecord(
                name="archangel.local",
                record_type=DNSRecordType.NS,
                value=ns_server
            )
            forward_zone.add_record(ns_record)
        
        # Default A records for infrastructure
        default_records = [
            ("dns01.archangel.local", "192.168.1.10"),
            ("dns02.archangel.local", "192.168.1.11"),
            ("dc01.archangel.local", "192.168.1.20"),
            ("dc02.archangel.local", "192.168.1.21"),
            ("web01.archangel.local", "192.168.10.10"),
            ("web02.archangel.local", "192.168.10.11"),
            ("db01.archangel.local", "192.168.20.10"),
            ("db02.archangel.local", "192.168.20.11"),
            ("mail01.archangel.local", "192.168.10.20"),
            ("file01.archangel.local", "192.168.20.20"),
        ]
        
        for hostname, ip_address in default_records:
            a_record = DNSRecord(
                name=hostname,
                record_type=DNSRecordType.A,
                value=ip_address
            )
            forward_zone.add_record(a_record)
        
        # CNAME records
        cname_records = [
            ("www.archangel.local", "web01.archangel.local"),
            ("intranet.archangel.local", "web02.archangel.local"),
            ("mail.archangel.local", "mail01.archangel.local"),
            ("files.archangel.local", "file01.archangel.local"),
        ]
        
        for alias, target in cname_records:
            cname_record = DNSRecord(
                name=alias,
                record_type=DNSRecordType.CNAME,
                value=target
            )
            forward_zone.add_record(cname_record)
        
        # MX records
        mx_record = DNSRecord(
            name="archangel.local",
            record_type=DNSRecordType.MX,
            value="mail01.archangel.local",
            priority=10
        )
        forward_zone.add_record(mx_record)
        
        # SRV records for Active Directory services
        srv_records = [
            ("_ldap._tcp.archangel.local", "dc01.archangel.local", 389, 0, 100),
            ("_kerberos._tcp.archangel.local", "dc01.archangel.local", 88, 0, 100),
            ("_gc._tcp.archangel.local", "dc01.archangel.local", 3268, 0, 100),
        ]
        
        for service_name, target, port, weight, priority in srv_records:
            srv_record = DNSRecord(
                name=service_name,
                record_type=DNSRecordType.SRV,
                value=target,
                port=port,
                weight=weight,
                priority=priority
            )
            forward_zone.add_record(srv_record)
        
        self.zones["archangel.local"] = forward_zone
        
        # Reverse zones
        self._create_reverse_zones()
    
    def _create_reverse_zones(self) -> None:
        """Create reverse DNS zones for PTR records."""
        
        reverse_zones = [
            "1.168.192.in-addr.arpa",  # 192.168.1.0/24
            "10.168.192.in-addr.arpa", # 192.168.10.0/24
            "20.168.192.in-addr.arpa", # 192.168.20.0/24
        ]
        
        for zone_name in reverse_zones:
            reverse_zone = DNSZone(
                name=zone_name,
                zone_type="primary",
                authoritative_servers=["dns01.archangel.local"]
            )
            
            # SOA record for reverse zone
            soa_record = DNSRecord(
                name=zone_name,
                record_type=DNSRecordType.SOA,
                value=f"dns01.archangel.local. admin.archangel.local. {reverse_zone.serial} {reverse_zone.refresh} {reverse_zone.retry} {reverse_zone.expire} {reverse_zone.minimum_ttl}"
            )
            reverse_zone.add_record(soa_record)
            
            self.zones[zone_name] = reverse_zone
        
        # Add PTR records for infrastructure hosts
        ptr_records = [
            ("10.1.168.192.in-addr.arpa", "dns01.archangel.local"),
            ("11.1.168.192.in-addr.arpa", "dns02.archangel.local"),
            ("20.1.168.192.in-addr.arpa", "dc01.archangel.local"),
            ("21.1.168.192.in-addr.arpa", "dc02.archangel.local"),
            ("10.10.168.192.in-addr.arpa", "web01.archangel.local"),
            ("11.10.168.192.in-addr.arpa", "web02.archangel.local"),
            ("10.20.168.192.in-addr.arpa", "db01.archangel.local"),
            ("11.20.168.192.in-addr.arpa", "db02.archangel.local"),
            ("20.10.168.192.in-addr.arpa", "mail01.archangel.local"),
            ("20.20.168.192.in-addr.arpa", "file01.archangel.local"),
        ]
        
        for ptr_name, hostname in ptr_records:
            # Determine which reverse zone this belongs to
            for zone_name in self.zones:
                if ptr_name.endswith(zone_name) and zone_name.endswith(".in-addr.arpa"):
                    ptr_record = DNSRecord(
                        name=ptr_name,
                        record_type=DNSRecordType.PTR,
                        value=hostname
                    )
                    self.zones[zone_name].add_record(ptr_record)
                    break
    
    def resolve_query(self, query_name: str, query_type: DNSRecordType, 
                     client_ip: str = "127.0.0.1") -> List[DNSRecord]:
        """
        Resolve a DNS query and return matching records.
        
        Args:
            query_name: The domain name to resolve
            query_type: The type of DNS record requested
            client_ip: IP address of the client making the query
            
        Returns:
            List of matching DNS records
        """
        
        start_time = time.time()
        query_id = len(self.query_log) + 1
        
        # Create query log entry
        query = DNSQuery(
            query_id=query_id,
            client_ip=client_ip,
            query_name=query_name,
            query_type=query_type
        )
        
        # Check cache first
        cache_key = f"{query_name}:{query_type.value}"
        if cache_key in self.cache:
            cached_records, cache_time = self.cache[cache_key]
            if datetime.now() - cache_time < timedelta(seconds=self.cache_ttl):
                query.cached = True
                query.response_code = 0  # NOERROR
                query.response_time_ms = (time.time() - start_time) * 1000
                self.query_log.append(query)
                return cached_records
        
        # Find appropriate zone
        zone = self._find_zone_for_query(query_name)
        if not zone:
            query.response_code = 3  # NXDOMAIN
            query.response_time_ms = (time.time() - start_time) * 1000
            self.query_log.append(query)
            return []
        
        # Get records from zone
        records = zone.get_records(query_name, query_type)
        
        # Handle CNAME resolution
        if not records and query_type == DNSRecordType.A:
            cname_records = zone.get_records(query_name, DNSRecordType.CNAME)
            if cname_records:
                # Follow CNAME chain
                target = cname_records[0].value
                a_records = zone.get_records(target, DNSRecordType.A)
                records = cname_records + a_records
        
        # Cache the results
        if records:
            self.cache[cache_key] = (records, datetime.now())
            query.response_code = 0  # NOERROR
        else:
            query.response_code = 3  # NXDOMAIN
        
        query.response_time_ms = (time.time() - start_time) * 1000
        self.query_log.append(query)
        
        self.logger.info(f"DNS query: {query_name} ({query_type.value}) from {client_ip} - {len(records)} records returned")
        
        return records
    
    def _find_zone_for_query(self, query_name: str) -> Optional[DNSZone]:
        """Find the most specific zone that can answer the query."""
        
        # Normalize query name
        query_name = query_name.lower().rstrip('.')
        
        # Find the longest matching zone
        best_match = None
        best_match_length = 0
        
        for zone_name, zone in self.zones.items():
            zone_name = zone_name.lower().rstrip('.')
            
            if query_name == zone_name or query_name.endswith('.' + zone_name):
                if len(zone_name) > best_match_length:
                    best_match = zone
                    best_match_length = len(zone_name)
        
        return best_match
    
    def add_dynamic_record(self, name: str, record_type: DNSRecordType, 
                          value: str, ttl: int = 3600) -> bool:
        """
        Add a dynamic DNS record.
        
        Args:
            name: Record name
            record_type: Type of DNS record
            value: Record value
            ttl: Time to live in seconds
            
        Returns:
            True if record was added successfully
        """
        
        zone = self._find_zone_for_query(name)
        if not zone:
            self.logger.error(f"No zone found for dynamic record: {name}")
            return False
        
        record = DNSRecord(
            name=name,
            record_type=record_type,
            value=value,
            ttl=ttl
        )
        
        zone.add_record(record)
        
        # Invalidate cache for this record
        cache_key = f"{name}:{record_type.value}"
        if cache_key in self.cache:
            del self.cache[cache_key]
        
        self.logger.info(f"Added dynamic DNS record: {name} {record_type.value} {value}")
        return True
    
    def remove_dynamic_record(self, name: str, record_type: DNSRecordType) -> bool:
        """Remove a dynamic DNS record."""
        
        zone = self._find_zone_for_query(name)
        if not zone:
            return False
        
        success = zone.remove_record(name, record_type)
        
        if success:
            # Invalidate cache
            cache_key = f"{name}:{record_type.value}"
            if cache_key in self.cache:
                del self.cache[cache_key]
            
            self.logger.info(f"Removed dynamic DNS record: {name} {record_type.value}")
        
        return success
    
    def get_zone_info(self, zone_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a DNS zone."""
        
        if zone_name not in self.zones:
            return None
        
        zone = self.zones[zone_name]
        
        return {
            "name": zone.name,
            "type": zone.zone_type,
            "serial": zone.serial,
            "refresh": zone.refresh,
            "retry": zone.retry,
            "expire": zone.expire,
            "minimum_ttl": zone.minimum_ttl,
            "authoritative_servers": zone.authoritative_servers,
            "record_count": sum(len(records) for records in zone.records.values())
        }
    
    def get_query_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get DNS query statistics for the specified time period."""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_queries = [q for q in self.query_log if q.timestamp >= cutoff_time]
        
        if not recent_queries:
            return {
                "total_queries": 0,
                "successful_queries": 0,
                "failed_queries": 0,
                "cached_queries": 0,
                "average_response_time_ms": 0,
                "top_queried_names": [],
                "query_types": {}
            }
        
        successful_queries = [q for q in recent_queries if q.response_code == 0]
        failed_queries = [q for q in recent_queries if q.response_code != 0]
        cached_queries = [q for q in recent_queries if q.cached]
        
        # Calculate average response time
        response_times = [q.response_time_ms for q in recent_queries if q.response_time_ms is not None]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Top queried names
        name_counts = {}
        for query in recent_queries:
            name_counts[query.query_name] = name_counts.get(query.query_name, 0) + 1
        
        top_names = sorted(name_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Query types distribution
        type_counts = {}
        for query in recent_queries:
            type_name = query.query_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return {
            "total_queries": len(recent_queries),
            "successful_queries": len(successful_queries),
            "failed_queries": len(failed_queries),
            "cached_queries": len(cached_queries),
            "average_response_time_ms": round(avg_response_time, 2),
            "top_queried_names": top_names,
            "query_types": type_counts
        }
    
    def export_zone_file(self, zone_name: str) -> Optional[str]:
        """Export a zone in standard DNS zone file format."""
        
        if zone_name not in self.zones:
            return None
        
        zone = self.zones[zone_name]
        lines = []
        
        # Zone header
        lines.append(f"; Zone file for {zone_name}")
        lines.append(f"; Generated on {datetime.now().isoformat()}")
        lines.append(f"$ORIGIN {zone_name}.")
        lines.append(f"$TTL {zone.minimum_ttl}")
        lines.append("")
        
        # Sort records by name and type
        all_records = []
        for name, records in zone.records.items():
            all_records.extend(records)
        
        all_records.sort(key=lambda r: (r.name, r.record_type.value))
        
        # Write records
        for record in all_records:
            name = record.name if record.name.endswith('.') else f"{record.name}."
            
            if record.record_type == DNSRecordType.SOA:
                lines.append(f"{name:<30} {record.ttl:<8} IN {record.record_type.value:<8} {record.value}")
            elif record.record_type == DNSRecordType.MX:
                lines.append(f"{name:<30} {record.ttl:<8} IN {record.record_type.value:<8} {record.priority} {record.value}")
            elif record.record_type == DNSRecordType.SRV:
                lines.append(f"{name:<30} {record.ttl:<8} IN {record.record_type.value:<8} {record.priority} {record.weight} {record.port} {record.value}")
            else:
                lines.append(f"{name:<30} {record.ttl:<8} IN {record.record_type.value:<8} {record.value}")
        
        return "\n".join(lines)
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get DNS server status information."""
        
        return {
            "server_name": self.server_name,
            "listen_port": self.listen_port,
            "running": self.running,
            "zones_count": len(self.zones),
            "total_queries": len(self.query_log),
            "cache_entries": len(self.cache),
            "uptime_seconds": time.time() - getattr(self, '_start_time', time.time())
        }