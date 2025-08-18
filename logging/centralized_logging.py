#!/usr/bin/env python3
"""
Centralized Logging System for Archangel AI Security Platform
Comprehensive log aggregation, parsing, and correlation system with SIEM integration
"""

import json
import logging
import logging.handlers
import os
import sys
import time
import asyncio
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
import hashlib
import threading
import queue
import socket
import struct
from pathlib import Path
import gzip
import shutil

# Advanced logging imports
try:
    import elasticsearch
    from elasticsearch.helpers import bulk
    HAS_ELASTICSEARCH = True
except ImportError:
    HAS_ELASTICSEARCH = False

try:
    import kafka
    from kafka import KafkaProducer, KafkaConsumer
    HAS_KAFKA = True
except ImportError:
    HAS_KAFKA = False

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

# Configure base logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """Standardized log levels with numeric values."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40
    FATAL = 50
    SECURITY = 60
    AUDIT = 70

class LogCategory(Enum):
    """Log categories for classification."""
    SYSTEM = "system"
    SECURITY = "security"  
    AUDIT = "audit"
    PERFORMANCE = "performance"
    APPLICATION = "application"
    NETWORK = "network"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    THREAT_DETECTION = "threat_detection"
    INCIDENT = "incident"
    COMPLIANCE = "compliance"

class EventSeverity(Enum):
    """Security event severity levels."""
    INFORMATIONAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5

@dataclass
class LogEntry:
    """Standardized log entry structure."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    level: LogLevel = LogLevel.INFO
    category: LogCategory = LogCategory.SYSTEM
    source: str = "archangel"
    component: str = "unknown"
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Security-specific fields
    severity: EventSeverity = EventSeverity.INFORMATIONAL
    threat_indicators: List[str] = field(default_factory=list)
    mitre_tactics: List[str] = field(default_factory=list)
    mitre_techniques: List[str] = field(default_factory=list)
    
    # Network context
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    source_port: Optional[int] = None
    destination_port: Optional[int] = None
    protocol: Optional[str] = None
    
    # File system context
    file_path: Optional[str] = None
    file_hash: Optional[str] = None
    file_size: Optional[int] = None
    
    # Process context
    process_id: Optional[int] = None
    process_name: Optional[str] = None
    parent_process_id: Optional[int] = None
    command_line: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
        
        # Ensure timezone is set
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
        
        # Generate hash for deduplication
        self._generate_hash()
    
    def _generate_hash(self):
        """Generate hash for log entry deduplication."""
        hash_data = f"{self.source}:{self.component}:{self.message}:{self.level.value}"
        self.metadata['entry_hash'] = hashlib.sha256(hash_data.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary."""
        data = asdict(self)
        
        # Convert enums to strings
        data['level'] = self.level.name
        data['category'] = self.category.value
        data['severity'] = self.severity.name
        data['timestamp'] = self.timestamp.isoformat()
        
        # Add computed fields
        data['@timestamp'] = self.timestamp.isoformat()
        data['event_id'] = self.correlation_id
        data['log_type'] = 'archangel'
        
        return data
    
    def to_json(self) -> str:
        """Convert log entry to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        """Create LogEntry from dictionary."""
        # Convert string enums back
        if isinstance(data.get('level'), str):
            data['level'] = LogLevel[data['level']]
        if isinstance(data.get('category'), str):
            data['category'] = LogCategory(data['category'])
        if isinstance(data.get('severity'), str):
            data['severity'] = EventSeverity[data['severity']]
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'LogEntry':
        """Create LogEntry from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

class LogAggregator:
    """Centralized log aggregation and processing system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.buffer_size = config.get('buffer_size', 1000)
        self.flush_interval = config.get('flush_interval', 5.0)  # seconds
        self.max_retries = config.get('max_retries', 3)
        
        # Buffers and queues
        self.log_buffer = queue.Queue(maxsize=self.buffer_size * 2)
        self.batch_buffer = []
        self.processed_hashes = set()  # For deduplication
        
        # Threading
        self.running = False
        self.processor_thread = None
        self.flusher_thread = None
        
        # Statistics
        self.stats = {
            'total_logs': 0,
            'processed_logs': 0,
            'failed_logs': 0,
            'duplicate_logs': 0,
            'buffer_overflows': 0,
            'last_flush': None
        }
        
        # Initialize outputs
        self.outputs = self._initialize_outputs()
        
        # Initialize filters and enrichers
        self.filters = self._initialize_filters()
        self.enrichers = self._initialize_enrichers()
    
    def _initialize_outputs(self) -> List[Any]:
        """Initialize log output destinations."""
        outputs = []
        
        # File output
        if self.config.get('file_output', {}).get('enabled', True):
            file_config = self.config.get('file_output', {})
            outputs.append(FileOutput(file_config))
        
        # Elasticsearch output
        if self.config.get('elasticsearch', {}).get('enabled', False) and HAS_ELASTICSEARCH:
            es_config = self.config.get('elasticsearch', {})
            outputs.append(ElasticsearchOutput(es_config))
        
        # Kafka output
        if self.config.get('kafka', {}).get('enabled', False) and HAS_KAFKA:
            kafka_config = self.config.get('kafka', {})
            outputs.append(KafkaOutput(kafka_config))
        
        # Syslog output
        if self.config.get('syslog', {}).get('enabled', False):
            syslog_config = self.config.get('syslog', {})
            outputs.append(SyslogOutput(syslog_config))
        
        # Redis output
        if self.config.get('redis', {}).get('enabled', False) and HAS_REDIS:
            redis_config = self.config.get('redis', {})
            outputs.append(RedisOutput(redis_config))
        
        # HTTP webhook output
        if self.config.get('webhook', {}).get('enabled', False):
            webhook_config = self.config.get('webhook', {})
            outputs.append(WebhookOutput(webhook_config))
        
        return outputs
    
    def _initialize_filters(self) -> List[Callable]:
        """Initialize log filtering functions."""
        filters = []
        
        # Level filter
        min_level = self.config.get('filters', {}).get('min_level', LogLevel.INFO)
        filters.append(lambda entry: entry.level.value >= min_level.value)
        
        # Category filter
        allowed_categories = self.config.get('filters', {}).get('categories', [])
        if allowed_categories:
            filters.append(lambda entry: entry.category.value in allowed_categories)
        
        # Source filter
        allowed_sources = self.config.get('filters', {}).get('sources', [])
        if allowed_sources:
            filters.append(lambda entry: entry.source in allowed_sources)
        
        # Regex filters
        regex_filters = self.config.get('filters', {}).get('regex', [])
        for regex_config in regex_filters:
            import re
            pattern = re.compile(regex_config['pattern'])
            field = regex_config.get('field', 'message')
            filters.append(lambda entry: bool(pattern.search(getattr(entry, field, ''))))
        
        return filters
    
    def _initialize_enrichers(self) -> List[Callable]:
        """Initialize log enrichment functions."""
        enrichers = []
        
        # Hostname enricher
        if self.config.get('enrichment', {}).get('add_hostname', True):
            hostname = socket.gethostname()
            enrichers.append(lambda entry: entry.metadata.update({'hostname': hostname}))
        
        # Process ID enricher
        if self.config.get('enrichment', {}).get('add_process_id', True):
            enrichers.append(lambda entry: entry.metadata.update({'pid': os.getpid()}))
        
        # Environment enricher
        if self.config.get('enrichment', {}).get('add_environment', True):
            env = os.getenv('ENVIRONMENT', 'unknown')
            enrichers.append(lambda entry: entry.metadata.update({'environment': env}))
        
        # GeoIP enricher (if source_ip exists)
        if self.config.get('enrichment', {}).get('add_geoip', False):
            enrichers.append(self._enrich_geoip)
        
        # Threat intelligence enricher
        if self.config.get('enrichment', {}).get('add_threat_intel', False):
            enrichers.append(self._enrich_threat_intel)
        
        return enrichers
    
    def start(self):
        """Start the log aggregator."""
        if self.running:
            logger.warning("Log aggregator is already running")
            return
        
        self.running = True
        
        # Start processor thread
        self.processor_thread = threading.Thread(target=self._processor_loop, daemon=True)
        self.processor_thread.start()
        
        # Start flusher thread
        self.flusher_thread = threading.Thread(target=self._flusher_loop, daemon=True)
        self.flusher_thread.start()
        
        logger.info("Log aggregator started")
    
    def stop(self):
        """Stop the log aggregator."""
        if not self.running:
            return
        
        self.running = False
        
        # Flush remaining logs
        self._flush_batch()
        
        # Wait for threads to finish
        if self.processor_thread:
            self.processor_thread.join(timeout=5)
        if self.flusher_thread:
            self.flusher_thread.join(timeout=5)
        
        logger.info("Log aggregator stopped")
    
    def ingest_log(self, log_entry: LogEntry):
        """Ingest a log entry for processing."""
        try:
            # Apply filters
            if not self._apply_filters(log_entry):
                return
            
            # Check for duplicates
            entry_hash = log_entry.metadata.get('entry_hash')
            if entry_hash in self.processed_hashes:
                self.stats['duplicate_logs'] += 1
                return
            
            # Add to buffer
            self.log_buffer.put_nowait(log_entry)
            self.stats['total_logs'] += 1
            
        except queue.Full:
            self.stats['buffer_overflows'] += 1
            logger.warning("Log buffer overflow, dropping log entry")
        except Exception as e:
            logger.error(f"Failed to ingest log entry: {e}")
    
    def ingest_logs(self, log_entries: List[LogEntry]):
        """Ingest multiple log entries."""
        for entry in log_entries:
            self.ingest_log(entry)
    
    def _processor_loop(self):
        """Main log processing loop."""
        while self.running:
            try:
                # Get log entry from buffer
                try:
                    log_entry = self.log_buffer.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Apply enrichers
                self._apply_enrichers(log_entry)
                
                # Add to batch buffer
                self.batch_buffer.append(log_entry)
                
                # Mark as processed
                entry_hash = log_entry.metadata.get('entry_hash')
                if entry_hash:
                    self.processed_hashes.add(entry_hash)
                
                self.stats['processed_logs'] += 1
                
                # Flush if batch is full
                if len(self.batch_buffer) >= self.buffer_size:
                    self._flush_batch()
                
            except Exception as e:
                logger.error(f"Error in processor loop: {e}")
                self.stats['failed_logs'] += 1
    
    def _flusher_loop(self):
        """Periodic batch flushing loop."""
        while self.running:
            time.sleep(self.flush_interval)
            if self.batch_buffer:
                self._flush_batch()
    
    def _apply_filters(self, log_entry: LogEntry) -> bool:
        """Apply filters to log entry."""
        for filter_func in self.filters:
            try:
                if not filter_func(log_entry):
                    return False
            except Exception as e:
                logger.warning(f"Filter function failed: {e}")
        return True
    
    def _apply_enrichers(self, log_entry: LogEntry):
        """Apply enrichers to log entry."""
        for enricher_func in self.enrichers:
            try:
                enricher_func(log_entry)
            except Exception as e:
                logger.warning(f"Enricher function failed: {e}")
    
    def _enrich_geoip(self, log_entry: LogEntry):
        """Enrich log entry with GeoIP information."""
        if log_entry.source_ip:
            # Placeholder for GeoIP enrichment
            # In real implementation, use GeoIP library
            log_entry.metadata['geoip'] = {
                'country': 'Unknown',
                'city': 'Unknown',
                'coordinates': {'lat': 0, 'lon': 0}
            }
    
    def _enrich_threat_intel(self, log_entry: LogEntry):
        """Enrich log entry with threat intelligence."""
        # Check IPs against threat intel feeds
        if log_entry.source_ip:
            # Placeholder for threat intel lookup
            log_entry.metadata['threat_intel'] = {
                'is_malicious': False,
                'reputation_score': 0,
                'categories': []
            }
    
    def _flush_batch(self):
        """Flush the current batch to outputs."""
        if not self.batch_buffer:
            return
        
        batch = self.batch_buffer.copy()
        self.batch_buffer.clear()
        
        # Send to all outputs
        for output in self.outputs:
            try:
                output.write_batch(batch)
            except Exception as e:
                logger.error(f"Failed to write to output {output.__class__.__name__}: {e}")
        
        self.stats['last_flush'] = datetime.now(timezone.utc)
        logger.debug(f"Flushed batch of {len(batch)} log entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        stats = self.stats.copy()
        stats['buffer_size'] = self.log_buffer.qsize()
        stats['batch_size'] = len(self.batch_buffer)
        stats['outputs_count'] = len(self.outputs)
        return stats


class LogOutput:
    """Base class for log outputs."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', True)
    
    def write_log(self, log_entry: LogEntry):
        """Write a single log entry."""
        raise NotImplementedError
    
    def write_batch(self, log_entries: List[LogEntry]):
        """Write a batch of log entries."""
        for entry in log_entries:
            if self.enabled:
                self.write_log(entry)


class FileOutput(LogOutput):
    """File-based log output with rotation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.log_dir = config.get('directory', './logs')
        self.file_pattern = config.get('file_pattern', 'archangel-{date}.log')
        self.max_file_size = config.get('max_file_size', 100 * 1024 * 1024)  # 100MB
        self.max_files = config.get('max_files', 30)
        self.compress_old = config.get('compress_old', True)
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Current file handle
        self.current_file = None
        self.current_file_path = None
        self.current_file_size = 0
    
    def _get_log_file_path(self) -> str:
        """Get current log file path."""
        date_str = datetime.now().strftime('%Y-%m-%d')
        filename = self.file_pattern.format(date=date_str)
        return os.path.join(self.log_dir, filename)
    
    def _rotate_if_needed(self):
        """Rotate log file if needed."""
        current_path = self._get_log_file_path()
        
        # Check if we need to open a new file
        if (self.current_file_path != current_path or 
            self.current_file_size > self.max_file_size):
            
            # Close current file
            if self.current_file:
                self.current_file.close()
            
            # Compress old file if configured
            if (self.current_file_path and 
                self.current_file_path != current_path and 
                self.compress_old and 
                os.path.exists(self.current_file_path)):
                self._compress_file(self.current_file_path)
            
            # Open new file
            self.current_file_path = current_path
            self.current_file = open(current_path, 'a', encoding='utf-8')
            self.current_file_size = os.path.getsize(current_path) if os.path.exists(current_path) else 0
    
    def _compress_file(self, file_path: str):
        """Compress a log file."""
        try:
            compressed_path = file_path + '.gz'
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(file_path)
            logger.debug(f"Compressed log file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to compress log file {file_path}: {e}")
    
    def write_log(self, log_entry: LogEntry):
        """Write log entry to file."""
        if not self.enabled:
            return
        
        try:
            self._rotate_if_needed()
            
            log_line = log_entry.to_json() + '\n'
            self.current_file.write(log_line)
            self.current_file.flush()
            
            self.current_file_size += len(log_line.encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Failed to write log to file: {e}")


class ElasticsearchOutput(LogOutput):
    """Elasticsearch log output."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not HAS_ELASTICSEARCH:
            logger.error("Elasticsearch not available - install elasticsearch package")
            self.enabled = False
            return
        
        self.hosts = config.get('hosts', ['localhost:9200'])
        self.index_pattern = config.get('index_pattern', 'archangel-logs-{date}')
        self.doc_type = config.get('doc_type', '_doc')
        self.timeout = config.get('timeout', 30)
        
        # Initialize Elasticsearch client
        try:
            self.client = elasticsearch.Elasticsearch(
                hosts=self.hosts,
                timeout=self.timeout,
                max_retries=3,
                retry_on_timeout=True
            )
            
            # Test connection
            if not self.client.ping():
                logger.error("Cannot connect to Elasticsearch")
                self.enabled = False
                
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch client: {e}")
            self.enabled = False
    
    def _get_index_name(self) -> str:
        """Get current index name."""
        date_str = datetime.now().strftime('%Y.%m.%d')
        return self.index_pattern.format(date=date_str)
    
    def write_log(self, log_entry: LogEntry):
        """Write log entry to Elasticsearch."""
        if not self.enabled:
            return
        
        try:
            index_name = self._get_index_name()
            doc = log_entry.to_dict()
            
            self.client.index(
                index=index_name,
                doc_type=self.doc_type,
                body=doc,
                id=log_entry.correlation_id
            )
            
        except Exception as e:
            logger.error(f"Failed to write log to Elasticsearch: {e}")
    
    def write_batch(self, log_entries: List[LogEntry]):
        """Write batch of log entries to Elasticsearch."""
        if not self.enabled or not log_entries:
            return
        
        try:
            actions = []
            index_name = self._get_index_name()
            
            for entry in log_entries:
                action = {
                    '_index': index_name,
                    '_type': self.doc_type,
                    '_id': entry.correlation_id,
                    '_source': entry.to_dict()
                }
                actions.append(action)
            
            # Bulk index
            success_count, failed_items = bulk(
                self.client,
                actions,
                chunk_size=500,
                request_timeout=self.timeout
            )
            
            if failed_items:
                logger.warning(f"Failed to index {len(failed_items)} documents to Elasticsearch")
            
        except Exception as e:
            logger.error(f"Failed to bulk write to Elasticsearch: {e}")


class SyslogOutput(LogOutput):
    """Syslog output for SIEM integration."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 514)
        self.protocol = config.get('protocol', 'UDP').upper()
        self.facility = config.get('facility', 16)  # Local use 0
        
        # Initialize socket
        if self.protocol == 'TCP':
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
        else:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    def write_log(self, log_entry: LogEntry):
        """Write log entry to syslog."""
        if not self.enabled:
            return
        
        try:
            # Calculate priority (facility * 8 + severity)
            severity_map = {
                LogLevel.TRACE: 7,    # Debug
                LogLevel.DEBUG: 7,    # Debug  
                LogLevel.INFO: 6,     # Informational
                LogLevel.WARN: 4,     # Warning
                LogLevel.ERROR: 3,    # Error
                LogLevel.FATAL: 2,    # Critical
                LogLevel.SECURITY: 1, # Alert
                LogLevel.AUDIT: 6     # Informational
            }
            
            severity = severity_map.get(log_entry.level, 6)
            priority = self.facility * 8 + severity
            
            # Format syslog message (RFC 3164)
            timestamp_str = log_entry.timestamp.strftime('%b %d %H:%M:%S')
            hostname = log_entry.metadata.get('hostname', 'archangel')
            tag = f"{log_entry.source}[{os.getpid()}]"
            
            message = f"<{priority}>{timestamp_str} {hostname} {tag}: {log_entry.to_json()}"
            
            # Send message
            if self.protocol == 'TCP':
                self.socket.send(message.encode('utf-8'))
            else:
                self.socket.sendto(message.encode('utf-8'), (self.host, self.port))
                
        except Exception as e:
            logger.error(f"Failed to write log to syslog: {e}")


class KafkaOutput(LogOutput):
    """Kafka output for streaming log data."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not HAS_KAFKA:
            logger.error("Kafka not available - install kafka-python package")
            self.enabled = False
            return
        
        self.bootstrap_servers = config.get('bootstrap_servers', ['localhost:9092'])
        self.topic = config.get('topic', 'archangel-logs')
        self.key_field = config.get('key_field', 'source')
        
        # Initialize Kafka producer
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: str(k).encode('utf-8') if k else None,
                acks='all',
                retries=3,
                max_in_flight_requests_per_connection=1
            )
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            self.enabled = False
    
    def write_log(self, log_entry: LogEntry):
        """Write log entry to Kafka."""
        if not self.enabled:
            return
        
        try:
            key = getattr(log_entry, self.key_field, None)
            value = log_entry.to_dict()
            
            self.producer.send(self.topic, key=key, value=value)
            
        except Exception as e:
            logger.error(f"Failed to write log to Kafka: {e}")
    
    def write_batch(self, log_entries: List[LogEntry]):
        """Write batch of log entries to Kafka."""
        if not self.enabled:
            return
        
        try:
            for entry in log_entries:
                key = getattr(entry, self.key_field, None)
                value = entry.to_dict()
                self.producer.send(self.topic, key=key, value=value)
            
            # Flush to ensure delivery
            self.producer.flush(timeout=30)
            
        except Exception as e:
            logger.error(f"Failed to batch write to Kafka: {e}")


class RedisOutput(LogOutput):
    """Redis output for real-time log streaming."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not HAS_REDIS:
            logger.error("Redis not available - install redis package")
            self.enabled = False
            return
        
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 6379)
        self.db = config.get('db', 0)
        self.key_pattern = config.get('key_pattern', 'archangel:logs:{category}')
        self.max_list_size = config.get('max_list_size', 10000)
        
        # Initialize Redis client
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=False
            )
            
            # Test connection
            self.client.ping()
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            self.enabled = False
    
    def write_log(self, log_entry: LogEntry):
        """Write log entry to Redis."""
        if not self.enabled:
            return
        
        try:
            key = self.key_pattern.format(category=log_entry.category.value)
            value = log_entry.to_json()
            
            # Add to list and trim to max size
            pipe = self.client.pipeline()
            pipe.lpush(key, value)
            pipe.ltrim(key, 0, self.max_list_size - 1)
            pipe.execute()
            
        except Exception as e:
            logger.error(f"Failed to write log to Redis: {e}")


class WebhookOutput(LogOutput):
    """HTTP webhook output for external integrations."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.url = config.get('url')
        self.method = config.get('method', 'POST').upper()
        self.headers = config.get('headers', {'Content-Type': 'application/json'})
        self.timeout = config.get('timeout', 30)
        self.batch_size = config.get('batch_size', 10)
        
        if not self.url:
            logger.error("Webhook URL not configured")
            self.enabled = False
    
    def write_log(self, log_entry: LogEntry):
        """Write log entry to webhook."""
        if not self.enabled:
            return
        
        try:
            import requests
            
            payload = log_entry.to_dict()
            response = requests.request(
                method=self.method,
                url=self.url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to write log to webhook: {e}")
    
    def write_batch(self, log_entries: List[LogEntry]):
        """Write batch of log entries to webhook."""
        if not self.enabled:
            return
        
        try:
            import requests
            
            # Split into smaller batches
            for i in range(0, len(log_entries), self.batch_size):
                batch = log_entries[i:i + self.batch_size]
                payload = {
                    'logs': [entry.to_dict() for entry in batch],
                    'count': len(batch)
                }
                
                response = requests.request(
                    method=self.method,
                    url=self.url,
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                
        except Exception as e:
            logger.error(f"Failed to batch write to webhook: {e}")


class ArchangelLogger:
    """High-level logger interface for Archangel components."""
    
    def __init__(self, component: str, source: str = "archangel", aggregator: Optional[LogAggregator] = None):
        self.component = component
        self.source = source
        self.aggregator = aggregator
        
        # Context for structured logging
        self.context = {}
    
    def set_context(self, **context):
        """Set logging context."""
        self.context.update(context)
    
    def clear_context(self):
        """Clear logging context."""
        self.context.clear()
    
    def _log(self, level: LogLevel, category: LogCategory, message: str, **kwargs):
        """Internal logging method."""
        log_entry = LogEntry(
            level=level,
            category=category,
            source=self.source,
            component=self.component,
            message=message,
            details=kwargs.get('details', {}),
            **{k: v for k, v in kwargs.items() if k != 'details'},
            **self.context
        )
        
        if self.aggregator:
            self.aggregator.ingest_log(log_entry)
        else:
            # Fallback to standard logging
            std_logger = logging.getLogger(f"{self.source}.{self.component}")
            std_logger.log(level.value, f"{message} {kwargs}")
    
    def trace(self, message: str, **kwargs):
        """Log trace message."""
        self._log(LogLevel.TRACE, LogCategory.SYSTEM, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(LogLevel.DEBUG, LogCategory.SYSTEM, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(LogLevel.INFO, LogCategory.SYSTEM, message, **kwargs)
    
    def warn(self, message: str, **kwargs):
        """Log warning message."""
        self._log(LogLevel.WARN, LogCategory.SYSTEM, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(LogLevel.ERROR, LogCategory.SYSTEM, message, **kwargs)
    
    def fatal(self, message: str, **kwargs):
        """Log fatal message."""
        self._log(LogLevel.FATAL, LogCategory.SYSTEM, message, **kwargs)
    
    def security(self, message: str, severity: EventSeverity = EventSeverity.MEDIUM, **kwargs):
        """Log security event."""
        self._log(LogLevel.SECURITY, LogCategory.SECURITY, message, severity=severity, **kwargs)
    
    def audit(self, message: str, **kwargs):
        """Log audit event."""
        self._log(LogLevel.AUDIT, LogCategory.AUDIT, message, **kwargs)
    
    def performance(self, message: str, **kwargs):
        """Log performance event."""
        self._log(LogLevel.INFO, LogCategory.PERFORMANCE, message, **kwargs)
    
    def threat_detection(self, message: str, indicators: List[str] = None, mitre_tactics: List[str] = None, **kwargs):
        """Log threat detection event."""
        kwargs['threat_indicators'] = indicators or []
        kwargs['mitre_tactics'] = mitre_tactics or []
        self._log(LogLevel.SECURITY, LogCategory.THREAT_DETECTION, message, severity=EventSeverity.HIGH, **kwargs)


def create_default_aggregator() -> LogAggregator:
    """Create default log aggregator configuration."""
    config = {
        'buffer_size': 1000,
        'flush_interval': 5.0,
        'file_output': {
            'enabled': True,
            'directory': './logs',
            'file_pattern': 'archangel-{date}.log',
            'max_file_size': 100 * 1024 * 1024,  # 100MB
            'compress_old': True
        },
        'elasticsearch': {
            'enabled': HAS_ELASTICSEARCH and os.getenv('ELASTICSEARCH_ENABLED', 'false').lower() == 'true',
            'hosts': os.getenv('ELASTICSEARCH_HOSTS', 'localhost:9200').split(','),
            'index_pattern': 'archangel-logs-{date}'
        },
        'syslog': {
            'enabled': os.getenv('SYSLOG_ENABLED', 'false').lower() == 'true',
            'host': os.getenv('SYSLOG_HOST', 'localhost'),
            'port': int(os.getenv('SYSLOG_PORT', '514')),
            'protocol': os.getenv('SYSLOG_PROTOCOL', 'UDP')
        },
        'filters': {
            'min_level': LogLevel.INFO
        },
        'enrichment': {
            'add_hostname': True,
            'add_process_id': True,
            'add_environment': True
        }
    }
    
    return LogAggregator(config)


# Global aggregator instance
_global_aggregator = None

def get_logger(component: str, source: str = "archangel") -> ArchangelLogger:
    """Get logger instance for component."""
    global _global_aggregator
    
    if _global_aggregator is None:
        _global_aggregator = create_default_aggregator()
        _global_aggregator.start()
    
    return ArchangelLogger(component, source, _global_aggregator)

def get_aggregator() -> LogAggregator:
    """Get global log aggregator."""
    global _global_aggregator
    
    if _global_aggregator is None:
        _global_aggregator = create_default_aggregator()
    
    return _global_aggregator


if __name__ == "__main__":
    # Example usage
    logger = get_logger("test_component")
    
    # Set context
    logger.set_context(user_id="user123", session_id="sess456")
    
    # Log various events
    logger.info("System started successfully")
    logger.security("Potential brute force attack detected", 
                   severity=EventSeverity.HIGH,
                   source_ip="192.168.1.100",
                   threat_indicators=["multiple_failed_logins", "suspicious_user_agent"])
    logger.audit("User logged in", user_id="user123", action="login")
    logger.performance("Query executed", duration=0.150, query="SELECT * FROM users")
    logger.threat_detection("Malware signature detected", 
                          indicators=["suspicious_file_hash"],
                          mitre_tactics=["TA0002"])  # Execution
    
    # Get stats
    aggregator = get_aggregator()
    print("Aggregator stats:", aggregator.get_stats())
    
    # Keep running for a bit to see logs processed
    time.sleep(10)
    
    # Stop aggregator
    aggregator.stop()