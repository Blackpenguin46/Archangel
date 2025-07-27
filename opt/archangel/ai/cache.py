"""
Decision Cache for Archangel AI

Implements a high-performance caching system for AI decisions to reduce
inference latency and improve system responsiveness.
"""

import time
import threading
import logging
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached decision entry"""
    key: str
    value: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired"""
        return time.time() > (self.timestamp + self.ttl)
    
    def touch(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_access = time.time()


class LRUCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired_removals': 0,
            'size': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            entry = self.cache.get(key)
            if entry is None:
                self.stats['misses'] += 1
                return None
            
            if entry.is_expired():
                del self.cache[key]
                self.stats['expired_removals'] += 1
                self.stats['misses'] += 1
                self.stats['size'] = len(self.cache)
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            entry.touch()
            self.stats['hits'] += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: float = 300.0):
        """Set value in cache"""
        with self.lock:
            current_time = time.time()
            
            if key in self.cache:
                # Update existing entry
                entry = self.cache[key]
                entry.value = value
                entry.timestamp = current_time
                entry.ttl = ttl
                entry.touch()
                self.cache.move_to_end(key)
            else:
                # Add new entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    timestamp=current_time,
                    ttl=ttl
                )
                self.cache[key] = entry
                
                # Evict oldest if necessary
                if len(self.cache) > self.max_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    self.stats['evictions'] += 1
            
            self.stats['size'] = len(self.cache)
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.stats['size'] = len(self.cache)
                return True
            return False
    
    def clear(self):
        """Clear all entries from cache"""
        with self.lock:
            self.cache.clear()
            self.stats['size'] = 0
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed"""
        with self.lock:
            current_time = time.time()
            expired_keys = []
            
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
            
            removed_count = len(expired_keys)
            self.stats['expired_removals'] += removed_count
            self.stats['size'] = len(self.cache)
            
            return removed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                **self.stats,
                'hit_rate_percent': hit_rate,
                'total_requests': total_requests
            }


class DecisionCache:
    """High-level decision cache for AI decisions"""
    
    def __init__(self, max_size: int = 10000, default_ttl: float = 300.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = LRUCache(max_size)
        self.cleanup_interval = 60.0  # 1 minute
        self.last_cleanup = time.time()
        
        # Cache partitioning for different decision types
        self.partitions = {
            'syscall': LRUCache(max_size // 4),
            'network': LRUCache(max_size // 4),
            'memory': LRUCache(max_size // 4),
            'process': LRUCache(max_size // 4),
        }
        
        logger.info(f"DecisionCache initialized with max_size={max_size}, default_ttl={default_ttl}s")
    
    def _get_partition_key(self, key: str) -> str:
        """Determine which partition to use based on key"""
        if 'syscall' in key:
            return 'syscall'
        elif 'network' in key:
            return 'network'
        elif 'memory' in key:
            return 'memory'
        elif 'process' in key:
            return 'process'
        else:
            return 'default'
    
    def get(self, key: str) -> Optional[Any]:
        """Get decision from cache"""
        self._maybe_cleanup()
        
        partition_key = self._get_partition_key(key)
        if partition_key in self.partitions:
            return self.partitions[partition_key].get(key)
        else:
            return self.cache.get(key)
    
    def set(self, key: str, decision: Any, ttl: Optional[float] = None):
        """Set decision in cache"""
        if ttl is None:
            ttl = self.default_ttl
        
        partition_key = self._get_partition_key(key)
        if partition_key in self.partitions:
            self.partitions[partition_key].set(key, decision, ttl)
        else:
            self.cache.set(key, decision, ttl)
        
        logger.debug(f"Cached decision with key={key[:16]}..., ttl={ttl}s")
    
    def delete(self, key: str) -> bool:
        """Delete decision from cache"""
        partition_key = self._get_partition_key(key)
        if partition_key in self.partitions:
            return self.partitions[partition_key].delete(key)
        else:
            return self.cache.delete(key)
    
    def clear(self):
        """Clear all cached decisions"""
        self.cache.clear()
        for partition in self.partitions.values():
            partition.clear()
        logger.info("Decision cache cleared")
    
    def _maybe_cleanup(self):
        """Perform cleanup if interval has passed"""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self.cleanup_expired()
            self.last_cleanup = current_time
    
    def cleanup_expired(self) -> int:
        """Remove expired entries from all partitions"""
        total_removed = 0
        
        total_removed += self.cache.cleanup_expired()
        for partition in self.partitions.values():
            total_removed += partition.cleanup_expired()
        
        if total_removed > 0:
            logger.debug(f"Cleaned up {total_removed} expired cache entries")
        
        return total_removed
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        main_stats = self.cache.get_stats()
        partition_stats = {}
        
        for name, partition in self.partitions.items():
            partition_stats[name] = partition.get_stats()
        
        total_size = main_stats['size'] + sum(stats['size'] for stats in partition_stats.values())
        total_hits = main_stats['hits'] + sum(stats['hits'] for stats in partition_stats.values())
        total_misses = main_stats['misses'] + sum(stats['misses'] for stats in partition_stats.values())
        total_requests = total_hits + total_misses
        overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'overall': {
                'total_size': total_size,
                'max_size': self.max_size,
                'utilization_percent': (total_size / self.max_size * 100),
                'total_hits': total_hits,
                'total_misses': total_misses,
                'total_requests': total_requests,
                'hit_rate_percent': overall_hit_rate,
            },
            'main_cache': main_stats,
            'partitions': partition_stats
        }
    
    def get_cache_keys(self, limit: int = 100) -> List[str]:
        """Get list of cache keys for debugging"""
        keys = []
        
        # Get keys from main cache
        with self.cache.lock:
            keys.extend(list(self.cache.cache.keys())[:limit//2])
        
        # Get keys from partitions
        remaining_limit = limit - len(keys)
        per_partition_limit = max(1, remaining_limit // len(self.partitions))
        
        for partition in self.partitions.values():
            with partition.lock:
                keys.extend(list(partition.cache.keys())[:per_partition_limit])
                if len(keys) >= limit:
                    break
        
        return keys[:limit]
    
    def warm_cache(self, decisions: List[Tuple[str, Any, float]]):
        """Pre-populate cache with decisions"""
        for key, decision, ttl in decisions:
            self.set(key, decision, ttl)
        
        logger.info(f"Warmed cache with {len(decisions)} decisions")


class CacheKeyGenerator:
    """Utility class for generating consistent cache keys"""
    
    @staticmethod
    def generate_key(data: Dict[str, Any], prefix: str = "") -> str:
        """Generate a consistent cache key from data"""
        # Sort keys for consistency
        sorted_data = json.dumps(data, sort_keys=True, separators=(',', ':'))
        
        # Create hash
        hash_obj = hashlib.sha256(sorted_data.encode('utf-8'))
        key_hash = hash_obj.hexdigest()[:16]  # Use first 16 characters
        
        if prefix:
            return f"{prefix}:{key_hash}"
        else:
            return key_hash
    
    @staticmethod
    def generate_syscall_key(syscall_num: int, process_name: str, args: List[Any]) -> str:
        """Generate cache key for syscall decisions"""
        data = {
            'type': 'syscall',
            'syscall_num': syscall_num,
            'process_name': process_name,
            'args_hash': hashlib.md5(str(args).encode()).hexdigest()[:8]
        }
        return CacheKeyGenerator.generate_key(data, 'syscall')
    
    @staticmethod
    def generate_network_key(src_ip: str, dst_ip: str, dst_port: int, protocol: str) -> str:
        """Generate cache key for network decisions"""
        data = {
            'type': 'network',
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'dst_port': dst_port,
            'protocol': protocol
        }
        return CacheKeyGenerator.generate_key(data, 'network')
    
    @staticmethod
    def generate_memory_key(address: int, access_type: str, size: int) -> str:
        """Generate cache key for memory decisions"""
        data = {
            'type': 'memory',
            'address': address,
            'access_type': access_type,
            'size': size
        }
        return CacheKeyGenerator.generate_key(data, 'memory')
    
    @staticmethod
    def generate_process_key(pid: int, process_name: str, parent_pid: int) -> str:
        """Generate cache key for process decisions"""
        data = {
            'type': 'process',
            'pid': pid,
            'process_name': process_name,
            'parent_pid': parent_pid
        }
        return CacheKeyGenerator.generate_key(data, 'process')