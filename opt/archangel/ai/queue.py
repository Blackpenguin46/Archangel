"""
Operation Queue for Archangel AI

Implements a priority-based operation queue for managing AI requests
and ensuring proper ordering and resource allocation.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import heapq
import uuid

logger = logging.getLogger(__name__)


class OperationPriority(Enum):
    """Operation priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class OperationStatus(Enum):
    """Operation status values"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Operation:
    """Represents an operation in the queue"""
    id: str
    type: str
    priority: OperationPriority
    data: Dict[str, Any]
    callback: Optional[Callable] = None
    context: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: OperationStatus = OperationStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: float = 30.0
    
    def __lt__(self, other):
        """Comparison for priority queue (higher priority first)"""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.created_at < other.created_at
    
    def is_expired(self) -> bool:
        """Check if operation has timed out"""
        if self.started_at is None:
            return False
        return time.time() - self.started_at > self.timeout_seconds
    
    def duration_ms(self) -> Optional[float]:
        """Get operation duration in milliseconds"""
        if self.started_at is None:
            return None
        end_time = self.completed_at or time.time()
        return (end_time - self.started_at) * 1000


class OperationQueue:
    """Priority-based operation queue with async processing"""
    
    def __init__(self, max_size: int = 1000, max_concurrent: int = 10):
        self.max_size = max_size
        self.max_concurrent = max_concurrent
        
        # Priority queue for pending operations
        self.pending_queue: List[Operation] = []
        self.pending_lock = threading.Lock()
        
        # Running operations
        self.running_operations: Dict[str, Operation] = {}
        self.running_lock = threading.Lock()
        
        # Completed operations (limited history)
        self.completed_operations: deque = deque(maxlen=1000)
        self.completed_lock = threading.Lock()
        
        # Processing control
        self.processing = False
        self.processor_task: Optional[asyncio.Task] = None
        self.operation_semaphore = asyncio.Semaphore(max_concurrent)
        
        # Statistics
        self.stats = {
            'total_operations': 0,
            'completed_operations': 0,
            'failed_operations': 0,
            'cancelled_operations': 0,
            'average_processing_time_ms': 0.0,
            'queue_size': 0,
            'running_count': 0,
        }
        
        logger.info(f"OperationQueue initialized with max_size={max_size}, max_concurrent={max_concurrent}")
    
    def enqueue(self, operation_type: str, data: Dict[str, Any], 
                priority: OperationPriority = OperationPriority.NORMAL,
                callback: Optional[Callable] = None,
                context: Optional[Dict[str, Any]] = None,
                timeout_seconds: float = 30.0) -> str:
        """Add operation to queue"""
        
        with self.pending_lock:
            if len(self.pending_queue) >= self.max_size:
                # Remove lowest priority operation if queue is full
                if self.pending_queue:
                    lowest_priority_op = min(self.pending_queue)
                    if lowest_priority_op.priority.value < priority.value:
                        self.pending_queue.remove(lowest_priority_op)
                        heapq.heapify(self.pending_queue)
                        logger.warning(f"Evicted operation {lowest_priority_op.id} to make room")
                    else:
                        raise RuntimeError("Operation queue is full and cannot accept higher priority operations")
                else:
                    raise RuntimeError("Operation queue is full")
            
            operation = Operation(
                id=str(uuid.uuid4()),
                type=operation_type,
                priority=priority,
                data=data,
                callback=callback,
                context=context,
                timeout_seconds=timeout_seconds
            )
            
            heapq.heappush(self.pending_queue, operation)
            self.stats['total_operations'] += 1
            self.stats['queue_size'] = len(self.pending_queue)
            
            logger.debug(f"Enqueued operation {operation.id} with priority {priority.name}")
            return operation.id
    
    def get_operation_status(self, operation_id: str) -> Optional[OperationStatus]:
        """Get status of an operation"""
        # Check running operations
        with self.running_lock:
            if operation_id in self.running_operations:
                return self.running_operations[operation_id].status
        
        # Check pending operations
        with self.pending_lock:
            for op in self.pending_queue:
                if op.id == operation_id:
                    return op.status
        
        # Check completed operations
        with self.completed_lock:
            for op in self.completed_operations:
                if op.id == operation_id:
                    return op.status
        
        return None
    
    def get_operation_result(self, operation_id: str) -> Optional[Any]:
        """Get result of a completed operation"""
        with self.completed_lock:
            for op in self.completed_operations:
                if op.id == operation_id:
                    return op.result
        return None
    
    def cancel_operation(self, operation_id: str) -> bool:
        """Cancel a pending or running operation"""
        # Try to cancel pending operation
        with self.pending_lock:
            for i, op in enumerate(self.pending_queue):
                if op.id == operation_id:
                    op.status = OperationStatus.CANCELLED
                    del self.pending_queue[i]
                    heapq.heapify(self.pending_queue)
                    self.stats['cancelled_operations'] += 1
                    self.stats['queue_size'] = len(self.pending_queue)
                    logger.info(f"Cancelled pending operation {operation_id}")
                    return True
        
        # Try to cancel running operation
        with self.running_lock:
            if operation_id in self.running_operations:
                op = self.running_operations[operation_id]
                op.status = OperationStatus.CANCELLED
                # Note: Actual cancellation of running task would need to be handled
                # by the operation processor
                logger.info(f"Marked running operation {operation_id} for cancellation")
                return True
        
        return False
    
    async def start_processing(self):
        """Start the operation processor"""
        if self.processing:
            logger.warning("Operation processor is already running")
            return
        
        self.processing = True
        self.processor_task = asyncio.create_task(self._process_operations())
        logger.info("Operation processor started")
    
    async def stop_processing(self):
        """Stop the operation processor"""
        if not self.processing:
            return
        
        self.processing = False
        
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all running operations
        with self.running_lock:
            for op in self.running_operations.values():
                op.status = OperationStatus.CANCELLED
        
        logger.info("Operation processor stopped")
    
    async def _process_operations(self):
        """Main operation processing loop"""
        while self.processing:
            try:
                # Get next operation from queue
                operation = self._get_next_operation()
                
                if operation is None:
                    # No operations available, wait a bit
                    await asyncio.sleep(0.1)
                    continue
                
                # Process operation with concurrency control
                async with self.operation_semaphore:
                    await self._process_single_operation(operation)
                
            except Exception as e:
                logger.error(f"Error in operation processor: {e}")
                await asyncio.sleep(1.0)
    
    def _get_next_operation(self) -> Optional[Operation]:
        """Get the next operation from the priority queue"""
        with self.pending_lock:
            if not self.pending_queue:
                return None
            
            operation = heapq.heappop(self.pending_queue)
            self.stats['queue_size'] = len(self.pending_queue)
            
            # Move to running operations
            with self.running_lock:
                self.running_operations[operation.id] = operation
                self.stats['running_count'] = len(self.running_operations)
            
            operation.status = OperationStatus.RUNNING
            operation.started_at = time.time()
            
            return operation
    
    async def _process_single_operation(self, operation: Operation):
        """Process a single operation"""
        try:
            logger.debug(f"Processing operation {operation.id} of type {operation.type}")
            
            # Simulate operation processing based on type
            result = await self._execute_operation(operation)
            
            # Mark as completed
            operation.status = OperationStatus.COMPLETED
            operation.completed_at = time.time()
            operation.result = result
            
            # Call callback if provided
            if operation.callback:
                try:
                    if asyncio.iscoroutinefunction(operation.callback):
                        await operation.callback(operation)
                    else:
                        operation.callback(operation)
                except Exception as e:
                    logger.error(f"Error in operation callback: {e}")
            
            self.stats['completed_operations'] += 1
            self._update_processing_time_stats(operation.duration_ms())
            
            logger.debug(f"Completed operation {operation.id} in {operation.duration_ms():.2f}ms")
            
        except Exception as e:
            # Mark as failed
            operation.status = OperationStatus.FAILED
            operation.completed_at = time.time()
            operation.error = str(e)
            
            # Retry if possible
            if operation.retry_count < operation.max_retries:
                operation.retry_count += 1
                operation.status = OperationStatus.PENDING
                operation.started_at = None
                operation.completed_at = None
                operation.error = None
                
                # Re-enqueue for retry
                with self.pending_lock:
                    heapq.heappush(self.pending_queue, operation)
                    self.stats['queue_size'] = len(self.pending_queue)
                
                logger.info(f"Retrying operation {operation.id} (attempt {operation.retry_count})")
            else:
                self.stats['failed_operations'] += 1
                logger.error(f"Operation {operation.id} failed: {e}")
        
        finally:
            # Move from running to completed
            with self.running_lock:
                if operation.id in self.running_operations:
                    del self.running_operations[operation.id]
                    self.stats['running_count'] = len(self.running_operations)
            
            with self.completed_lock:
                self.completed_operations.append(operation)
    
    async def _execute_operation(self, operation: Operation) -> Any:
        """Execute the actual operation logic"""
        # This is where specific operation types would be handled
        # For now, we'll simulate different operation types
        
        if operation.type == "ai_inference":
            # Simulate AI inference
            await asyncio.sleep(0.1)  # Simulate processing time
            return {"inference_result": "processed", "confidence": 0.85}
        
        elif operation.type == "kernel_request":
            # Simulate kernel request processing
            await asyncio.sleep(0.05)
            return {"decision": "allow", "reasoning": "safe operation"}
        
        elif operation.type == "complex_analysis":
            # Simulate complex analysis
            await asyncio.sleep(0.5)
            return {"analysis": "detailed_results", "recommendations": ["action1", "action2"]}
        
        else:
            # Default operation
            await asyncio.sleep(0.01)
            return {"status": "completed", "data": operation.data}
    
    def _update_processing_time_stats(self, processing_time_ms: Optional[float]):
        """Update average processing time statistics"""
        if processing_time_ms is None:
            return
        
        current_avg = self.stats['average_processing_time_ms']
        completed_ops = self.stats['completed_operations']
        
        if completed_ops == 1:
            self.stats['average_processing_time_ms'] = processing_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats['average_processing_time_ms'] = (
                alpha * processing_time_ms + (1 - alpha) * current_avg
            )
    
    def get_queue_info(self) -> Dict[str, Any]:
        """Get detailed queue information"""
        with self.pending_lock:
            pending_by_priority = {}
            for op in self.pending_queue:
                priority_name = op.priority.name
                pending_by_priority[priority_name] = pending_by_priority.get(priority_name, 0) + 1
        
        with self.running_lock:
            running_operations = [
                {
                    'id': op.id,
                    'type': op.type,
                    'priority': op.priority.name,
                    'started_at': op.started_at,
                    'duration_ms': op.duration_ms()
                }
                for op in self.running_operations.values()
            ]
        
        return {
            'pending_count': len(self.pending_queue),
            'running_count': len(self.running_operations),
            'completed_count': len(self.completed_operations),
            'pending_by_priority': pending_by_priority,
            'running_operations': running_operations,
            'max_size': self.max_size,
            'max_concurrent': self.max_concurrent,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return self.stats.copy()
    
    def clear_completed_history(self):
        """Clear completed operations history"""
        with self.completed_lock:
            self.completed_operations.clear()
        logger.info("Cleared completed operations history")