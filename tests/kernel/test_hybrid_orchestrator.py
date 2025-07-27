#!/usr/bin/env python3
"""
Test script for HybridAIOrchestrator

Tests the kernel-userspace communication bridge, AI model management,
decision caching, and operation queue functionality.
"""

import asyncio
import logging
import sys
import os
import time
from typing import Dict, Any

# Add the opt/archangel directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'opt', 'archangel'))

from ai.orchestrator import (
    HybridAIOrchestrator, 
    KernelRequest, 
    AIEngineType, 
    MessagePriority,
    create_orchestrator
)
from ai.models import ModelType, InferenceRequest
from ai.cache import CacheKeyGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_orchestrator_initialization():
    """Test orchestrator initialization"""
    logger.info("Testing orchestrator initialization...")
    
    config = {
        'cache_size': 1000,
        'cache_ttl': 300,
        'queue_size': 500,
        'models': {
            'llm_planner': {
                'type': 'llm_planner',
                'path': '/opt/archangel/models/codellama-13b',
                'enabled': True
            }
        }
    }
    
    orchestrator = create_orchestrator(config)
    
    # Initialize orchestrator
    success = await orchestrator.initialize()
    assert success, "Orchestrator initialization failed"
    
    # Check that components are initialized
    assert orchestrator.kernel_bridge is not None
    assert orchestrator.model_manager is not None
    assert orchestrator.decision_cache is not None
    assert orchestrator.operation_queue is not None
    
    logger.info("✓ Orchestrator initialization test passed")
    
    await orchestrator.shutdown()
    return True


async def test_kernel_request_processing():
    """Test processing of kernel requests"""
    logger.info("Testing kernel request processing...")
    
    orchestrator = create_orchestrator()
    await orchestrator.initialize()
    
    # Create test kernel requests
    test_requests = [
        KernelRequest(
            engine_type=AIEngineType.SYSCALL,
            request_id="test_syscall_1",
            timestamp=time.time(),
            priority=MessagePriority.NORMAL,
            data={
                'syscall_num': 1,  # read
                'process_name': 'test_process',
                'args': [0, 'buffer', 1024]
            }
        ),
        KernelRequest(
            engine_type=AIEngineType.NETWORK,
            request_id="test_network_1",
            timestamp=time.time(),
            priority=MessagePriority.HIGH,
            data={
                'src_ip': '192.168.1.100',
                'dst_ip': '192.168.1.1',
                'dst_port': 80,
                'protocol': 'tcp'
            }
        ),
        KernelRequest(
            engine_type=AIEngineType.MEMORY,
            request_id="test_memory_1",
            timestamp=time.time(),
            priority=MessagePriority.CRITICAL,
            data={
                'address': 0x7fff12345000,
                'access_type': 'write',
                'size': 8
            }
        )
    ]
    
    # Process requests and verify responses
    for request in test_requests:
        decision = await orchestrator.process_kernel_request(request)
        
        assert decision is not None, f"No decision for request {request.request_id}"
        assert decision.decision is not None, "Decision value is None"
        assert decision.confidence > 0, "Confidence should be positive"
        assert decision.reasoning, "Reasoning should be provided"
        
        logger.info(f"✓ Request {request.request_id}: {decision.decision} (confidence: {decision.confidence:.2f})")
    
    logger.info("✓ Kernel request processing test passed")
    
    await orchestrator.shutdown()
    return True


async def test_decision_caching():
    """Test decision caching functionality"""
    logger.info("Testing decision caching...")
    
    orchestrator = create_orchestrator()
    await orchestrator.initialize()
    
    # Create identical requests to test caching
    request1 = KernelRequest(
        engine_type=AIEngineType.SYSCALL,
        request_id="cache_test_1",
        timestamp=time.time(),
        priority=MessagePriority.NORMAL,
        data={
            'syscall_num': 2,  # write
            'process_name': 'cache_test',
            'args': [1, 'data', 4]
        }
    )
    
    request2 = KernelRequest(
        engine_type=AIEngineType.SYSCALL,
        request_id="cache_test_2",
        timestamp=time.time(),
        priority=MessagePriority.NORMAL,
        data={
            'syscall_num': 2,  # write (same as request1)
            'process_name': 'cache_test',
            'args': [1, 'data', 4]
        }
    )
    
    # Process first request (should be cache miss)
    start_stats = orchestrator.stats.copy()
    decision1 = await orchestrator.process_kernel_request(request1)
    
    # Process second request (should be cache hit)
    decision2 = await orchestrator.process_kernel_request(request2)
    
    # Verify caching worked
    assert orchestrator.stats['cache_hits'] > start_stats['cache_hits'], "Cache hit not recorded"
    assert decision1.decision == decision2.decision, "Cached decisions should be identical"
    
    # Test cache statistics
    cache_stats = orchestrator.decision_cache.get_statistics()
    assert cache_stats['overall']['total_hits'] > 0, "No cache hits recorded"
    
    logger.info("✓ Decision caching test passed")
    
    await orchestrator.shutdown()
    return True


async def test_ai_model_integration():
    """Test AI model integration"""
    logger.info("Testing AI model integration...")
    
    orchestrator = create_orchestrator()
    await orchestrator.initialize()
    
    # Test different model types
    model_tests = [
        (ModelType.LLM_PLANNER, "Plan a penetration test for target network"),
        (ModelType.ANALYZER, {"vulnerability": "SQL injection", "severity": "high"}),
        (ModelType.GENERATOR, {"type": "exploit", "target": "web_app"}),
        (ModelType.REPORTER, {"findings": ["vuln1", "vuln2"], "format": "executive"})
    ]
    
    for model_type, test_data in model_tests:
        request = InferenceRequest(
            model_type=model_type,
            input_data=test_data,
            priority=2
        )
        
        result = await orchestrator.model_manager.infer(request)
        
        assert result is not None, f"No result from {model_type.value} model"
        assert result.output is not None, f"No output from {model_type.value} model"
        assert result.confidence > 0, f"Invalid confidence from {model_type.value} model"
        
        logger.info(f"✓ {model_type.value} model test passed (confidence: {result.confidence:.2f})")
    
    logger.info("✓ AI model integration test passed")
    
    await orchestrator.shutdown()
    return True


async def test_operation_queue():
    """Test operation queue functionality"""
    logger.info("Testing operation queue...")
    
    orchestrator = create_orchestrator()
    await orchestrator.initialize()
    
    # Start queue processing
    await orchestrator.operation_queue.start_processing()
    
    # Enqueue test operations
    operation_ids = []
    for i in range(5):
        op_id = orchestrator.operation_queue.enqueue(
            operation_type="test_operation",
            data={"test_id": i},
            priority=MessagePriority.NORMAL if i % 2 == 0 else MessagePriority.HIGH
        )
        operation_ids.append(op_id)
    
    # Wait for operations to complete
    await asyncio.sleep(1.0)
    
    # Check operation results
    completed_count = 0
    for op_id in operation_ids:
        status = orchestrator.operation_queue.get_operation_status(op_id)
        if status and status.value == "completed":
            completed_count += 1
    
    assert completed_count > 0, "No operations completed"
    
    # Test queue statistics
    queue_stats = orchestrator.operation_queue.get_statistics()
    assert queue_stats['total_operations'] >= 5, "Operations not recorded"
    
    logger.info(f"✓ Operation queue test passed ({completed_count} operations completed)")
    
    await orchestrator.operation_queue.stop_processing()
    await orchestrator.shutdown()
    return True


async def test_performance_monitoring():
    """Test performance monitoring"""
    logger.info("Testing performance monitoring...")
    
    orchestrator = create_orchestrator()
    await orchestrator.initialize()
    
    # Generate some load
    requests = []
    for i in range(10):
        request = KernelRequest(
            engine_type=AIEngineType.SYSCALL,
            request_id=f"perf_test_{i}",
            timestamp=time.time(),
            priority=MessagePriority.NORMAL,
            data={'syscall_num': i % 10, 'process_name': f'test_{i}'}
        )
        requests.append(request)
    
    # Process requests
    start_time = time.time()
    for request in requests:
        await orchestrator.process_kernel_request(request)
    end_time = time.time()
    
    # Check performance statistics
    stats = orchestrator.get_statistics()
    
    assert stats['requests_processed'] >= 10, "Not all requests processed"
    assert stats['average_response_time_ms'] > 0, "No response time recorded"
    
    total_time_ms = (end_time - start_time) * 1000
    avg_time_per_request = total_time_ms / len(requests)
    
    logger.info(f"✓ Performance test passed:")
    logger.info(f"  - Total requests: {stats['requests_processed']}")
    logger.info(f"  - Average response time: {stats['average_response_time_ms']:.2f}ms")
    logger.info(f"  - Cache hit rate: {stats['cache_hits']}/{stats['cache_hits'] + stats['cache_misses']}")
    logger.info(f"  - Fast path decisions: {stats['fast_path_decisions']}")
    
    await orchestrator.shutdown()
    return True


async def test_cache_key_generation():
    """Test cache key generation utilities"""
    logger.info("Testing cache key generation...")
    
    # Test syscall key generation
    key1 = CacheKeyGenerator.generate_syscall_key(1, "test_process", [0, "buffer", 1024])
    key2 = CacheKeyGenerator.generate_syscall_key(1, "test_process", [0, "buffer", 1024])
    key3 = CacheKeyGenerator.generate_syscall_key(2, "test_process", [0, "buffer", 1024])
    
    assert key1 == key2, "Identical syscall data should generate same key"
    assert key1 != key3, "Different syscall data should generate different keys"
    
    # Test network key generation
    net_key1 = CacheKeyGenerator.generate_network_key("192.168.1.1", "192.168.1.2", 80, "tcp")
    net_key2 = CacheKeyGenerator.generate_network_key("192.168.1.1", "192.168.1.2", 80, "tcp")
    net_key3 = CacheKeyGenerator.generate_network_key("192.168.1.1", "192.168.1.2", 443, "tcp")
    
    assert net_key1 == net_key2, "Identical network data should generate same key"
    assert net_key1 != net_key3, "Different network data should generate different keys"
    
    logger.info("✓ Cache key generation test passed")
    return True


async def run_all_tests():
    """Run all tests"""
    logger.info("Starting HybridAIOrchestrator tests...")
    
    tests = [
        test_cache_key_generation,
        test_orchestrator_initialization,
        test_kernel_request_processing,
        test_decision_caching,
        test_ai_model_integration,
        test_operation_queue,
        test_performance_monitoring,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} failed: {e}")
            failed += 1
    
    logger.info(f"\nTest Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)