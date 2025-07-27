# Archangel Kernel-Userspace Communication Bridge

## Overview

The Archangel communication bridge provides high-performance, low-latency communication between kernel AI modules and userspace components. It implements a hybrid architecture with shared memory ring buffers, zero-copy DMA transfers, and event-driven notifications.

## Architecture

### Core Components

1. **Communication Manager** (`archangel_comm_manager`)
   - Manages multiple communication channels
   - Handles channel lifecycle and resource allocation
   - Provides unified interface for kernel-userspace communication

2. **Communication Channels** (`archangel_comm_channel`)
   - Bidirectional communication paths
   - Each channel contains SPSC queues, DMA pools, and event notification
   - Support for up to 16 concurrent channels

3. **SPSC Queues** (`archangel_spsc_queue`)
   - Lock-free Single Producer Single Consumer queues
   - Optimized for high-throughput, low-latency messaging
   - Automatic wrap-around handling and cache optimization

4. **DMA Buffer Pool** (`archangel_dma_pool`)
   - Zero-copy transfers for large data (>1KB)
   - Pre-allocated coherent DMA buffers
   - Automatic buffer management and recycling

5. **Event Notification System**
   - eventfd-based signaling for microsecond latency
   - Asynchronous notification between kernel and userspace
   - Integration with kernel wait queues

## Performance Characteristics

- **Throughput**: >1M messages/sec kernel-userspace
- **Latency**: <10μs for small messages, <100μs for DMA transfers
- **Memory Usage**: <1MB per channel for ring buffers and DMA pool
- **CPU Overhead**: <1% for typical workloads

## Key Features

### Lock-Free SPSC Queues
- Single producer, single consumer design eliminates locking overhead
- Cache-optimized with separate cachelines for producer and consumer
- Automatic power-of-2 sizing for efficient modulo operations
- Support for variable-length messages with size headers

### Zero-Copy DMA Transfers
- Automatic DMA usage for messages >1KB threshold
- Coherent DMA memory allocation for hardware compatibility
- Scatter-gather list support for complex data structures
- Buffer pooling to minimize allocation overhead

### Event-Driven Notifications
- eventfd integration for userspace notification
- Kernel wait queue integration for kernel-side blocking
- Low-latency signaling with memory barriers
- Support for both blocking and non-blocking operations

### Shared Memory Support
- Optional shared memory regions for bulk data transfer
- Page-aligned allocation with proper memory management
- Integration with kernel VMA subsystem
- Support for memory mapping to userspace

## Message Format

### Message Header
```c
struct archangel_msg_header {
    u32 magic;          // Magic number (0x41524348 "ARCH")
    u8 type;            // Message type (AI_REQUEST, AI_RESPONSE, etc.)
    u8 priority;        // Message priority (LOW, NORMAL, HIGH, CRITICAL)
    u16 flags;          // Message flags (DMA, etc.)
    u32 size;           // Total message size including header
    u64 sequence;       // Sequence number for ordering
    u64 timestamp;      // Timestamp when message was created
} __packed;
```

### Message Types
- `ARCHANGEL_MSG_AI_REQUEST`: AI inference requests
- `ARCHANGEL_MSG_AI_RESPONSE`: AI inference responses
- `ARCHANGEL_MSG_SYSCALL_EVENT`: System call events
- `ARCHANGEL_MSG_NETWORK_EVENT`: Network packet events
- `ARCHANGEL_MSG_MEMORY_EVENT`: Memory access events
- `ARCHANGEL_MSG_CONTROL`: Control messages
- `ARCHANGEL_MSG_STATS`: Statistics updates

## API Reference

### Channel Management
```c
int archangel_comm_channel_create(u32 *channel_id);
void archangel_comm_channel_destroy(u32 channel_id);
struct archangel_comm_channel *archangel_comm_channel_get(u32 channel_id);
```

### Message Operations
```c
int archangel_comm_send_message(u32 channel_id, enum archangel_msg_type type,
                               enum archangel_msg_priority priority,
                               const void *data, u32 size);

int archangel_comm_receive_message(u32 channel_id, enum archangel_msg_type *type,
                                  void *data, u32 max_size);
```

### Event Notification
```c
int archangel_comm_notify_kernel(u32 channel_id);
int archangel_comm_notify_user(u32 channel_id);
```

### Shared Memory
```c
int archangel_comm_setup_shared_memory(u32 channel_id, size_t size);
void archangel_comm_cleanup_shared_memory(u32 channel_id);
```

## Usage Examples

### Kernel Side - Sending AI Request
```c
struct ai_request req = {
    .operation = AI_CLASSIFY_PACKET,
    .data_size = packet_size,
    .priority = HIGH_PRIORITY
};

int ret = archangel_comm_send_message(0, ARCHANGEL_MSG_AI_REQUEST,
                                     ARCHANGEL_PRIORITY_HIGH,
                                     &req, sizeof(req));
if (ret < 0) {
    pr_err("Failed to send AI request: %d\n", ret);
}
```

### Kernel Side - Receiving AI Response
```c
struct ai_response resp;
enum archangel_msg_type type;

int ret = archangel_comm_receive_message(0, &type, &resp, sizeof(resp));
if (ret > 0 && type == ARCHANGEL_MSG_AI_RESPONSE) {
    /* Process AI response */
    handle_ai_response(&resp);
}
```

## Statistics and Monitoring

The communication bridge provides comprehensive statistics through the proc filesystem:

- `/proc/archangel/comm` - Communication statistics
- Per-channel message counts, byte counts, error rates
- Queue utilization and DMA transfer statistics
- Event notification performance metrics

## Error Handling

### Queue Full Conditions
- Automatic detection of queue full conditions
- Statistics tracking for queue full events
- Graceful degradation with error reporting

### DMA Allocation Failures
- Fallback to regular message queuing
- Buffer pool management with recycling
- Error statistics and monitoring

### Communication Failures
- Robust error detection and reporting
- Automatic channel recovery mechanisms
- Comprehensive error statistics

## Integration with Core Module

The communication bridge integrates seamlessly with the Archangel core module:

1. **Initialization**: Called during core module initialization
2. **AI Engine Integration**: Provides communication channels for AI engines
3. **Statistics Integration**: Unified statistics reporting
4. **Resource Management**: Integrated with core resource monitoring

## Testing

Comprehensive testing is provided through:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end communication testing
- **Performance Tests**: Throughput and latency benchmarking
- **Stress Tests**: High-load and error condition testing

## Future Enhancements

- **NUMA Awareness**: Optimize for NUMA topology
- **Hardware Acceleration**: Integration with RDMA and other high-speed interconnects
- **Compression**: Optional message compression for bandwidth optimization
- **Encryption**: Secure communication channels for sensitive data