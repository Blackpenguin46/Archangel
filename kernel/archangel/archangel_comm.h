#ifndef _ARCHANGEL_COMM_H
#define _ARCHANGEL_COMM_H

#include <linux/types.h>
#include <linux/spinlock.h>
#include <linux/wait.h>
#include <linux/eventfd.h>
#include <linux/dma-mapping.h>
#include <linux/scatterlist.h>
#include <linux/mm.h>
#include <linux/mman.h>
#include <linux/file.h>
#include <linux/anon_inodes.h>

/* Communication constants */
#define ARCHANGEL_COMM_RING_SIZE 4096
#define ARCHANGEL_COMM_DMA_THRESHOLD 1024
#define ARCHANGEL_COMM_MAX_CHANNELS 16
#define ARCHANGEL_COMM_MAGIC 0x41524348  /* "ARCH" */

/* Message types */
enum archangel_msg_type {
    ARCHANGEL_MSG_AI_REQUEST = 1,
    ARCHANGEL_MSG_AI_RESPONSE,
    ARCHANGEL_MSG_SYSCALL_EVENT,
    ARCHANGEL_MSG_NETWORK_EVENT,
    ARCHANGEL_MSG_MEMORY_EVENT,
    ARCHANGEL_MSG_CONTROL,
    ARCHANGEL_MSG_STATS,
    ARCHANGEL_MSG_MAX
};

/* Message priorities */
enum archangel_msg_priority {
    ARCHANGEL_PRIORITY_LOW = 0,
    ARCHANGEL_PRIORITY_NORMAL,
    ARCHANGEL_PRIORITY_HIGH,
    ARCHANGEL_PRIORITY_CRITICAL
};

/**
 * struct archangel_msg_header - Message header for communication
 * @magic: Magic number for validation
 * @type: Message type
 * @priority: Message priority
 * @size: Total message size including header
 * @sequence: Sequence number for ordering
 * @timestamp: Timestamp when message was created
 * @flags: Message flags
 */
struct archangel_msg_header {
    u32 magic;
    u8 type;
    u8 priority;
    u16 flags;
    u32 size;
    u64 sequence;
    u64 timestamp;
} __packed;

/**
 * struct archangel_spsc_queue - Single Producer Single Consumer queue
 * @buffer: Ring buffer for messages
 * @size: Size of the ring buffer
 * @head: Producer head position (only modified by producer)
 * @tail: Consumer tail position (only modified by consumer)
 * @cached_head: Cached head for consumer (reduces cache misses)
 * @cached_tail: Cached tail for producer (reduces cache misses)
 */
struct archangel_spsc_queue {
    void *buffer;
    u32 size;
    u32 head;
    u32 tail;
    u32 cached_head;
    u32 cached_tail;
} ____cacheline_aligned;

/**
 * struct archangel_dma_buffer - DMA buffer for zero-copy transfers
 * @vaddr: Virtual address of buffer
 * @dma_addr: DMA address for hardware
 * @size: Size of buffer
 * @sg_list: Scatter-gather list
 * @nents: Number of scatter-gather entries
 * @in_use: Buffer usage flag
 */
struct archangel_dma_buffer {
    void *vaddr;
    dma_addr_t dma_addr;
    size_t size;
    struct scatterlist *sg_list;
    int nents;
    atomic_t in_use;
};

/**
 * struct archangel_dma_pool - Pool of DMA buffers
 * @buffers: Array of DMA buffers
 * @count: Number of buffers in pool
 * @next_free: Next free buffer index
 * @lock: Spinlock for pool access
 * @device: Device for DMA operations
 */
struct archangel_dma_pool {
    struct archangel_dma_buffer *buffers;
    u32 count;
    u32 next_free;
    spinlock_t lock;
    struct device *device;
};

/**
 * struct archangel_comm_channel - Communication channel
 * @id: Channel identifier
 * @kernel_to_user: Kernel to userspace queue
 * @user_to_kernel: Userspace to kernel queue
 * @dma_pool: DMA buffer pool for large transfers
 * @kernel_eventfd: Event notification for kernel events
 * @user_eventfd: Event notification for userspace events
 * @shared_vma: Shared memory VMA
 * @shared_pages: Shared memory pages
 * @shared_size: Size of shared memory
 * @sequence_counter: Message sequence counter
 * @stats: Channel statistics
 * @lock: Channel lock
 * @active: Channel active flag
 */
struct archangel_comm_channel {
    u32 id;
    
    /* Lock-free SPSC queues */
    struct archangel_spsc_queue kernel_to_user;
    struct archangel_spsc_queue user_to_kernel;
    
    /* Zero-copy DMA transfers */
    struct archangel_dma_pool dma_pool;
    
    /* Event notification */
    struct eventfd_ctx *kernel_eventfd;
    struct eventfd_ctx *user_eventfd;
    
    /* Shared memory */
    struct vm_area_struct *shared_vma;
    struct page **shared_pages;
    size_t shared_size;
    
    /* Synchronization */
    atomic64_t sequence_counter;
    
    /* Statistics */
    struct {
        atomic64_t messages_sent;
        atomic64_t messages_received;
        atomic64_t bytes_sent;
        atomic64_t bytes_received;
        atomic64_t dma_transfers;
        atomic64_t queue_full_events;
        atomic64_t errors;
    } stats;
    
    spinlock_t lock;
    bool active;
};

/**
 * struct archangel_comm_manager - Communication manager
 * @channels: Array of communication channels
 * @channel_count: Number of active channels
 * @default_channel: Default channel for communication
 * @lock: Manager lock
 * @initialized: Initialization flag
 */
struct archangel_comm_manager {
    struct archangel_comm_channel channels[ARCHANGEL_COMM_MAX_CHANNELS];
    u32 channel_count;
    u32 default_channel;
    spinlock_t lock;
    bool initialized;
};

/* Global communication manager */
extern struct archangel_comm_manager *archangel_comm_mgr;

/* Communication manager functions */
int archangel_comm_manager_init(void);
void archangel_comm_manager_cleanup(void);

/* Channel management */
int archangel_comm_channel_create(u32 *channel_id);
void archangel_comm_channel_destroy(u32 channel_id);
struct archangel_comm_channel *archangel_comm_channel_get(u32 channel_id);

/* SPSC queue operations */
int archangel_spsc_queue_init(struct archangel_spsc_queue *queue, u32 size);
void archangel_spsc_queue_cleanup(struct archangel_spsc_queue *queue);
int archangel_spsc_queue_push(struct archangel_spsc_queue *queue, const void *data, u32 size);
int archangel_spsc_queue_pop(struct archangel_spsc_queue *queue, void *data, u32 max_size);
bool archangel_spsc_queue_empty(struct archangel_spsc_queue *queue);
bool archangel_spsc_queue_full(struct archangel_spsc_queue *queue);
u32 archangel_spsc_queue_available_space(struct archangel_spsc_queue *queue);

/* DMA operations */
int archangel_dma_pool_init(struct archangel_dma_pool *pool, struct device *dev, u32 buffer_count, size_t buffer_size);
void archangel_dma_pool_cleanup(struct archangel_dma_pool *pool);
struct archangel_dma_buffer *archangel_dma_buffer_alloc(struct archangel_dma_pool *pool);
void archangel_dma_buffer_free(struct archangel_dma_pool *pool, struct archangel_dma_buffer *buffer);

/* Message operations */
int archangel_comm_send_message(u32 channel_id, enum archangel_msg_type type, 
                               enum archangel_msg_priority priority, 
                               const void *data, u32 size);
int archangel_comm_receive_message(u32 channel_id, enum archangel_msg_type *type,
                                  void *data, u32 max_size);

/* Event notification */
int archangel_comm_notify_kernel(u32 channel_id);
int archangel_comm_notify_user(u32 channel_id);

/* Shared memory operations */
int archangel_comm_setup_shared_memory(u32 channel_id, size_t size);
void archangel_comm_cleanup_shared_memory(u32 channel_id);

/* Statistics and monitoring */
void archangel_comm_update_stats(u32 channel_id, const char *event, u64 value);
void archangel_comm_get_stats(u32 channel_id, struct seq_file *m);

/* Utility functions */
static inline bool archangel_comm_is_initialized(void)
{
    return archangel_comm_mgr && archangel_comm_mgr->initialized;
}

static inline u64 archangel_comm_get_timestamp(void)
{
    return ktime_get_ns();
}

static inline u32 archangel_comm_next_power_of_2(u32 n)
{
    return 1U << (32 - __builtin_clz(n - 1));
}

#endif /* _ARCHANGEL_COMM_H */