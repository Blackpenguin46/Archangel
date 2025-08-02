/*
 * Archangel Linux - Kernel-Userspace Communication
 * High-speed shared memory communication bridge with ring buffers and DMA
 * Optimized for <1ms response times
 */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/mman.h>
#include <linux/vmalloc.h>
#include <linux/proc_fs.h>
#include <linux/mutex.h>
#include <linux/spinlock.h>
#include <linux/wait.h>
#include <linux/sched.h>
#include <linux/atomic.h>
#include <linux/ktime.h>
#include <linux/uaccess.h>
#include <linux/dma-mapping.h>
#include <linux/workqueue.h>
#include <linux/interrupt.h>
#include <linux/cache.h>

#include "../include/archangel.h"

/* Performance optimization constants */
#define ARCHANGEL_RING_BUFFER_SIZE      (64 * 1024)    /* 64KB ring buffers */
#define ARCHANGEL_MAX_BATCH_SIZE        32              /* Max messages per batch */
#define ARCHANGEL_CACHE_LINE_SIZE       L1_CACHE_BYTES  /* CPU cache line alignment */
#define ARCHANGEL_PREFETCH_DISTANCE     8               /* Cache prefetch distance */
#define ARCHANGEL_DMA_COHERENT_SIZE     (1024 * 1024)   /* 1MB DMA coherent buffer */

/* Lock-free ring buffer structure (optimized for performance) */
struct archangel_ring_buffer {
    /* Producer/consumer indices (cache-line aligned) */
    volatile u32 producer_index ____cacheline_aligned;
    volatile u32 consumer_index ____cacheline_aligned;
    
    /* Buffer metadata */
    u32 buffer_size;
    u32 buffer_mask;    /* size - 1, for fast modulo */
    
    /* Performance counters */
    atomic64_t total_messages;
    atomic64_t dropped_messages;
    atomic64_t batch_count;
    
    /* Buffer data (cache-line aligned) */
    struct archangel_message buffer[0] ____cacheline_aligned;
};

/* High-performance decision cache */
struct archangel_decision_cache {
    /* Cache entries (hash table) */
    struct {
        u64 context_hash;
        enum archangel_decision decision;
        u64 timestamp;
        atomic_t access_count;
    } entries[4096] ____cacheline_aligned;
    
    /* Cache statistics */
    atomic64_t cache_hits;
    atomic64_t cache_misses;
    atomic64_t cache_evictions;
    
    /* Hash seed for cache lookups */
    u32 hash_seed;
} ____cacheline_aligned;

/* Enhanced communication state with performance optimizations */
static struct {
    bool initialized;
    
    /* High-performance ring buffers */
    struct archangel_ring_buffer *kernel_to_user_ring;
    struct archangel_ring_buffer *user_to_kernel_ring;
    void *ring_buffer_memory;
    size_t ring_buffer_total_size;
    
    /* DMA coherent memory for zero-copy operations */
    void *dma_coherent_buffer;
    dma_addr_t dma_handle;
    size_t dma_buffer_size;
    
    /* Decision cache for <1ms responses */
    struct archangel_decision_cache *decision_cache;
    
    /* Shared memory (legacy compatibility) */
    struct archangel_shared_memory *shared_mem;
    unsigned long shared_mem_size;
    struct page **shared_pages;
    int num_pages;
    
    /* Performance statistics */
    atomic64_t messages_sent;
    atomic64_t messages_received;
    atomic64_t ring_buffer_overruns;
    atomic64_t zero_copy_operations;
    atomic64_t fast_path_decisions;
    atomic64_t avg_response_time_ns;
    atomic64_t min_response_time_ns;
    atomic64_t max_response_time_ns;
    
    /* High-resolution timing */
    ktime_t last_message_time;
    atomic64_t message_rate_per_sec;
    
    /* Work queue for deferred processing */
    struct workqueue_struct *comm_workqueue;
    struct work_struct batch_work;
    
    /* Synchronization (minimal locking) */
    struct mutex comm_lock;
    
} comm_state ____cacheline_aligned = {
    .initialized = false,
    .comm_lock = __MUTEX_INITIALIZER(comm_state.comm_lock),
};

/* Forward declarations */
static int alloc_shared_memory(void);
static void free_shared_memory(void);
static int init_ring_buffers(void);
static void cleanup_ring_buffers(void);
static int init_decision_cache(void);
static void cleanup_decision_cache(void);
static int init_dma_buffers(void);
static void cleanup_dma_buffers(void);

/* High-performance ring buffer operations */
static inline int ring_buffer_produce(struct archangel_ring_buffer *ring, 
                                     const struct archangel_message *msg);
static inline int ring_buffer_consume(struct archangel_ring_buffer *ring, 
                                     struct archangel_message *msg);
static inline bool ring_buffer_is_empty(struct archangel_ring_buffer *ring);
static inline bool ring_buffer_is_full(struct archangel_ring_buffer *ring);

/* Decision cache operations */
static inline enum archangel_decision cache_lookup_decision(
    struct archangel_security_context *ctx);
static inline void cache_store_decision(struct archangel_security_context *ctx, 
                                       enum archangel_decision decision);
static inline u64 hash_security_context(struct archangel_security_context *ctx);

/* Batch processing */
static void batch_process_work(struct work_struct *work);
static int process_message_batch(struct archangel_message *messages, int count);

/* Legacy queue operations (for compatibility) */
static int enqueue_message(struct archangel_message *queue_base, volatile u32 *head, 
                          volatile u32 *tail, u32 queue_size, spinlock_t *lock,
                          const struct archangel_message *msg);
static int dequeue_message(struct archangel_message *queue_base, volatile u32 *head,
                          volatile u32 *tail, u32 queue_size, spinlock_t *lock,
                          struct archangel_message *msg);

/* Communication device file operations */
static int archangel_comm_open(struct inode *inode, struct file *file);
static int archangel_comm_release(struct inode *inode, struct file *file);
static long archangel_comm_ioctl(struct file *file, unsigned int cmd, unsigned long arg);
static int archangel_comm_mmap(struct file *file, struct vm_area_struct *vma);

static const struct proc_ops archangel_comm_fops = {
    .proc_open = archangel_comm_open,
    .proc_release = archangel_comm_release,
    .proc_ioctl = archangel_comm_ioctl,
    .proc_mmap = archangel_comm_mmap,
};

static struct proc_dir_entry *comm_proc_entry = NULL;

/*
 * Initialize enhanced kernel-userspace communication
 */
int archangel_init_communication(void)
{
    int ret;
    
    archangel_info("Initializing enhanced kernel-userspace communication");
    
    mutex_lock(&comm_state.comm_lock);
    
    if (comm_state.initialized) {
        mutex_unlock(&comm_state.comm_lock);
        return 0;
    }
    
    /* Initialize high-performance ring buffers */
    ret = init_ring_buffers();
    if (ret) {
        archangel_err("Failed to initialize ring buffers: %d", ret);
        goto error;
    }
    
    /* Initialize decision cache for fast responses */
    ret = init_decision_cache();
    if (ret) {
        archangel_err("Failed to initialize decision cache: %d", ret);
        goto error_ring_buffers;
    }
    
    /* Initialize DMA coherent buffers for zero-copy operations */
    ret = init_dma_buffers();
    if (ret) {
        archangel_warn("Failed to initialize DMA buffers: %d (continuing without DMA)", ret);
        /* Continue without DMA - not critical */
    }
    
    /* Allocate shared memory (legacy compatibility) */
    ret = alloc_shared_memory();
    if (ret) {
        archangel_err("Failed to allocate shared memory: %d", ret);
        goto error_dma;
    }
    
    /* Create high-priority workqueue for batch processing */
    comm_state.comm_workqueue = alloc_workqueue("archangel_comm", 
                                               WQ_HIGHPRI | WQ_CPU_INTENSIVE, 1);
    if (!comm_state.comm_workqueue) {
        archangel_err("Failed to create communication workqueue");
        ret = -ENOMEM;
        goto error_shared_mem;
    }
    
    INIT_WORK(&comm_state.batch_work, batch_process_work);
    
    /* Create communication device in proc */
    comm_proc_entry = proc_create("archangel_comm", 0666, NULL, &archangel_comm_fops);
    if (!comm_proc_entry) {
        archangel_err("Failed to create communication proc entry");
        ret = -ENOMEM;
        goto error_workqueue;
    }
    
    /* Initialize performance statistics */
    atomic64_set(&comm_state.messages_sent, 0);
    atomic64_set(&comm_state.messages_received, 0);
    atomic64_set(&comm_state.ring_buffer_overruns, 0);
    atomic64_set(&comm_state.zero_copy_operations, 0);
    atomic64_set(&comm_state.fast_path_decisions, 0);
    atomic64_set(&comm_state.avg_response_time_ns, 0);
    atomic64_set(&comm_state.min_response_time_ns, UINT64_MAX);
    atomic64_set(&comm_state.max_response_time_ns, 0);
    atomic64_set(&comm_state.message_rate_per_sec, 0);
    
    comm_state.last_message_time = ktime_get();
    comm_state.initialized = true;
    mutex_unlock(&comm_state.comm_lock);
    
    archangel_info("Enhanced communication initialized: ring buffers, decision cache, %s",
                   comm_state.dma_coherent_buffer ? "DMA enabled" : "DMA disabled");
    
    return 0;

error_workqueue:
    destroy_workqueue(comm_state.comm_workqueue);
error_shared_mem:
    free_shared_memory();
error_dma:
    cleanup_dma_buffers();
    cleanup_decision_cache();
error_ring_buffers:
    cleanup_ring_buffers();
error:
    mutex_unlock(&comm_state.comm_lock);
    return ret;
}

/*
 * Cleanup kernel-userspace communication
 */
void archangel_cleanup_communication(void)
{
    archangel_info("Cleaning up kernel-userspace communication");
    
    mutex_lock(&comm_state.comm_lock);
    
    if (!comm_state.initialized) {
        mutex_unlock(&comm_state.comm_lock);
        return;
    }
    
    /* Remove proc entry */
    if (comm_proc_entry) {
        proc_remove(comm_proc_entry);
        comm_proc_entry = NULL;
    }
    
    /* Cleanup message queues */
    cleanup_message_queues();
    
    /* Free shared memory */
    free_shared_memory();
    
    comm_state.initialized = false;
    mutex_unlock(&comm_state.comm_lock);
    
    archangel_info("Communication cleanup complete, sent %lld messages, received %lld",
                   atomic64_read(&comm_state.messages_sent),
                   atomic64_read(&comm_state.messages_received));
}

/*
 * High-performance send message to userspace with decision cache
 */
int archangel_send_message(enum archangel_msg_type type, const void *data, u32 size)
{
    struct archangel_message msg;
    ktime_t start_time, end_time;
    u64 response_time_ns;
    int ret;
    
    if (!comm_state.initialized)
        return -ENOTCONN;
    
    if (size > ARCHANGEL_MAX_MSG_SIZE - sizeof(struct archangel_message))
        return -EMSGSIZE;
    
    start_time = ktime_get();
    
    /* Fast path: Check decision cache for analysis requests */
    if (type == ARCHANGEL_MSG_ANALYSIS_REQUEST && data && size >= sizeof(struct archangel_security_context)) {
        struct archangel_security_context *ctx = (struct archangel_security_context *)data;
        enum archangel_decision cached_decision = cache_lookup_decision(ctx);
        
        if (cached_decision != ARCHANGEL_UNKNOWN) {
            atomic64_inc(&comm_state.fast_path_decisions);
            
            /* Return cached decision immediately - sub-microsecond response */
            end_time = ktime_get();
            response_time_ns = ktime_to_ns(ktime_sub(end_time, start_time));
            
            /* Update timing statistics */
            atomic64_set(&comm_state.avg_response_time_ns, 
                        (atomic64_read(&comm_state.avg_response_time_ns) + response_time_ns) / 2);
            
            if (response_time_ns < atomic64_read(&comm_state.min_response_time_ns))
                atomic64_set(&comm_state.min_response_time_ns, response_time_ns);
            
            if (response_time_ns > atomic64_read(&comm_state.max_response_time_ns))
                atomic64_set(&comm_state.max_response_time_ns, response_time_ns);
            
            archangel_debug("Fast path decision for PID %u: %d (%llu ns)", 
                           ctx->pid, cached_decision, response_time_ns);
            
            return cached_decision; /* Return decision directly */
        }
    }
    
    /* Prepare message */
    memset(&msg, 0, sizeof(msg));
    msg.type = type;
    msg.sequence = atomic64_inc_return(&comm_state.messages_sent);
    msg.data_size = size;
    msg.timestamp = ktime_get_ns();
    msg.flags = 0;
    
    /* Copy data directly into message */
    if (data && size > 0) {
        if (size <= sizeof(msg.data)) {
            memcpy(msg.data, data, size);
        } else {
            /* Use DMA buffer for large messages if available */
            if (comm_state.dma_coherent_buffer && size <= comm_state.dma_buffer_size) {
                memcpy(comm_state.dma_coherent_buffer, data, size);
                msg.flags |= 0x0001; /* DMA flag */
                atomic64_inc(&comm_state.zero_copy_operations);
            } else {
                memcpy(msg.data, data, min_t(u32, size, sizeof(msg.data)));
            }
        }
    }
    
    /* Use high-performance ring buffer */
    if (comm_state.kernel_to_user_ring) {
        ret = ring_buffer_produce(comm_state.kernel_to_user_ring, &msg);
        if (ret != 0) {
            atomic64_inc(&comm_state.ring_buffer_overruns);
        }
    } else {
        /* Fallback to legacy queue (should not happen in normal operation) */
        ret = -ENOSYS;
    }
    
    if (ret == 0) {
        /* Update message rate tracking */
        ktime_t current_time = ktime_get();
        u64 time_diff_ns = ktime_to_ns(ktime_sub(current_time, comm_state.last_message_time));
        if (time_diff_ns > 1000000000ULL) { /* 1 second */
            atomic64_set(&comm_state.message_rate_per_sec, 
                        atomic64_read(&comm_state.messages_sent));
            comm_state.last_message_time = current_time;
        }
        
        archangel_debug("Sent message type %d, size %u via ring buffer", type, size);
    } else {
        archangel_warn("Failed to send message: ring buffer full");
    }
    
    /* Update timing statistics */
    end_time = ktime_get();
    response_time_ns = ktime_to_ns(ktime_sub(end_time, start_time));
    
    atomic64_set(&comm_state.avg_response_time_ns, 
                (atomic64_read(&comm_state.avg_response_time_ns) + response_time_ns) / 2);
    
    if (response_time_ns < atomic64_read(&comm_state.min_response_time_ns))
        atomic64_set(&comm_state.min_response_time_ns, response_time_ns);
    
    if (response_time_ns > atomic64_read(&comm_state.max_response_time_ns))
        atomic64_set(&comm_state.max_response_time_ns, response_time_ns);
    
    return ret;
}

/*
 * Receive message from userspace
 */
int archangel_receive_message(struct archangel_message **msg)
{
    struct archangel_message temp_msg;
    struct archangel_message *full_msg;
    int ret;
    
    if (!comm_state.initialized || !msg)
        return -EINVAL;
    
    /* Try to dequeue message header */
    ret = dequeue_message(comm_state.user_to_kernel_queue.messages,
                         &comm_state.user_to_kernel_queue.head,
                         &comm_state.user_to_kernel_queue.tail,
                         comm_state.user_to_kernel_queue.size,
                         &comm_state.user_to_kernel_queue.lock,
                         &temp_msg);
    
    if (ret != 0) {
        if (ret == -EAGAIN)
            atomic64_inc(&comm_state.queue_empty_errors);
        return ret;
    }
    
    /* Allocate memory for full message */
    full_msg = kmalloc(sizeof(struct archangel_message) + temp_msg.data_size, GFP_KERNEL);
    if (!full_msg)
        return -ENOMEM;
    
    /* Copy message */
    memcpy(full_msg, &temp_msg, sizeof(struct archangel_message));
    if (temp_msg.data_size > 0)
        memcpy(full_msg->data, temp_msg.data, temp_msg.data_size);
    
    atomic64_inc(&comm_state.messages_received);
    
    archangel_debug("Received message type %d, size %u from userspace", 
                   full_msg->type, full_msg->data_size);
    
    *msg = full_msg;
    return 0;
}

/*
 * Allocate shared memory
 */
static int alloc_shared_memory(void)
{
    int i;
    
    comm_state.shared_mem_size = ARCHANGEL_SHARED_MEM_SIZE;
    comm_state.num_pages = (comm_state.shared_mem_size + PAGE_SIZE - 1) / PAGE_SIZE;
    
    /* Allocate page array */
    comm_state.shared_pages = kmalloc(comm_state.num_pages * sizeof(struct page *), 
                                     GFP_KERNEL);
    if (!comm_state.shared_pages)
        return -ENOMEM;
    
    /* Allocate pages */
    for (i = 0; i < comm_state.num_pages; i++) {
        comm_state.shared_pages[i] = alloc_page(GFP_KERNEL | __GFP_ZERO);
        if (!comm_state.shared_pages[i]) {
            /* Cleanup allocated pages */
            while (--i >= 0)
                __free_page(comm_state.shared_pages[i]);
            kfree(comm_state.shared_pages);
            return -ENOMEM;
        }
    }
    
    /* Map pages to virtual memory */
    comm_state.shared_mem = vmap(comm_state.shared_pages, comm_state.num_pages, 
                                VM_MAP, PAGE_KERNEL);
    if (!comm_state.shared_mem) {
        /* Cleanup pages */
        for (i = 0; i < comm_state.num_pages; i++)
            __free_page(comm_state.shared_pages[i]);
        kfree(comm_state.shared_pages);
        return -ENOMEM;
    }
    
    memset(comm_state.shared_mem, 0, comm_state.shared_mem_size);
    
    archangel_debug("Allocated %d pages (%lu bytes) for shared memory", 
                   comm_state.num_pages, comm_state.shared_mem_size);
    
    return 0;
}

/*
 * Free shared memory
 */
static void free_shared_memory(void)
{
    int i;
    
    if (comm_state.shared_mem) {
        vunmap(comm_state.shared_mem);
        comm_state.shared_mem = NULL;
    }
    
    if (comm_state.shared_pages) {
        for (i = 0; i < comm_state.num_pages; i++) {
            if (comm_state.shared_pages[i])
                __free_page(comm_state.shared_pages[i]);
        }
        kfree(comm_state.shared_pages);
        comm_state.shared_pages = NULL;
    }
    
    comm_state.num_pages = 0;
    comm_state.shared_mem_size = 0;
}

/*
 * Initialize message queues
 */
static int init_message_queues(void)
{
    u32 queue_size = 1024; /* Number of messages per queue */
    
    /* Initialize kernel to user queue */
    comm_state.kernel_to_user_queue.messages = 
        kmalloc(queue_size * sizeof(struct archangel_message), GFP_KERNEL);
    if (!comm_state.kernel_to_user_queue.messages)
        return -ENOMEM;
    
    comm_state.kernel_to_user_queue.head = 0;
    comm_state.kernel_to_user_queue.tail = 0;
    comm_state.kernel_to_user_queue.size = queue_size;
    spin_lock_init(&comm_state.kernel_to_user_queue.lock);
    init_waitqueue_head(&comm_state.kernel_to_user_queue.wait_queue);
    
    /* Initialize user to kernel queue */
    comm_state.user_to_kernel_queue.messages = 
        kmalloc(queue_size * sizeof(struct archangel_message), GFP_KERNEL);
    if (!comm_state.user_to_kernel_queue.messages) {
        kfree(comm_state.kernel_to_user_queue.messages);
        return -ENOMEM;
    }
    
    comm_state.user_to_kernel_queue.head = 0;
    comm_state.user_to_kernel_queue.tail = 0;
    comm_state.user_to_kernel_queue.size = queue_size;
    spin_lock_init(&comm_state.user_to_kernel_queue.lock);
    init_waitqueue_head(&comm_state.user_to_kernel_queue.wait_queue);
    
    archangel_debug("Initialized message queues with %u slots each", queue_size);
    
    return 0;
}

/*
 * Cleanup message queues
 */
static void cleanup_message_queues(void)
{
    if (comm_state.kernel_to_user_queue.messages) {
        kfree(comm_state.kernel_to_user_queue.messages);
        comm_state.kernel_to_user_queue.messages = NULL;
    }
    
    if (comm_state.user_to_kernel_queue.messages) {
        kfree(comm_state.user_to_kernel_queue.messages);
        comm_state.user_to_kernel_queue.messages = NULL;
    }
}

/*
 * Enqueue message (lock-free circular buffer)
 */
static int enqueue_message(struct archangel_message *queue_base, volatile u32 *head, 
                          volatile u32 *tail, u32 queue_size, spinlock_t *lock,
                          const struct archangel_message *msg)
{
    unsigned long flags;
    u32 next_tail;
    
    spin_lock_irqsave(lock, flags);
    
    next_tail = (*tail + 1) % queue_size;
    
    /* Check if queue is full */
    if (next_tail == *head) {
        spin_unlock_irqrestore(lock, flags);
        return -EAGAIN;
    }
    
    /* Copy message */
    memcpy(&queue_base[*tail], msg, sizeof(struct archangel_message) + msg->data_size);
    
    /* Update tail */
    *tail = next_tail;
    
    spin_unlock_irqrestore(lock, flags);
    
    return 0;
}

/*
 * Dequeue message (lock-free circular buffer)
 */
static int dequeue_message(struct archangel_message *queue_base, volatile u32 *head,
                          volatile u32 *tail, u32 queue_size, spinlock_t *lock,
                          struct archangel_message *msg)
{
    unsigned long flags;
    
    spin_lock_irqsave(lock, flags);
    
    /* Check if queue is empty */
    if (*head == *tail) {
        spin_unlock_irqrestore(lock, flags);
        return -EAGAIN;
    }
    
    /* Copy message */
    memcpy(msg, &queue_base[*head], sizeof(struct archangel_message));
    if (msg->data_size > 0)
        memcpy(msg->data, queue_base[*head].data, msg->data_size);
    
    /* Update head */
    *head = (*head + 1) % queue_size;
    
    spin_unlock_irqrestore(lock, flags);
    
    return 0;
}

/*
 * Communication device file operations
 */
static int archangel_comm_open(struct inode *inode, struct file *file)
{
    archangel_debug("Communication device opened by PID %d", current->pid);
    return 0;
}

static int archangel_comm_release(struct inode *inode, struct file *file)
{
    archangel_debug("Communication device closed by PID %d", current->pid);
    return 0;
}

static long archangel_comm_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    switch (cmd) {
    case ARCHANGEL_IOC_GET_STATS: {
        struct archangel_stats stats;
        archangel_get_stats(&stats);
        
        /* Add communication statistics */
        // stats.messages_sent = atomic64_read(&comm_state.messages_sent);
        // stats.messages_received = atomic64_read(&comm_state.messages_received);
        
        if (copy_to_user((void __user *)arg, &stats, sizeof(stats)))
            return -EFAULT;
        return 0;
    }
    
    default:
        return -ENOTTY;
    }
}

static int archangel_comm_mmap(struct file *file, struct vm_area_struct *vma)
{
    unsigned long size = vma->vm_end - vma->vm_start;
    unsigned long offset = vma->vm_pgoff << PAGE_SHIFT;
    int i;
    
    /* Check size and offset */
    if (offset + size > comm_state.shared_mem_size)
        return -EINVAL;
    
    /* Map pages to userspace */
    for (i = 0; i < comm_state.num_pages; i++) {
        if (remap_pfn_range(vma, 
                           vma->vm_start + i * PAGE_SIZE,
                           page_to_pfn(comm_state.shared_pages[i]),
                           PAGE_SIZE,
                           vma->vm_page_prot)) {
            return -EAGAIN;
        }
    }
    
    archangel_debug("Mapped %lu bytes of shared memory to userspace", size);
    
    return 0;
}

/*
 * Get communication statistics
 */
void archangel_get_comm_stats(u64 *sent, u64 *received, u64 *queue_full, u64 *queue_empty)
{
    if (sent)
        *sent = atomic64_read(&comm_state.messages_sent);
    if (received)
        *received = atomic64_read(&comm_state.messages_received);
    if (queue_full)
        *queue_full = atomic64_read(&comm_state.ring_buffer_overruns);
    if (queue_empty)
        *queue_empty = 0; /* Ring buffers don't have empty errors */
}

/*
 * Initialize high-performance ring buffers
 */
static int init_ring_buffers(void)
{
    size_t ring_size = ARCHANGEL_RING_BUFFER_SIZE / sizeof(struct archangel_message);
    size_t total_size = sizeof(struct archangel_ring_buffer) + 
                       (ring_size * sizeof(struct archangel_message)) * 2;
    
    /* Allocate aligned memory for both ring buffers */
    comm_state.ring_buffer_memory = kmalloc(total_size, GFP_KERNEL | __GFP_ZERO);
    if (!comm_state.ring_buffer_memory)
        return -ENOMEM;
    
    comm_state.ring_buffer_total_size = total_size;
    
    /* Setup kernel-to-user ring buffer */
    comm_state.kernel_to_user_ring = (struct archangel_ring_buffer *)comm_state.ring_buffer_memory;
    comm_state.kernel_to_user_ring->buffer_size = ring_size;
    comm_state.kernel_to_user_ring->buffer_mask = ring_size - 1; /* Must be power of 2 */
    comm_state.kernel_to_user_ring->producer_index = 0;
    comm_state.kernel_to_user_ring->consumer_index = 0;
    atomic64_set(&comm_state.kernel_to_user_ring->total_messages, 0);
    atomic64_set(&comm_state.kernel_to_user_ring->dropped_messages, 0);
    atomic64_set(&comm_state.kernel_to_user_ring->batch_count, 0);
    
    /* Setup user-to-kernel ring buffer */
    char *second_ring_addr = (char *)comm_state.ring_buffer_memory + 
                            sizeof(struct archangel_ring_buffer) + 
                            (ring_size * sizeof(struct archangel_message));
    comm_state.user_to_kernel_ring = (struct archangel_ring_buffer *)second_ring_addr;
    comm_state.user_to_kernel_ring->buffer_size = ring_size;
    comm_state.user_to_kernel_ring->buffer_mask = ring_size - 1;
    comm_state.user_to_kernel_ring->producer_index = 0;
    comm_state.user_to_kernel_ring->consumer_index = 0;
    atomic64_set(&comm_state.user_to_kernel_ring->total_messages, 0);
    atomic64_set(&comm_state.user_to_kernel_ring->dropped_messages, 0);
    atomic64_set(&comm_state.user_to_kernel_ring->batch_count, 0);
    
    archangel_info("Initialized ring buffers: %zu messages each (%zu bytes total)",
                   ring_size, total_size);
    
    return 0;
}

/*
 * Cleanup ring buffers
 */
static void cleanup_ring_buffers(void)
{
    if (comm_state.ring_buffer_memory) {
        kfree(comm_state.ring_buffer_memory);
        comm_state.ring_buffer_memory = NULL;
    }
    
    comm_state.kernel_to_user_ring = NULL;
    comm_state.user_to_kernel_ring = NULL;
    comm_state.ring_buffer_total_size = 0;
}

/*
 * Initialize decision cache
 */
static int init_decision_cache(void)
{
    size_t cache_size = sizeof(struct archangel_decision_cache);
    
    comm_state.decision_cache = kmalloc(cache_size, GFP_KERNEL | __GFP_ZERO);
    if (!comm_state.decision_cache)
        return -ENOMEM;
    
    /* Initialize cache statistics */
    atomic64_set(&comm_state.decision_cache->cache_hits, 0);
    atomic64_set(&comm_state.decision_cache->cache_misses, 0);
    atomic64_set(&comm_state.decision_cache->cache_evictions, 0);
    
    /* Generate random hash seed */
    get_random_bytes(&comm_state.decision_cache->hash_seed, sizeof(u32));
    
    archangel_info("Initialized decision cache with %zu entries",
                   ARRAY_SIZE(comm_state.decision_cache->entries));
    
    return 0;
}

/*
 * Cleanup decision cache
 */
static void cleanup_decision_cache(void)
{
    if (comm_state.decision_cache) {
        kfree(comm_state.decision_cache);
        comm_state.decision_cache = NULL;
    }
}

/*
 * Initialize DMA coherent buffers
 */
static int init_dma_buffers(void)
{
    /* Try to allocate DMA coherent memory for zero-copy operations */
    comm_state.dma_buffer_size = ARCHANGEL_DMA_COHERENT_SIZE;
    comm_state.dma_coherent_buffer = dma_alloc_coherent(NULL, 
                                                       comm_state.dma_buffer_size,
                                                       &comm_state.dma_handle,
                                                       GFP_KERNEL);
    
    if (!comm_state.dma_coherent_buffer) {
        comm_state.dma_buffer_size = 0;
        return -ENOMEM;
    }
    
    archangel_info("Initialized DMA coherent buffer: %zu bytes at 0x%llx",
                   comm_state.dma_buffer_size, (u64)comm_state.dma_handle);
    
    return 0;
}

/*
 * Cleanup DMA buffers
 */
static void cleanup_dma_buffers(void)
{
    if (comm_state.dma_coherent_buffer) {
        dma_free_coherent(NULL, comm_state.dma_buffer_size,
                         comm_state.dma_coherent_buffer, comm_state.dma_handle);
        comm_state.dma_coherent_buffer = NULL;
        comm_state.dma_buffer_size = 0;
    }
}

/*
 * High-performance ring buffer produce (lock-free)
 */
static inline int ring_buffer_produce(struct archangel_ring_buffer *ring, 
                                     const struct archangel_message *msg)
{
    u32 producer_index, next_producer_index;
    u32 consumer_index;
    
    if (!ring || !msg)
        return -EINVAL;
    
    producer_index = ring->producer_index;
    next_producer_index = (producer_index + 1) & ring->buffer_mask;
    
    /* Check if buffer is full */
    consumer_index = ring->consumer_index;
    if (next_producer_index == consumer_index) {
        atomic64_inc(&ring->dropped_messages);
        return -EAGAIN;
    }
    
    /* Copy message into ring buffer */
    memcpy(&ring->buffer[producer_index], msg, sizeof(struct archangel_message));
    
    /* Ensure memory ordering */
    smp_wmb();
    
    /* Update producer index atomically */
    ring->producer_index = next_producer_index;
    atomic64_inc(&ring->total_messages);
    
    return 0;
}

/*
 * High-performance ring buffer consume (lock-free)
 */
static inline int ring_buffer_consume(struct archangel_ring_buffer *ring, 
                                     struct archangel_message *msg)
{
    u32 consumer_index, next_consumer_index;
    u32 producer_index;
    
    if (!ring || !msg)
        return -EINVAL;
    
    consumer_index = ring->consumer_index;
    producer_index = ring->producer_index;
    
    /* Check if buffer is empty */
    if (consumer_index == producer_index)
        return -EAGAIN;
    
    /* Copy message from ring buffer */
    memcpy(msg, &ring->buffer[consumer_index], sizeof(struct archangel_message));
    
    /* Ensure memory ordering */
    smp_rmb();
    
    /* Update consumer index */
    next_consumer_index = (consumer_index + 1) & ring->buffer_mask;
    ring->consumer_index = next_consumer_index;
    
    return 0;
}

/*
 * Check if ring buffer is empty
 */
static inline bool ring_buffer_is_empty(struct archangel_ring_buffer *ring)
{
    return ring->consumer_index == ring->producer_index;
}

/*
 * Check if ring buffer is full
 */
static inline bool ring_buffer_is_full(struct archangel_ring_buffer *ring)
{
    u32 next_producer = (ring->producer_index + 1) & ring->buffer_mask;
    return next_producer == ring->consumer_index;
}

/*
 * Hash security context for cache lookup
 */
static inline u64 hash_security_context(struct archangel_security_context *ctx)
{
    if (!ctx)
        return 0;
    
    /* Create hash from key context fields */
    u64 hash = ctx->pid;
    hash = hash * 31 + ctx->uid;
    hash = hash * 31 + ctx->syscall_nr;
    hash = hash * 31 + ctx->flags;
    
    /* Include command name in hash */
    int i;
    for (i = 0; i < sizeof(ctx->comm) && ctx->comm[i]; i++) {
        hash = hash * 31 + ctx->comm[i];
    }
    
    return hash ^ comm_state.decision_cache->hash_seed;
}

/*
 * Look up decision in cache
 */
static inline enum archangel_decision cache_lookup_decision(
    struct archangel_security_context *ctx)
{
    u64 context_hash;
    u32 cache_index;
    u64 current_time;
    
    if (!comm_state.decision_cache || !ctx)
        return ARCHANGEL_UNKNOWN;
    
    context_hash = hash_security_context(ctx);
    cache_index = context_hash & (ARRAY_SIZE(comm_state.decision_cache->entries) - 1);
    current_time = ktime_get_ns();
    
    /* Check if cache entry is valid and not expired (5 second TTL) */
    if (comm_state.decision_cache->entries[cache_index].context_hash == context_hash &&
        (current_time - comm_state.decision_cache->entries[cache_index].timestamp) < 5000000000ULL) {
        
        atomic_inc(&comm_state.decision_cache->entries[cache_index].access_count);
        atomic64_inc(&comm_state.decision_cache->cache_hits);
        
        return comm_state.decision_cache->entries[cache_index].decision;
    }
    
    atomic64_inc(&comm_state.decision_cache->cache_misses);
    return ARCHANGEL_UNKNOWN;
}

/*
 * Store decision in cache
 */
static inline void cache_store_decision(struct archangel_security_context *ctx, 
                                       enum archangel_decision decision)
{
    u64 context_hash;
    u32 cache_index;
    
    if (!comm_state.decision_cache || !ctx)
        return;
    
    context_hash = hash_security_context(ctx);
    cache_index = context_hash & (ARRAY_SIZE(comm_state.decision_cache->entries) - 1);
    
    /* Check if we're evicting an existing entry */
    if (comm_state.decision_cache->entries[cache_index].context_hash != 0 &&
        comm_state.decision_cache->entries[cache_index].context_hash != context_hash) {
        atomic64_inc(&comm_state.decision_cache->cache_evictions);
    }
    
    /* Store new cache entry */
    comm_state.decision_cache->entries[cache_index].context_hash = context_hash;
    comm_state.decision_cache->entries[cache_index].decision = decision;
    comm_state.decision_cache->entries[cache_index].timestamp = ktime_get_ns();
    atomic_set(&comm_state.decision_cache->entries[cache_index].access_count, 1);
}

/*
 * Batch processing work function
 */
static void batch_process_work(struct work_struct *work)
{
    struct archangel_message messages[ARCHANGEL_MAX_BATCH_SIZE];
    int msg_count = 0;
    int i;
    
    /* Consume messages from ring buffer in batches */
    while (msg_count < ARCHANGEL_MAX_BATCH_SIZE && 
           !ring_buffer_is_empty(comm_state.user_to_kernel_ring)) {
        
        if (ring_buffer_consume(comm_state.user_to_kernel_ring, 
                               &messages[msg_count]) == 0) {
            msg_count++;
        } else {
            break;
        }
    }
    
    if (msg_count > 0) {
        /* Process batch of messages */
        process_message_batch(messages, msg_count);
        atomic64_inc(&comm_state.user_to_kernel_ring->batch_count);
    }
}

/*
 * Process batch of messages
 */
static int process_message_batch(struct archangel_message *messages, int count)
{
    int i;
    
    for (i = 0; i < count; i++) {
        struct archangel_message *msg = &messages[i];
        
        /* Process message based on type */
        switch (msg->type) {
        case ARCHANGEL_MSG_ANALYSIS_RESPONSE:
            /* Handle analysis response from userspace AI */
            if (msg->data_size >= sizeof(struct archangel_security_context) + sizeof(enum archangel_decision)) {
                struct archangel_security_context *ctx = (struct archangel_security_context *)msg->data;
                enum archangel_decision *decision = (enum archangel_decision *)(msg->data + sizeof(*ctx));
                
                /* Cache the decision for future fast lookups */
                cache_store_decision(ctx, *decision);
                
                archangel_debug("Cached decision %d for PID %u", *decision, ctx->pid);
            }
            break;
            
        case ARCHANGEL_MSG_RULE_UPDATE:
            /* Handle rule updates from userspace */
            archangel_debug("Received rule update from userspace");
            break;
            
        default:
            archangel_debug("Received message type %d from userspace", msg->type);
            break;
        }
        
        atomic64_inc(&comm_state.messages_received);
    }
    
    return count;
}

/*
 * Enhanced make decision function with cache integration
 */
enum archangel_decision archangel_make_decision_cached(struct archangel_security_context *ctx)
{
    enum archangel_decision decision;
    
    if (!ctx)
        return ARCHANGEL_UNKNOWN;
    
    /* Check cache first for sub-microsecond responses */
    decision = cache_lookup_decision(ctx);
    if (decision != ARCHANGEL_UNKNOWN) {
        atomic64_inc(&comm_state.fast_path_decisions);
        return decision;
    }
    
    /* Fall back to normal decision making process */
    decision = archangel_make_decision(ctx);
    
    /* Cache the decision for future fast lookups */
    if (decision != ARCHANGEL_UNKNOWN && decision != ARCHANGEL_DEFER_TO_USERSPACE) {
        cache_store_decision(ctx, decision);
    }
    
    return decision;
}

/*
 * Get enhanced communication statistics
 */
void archangel_get_enhanced_comm_stats(u64 *sent, u64 *received, u64 *overruns, 
                                      u64 *zero_copy_ops, u64 *fast_path_decisions,
                                      u64 *avg_response_ns, u64 *min_response_ns, 
                                      u64 *max_response_ns, u64 *cache_hits, u64 *cache_misses)
{
    if (sent)
        *sent = atomic64_read(&comm_state.messages_sent);
    if (received)
        *received = atomic64_read(&comm_state.messages_received);
    if (overruns)
        *overruns = atomic64_read(&comm_state.ring_buffer_overruns);
    if (zero_copy_ops)
        *zero_copy_ops = atomic64_read(&comm_state.zero_copy_operations);
    if (fast_path_decisions)
        *fast_path_decisions = atomic64_read(&comm_state.fast_path_decisions);
    if (avg_response_ns)
        *avg_response_ns = atomic64_read(&comm_state.avg_response_time_ns);
    if (min_response_ns)
        *min_response_ns = atomic64_read(&comm_state.min_response_time_ns);
    if (max_response_ns)
        *max_response_ns = atomic64_read(&comm_state.max_response_time_ns);
    
    if (comm_state.decision_cache) {
        if (cache_hits)
            *cache_hits = atomic64_read(&comm_state.decision_cache->cache_hits);
        if (cache_misses)
            *cache_misses = atomic64_read(&comm_state.decision_cache->cache_misses);
    }
}