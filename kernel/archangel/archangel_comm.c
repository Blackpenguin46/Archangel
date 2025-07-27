#include "archangel_core.h"
#include "archangel_comm.h"

/* Global communication manager */
struct archangel_comm_manager *archangel_comm_mgr = NULL;

/**
 * archangel_comm_manager_init - Initialize the communication manager
 * 
 * Returns: 0 on success, negative error code on failure
 */
int archangel_comm_manager_init(void)
{
    int ret, i;
    
    if (archangel_comm_mgr) {
        pr_warn("archangel_comm: Communication manager already initialized\n");
        return -EEXIST;
    }
    
    archangel_comm_mgr = kzalloc(sizeof(*archangel_comm_mgr), GFP_KERNEL);
    if (!archangel_comm_mgr) {
        pr_err("archangel_comm: Failed to allocate communication manager\n");
        return -ENOMEM;
    }
    
    /* Initialize manager structure */
    spin_lock_init(&archangel_comm_mgr->lock);
    archangel_comm_mgr->channel_count = 0;
    archangel_comm_mgr->default_channel = 0;
    
    /* Initialize all channels as inactive */
    for (i = 0; i < ARCHANGEL_COMM_MAX_CHANNELS; i++) {
        struct archangel_comm_channel *channel = &archangel_comm_mgr->channels[i];
        
        channel->id = i;
        channel->active = false;
        spin_lock_init(&channel->lock);
        atomic64_set(&channel->sequence_counter, 0);
        
        /* Initialize statistics */
        atomic64_set(&channel->stats.messages_sent, 0);
        atomic64_set(&channel->stats.messages_received, 0);
        atomic64_set(&channel->stats.bytes_sent, 0);
        atomic64_set(&channel->stats.bytes_received, 0);
        atomic64_set(&channel->stats.dma_transfers, 0);
        atomic64_set(&channel->stats.queue_full_events, 0);
        atomic64_set(&channel->stats.errors, 0);
    }
    
    /* Create default channel */
    ret = archangel_comm_channel_create(&archangel_comm_mgr->default_channel);
    if (ret) {
        pr_err("archangel_comm: Failed to create default channel: %d\n", ret);
        kfree(archangel_comm_mgr);
        archangel_comm_mgr = NULL;
        return ret;
    }
    
    archangel_comm_mgr->initialized = true;
    
    pr_info("archangel_comm: Communication manager initialized with %d channels\n", 
            archangel_comm_mgr->channel_count);
    return 0;
}

/**
 * archangel_comm_manager_cleanup - Clean up the communication manager
 */
void archangel_comm_manager_cleanup(void)
{
    int i;
    
    if (!archangel_comm_mgr)
        return;
    
    archangel_comm_mgr->initialized = false;
    
    /* Destroy all active channels */
    for (i = 0; i < ARCHANGEL_COMM_MAX_CHANNELS; i++) {
        if (archangel_comm_mgr->channels[i].active) {
            archangel_comm_channel_destroy(i);
        }
    }
    
    kfree(archangel_comm_mgr);
    archangel_comm_mgr = NULL;
    
    pr_info("archangel_comm: Communication manager cleaned up\n");
}

/**
 * archangel_comm_channel_create - Create a new communication channel
 * @channel_id: Pointer to store the created channel ID
 * 
 * Returns: 0 on success, negative error code on failure
 */
int archangel_comm_channel_create(u32 *channel_id)
{
    struct archangel_comm_channel *channel;
    unsigned long flags;
    int ret, i;
    
    if (!archangel_comm_is_initialized() || !channel_id) {
        return -EINVAL;
    }
    
    spin_lock_irqsave(&archangel_comm_mgr->lock, flags);
    
    /* Find an inactive channel */
    for (i = 0; i < ARCHANGEL_COMM_MAX_CHANNELS; i++) {
        if (!archangel_comm_mgr->channels[i].active) {
            break;
        }
    }
    
    if (i >= ARCHANGEL_COMM_MAX_CHANNELS) {
        spin_unlock_irqrestore(&archangel_comm_mgr->lock, flags);
        pr_err("archangel_comm: No available channels\n");
        return -ENOSPC;
    }
    
    channel = &archangel_comm_mgr->channels[i];
    channel->active = true;
    archangel_comm_mgr->channel_count++;
    *channel_id = i;
    
    spin_unlock_irqrestore(&archangel_comm_mgr->lock, flags);
    
    /* Initialize SPSC queues */
    ret = archangel_spsc_queue_init(&channel->kernel_to_user, ARCHANGEL_COMM_RING_SIZE);
    if (ret) {
        pr_err("archangel_comm: Failed to initialize kernel-to-user queue: %d\n", ret);
        goto cleanup_channel;
    }
    
    ret = archangel_spsc_queue_init(&channel->user_to_kernel, ARCHANGEL_COMM_RING_SIZE);
    if (ret) {
        pr_err("archangel_comm: Failed to initialize user-to-kernel queue: %d\n", ret);
        goto cleanup_k2u_queue;
    }
    
    /* Initialize DMA pool */
    ret = archangel_dma_pool_init(&channel->dma_pool, NULL, 16, PAGE_SIZE * 4);
    if (ret) {
        pr_err("archangel_comm: Failed to initialize DMA pool: %d\n", ret);
        goto cleanup_u2k_queue;
    }
    
    pr_info("archangel_comm: Channel %u created successfully\n", i);
    return 0;

cleanup_u2k_queue:
    archangel_spsc_queue_cleanup(&channel->user_to_kernel);
cleanup_k2u_queue:
    archangel_spsc_queue_cleanup(&channel->kernel_to_user);
cleanup_channel:
    spin_lock_irqsave(&archangel_comm_mgr->lock, flags);
    channel->active = false;
    archangel_comm_mgr->channel_count--;
    spin_unlock_irqrestore(&archangel_comm_mgr->lock, flags);
    return ret;
}

/**
 * archangel_comm_channel_destroy - Destroy a communication channel
 * @channel_id: Channel ID to destroy
 */
void archangel_comm_channel_destroy(u32 channel_id)
{
    struct archangel_comm_channel *channel;
    unsigned long flags;
    
    if (!archangel_comm_is_initialized() || channel_id >= ARCHANGEL_COMM_MAX_CHANNELS) {
        return;
    }
    
    channel = &archangel_comm_mgr->channels[channel_id];
    
    if (!channel->active) {
        return;
    }
    
    /* Clean up shared memory */
    archangel_comm_cleanup_shared_memory(channel_id);
    
    /* Clean up event contexts */
    if (channel->kernel_eventfd) {
        eventfd_ctx_put(channel->kernel_eventfd);
        channel->kernel_eventfd = NULL;
    }
    
    if (channel->user_eventfd) {
        eventfd_ctx_put(channel->user_eventfd);
        channel->user_eventfd = NULL;
    }
    
    /* Clean up DMA pool */
    archangel_dma_pool_cleanup(&channel->dma_pool);
    
    /* Clean up SPSC queues */
    archangel_spsc_queue_cleanup(&channel->user_to_kernel);
    archangel_spsc_queue_cleanup(&channel->kernel_to_user);
    
    /* Mark channel as inactive */
    spin_lock_irqsave(&archangel_comm_mgr->lock, flags);
    channel->active = false;
    archangel_comm_mgr->channel_count--;
    spin_unlock_irqrestore(&archangel_comm_mgr->lock, flags);
    
    pr_info("archangel_comm: Channel %u destroyed\n", channel_id);
}

/**
 * archangel_comm_channel_get - Get a communication channel by ID
 * @channel_id: Channel ID
 * 
 * Returns: Channel pointer on success, NULL on failure
 */
struct archangel_comm_channel *archangel_comm_channel_get(u32 channel_id)
{
    if (!archangel_comm_is_initialized() || channel_id >= ARCHANGEL_COMM_MAX_CHANNELS) {
        return NULL;
    }
    
    if (!archangel_comm_mgr->channels[channel_id].active) {
        return NULL;
    }
    
    return &archangel_comm_mgr->channels[channel_id];
}
/**
 *
 archangel_spsc_queue_init - Initialize a Single Producer Single Consumer queue
 * @queue: Queue to initialize
 * @size: Size of the queue buffer
 * 
 * Returns: 0 on success, negative error code on failure
 */
int archangel_spsc_queue_init(struct archangel_spsc_queue *queue, u32 size)
{
    if (!queue || size == 0) {
        return -EINVAL;
    }
    
    /* Ensure size is power of 2 for efficient modulo operations */
    size = archangel_comm_next_power_of_2(size);
    
    queue->buffer = kzalloc(size, GFP_KERNEL);
    if (!queue->buffer) {
        pr_err("archangel_comm: Failed to allocate queue buffer of size %u\n", size);
        return -ENOMEM;
    }
    
    queue->size = size;
    queue->head = 0;
    queue->tail = 0;
    queue->cached_head = 0;
    queue->cached_tail = 0;
    
    pr_debug("archangel_comm: SPSC queue initialized with size %u\n", size);
    return 0;
}

/**
 * archangel_spsc_queue_cleanup - Clean up a SPSC queue
 * @queue: Queue to clean up
 */
void archangel_spsc_queue_cleanup(struct archangel_spsc_queue *queue)
{
    if (!queue)
        return;
    
    kfree(queue->buffer);
    queue->buffer = NULL;
    queue->size = 0;
    queue->head = 0;
    queue->tail = 0;
    queue->cached_head = 0;
    queue->cached_tail = 0;
}

/**
 * archangel_spsc_queue_push - Push data to SPSC queue (producer side)
 * @queue: Queue to push to
 * @data: Data to push
 * @size: Size of data
 * 
 * Returns: 0 on success, negative error code on failure
 */
int archangel_spsc_queue_push(struct archangel_spsc_queue *queue, const void *data, u32 size)
{
    u32 head, next_head, tail;
    u32 available_space;
    u32 total_size;
    
    if (!queue || !data || size == 0) {
        return -EINVAL;
    }
    
    /* Include size header in total size */
    total_size = size + sizeof(u32);
    
    head = queue->head;
    
    /* Check cached tail first to avoid cache miss */
    tail = queue->cached_tail;
    if (head - tail >= queue->size - total_size) {
        /* Cache miss, read actual tail */
        tail = READ_ONCE(queue->tail);
        queue->cached_tail = tail;
        
        /* Check again with actual tail */
        if (head - tail >= queue->size - total_size) {
            return -ENOSPC; /* Queue full */
        }
    }
    
    /* Calculate available space */
    available_space = queue->size - (head - tail);
    if (available_space < total_size) {
        return -ENOSPC;
    }
    
    /* Write size header first */
    *((u32 *)((char *)queue->buffer + (head & (queue->size - 1)))) = size;
    head += sizeof(u32);
    
    /* Handle wrap-around for data */
    u32 head_pos = head & (queue->size - 1);
    u32 remaining = queue->size - head_pos;
    
    if (remaining >= size) {
        /* No wrap-around needed */
        memcpy((char *)queue->buffer + head_pos, data, size);
    } else {
        /* Handle wrap-around */
        memcpy((char *)queue->buffer + head_pos, data, remaining);
        memcpy(queue->buffer, (char *)data + remaining, size - remaining);
    }
    
    head += size;
    
    /* Memory barrier to ensure data is written before updating head */
    smp_wmb();
    
    /* Update head (visible to consumer) */
    WRITE_ONCE(queue->head, head);
    
    return 0;
}

/**
 * archangel_spsc_queue_pop - Pop data from SPSC queue (consumer side)
 * @queue: Queue to pop from
 * @data: Buffer to store popped data
 * @max_size: Maximum size of data buffer
 * 
 * Returns: Size of popped data on success, negative error code on failure
 */
int archangel_spsc_queue_pop(struct archangel_spsc_queue *queue, void *data, u32 max_size)
{
    u32 head, tail;
    u32 data_size;
    u32 tail_pos;
    u32 remaining;
    
    if (!queue || !data || max_size == 0) {
        return -EINVAL;
    }
    
    tail = queue->tail;
    
    /* Check cached head first */
    head = queue->cached_head;
    if (head == tail) {
        /* Cache miss, read actual head */
        head = READ_ONCE(queue->head);
        queue->cached_head = head;
        
        /* Check again with actual head */
        if (head == tail) {
            return -ENODATA; /* Queue empty */
        }
    }
    
    /* Read size header */
    data_size = *((u32 *)((char *)queue->buffer + (tail & (queue->size - 1))));
    tail += sizeof(u32);
    
    /* Validate data size */
    if (data_size == 0 || data_size > max_size) {
        pr_err("archangel_comm: Invalid data size %u (max %u)\n", data_size, max_size);
        return -EINVAL;
    }
    
    /* Ensure we have enough data available */
    if (head - tail < data_size) {
        pr_err("archangel_comm: Insufficient data available\n");
        return -ENODATA;
    }
    
    /* Read data with potential wrap-around */
    tail_pos = tail & (queue->size - 1);
    remaining = queue->size - tail_pos;
    
    if (remaining >= data_size) {
        /* No wrap-around needed */
        memcpy(data, (char *)queue->buffer + tail_pos, data_size);
    } else {
        /* Handle wrap-around */
        memcpy(data, (char *)queue->buffer + tail_pos, remaining);
        memcpy((char *)data + remaining, queue->buffer, data_size - remaining);
    }
    
    tail += data_size;
    
    /* Memory barrier to ensure data is read before updating tail */
    smp_rmb();
    
    /* Update tail (visible to producer) */
    WRITE_ONCE(queue->tail, tail);
    
    return data_size;
}

/**
 * archangel_spsc_queue_empty - Check if queue is empty
 * @queue: Queue to check
 * 
 * Returns: true if empty, false otherwise
 */
bool archangel_spsc_queue_empty(struct archangel_spsc_queue *queue)
{
    if (!queue) {
        return true;
    }
    
    return READ_ONCE(queue->head) == READ_ONCE(queue->tail);
}

/**
 * archangel_spsc_queue_full - Check if queue is full
 * @queue: Queue to check
 * 
 * Returns: true if full, false otherwise
 */
bool archangel_spsc_queue_full(struct archangel_spsc_queue *queue)
{
    u32 head, tail;
    
    if (!queue) {
        return true;
    }
    
    head = READ_ONCE(queue->head);
    tail = READ_ONCE(queue->tail);
    
    return (head - tail) >= (queue->size - sizeof(u32) - 1);
}

/**
 * archangel_spsc_queue_available_space - Get available space in queue
 * @queue: Queue to check
 * 
 * Returns: Available space in bytes
 */
u32 archangel_spsc_queue_available_space(struct archangel_spsc_queue *queue)
{
    u32 head, tail;
    
    if (!queue) {
        return 0;
    }
    
    head = READ_ONCE(queue->head);
    tail = READ_ONCE(queue->tail);
    
    return queue->size - (head - tail);
}/**

 * archangel_dma_pool_init - Initialize DMA buffer pool
 * @pool: DMA pool to initialize
 * @dev: Device for DMA operations (can be NULL for coherent allocation)
 * @buffer_count: Number of buffers in pool
 * @buffer_size: Size of each buffer
 * 
 * Returns: 0 on success, negative error code on failure
 */
int archangel_dma_pool_init(struct archangel_dma_pool *pool, struct device *dev, 
                           u32 buffer_count, size_t buffer_size)
{
    int i, ret;
    
    if (!pool || buffer_count == 0 || buffer_size == 0) {
        return -EINVAL;
    }
    
    pool->buffers = kzalloc(buffer_count * sizeof(struct archangel_dma_buffer), GFP_KERNEL);
    if (!pool->buffers) {
        pr_err("archangel_comm: Failed to allocate DMA buffer array\n");
        return -ENOMEM;
    }
    
    pool->count = buffer_count;
    pool->next_free = 0;
    pool->device = dev;
    spin_lock_init(&pool->lock);
    
    /* Initialize each DMA buffer */
    for (i = 0; i < buffer_count; i++) {
        struct archangel_dma_buffer *buffer = &pool->buffers[i];
        
        /* Allocate coherent DMA memory */
        if (dev) {
            buffer->vaddr = dma_alloc_coherent(dev, buffer_size, 
                                             &buffer->dma_addr, GFP_KERNEL);
        } else {
            /* Use regular kernel memory if no device */
            buffer->vaddr = kzalloc(buffer_size, GFP_KERNEL);
            buffer->dma_addr = 0;
        }
        
        if (!buffer->vaddr) {
            pr_err("archangel_comm: Failed to allocate DMA buffer %d\n", i);
            ret = -ENOMEM;
            goto cleanup_buffers;
        }
        
        buffer->size = buffer_size;
        atomic_set(&buffer->in_use, 0);
        
        /* Allocate scatter-gather list */
        buffer->sg_list = kzalloc(sizeof(struct scatterlist), GFP_KERNEL);
        if (!buffer->sg_list) {
            pr_err("archangel_comm: Failed to allocate scatter-gather list for buffer %d\n", i);
            if (dev) {
                dma_free_coherent(dev, buffer_size, buffer->vaddr, buffer->dma_addr);
            } else {
                kfree(buffer->vaddr);
            }
            ret = -ENOMEM;
            goto cleanup_buffers;
        }
        
        /* Initialize scatter-gather list */
        sg_init_one(buffer->sg_list, buffer->vaddr, buffer_size);
        buffer->nents = 1;
    }
    
    pr_info("archangel_comm: DMA pool initialized with %u buffers of %zu bytes each\n", 
            buffer_count, buffer_size);
    return 0;

cleanup_buffers:
    for (i = i - 1; i >= 0; i--) {
        struct archangel_dma_buffer *buffer = &pool->buffers[i];
        
        kfree(buffer->sg_list);
        if (dev) {
            dma_free_coherent(dev, buffer->size, buffer->vaddr, buffer->dma_addr);
        } else {
            kfree(buffer->vaddr);
        }
    }
    kfree(pool->buffers);
    pool->buffers = NULL;
    return ret;
}

/**
 * archangel_dma_pool_cleanup - Clean up DMA buffer pool
 * @pool: DMA pool to clean up
 */
void archangel_dma_pool_cleanup(struct archangel_dma_pool *pool)
{
    int i;
    
    if (!pool || !pool->buffers) {
        return;
    }
    
    /* Free all DMA buffers */
    for (i = 0; i < pool->count; i++) {
        struct archangel_dma_buffer *buffer = &pool->buffers[i];
        
        if (atomic_read(&buffer->in_use)) {
            pr_warn("archangel_comm: DMA buffer %d still in use during cleanup\n", i);
        }
        
        kfree(buffer->sg_list);
        
        if (pool->device) {
            dma_free_coherent(pool->device, buffer->size, 
                            buffer->vaddr, buffer->dma_addr);
        } else {
            kfree(buffer->vaddr);
        }
    }
    
    kfree(pool->buffers);
    pool->buffers = NULL;
    pool->count = 0;
    
    pr_info("archangel_comm: DMA pool cleaned up\n");
}

/**
 * archangel_dma_buffer_alloc - Allocate a DMA buffer from pool
 * @pool: DMA pool to allocate from
 * 
 * Returns: DMA buffer pointer on success, NULL on failure
 */
struct archangel_dma_buffer *archangel_dma_buffer_alloc(struct archangel_dma_pool *pool)
{
    struct archangel_dma_buffer *buffer = NULL;
    unsigned long flags;
    u32 start_idx, i;
    
    if (!pool || !pool->buffers) {
        return NULL;
    }
    
    spin_lock_irqsave(&pool->lock, flags);
    
    start_idx = pool->next_free;
    
    /* Find a free buffer using round-robin */
    for (i = 0; i < pool->count; i++) {
        u32 idx = (start_idx + i) % pool->count;
        struct archangel_dma_buffer *candidate = &pool->buffers[idx];
        
        if (atomic_cmpxchg(&candidate->in_use, 0, 1) == 0) {
            buffer = candidate;
            pool->next_free = (idx + 1) % pool->count;
            break;
        }
    }
    
    spin_unlock_irqrestore(&pool->lock, flags);
    
    if (!buffer) {
        pr_debug("archangel_comm: No free DMA buffers available\n");
    }
    
    return buffer;
}

/**
 * archangel_dma_buffer_free - Free a DMA buffer back to pool
 * @pool: DMA pool to return buffer to
 * @buffer: Buffer to free
 */
void archangel_dma_buffer_free(struct archangel_dma_pool *pool, struct archangel_dma_buffer *buffer)
{
    if (!pool || !buffer) {
        return;
    }
    
    /* Verify buffer belongs to this pool */
    if (buffer < pool->buffers || buffer >= pool->buffers + pool->count) {
        pr_err("archangel_comm: Invalid buffer pointer for pool\n");
        return;
    }
    
    /* Mark buffer as free */
    atomic_set(&buffer->in_use, 0);
}/**
 * a
rchangel_comm_send_message - Send a message through communication channel
 * @channel_id: Channel ID to send through
 * @type: Message type
 * @priority: Message priority
 * @data: Message data
 * @size: Size of message data
 * 
 * Returns: 0 on success, negative error code on failure
 */
int archangel_comm_send_message(u32 channel_id, enum archangel_msg_type type,
                               enum archangel_msg_priority priority,
                               const void *data, u32 size)
{
    struct archangel_comm_channel *channel;
    struct archangel_msg_header header;
    struct archangel_dma_buffer *dma_buffer = NULL;
    int ret;
    bool use_dma = false;
    
    channel = archangel_comm_channel_get(channel_id);
    if (!channel) {
        return -EINVAL;
    }
    
    if (type >= ARCHANGEL_MSG_MAX || !data || size == 0) {
        return -EINVAL;
    }
    
    /* Prepare message header */
    header.magic = ARCHANGEL_COMM_MAGIC;
    header.type = type;
    header.priority = priority;
    header.size = sizeof(header) + size;
    header.sequence = atomic64_inc_return(&channel->sequence_counter);
    header.timestamp = archangel_comm_get_timestamp();
    header.flags = 0;
    
    /* Decide whether to use DMA for large messages */
    if (size > ARCHANGEL_COMM_DMA_THRESHOLD) {
        dma_buffer = archangel_dma_buffer_alloc(&channel->dma_pool);
        if (dma_buffer) {
            use_dma = true;
            header.flags |= 0x01; /* DMA flag */
        }
    }
    
    if (use_dma) {
        /* Copy data to DMA buffer */
        if (size > dma_buffer->size) {
            pr_err("archangel_comm: Message size %u exceeds DMA buffer size %zu\n", 
                   size, dma_buffer->size);
            archangel_dma_buffer_free(&channel->dma_pool, dma_buffer);
            return -EINVAL;
        }
        
        memcpy(dma_buffer->vaddr, data, size);
        
        /* Send header with DMA buffer reference */
        ret = archangel_spsc_queue_push(&channel->kernel_to_user, &header, sizeof(header));
        if (ret) {
            pr_err("archangel_comm: Failed to send DMA message header: %d\n", ret);
            archangel_dma_buffer_free(&channel->dma_pool, dma_buffer);
            archangel_comm_update_stats(channel_id, "error", 1);
            return ret;
        }
        
        /* Send DMA buffer pointer */
        ret = archangel_spsc_queue_push(&channel->kernel_to_user, &dma_buffer, sizeof(dma_buffer));
        if (ret) {
            pr_err("archangel_comm: Failed to send DMA buffer pointer: %d\n", ret);
            archangel_dma_buffer_free(&channel->dma_pool, dma_buffer);
            archangel_comm_update_stats(channel_id, "error", 1);
            return ret;
        }
        
        archangel_comm_update_stats(channel_id, "dma_transfer", 1);
    } else {
        /* Send header first */
        ret = archangel_spsc_queue_push(&channel->kernel_to_user, &header, sizeof(header));
        if (ret) {
            if (ret == -ENOSPC) {
                archangel_comm_update_stats(channel_id, "queue_full", 1);
            } else {
                archangel_comm_update_stats(channel_id, "error", 1);
            }
            return ret;
        }
        
        /* Send data */
        ret = archangel_spsc_queue_push(&channel->kernel_to_user, data, size);
        if (ret) {
            pr_err("archangel_comm: Failed to send message data: %d\n", ret);
            archangel_comm_update_stats(channel_id, "error", 1);
            return ret;
        }
    }
    
    /* Update statistics */
    archangel_comm_update_stats(channel_id, "message_sent", 1);
    archangel_comm_update_stats(channel_id, "bytes_sent", header.size);
    
    /* Notify userspace */
    ret = archangel_comm_notify_user(channel_id);
    if (ret) {
        pr_debug("archangel_comm: Failed to notify userspace: %d\n", ret);
    }
    
    return 0;
}

/**
 * archangel_comm_receive_message - Receive a message from communication channel
 * @channel_id: Channel ID to receive from
 * @type: Pointer to store message type
 * @data: Buffer to store message data
 * @max_size: Maximum size of data buffer
 * 
 * Returns: Size of received data on success, negative error code on failure
 */
int archangel_comm_receive_message(u32 channel_id, enum archangel_msg_type *type,
                                  void *data, u32 max_size)
{
    struct archangel_comm_channel *channel;
    struct archangel_msg_header header;
    struct archangel_dma_buffer *dma_buffer;
    int ret;
    u32 data_size;
    
    channel = archangel_comm_channel_get(channel_id);
    if (!channel || !type || !data || max_size == 0) {
        return -EINVAL;
    }
    
    /* Receive header first */
    ret = archangel_spsc_queue_pop(&channel->user_to_kernel, &header, sizeof(header));
    if (ret < 0) {
        return ret; /* Queue empty or error */
    }
    
    if (ret != sizeof(header)) {
        pr_err("archangel_comm: Invalid header size received: %d\n", ret);
        archangel_comm_update_stats(channel_id, "error", 1);
        return -EINVAL;
    }
    
    /* Validate header */
    if (header.magic != ARCHANGEL_COMM_MAGIC) {
        pr_err("archangel_comm: Invalid magic number: 0x%x\n", header.magic);
        archangel_comm_update_stats(channel_id, "error", 1);
        return -EINVAL;
    }
    
    if (header.type >= ARCHANGEL_MSG_MAX) {
        pr_err("archangel_comm: Invalid message type: %u\n", header.type);
        archangel_comm_update_stats(channel_id, "error", 1);
        return -EINVAL;
    }
    
    *type = header.type;
    data_size = header.size - sizeof(header);
    
    if (data_size > max_size) {
        pr_err("archangel_comm: Message data size %u exceeds buffer size %u\n", 
               data_size, max_size);
        archangel_comm_update_stats(channel_id, "error", 1);
        return -EINVAL;
    }
    
    /* Check if message uses DMA */
    if (header.flags & 0x01) {
        /* Receive DMA buffer pointer */
        ret = archangel_spsc_queue_pop(&channel->user_to_kernel, &dma_buffer, sizeof(dma_buffer));
        if (ret != sizeof(dma_buffer)) {
            pr_err("archangel_comm: Failed to receive DMA buffer pointer: %d\n", ret);
            archangel_comm_update_stats(channel_id, "error", 1);
            return -EINVAL;
        }
        
        /* Copy data from DMA buffer */
        memcpy(data, dma_buffer->vaddr, data_size);
        
        /* Free DMA buffer */
        archangel_dma_buffer_free(&channel->dma_pool, dma_buffer);
        
        archangel_comm_update_stats(channel_id, "dma_transfer", 1);
    } else {
        /* Receive data directly */
        ret = archangel_spsc_queue_pop(&channel->user_to_kernel, data, data_size);
        if (ret != data_size) {
            pr_err("archangel_comm: Failed to receive message data: %d\n", ret);
            archangel_comm_update_stats(channel_id, "error", 1);
            return -EINVAL;
        }
    }
    
    /* Update statistics */
    archangel_comm_update_stats(channel_id, "message_received", 1);
    archangel_comm_update_stats(channel_id, "bytes_received", header.size);
    
    return data_size;
}

/**
 * archangel_comm_notify_kernel - Notify kernel of new message
 * @channel_id: Channel ID
 * 
 * Returns: 0 on success, negative error code on failure
 */
int archangel_comm_notify_kernel(u32 channel_id)
{
    struct archangel_comm_channel *channel;
    
    channel = archangel_comm_channel_get(channel_id);
    if (!channel) {
        return -EINVAL;
    }
    
    if (channel->kernel_eventfd) {
        eventfd_signal(channel->kernel_eventfd, 1);
    }
    
    /* Wake up any waiting kernel threads */
    if (archangel_is_initialized()) {
        wake_up_interruptible(&archangel_ai->comm.wait_queue);
    }
    
    return 0;
}

/**
 * archangel_comm_notify_user - Notify userspace of new message
 * @channel_id: Channel ID
 * 
 * Returns: 0 on success, negative error code on failure
 */
int archangel_comm_notify_user(u32 channel_id)
{
    struct archangel_comm_channel *channel;
    
    channel = archangel_comm_channel_get(channel_id);
    if (!channel) {
        return -EINVAL;
    }
    
    if (channel->user_eventfd) {
        eventfd_signal(channel->user_eventfd, 1);
        return 0;
    }
    
    return -ENOTCONN;
}/**
 * arc
hangel_comm_setup_shared_memory - Set up shared memory for channel
 * @channel_id: Channel ID
 * @size: Size of shared memory region
 * 
 * Returns: 0 on success, negative error code on failure
 */
int archangel_comm_setup_shared_memory(u32 channel_id, size_t size)
{
    struct archangel_comm_channel *channel;
    int num_pages, i;
    
    channel = archangel_comm_channel_get(channel_id);
    if (!channel) {
        return -EINVAL;
    }
    
    if (channel->shared_pages) {
        pr_warn("archangel_comm: Shared memory already set up for channel %u\n", channel_id);
        return -EEXIST;
    }
    
    /* Round up to page boundary */
    size = PAGE_ALIGN(size);
    num_pages = size >> PAGE_SHIFT;
    
    /* Allocate page array */
    channel->shared_pages = kzalloc(num_pages * sizeof(struct page *), GFP_KERNEL);
    if (!channel->shared_pages) {
        pr_err("archangel_comm: Failed to allocate page array\n");
        return -ENOMEM;
    }
    
    /* Allocate pages */
    for (i = 0; i < num_pages; i++) {
        channel->shared_pages[i] = alloc_page(GFP_KERNEL | __GFP_ZERO);
        if (!channel->shared_pages[i]) {
            pr_err("archangel_comm: Failed to allocate page %d\n", i);
            goto cleanup_pages;
        }
    }
    
    channel->shared_size = size;
    
    pr_info("archangel_comm: Shared memory set up for channel %u: %zu bytes (%d pages)\n",
            channel_id, size, num_pages);
    return 0;

cleanup_pages:
    for (i = i - 1; i >= 0; i--) {
        __free_page(channel->shared_pages[i]);
    }
    kfree(channel->shared_pages);
    channel->shared_pages = NULL;
    return -ENOMEM;
}

/**
 * archangel_comm_cleanup_shared_memory - Clean up shared memory for channel
 * @channel_id: Channel ID
 */
void archangel_comm_cleanup_shared_memory(u32 channel_id)
{
    struct archangel_comm_channel *channel;
    int num_pages, i;
    
    channel = archangel_comm_channel_get(channel_id);
    if (!channel || !channel->shared_pages) {
        return;
    }
    
    num_pages = channel->shared_size >> PAGE_SHIFT;
    
    /* Free all pages */
    for (i = 0; i < num_pages; i++) {
        __free_page(channel->shared_pages[i]);
    }
    
    kfree(channel->shared_pages);
    channel->shared_pages = NULL;
    channel->shared_size = 0;
    
    pr_info("archangel_comm: Shared memory cleaned up for channel %u\n", channel_id);
}

/**
 * archangel_comm_update_stats - Update channel statistics
 * @channel_id: Channel ID
 * @event: Event name
 * @value: Value to add
 */
void archangel_comm_update_stats(u32 channel_id, const char *event, u64 value)
{
    struct archangel_comm_channel *channel;
    
    channel = archangel_comm_channel_get(channel_id);
    if (!channel || !event) {
        return;
    }
    
    if (strcmp(event, "message_sent") == 0) {
        atomic64_add(value, &channel->stats.messages_sent);
    } else if (strcmp(event, "message_received") == 0) {
        atomic64_add(value, &channel->stats.messages_received);
    } else if (strcmp(event, "bytes_sent") == 0) {
        atomic64_add(value, &channel->stats.bytes_sent);
    } else if (strcmp(event, "bytes_received") == 0) {
        atomic64_add(value, &channel->stats.bytes_received);
    } else if (strcmp(event, "dma_transfer") == 0) {
        atomic64_add(value, &channel->stats.dma_transfers);
    } else if (strcmp(event, "queue_full") == 0) {
        atomic64_add(value, &channel->stats.queue_full_events);
    } else if (strcmp(event, "error") == 0) {
        atomic64_add(value, &channel->stats.errors);
    }
}

/**
 * archangel_comm_get_stats - Get channel statistics for proc display
 * @channel_id: Channel ID
 * @m: Seq file for output
 */
void archangel_comm_get_stats(u32 channel_id, struct seq_file *m)
{
    struct archangel_comm_channel *channel;
    
    channel = archangel_comm_channel_get(channel_id);
    if (!channel) {
        seq_printf(m, "Channel %u: Not found\n", channel_id);
        return;
    }
    
    seq_printf(m, "Channel %u Statistics:\n", channel_id);
    seq_printf(m, "  Status: %s\n", channel->active ? "Active" : "Inactive");
    seq_printf(m, "  Messages sent: %llu\n", 
               atomic64_read(&channel->stats.messages_sent));
    seq_printf(m, "  Messages received: %llu\n", 
               atomic64_read(&channel->stats.messages_received));
    seq_printf(m, "  Bytes sent: %llu\n", 
               atomic64_read(&channel->stats.bytes_sent));
    seq_printf(m, "  Bytes received: %llu\n", 
               atomic64_read(&channel->stats.bytes_received));
    seq_printf(m, "  DMA transfers: %llu\n", 
               atomic64_read(&channel->stats.dma_transfers));
    seq_printf(m, "  Queue full events: %llu\n", 
               atomic64_read(&channel->stats.queue_full_events));
    seq_printf(m, "  Errors: %llu\n", 
               atomic64_read(&channel->stats.errors));
    seq_printf(m, "  Sequence counter: %llu\n", 
               atomic64_read(&channel->sequence_counter));
    
    /* Queue statistics */
    seq_printf(m, "  Kernel->User queue:\n");
    seq_printf(m, "    Empty: %s\n", 
               archangel_spsc_queue_empty(&channel->kernel_to_user) ? "Yes" : "No");
    seq_printf(m, "    Available space: %u bytes\n", 
               archangel_spsc_queue_available_space(&channel->kernel_to_user));
    
    seq_printf(m, "  User->Kernel queue:\n");
    seq_printf(m, "    Empty: %s\n", 
               archangel_spsc_queue_empty(&channel->user_to_kernel) ? "Yes" : "No");
    seq_printf(m, "    Available space: %u bytes\n", 
               archangel_spsc_queue_available_space(&channel->user_to_kernel));
    
    /* Shared memory info */
    if (channel->shared_pages) {
        seq_printf(m, "  Shared memory: %zu bytes (%zu pages)\n", 
                   channel->shared_size, channel->shared_size >> PAGE_SHIFT);
    } else {
        seq_printf(m, "  Shared memory: Not allocated\n");
    }
    
    /* Event notification status */
    seq_printf(m, "  Kernel eventfd: %s\n", 
               channel->kernel_eventfd ? "Connected" : "Not connected");
    seq_printf(m, "  User eventfd: %s\n", 
               channel->user_eventfd ? "Connected" : "Not connected");
}

/* Export symbols for other Archangel modules */
EXPORT_SYMBOL(archangel_comm_mgr);
EXPORT_SYMBOL(archangel_comm_manager_init);
EXPORT_SYMBOL(archangel_comm_manager_cleanup);
EXPORT_SYMBOL(archangel_comm_channel_create);
EXPORT_SYMBOL(archangel_comm_channel_destroy);
EXPORT_SYMBOL(archangel_comm_channel_get);
EXPORT_SYMBOL(archangel_comm_send_message);
EXPORT_SYMBOL(archangel_comm_receive_message);
EXPORT_SYMBOL(archangel_comm_notify_kernel);
EXPORT_SYMBOL(archangel_comm_notify_user);
EXPORT_SYMBOL(archangel_comm_setup_shared_memory);
EXPORT_SYMBOL(archangel_comm_cleanup_shared_memory);
EXPORT_SYMBOL(archangel_comm_update_stats);
EXPORT_SYMBOL(archangel_comm_get_stats);