/*
 * Archangel Linux - Kernel-Userspace Communication
 * High-speed shared memory communication bridge
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

#include "../include/archangel.h"

/* Communication state */
static struct {
    bool initialized;
    
    /* Shared memory */
    struct archangel_shared_memory *shared_mem;
    unsigned long shared_mem_size;
    struct page **shared_pages;
    int num_pages;
    
    /* Message queues */
    struct {
        struct archangel_message *messages;
        volatile u32 head;
        volatile u32 tail;
        u32 size;
        spinlock_t lock;
        wait_queue_head_t wait_queue;
    } kernel_to_user_queue, user_to_kernel_queue;
    
    /* Statistics */
    atomic64_t messages_sent;
    atomic64_t messages_received;
    atomic64_t queue_full_errors;
    atomic64_t queue_empty_errors;
    
    /* Synchronization */
    struct mutex comm_lock;
    
} comm_state = {
    .initialized = false,
    .comm_lock = __MUTEX_INITIALIZER(comm_state.comm_lock),
};

/* Forward declarations */
static int alloc_shared_memory(void);
static void free_shared_memory(void);
static int init_message_queues(void);
static void cleanup_message_queues(void);
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
 * Initialize kernel-userspace communication
 */
int archangel_init_communication(void)
{
    int ret;
    
    archangel_info("Initializing kernel-userspace communication");
    
    mutex_lock(&comm_state.comm_lock);
    
    if (comm_state.initialized) {
        mutex_unlock(&comm_state.comm_lock);
        return 0;
    }
    
    /* Allocate shared memory */
    ret = alloc_shared_memory();
    if (ret) {
        archangel_err("Failed to allocate shared memory: %d", ret);
        goto error;
    }
    
    /* Initialize message queues */
    ret = init_message_queues();
    if (ret) {
        archangel_err("Failed to initialize message queues: %d", ret);
        goto error_shared_mem;
    }
    
    /* Create communication device in proc */
    comm_proc_entry = proc_create("archangel_comm", 0666, NULL, &archangel_comm_fops);
    if (!comm_proc_entry) {
        archangel_err("Failed to create communication proc entry");
        ret = -ENOMEM;
        goto error_queues;
    }
    
    /* Initialize statistics */
    atomic64_set(&comm_state.messages_sent, 0);
    atomic64_set(&comm_state.messages_received, 0);
    atomic64_set(&comm_state.queue_full_errors, 0);
    atomic64_set(&comm_state.queue_empty_errors, 0);
    
    comm_state.initialized = true;
    mutex_unlock(&comm_state.comm_lock);
    
    archangel_info("Communication initialized with %lu bytes shared memory", 
                   comm_state.shared_mem_size);
    
    return 0;

error_queues:
    cleanup_message_queues();
error_shared_mem:
    free_shared_memory();
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
 * Send message to userspace
 */
int archangel_send_message(enum archangel_msg_type type, const void *data, u32 size)
{
    struct archangel_message *msg;
    u32 total_size;
    int ret;
    
    if (!comm_state.initialized)
        return -ENOTCONN;
    
    if (size > ARCHANGEL_MAX_MSG_SIZE - sizeof(struct archangel_message))
        return -EMSGSIZE;
    
    total_size = sizeof(struct archangel_message) + size;
    msg = kmalloc(total_size, GFP_KERNEL);
    if (!msg)
        return -ENOMEM;
    
    /* Fill message header */
    msg->type = type;
    msg->sequence = atomic64_inc_return(&comm_state.messages_sent);
    msg->data_size = size;
    msg->timestamp = ktime_get_ns();
    msg->flags = 0;
    
    /* Copy data */
    if (data && size > 0)
        memcpy(msg->data, data, size);
    
    /* Enqueue message */
    ret = enqueue_message(comm_state.kernel_to_user_queue.messages,
                         &comm_state.kernel_to_user_queue.head,
                         &comm_state.kernel_to_user_queue.tail,
                         comm_state.kernel_to_user_queue.size,
                         &comm_state.kernel_to_user_queue.lock,
                         msg);
    
    if (ret == 0) {
        /* Wake up userspace readers */
        wake_up_interruptible(&comm_state.kernel_to_user_queue.wait_queue);
        
        archangel_debug("Sent message type %d, size %u to userspace", type, size);
    } else {
        atomic64_inc(&comm_state.queue_full_errors);
        archangel_warn("Failed to send message to userspace: queue full");
    }
    
    kfree(msg);
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
        *queue_full = atomic64_read(&comm_state.queue_full_errors);
    if (queue_empty)
        *queue_empty = atomic64_read(&comm_state.queue_empty_errors);
}