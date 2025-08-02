/*
 * Archangel Linux - Memory Protection and Process Monitoring
 * Advanced memory protection, heap analysis, and process behavior monitoring
 */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/mm.h>
#include <linux/mman.h>
#include <linux/sched.h>
#include <linux/sched/mm.h>
#include <linux/sched/task.h>
#include <linux/string.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/kprobes.h>
#include <linux/ptrace.h>
#include <linux/atomic.h>
#include <linux/spinlock.h>
#include <linux/rbtree.h>
#include <linux/hash.h>
#include <linux/ktime.h>

#include "../include/archangel.h"

/* Memory protection configuration */
#define MAX_TRACKED_PROCESSES    1024
#define MAX_MEMORY_REGIONS       512
#define SUSPICIOUS_ALLOC_SIZE    (1024 * 1024 * 100)  /* 100MB */
#define HEAP_SPRAY_THRESHOLD     20                    /* Allocations per second */
#define ROP_GADGET_SCAN_SIZE     4096                  /* Bytes to scan for ROP gadgets */

/* Memory region types */
enum memory_region_type {
    REGION_HEAP = 1,
    REGION_STACK,
    REGION_CODE,
    REGION_DATA,
    REGION_SHARED,
    REGION_UNKNOWN
};

/* Memory protection flags */
#define MEM_PROT_EXECUTABLE     0x0001
#define MEM_PROT_WRITABLE       0x0002
#define MEM_PROT_READABLE       0x0004
#define MEM_PROT_SUSPICIOUS     0x0008
#define MEM_PROT_HEAP_SPRAY     0x0010
#define MEM_PROT_ROP_GADGETS    0x0020
#define MEM_PROT_SHELLCODE      0x0040

/* Memory region structure */
struct memory_region {
    unsigned long start_addr;
    unsigned long end_addr;
    enum memory_region_type type;
    u32 protection_flags;
    u32 access_count;
    u64 last_access;
    u64 creation_time;
    struct rb_node rb_node;
};

/* Process memory profile */
struct process_memory_profile {
    u32 pid;
    char comm[16];
    
    /* Memory statistics */
    atomic64_t total_allocations;
    atomic64_t total_frees;
    atomic64_t current_heap_size;
    atomic64_t peak_heap_size;
    atomic64_t suspicious_allocations;
    
    /* Behavior metrics */
    u32 alloc_rate_per_sec;
    u64 last_alloc_time;
    u32 recent_alloc_count;
    u64 alloc_window_start;
    
    /* Memory regions */
    struct rb_root memory_regions;
    spinlock_t regions_lock;
    
    /* Hash table linkage */
    struct hlist_node hash_node;
    u64 creation_time;
    atomic_t ref_count;
};

/* Memory protection state */
static struct {
    bool enabled;
    atomic64_t total_allocations;
    atomic64_t suspicious_activities;
    atomic64_t blocked_allocations;
    
    /* Process tracking */
    struct hlist_head process_hash[256];
    spinlock_t process_hash_lock;
    atomic_t tracked_processes;
    
    /* Kprobes for memory operations */
    struct kprobe kmalloc_probe;
    struct kprobe kfree_probe;
    struct kprobe mmap_probe;
    struct kprobe munmap_probe;
    bool probes_registered;
    
} memory_state = {
    .enabled = false,
    .process_hash_lock = __SPIN_LOCK_UNLOCKED(memory_state.process_hash_lock),
};

/* Forward declarations */
static struct process_memory_profile *get_process_profile(u32 pid);
static void put_process_profile(struct process_memory_profile *profile);
static int analyze_memory_allocation(unsigned long addr, size_t size, gfp_t flags);
static int analyze_memory_free(unsigned long addr);
static int analyze_memory_mapping(unsigned long addr, size_t len, int prot);
static bool detect_heap_spray(struct process_memory_profile *profile);
static bool detect_rop_gadgets(unsigned long addr, size_t size);
static bool detect_shellcode_patterns(unsigned long addr, size_t size);
static int memory_protection_decision(struct process_memory_profile *profile, 
                                    unsigned long addr, size_t size, u32 flags);

/* Kprobe handlers */
static int kmalloc_handler(struct kprobe *p, struct pt_regs *regs);
static int kfree_handler(struct kprobe *p, struct pt_regs *regs);
static int mmap_handler(struct kprobe *p, struct pt_regs *regs);
static int munmap_handler(struct kprobe *p, struct pt_regs *regs);

/*
 * Initialize memory protection system
 */
int archangel_memory_init(void)
{
    int ret, i;
    
    archangel_info("Initializing memory protection system");
    
    /* Initialize process hash table */
    for (i = 0; i < ARRAY_SIZE(memory_state.process_hash); i++) {
        INIT_HLIST_HEAD(&memory_state.process_hash[i]);
    }
    
    atomic_set(&memory_state.tracked_processes, 0);
    atomic64_set(&memory_state.total_allocations, 0);
    atomic64_set(&memory_state.suspicious_activities, 0);
    atomic64_set(&memory_state.blocked_allocations, 0);
    
    /* Register kprobes for memory operations */
    memory_state.kmalloc_probe.symbol_name = "__kmalloc";
    memory_state.kmalloc_probe.pre_handler = kmalloc_handler;
    ret = register_kprobe(&memory_state.kmalloc_probe);
    if (ret < 0) {
        archangel_warn("Failed to register kmalloc kprobe: %d", ret);
    }
    
    memory_state.kfree_probe.symbol_name = "kfree";
    memory_state.kfree_probe.pre_handler = kfree_handler;
    ret = register_kprobe(&memory_state.kfree_probe);
    if (ret < 0) {
        archangel_warn("Failed to register kfree kprobe: %d", ret);
    }
    
    memory_state.mmap_probe.symbol_name = "do_mmap";
    memory_state.mmap_probe.pre_handler = mmap_handler;
    ret = register_kprobe(&memory_state.mmap_probe);
    if (ret < 0) {
        archangel_warn("Failed to register mmap kprobe: %d", ret);
    }
    
    memory_state.munmap_probe.symbol_name = "do_munmap";
    memory_state.munmap_probe.pre_handler = munmap_handler;
    ret = register_kprobe(&memory_state.munmap_probe);
    if (ret < 0) {
        archangel_warn("Failed to register munmap kprobe: %d", ret);
    }
    
    memory_state.probes_registered = true;
    memory_state.enabled = true;
    
    archangel_info("Memory protection system initialized");
    return 0;
}

/*
 * Cleanup memory protection system
 */
void archangel_memory_cleanup(void)
{
    struct process_memory_profile *profile;
    struct hlist_node *tmp;
    int i;
    
    archangel_info("Cleaning up memory protection system");
    
    memory_state.enabled = false;
    
    /* Unregister kprobes */
    if (memory_state.probes_registered) {
        unregister_kprobe(&memory_state.kmalloc_probe);
        unregister_kprobe(&memory_state.kfree_probe);
        unregister_kprobe(&memory_state.mmap_probe);
        unregister_kprobe(&memory_state.munmap_probe);
        memory_state.probes_registered = false;
    }
    
    /* Cleanup process profiles */
    spin_lock(&memory_state.process_hash_lock);
    for (i = 0; i < ARRAY_SIZE(memory_state.process_hash); i++) {
        hlist_for_each_entry_safe(profile, tmp, &memory_state.process_hash[i], hash_node) {
            hlist_del(&profile->hash_node);
            put_process_profile(profile);
        }
    }
    spin_unlock(&memory_state.process_hash_lock);
    
    archangel_info("Memory protection cleaned up, tracked %lld allocations, "
                   "detected %lld suspicious activities",
                   atomic64_read(&memory_state.total_allocations),
                   atomic64_read(&memory_state.suspicious_activities));
}

/*
 * Get or create process memory profile
 */
static struct process_memory_profile *get_process_profile(u32 pid)
{
    struct process_memory_profile *profile;
    u32 hash = hash_32(pid, 8);
    
    /* Try to find existing profile */
    spin_lock(&memory_state.process_hash_lock);
    hlist_for_each_entry(profile, &memory_state.process_hash[hash], hash_node) {
        if (profile->pid == pid) {
            atomic_inc(&profile->ref_count);
            spin_unlock(&memory_state.process_hash_lock);
            return profile;
        }
    }
    spin_unlock(&memory_state.process_hash_lock);
    
    /* Create new profile */
    profile = kzalloc(sizeof(*profile), GFP_ATOMIC);
    if (!profile)
        return NULL;
    
    profile->pid = pid;
    strncpy(profile->comm, current->comm, sizeof(profile->comm) - 1);
    profile->memory_regions = RB_ROOT;
    spin_lock_init(&profile->regions_lock);
    atomic_set(&profile->ref_count, 1);
    profile->creation_time = ktime_get_ns();
    profile->alloc_window_start = profile->creation_time;
    
    /* Add to hash table */
    spin_lock(&memory_state.process_hash_lock);
    hlist_add_head(&profile->hash_node, &memory_state.process_hash[hash]);
    atomic_inc(&memory_state.tracked_processes);
    spin_unlock(&memory_state.process_hash_lock);
    
    return profile;
}

/*
 * Release process memory profile reference
 */
static void put_process_profile(struct process_memory_profile *profile)
{
    if (!profile)
        return;
    
    if (atomic_dec_and_test(&profile->ref_count)) {
        /* Free memory regions */
        struct rb_node *node;
        struct memory_region *region;
        
        while ((node = rb_first(&profile->memory_regions))) {
            region = rb_entry(node, struct memory_region, rb_node);
            rb_erase(node, &profile->memory_regions);
            kfree(region);
        }
        
        kfree(profile);
        atomic_dec(&memory_state.tracked_processes);
    }
}

/*
 * Analyze memory allocation for threats
 */
static int analyze_memory_allocation(unsigned long addr, size_t size, gfp_t flags)
{
    struct process_memory_profile *profile;
    u64 current_time;
    bool suspicious = false;
    u32 prot_flags = 0;
    
    if (!memory_state.enabled)
        return 0;
    
    atomic64_inc(&memory_state.total_allocations);
    current_time = ktime_get_ns();
    
    profile = get_process_profile(current->pid);
    if (!profile)
        return 0;
    
    /* Update allocation statistics */
    atomic64_inc(&profile->total_allocations);
    atomic64_add(size, &profile->current_heap_size);
    
    if (atomic64_read(&profile->current_heap_size) > atomic64_read(&profile->peak_heap_size)) {
        atomic64_set(&profile->peak_heap_size, atomic64_read(&profile->current_heap_size));
    }
    
    /* Update allocation rate tracking */
    if (current_time - profile->alloc_window_start > 1000000000ULL) { /* 1 second window */
        profile->alloc_rate_per_sec = profile->recent_alloc_count;
        profile->recent_alloc_count = 0;
        profile->alloc_window_start = current_time;
    }
    profile->recent_alloc_count++;
    profile->last_alloc_time = current_time;
    
    /* Check for suspicious patterns */
    
    /* Large allocation detection */
    if (size > SUSPICIOUS_ALLOC_SIZE) {
        prot_flags |= MEM_PROT_SUSPICIOUS;
        suspicious = true;
        archangel_warn("Large allocation detected: PID %u, size %zu bytes", 
                      current->pid, size);
    }
    
    /* Heap spray detection */
    if (detect_heap_spray(profile)) {
        prot_flags |= MEM_PROT_HEAP_SPRAY;
        suspicious = true;
        archangel_warn("Potential heap spray detected: PID %u, rate %u/sec",
                      current->pid, profile->alloc_rate_per_sec);
    }
    
    /* ROP gadget detection */
    if (detect_rop_gadgets(addr, size)) {
        prot_flags |= MEM_PROT_ROP_GADGETS;
        suspicious = true;
        archangel_warn("ROP gadgets detected in allocation: PID %u, addr 0x%lx",
                      current->pid, addr);
    }
    
    /* Shellcode pattern detection */
    if (detect_shellcode_patterns(addr, size)) {
        prot_flags |= MEM_PROT_SHELLCODE;
        suspicious = true;
        archangel_warn("Shellcode patterns detected: PID %u, addr 0x%lx",
                      current->pid, addr);
    }
    
    if (suspicious) {
        atomic64_inc(&profile->suspicious_allocations);
        atomic64_inc(&memory_state.suspicious_activities);
        
        /* Make protection decision */
        int decision = memory_protection_decision(profile, addr, size, prot_flags);
        if (decision == ARCHANGEL_DENY) {
            atomic64_inc(&memory_state.blocked_allocations);
            put_process_profile(profile);
            return -EPERM;
        }
    }
    
    put_process_profile(profile);
    return 0;
}

/*
 * Analyze memory free operations
 */
static int analyze_memory_free(unsigned long addr)
{
    struct process_memory_profile *profile;
    
    if (!memory_state.enabled || !addr)
        return 0;
    
    profile = get_process_profile(current->pid);
    if (!profile)
        return 0;
    
    atomic64_inc(&profile->total_frees);
    
    /* Note: In a full implementation, we would track individual allocations
     * to properly update current_heap_size on free */
    
    put_process_profile(profile);
    return 0;
}

/*
 * Detect heap spray attacks
 */
static bool detect_heap_spray(struct process_memory_profile *profile)
{
    return profile->alloc_rate_per_sec > HEAP_SPRAY_THRESHOLD;
}

/*
 * Detect ROP gadgets in memory
 */
static bool detect_rop_gadgets(unsigned long addr, size_t size)
{
    /* Simplified ROP gadget detection - looks for common x86_64 patterns */
    u8 *data;
    size_t scan_size = min(size, (size_t)ROP_GADGET_SCAN_SIZE);
    int i;
    int gadget_count = 0;
    
    if (!addr || !size)
        return false;
    
    /* Map the memory for scanning */
    data = (u8 *)addr;
    
    /* Look for common ROP gadget patterns */
    for (i = 0; i < scan_size - 2; i++) {
        /* pop rdi; ret (0x5f 0xc3) */
        if (data[i] == 0x5f && data[i+1] == 0xc3)
            gadget_count++;
        /* pop rsi; ret (0x5e 0xc3) */
        else if (data[i] == 0x5e && data[i+1] == 0xc3)
            gadget_count++;
        /* pop rdx; ret (0x5a 0xc3) */
        else if (data[i] == 0x5a && data[i+1] == 0xc3)
            gadget_count++;
        /* mov rdi, rsp; ret (0x48 0x89 0xe7 0xc3) */
        else if (i < scan_size - 3 && data[i] == 0x48 && data[i+1] == 0x89 && 
                 data[i+2] == 0xe7 && data[i+3] == 0xc3)
            gadget_count++;
    }
    
    /* Threshold for ROP chain detection */
    return gadget_count > 5;
}

/*
 * Detect shellcode patterns
 */
static bool detect_shellcode_patterns(unsigned long addr, size_t size)
{
    u8 *data;
    size_t scan_size = min(size, (size_t)ROP_GADGET_SCAN_SIZE);
    int i;
    int pattern_count = 0;
    
    if (!addr || !size)
        return false;
    
    data = (u8 *)addr;
    
    /* Look for common shellcode patterns */
    for (i = 0; i < scan_size - 4; i++) {
        /* NOP sled (0x90) */
        if (data[i] == 0x90 && data[i+1] == 0x90 && data[i+2] == 0x90)
            pattern_count++;
        /* execve syscall pattern (0x48 0x31 0xc0 0xb0 0x3b) */
        else if (data[i] == 0x48 && data[i+1] == 0x31 && data[i+2] == 0xc0 && 
                 data[i+3] == 0xb0 && data[i+4] == 0x3b)
            pattern_count += 5;
        /* Common shellcode header (0xeb 0x??) - short jump */
        else if (data[i] == 0xeb)
            pattern_count++;
    }
    
    return pattern_count > 10;
}

/*
 * Make memory protection decision
 */
static int memory_protection_decision(struct process_memory_profile *profile, 
                                    unsigned long addr, size_t size, u32 flags)
{
    struct archangel_security_context *ctx;
    enum archangel_decision decision;
    
    /* Create security context for AI analysis */
    ctx = kmalloc(sizeof(*ctx) + 64, GFP_ATOMIC);
    if (!ctx)
        return ARCHANGEL_ALLOW; /* Fail open */
    
    memset(ctx, 0, sizeof(*ctx) + 64);
    ctx->pid = profile->pid;
    ctx->uid = current_uid().val;
    ctx->syscall_nr = -1; /* Not a syscall */
    ctx->timestamp = ktime_get_ns();
    ctx->flags = flags;
    strncpy(ctx->comm, profile->comm, sizeof(ctx->comm) - 1);
    ctx->data_size = 64;
    
    /* Add memory-specific data */
    *((unsigned long *)ctx->data) = addr;
    *((size_t *)(ctx->data + 8)) = size;
    *((u64 *)(ctx->data + 16)) = atomic64_read(&profile->suspicious_allocations);
    *((u32 *)(ctx->data + 24)) = profile->alloc_rate_per_sec;
    
    /* Get AI decision */
    decision = archangel_make_decision(ctx);
    
    kfree(ctx);
    return decision;
}

/* Kprobe handlers */
static int kmalloc_handler(struct kprobe *p, struct pt_regs *regs)
{
    size_t size;
    gfp_t flags;
    
#ifdef CONFIG_X86_64
    size = regs->di;    /* First argument */
    flags = regs->si;   /* Second argument */
#else
    return 0; /* Skip on other architectures */
#endif
    
    /* Analyze the allocation */
    return analyze_memory_allocation(0, size, flags); /* Address not available in pre_handler */
}

static int kfree_handler(struct kprobe *p, struct pt_regs *regs)
{
    unsigned long addr;
    
#ifdef CONFIG_X86_64
    addr = regs->di;    /* First argument */
#else
    return 0;
#endif
    
    return analyze_memory_free(addr);
}

static int mmap_handler(struct kprobe *p, struct pt_regs *regs)
{
    unsigned long addr, len;
    int prot;
    
#ifdef CONFIG_X86_64
    addr = regs->di;    /* addr */
    len = regs->si;     /* len */
    prot = regs->dx;    /* prot */
#else
    return 0;
#endif
    
    return analyze_memory_mapping(addr, len, prot);
}

static int munmap_handler(struct kprobe *p, struct pt_regs *regs)
{
    /* For now, just track munmap calls */
    return 0;
}

/*
 * Analyze memory mapping operations
 */
static int analyze_memory_mapping(unsigned long addr, size_t len, int prot)
{
    struct process_memory_profile *profile;
    bool suspicious = false;
    u32 prot_flags = 0;
    
    if (!memory_state.enabled)
        return 0;
    
    profile = get_process_profile(current->pid);
    if (!profile)
        return 0;
    
    /* Check for suspicious mapping patterns */
    if ((prot & PROT_READ) && (prot & PROT_WRITE) && (prot & PROT_EXEC)) {
        /* RWX mappings are highly suspicious */
        prot_flags |= MEM_PROT_EXECUTABLE | MEM_PROT_WRITABLE | MEM_PROT_SUSPICIOUS;
        suspicious = true;
        archangel_warn("RWX memory mapping: PID %u, addr 0x%lx, len %zu",
                      current->pid, addr, len);
    }
    
    if (suspicious) {
        atomic64_inc(&profile->suspicious_allocations);
        atomic64_inc(&memory_state.suspicious_activities);
        
        int decision = memory_protection_decision(profile, addr, len, prot_flags);
        if (decision == ARCHANGEL_DENY) {
            put_process_profile(profile);
            return -EPERM;
        }
    }
    
    put_process_profile(profile);
    return 0;
}

/*
 * Get memory protection statistics
 */
void archangel_memory_get_stats(u64 *total_allocs, u64 *suspicious_activities, 
                               u64 *blocked_allocs, u32 *tracked_processes)
{
    if (total_allocs)
        *total_allocs = atomic64_read(&memory_state.total_allocations);
    if (suspicious_activities)
        *suspicious_activities = atomic64_read(&memory_state.suspicious_activities);
    if (blocked_allocs)
        *blocked_allocs = atomic64_read(&memory_state.blocked_allocations);
    if (tracked_processes)
        *tracked_processes = atomic_read(&memory_state.tracked_processes);
}

/*
 * Enable/disable memory protection
 */
void archangel_memory_set_enabled(bool enabled)
{
    memory_state.enabled = enabled;
    archangel_info("Memory protection %s", enabled ? "enabled" : "disabled");
}

/*
 * Check if memory protection is enabled
 */
bool archangel_memory_is_enabled(void)
{
    return memory_state.enabled;
}