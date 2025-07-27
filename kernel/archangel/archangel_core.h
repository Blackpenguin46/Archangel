#ifndef _ARCHANGEL_CORE_H
#define _ARCHANGEL_CORE_H

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/atomic.h>
#include <linux/wait.h>
#include <linux/spinlock.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/uaccess.h>

/* Version information */
#define ARCHANGEL_VERSION_MAJOR 1
#define ARCHANGEL_VERSION_MINOR 0
#define ARCHANGEL_VERSION_PATCH 0

/* Performance constraints */
#define ARCHANGEL_MAX_INFERENCE_NS 1000000  /* 1ms max */
#define ARCHANGEL_MAX_MEMORY_KB 10240       /* 10MB max */
#define ARCHANGEL_MAX_CPU_PERCENT 5         /* 5% max */

/* Communication buffer sizes */
#define ARCHANGEL_RING_BUFFER_SIZE 4096
#define ARCHANGEL_DMA_THRESHOLD 1024

/* Forward declarations */
struct archangel_ai_engine;
struct archangel_comm_channel;
struct archangel_stats;

/* AI Engine types */
enum archangel_engine_type {
    ARCHANGEL_ENGINE_SYSCALL = 0,
    ARCHANGEL_ENGINE_NETWORK,
    ARCHANGEL_ENGINE_MEMORY,
    ARCHANGEL_ENGINE_PROCESS,
    ARCHANGEL_ENGINE_MAX
};

/* AI Engine status */
enum archangel_engine_status {
    ARCHANGEL_ENGINE_INACTIVE = 0,
    ARCHANGEL_ENGINE_ACTIVE,
    ARCHANGEL_ENGINE_ERROR,
    ARCHANGEL_ENGINE_OVERLOAD
};

/**
 * struct archangel_ai_engine - Base AI engine structure
 * @type: Engine type identifier
 * @status: Current engine status
 * @inference_count: Number of inferences performed
 * @avg_inference_time_ns: Average inference time in nanoseconds
 * @memory_usage_kb: Current memory usage in KB
 * @cpu_usage_percent: Current CPU usage percentage
 * @last_inference_time: Timestamp of last inference
 * @lock: Engine-specific spinlock
 */
struct archangel_ai_engine {
    enum archangel_engine_type type;
    enum archangel_engine_status status;
    atomic64_t inference_count;
    u64 avg_inference_time_ns;
    u32 memory_usage_kb;
    u8 cpu_usage_percent;
    u64 last_inference_time;
    spinlock_t lock;
};

/**
 * struct archangel_resource_monitor - Resource usage monitoring
 * @total_memory_kb: Total memory allocated by all engines
 * @peak_memory_kb: Peak memory usage
 * @total_cpu_percent: Total CPU usage across all engines
 * @active_engines: Bitmask of active engines
 * @overload_count: Number of resource overload events
 * @last_check_time: Last resource check timestamp
 */
struct archangel_resource_monitor {
    atomic_t total_memory_kb;
    u32 peak_memory_kb;
    atomic_t total_cpu_percent;
    unsigned long active_engines;
    atomic64_t overload_count;
    u64 last_check_time;
};

/**
 * struct archangel_kernel_ai - Main AI coordination structure
 * @engines: AI engines for different subsystems
 * @comm: Communication channels with userspace
 * @limits: Real-time performance constraints
 * @stats: Runtime statistics
 * @resource_monitor: Resource usage monitoring
 * @lock: Spinlock for thread-safe access
 * @initialized: Initialization status
 */
struct archangel_kernel_ai {
    /* AI engines */
    struct {
        struct archangel_ai_engine *syscall_filter;
        struct archangel_ai_engine *network_ids;
        struct archangel_ai_engine *memory_patterns;
        struct archangel_ai_engine *process_monitor;
    } engines;
    
    /* Communication channels */
    struct {
        void *to_userspace;
        void *from_userspace;
        void *zero_copy_pool;
        wait_queue_head_t wait_queue;
    } comm;
    
    /* Real-time constraints */
    struct {
        u64 max_inference_ns;
        u32 max_memory_kb;
        u8 max_cpu_percent;
    } limits;
    
    /* Statistics */
    struct {
        atomic64_t inferences;
        atomic64_t cache_hits;
        atomic64_t deferrals;
        atomic64_t blocks;
    } stats;
    
    /* Resource monitoring */
    struct archangel_resource_monitor resource_monitor;
    
    spinlock_t lock;
    bool initialized;
};

/* Global AI instance */
extern struct archangel_kernel_ai *archangel_ai;

/* Core functions */
int archangel_core_init(void);
void archangel_core_exit(void);
int archangel_ai_initialize(void);
void archangel_ai_cleanup(void);

/* Statistics and monitoring */
int archangel_stats_init(void);
void archangel_stats_cleanup(void);
void archangel_stats_update(const char *event, u64 value);

/* Communication interface */
int archangel_comm_init(void);
void archangel_comm_cleanup(void);

/* AI Engine management */
int archangel_engine_init(struct archangel_ai_engine **engine, enum archangel_engine_type type);
void archangel_engine_cleanup(struct archangel_ai_engine *engine);
int archangel_engine_register(struct archangel_ai_engine *engine);
void archangel_engine_unregister(struct archangel_ai_engine *engine);
void archangel_engine_update_stats(struct archangel_ai_engine *engine, u64 inference_time_ns);
bool archangel_engine_check_limits(struct archangel_ai_engine *engine);

/* Resource management */
int archangel_resource_monitor_init(void);
void archangel_resource_monitor_cleanup(void);
void archangel_resource_monitor_update(void);
bool archangel_resource_check_limits(void);

/* Utility functions */
static inline bool archangel_is_initialized(void)
{
    return archangel_ai && archangel_ai->initialized;
}

static inline void archangel_inc_stat(atomic64_t *stat)
{
    atomic64_inc(stat);
}

#endif /* _ARCHANGEL_CORE_H */