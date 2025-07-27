#include "archangel_core.h"
#include "archangel_comm.h"

/* Module information */
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Archangel Linux Development Team");
MODULE_DESCRIPTION("Archangel AI Security Operating System - Core Kernel Module");
MODULE_VERSION("1.0.0");

/* Global AI instance */
struct archangel_kernel_ai *archangel_ai = NULL;

/* Proc filesystem entry */
static struct proc_dir_entry *archangel_proc_dir = NULL;
static struct proc_dir_entry *archangel_stats_entry = NULL;
static struct proc_dir_entry *archangel_comm_entry = NULL;

/**
 * archangel_ai_initialize - Initialize the AI coordination structure
 * 
 * Returns: 0 on success, negative error code on failure
 */
int archangel_ai_initialize(void)
{
    int ret;
    
    if (archangel_ai) {
        pr_warn("archangel_core: AI already initialized\n");
        return -EEXIST;
    }
    
    archangel_ai = kzalloc(sizeof(*archangel_ai), GFP_KERNEL);
    if (!archangel_ai) {
        pr_err("archangel_core: Failed to allocate AI structure\n");
        return -ENOMEM;
    }
    
    /* Initialize performance limits */
    archangel_ai->limits.max_inference_ns = ARCHANGEL_MAX_INFERENCE_NS;
    archangel_ai->limits.max_memory_kb = ARCHANGEL_MAX_MEMORY_KB;
    archangel_ai->limits.max_cpu_percent = ARCHANGEL_MAX_CPU_PERCENT;
    
    /* Initialize statistics */
    atomic64_set(&archangel_ai->stats.inferences, 0);
    atomic64_set(&archangel_ai->stats.cache_hits, 0);
    atomic64_set(&archangel_ai->stats.deferrals, 0);
    atomic64_set(&archangel_ai->stats.blocks, 0);
    
    /* Initialize synchronization */
    spin_lock_init(&archangel_ai->lock);
    init_waitqueue_head(&archangel_ai->comm.wait_queue);
    
    /* Initialize resource monitor */
    ret = archangel_resource_monitor_init();
    if (ret) {
        pr_err("archangel_core: Failed to initialize resource monitor: %d\n", ret);
        kfree(archangel_ai);
        archangel_ai = NULL;
        return ret;
    }
    
    /* Initialize AI engines */
    ret = archangel_engine_init(&archangel_ai->engines.syscall_filter, ARCHANGEL_ENGINE_SYSCALL);
    if (ret) {
        pr_err("archangel_core: Failed to initialize syscall engine: %d\n", ret);
        goto cleanup_resource_monitor;
    }
    
    ret = archangel_engine_init(&archangel_ai->engines.network_ids, ARCHANGEL_ENGINE_NETWORK);
    if (ret) {
        pr_err("archangel_core: Failed to initialize network engine: %d\n", ret);
        goto cleanup_syscall_engine;
    }
    
    ret = archangel_engine_init(&archangel_ai->engines.memory_patterns, ARCHANGEL_ENGINE_MEMORY);
    if (ret) {
        pr_err("archangel_core: Failed to initialize memory engine: %d\n", ret);
        goto cleanup_network_engine;
    }
    
    ret = archangel_engine_init(&archangel_ai->engines.process_monitor, ARCHANGEL_ENGINE_PROCESS);
    if (ret) {
        pr_err("archangel_core: Failed to initialize process engine: %d\n", ret);
        goto cleanup_memory_engine;
    }
    
    archangel_ai->initialized = true;
    
    pr_info("archangel_core: AI coordination structure initialized with %d engines\n", ARCHANGEL_ENGINE_MAX);
    return 0;

cleanup_memory_engine:
    archangel_engine_cleanup(archangel_ai->engines.memory_patterns);
cleanup_network_engine:
    archangel_engine_cleanup(archangel_ai->engines.network_ids);
cleanup_syscall_engine:
    archangel_engine_cleanup(archangel_ai->engines.syscall_filter);
cleanup_resource_monitor:
    archangel_resource_monitor_cleanup();
    kfree(archangel_ai);
    archangel_ai = NULL;
    return ret;
}

/**
 * archangel_ai_cleanup - Clean up the AI coordination structure
 */
void archangel_ai_cleanup(void)
{
    if (!archangel_ai)
        return;
    
    archangel_ai->initialized = false;
    
    /* Clean up AI engines */
    if (archangel_ai->engines.process_monitor) {
        archangel_engine_cleanup(archangel_ai->engines.process_monitor);
        archangel_ai->engines.process_monitor = NULL;
    }
    
    if (archangel_ai->engines.memory_patterns) {
        archangel_engine_cleanup(archangel_ai->engines.memory_patterns);
        archangel_ai->engines.memory_patterns = NULL;
    }
    
    if (archangel_ai->engines.network_ids) {
        archangel_engine_cleanup(archangel_ai->engines.network_ids);
        archangel_ai->engines.network_ids = NULL;
    }
    
    if (archangel_ai->engines.syscall_filter) {
        archangel_engine_cleanup(archangel_ai->engines.syscall_filter);
        archangel_ai->engines.syscall_filter = NULL;
    }
    
    /* Clean up resource monitor */
    archangel_resource_monitor_cleanup();
    
    /* Clean up communication channels (placeholders for now) */
    archangel_ai->comm.to_userspace = NULL;
    archangel_ai->comm.from_userspace = NULL;
    archangel_ai->comm.zero_copy_pool = NULL;
    
    kfree(archangel_ai);
    archangel_ai = NULL;
    
    pr_info("archangel_core: AI coordination structure cleaned up\n");
}

/**
 * archangel_stats_show - Show statistics in proc filesystem
 */
static int archangel_stats_show(struct seq_file *m, void *v)
{
    const char *engine_names[] = {"Syscall", "Network", "Memory", "Process"};
    const char *status_names[] = {"Inactive", "Active", "Error", "Overload"};
    struct archangel_ai_engine *engines[ARCHANGEL_ENGINE_MAX];
    int i;
    
    if (!archangel_is_initialized()) {
        seq_printf(m, "Archangel AI: Not initialized\n");
        return 0;
    }
    
    /* Update resource monitoring before displaying stats */
    archangel_resource_monitor_update();
    
    seq_printf(m, "Archangel AI Statistics:\n");
    seq_printf(m, "  Version: %d.%d.%d\n", 
               ARCHANGEL_VERSION_MAJOR, 
               ARCHANGEL_VERSION_MINOR, 
               ARCHANGEL_VERSION_PATCH);
    seq_printf(m, "  Status: %s\n", 
               archangel_ai->initialized ? "Initialized" : "Not initialized");
    
    seq_printf(m, "\nGlobal Statistics:\n");
    seq_printf(m, "  Total inferences: %llu\n", 
               atomic64_read(&archangel_ai->stats.inferences));
    seq_printf(m, "  Cache hits: %llu\n", 
               atomic64_read(&archangel_ai->stats.cache_hits));
    seq_printf(m, "  Deferrals: %llu\n", 
               atomic64_read(&archangel_ai->stats.deferrals));
    seq_printf(m, "  Blocks: %llu\n", 
               atomic64_read(&archangel_ai->stats.blocks));
    
    seq_printf(m, "\nResource Usage:\n");
    seq_printf(m, "  Total memory: %u KB\n", 
               atomic_read(&archangel_ai->resource_monitor.total_memory_kb));
    seq_printf(m, "  Peak memory: %u KB\n", 
               archangel_ai->resource_monitor.peak_memory_kb);
    seq_printf(m, "  Total CPU: %u%%\n", 
               atomic_read(&archangel_ai->resource_monitor.total_cpu_percent));
    seq_printf(m, "  Active engines: 0x%lx\n", 
               archangel_ai->resource_monitor.active_engines);
    seq_printf(m, "  Overload events: %llu\n", 
               atomic64_read(&archangel_ai->resource_monitor.overload_count));
    
    seq_printf(m, "\nResource Limits:\n");
    seq_printf(m, "  Max inference time: %llu ns (1ms)\n", 
               archangel_ai->limits.max_inference_ns);
    seq_printf(m, "  Max memory: %u KB (10MB)\n", 
               archangel_ai->limits.max_memory_kb);
    seq_printf(m, "  Max CPU: %u%%\n", 
               archangel_ai->limits.max_cpu_percent);
    
    /* Collect engine pointers */
    engines[ARCHANGEL_ENGINE_SYSCALL] = archangel_ai->engines.syscall_filter;
    engines[ARCHANGEL_ENGINE_NETWORK] = archangel_ai->engines.network_ids;
    engines[ARCHANGEL_ENGINE_MEMORY] = archangel_ai->engines.memory_patterns;
    engines[ARCHANGEL_ENGINE_PROCESS] = archangel_ai->engines.process_monitor;
    
    seq_printf(m, "\nAI Engines:\n");
    for (i = 0; i < ARCHANGEL_ENGINE_MAX; i++) {
        struct archangel_ai_engine *engine = engines[i];
        if (!engine) {
            seq_printf(m, "  %s Engine: Not initialized\n", engine_names[i]);
            continue;
        }
        
        seq_printf(m, "  %s Engine:\n", engine_names[i]);
        seq_printf(m, "    Status: %s\n", 
                   engine->status < 4 ? status_names[engine->status] : "Unknown");
        seq_printf(m, "    Inferences: %llu\n", 
                   atomic64_read(&engine->inference_count));
        seq_printf(m, "    Avg inference time: %llu ns\n", 
                   engine->avg_inference_time_ns);
        seq_printf(m, "    Memory usage: %u KB\n", 
                   engine->memory_usage_kb);
        seq_printf(m, "    CPU usage: %u%%\n", 
                   engine->cpu_usage_percent);
        if (engine->last_inference_time > 0) {
            u64 time_since = ktime_get_ns() - engine->last_inference_time;
            seq_printf(m, "    Last inference: %llu ns ago\n", time_since);
        } else {
            seq_printf(m, "    Last inference: Never\n");
        }
    }
    
    return 0;
}

static int archangel_stats_open(struct inode *inode, struct file *file)
{
    return single_open(file, archangel_stats_show, NULL);
}

static const struct proc_ops archangel_stats_ops = {
    .proc_open = archangel_stats_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

/**
 * archangel_comm_show - Show communication statistics in proc filesystem
 */
static int archangel_comm_show(struct seq_file *m, void *v)
{
    int i;
    
    if (!archangel_comm_is_initialized()) {
        seq_printf(m, "Archangel Communication: Not initialized\n");
        return 0;
    }
    
    seq_printf(m, "Archangel Communication Statistics:\n");
    seq_printf(m, "  Manager status: %s\n", 
               archangel_comm_mgr->initialized ? "Initialized" : "Not initialized");
    seq_printf(m, "  Active channels: %u/%u\n", 
               archangel_comm_mgr->channel_count, ARCHANGEL_COMM_MAX_CHANNELS);
    seq_printf(m, "  Default channel: %u\n", 
               archangel_comm_mgr->default_channel);
    
    seq_printf(m, "\nChannel Details:\n");
    for (i = 0; i < ARCHANGEL_COMM_MAX_CHANNELS; i++) {
        if (archangel_comm_mgr->channels[i].active) {
            archangel_comm_get_stats(i, m);
            seq_printf(m, "\n");
        }
    }
    
    return 0;
}

static int archangel_comm_open(struct inode *inode, struct file *file)
{
    return single_open(file, archangel_comm_show, NULL);
}

static const struct proc_ops archangel_comm_ops = {
    .proc_open = archangel_comm_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

/**
 * archangel_stats_init - Initialize statistics and proc filesystem
 */
int archangel_stats_init(void)
{
    archangel_proc_dir = proc_mkdir("archangel", NULL);
    if (!archangel_proc_dir) {
        pr_err("archangel_core: Failed to create proc directory\n");
        return -ENOMEM;
    }
    
    archangel_stats_entry = proc_create("stats", 0444, archangel_proc_dir, 
                                       &archangel_stats_ops);
    if (!archangel_stats_entry) {
        pr_err("archangel_core: Failed to create stats proc entry\n");
        proc_remove(archangel_proc_dir);
        return -ENOMEM;
    }
    
    archangel_comm_entry = proc_create("comm", 0444, archangel_proc_dir, 
                                      &archangel_comm_ops);
    if (!archangel_comm_entry) {
        pr_err("archangel_core: Failed to create comm proc entry\n");
        proc_remove(archangel_stats_entry);
        proc_remove(archangel_proc_dir);
        return -ENOMEM;
    }
    
    pr_info("archangel_core: Statistics interface initialized\n");
    return 0;
}

/**
 * archangel_stats_cleanup - Clean up statistics and proc filesystem
 */
void archangel_stats_cleanup(void)
{
    if (archangel_comm_entry) {
        proc_remove(archangel_comm_entry);
        archangel_comm_entry = NULL;
    }
    
    if (archangel_stats_entry) {
        proc_remove(archangel_stats_entry);
        archangel_stats_entry = NULL;
    }
    
    if (archangel_proc_dir) {
        proc_remove(archangel_proc_dir);
        archangel_proc_dir = NULL;
    }
    
    pr_info("archangel_core: Statistics interface cleaned up\n");
}

/**
 * archangel_stats_update - Update a statistic
 */
void archangel_stats_update(const char *event, u64 value)
{
    if (!archangel_is_initialized())
        return;
    
    /* Simple event-based statistics update */
    if (strcmp(event, "inference") == 0) {
        archangel_inc_stat(&archangel_ai->stats.inferences);
    } else if (strcmp(event, "cache_hit") == 0) {
        archangel_inc_stat(&archangel_ai->stats.cache_hits);
    } else if (strcmp(event, "deferral") == 0) {
        archangel_inc_stat(&archangel_ai->stats.deferrals);
    } else if (strcmp(event, "block") == 0) {
        archangel_inc_stat(&archangel_ai->stats.blocks);
    }
}

/**
 * archangel_comm_init - Initialize communication infrastructure
 */
int archangel_comm_init(void)
{
    int ret;
    
    if (!archangel_is_initialized()) {
        pr_err("archangel_core: AI not initialized for communication setup\n");
        return -EINVAL;
    }
    
    /* Initialize communication manager */
    ret = archangel_comm_manager_init();
    if (ret) {
        pr_err("archangel_core: Failed to initialize communication manager: %d\n", ret);
        return ret;
    }
    
    /* Update AI structure with communication references */
    if (archangel_comm_mgr && archangel_comm_mgr->channel_count > 0) {
        struct archangel_comm_channel *default_channel = 
            archangel_comm_channel_get(archangel_comm_mgr->default_channel);
        
        if (default_channel) {
            archangel_ai->comm.to_userspace = &default_channel->kernel_to_user;
            archangel_ai->comm.from_userspace = &default_channel->user_to_kernel;
            archangel_ai->comm.zero_copy_pool = &default_channel->dma_pool;
        }
    }
    
    pr_info("archangel_core: Communication infrastructure initialized with %u channels\n",
            archangel_comm_mgr ? archangel_comm_mgr->channel_count : 0);
    return 0;
}

/**
 * archangel_comm_cleanup - Clean up communication infrastructure
 */
void archangel_comm_cleanup(void)
{
    /* Clear communication references from AI structure */
    if (archangel_ai) {
        archangel_ai->comm.to_userspace = NULL;
        archangel_ai->comm.from_userspace = NULL;
        archangel_ai->comm.zero_copy_pool = NULL;
    }
    
    /* Clean up communication manager */
    archangel_comm_manager_cleanup();
    
    pr_info("archangel_core: Communication infrastructure cleaned up\n");
}

/**
 * archangel_engine_init - Initialize an AI engine
 * @engine: Pointer to engine pointer to initialize
 * @type: Type of engine to initialize
 * 
 * Returns: 0 on success, negative error code on failure
 */
int archangel_engine_init(struct archangel_ai_engine **engine, enum archangel_engine_type type)
{
    struct archangel_ai_engine *new_engine;
    
    if (!engine || type >= ARCHANGEL_ENGINE_MAX) {
        pr_err("archangel_core: Invalid engine parameters\n");
        return -EINVAL;
    }
    
    new_engine = kzalloc(sizeof(*new_engine), GFP_KERNEL);
    if (!new_engine) {
        pr_err("archangel_core: Failed to allocate engine structure\n");
        return -ENOMEM;
    }
    
    /* Initialize engine structure */
    new_engine->type = type;
    new_engine->status = ARCHANGEL_ENGINE_INACTIVE;
    atomic64_set(&new_engine->inference_count, 0);
    new_engine->avg_inference_time_ns = 0;
    new_engine->memory_usage_kb = 0;
    new_engine->cpu_usage_percent = 0;
    new_engine->last_inference_time = 0;
    spin_lock_init(&new_engine->lock);
    
    *engine = new_engine;
    
    pr_info("archangel_core: AI engine %d initialized\n", type);
    return 0;
}

/**
 * archangel_engine_cleanup - Clean up an AI engine
 * @engine: Engine to clean up
 */
void archangel_engine_cleanup(struct archangel_ai_engine *engine)
{
    if (!engine)
        return;
    
    engine->status = ARCHANGEL_ENGINE_INACTIVE;
    
    pr_info("archangel_core: AI engine %d cleaned up\n", engine->type);
    kfree(engine);
}

/**
 * archangel_engine_register - Register an engine as active
 * @engine: Engine to register
 * 
 * Returns: 0 on success, negative error code on failure
 */
int archangel_engine_register(struct archangel_ai_engine *engine)
{
    unsigned long flags;
    
    if (!engine || !archangel_is_initialized()) {
        return -EINVAL;
    }
    
    spin_lock_irqsave(&archangel_ai->lock, flags);
    
    /* Set engine as active in resource monitor */
    set_bit(engine->type, &archangel_ai->resource_monitor.active_engines);
    engine->status = ARCHANGEL_ENGINE_ACTIVE;
    
    spin_unlock_irqrestore(&archangel_ai->lock, flags);
    
    pr_info("archangel_core: AI engine %d registered as active\n", engine->type);
    return 0;
}

/**
 * archangel_engine_unregister - Unregister an active engine
 * @engine: Engine to unregister
 */
void archangel_engine_unregister(struct archangel_ai_engine *engine)
{
    unsigned long flags;
    
    if (!engine || !archangel_is_initialized())
        return;
    
    spin_lock_irqsave(&archangel_ai->lock, flags);
    
    /* Clear engine from active list */
    clear_bit(engine->type, &archangel_ai->resource_monitor.active_engines);
    engine->status = ARCHANGEL_ENGINE_INACTIVE;
    
    spin_unlock_irqrestore(&archangel_ai->lock, flags);
    
    pr_info("archangel_core: AI engine %d unregistered\n", engine->type);
}

/**
 * archangel_engine_update_stats - Update engine statistics after inference
 * @engine: Engine to update
 * @inference_time_ns: Time taken for inference in nanoseconds
 */
void archangel_engine_update_stats(struct archangel_ai_engine *engine, u64 inference_time_ns)
{
    unsigned long flags;
    u64 count;
    
    if (!engine)
        return;
    
    spin_lock_irqsave(&engine->lock, flags);
    
    /* Update inference count */
    atomic64_inc(&engine->inference_count);
    count = atomic64_read(&engine->inference_count);
    
    /* Update average inference time using exponential moving average */
    if (engine->avg_inference_time_ns == 0) {
        engine->avg_inference_time_ns = inference_time_ns;
    } else {
        /* EMA with alpha = 0.1 */
        engine->avg_inference_time_ns = (engine->avg_inference_time_ns * 9 + inference_time_ns) / 10;
    }
    
    engine->last_inference_time = ktime_get_ns();
    
    spin_unlock_irqrestore(&engine->lock, flags);
    
    /* Update global statistics */
    archangel_stats_update("inference", 1);
    
    /* Check if inference time exceeds limits */
    if (inference_time_ns > archangel_ai->limits.max_inference_ns) {
        pr_warn("archangel_core: Engine %d inference time %llu ns exceeds limit %llu ns\n",
                engine->type, inference_time_ns, archangel_ai->limits.max_inference_ns);
    }
}

/**
 * archangel_engine_check_limits - Check if engine is within resource limits
 * @engine: Engine to check
 * 
 * Returns: true if within limits, false otherwise
 */
bool archangel_engine_check_limits(struct archangel_ai_engine *engine)
{
    if (!engine || !archangel_is_initialized())
        return false;
    
    /* Check memory limit */
    if (engine->memory_usage_kb > archangel_ai->limits.max_memory_kb / ARCHANGEL_ENGINE_MAX) {
        engine->status = ARCHANGEL_ENGINE_OVERLOAD;
        pr_warn("archangel_core: Engine %d memory usage %u KB exceeds per-engine limit\n",
                engine->type, engine->memory_usage_kb);
        return false;
    }
    
    /* Check CPU limit */
    if (engine->cpu_usage_percent > archangel_ai->limits.max_cpu_percent / ARCHANGEL_ENGINE_MAX) {
        engine->status = ARCHANGEL_ENGINE_OVERLOAD;
        pr_warn("archangel_core: Engine %d CPU usage %u%% exceeds per-engine limit\n",
                engine->type, engine->cpu_usage_percent);
        return false;
    }
    
    /* Check inference time limit */
    if (engine->avg_inference_time_ns > archangel_ai->limits.max_inference_ns) {
        engine->status = ARCHANGEL_ENGINE_OVERLOAD;
        pr_warn("archangel_core: Engine %d average inference time %llu ns exceeds limit\n",
                engine->type, engine->avg_inference_time_ns);
        return false;
    }
    
    if (engine->status == ARCHANGEL_ENGINE_OVERLOAD) {
        engine->status = ARCHANGEL_ENGINE_ACTIVE;
    }
    
    return true;
}

/**
 * archangel_resource_monitor_init - Initialize resource monitoring
 * 
 * Returns: 0 on success, negative error code on failure
 */
int archangel_resource_monitor_init(void)
{
    if (!archangel_ai) {
        pr_err("archangel_core: AI structure not allocated for resource monitor\n");
        return -EINVAL;
    }
    
    /* Initialize resource monitor */
    atomic_set(&archangel_ai->resource_monitor.total_memory_kb, 0);
    archangel_ai->resource_monitor.peak_memory_kb = 0;
    atomic_set(&archangel_ai->resource_monitor.total_cpu_percent, 0);
    archangel_ai->resource_monitor.active_engines = 0;
    atomic64_set(&archangel_ai->resource_monitor.overload_count, 0);
    archangel_ai->resource_monitor.last_check_time = ktime_get_ns();
    
    pr_info("archangel_core: Resource monitor initialized\n");
    return 0;
}

/**
 * archangel_resource_monitor_cleanup - Clean up resource monitoring
 */
void archangel_resource_monitor_cleanup(void)
{
    if (!archangel_ai)
        return;
    
    /* Reset resource monitor */
    atomic_set(&archangel_ai->resource_monitor.total_memory_kb, 0);
    archangel_ai->resource_monitor.peak_memory_kb = 0;
    atomic_set(&archangel_ai->resource_monitor.total_cpu_percent, 0);
    archangel_ai->resource_monitor.active_engines = 0;
    
    pr_info("archangel_core: Resource monitor cleaned up\n");
}

/**
 * archangel_resource_monitor_update - Update resource usage statistics
 */
void archangel_resource_monitor_update(void)
{
    u32 total_memory = 0;
    u32 total_cpu = 0;
    unsigned long flags;
    
    if (!archangel_is_initialized())
        return;
    
    spin_lock_irqsave(&archangel_ai->lock, flags);
    
    /* Calculate total resource usage across all engines */
    if (archangel_ai->engines.syscall_filter) {
        total_memory += archangel_ai->engines.syscall_filter->memory_usage_kb;
        total_cpu += archangel_ai->engines.syscall_filter->cpu_usage_percent;
    }
    
    if (archangel_ai->engines.network_ids) {
        total_memory += archangel_ai->engines.network_ids->memory_usage_kb;
        total_cpu += archangel_ai->engines.network_ids->cpu_usage_percent;
    }
    
    if (archangel_ai->engines.memory_patterns) {
        total_memory += archangel_ai->engines.memory_patterns->memory_usage_kb;
        total_cpu += archangel_ai->engines.memory_patterns->cpu_usage_percent;
    }
    
    if (archangel_ai->engines.process_monitor) {
        total_memory += archangel_ai->engines.process_monitor->memory_usage_kb;
        total_cpu += archangel_ai->engines.process_monitor->cpu_usage_percent;
    }
    
    /* Update totals */
    atomic_set(&archangel_ai->resource_monitor.total_memory_kb, total_memory);
    atomic_set(&archangel_ai->resource_monitor.total_cpu_percent, total_cpu);
    
    /* Update peak memory usage */
    if (total_memory > archangel_ai->resource_monitor.peak_memory_kb) {
        archangel_ai->resource_monitor.peak_memory_kb = total_memory;
    }
    
    archangel_ai->resource_monitor.last_check_time = ktime_get_ns();
    
    spin_unlock_irqrestore(&archangel_ai->lock, flags);
}

/**
 * archangel_resource_check_limits - Check if system is within resource limits
 * 
 * Returns: true if within limits, false otherwise
 */
bool archangel_resource_check_limits(void)
{
    u32 total_memory, total_cpu;
    bool within_limits = true;
    
    if (!archangel_is_initialized())
        return false;
    
    /* Update resource usage first */
    archangel_resource_monitor_update();
    
    total_memory = atomic_read(&archangel_ai->resource_monitor.total_memory_kb);
    total_cpu = atomic_read(&archangel_ai->resource_monitor.total_cpu_percent);
    
    /* Check memory limit */
    if (total_memory > archangel_ai->limits.max_memory_kb) {
        pr_warn("archangel_core: Total memory usage %u KB exceeds limit %u KB\n",
                total_memory, archangel_ai->limits.max_memory_kb);
        atomic64_inc(&archangel_ai->resource_monitor.overload_count);
        within_limits = false;
    }
    
    /* Check CPU limit */
    if (total_cpu > archangel_ai->limits.max_cpu_percent) {
        pr_warn("archangel_core: Total CPU usage %u%% exceeds limit %u%%\n",
                total_cpu, archangel_ai->limits.max_cpu_percent);
        atomic64_inc(&archangel_ai->resource_monitor.overload_count);
        within_limits = false;
    }
    
    return within_limits;
}

/**
 * archangel_core_init - Module initialization function
 */
static int __init archangel_core_init(void)
{
    int ret;
    
    pr_info("archangel_core: Initializing Archangel AI Security Operating System\n");
    pr_info("archangel_core: Version %d.%d.%d\n", 
            ARCHANGEL_VERSION_MAJOR, 
            ARCHANGEL_VERSION_MINOR, 
            ARCHANGEL_VERSION_PATCH);
    
    /* Initialize AI coordination structure */
    ret = archangel_ai_initialize();
    if (ret) {
        pr_err("archangel_core: Failed to initialize AI structure: %d\n", ret);
        return ret;
    }
    
    /* Initialize statistics and proc interface */
    ret = archangel_stats_init();
    if (ret) {
        pr_err("archangel_core: Failed to initialize statistics: %d\n", ret);
        archangel_ai_cleanup();
        return ret;
    }
    
    /* Initialize communication infrastructure */
    ret = archangel_comm_init();
    if (ret) {
        pr_err("archangel_core: Failed to initialize communication: %d\n", ret);
        archangel_stats_cleanup();
        archangel_ai_cleanup();
        return ret;
    }
    
    pr_info("archangel_core: Module loaded successfully\n");
    return 0;
}

/**
 * archangel_core_exit - Module cleanup function
 */
static void __exit archangel_core_exit(void)
{
    pr_info("archangel_core: Shutting down Archangel AI Security Operating System\n");
    
    /* Clean up in reverse order */
    archangel_comm_cleanup();
    archangel_stats_cleanup();
    archangel_ai_cleanup();
    
    pr_info("archangel_core: Module unloaded successfully\n");
}

module_init(archangel_core_init);
module_exit(archangel_core_exit);

/* Export symbols for other Archangel modules */
EXPORT_SYMBOL(archangel_ai);
EXPORT_SYMBOL(archangel_stats_update);
EXPORT_SYMBOL(archangel_is_initialized);
EXPORT_SYMBOL(archangel_engine_register);
EXPORT_SYMBOL(archangel_engine_unregister);
EXPORT_SYMBOL(archangel_engine_update_stats);
EXPORT_SYMBOL(archangel_engine_check_limits);
EXPORT_SYMBOL(archangel_resource_check_limits);