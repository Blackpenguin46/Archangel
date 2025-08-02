/*
 * Archangel Linux - AI Security Expert Kernel Module
 * Main module initialization and core functionality
 */

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/version.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/ktime.h>
#include <linux/spinlock.h>
#include <linux/atomic.h>
#include <linux/mm.h>
#include <linux/vmalloc.h>

#include "../include/archangel.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Archangel AI Team");
MODULE_DESCRIPTION("AI Security Expert Kernel Module");
MODULE_VERSION(ARCHANGEL_VERSION);

/* Global module state */
static struct {
    bool initialized;
    u32 mode_flags;
    atomic64_t boot_time;
    struct mutex lock;
    
    /* Statistics */
    struct archangel_stats stats;
    spinlock_t stats_lock;
    
    /* Rule management */
    struct archangel_rule *rules[ARCHANGEL_MAX_RULES];
    u32 rule_count;
    struct mutex rules_lock;
    
    /* Shared memory */
    struct archangel_shared_memory *shared_mem;
    unsigned long shared_mem_size;
    
    /* Proc filesystem */
    struct proc_dir_entry *proc_root;
    struct proc_dir_entry *proc_entries[8];
    
} archangel_state = {
    .initialized = false,
    .mode_flags = ARCHANGEL_MODE_ACTIVE,
    .stats_lock = __SPIN_LOCK_UNLOCKED(archangel_state.stats_lock),
    .rules_lock = __MUTEX_INITIALIZER(archangel_state.rules_lock),
    .lock = __MUTEX_INITIALIZER(archangel_state.lock)
};

/* Forward declarations */
static int archangel_proc_init(void);
static void archangel_proc_cleanup(void);
static enum archangel_decision archangel_evaluate_rules(struct archangel_security_context *ctx);
static void archangel_update_stats(enum archangel_decision decision, u64 decision_time);

/* External function declarations */
extern int archangel_init_communication(void);
extern void archangel_cleanup_communication(void);
extern int archangel_syscall_init(void);
extern void archangel_syscall_cleanup(void);
extern int archangel_memory_init(void);
extern void archangel_memory_cleanup(void);
extern int archangel_network_init(void);
extern void archangel_network_cleanup(void);

/* Enhanced statistics functions */
extern void archangel_memory_get_stats(u64 *total_allocs, u64 *suspicious_activities, 
                                       u64 *blocked_allocs, u32 *tracked_processes);
extern void archangel_network_get_stats(u64 *total_packets, u64 *blocked_packets, 
                                        u64 *suspicious_packets, u64 *ddos_packets,
                                        u64 *malicious_payloads, u32 *tracked_connections,
                                        u32 *tracked_hosts);
extern void archangel_get_enhanced_comm_stats(u64 *sent, u64 *received, u64 *overruns, 
                                              u64 *zero_copy_ops, u64 *fast_path_decisions,
                                              u64 *avg_response_ns, u64 *min_response_ns, 
                                              u64 *max_response_ns, u64 *cache_hits, u64 *cache_misses);

/*
 * Core decision making function
 * This is the heart of the kernel-space security logic
 */
enum archangel_decision archangel_make_decision(struct archangel_security_context *ctx)
{
    ktime_t start_time, end_time;
    enum archangel_decision decision;
    u64 decision_time_ns;
    
    if (!archangel_state.initialized || !ctx)
        return ARCHANGEL_UNKNOWN;
    
    start_time = ktime_get();
    
    /* Try rule-based decision first (fast path) */
    decision = archangel_evaluate_rules(ctx);
    
    /* If no rule matches, defer to userspace AI */
    if (decision == ARCHANGEL_UNKNOWN) {
        decision = ARCHANGEL_DEFER_TO_USERSPACE;
        
        /* Send analysis request to userspace AI */
        archangel_send_message(ARCHANGEL_MSG_ANALYSIS_REQUEST, ctx, 
                              sizeof(*ctx) + ctx->data_size);
    }
    
    end_time = ktime_get();
    decision_time_ns = ktime_to_ns(ktime_sub(end_time, start_time));
    
    /* Update statistics */
    archangel_update_stats(decision, decision_time_ns);
    
    archangel_debug("Decision for PID %u, syscall %u: %d (took %llu ns)",
                   ctx->pid, ctx->syscall_nr, decision, decision_time_ns);
    
    return decision;
}

/*
 * Evaluate security rules against context
 * Fast kernel-space rule matching
 */
static enum archangel_decision archangel_evaluate_rules(struct archangel_security_context *ctx)
{
    struct archangel_rule *rule;
    u32 i;
    enum archangel_decision decision = ARCHANGEL_UNKNOWN;
    
    if (!ctx)
        return ARCHANGEL_UNKNOWN;
    
    mutex_lock(&archangel_state.rules_lock);
    
    /* Evaluate rules in priority order */
    for (i = 0; i < archangel_state.rule_count; i++) {
        rule = archangel_state.rules[i];
        if (!rule)
            continue;
        
        /* Simple rule matching - can be extended */
        if ((ctx->syscall_nr & rule->condition_mask) == rule->condition_values) {
            decision = rule->action;
            rule->match_count++;
            rule->last_matched = ktime_get_ns();
            
            archangel_debug("Rule %u matched for syscall %u", 
                           rule->id, ctx->syscall_nr);
            break;
        }
    }
    
    mutex_unlock(&archangel_state.rules_lock);
    
    return decision;
}

/*
 * Update module statistics
 */
static void archangel_update_stats(enum archangel_decision decision, u64 decision_time)
{
    unsigned long flags;
    
    spin_lock_irqsave(&archangel_state.stats_lock, flags);
    
    archangel_state.stats.total_decisions++;
    
    switch (decision) {
    case ARCHANGEL_ALLOW:
        archangel_state.stats.allow_decisions++;
        break;
    case ARCHANGEL_DENY:
        archangel_state.stats.deny_decisions++;
        break;
    case ARCHANGEL_MONITOR:
        archangel_state.stats.monitor_decisions++;
        break;
    case ARCHANGEL_DEFER_TO_USERSPACE:
        archangel_state.stats.deferred_decisions++;
        break;
    default:
        break;
    }
    
    /* Update timing statistics */
    if (decision_time > archangel_state.stats.max_decision_time_ns)
        archangel_state.stats.max_decision_time_ns = decision_time;
    
    /* Simple moving average for decision time */
    archangel_state.stats.avg_decision_time_ns = 
        (archangel_state.stats.avg_decision_time_ns + decision_time) / 2;
    
    spin_unlock_irqrestore(&archangel_state.stats_lock, flags);
}

/*
 * Add a new security rule
 */
int archangel_add_rule(const struct archangel_rule *rule)
{
    struct archangel_rule *new_rule;
    u32 i;
    
    if (!rule || archangel_state.rule_count >= ARCHANGEL_MAX_RULES)
        return -EINVAL;
    
    new_rule = kmalloc(sizeof(*new_rule), GFP_KERNEL);
    if (!new_rule)
        return -ENOMEM;
    
    memcpy(new_rule, rule, sizeof(*new_rule));
    new_rule->created_time = ktime_get_ns();
    new_rule->last_matched = 0;
    new_rule->match_count = 0;
    
    mutex_lock(&archangel_state.rules_lock);
    
    /* Find insertion point based on priority */
    for (i = 0; i < archangel_state.rule_count; i++) {
        if (!archangel_state.rules[i] || 
            archangel_state.rules[i]->priority > new_rule->priority) {
            break;
        }
    }
    
    /* Shift rules if necessary */
    if (i < archangel_state.rule_count) {
        memmove(&archangel_state.rules[i + 1], &archangel_state.rules[i],
                (archangel_state.rule_count - i) * sizeof(struct archangel_rule *));
    }
    
    archangel_state.rules[i] = new_rule;
    archangel_state.rule_count++;
    
    mutex_unlock(&archangel_state.rules_lock);
    
    archangel_info("Added security rule %u with priority %u", 
                   new_rule->id, new_rule->priority);
    
    return 0;
}

/*
 * Remove a security rule
 */
int archangel_remove_rule(u32 rule_id)
{
    u32 i, j;
    struct archangel_rule *rule = NULL;
    
    mutex_lock(&archangel_state.rules_lock);
    
    /* Find rule by ID */
    for (i = 0; i < archangel_state.rule_count; i++) {
        if (archangel_state.rules[i] && archangel_state.rules[i]->id == rule_id) {
            rule = archangel_state.rules[i];
            
            /* Shift remaining rules */
            for (j = i; j < archangel_state.rule_count - 1; j++) {
                archangel_state.rules[j] = archangel_state.rules[j + 1];
            }
            
            archangel_state.rule_count--;
            archangel_state.rules[archangel_state.rule_count] = NULL;
            break;
        }
    }
    
    mutex_unlock(&archangel_state.rules_lock);
    
    if (rule) {
        kfree(rule);
        archangel_info("Removed security rule %u", rule_id);
        return 0;
    }
    
    return -ENOENT;
}

/*
 * Get current statistics
 */
void archangel_get_stats(struct archangel_stats *stats)
{
    unsigned long flags;
    u64 uptime_ns;
    
    if (!stats)
        return;
    
    spin_lock_irqsave(&archangel_state.stats_lock, flags);
    memcpy(stats, &archangel_state.stats, sizeof(*stats));
    spin_unlock_irqrestore(&archangel_state.stats_lock, flags);
    
    /* Calculate uptime */
    uptime_ns = ktime_get_ns() - atomic64_read(&archangel_state.boot_time);
    stats->uptime_seconds = uptime_ns / 1000000000ULL;
}

/* Message functions are implemented in archangel_communication.c */

/*
 * Module initialization
 */
static int __init archangel_init(void)
{
    int ret;
    
    archangel_info("Initializing Archangel AI Security Expert v%s", ARCHANGEL_VERSION);
    
    /* Initialize module state */
    mutex_lock(&archangel_state.lock);
    
    memset(&archangel_state.stats, 0, sizeof(archangel_state.stats));
    atomic64_set(&archangel_state.boot_time, ktime_get_ns());
    archangel_state.rule_count = 0;
    
    /* Initialize communication system */
    ret = archangel_init_communication();
    if (ret) {
        archangel_err("Failed to initialize communication: %d", ret);
        goto error;
    }
    
    /* Initialize proc filesystem */
    ret = archangel_proc_init();
    if (ret) {
        archangel_err("Failed to initialize proc filesystem: %d", ret);
        goto error_comm;
    }
    
    /* Initialize syscall interception */
    ret = archangel_syscall_init();
    if (ret) {
        archangel_warn("Failed to initialize syscall interception: %d", ret);
        /* Continue without syscall interception */
    }
    
    /* Initialize memory protection system */
    ret = archangel_memory_init();
    if (ret) {
        archangel_warn("Failed to initialize memory protection: %d", ret);
        /* Continue without memory protection */
    }
    
    /* Initialize network analysis system */
    ret = archangel_network_init();
    if (ret) {
        archangel_warn("Failed to initialize network analysis: %d", ret);
        /* Continue without network analysis */
    }
    
    /* Allocate shared memory */
    archangel_state.shared_mem_size = ARCHANGEL_SHARED_MEM_SIZE;
    archangel_state.shared_mem = vmalloc(archangel_state.shared_mem_size);
    if (!archangel_state.shared_mem) {
        archangel_err("Failed to allocate shared memory");
        ret = -ENOMEM;
        goto error_syscall;
    }
    
    memset(archangel_state.shared_mem, 0, archangel_state.shared_mem_size);
    
    archangel_state.initialized = true;
    mutex_unlock(&archangel_state.lock);
    
    archangel_info("Archangel module loaded successfully");
    archangel_info("Mode: %s%s%s%s", 
                   (archangel_state.mode_flags & ARCHANGEL_MODE_ACTIVE) ? "ACTIVE " : "",
                   (archangel_state.mode_flags & ARCHANGEL_MODE_LEARNING) ? "LEARNING " : "",
                   (archangel_state.mode_flags & ARCHANGEL_MODE_MONITORING) ? "MONITORING " : "",
                   (archangel_state.mode_flags & ARCHANGEL_MODE_DEBUG) ? "DEBUG" : "");
    
    return 0;

error_syscall:
    archangel_syscall_cleanup();
    archangel_proc_cleanup();
error_comm:
    archangel_cleanup_communication();
error:
    archangel_state.initialized = false;
    mutex_unlock(&archangel_state.lock);
    return ret;
}

/*
 * Module cleanup
 */
static void __exit archangel_exit(void)
{
    u32 i;
    
    archangel_info("Unloading Archangel AI Security Expert");
    
    mutex_lock(&archangel_state.lock);
    archangel_state.initialized = false;
    
    /* Cleanup all subsystems */
    archangel_network_cleanup();
    archangel_memory_cleanup();
    archangel_syscall_cleanup();
    
    /* Cleanup shared memory */
    if (archangel_state.shared_mem) {
        vfree(archangel_state.shared_mem);
        archangel_state.shared_mem = NULL;
    }
    
    /* Cleanup rules */
    mutex_lock(&archangel_state.rules_lock);
    for (i = 0; i < archangel_state.rule_count; i++) {
        if (archangel_state.rules[i]) {
            kfree(archangel_state.rules[i]);
            archangel_state.rules[i] = NULL;
        }
    }
    archangel_state.rule_count = 0;
    mutex_unlock(&archangel_state.rules_lock);
    
    /* Cleanup proc filesystem */
    archangel_proc_cleanup();
    
    /* Cleanup communication */
    archangel_cleanup_communication();
    
    mutex_unlock(&archangel_state.lock);
    
    archangel_info("Archangel module unloaded");
}

/* Proc filesystem implementation */
static int archangel_proc_status_show(struct seq_file *m, void *v)
{
    seq_printf(m, "Archangel AI Security Expert v%s\n", ARCHANGEL_VERSION);
    seq_printf(m, "Status: %s\n", archangel_state.initialized ? "Active" : "Inactive");
    seq_printf(m, "Mode: 0x%08x\n", archangel_state.mode_flags);
    seq_printf(m, "Rules: %u/%u\n", archangel_state.rule_count, ARCHANGEL_MAX_RULES);
    seq_printf(m, "Shared Memory: %lu bytes\n", archangel_state.shared_mem_size);
    
    return 0;
}

static int archangel_proc_stats_show(struct seq_file *m, void *v)
{
    struct archangel_stats stats;
    
    /* Core module statistics */
    archangel_get_stats(&stats);
    
    seq_printf(m, "=== Core Module Statistics ===\n");
    seq_printf(m, "Total Decisions: %llu\n", stats.total_decisions);
    seq_printf(m, "Allow: %llu\n", stats.allow_decisions);
    seq_printf(m, "Deny: %llu\n", stats.deny_decisions);
    seq_printf(m, "Monitor: %llu\n", stats.monitor_decisions);
    seq_printf(m, "Deferred: %llu\n", stats.deferred_decisions);
    seq_printf(m, "Rule Matches: %llu\n", stats.rule_matches);
    seq_printf(m, "Cache Hits: %llu\n", stats.cache_hits);
    seq_printf(m, "Cache Misses: %llu\n", stats.cache_misses);
    seq_printf(m, "Userspace Requests: %llu\n", stats.userspace_requests);
    seq_printf(m, "Userspace Responses: %llu\n", stats.userspace_responses);
    seq_printf(m, "Avg Decision Time: %llu ns\n", stats.avg_decision_time_ns);
    seq_printf(m, "Max Decision Time: %llu ns\n", stats.max_decision_time_ns);
    seq_printf(m, "Uptime: %llu seconds\n", stats.uptime_seconds);
    
    /* Enhanced communication statistics */
    u64 sent, received, overruns, zero_copy_ops, fast_path_decisions;
    u64 avg_response_ns, min_response_ns, max_response_ns, cache_hits, cache_misses;
    
    archangel_get_enhanced_comm_stats(&sent, &received, &overruns, &zero_copy_ops,
                                     &fast_path_decisions, &avg_response_ns, 
                                     &min_response_ns, &max_response_ns,
                                     &cache_hits, &cache_misses);
    
    seq_printf(m, "\n=== Enhanced Communication Statistics ===\n");
    seq_printf(m, "Messages Sent: %llu\n", sent);
    seq_printf(m, "Messages Received: %llu\n", received);
    seq_printf(m, "Ring Buffer Overruns: %llu\n", overruns);
    seq_printf(m, "Zero-Copy Operations: %llu\n", zero_copy_ops);
    seq_printf(m, "Fast Path Decisions: %llu\n", fast_path_decisions);
    seq_printf(m, "Avg Response Time: %llu ns\n", avg_response_ns);
    seq_printf(m, "Min Response Time: %llu ns\n", min_response_ns);
    seq_printf(m, "Max Response Time: %llu ns\n", max_response_ns);
    seq_printf(m, "Decision Cache Hits: %llu\n", cache_hits);
    seq_printf(m, "Decision Cache Misses: %llu\n", cache_misses);
    
    /* Memory protection statistics */
    u64 total_allocs, suspicious_activities, blocked_allocs;
    u32 tracked_processes;
    
    archangel_memory_get_stats(&total_allocs, &suspicious_activities, 
                              &blocked_allocs, &tracked_processes);
    
    seq_printf(m, "\n=== Memory Protection Statistics ===\n");
    seq_printf(m, "Total Allocations: %llu\n", total_allocs);
    seq_printf(m, "Suspicious Activities: %llu\n", suspicious_activities);
    seq_printf(m, "Blocked Allocations: %llu\n", blocked_allocs);
    seq_printf(m, "Tracked Processes: %u\n", tracked_processes);
    
    /* Network analysis statistics */
    u64 total_packets, blocked_packets, suspicious_packets, ddos_packets, malicious_payloads;
    u32 tracked_connections, tracked_hosts;
    
    archangel_network_get_stats(&total_packets, &blocked_packets, &suspicious_packets,
                               &ddos_packets, &malicious_payloads, &tracked_connections,
                               &tracked_hosts);
    
    seq_printf(m, "\n=== Network Analysis Statistics ===\n");
    seq_printf(m, "Total Packets: %llu\n", total_packets);
    seq_printf(m, "Blocked Packets: %llu\n", blocked_packets);
    seq_printf(m, "Suspicious Packets: %llu\n", suspicious_packets);
    seq_printf(m, "DDoS Packets: %llu\n", ddos_packets);
    seq_printf(m, "Malicious Payloads: %llu\n", malicious_payloads);
    seq_printf(m, "Tracked Connections: %u\n", tracked_connections);
    seq_printf(m, "Tracked Hosts: %u\n", tracked_hosts);
    
    return 0;
}

static int archangel_proc_rules_show(struct seq_file *m, void *v)
{
    u32 i;
    struct archangel_rule *rule;
    
    seq_printf(m, "ID\tPriority\tAction\tConfidence\tMatches\tDescription\n");
    
    mutex_lock(&archangel_state.rules_lock);
    for (i = 0; i < archangel_state.rule_count; i++) {
        rule = archangel_state.rules[i];
        if (!rule)
            continue;
        
        seq_printf(m, "%u\t%u\t%d\t%d\t%u\t%s\n",
                   rule->id, rule->priority, rule->action, 
                   rule->confidence, rule->match_count, rule->description);
    }
    mutex_unlock(&archangel_state.rules_lock);
    
    return 0;
}

/* Proc file open functions */
static int archangel_proc_status_open(struct inode *inode, struct file *file)
{
    return single_open(file, archangel_proc_status_show, NULL);
}

static int archangel_proc_stats_open(struct inode *inode, struct file *file)
{
    return single_open(file, archangel_proc_stats_show, NULL);
}

static int archangel_proc_rules_open(struct inode *inode, struct file *file)
{
    return single_open(file, archangel_proc_rules_show, NULL);
}

/* Proc control write function */
static ssize_t archangel_proc_control_write(struct file *file, const char __user *buffer,
                                           size_t count, loff_t *pos)
{
    char cmd[256];
    struct archangel_rule rule;
    u32 rule_id;
    
    if (count > sizeof(cmd) - 1)
        return -EINVAL;
    
    if (copy_from_user(cmd, buffer, count))
        return -EFAULT;
    
    cmd[count] = '\0';
    
    /* Parse commands */
    if (strncmp(cmd, "add_rule", 8) == 0) {
        /* Simple rule format: add_rule <id> <priority> <action> <description> */
        if (sscanf(cmd, "add_rule %u %u %d %63s", 
                   &rule.id, &rule.priority, (int*)&rule.action, rule.description) == 4) {
            rule.condition_mask = 0xFFFFFFFF;
            rule.condition_values = 0;
            rule.confidence = ARCHANGEL_CONFIDENCE_HIGH;
            
            if (archangel_add_rule(&rule) == 0) {
                archangel_info("Added rule %u via proc interface", rule.id);
            }
        }
    } else if (strncmp(cmd, "del_rule", 8) == 0) {
        if (sscanf(cmd, "del_rule %u", &rule_id) == 1) {
            if (archangel_remove_rule(rule_id) == 0) {
                archangel_info("Removed rule %u via proc interface", rule_id);
            }
        }
    } else if (strncmp(cmd, "clear_rules", 11) == 0) {
        mutex_lock(&archangel_state.rules_lock);
        for (rule_id = 0; rule_id < archangel_state.rule_count; rule_id++) {
            if (archangel_state.rules[rule_id]) {
                kfree(archangel_state.rules[rule_id]);
                archangel_state.rules[rule_id] = NULL;
            }
        }
        archangel_state.rule_count = 0;
        mutex_unlock(&archangel_state.rules_lock);
        archangel_info("Cleared all rules via proc interface");
    }
    
    return count;
}

/* Proc file operations */
static const struct proc_ops archangel_proc_status_ops = {
    .proc_open = archangel_proc_status_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

static const struct proc_ops archangel_proc_stats_ops = {
    .proc_open = archangel_proc_stats_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

static const struct proc_ops archangel_proc_rules_ops = {
    .proc_open = archangel_proc_rules_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

static const struct proc_ops archangel_proc_control_ops = {
    .proc_write = archangel_proc_control_write,
};

/*
 * Initialize proc filesystem entries
 */
static int archangel_proc_init(void)
{
    /* Create root directory */
    archangel_state.proc_root = proc_mkdir(ARCHANGEL_PROC_ROOT, NULL);
    if (!archangel_state.proc_root) {
        archangel_err("Failed to create proc root directory");
        return -ENOMEM;
    }
    
    /* Create proc entries */
    archangel_state.proc_entries[0] = proc_create(ARCHANGEL_PROC_STATUS, 0444,
                                                 archangel_state.proc_root,
                                                 &archangel_proc_status_ops);
    
    archangel_state.proc_entries[1] = proc_create(ARCHANGEL_PROC_STATS, 0444,
                                                 archangel_state.proc_root,
                                                 &archangel_proc_stats_ops);
    
    archangel_state.proc_entries[2] = proc_create(ARCHANGEL_PROC_RULES, 0444,
                                                 archangel_state.proc_root,
                                                 &archangel_proc_rules_ops);
    
    archangel_state.proc_entries[3] = proc_create(ARCHANGEL_PROC_CONTROL, 0200,
                                                 archangel_state.proc_root,
                                                 &archangel_proc_control_ops);
    
    archangel_info("Proc filesystem initialized at /proc/%s", ARCHANGEL_PROC_ROOT);
    return 0;
}

/*
 * Cleanup proc filesystem entries
 */
static void archangel_proc_cleanup(void)
{
    if (archangel_state.proc_root) {
        proc_remove(archangel_state.proc_root);
        archangel_state.proc_root = NULL;
    }
    
    archangel_info("Proc filesystem cleaned up");
}

module_init(archangel_init);
module_exit(archangel_exit);