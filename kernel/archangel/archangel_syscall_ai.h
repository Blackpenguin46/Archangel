#ifndef _ARCHANGEL_SYSCALL_AI_H
#define _ARCHANGEL_SYSCALL_AI_H

#include "archangel_core.h"
#include "archangel_comm.h"
#include <linux/syscalls.h>
#include <linux/ptrace.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/radix-tree.h>
#include <linux/hash.h>
#include <linux/jhash.h>
#include <linux/rbtree.h>
#include <linux/ktime.h>

/* Syscall AI constants */
#define ARCHANGEL_SYSCALL_MAX_PATTERNS 1024
#define ARCHANGEL_SYSCALL_CACHE_SIZE 4096
#define ARCHANGEL_SYSCALL_PROFILE_ENTRIES 65536
#define ARCHANGEL_SYSCALL_DECISION_TIMEOUT_NS 100000  /* 100μs */
#define ARCHANGEL_SYSCALL_RISK_THRESHOLD 75
#define ARCHANGEL_SYSCALL_PATTERN_MAX_LEN 256

/* Syscall decision types */
enum archangel_syscall_decision {
    ARCHANGEL_SYSCALL_ALLOW = 0,
    ARCHANGEL_SYSCALL_BLOCK,
    ARCHANGEL_SYSCALL_DEFER,
    ARCHANGEL_SYSCALL_MONITOR,
    ARCHANGEL_SYSCALL_UNKNOWN
};

/* Risk levels */
enum archangel_risk_level {
    ARCHANGEL_RISK_LOW = 0,
    ARCHANGEL_RISK_MEDIUM = 25,
    ARCHANGEL_RISK_HIGH = 50,
    ARCHANGEL_RISK_CRITICAL = 75
};

/* Pattern types for malicious behavior detection */
enum archangel_pattern_type {
    ARCHANGEL_PATTERN_SEQUENCE = 0,  /* Syscall sequence patterns */
    ARCHANGEL_PATTERN_FREQUENCY,     /* Frequency-based patterns */
    ARCHANGEL_PATTERN_ARGUMENT,      /* Argument-based patterns */
    ARCHANGEL_PATTERN_TIMING,        /* Timing-based patterns */
    ARCHANGEL_PATTERN_MAX
};

/**
 * struct archangel_syscall_pattern - Malicious pattern definition
 * @type: Pattern type
 * @syscalls: Array of syscall numbers in pattern
 * @count: Number of syscalls in pattern
 * @risk_score: Risk score for this pattern (0-100)
 * @description: Human-readable description
 * @enabled: Whether pattern is active
 */
struct archangel_syscall_pattern {
    enum archangel_pattern_type type;
    u16 syscalls[16];
    u8 count;
    u8 risk_score;
    char description[64];
    bool enabled;
};

/**
 * struct archangel_decision_node - Decision tree node
 * @syscall_nr: System call number
 * @condition: Condition to check (bitmask)
 * @threshold: Threshold value for condition
 * @decision: Decision if condition matches
 * @left: Left child (condition false)
 * @right: Right child (condition true)
 */
struct archangel_decision_node {
    u16 syscall_nr;
    u32 condition;
    u32 threshold;
    enum archangel_syscall_decision decision;
    struct archangel_decision_node *left;
    struct archangel_decision_node *right;
};

/**
 * struct archangel_decision_tree - Pre-compiled decision tree
 * @root: Root node of the tree
 * @node_count: Number of nodes in tree
 * @max_depth: Maximum depth of tree
 * @last_update: Last update timestamp
 */
struct archangel_decision_tree {
    struct archangel_decision_node *root;
    u32 node_count;
    u8 max_depth;
    u64 last_update;
};

/**
 * struct archangel_process_profile - Per-process behavioral profile
 * @pid: Process ID
 * @comm: Process command name
 * @syscall_counts: Count of each syscall type
 * @recent_syscalls: Ring buffer of recent syscalls
 * @recent_head: Head of recent syscalls ring buffer
 * @risk_score: Current risk score (0-100)
 * @creation_time: Process creation timestamp
 * @last_activity: Last syscall timestamp
 * @pattern_matches: Number of pattern matches
 * @suspicious_flags: Bitmask of suspicious behaviors
 * @rb_node: Red-black tree node for efficient lookup
 */
struct archangel_process_profile {
    pid_t pid;
    char comm[TASK_COMM_LEN];
    atomic_t syscall_counts[__NR_syscalls];
    u16 recent_syscalls[32];
    u8 recent_head;
    atomic_t risk_score;
    u64 creation_time;
    u64 last_activity;
    atomic_t pattern_matches;
    u32 suspicious_flags;
    struct rb_node rb_node;
};

/**
 * struct archangel_decision_cache_entry - Cached decision entry
 * @key: Cache key (hash of syscall + context)
 * @decision: Cached decision
 * @timestamp: When decision was cached
 * @hit_count: Number of cache hits
 * @next: Next entry in hash chain
 */
struct archangel_decision_cache_entry {
    u64 key;
    enum archangel_syscall_decision decision;
    u64 timestamp;
    atomic_t hit_count;
    struct archangel_decision_cache_entry *next;
};

/**
 * struct archangel_decision_cache - Decision cache for fast lookups
 * @entries: Hash table of cache entries
 * @size: Size of hash table
 * @mask: Hash mask (size - 1)
 * @hit_count: Total cache hits
 * @miss_count: Total cache misses
 * @lock: Cache lock
 */
struct archangel_decision_cache {
    struct archangel_decision_cache_entry **entries;
    u32 size;
    u32 mask;
    atomic64_t hit_count;
    atomic64_t miss_count;
    spinlock_t lock;
};

/**
 * struct archangel_syscall_ai_engine - Main syscall AI engine
 * @base: Base AI engine structure
 * @decision_trees: Pre-compiled decision trees for each syscall
 * @patterns: Malicious pattern database
 * @pattern_count: Number of active patterns
 * @process_profiles: Red-black tree of process profiles
 * @profile_lock: Lock for process profiles tree
 * @decision_cache: Fast decision cache
 * @userspace_defer_queue: Queue for userspace deferral
 * @stats: Engine-specific statistics
 * @enabled: Whether engine is enabled
 */
struct archangel_syscall_ai_engine {
    struct archangel_ai_engine base;
    
    /* Decision trees for fast inference */
    struct archangel_decision_tree decision_trees[__NR_syscalls];
    
    /* Pattern matching */
    struct archangel_syscall_pattern patterns[ARCHANGEL_SYSCALL_MAX_PATTERNS];
    atomic_t pattern_count;
    
    /* Per-process behavioral profiling */
    struct rb_root process_profiles;
    spinlock_t profile_lock;
    
    /* Decision caching */
    struct archangel_decision_cache decision_cache;
    
    /* Userspace deferral */
    struct archangel_spsc_queue userspace_defer_queue;
    
    /* Statistics */
    struct {
        atomic64_t syscalls_analyzed;
        atomic64_t decisions_cached;
        atomic64_t patterns_matched;
        atomic64_t processes_profiled;
        atomic64_t blocks_issued;
        atomic64_t deferrals_issued;
        atomic64_t userspace_requests;
    } stats;
    
    bool enabled;
};

/**
 * struct archangel_syscall_context - Context for syscall analysis
 * @regs: CPU registers at syscall entry
 * @syscall_nr: System call number
 * @args: System call arguments
 * @task: Current task
 * @pid: Process ID
 * @timestamp: Syscall timestamp
 */
struct archangel_syscall_context {
    struct pt_regs *regs;
    long syscall_nr;
    unsigned long args[6];
    struct task_struct *task;
    pid_t pid;
    u64 timestamp;
};

/**
 * struct archangel_userspace_defer_request - Userspace deferral request
 * @context: Syscall context
 * @profile: Process profile
 * @risk_score: Calculated risk score
 * @pattern_matches: Matched patterns
 * @timeout: Request timeout
 */
struct archangel_userspace_defer_request {
    struct archangel_syscall_context context;
    struct archangel_process_profile *profile;
    u8 risk_score;
    u32 pattern_matches;
    u64 timeout;
};

/* Global syscall AI engine */
extern struct archangel_syscall_ai_engine *archangel_syscall_ai;

/* Core functions */
int archangel_syscall_ai_init(void);
void archangel_syscall_ai_cleanup(void);
int archangel_syscall_ai_enable(void);
void archangel_syscall_ai_disable(void);

/* Main syscall interception */
enum archangel_syscall_decision ai_syscall_intercept(struct pt_regs *regs);

/* Decision tree operations */
int archangel_decision_tree_init(struct archangel_decision_tree *tree, u16 syscall_nr);
void archangel_decision_tree_cleanup(struct archangel_decision_tree *tree);
enum archangel_syscall_decision archangel_decision_tree_evaluate(
    struct archangel_decision_tree *tree, 
    struct archangel_syscall_context *context);

/* Pattern matching */
int archangel_pattern_init(void);
void archangel_pattern_cleanup(void);
int archangel_pattern_add(struct archangel_syscall_pattern *pattern);
void archangel_pattern_remove(u32 pattern_id);
u32 archangel_pattern_match(struct archangel_process_profile *profile, 
                           struct archangel_syscall_context *context);

/* Process profiling */
struct archangel_process_profile *archangel_profile_get_or_create(pid_t pid);
void archangel_profile_update(struct archangel_process_profile *profile, 
                             struct archangel_syscall_context *context);
void archangel_profile_cleanup_expired(void);
u8 archangel_profile_calculate_risk(struct archangel_process_profile *profile);

/* Decision caching */
int archangel_decision_cache_init(struct archangel_decision_cache *cache, u32 size);
void archangel_decision_cache_cleanup(struct archangel_decision_cache *cache);
enum archangel_syscall_decision archangel_decision_cache_lookup(
    struct archangel_decision_cache *cache, u64 key);
void archangel_decision_cache_store(struct archangel_decision_cache *cache, 
                                   u64 key, enum archangel_syscall_decision decision);

/* Userspace deferral */
int archangel_userspace_defer_init(void);
void archangel_userspace_defer_cleanup(void);
int archangel_userspace_defer_request(struct archangel_syscall_context *context,
                                     struct archangel_process_profile *profile);
enum archangel_syscall_decision archangel_userspace_defer_wait_response(u64 request_id);

/* Utility functions */
static inline u64 archangel_syscall_hash_context(struct archangel_syscall_context *context)
{
    return jhash_3words(context->syscall_nr, context->pid, 
                       (u32)(context->timestamp >> 32), 0x12345678);
}

static inline bool archangel_syscall_is_high_risk(long syscall_nr)
{
    /* High-risk syscalls that require careful analysis */
    switch (syscall_nr) {
    case __NR_execve:
    case __NR_execveat:
    case __NR_ptrace:
    case __NR_mount:
    case __NR_umount2:
    case __NR_init_module:
    case __NR_delete_module:
    case __NR_kexec_load:
    case __NR_reboot:
        return true;
    default:
        return false;
    }
}

static inline bool archangel_syscall_ai_is_enabled(void)
{
    return archangel_syscall_ai && archangel_syscall_ai->enabled;
}

#endif /* _ARCHANGEL_SYSCALL_AI_H */