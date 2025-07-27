#include "archangel_syscall_ai.h"

/* Module information */
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Archangel Linux Development Team");
MODULE_DESCRIPTION("Archangel AI Security Operating System - Syscall AI Filter");
MODULE_VERSION("1.0.0");

/* Global syscall AI engine */
struct archangel_syscall_ai_engine *archangel_syscall_ai = NULL;

/* Original syscall table pointer */
static void **original_sys_call_table = NULL;
static void *original_syscalls[__NR_syscalls];

/* Default malicious patterns */
static struct archangel_syscall_pattern default_patterns[] = {
    {
        .type = ARCHANGEL_PATTERN_SEQUENCE,
        .syscalls = {__NR_socket, __NR_connect, __NR_write},
        .count = 3,
        .risk_score = 60,
        .description = "Network connection pattern",
        .enabled = true
    },
    {
        .type = ARCHANGEL_PATTERN_SEQUENCE,
        .syscalls = {__NR_openat, __NR_read, __NR_write, __NR_close},
        .count = 4,
        .risk_score = 40,
        .description = "File access pattern",
        .enabled = true
    },
    {
        .type = ARCHANGEL_PATTERN_SEQUENCE,
        .syscalls = {__NR_ptrace, __NR_wait4},
        .count = 2,
        .risk_score = 85,
        .description = "Process debugging pattern",
        .enabled = true
    },
    {
        .type = ARCHANGEL_PATTERN_SEQUENCE,
        .syscalls = {__NR_execve, __NR_setuid, __NR_setgid},
        .count = 3,
        .risk_score = 90,
        .description = "Privilege escalation pattern",
        .enabled = true
    }
};

/**
 * archangel_syscall_ai_init - Initialize the syscall AI engine
 * 
 * Returns: 0 on success, negative error code on failure
 */
int archangel_syscall_ai_init(void)
{
    int ret, i;
    
    if (archangel_syscall_ai) {
        pr_warn("archangel_syscall_ai: Engine already initialized\n");
        return -EEXIST;
    }
    
    if (!archangel_is_initialized()) {
        pr_err("archangel_syscall_ai: Core AI not initialized\n");
        return -EINVAL;
    }
    
    /* Allocate engine structure */
    archangel_syscall_ai = kzalloc(sizeof(*archangel_syscall_ai), GFP_KERNEL);
    if (!archangel_syscall_ai) {
        pr_err("archangel_syscall_ai: Failed to allocate engine structure\n");
        return -ENOMEM;
    }
    
    /* Initialize base engine */
    archangel_syscall_ai->base.type = ARCHANGEL_ENGINE_SYSCALL;
    archangel_syscall_ai->base.status = ARCHANGEL_ENGINE_INACTIVE;
    atomic64_set(&archangel_syscall_ai->base.inference_count, 0);
    archangel_syscall_ai->base.avg_inference_time_ns = 0;
    archangel_syscall_ai->base.memory_usage_kb = sizeof(*archangel_syscall_ai) / 1024;
    archangel_syscall_ai->base.cpu_usage_percent = 0;
    archangel_syscall_ai->base.last_inference_time = 0;
    spin_lock_init(&archangel_syscall_ai->base.lock);
    
    /* Initialize decision trees */
    for (i = 0; i < __NR_syscalls; i++) {
        ret = archangel_decision_tree_init(&archangel_syscall_ai->decision_trees[i], i);
        if (ret) {
            pr_err("archangel_syscall_ai: Failed to initialize decision tree for syscall %d: %d\n", i, ret);
            goto cleanup_trees;
        }
    }
    
    /* Initialize pattern matching */
    ret = archangel_pattern_init();
    if (ret) {
        pr_err("archangel_syscall_ai: Failed to initialize pattern matching: %d\n", ret);
        goto cleanup_trees;
    }
    
    /* Initialize process profiling */
    archangel_syscall_ai->process_profiles = RB_ROOT;
    spin_lock_init(&archangel_syscall_ai->profile_lock);
    
    /* Initialize decision cache */
    ret = archangel_decision_cache_init(&archangel_syscall_ai->decision_cache, 
                                       ARCHANGEL_SYSCALL_CACHE_SIZE);
    if (ret) {
        pr_err("archangel_syscall_ai: Failed to initialize decision cache: %d\n", ret);
        goto cleanup_patterns;
    }
    
    /* Initialize userspace deferral queue */
    ret = archangel_spsc_queue_init(&archangel_syscall_ai->userspace_defer_queue, 
                                   ARCHANGEL_COMM_RING_SIZE);
    if (ret) {
        pr_err("archangel_syscall_ai: Failed to initialize deferral queue: %d\n", ret);
        goto cleanup_cache;
    }
    
    /* Initialize statistics */
    atomic64_set(&archangel_syscall_ai->stats.syscalls_analyzed, 0);
    atomic64_set(&archangel_syscall_ai->stats.decisions_cached, 0);
    atomic64_set(&archangel_syscall_ai->stats.patterns_matched, 0);
    atomic64_set(&archangel_syscall_ai->stats.processes_profiled, 0);
    atomic64_set(&archangel_syscall_ai->stats.blocks_issued, 0);
    atomic64_set(&archangel_syscall_ai->stats.deferrals_issued, 0);
    atomic64_set(&archangel_syscall_ai->stats.userspace_requests, 0);
    
    archangel_syscall_ai->enabled = false;
    
    /* Register with core AI */
    ret = archangel_engine_register(&archangel_syscall_ai->base);
    if (ret) {
        pr_err("archangel_syscall_ai: Failed to register with core AI: %d\n", ret);
        goto cleanup_queue;
    }
    
    pr_info("archangel_syscall_ai: Syscall AI engine initialized with %d decision trees\n", 
            __NR_syscalls);
    return 0;

cleanup_queue:
    archangel_spsc_queue_cleanup(&archangel_syscall_ai->userspace_defer_queue);
cleanup_cache:
    archangel_decision_cache_cleanup(&archangel_syscall_ai->decision_cache);
cleanup_patterns:
    archangel_pattern_cleanup();
cleanup_trees:
    for (i = i - 1; i >= 0; i--) {
        archangel_decision_tree_cleanup(&archangel_syscall_ai->decision_trees[i]);
    }
    kfree(archangel_syscall_ai);
    archangel_syscall_ai = NULL;
    return ret;
}

/**
 * archangel_syscall_ai_cleanup - Clean up the syscall AI engine
 */
void archangel_syscall_ai_cleanup(void)
{
    int i;
    
    if (!archangel_syscall_ai)
        return;
    
    /* Disable engine first */
    archangel_syscall_ai_disable();
    
    /* Unregister from core AI */
    archangel_engine_unregister(&archangel_syscall_ai->base);
    
    /* Clean up userspace deferral queue */
    archangel_spsc_queue_cleanup(&archangel_syscall_ai->userspace_defer_queue);
    
    /* Clean up decision cache */
    archangel_decision_cache_cleanup(&archangel_syscall_ai->decision_cache);
    
    /* Clean up pattern matching */
    archangel_pattern_cleanup();
    
    /* Clean up decision trees */
    for (i = 0; i < __NR_syscalls; i++) {
        archangel_decision_tree_cleanup(&archangel_syscall_ai->decision_trees[i]);
    }
    
    /* Clean up process profiles */
    archangel_profile_cleanup_expired();
    
    kfree(archangel_syscall_ai);
    archangel_syscall_ai = NULL;
    
    pr_info("archangel_syscall_ai: Syscall AI engine cleaned up\n");
}

/**
 * archangel_syscall_ai_enable - Enable syscall interception
 * 
 * Returns: 0 on success, negative error code on failure
 */
int archangel_syscall_ai_enable(void)
{
    if (!archangel_syscall_ai) {
        pr_err("archangel_syscall_ai: Engine not initialized\n");
        return -EINVAL;
    }
    
    if (archangel_syscall_ai->enabled) {
        pr_warn("archangel_syscall_ai: Engine already enabled\n");
        return 0;
    }
    
    /* Find syscall table */
    original_sys_call_table = (void **)kallsyms_lookup_name("sys_call_table");
    if (!original_sys_call_table) {
        pr_err("archangel_syscall_ai: Failed to find sys_call_table\n");
        return -ENOENT;
    }
    
    /* Note: In a real implementation, we would hook specific syscalls */
    /* For now, we'll use a different approach via kprobes or similar */
    /* This is a placeholder for syscall table modification */
    
    archangel_syscall_ai->enabled = true;
    archangel_syscall_ai->base.status = ARCHANGEL_ENGINE_ACTIVE;
    
    pr_info("archangel_syscall_ai: Syscall interception enabled\n");
    return 0;
}

/**
 * archangel_syscall_ai_disable - Disable syscall interception
 */
void archangel_syscall_ai_disable(void)
{
    if (!archangel_syscall_ai || !archangel_syscall_ai->enabled)
        return;
    
    /* Restore original syscalls */
    if (original_sys_call_table) {
        /* Note: In a real implementation, we would restore syscall table */
        /* This is a placeholder for syscall table restoration */
    }
    
    archangel_syscall_ai->enabled = false;
    archangel_syscall_ai->base.status = ARCHANGEL_ENGINE_INACTIVE;
    
    pr_info("archangel_syscall_ai: Syscall interception disabled\n");
}

/**
 * ai_syscall_intercept - Main syscall interception function
 * @regs: CPU registers at syscall entry
 * 
 * Returns: Decision for the syscall
 */
enum archangel_syscall_decision ai_syscall_intercept(struct pt_regs *regs)
{
    struct archangel_syscall_context context;
    struct archangel_process_profile *profile;
    enum archangel_syscall_decision decision;
    u64 start_time, inference_time;
    u64 cache_key;
    u32 pattern_matches;
    u8 risk_score;
    
    if (!archangel_syscall_ai_is_enabled()) {
        return ARCHANGEL_SYSCALL_ALLOW;
    }
    
    start_time = ktime_get_ns();
    
    /* Prepare syscall context */
    context.regs = regs;
    /* Get syscall number - architecture specific */
#ifdef CONFIG_X86_64
    context.syscall_nr = regs->orig_ax;
#elif defined(CONFIG_ARM64)
    context.syscall_nr = regs->syscallno;
#else
    context.syscall_nr = -1; /* Unknown architecture */
#endif
    context.task = current;
    context.pid = current->pid;
    context.timestamp = start_time;
    
    /* Get syscall arguments - architecture specific */
#ifdef CONFIG_X86_64
    context.args[0] = regs->di;
    context.args[1] = regs->si;
    context.args[2] = regs->dx;
    context.args[3] = regs->r10;
    context.args[4] = regs->r8;
    context.args[5] = regs->r9;
#elif defined(CONFIG_ARM64)
    context.args[0] = regs->regs[0];
    context.args[1] = regs->regs[1];
    context.args[2] = regs->regs[2];
    context.args[3] = regs->regs[3];
    context.args[4] = regs->regs[4];
    context.args[5] = regs->regs[5];
#else
    /* Generic fallback */
    memset(context.args, 0, sizeof(context.args));
#endif
    
    /* Check decision cache first */
    cache_key = archangel_syscall_hash_context(&context);
    decision = archangel_decision_cache_lookup(&archangel_syscall_ai->decision_cache, cache_key);
    if (decision != ARCHANGEL_SYSCALL_UNKNOWN) {
        atomic64_inc(&archangel_syscall_ai->stats.decisions_cached);
        goto update_stats;
    }
    
    /* Get or create process profile */
    profile = archangel_profile_get_or_create(context.pid);
    if (!profile) {
        pr_warn("archangel_syscall_ai: Failed to get process profile for PID %d\n", context.pid);
        decision = ARCHANGEL_SYSCALL_ALLOW;
        goto update_stats;
    }
    
    /* Update process profile */
    archangel_profile_update(profile, &context);
    
    /* Check for pattern matches */
    pattern_matches = archangel_pattern_match(profile, &context);
    if (pattern_matches > 0) {
        atomic64_inc(&archangel_syscall_ai->stats.patterns_matched);
    }
    
    /* Calculate risk score */
    risk_score = archangel_profile_calculate_risk(profile);
    
    /* Use decision tree for fast inference */
    if (context.syscall_nr < __NR_syscalls) {
        decision = archangel_decision_tree_evaluate(
            &archangel_syscall_ai->decision_trees[context.syscall_nr], &context);
    } else {
        decision = ARCHANGEL_SYSCALL_ALLOW;
    }
    
    /* Override decision based on risk score and patterns */
    if (risk_score >= ARCHANGEL_SYSCALL_RISK_THRESHOLD || pattern_matches > 2) {
        if (archangel_syscall_is_high_risk(context.syscall_nr)) {
            decision = ARCHANGEL_SYSCALL_BLOCK;
            atomic64_inc(&archangel_syscall_ai->stats.blocks_issued);
        } else {
            decision = ARCHANGEL_SYSCALL_DEFER;
            atomic64_inc(&archangel_syscall_ai->stats.deferrals_issued);
            
            /* Send to userspace for complex analysis */
            archangel_userspace_defer_request(&context, profile);
        }
    }
    
    /* Cache the decision */
    archangel_decision_cache_store(&archangel_syscall_ai->decision_cache, cache_key, decision);
    
update_stats:
    /* Update statistics */
    inference_time = ktime_get_ns() - start_time;
    archangel_engine_update_stats(&archangel_syscall_ai->base, inference_time);
    atomic64_inc(&archangel_syscall_ai->stats.syscalls_analyzed);
    
    /* Check resource limits */
    if (!archangel_engine_check_limits(&archangel_syscall_ai->base)) {
        pr_warn("archangel_syscall_ai: Engine exceeding resource limits\n");
    }
    
    return decision;
}

/**
 * archangel_decision_tree_init - Initialize a decision tree for a syscall
 * @tree: Decision tree to initialize
 * @syscall_nr: System call number
 * 
 * Returns: 0 on success, negative error code on failure
 */
int archangel_decision_tree_init(struct archangel_decision_tree *tree, u16 syscall_nr)
{
    if (!tree) {
        return -EINVAL;
    }
    
    /* Initialize tree structure */
    tree->root = NULL;
    tree->node_count = 0;
    tree->max_depth = 0;
    tree->last_update = ktime_get_ns();
    
    /* Create simple decision tree based on syscall type */
    tree->root = kzalloc(sizeof(struct archangel_decision_node), GFP_KERNEL);
    if (!tree->root) {
        return -ENOMEM;
    }
    
    tree->root->syscall_nr = syscall_nr;
    tree->root->condition = 0; /* No condition for root */
    tree->root->threshold = 0;
    tree->root->left = NULL;
    tree->root->right = NULL;
    
    /* Set default decision based on syscall risk level */
    if (archangel_syscall_is_high_risk(syscall_nr)) {
        tree->root->decision = ARCHANGEL_SYSCALL_MONITOR;
    } else {
        tree->root->decision = ARCHANGEL_SYSCALL_ALLOW;
    }
    
    tree->node_count = 1;
    tree->max_depth = 1;
    
    return 0;
}

/**
 * archangel_decision_tree_cleanup - Clean up a decision tree
 * @tree: Decision tree to clean up
 */
void archangel_decision_tree_cleanup(struct archangel_decision_tree *tree)
{
    if (!tree || !tree->root)
        return;
    
    /* Simple cleanup for single-node tree */
    kfree(tree->root);
    tree->root = NULL;
    tree->node_count = 0;
    tree->max_depth = 0;
}

/**
 * archangel_decision_tree_evaluate - Evaluate decision tree for syscall
 * @tree: Decision tree to evaluate
 * @context: Syscall context
 * 
 * Returns: Decision from tree evaluation
 */
enum archangel_syscall_decision archangel_decision_tree_evaluate(
    struct archangel_decision_tree *tree, 
    struct archangel_syscall_context *context)
{
    struct archangel_decision_node *node;
    
    if (!tree || !tree->root || !context) {
        return ARCHANGEL_SYSCALL_ALLOW;
    }
    
    node = tree->root;
    
    /* Simple evaluation for single-node tree */
    /* In a full implementation, this would traverse the tree based on conditions */
    return node->decision;
}

/**
 * archangel_pattern_init - Initialize pattern matching system
 * 
 * Returns: 0 on success, negative error code on failure
 */
int archangel_pattern_init(void)
{
    int i, ret;
    
    if (!archangel_syscall_ai) {
        return -EINVAL;
    }
    
    /* Initialize pattern array */
    memset(archangel_syscall_ai->patterns, 0, sizeof(archangel_syscall_ai->patterns));
    atomic_set(&archangel_syscall_ai->pattern_count, 0);
    
    /* Load default patterns */
    for (i = 0; i < ARRAY_SIZE(default_patterns); i++) {
        ret = archangel_pattern_add(&default_patterns[i]);
        if (ret) {
            pr_warn("archangel_syscall_ai: Failed to add default pattern %d: %d\n", i, ret);
        }
    }
    
    pr_info("archangel_syscall_ai: Pattern matching initialized with %d patterns\n",
            atomic_read(&archangel_syscall_ai->pattern_count));
    return 0;
}

/**
 * archangel_pattern_cleanup - Clean up pattern matching system
 */
void archangel_pattern_cleanup(void)
{
    if (!archangel_syscall_ai)
        return;
    
    /* Clear all patterns */
    memset(archangel_syscall_ai->patterns, 0, sizeof(archangel_syscall_ai->patterns));
    atomic_set(&archangel_syscall_ai->pattern_count, 0);
    
    pr_info("archangel_syscall_ai: Pattern matching cleaned up\n");
}

/**
 * archangel_pattern_add - Add a new pattern to the database
 * @pattern: Pattern to add
 * 
 * Returns: Pattern ID on success, negative error code on failure
 */
int archangel_pattern_add(struct archangel_syscall_pattern *pattern)
{
    int pattern_id;
    int current_count;
    
    if (!pattern || !archangel_syscall_ai) {
        return -EINVAL;
    }
    
    current_count = atomic_read(&archangel_syscall_ai->pattern_count);
    if (current_count >= ARCHANGEL_SYSCALL_MAX_PATTERNS) {
        pr_err("archangel_syscall_ai: Pattern database full\n");
        return -ENOSPC;
    }
    
    /* Find empty slot */
    for (pattern_id = 0; pattern_id < ARCHANGEL_SYSCALL_MAX_PATTERNS; pattern_id++) {
        if (!archangel_syscall_ai->patterns[pattern_id].enabled) {
            break;
        }
    }
    
    if (pattern_id >= ARCHANGEL_SYSCALL_MAX_PATTERNS) {
        return -ENOSPC;
    }
    
    /* Copy pattern */
    memcpy(&archangel_syscall_ai->patterns[pattern_id], pattern, sizeof(*pattern));
    archangel_syscall_ai->patterns[pattern_id].enabled = true;
    
    atomic_inc(&archangel_syscall_ai->pattern_count);
    
    pr_debug("archangel_syscall_ai: Added pattern %d: %s\n", pattern_id, pattern->description);
    return pattern_id;
}

/**
 * archangel_pattern_remove - Remove a pattern from the database
 * @pattern_id: Pattern ID to remove
 */
void archangel_pattern_remove(u32 pattern_id)
{
    if (!archangel_syscall_ai || pattern_id >= ARCHANGEL_SYSCALL_MAX_PATTERNS) {
        return;
    }
    
    if (archangel_syscall_ai->patterns[pattern_id].enabled) {
        archangel_syscall_ai->patterns[pattern_id].enabled = false;
        atomic_dec(&archangel_syscall_ai->pattern_count);
        
        pr_debug("archangel_syscall_ai: Removed pattern %u\n", pattern_id);
    }
}

/**
 * archangel_pattern_match - Check for pattern matches in process behavior
 * @profile: Process profile to check
 * @context: Current syscall context
 * 
 * Returns: Number of patterns matched
 */
u32 archangel_pattern_match(struct archangel_process_profile *profile, 
                           struct archangel_syscall_context *context)
{
    u32 matches = 0;
    int i, j, k;
    
    if (!profile || !context || !archangel_syscall_ai) {
        return 0;
    }
    
    /* Check each active pattern */
    for (i = 0; i < ARCHANGEL_SYSCALL_MAX_PATTERNS; i++) {
        struct archangel_syscall_pattern *pattern = &archangel_syscall_ai->patterns[i];
        
        if (!pattern->enabled) {
            continue;
        }
        
        /* Check sequence patterns */
        if (pattern->type == ARCHANGEL_PATTERN_SEQUENCE) {
            bool match = true;
            
            /* Check if recent syscalls match the pattern */
            for (j = 0; j < pattern->count; j++) {
                bool found = false;
                
                /* Look for syscall in recent history */
                for (k = 0; k < 32; k++) {
                    int idx = (profile->recent_head - k - 1) & 31;
                    if (profile->recent_syscalls[idx] == pattern->syscalls[j]) {
                        found = true;
                        break;
                    }
                }
                
                if (!found) {
                    match = false;
                    break;
                }
            }
            
            if (match) {
                matches++;
                atomic_inc(&profile->pattern_matches);
                
                /* Update risk score based on pattern */
                atomic_add(pattern->risk_score / 4, &profile->risk_score);
                
                pr_debug("archangel_syscall_ai: Pattern match for PID %d: %s\n",
                        profile->pid, pattern->description);
            }
        }
    }
    
    return matches;
}

/**
 * archangel_profile_get_or_create - Get or create process profile
 * @pid: Process ID
 * 
 * Returns: Process profile pointer or NULL on failure
 */
struct archangel_process_profile *archangel_profile_get_or_create(pid_t pid)
{
    struct archangel_process_profile *profile;
    struct rb_node **new, *parent = NULL;
    unsigned long flags;
    int i;
    
    if (!archangel_syscall_ai) {
        return NULL;
    }
    
    spin_lock_irqsave(&archangel_syscall_ai->profile_lock, flags);
    
    /* Search for existing profile */
    new = &archangel_syscall_ai->process_profiles.rb_node;
    while (*new) {
        profile = rb_entry(*new, struct archangel_process_profile, rb_node);
        parent = *new;
        
        if (pid < profile->pid) {
            new = &((*new)->rb_left);
        } else if (pid > profile->pid) {
            new = &((*new)->rb_right);
        } else {
            /* Found existing profile */
            spin_unlock_irqrestore(&archangel_syscall_ai->profile_lock, flags);
            return profile;
        }
    }
    
    spin_unlock_irqrestore(&archangel_syscall_ai->profile_lock, flags);
    
    /* Create new profile */
    profile = kzalloc(sizeof(*profile), GFP_ATOMIC);
    if (!profile) {
        pr_err("archangel_syscall_ai: Failed to allocate process profile\n");
        return NULL;
    }
    
    /* Initialize profile */
    profile->pid = pid;
    get_task_comm(profile->comm, current);
    
    for (i = 0; i < __NR_syscalls; i++) {
        atomic_set(&profile->syscall_counts[i], 0);
    }
    
    memset(profile->recent_syscalls, 0, sizeof(profile->recent_syscalls));
    profile->recent_head = 0;
    atomic_set(&profile->risk_score, 0);
    profile->creation_time = ktime_get_ns();
    profile->last_activity = profile->creation_time;
    atomic_set(&profile->pattern_matches, 0);
    profile->suspicious_flags = 0;
    
    /* Insert into red-black tree */
    spin_lock_irqsave(&archangel_syscall_ai->profile_lock, flags);
    rb_link_node(&profile->rb_node, parent, new);
    rb_insert_color(&profile->rb_node, &archangel_syscall_ai->process_profiles);
    spin_unlock_irqrestore(&archangel_syscall_ai->profile_lock, flags);
    
    atomic64_inc(&archangel_syscall_ai->stats.processes_profiled);
    
    pr_debug("archangel_syscall_ai: Created profile for PID %d (%s)\n", pid, profile->comm);
    return profile;
}

/**
 * archangel_profile_update - Update process profile with syscall information
 * @profile: Process profile to update
 * @context: Syscall context
 */
void archangel_profile_update(struct archangel_process_profile *profile, 
                             struct archangel_syscall_context *context)
{
    if (!profile || !context) {
        return;
    }
    
    /* Update syscall count */
    if (context->syscall_nr < __NR_syscalls) {
        atomic_inc(&profile->syscall_counts[context->syscall_nr]);
    }
    
    /* Update recent syscalls ring buffer */
    profile->recent_syscalls[profile->recent_head] = context->syscall_nr;
    profile->recent_head = (profile->recent_head + 1) & 31;
    
    /* Update activity timestamp */
    profile->last_activity = context->timestamp;
    
    /* Check for suspicious behavior */
    if (archangel_syscall_is_high_risk(context->syscall_nr)) {
        profile->suspicious_flags |= (1 << 0); /* High-risk syscall flag */
    }
    
    /* Update risk score based on syscall frequency */
    if (atomic_read(&profile->syscall_counts[context->syscall_nr]) > 100) {
        atomic_add(1, &profile->risk_score);
    }
}

/**
 * archangel_profile_cleanup_expired - Clean up expired process profiles
 */
void archangel_profile_cleanup_expired(void)
{
    struct archangel_process_profile *profile, *next;
    struct rb_node *node;
    unsigned long flags;
    u64 current_time, expiry_time;
    
    if (!archangel_syscall_ai) {
        return;
    }
    
    current_time = ktime_get_ns();
    expiry_time = 300ULL * NSEC_PER_SEC; /* 5 minutes */
    
    spin_lock_irqsave(&archangel_syscall_ai->profile_lock, flags);
    
    /* Iterate through all profiles */
    for (node = rb_first(&archangel_syscall_ai->process_profiles); node; ) {
        profile = rb_entry(node, struct archangel_process_profile, rb_node);
        node = rb_next(node);
        
        /* Check if profile is expired */
        if (current_time - profile->last_activity > expiry_time) {
            pr_debug("archangel_syscall_ai: Cleaning up expired profile for PID %d\n", 
                    profile->pid);
            
            rb_erase(&profile->rb_node, &archangel_syscall_ai->process_profiles);
            kfree(profile);
        }
    }
    
    spin_unlock_irqrestore(&archangel_syscall_ai->profile_lock, flags);
}

/**
 * archangel_profile_calculate_risk - Calculate risk score for process
 * @profile: Process profile
 * 
 * Returns: Risk score (0-100)
 */
u8 archangel_profile_calculate_risk(struct archangel_process_profile *profile)
{
    u32 base_risk, pattern_risk, frequency_risk, time_risk;
    u64 age, current_time;
    int i;
    
    if (!profile) {
        return 0;
    }
    
    base_risk = atomic_read(&profile->risk_score);
    pattern_risk = atomic_read(&profile->pattern_matches) * 10;
    
    /* Calculate frequency-based risk */
    frequency_risk = 0;
    for (i = 0; i < __NR_syscalls; i++) {
        int count = atomic_read(&profile->syscall_counts[i]);
        if (count > 1000) {
            frequency_risk += 5;
        }
    }
    
    /* Calculate time-based risk (newer processes are riskier) */
    current_time = ktime_get_ns();
    age = current_time - profile->creation_time;
    if (age < 60ULL * NSEC_PER_SEC) { /* Less than 1 minute old */
        time_risk = 20;
    } else if (age < 300ULL * NSEC_PER_SEC) { /* Less than 5 minutes old */
        time_risk = 10;
    } else {
        time_risk = 0;
    }
    
    /* Combine risk factors */
    u32 total_risk = base_risk + pattern_risk + frequency_risk + time_risk;
    
    /* Cap at 100 */
    return (u8)min(total_risk, 100U);
}/**
 * 
archangel_decision_cache_init - Initialize decision cache
 * @cache: Decision cache to initialize
 * @size: Size of cache hash table
 * 
 * Returns: 0 on success, negative error code on failure
 */
int archangel_decision_cache_init(struct archangel_decision_cache *cache, u32 size)
{
    int i;
    
    if (!cache || size == 0) {
        return -EINVAL;
    }
    
    /* Ensure size is power of 2 */
    size = archangel_comm_next_power_of_2(size);
    
    cache->entries = kzalloc(size * sizeof(struct archangel_decision_cache_entry *), GFP_KERNEL);
    if (!cache->entries) {
        pr_err("archangel_syscall_ai: Failed to allocate cache entries\n");
        return -ENOMEM;
    }
    
    cache->size = size;
    cache->mask = size - 1;
    atomic64_set(&cache->hit_count, 0);
    atomic64_set(&cache->miss_count, 0);
    spin_lock_init(&cache->lock);
    
    /* Initialize hash table */
    for (i = 0; i < size; i++) {
        cache->entries[i] = NULL;
    }
    
    pr_debug("archangel_syscall_ai: Decision cache initialized with %u entries\n", size);
    return 0;
}

/**
 * archangel_decision_cache_cleanup - Clean up decision cache
 * @cache: Decision cache to clean up
 */
void archangel_decision_cache_cleanup(struct archangel_decision_cache *cache)
{
    struct archangel_decision_cache_entry *entry, *next;
    unsigned long flags;
    int i;
    
    if (!cache || !cache->entries) {
        return;
    }
    
    spin_lock_irqsave(&cache->lock, flags);
    
    /* Free all cache entries */
    for (i = 0; i < cache->size; i++) {
        entry = cache->entries[i];
        while (entry) {
            next = entry->next;
            kfree(entry);
            entry = next;
        }
        cache->entries[i] = NULL;
    }
    
    spin_unlock_irqrestore(&cache->lock, flags);
    
    kfree(cache->entries);
    cache->entries = NULL;
    cache->size = 0;
    
    pr_debug("archangel_syscall_ai: Decision cache cleaned up\n");
}

/**
 * archangel_decision_cache_lookup - Look up cached decision
 * @cache: Decision cache
 * @key: Cache key
 * 
 * Returns: Cached decision or ARCHANGEL_SYSCALL_UNKNOWN if not found
 */
enum archangel_syscall_decision archangel_decision_cache_lookup(
    struct archangel_decision_cache *cache, u64 key)
{
    struct archangel_decision_cache_entry *entry;
    enum archangel_syscall_decision decision = ARCHANGEL_SYSCALL_UNKNOWN;
    unsigned long flags;
    u32 hash;
    u64 current_time;
    
    if (!cache || !cache->entries) {
        return ARCHANGEL_SYSCALL_UNKNOWN;
    }
    
    hash = key & cache->mask;
    current_time = ktime_get_ns();
    
    spin_lock_irqsave(&cache->lock, flags);
    
    /* Search hash chain */
    entry = cache->entries[hash];
    while (entry) {
        if (entry->key == key) {
            /* Check if entry is still valid (not older than 1 second) */
            if (current_time - entry->timestamp < NSEC_PER_SEC) {
                decision = entry->decision;
                atomic_inc(&entry->hit_count);
                atomic64_inc(&cache->hit_count);
            } else {
                /* Entry expired, remove it */
                /* For simplicity, we'll just mark it as invalid */
                entry->decision = ARCHANGEL_SYSCALL_UNKNOWN;
            }
            break;
        }
        entry = entry->next;
    }
    
    if (decision == ARCHANGEL_SYSCALL_UNKNOWN) {
        atomic64_inc(&cache->miss_count);
    }
    
    spin_unlock_irqrestore(&cache->lock, flags);
    
    return decision;
}

/**
 * archangel_decision_cache_store - Store decision in cache
 * @cache: Decision cache
 * @key: Cache key
 * @decision: Decision to store
 */
void archangel_decision_cache_store(struct archangel_decision_cache *cache, 
                                   u64 key, enum archangel_syscall_decision decision)
{
    struct archangel_decision_cache_entry *entry, *new_entry;
    unsigned long flags;
    u32 hash;
    
    if (!cache || !cache->entries || decision == ARCHANGEL_SYSCALL_UNKNOWN) {
        return;
    }
    
    hash = key & cache->mask;
    
    /* Allocate new entry */
    new_entry = kzalloc(sizeof(*new_entry), GFP_ATOMIC);
    if (!new_entry) {
        pr_debug("archangel_syscall_ai: Failed to allocate cache entry\n");
        return;
    }
    
    new_entry->key = key;
    new_entry->decision = decision;
    new_entry->timestamp = ktime_get_ns();
    atomic_set(&new_entry->hit_count, 0);
    new_entry->next = NULL;
    
    spin_lock_irqsave(&cache->lock, flags);
    
    /* Check if entry already exists */
    entry = cache->entries[hash];
    while (entry) {
        if (entry->key == key) {
            /* Update existing entry */
            entry->decision = decision;
            entry->timestamp = new_entry->timestamp;
            spin_unlock_irqrestore(&cache->lock, flags);
            kfree(new_entry);
            return;
        }
        entry = entry->next;
    }
    
    /* Add new entry to head of chain */
    new_entry->next = cache->entries[hash];
    cache->entries[hash] = new_entry;
    
    spin_unlock_irqrestore(&cache->lock, flags);
}

/**
 * archangel_userspace_defer_init - Initialize userspace deferral system
 * 
 * Returns: 0 on success, negative error code on failure
 */
int archangel_userspace_defer_init(void)
{
    if (!archangel_syscall_ai) {
        return -EINVAL;
    }
    
    /* Userspace deferral queue is already initialized in main init function */
    pr_info("archangel_syscall_ai: Userspace deferral system initialized\n");
    return 0;
}

/**
 * archangel_userspace_defer_cleanup - Clean up userspace deferral system
 */
void archangel_userspace_defer_cleanup(void)
{
    /* Queue cleanup is handled in main cleanup function */
    pr_info("archangel_syscall_ai: Userspace deferral system cleaned up\n");
}

/**
 * archangel_userspace_defer_request - Send syscall to userspace for analysis
 * @context: Syscall context
 * @profile: Process profile
 * 
 * Returns: 0 on success, negative error code on failure
 */
int archangel_userspace_defer_request(struct archangel_syscall_context *context,
                                     struct archangel_process_profile *profile)
{
    struct archangel_userspace_defer_request request;
    int ret;
    
    if (!context || !profile || !archangel_syscall_ai) {
        return -EINVAL;
    }
    
    /* Prepare deferral request */
    memcpy(&request.context, context, sizeof(request.context));
    request.profile = profile;
    request.risk_score = archangel_profile_calculate_risk(profile);
    request.pattern_matches = atomic_read(&profile->pattern_matches);
    request.timeout = ktime_get_ns() + ARCHANGEL_SYSCALL_DECISION_TIMEOUT_NS;
    
    /* Send to userspace via communication channel */
    if (archangel_comm_is_initialized()) {
        ret = archangel_comm_send_message(0, /* Default channel */
                                         ARCHANGEL_MSG_SYSCALL_EVENT,
                                         ARCHANGEL_PRIORITY_HIGH,
                                         &request, sizeof(request));
        if (ret) {
            pr_debug("archangel_syscall_ai: Failed to send deferral request: %d\n", ret);
            return ret;
        }
        
        atomic64_inc(&archangel_syscall_ai->stats.userspace_requests);
    }
    
    return 0;
}

/**
 * archangel_userspace_defer_wait_response - Wait for userspace response
 * @request_id: Request ID to wait for
 * 
 * Returns: Decision from userspace or timeout decision
 */
enum archangel_syscall_decision archangel_userspace_defer_wait_response(u64 request_id)
{
    /* For now, return a default decision after timeout */
    /* In a full implementation, this would wait for userspace response */
    return ARCHANGEL_SYSCALL_ALLOW;
}

/**
 * archangel_syscall_ai_module_init - Module initialization
 */
static int __init archangel_syscall_ai_module_init(void)
{
    int ret;
    
    pr_info("archangel_syscall_ai: Initializing Syscall AI Filter Module\n");
    
    /* Initialize syscall AI engine */
    ret = archangel_syscall_ai_init();
    if (ret) {
        pr_err("archangel_syscall_ai: Failed to initialize engine: %d\n", ret);
        return ret;
    }
    
    /* Initialize userspace deferral */
    ret = archangel_userspace_defer_init();
    if (ret) {
        pr_err("archangel_syscall_ai: Failed to initialize userspace deferral: %d\n", ret);
        archangel_syscall_ai_cleanup();
        return ret;
    }
    
    pr_info("archangel_syscall_ai: Module loaded successfully\n");
    return 0;
}

/**
 * archangel_syscall_ai_module_exit - Module cleanup
 */
static void __exit archangel_syscall_ai_module_exit(void)
{
    pr_info("archangel_syscall_ai: Shutting down Syscall AI Filter Module\n");
    
    /* Clean up userspace deferral */
    archangel_userspace_defer_cleanup();
    
    /* Clean up syscall AI engine */
    archangel_syscall_ai_cleanup();
    
    pr_info("archangel_syscall_ai: Module unloaded successfully\n");
}

module_init(archangel_syscall_ai_module_init);
module_exit(archangel_syscall_ai_module_exit);

/* Export symbols for other modules */
EXPORT_SYMBOL(ai_syscall_intercept);
EXPORT_SYMBOL(archangel_syscall_ai_enable);
EXPORT_SYMBOL(archangel_syscall_ai_disable);
EXPORT_SYMBOL(archangel_syscall_ai_is_enabled);