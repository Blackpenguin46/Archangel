/*
 * Archangel Linux - AI Security Expert Kernel Module
 * Main header file for kernel-userspace communication
 */

#ifndef _ARCHANGEL_H
#define _ARCHANGEL_H

#include <linux/types.h>
#include <linux/ioctl.h>

#define ARCHANGEL_MODULE_NAME "archangel"
#define ARCHANGEL_VERSION "0.1.0"

/* Maximum sizes for communication */
#define ARCHANGEL_MAX_MSG_SIZE      4096
#define ARCHANGEL_MAX_TARGETS       256
#define ARCHANGEL_MAX_RULES         1024
#define ARCHANGEL_SHARED_MEM_SIZE   (1024 * 1024)  /* 1MB shared memory */

/* Communication channel types */
enum archangel_channel {
    ARCHANGEL_CHANNEL_CONTROL = 0,
    ARCHANGEL_CHANNEL_DATA,
    ARCHANGEL_CHANNEL_EVENTS,
    ARCHANGEL_CHANNEL_MAX
};

/* Message types for kernel-userspace communication */
enum archangel_msg_type {
    ARCHANGEL_MSG_PING = 1,
    ARCHANGEL_MSG_PONG,
    ARCHANGEL_MSG_ANALYSIS_REQUEST,
    ARCHANGEL_MSG_ANALYSIS_RESPONSE,
    ARCHANGEL_MSG_RULE_UPDATE,
    ARCHANGEL_MSG_EVENT_NOTIFICATION,
    ARCHANGEL_MSG_STATUS_REQUEST,
    ARCHANGEL_MSG_STATUS_RESPONSE,
    ARCHANGEL_MSG_SHUTDOWN
};

/* Security decision types */
enum archangel_decision {
    ARCHANGEL_ALLOW = 0,
    ARCHANGEL_DENY,
    ARCHANGEL_MONITOR,
    ARCHANGEL_DEFER_TO_USERSPACE,
    ARCHANGEL_UNKNOWN
};

/* Confidence levels for AI decisions */
enum archangel_confidence {
    ARCHANGEL_CONFIDENCE_LOW = 0,
    ARCHANGEL_CONFIDENCE_MEDIUM,
    ARCHANGEL_CONFIDENCE_HIGH,
    ARCHANGEL_CONFIDENCE_VERY_HIGH
};

/* Security context structure */
struct archangel_security_context {
    u32 pid;                    /* Process ID */
    u32 uid;                    /* User ID */
    u32 syscall_nr;             /* System call number */
    u64 timestamp;              /* Timestamp in nanoseconds */
    u32 flags;                  /* Context flags */
    char comm[16];              /* Process command name */
    u32 data_size;              /* Size of additional data */
    u8 data[0];                 /* Variable length data */
} __packed;

/* Security rule structure */
struct archangel_rule {
    u32 id;                     /* Rule ID */
    u32 priority;               /* Rule priority (lower = higher priority) */
    u32 condition_mask;         /* Condition bitmask */
    u32 condition_values;       /* Expected values */
    enum archangel_decision action;  /* Action to take */
    enum archangel_confidence confidence;  /* AI confidence in rule */
    u64 created_time;           /* Rule creation timestamp */
    u64 last_matched;           /* Last time rule matched */
    u32 match_count;            /* Number of times matched */
    char description[64];       /* Human readable description */
} __packed;

/* Communication message structure */
struct archangel_message {
    enum archangel_msg_type type;
    u32 sequence;               /* Message sequence number */
    u32 data_size;              /* Size of message data */
    u64 timestamp;              /* Message timestamp */
    u32 flags;                  /* Message flags */
    u8 data[0];                 /* Variable length message data */
} __packed;

/* Statistics structure */
struct archangel_stats {
    u64 total_decisions;        /* Total security decisions made */
    u64 allow_decisions;        /* Number of ALLOW decisions */
    u64 deny_decisions;         /* Number of DENY decisions */
    u64 monitor_decisions;      /* Number of MONITOR decisions */
    u64 deferred_decisions;     /* Decisions deferred to userspace */
    u64 rule_matches;           /* Total rule matches */
    u64 cache_hits;             /* Decision cache hits */
    u64 cache_misses;           /* Decision cache misses */
    u64 userspace_requests;     /* Requests sent to userspace */
    u64 userspace_responses;    /* Responses from userspace */
    u64 avg_decision_time_ns;   /* Average decision time in nanoseconds */
    u64 max_decision_time_ns;   /* Maximum decision time */
    u64 uptime_seconds;         /* Module uptime in seconds */
} __packed;

/* IOCTL commands */
#define ARCHANGEL_IOC_MAGIC 'A'
#define ARCHANGEL_IOC_GET_STATS     _IOR(ARCHANGEL_IOC_MAGIC, 1, struct archangel_stats)
#define ARCHANGEL_IOC_ADD_RULE      _IOW(ARCHANGEL_IOC_MAGIC, 2, struct archangel_rule)
#define ARCHANGEL_IOC_DEL_RULE      _IOW(ARCHANGEL_IOC_MAGIC, 3, u32)
#define ARCHANGEL_IOC_CLEAR_RULES   _IO(ARCHANGEL_IOC_MAGIC, 4)
#define ARCHANGEL_IOC_GET_STATUS    _IOR(ARCHANGEL_IOC_MAGIC, 5, u32)
#define ARCHANGEL_IOC_SET_MODE      _IOW(ARCHANGEL_IOC_MAGIC, 6, u32)

/* Mode flags */
#define ARCHANGEL_MODE_ACTIVE       0x0001
#define ARCHANGEL_MODE_LEARNING     0x0002
#define ARCHANGEL_MODE_MONITORING   0x0004
#define ARCHANGEL_MODE_DEBUG        0x0008

/* Proc filesystem paths */
#define ARCHANGEL_PROC_ROOT         "archangel"
#define ARCHANGEL_PROC_STATUS       "status"
#define ARCHANGEL_PROC_STATS        "stats"
#define ARCHANGEL_PROC_RULES        "rules"
#define ARCHANGEL_PROC_LOG          "log"
#define ARCHANGEL_PROC_CONTROL      "control"

/* Shared memory structure */
struct archangel_shared_memory {
    /* Control section */
    struct {
        volatile u32 kernel_sequence;
        volatile u32 user_sequence;
        volatile u32 kernel_flags;
        volatile u32 user_flags;
    } control;
    
    /* Message queues */
    struct {
        volatile u32 head;
        volatile u32 tail;
        u32 size;
        struct archangel_message messages[0];
    } kernel_to_user_queue;
    
    struct {
        volatile u32 head; 
        volatile u32 tail;
        u32 size;
        struct archangel_message messages[0];
    } user_to_kernel_queue;
    
    /* Event buffer */
    struct {
        volatile u32 head;
        volatile u32 tail;
        u32 size;
        u8 events[0];
    } event_buffer;
} __packed;

#ifdef __KERNEL__

/* Kernel-only definitions */

/* Function prototypes */
int archangel_init_communication(void);
void archangel_cleanup_communication(void);
int archangel_send_message(enum archangel_msg_type type, const void *data, u32 size);
int archangel_receive_message(struct archangel_message **msg);
enum archangel_decision archangel_make_decision(struct archangel_security_context *ctx);
int archangel_add_rule(const struct archangel_rule *rule);
int archangel_remove_rule(u32 rule_id);
void archangel_get_stats(struct archangel_stats *stats);

/* Syscall interception prototypes */
int archangel_syscall_init(void);
void archangel_syscall_cleanup(void);
u64 archangel_syscall_get_intercept_count(void);
void archangel_syscall_set_enabled(bool enabled);
bool archangel_syscall_is_enabled(void);
const char *archangel_syscall_name(long syscall_nr);

/* Communication prototypes */
void archangel_get_comm_stats(u64 *sent, u64 *received, u64 *queue_full, u64 *queue_empty);

/* Logging macros */
#define archangel_info(fmt, ...) \
    pr_info("archangel: " fmt, ##__VA_ARGS__)

#define archangel_warn(fmt, ...) \
    pr_warn("archangel: " fmt, ##__VA_ARGS__)

#define archangel_err(fmt, ...) \
    pr_err("archangel: " fmt, ##__VA_ARGS__)

#define archangel_debug(fmt, ...) \
    pr_debug("archangel: " fmt, ##__VA_ARGS__)

#endif /* __KERNEL__ */

#endif /* _ARCHANGEL_H */