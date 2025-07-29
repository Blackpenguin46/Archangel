/*
 * Archangel Linux - System Call Interception
 * Monitors critical system calls for security analysis
 */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/kprobes.h>
#include <linux/ptrace.h>
#include <linux/syscalls.h>
#include <linux/sched.h>
#include <linux/sched/task.h>
#include <linux/string.h>
#include <linux/uaccess.h>
#include <linux/slab.h>

#include "../include/archangel.h"

/* Critical syscalls to monitor */
static const long monitored_syscalls[] = {
    __NR_execve,        /* Process execution */
    __NR_open,          /* File opening */
    __NR_openat,        /* File opening (modern) */
    __NR_ptrace,        /* Process tracing */
    __NR_mount,         /* Filesystem mounting */
    __NR_umount2,       /* Filesystem unmounting */
    __NR_init_module,   /* Kernel module loading */
    __NR_delete_module, /* Kernel module unloading */
    __NR_socket,        /* Network socket creation */
    __NR_bind,          /* Network binding */
    __NR_connect,       /* Network connection */
    __NR_listen,        /* Network listening */
    __NR_accept,        /* Network accepting */
    __NR_kill,          /* Signal sending */
    __NR_setuid,        /* User ID change */
    __NR_setgid,        /* Group ID change */
    __NR_chmod,         /* Permission change */
    __NR_chown,         /* Ownership change */
};

#define NUM_MONITORED_SYSCALLS (sizeof(monitored_syscalls) / sizeof(monitored_syscalls[0]))

/* Syscall interception state */
static struct {
    bool enabled;
    atomic64_t intercept_count;
    struct kprobe kprobes[NUM_MONITORED_SYSCALLS];
    bool kprobes_registered[NUM_MONITORED_SYSCALLS];
} syscall_state = {
    .enabled = false,
};

/* Forward declarations */
static int archangel_syscall_handler(struct kprobe *p, struct pt_regs *regs);
static bool is_monitored_syscall(long syscall_nr);
static int create_security_context(struct pt_regs *regs, long syscall_nr, 
                                 struct archangel_security_context **ctx);

/*
 * Initialize syscall interception
 */
int archangel_syscall_init(void)
{
    int ret, i;
    char probe_name[32];
    
    archangel_info("Initializing syscall interception");
    
    /* Register kprobes for monitored syscalls */
    for (i = 0; i < NUM_MONITORED_SYSCALLS; i++) {
        memset(&syscall_state.kprobes[i], 0, sizeof(struct kprobe));
        
        /* Create probe symbol name */
        snprintf(probe_name, sizeof(probe_name), "__x64_sys_%s", 
                 archangel_syscall_name(monitored_syscalls[i]));
        
        syscall_state.kprobes[i].symbol_name = kstrdup(probe_name, GFP_KERNEL);
        if (!syscall_state.kprobes[i].symbol_name) {
            ret = -ENOMEM;
            goto cleanup;
        }
        
        syscall_state.kprobes[i].pre_handler = archangel_syscall_handler;
        
        ret = register_kprobe(&syscall_state.kprobes[i]);
        if (ret < 0) {
            archangel_warn("Failed to register kprobe for syscall %ld: %d", 
                          monitored_syscalls[i], ret);
            kfree(syscall_state.kprobes[i].symbol_name);
            continue;
        }
        
        syscall_state.kprobes_registered[i] = true;
        archangel_debug("Registered kprobe for syscall %ld (%s)", 
                       monitored_syscalls[i], probe_name);
    }
    
    syscall_state.enabled = true;
    atomic64_set(&syscall_state.intercept_count, 0);
    
    archangel_info("Syscall interception initialized, monitoring %zu syscalls", 
                   NUM_MONITORED_SYSCALLS);
    
    return 0;

cleanup:
    archangel_syscall_cleanup();
    return ret;
}

/*
 * Cleanup syscall interception
 */
void archangel_syscall_cleanup(void)
{
    int i;
    
    archangel_info("Cleaning up syscall interception");
    
    syscall_state.enabled = false;
    
    /* Unregister all kprobes */
    for (i = 0; i < NUM_MONITORED_SYSCALLS; i++) {
        if (syscall_state.kprobes_registered[i]) {
            unregister_kprobe(&syscall_state.kprobes[i]);
            syscall_state.kprobes_registered[i] = false;
        }
        
        if (syscall_state.kprobes[i].symbol_name) {
            kfree(syscall_state.kprobes[i].symbol_name);
            syscall_state.kprobes[i].symbol_name = NULL;
        }
    }
    
    archangel_info("Syscall interception cleaned up, %lld calls intercepted", 
                   atomic64_read(&syscall_state.intercept_count));
}

/*
 * Main syscall interception handler
 */
static int archangel_syscall_handler(struct kprobe *p, struct pt_regs *regs)
{
    struct archangel_security_context *ctx = NULL;
    enum archangel_decision decision;
    long syscall_nr;
    int ret = 0; /* Continue with normal syscall execution */
    
    if (!syscall_state.enabled)
        return 0;
    
    atomic64_inc(&syscall_state.intercept_count);
    
    /* Get syscall number from registers */
#ifdef CONFIG_X86_64
    syscall_nr = regs->orig_ax;
#else
    syscall_nr = -1; /* Fallback for other architectures */
#endif
    
    /* Only process monitored syscalls */
    if (!is_monitored_syscall(syscall_nr))
        return 0;
    
    /* Create security context */
    if (create_security_context(regs, syscall_nr, &ctx) != 0)
        return 0; /* Continue on error */
    
    /* Make security decision */
    decision = archangel_make_decision(ctx);
    
    /* Log interesting decisions */
    if (decision != ARCHANGEL_ALLOW) {
        archangel_info("Syscall %ld from PID %u (%s): %s", 
                      syscall_nr, current->pid, current->comm,
                      (decision == ARCHANGEL_DENY) ? "DENIED" :
                      (decision == ARCHANGEL_MONITOR) ? "MONITORED" : "DEFERRED");
    }
    
    /* Apply decision */
    switch (decision) {
    case ARCHANGEL_DENY:
        /* Block syscall by returning error */
        regs->ax = -EPERM;
        ret = 1; /* Skip original syscall */
        break;
        
    case ARCHANGEL_MONITOR:
        /* Allow but log the syscall */
        archangel_debug("Monitoring syscall %ld from PID %u", syscall_nr, current->pid);
        break;
        
    case ARCHANGEL_ALLOW:
    case ARCHANGEL_DEFER_TO_USERSPACE:
    default:
        /* Allow syscall to proceed normally */
        break;
    }
    
    /* Cleanup */
    if (ctx)
        kfree(ctx);
    
    return ret;
}

/*
 * Check if syscall is monitored
 */
static bool is_monitored_syscall(long syscall_nr)
{
    int i;
    
    for (i = 0; i < NUM_MONITORED_SYSCALLS; i++) {
        if (monitored_syscalls[i] == syscall_nr)
            return true;
    }
    
    return false;
}

/*
 * Create security context from syscall
 */
static int create_security_context(struct pt_regs *regs, long syscall_nr,
                                 struct archangel_security_context **ctx)
{
    struct archangel_security_context *new_ctx;
    size_t ctx_size;
    
    if (!ctx)
        return -EINVAL;
    
    /* Allocate context structure */
    ctx_size = sizeof(struct archangel_security_context) + 256; /* Extra data space */
    new_ctx = kmalloc(ctx_size, GFP_ATOMIC); /* Must use GFP_ATOMIC in kprobe */
    if (!new_ctx)
        return -ENOMEM;
    
    memset(new_ctx, 0, ctx_size);
    
    /* Fill in context information */
    new_ctx->pid = current->pid;
    new_ctx->uid = current_uid().val;
    new_ctx->syscall_nr = syscall_nr;
    new_ctx->timestamp = ktime_get_ns();
    new_ctx->flags = 0;
    
    /* Copy process command name */
    strncpy(new_ctx->comm, current->comm, sizeof(new_ctx->comm) - 1);
    new_ctx->comm[sizeof(new_ctx->comm) - 1] = '\0';
    
    /* Add syscall-specific data */
    switch (syscall_nr) {
    case __NR_execve:
        /* For execve, we could extract filename */
        new_ctx->flags |= 0x0001; /* EXECVE flag */
        break;
        
    case __NR_open:
    case __NR_openat:
        /* For file operations, we could extract filename */
        new_ctx->flags |= 0x0002; /* FILE_ACCESS flag */
        break;
        
    case __NR_socket:
        /* For network operations */
        new_ctx->flags |= 0x0004; /* NETWORK flag */
        break;
        
    case __NR_ptrace:
        /* For debugging operations */
        new_ctx->flags |= 0x0008; /* DEBUG flag */
        break;
        
    default:
        break;
    }
    
    new_ctx->data_size = 0; /* No additional data for now */
    
    *ctx = new_ctx;
    return 0;
}

/*
 * Get syscall interception statistics
 */
u64 archangel_syscall_get_intercept_count(void)
{
    return atomic64_read(&syscall_state.intercept_count);
}

/*
 * Enable/disable syscall interception
 */
void archangel_syscall_set_enabled(bool enabled)
{
    syscall_state.enabled = enabled;
    archangel_info("Syscall interception %s", enabled ? "enabled" : "disabled");
}

/*
 * Check if syscall interception is enabled
 */
bool archangel_syscall_is_enabled(void)
{
    return syscall_state.enabled;
}

/*
 * Get syscall name from number (simplified version)
 */
const char *archangel_syscall_name(long syscall_nr)
{
    switch (syscall_nr) {
    case __NR_execve: return "execve";
    case __NR_open: return "open";
    case __NR_openat: return "openat";
    case __NR_ptrace: return "ptrace";
    case __NR_mount: return "mount";
    case __NR_umount2: return "umount2";
    case __NR_init_module: return "init_module";
    case __NR_delete_module: return "delete_module";
    case __NR_socket: return "socket";
    case __NR_bind: return "bind";
    case __NR_connect: return "connect";
    case __NR_listen: return "listen";
    case __NR_accept: return "accept";
    case __NR_kill: return "kill";
    case __NR_setuid: return "setuid";
    case __NR_setgid: return "setgid";
    case __NR_chmod: return "chmod";
    case __NR_chown: return "chown";
    default: return "unknown";
    }
}