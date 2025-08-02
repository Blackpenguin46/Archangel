/*
 * Archangel Linux - Network Packet Inspection and Analysis
 * Real-time network traffic analysis for threat detection
 */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/netfilter.h>
#include <linux/netfilter_ipv4.h>
#include <linux/netfilter_ipv6.h>
#include <linux/ip.h>
#include <linux/ipv6.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <linux/icmp.h>
#include <linux/skbuff.h>
#include <linux/string.h>
#include <linux/slab.h>
#include <linux/atomic.h>
#include <linux/spinlock.h>
#include <linux/hash.h>
#include <linux/ktime.h>
#include <linux/jhash.h>
#include <net/ip.h>
#include <net/tcp.h>
#include <net/udp.h>

#include "../include/archangel.h"

/* Network analysis configuration */
#define MAX_TRACKED_CONNECTIONS    2048
#define MAX_PAYLOAD_ANALYSIS       1024
#define DDOS_THRESHOLD_PPS         1000    /* Packets per second */
#define PORT_SCAN_THRESHOLD        20      /* Unique ports per minute */
#define CONNECTION_TIMEOUT         300     /* Seconds */
#define SUSPICIOUS_PAYLOAD_BYTES   8

/* Network threat types */
enum network_threat_type {
    THREAT_NONE = 0,
    THREAT_PORT_SCAN,
    THREAT_DDOS,
    THREAT_MALICIOUS_PAYLOAD,
    THREAT_SUSPICIOUS_CONNECTION,
    THREAT_DATA_EXFILTRATION,
    THREAT_COMMAND_INJECTION,
    THREAT_BRUTE_FORCE
};

/* Network connection flags */
#define NET_FLAG_SUSPICIOUS         0x0001
#define NET_FLAG_HIGH_VOLUME        0x0002
#define NET_FLAG_PORT_SCAN          0x0004
#define NET_FLAG_MALICIOUS_PAYLOAD  0x0008
#define NET_FLAG_ENCRYPTED          0x0010
#define NET_FLAG_EXTERNAL           0x0020
#define NET_FLAG_BLOCKED            0x0040

/* Connection tracking structure */
struct network_connection {
    u32 src_ip;
    u32 dst_ip;
    u16 src_port;
    u16 dst_port;
    u8 protocol;
    
    /* Statistics */
    atomic64_t packet_count;
    atomic64_t byte_count;
    u64 first_seen;
    u64 last_seen;
    u32 flags;
    
    /* Threat analysis */
    enum network_threat_type threat_type;
    u32 threat_score;
    u32 unique_ports;
    u64 unique_port_bitmap; /* For port scan detection */
    
    /* Hash table linkage */
    struct hlist_node hash_node;
    atomic_t ref_count;
};

/* Host tracking for behavioral analysis */
struct network_host {
    u32 ip_address;
    
    /* Connection statistics */
    atomic_t active_connections;
    atomic64_t total_packets;
    atomic64_t total_bytes;
    u64 first_seen;
    u64 last_activity;
    
    /* Rate limiting */
    u32 packets_per_sec;
    u64 rate_window_start;
    u32 packets_in_window;
    
    /* Port scan detection */
    u16 accessed_ports[64]; /* Recent ports accessed */
    u8 port_count;
    u64 port_scan_window;
    
    /* Threat assessment */
    u32 threat_score;
    u32 flags;
    
    struct hlist_node hash_node;
    atomic_t ref_count;
};

/* Network analysis state */
static struct {
    bool enabled;
    
    /* Statistics */
    atomic64_t total_packets;
    atomic64_t blocked_packets;
    atomic64_t suspicious_packets;
    atomic64_t ddos_packets;
    atomic64_t malicious_payloads;
    
    /* Connection tracking */
    struct hlist_head connection_hash[512];
    spinlock_t connection_lock;
    atomic_t tracked_connections;
    
    /* Host tracking */
    struct hlist_head host_hash[256];
    spinlock_t host_lock;
    atomic_t tracked_hosts;
    
    /* Netfilter hooks */
    struct nf_hook_ops nf_hook_in;
    struct nf_hook_ops nf_hook_out;
    struct nf_hook_ops nf_hook_forward;
    bool hooks_registered;
    
} network_state = {
    .enabled = false,
    .connection_lock = __SPIN_LOCK_UNLOCKED(network_state.connection_lock),
    .host_lock = __SPIN_LOCK_UNLOCKED(network_state.host_lock),
};

/* Malicious payload signatures */
static const char *malicious_signatures[] = {
    "/bin/sh",          /* Shell execution */
    "/bin/bash",        /* Bash execution */
    "cmd.exe",          /* Windows command prompt */
    "powershell",       /* PowerShell */
    "eval(",            /* Code evaluation */
    "exec(",            /* Code execution */
    "system(",          /* System calls */
    "SELECT * FROM",    /* SQL injection */
    "UNION SELECT",     /* SQL injection */
    "' OR '1'='1",      /* SQL injection */
    "../../../",        /* Directory traversal */
    "../../../../",     /* Directory traversal */
    "<script",          /* XSS attack */
    "javascript:",      /* XSS attack */
    "alert(",           /* XSS alert */
};

#define NUM_MALICIOUS_SIGNATURES (sizeof(malicious_signatures) / sizeof(malicious_signatures[0]))

/* Forward declarations */
static struct network_connection *get_connection(u32 src_ip, u32 dst_ip, 
                                               u16 src_port, u16 dst_port, u8 protocol);
static void put_connection(struct network_connection *conn);
static struct network_host *get_host(u32 ip_address);
static void put_host(struct network_host *host);
static enum network_threat_type analyze_packet_content(struct sk_buff *skb);
static bool detect_port_scan(struct network_host *host, u16 port);
static bool detect_ddos(struct network_host *host);
static bool contains_malicious_payload(const char *data, size_t len);
static int make_network_decision(struct network_connection *conn, 
                               struct sk_buff *skb, enum network_threat_type threat);

/* Netfilter hook functions */
static unsigned int archangel_nf_hook_in(void *priv, struct sk_buff *skb,
                                        const struct nf_hook_state *state);
static unsigned int archangel_nf_hook_out(void *priv, struct sk_buff *skb,
                                         const struct nf_hook_state *state);
static unsigned int archangel_nf_hook_forward(void *priv, struct sk_buff *skb,
                                            const struct nf_hook_state *state);

/*
 * Initialize network analysis system
 */
int archangel_network_init(void)
{
    int ret, i;
    
    archangel_info("Initializing network analysis system");
    
    /* Initialize hash tables */
    for (i = 0; i < ARRAY_SIZE(network_state.connection_hash); i++) {
        INIT_HLIST_HEAD(&network_state.connection_hash[i]);
    }
    
    for (i = 0; i < ARRAY_SIZE(network_state.host_hash); i++) {
        INIT_HLIST_HEAD(&network_state.host_hash[i]);
    }
    
    /* Initialize statistics */
    atomic64_set(&network_state.total_packets, 0);
    atomic64_set(&network_state.blocked_packets, 0);
    atomic64_set(&network_state.suspicious_packets, 0);
    atomic64_set(&network_state.ddos_packets, 0);
    atomic64_set(&network_state.malicious_payloads, 0);
    atomic_set(&network_state.tracked_connections, 0);
    atomic_set(&network_state.tracked_hosts, 0);
    
    /* Register netfilter hooks */
    network_state.nf_hook_in.hook = archangel_nf_hook_in;
    network_state.nf_hook_in.hooknum = NF_INET_PRE_ROUTING;
    network_state.nf_hook_in.pf = PF_INET;
    network_state.nf_hook_in.priority = NF_IP_PRI_FIRST;
    
    ret = nf_register_net_hook(&init_net, &network_state.nf_hook_in);
    if (ret < 0) {
        archangel_err("Failed to register input netfilter hook: %d", ret);
        return ret;
    }
    
    network_state.nf_hook_out.hook = archangel_nf_hook_out;
    network_state.nf_hook_out.hooknum = NF_INET_POST_ROUTING;
    network_state.nf_hook_out.pf = PF_INET;
    network_state.nf_hook_out.priority = NF_IP_PRI_FIRST;
    
    ret = nf_register_net_hook(&init_net, &network_state.nf_hook_out);
    if (ret < 0) {
        archangel_err("Failed to register output netfilter hook: %d", ret);
        nf_unregister_net_hook(&init_net, &network_state.nf_hook_in);
        return ret;
    }
    
    network_state.nf_hook_forward.hook = archangel_nf_hook_forward;
    network_state.nf_hook_forward.hooknum = NF_INET_FORWARD;
    network_state.nf_hook_forward.pf = PF_INET;
    network_state.nf_hook_forward.priority = NF_IP_PRI_FIRST;
    
    ret = nf_register_net_hook(&init_net, &network_state.nf_hook_forward);
    if (ret < 0) {
        archangel_err("Failed to register forward netfilter hook: %d", ret);
        nf_unregister_net_hook(&init_net, &network_state.nf_hook_in);
        nf_unregister_net_hook(&init_net, &network_state.nf_hook_out);
        return ret;
    }
    
    network_state.hooks_registered = true;
    network_state.enabled = true;
    
    archangel_info("Network analysis system initialized with netfilter hooks");
    return 0;
}

/*
 * Cleanup network analysis system
 */
void archangel_network_cleanup(void)
{
    struct network_connection *conn;
    struct network_host *host;
    struct hlist_node *tmp;
    int i;
    
    archangel_info("Cleaning up network analysis system");
    
    network_state.enabled = false;
    
    /* Unregister netfilter hooks */
    if (network_state.hooks_registered) {
        nf_unregister_net_hook(&init_net, &network_state.nf_hook_in);
        nf_unregister_net_hook(&init_net, &network_state.nf_hook_out);
        nf_unregister_net_hook(&init_net, &network_state.nf_hook_forward);
        network_state.hooks_registered = false;
    }
    
    /* Cleanup connections */
    spin_lock(&network_state.connection_lock);
    for (i = 0; i < ARRAY_SIZE(network_state.connection_hash); i++) {
        hlist_for_each_entry_safe(conn, tmp, &network_state.connection_hash[i], hash_node) {
            hlist_del(&conn->hash_node);
            put_connection(conn);
        }
    }
    spin_unlock(&network_state.connection_lock);
    
    /* Cleanup hosts */
    spin_lock(&network_state.host_lock);
    for (i = 0; i < ARRAY_SIZE(network_state.host_hash); i++) {
        hlist_for_each_entry_safe(host, tmp, &network_state.host_hash[i], hash_node) {
            hlist_del(&host->hash_node);
            put_host(host);
        }
    }
    spin_unlock(&network_state.host_lock);
    
    archangel_info("Network analysis cleaned up, analyzed %lld packets, "
                   "blocked %lld suspicious packets",
                   atomic64_read(&network_state.total_packets),
                   atomic64_read(&network_state.blocked_packets));
}

/*
 * Main netfilter hook for incoming packets
 */
static unsigned int archangel_nf_hook_in(void *priv, struct sk_buff *skb,
                                        const struct nf_hook_state *state)
{
    struct iphdr *iph;
    struct network_connection *conn;
    struct network_host *src_host;
    enum network_threat_type threat;
    u32 src_ip, dst_ip;
    u16 src_port = 0, dst_port = 0;
    u8 protocol;
    u64 current_time;
    
    if (!network_state.enabled || !skb)
        return NF_ACCEPT;
    
    atomic64_inc(&network_state.total_packets);
    current_time = ktime_get_ns();
    
    /* Parse IP header */
    iph = ip_hdr(skb);
    if (!iph)
        return NF_ACCEPT;
    
    src_ip = ntohl(iph->saddr);
    dst_ip = ntohl(iph->daddr);
    protocol = iph->protocol;
    
    /* Extract port information for TCP/UDP */
    if (protocol == IPPROTO_TCP) {
        struct tcphdr *tcph = tcp_hdr(skb);
        if (tcph) {
            src_port = ntohs(tcph->source);
            dst_port = ntohs(tcph->dest);
        }
    } else if (protocol == IPPROTO_UDP) {
        struct udphdr *udph = udp_hdr(skb);
        if (udph) {
            src_port = ntohs(udph->source);
            dst_port = ntohs(udph->dest);
        }
    }
    
    /* Get or create connection tracking */
    conn = get_connection(src_ip, dst_ip, src_port, dst_port, protocol);
    if (conn) {
        atomic64_inc(&conn->packet_count);
        atomic64_add(skb->len, &conn->byte_count);
        conn->last_seen = current_time;
    }
    
    /* Get or create host tracking */
    src_host = get_host(src_ip);
    if (src_host) {
        atomic64_inc(&src_host->total_packets);
        atomic64_add(skb->len, &src_host->total_bytes);
        src_host->last_activity = current_time;
        
        /* Update rate tracking */
        if (current_time - src_host->rate_window_start > 1000000000ULL) { /* 1 second */
            src_host->packets_per_sec = src_host->packets_in_window;
            src_host->packets_in_window = 0;
            src_host->rate_window_start = current_time;
        }
        src_host->packets_in_window++;
        
        /* Check for threats */
        
        /* DDoS detection */
        if (detect_ddos(src_host)) {
            if (conn) conn->threat_type = THREAT_DDOS;
            atomic64_inc(&network_state.ddos_packets);
            archangel_warn("DDoS attack detected from %pI4, rate: %u pps",
                          &iph->saddr, src_host->packets_per_sec);
        }
        
        /* Port scan detection */
        if (dst_port && detect_port_scan(src_host, dst_port)) {
            if (conn) {
                conn->threat_type = THREAT_PORT_SCAN;
                conn->flags |= NET_FLAG_PORT_SCAN;
            }
            archangel_warn("Port scan detected from %pI4, port %u",
                          &iph->saddr, dst_port);
        }
    }
    
    /* Analyze packet content */
    threat = analyze_packet_content(skb);
    if (threat != THREAT_NONE) {
        if (conn) {
            conn->threat_type = threat;
            conn->flags |= NET_FLAG_MALICIOUS_PAYLOAD;
        }
        atomic64_inc(&network_state.suspicious_packets);
        
        if (threat == THREAT_MALICIOUS_PAYLOAD) {
            atomic64_inc(&network_state.malicious_payloads);
            archangel_warn("Malicious payload detected from %pI4:%u to %pI4:%u",
                          &iph->saddr, src_port, &iph->daddr, dst_port);
        }
    }
    
    /* Make security decision */
    if (conn && (conn->threat_type != THREAT_NONE || threat != THREAT_NONE)) {
        int decision = make_network_decision(conn, skb, threat);
        if (decision == ARCHANGEL_DENY) {
            atomic64_inc(&network_state.blocked_packets);
            if (conn) conn->flags |= NET_FLAG_BLOCKED;
            
            /* Cleanup references */
            if (conn) put_connection(conn);
            if (src_host) put_host(src_host);
            
            return NF_DROP;
        }
    }
    
    /* Cleanup references */
    if (conn) put_connection(conn);
    if (src_host) put_host(src_host);
    
    return NF_ACCEPT;
}

/*
 * Netfilter hook for outgoing packets
 */
static unsigned int archangel_nf_hook_out(void *priv, struct sk_buff *skb,
                                         const struct nf_hook_state *state)
{
    /* For now, just monitor outgoing traffic */
    if (network_state.enabled && skb) {
        atomic64_inc(&network_state.total_packets);
        
        /* Could add data exfiltration detection here */
    }
    
    return NF_ACCEPT;
}

/*
 * Netfilter hook for forwarded packets
 */
static unsigned int archangel_nf_hook_forward(void *priv, struct sk_buff *skb,
                                            const struct nf_hook_state *state)
{
    /* Monitor forwarded traffic */
    return archangel_nf_hook_in(priv, skb, state);
}

/*
 * Get or create network connection tracking
 */
static struct network_connection *get_connection(u32 src_ip, u32 dst_ip, 
                                               u16 src_port, u16 dst_port, u8 protocol)
{
    struct network_connection *conn;
    u32 hash;
    u64 current_time = ktime_get_ns();
    
    /* Create hash from connection tuple */
    hash = jhash_3words(src_ip ^ dst_ip, 
                       (src_port << 16) | dst_port, 
                       protocol, 0) & (ARRAY_SIZE(network_state.connection_hash) - 1);
    
    /* Try to find existing connection */
    spin_lock(&network_state.connection_lock);
    hlist_for_each_entry(conn, &network_state.connection_hash[hash], hash_node) {
        if (conn->src_ip == src_ip && conn->dst_ip == dst_ip &&
            conn->src_port == src_port && conn->dst_port == dst_port &&
            conn->protocol == protocol) {
            atomic_inc(&conn->ref_count);
            spin_unlock(&network_state.connection_lock);
            return conn;
        }
    }
    spin_unlock(&network_state.connection_lock);
    
    /* Create new connection */
    conn = kzalloc(sizeof(*conn), GFP_ATOMIC);
    if (!conn)
        return NULL;
    
    conn->src_ip = src_ip;
    conn->dst_ip = dst_ip;
    conn->src_port = src_port;
    conn->dst_port = dst_port;
    conn->protocol = protocol;
    atomic64_set(&conn->packet_count, 0);
    atomic64_set(&conn->byte_count, 0);
    conn->first_seen = current_time;
    conn->last_seen = current_time;
    conn->threat_type = THREAT_NONE;
    atomic_set(&conn->ref_count, 1);
    
    /* Add to hash table */
    spin_lock(&network_state.connection_lock);
    hlist_add_head(&conn->hash_node, &network_state.connection_hash[hash]);
    atomic_inc(&network_state.tracked_connections);
    spin_unlock(&network_state.connection_lock);
    
    return conn;
}

/*
 * Release connection reference
 */
static void put_connection(struct network_connection *conn)
{
    if (!conn)
        return;
    
    if (atomic_dec_and_test(&conn->ref_count)) {
        kfree(conn);
        atomic_dec(&network_state.tracked_connections);
    }
}

/*
 * Get or create host tracking entry
 */
static struct network_host *get_host(u32 ip_address)
{
    struct network_host *host;
    u32 hash = hash_32(ip_address, 8);
    u64 current_time = ktime_get_ns();
    
    /* Try to find existing host */
    spin_lock(&network_state.host_lock);
    hlist_for_each_entry(host, &network_state.host_hash[hash], hash_node) {
        if (host->ip_address == ip_address) {
            atomic_inc(&host->ref_count);
            spin_unlock(&network_state.host_lock);
            return host;
        }
    }
    spin_unlock(&network_state.host_lock);
    
    /* Create new host */
    host = kzalloc(sizeof(*host), GFP_ATOMIC);
    if (!host)
        return NULL;
    
    host->ip_address = ip_address;
    host->first_seen = current_time;
    host->last_activity = current_time;
    host->rate_window_start = current_time;
    host->port_scan_window = current_time;
    atomic_set(&host->ref_count, 1);
    
    /* Add to hash table */
    spin_lock(&network_state.host_lock);
    hlist_add_head(&host->hash_node, &network_state.host_hash[hash]);
    atomic_inc(&network_state.tracked_hosts);
    spin_unlock(&network_state.host_lock);
    
    return host;
}

/*
 * Release host reference
 */
static void put_host(struct network_host *host)
{
    if (!host)
        return;
    
    if (atomic_dec_and_test(&host->ref_count)) {
        kfree(host);
        atomic_dec(&network_state.tracked_hosts);
    }
}

/*
 * Analyze packet content for threats
 */
static enum network_threat_type analyze_packet_content(struct sk_buff *skb)
{
    const char *payload;
    size_t payload_len;
    struct iphdr *iph;
    struct tcphdr *tcph;
    struct udphdr *udph;
    
    if (!skb)
        return THREAT_NONE;
    
    iph = ip_hdr(skb);
    if (!iph)
        return THREAT_NONE;
    
    /* Get payload data */
    if (iph->protocol == IPPROTO_TCP) {
        tcph = tcp_hdr(skb);
        if (!tcph)
            return THREAT_NONE;
        
        payload = (char *)tcph + (tcph->doff * 4);
        payload_len = skb->len - (payload - (char *)iph);
    } else if (iph->protocol == IPPROTO_UDP) {
        udph = udp_hdr(skb);
        if (!udph)
            return THREAT_NONE;
        
        payload = (char *)udph + sizeof(struct udphdr);
        payload_len = ntohs(udph->len) - sizeof(struct udphdr);
    } else {
        return THREAT_NONE;
    }
    
    /* Limit analysis size */
    if (payload_len > MAX_PAYLOAD_ANALYSIS)
        payload_len = MAX_PAYLOAD_ANALYSIS;
    
    if (payload_len < SUSPICIOUS_PAYLOAD_BYTES)
        return THREAT_NONE;
    
    /* Check for malicious content */
    if (contains_malicious_payload(payload, payload_len)) {
        return THREAT_MALICIOUS_PAYLOAD;
    }
    
    return THREAT_NONE;
}

/*
 * Detect port scanning behavior
 */
static bool detect_port_scan(struct network_host *host, u16 port)
{
    u64 current_time = ktime_get_ns();
    int i;
    bool port_exists = false;
    
    /* Reset window if it's been too long */
    if (current_time - host->port_scan_window > 60000000000ULL) { /* 1 minute */
        host->port_count = 0;
        host->port_scan_window = current_time;
    }
    
    /* Check if port already seen */
    for (i = 0; i < host->port_count; i++) {
        if (host->accessed_ports[i] == port) {
            port_exists = true;
            break;
        }
    }
    
    /* Add new port */
    if (!port_exists && host->port_count < ARRAY_SIZE(host->accessed_ports)) {
        host->accessed_ports[host->port_count++] = port;
    }
    
    return host->port_count > PORT_SCAN_THRESHOLD;
}

/*
 * Detect DDoS attacks
 */
static bool detect_ddos(struct network_host *host)
{
    return host->packets_per_sec > DDOS_THRESHOLD_PPS;
}

/*
 * Check for malicious payload signatures
 */
static bool contains_malicious_payload(const char *data, size_t len)
{
    int i;
    size_t sig_len;
    
    if (!data || len == 0)
        return false;
    
    for (i = 0; i < NUM_MALICIOUS_SIGNATURES; i++) {
        sig_len = strlen(malicious_signatures[i]);
        if (sig_len <= len) {
            if (strnstr(data, malicious_signatures[i], len)) {
                return true;
            }
        }
    }
    
    return false;
}

/*
 * Make network security decision
 */
static int make_network_decision(struct network_connection *conn, 
                               struct sk_buff *skb, enum network_threat_type threat)
{
    struct archangel_security_context *ctx;
    enum archangel_decision decision;
    struct iphdr *iph;
    
    if (!conn || !skb)
        return ARCHANGEL_ALLOW;
    
    iph = ip_hdr(skb);
    if (!iph)
        return ARCHANGEL_ALLOW;
    
    /* Create security context for AI analysis */
    ctx = kmalloc(sizeof(*ctx) + 128, GFP_ATOMIC);
    if (!ctx)
        return ARCHANGEL_ALLOW; /* Fail open */
    
    memset(ctx, 0, sizeof(*ctx) + 128);
    ctx->pid = 0; /* Network context, no specific PID */
    ctx->uid = 0;
    ctx->syscall_nr = -2; /* Network analysis marker */
    ctx->timestamp = ktime_get_ns();
    ctx->flags = conn->flags;
    strcpy(ctx->comm, "network");
    ctx->data_size = 128;
    
    /* Add network-specific data */
    *((u32 *)ctx->data) = conn->src_ip;
    *((u32 *)(ctx->data + 4)) = conn->dst_ip;
    *((u16 *)(ctx->data + 8)) = conn->src_port;
    *((u16 *)(ctx->data + 10)) = conn->dst_port;
    *((u8 *)(ctx->data + 12)) = conn->protocol;
    *((u32 *)(ctx->data + 16)) = (u32)threat;
    *((u64 *)(ctx->data + 20)) = atomic64_read(&conn->packet_count);
    *((u64 *)(ctx->data + 28)) = atomic64_read(&conn->byte_count);
    *((u32 *)(ctx->data + 36)) = conn->threat_score;
    
    /* Get AI decision */
    decision = archangel_make_decision(ctx);
    
    kfree(ctx);
    return decision;
}

/*
 * Get network analysis statistics
 */
void archangel_network_get_stats(u64 *total_packets, u64 *blocked_packets, 
                                u64 *suspicious_packets, u64 *ddos_packets,
                                u64 *malicious_payloads, u32 *tracked_connections,
                                u32 *tracked_hosts)
{
    if (total_packets)
        *total_packets = atomic64_read(&network_state.total_packets);
    if (blocked_packets)
        *blocked_packets = atomic64_read(&network_state.blocked_packets);
    if (suspicious_packets)
        *suspicious_packets = atomic64_read(&network_state.suspicious_packets);
    if (ddos_packets)
        *ddos_packets = atomic64_read(&network_state.ddos_packets);
    if (malicious_payloads)
        *malicious_payloads = atomic64_read(&network_state.malicious_payloads);
    if (tracked_connections)
        *tracked_connections = atomic_read(&network_state.tracked_connections);
    if (tracked_hosts)
        *tracked_hosts = atomic_read(&network_state.tracked_hosts);
}

/*
 * Enable/disable network analysis
 */
void archangel_network_set_enabled(bool enabled)
{
    network_state.enabled = enabled;
    archangel_info("Network analysis %s", enabled ? "enabled" : "disabled");
}

/*
 * Check if network analysis is enabled
 */
bool archangel_network_is_enabled(void)
{
    return network_state.enabled;
}