#ifndef _ARCHANGEL_NETWORK_AI_H
#define _ARCHANGEL_NETWORK_AI_H

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/netfilter.h>
#include <linux/netfilter_ipv4.h>
#include <linux/netfilter_ipv6.h>
#include <linux/ip.h>
#include <linux/ipv6.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <linux/icmp.h>
#include <linux/skbuff.h>
#include <linux/spinlock.h>
#include <linux/atomic.h>
#include <linux/slab.h>
#include <linux/hash.h>
#include <linux/jhash.h>
#include <linux/random.h>
#include <linux/cpufeature.h>
#include <asm/fpu/api.h>
#include "archangel_core.h"

/* Network AI configuration */
#define ARCHANGEL_NET_AI_VERSION "1.0.0"
#define ARCHANGEL_NET_AI_MAX_FLOWS 65536
#define ARCHANGEL_NET_AI_FLOW_TIMEOUT_MS 30000
#define ARCHANGEL_NET_AI_FEATURE_SIZE 64
#define ARCHANGEL_NET_AI_PATTERN_CACHE_SIZE 1024
#define ARCHANGEL_NET_AI_STEALTH_SIGNATURES 256

/* Performance targets */
#define ARCHANGEL_NET_AI_MAX_LATENCY_NS 1000000  /* 1ms max */
#define ARCHANGEL_NET_AI_TARGET_THROUGHPUT 1000000  /* 1M packets/sec */

/* Network AI decision types */
enum archangel_net_decision {
    ARCHANGEL_NET_ALLOW = 0,
    ARCHANGEL_NET_DROP,
    ARCHANGEL_NET_MODIFY,
    ARCHANGEL_NET_DEFER,
    ARCHANGEL_NET_LOG
};

/* Packet classification types */
enum archangel_packet_class {
    ARCHANGEL_PKT_NORMAL = 0,
    ARCHANGEL_PKT_SUSPICIOUS,
    ARCHANGEL_PKT_MALICIOUS,
    ARCHANGEL_PKT_STEALTH_TARGET,
    ARCHANGEL_PKT_EXPLOIT_ATTEMPT
};

/* Stealth operation modes */
enum archangel_stealth_mode {
    ARCHANGEL_STEALTH_OFF = 0,
    ARCHANGEL_STEALTH_PASSIVE,
    ARCHANGEL_STEALTH_ACTIVE,
    ARCHANGEL_STEALTH_AGGRESSIVE
};

/* Hardware acceleration capabilities */
struct archangel_hw_caps {
    bool avx2_available;
    bool vnni_available;
    bool sse4_available;
    bool aes_ni_available;
    u8 simd_width;
    u32 cache_line_size;
};

/* Network flow tracking */
struct archangel_flow_key {
    __be32 src_ip;
    __be32 dst_ip;
    __be16 src_port;
    __be16 dst_port;
    u8 protocol;
    u8 padding[3];
} __packed;

struct archangel_flow_stats {
    u64 packets;
    u64 bytes;
    u64 first_seen;
    u64 last_seen;
    u32 flags;
    enum archangel_packet_class classification;
    u16 anomaly_score;
    u16 risk_level;
};

struct archangel_flow_entry {
    struct archangel_flow_key key;
    struct archangel_flow_stats stats;
    struct hlist_node hash_node;
    struct rcu_head rcu;
    spinlock_t lock;
};

/* Packet feature extraction */
struct archangel_packet_features {
    /* Basic packet info */
    u16 packet_size;
    u8 protocol;
    u8 ttl;
    u16 flags;
    u16 window_size;
    
    /* Timing features */
    u32 inter_arrival_time;
    u32 flow_duration;
    
    /* Statistical features */
    u16 payload_entropy;
    u16 header_anomaly_score;
    
    /* Pattern matching results */
    u32 signature_matches;
    u16 behavioral_score;
    
    /* Hardware-accelerated features */
    u8 simd_features[32];
    
    /* Reserved for future use */
    u8 reserved[8];
} __packed;

/* ML classifier structure */
struct archangel_ml_classifier {
    /* Decision tree nodes */
    struct {
        u16 feature_index;
        u16 threshold;
        u16 left_child;
        u16 right_child;
        u8 is_leaf;
        u8 class_id;
    } *decision_tree;
    
    u16 tree_size;
    u16 max_depth;
    
    /* Feature scaling parameters */
    u16 feature_min[ARCHANGEL_NET_AI_FEATURE_SIZE];
    u16 feature_max[ARCHANGEL_NET_AI_FEATURE_SIZE];
    
    /* Performance counters */
    atomic64_t classifications;
    atomic64_t cache_hits;
    u64 avg_inference_time_ns;
    
    spinlock_t lock;
};

/* Anomaly detection engine */
struct archangel_anomaly_detector {
    /* Streaming statistics for each feature */
    struct {
        u64 sum;
        u64 sum_squares;
        u32 count;
        u16 mean;
        u16 variance;
    } feature_stats[ARCHANGEL_NET_AI_FEATURE_SIZE];
    
    /* Anomaly thresholds */
    u16 anomaly_threshold;
    u16 alert_threshold;
    
    /* Detection counters */
    atomic64_t anomalies_detected;
    atomic64_t false_positives;
    
    spinlock_t lock;
};

/* Stealth signature modification */
struct archangel_stealth_signature {
    u32 original_hash;
    u32 modified_hash;
    u16 offset;
    u16 length;
    u8 original_bytes[16];
    u8 modified_bytes[16];
    u32 usage_count;
    u64 last_used;
};

struct archangel_stealth_engine {
    struct archangel_stealth_signature signatures[ARCHANGEL_NET_AI_STEALTH_SIGNATURES];
    enum archangel_stealth_mode mode;
    u32 signature_count;
    u32 modification_seed;
    atomic64_t packets_modified;
    atomic64_t stealth_hits;
    spinlock_t lock;
};

/* Main network AI engine */
struct archangel_network_ai_engine {
    /* Base AI engine */
    struct archangel_ai_engine base;
    
    /* ML components */
    struct archangel_ml_classifier classifier;
    struct archangel_anomaly_detector anomaly_detector;
    struct archangel_stealth_engine stealth_engine;
    
    /* Flow tracking */
    struct hlist_head flow_table[ARCHANGEL_NET_AI_MAX_FLOWS];
    atomic_t flow_count;
    spinlock_t flow_lock;
    
    /* Hardware capabilities */
    struct archangel_hw_caps hw_caps;
    
    /* Pattern cache for fast lookups */
    struct {
        u32 hash;
        enum archangel_net_decision decision;
        u64 timestamp;
    } pattern_cache[ARCHANGEL_NET_AI_PATTERN_CACHE_SIZE];
    
    u32 cache_index;
    
    /* Performance statistics */
    atomic64_t packets_processed;
    atomic64_t packets_dropped;
    atomic64_t packets_modified;
    atomic64_t cache_hits;
    atomic64_t hw_accelerated;
    
    /* Configuration */
    bool enabled;
    bool stealth_mode_active;
    u8 sensitivity_level;
    u8 performance_mode;
    
    spinlock_t lock;
};

/* Global network AI instance */
extern struct archangel_network_ai_engine *archangel_net_ai;

/* Netfilter hooks */
extern struct nf_hook_ops archangel_nf_hooks[];

/* Core functions */
int archangel_network_ai_init(void);
void archangel_network_ai_cleanup(void);
int archangel_network_ai_enable(void);
void archangel_network_ai_disable(void);

/* Netfilter hook functions */
unsigned int archangel_ai_netfilter_hook_ipv4(void *priv,
                                             struct sk_buff *skb,
                                             const struct nf_hook_state *state);
unsigned int archangel_ai_netfilter_hook_ipv6(void *priv,
                                             struct sk_buff *skb,
                                             const struct nf_hook_state *state);

/* Packet processing */
enum archangel_net_decision archangel_ai_classify_packet(struct sk_buff *skb,
                                                       struct archangel_packet_features *features);
int archangel_ai_extract_features(struct sk_buff *skb,
                                struct archangel_packet_features *features);
int archangel_ai_modify_packet_stealth(struct sk_buff *skb,
                                     const struct archangel_packet_features *features);

/* Flow management */
struct archangel_flow_entry *archangel_flow_lookup(const struct archangel_flow_key *key);
struct archangel_flow_entry *archangel_flow_create(const struct archangel_flow_key *key);
void archangel_flow_update(struct archangel_flow_entry *flow,
                         const struct archangel_packet_features *features);
void archangel_flow_cleanup_expired(void);

/* ML classifier functions */
int archangel_ml_classifier_init(struct archangel_ml_classifier *classifier);
void archangel_ml_classifier_cleanup(struct archangel_ml_classifier *classifier);
enum archangel_packet_class archangel_ml_classify(struct archangel_ml_classifier *classifier,
                                                const struct archangel_packet_features *features);

/* Anomaly detection functions */
int archangel_anomaly_detector_init(struct archangel_anomaly_detector *detector);
void archangel_anomaly_detector_cleanup(struct archangel_anomaly_detector *detector);
u16 archangel_anomaly_detect(struct archangel_anomaly_detector *detector,
                           const struct archangel_packet_features *features);
void archangel_anomaly_update_stats(struct archangel_anomaly_detector *detector,
                                  const struct archangel_packet_features *features);

/* Stealth engine functions */
int archangel_stealth_engine_init(struct archangel_stealth_engine *engine);
void archangel_stealth_engine_cleanup(struct archangel_stealth_engine *engine);
int archangel_stealth_modify_packet(struct archangel_stealth_engine *engine,
                                  struct sk_buff *skb,
                                  const struct archangel_packet_features *features);
void archangel_stealth_set_mode(enum archangel_stealth_mode mode);

/* Hardware acceleration functions */
void archangel_hw_caps_detect(struct archangel_hw_caps *caps);
int archangel_simd_extract_features(const struct sk_buff *skb,
                                  struct archangel_packet_features *features);
bool archangel_hw_accelerated_classify(const struct archangel_packet_features *features,
                                     enum archangel_packet_class *result);

/* Utility functions */
static inline u32 archangel_flow_hash(const struct archangel_flow_key *key)
{
    return jhash(key, sizeof(*key), 0) & (ARCHANGEL_NET_AI_MAX_FLOWS - 1);
}

static inline bool archangel_is_stealth_active(void)
{
    return archangel_net_ai && archangel_net_ai->stealth_mode_active;
}

static inline void archangel_net_ai_inc_stat(atomic64_t *stat)
{
    atomic64_inc(stat);
}

/* Performance monitoring */
void archangel_network_ai_update_performance(u64 processing_time_ns);
void archangel_network_ai_get_stats(struct seq_file *m);

#endif /* _ARCHANGEL_NETWORK_AI_H */