# Kubernetes Deployment Guide

This guide covers deploying the Archangel Autonomous AI Evolution system on Kubernetes for production environments with high availability, scalability, and resilience.

## Prerequisites

### Cluster Requirements
- **Kubernetes**: 1.24+ (1.26+ recommended)
- **Nodes**: 3+ nodes (5+ recommended for HA)
- **CPU**: 16+ cores total (32+ recommended)
- **RAM**: 64GB+ total (128GB+ recommended)
- **Storage**: 500GB+ persistent storage
- **Network**: CNI plugin (Calico, Flannel, or Weave)

### Required Tools
- `kubectl` 1.24+
- `helm` 3.8+
- `kustomize` 4.5+
- `docker` or `podman`

### Cluster Features
- **Storage Classes**: Dynamic provisioning support
- **Load Balancer**: MetalLB, cloud provider LB, or ingress controller
- **DNS**: CoreDNS or equivalent
- **Monitoring**: Prometheus operator (optional but recommended)

## Quick Start

### 1. Cluster Preparation
```bash
# Verify cluster access
kubectl cluster-info
kubectl get nodes

# Create namespace
kubectl create namespace archangel

# Set default namespace
kubectl config set-context --current --namespace=archangel
```

### 2. Install Dependencies
```bash
# Add Helm repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Redis
helm install redis bitnami/redis \
  --set auth.password=secure_redis_password \
  --set master.persistence.size=10Gi

# Install PostgreSQL
helm install postgresql bitnami/postgresql \
  --set auth.postgresPassword=secure_postgres_password \
  --set primary.persistence.size=50Gi
```

### 3. Deploy Archangel
```bash
# Apply configurations
kubectl apply -k k8s/overlays/production

# Verify deployment
kubectl get pods
kubectl get services
```

## Detailed Configuration

### Namespace and RBAC

```yaml
# k8s/base/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: archangel
  labels:
    name: archangel
    security.istio.io/tlsMode: istio
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: archangel-service-account
  namespace: archangel
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: archangel-cluster-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: archangel-cluster-role-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: archangel-cluster-role
subjects:
- kind: ServiceAccount
  name: archangel-service-account
  namespace: archangel
```

### ConfigMaps and Secrets

```yaml
# k8s/base/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: archangel-config
  namespace: archangel
data:
  LOG_LEVEL: "INFO"
  DEPLOYMENT_MODE: "production"
  REDIS_HOST: "redis-master"
  POSTGRES_HOST: "postgresql"
  CHROMADB_HOST: "chromadb"
  NETWORK_SUBNET: "10.244.0.0/16"
  TLS_ENABLED: "true"
  PROMETHEUS_ENABLED: "true"
---
apiVersion: v1
kind: Secret
metadata:
  name: archangel-secrets
  namespace: archangel
type: Opaque
stringData:
  OPENAI_API_KEY: "your_openai_api_key_here"
  REDIS_PASSWORD: "secure_redis_password"
  POSTGRES_PASSWORD: "secure_postgres_password"
  ENCRYPTION_KEY: "your_32_character_encryption_key_here"
  JWT_SECRET: "your_jwt_secret_key_here"
```

### Core Services Deployment

#### Coordinator Service
```yaml
# k8s/base/coordinator.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: coordinator
  namespace: archangel
  labels:
    app: coordinator
    component: orchestration
spec:
  replicas: 2
  selector:
    matchLabels:
      app: coordinator
  template:
    metadata:
      labels:
        app: coordinator
        component: orchestration
    spec:
      serviceAccountName: archangel-service-account
      containers:
      - name: coordinator
        image: archangel/coordinator:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 8001
          name: grpc
        env:
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: archangel-config
              key: LOG_LEVEL
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: archangel-secrets
              key: OPENAI_API_KEY
        - name: REDIS_URL
          value: "redis://$(REDIS_PASSWORD)@redis-master:6379"
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: archangel-secrets
              key: REDIS_PASSWORD
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: config-volume
        configMap:
          name: archangel-config
      - name: logs-volume
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: coordinator
  namespace: archangel
  labels:
    app: coordinator
spec:
  selector:
    app: coordinator
  ports:
  - name: http
    port: 8000
    targetPort: 8000
  - name: grpc
    port: 8001
    targetPort: 8001
  type: ClusterIP
```

#### Red Team Agents
```yaml
# k8s/base/red-team.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: red-team-agents
  namespace: archangel
  labels:
    app: red-team
    team: red
spec:
  replicas: 4
  selector:
    matchLabels:
      app: red-team
  template:
    metadata:
      labels:
        app: red-team
        team: red
    spec:
      serviceAccountName: archangel-service-account
      containers:
      - name: red-team-agent
        image: archangel/red-team:latest
        env:
        - name: AGENT_TYPE
          value: "auto"  # Auto-assign agent type
        - name: TEAM
          value: "RED_TEAM"
        - name: COORDINATOR_URL
          value: "http://coordinator:8000"
        - name: REDIS_URL
          value: "redis://$(REDIS_PASSWORD)@redis-master:6379"
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: archangel-secrets
              key: REDIS_PASSWORD
        resources:
          requests:
            cpu: 250m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 2Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Blue Team Agents
```yaml
# k8s/base/blue-team.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: blue-team-agents
  namespace: archangel
  labels:
    app: blue-team
    team: blue
spec:
  replicas: 4
  selector:
    matchLabels:
      app: blue-team
  template:
    metadata:
      labels:
        app: blue-team
        team: blue
    spec:
      serviceAccountName: archangel-service-account
      containers:
      - name: blue-team-agent
        image: archangel/blue-team:latest
        env:
        - name: AGENT_TYPE
          value: "auto"  # Auto-assign agent type
        - name: TEAM
          value: "BLUE_TEAM"
        - name: COORDINATOR_URL
          value: "http://coordinator:8000"
        - name: REDIS_URL
          value: "redis://$(REDIS_PASSWORD)@redis-master:6379"
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: archangel-secrets
              key: REDIS_PASSWORD
        resources:
          requests:
            cpu: 250m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 2Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Mock Enterprise Environment

```yaml
# k8s/base/mock-enterprise.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-server
  namespace: archangel
  labels:
    app: web-server
    tier: dmz
spec:
  replicas: 2
  selector:
    matchLabels:
      app: web-server
  template:
    metadata:
      labels:
        app: web-server
        tier: dmz
    spec:
      containers:
      - name: web-server
        image: archangel/vulnerable-web:latest
        ports:
        - containerPort: 80
        - containerPort: 443
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
          limits:
            cpu: 500m
            memory: 1Gi
        volumeMounts:
        - name: web-content
          mountPath: /var/www/html
      volumes:
      - name: web-content
        persistentVolumeClaim:
          claimName: web-content-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: web-server
  namespace: archangel
spec:
  selector:
    app: web-server
  ports:
  - name: http
    port: 80
    targetPort: 80
  - name: https
    port: 443
    targetPort: 443
  type: LoadBalancer
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: web-content-pvc
  namespace: archangel
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: fast-ssd
```

### Vector Database (ChromaDB)

```yaml
# k8s/base/chromadb.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: chromadb
  namespace: archangel
spec:
  serviceName: chromadb
  replicas: 1
  selector:
    matchLabels:
      app: chromadb
  template:
    metadata:
      labels:
        app: chromadb
    spec:
      containers:
      - name: chromadb
        image: chromadb/chroma:latest
        ports:
        - containerPort: 8000
        env:
        - name: CHROMA_SERVER_HOST
          value: "0.0.0.0"
        - name: CHROMA_SERVER_HTTP_PORT
          value: "8000"
        resources:
          requests:
            cpu: 500m
            memory: 2Gi
          limits:
            cpu: 2000m
            memory: 8Gi
        volumeMounts:
        - name: chromadb-data
          mountPath: /chroma/chroma
        livenessProbe:
          httpGet:
            path: /api/v1/heartbeat
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
  volumeClaimTemplates:
  - metadata:
      name: chromadb-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
      storageClassName: fast-ssd
---
apiVersion: v1
kind: Service
metadata:
  name: chromadb
  namespace: archangel
spec:
  selector:
    app: chromadb
  ports:
  - port: 8000
    targetPort: 8000
  clusterIP: None
```

## Kustomization Structure

### Base Configuration
```yaml
# k8s/base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
- namespace.yaml
- configmap.yaml
- secrets.yaml
- coordinator.yaml
- red-team.yaml
- blue-team.yaml
- mock-enterprise.yaml
- chromadb.yaml

commonLabels:
  app.kubernetes.io/name: archangel
  app.kubernetes.io/version: "1.0.0"
  app.kubernetes.io/component: autonomous-ai-evolution

images:
- name: archangel/coordinator
  newTag: latest
- name: archangel/red-team
  newTag: latest
- name: archangel/blue-team
  newTag: latest
```

### Production Overlay
```yaml
# k8s/overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

bases:
- ../../base

patchesStrategicMerge:
- coordinator-production.yaml
- agents-production.yaml

configMapGenerator:
- name: archangel-config
  behavior: merge
  literals:
  - DEPLOYMENT_MODE=production
  - LOG_LEVEL=INFO
  - PROMETHEUS_ENABLED=true
  - GRAFANA_ENABLED=true

replicas:
- name: coordinator
  count: 3
- name: red-team-agents
  count: 6
- name: blue-team-agents
  count: 6

images:
- name: archangel/coordinator
  newTag: v1.0.0
- name: archangel/red-team
  newTag: v1.0.0
- name: archangel/blue-team
  newTag: v1.0.0
```

## Monitoring and Observability

### Prometheus ServiceMonitor
```yaml
# k8s/base/monitoring.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: archangel-metrics
  namespace: archangel
  labels:
    app: archangel
spec:
  selector:
    matchLabels:
      app: coordinator
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
---
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: archangel-alerts
  namespace: archangel
spec:
  groups:
  - name: archangel.rules
    rules:
    - alert: ArchangelCoordinatorDown
      expr: up{job="coordinator"} == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "Archangel coordinator is down"
        description: "Coordinator has been down for more than 1 minute"
    
    - alert: HighAgentFailureRate
      expr: rate(agent_failures_total[5m]) > 0.1
      for: 2m
      labels:
        severity: warning
      annotations:
        summary: "High agent failure rate detected"
        description: "Agent failure rate is {{ $value }} failures per second"
```

### Grafana Dashboard ConfigMap
```yaml
# k8s/base/grafana-dashboard.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: archangel-dashboard
  namespace: archangel
  labels:
    grafana_dashboard: "1"
data:
  archangel-overview.json: |
    {
      "dashboard": {
        "title": "Archangel System Overview",
        "panels": [
          {
            "title": "Active Agents",
            "type": "stat",
            "targets": [
              {
                "expr": "sum(up{job=~\".*-team.*\"})",
                "legendFormat": "Active Agents"
              }
            ]
          },
          {
            "title": "Agent Performance",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(agent_actions_total[5m])",
                "legendFormat": "Actions per second"
              }
            ]
          }
        ]
      }
    }
```

## Deployment Procedures

### Initial Deployment

```bash
#!/bin/bash
# scripts/deploy-k8s.sh

set -e

echo "🚀 Deploying Archangel to Kubernetes..."

# Verify prerequisites
echo "Checking prerequisites..."
kubectl version --client
helm version

# Create namespace
echo "Creating namespace..."
kubectl create namespace archangel --dry-run=client -o yaml | kubectl apply -f -

# Install dependencies
echo "Installing dependencies..."
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Install Redis
helm upgrade --install redis bitnami/redis \
  --namespace archangel \
  --set auth.password=secure_redis_password \
  --set master.persistence.size=10Gi \
  --set replica.persistence.size=10Gi \
  --wait

# Install PostgreSQL
helm upgrade --install postgresql bitnami/postgresql \
  --namespace archangel \
  --set auth.postgresPassword=secure_postgres_password \
  --set primary.persistence.size=50Gi \
  --wait

# Deploy Archangel components
echo "Deploying Archangel components..."
kubectl apply -k k8s/overlays/production

# Wait for deployment
echo "Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/coordinator -n archangel
kubectl wait --for=condition=available --timeout=300s deployment/red-team-agents -n archangel
kubectl wait --for=condition=available --timeout=300s deployment/blue-team-agents -n archangel

echo "✅ Deployment complete!"
echo "Access Grafana: kubectl port-forward svc/grafana 3000:3000 -n archangel"
echo "Access API: kubectl port-forward svc/coordinator 8000:8000 -n archangel"
```

### Rolling Updates

```bash
# Update image tags
kubectl set image deployment/coordinator coordinator=archangel/coordinator:v1.1.0 -n archangel

# Monitor rollout
kubectl rollout status deployment/coordinator -n archangel

# Rollback if needed
kubectl rollout undo deployment/coordinator -n archangel
```

### Scaling Operations

```bash
# Scale agents based on load
kubectl scale deployment red-team-agents --replicas=8 -n archangel
kubectl scale deployment blue-team-agents --replicas=8 -n archangel

# Auto-scaling with HPA
kubectl autoscale deployment coordinator --cpu-percent=70 --min=2 --max=10 -n archangel
```

## High Availability Configuration

### Multi-Zone Deployment
```yaml
# k8s/overlays/production/coordinator-ha.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: coordinator
spec:
  replicas: 3
  template:
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - coordinator
            topologyKey: kubernetes.io/hostname
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            preference:
              matchExpressions:
              - key: topology.kubernetes.io/zone
                operator: In
                values:
                - zone-a
                - zone-b
                - zone-c
```

### Database High Availability
```bash
# Install PostgreSQL with HA
helm upgrade --install postgresql bitnami/postgresql-ha \
  --namespace archangel \
  --set postgresql.replicaCount=3 \
  --set pgpool.replicaCount=2 \
  --set persistence.size=100Gi
```

## Security Configuration

### Network Policies
```yaml
# k8s/base/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: archangel-network-policy
  namespace: archangel
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: archangel
    - namespaceSelector:
        matchLabels:
          name: monitoring
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: archangel
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS for external APIs
    - protocol: TCP
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
```

### Pod Security Standards
```yaml
# k8s/base/pod-security.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: archangel
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

## Backup and Disaster Recovery

### Backup Strategy
```bash
#!/bin/bash
# scripts/backup-k8s.sh

# Backup PostgreSQL
kubectl exec -n archangel postgresql-0 -- pg_dump -U postgres archangel > backup-$(date +%Y%m%d).sql

# Backup ChromaDB
kubectl exec -n archangel chromadb-0 -- tar czf - /chroma/chroma | cat > chromadb-backup-$(date +%Y%m%d).tar.gz

# Backup Kubernetes manifests
kubectl get all -n archangel -o yaml > k8s-backup-$(date +%Y%m%d).yaml
```

### Disaster Recovery
```bash
#!/bin/bash
# scripts/restore-k8s.sh

# Restore database
kubectl exec -i -n archangel postgresql-0 -- psql -U postgres archangel < backup-20240101.sql

# Restore ChromaDB
kubectl exec -i -n archangel chromadb-0 -- tar xzf - -C / < chromadb-backup-20240101.tar.gz

# Restart services
kubectl rollout restart deployment/coordinator -n archangel
```

## Troubleshooting

### Common Issues

#### Pod Startup Issues
```bash
# Check pod status
kubectl get pods -n archangel

# Describe problematic pod
kubectl describe pod <pod-name> -n archangel

# Check logs
kubectl logs <pod-name> -n archangel --previous
```

#### Resource Issues
```bash
# Check resource usage
kubectl top nodes
kubectl top pods -n archangel

# Check resource quotas
kubectl describe resourcequota -n archangel
```

#### Network Issues
```bash
# Test service connectivity
kubectl exec -n archangel <pod-name> -- nslookup coordinator

# Check network policies
kubectl get networkpolicy -n archangel
```

### Debug Commands
```bash
# Port forward for debugging
kubectl port-forward svc/coordinator 8000:8000 -n archangel

# Execute commands in pods
kubectl exec -it <pod-name> -n archangel -- /bin/bash

# Check events
kubectl get events -n archangel --sort-by='.lastTimestamp'
```

## Performance Optimization

### Resource Tuning
- Set appropriate CPU and memory limits
- Use node affinity for optimal placement
- Configure horizontal pod autoscaling
- Implement vertical pod autoscaling for long-running workloads

### Storage Optimization
- Use fast SSD storage classes for databases
- Configure appropriate volume sizes
- Implement backup and retention policies
- Monitor storage usage and growth

### Network Optimization
- Use service mesh for advanced traffic management
- Configure ingress controllers for external access
- Implement proper load balancing strategies
- Monitor network latency and throughput

---

*Next: [Cloud Deployment Guide](cloud.md) for cloud-specific configurations*