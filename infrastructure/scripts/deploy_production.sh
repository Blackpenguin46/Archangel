#!/bin/bash
set -euo pipefail

# Archangel Production Deployment Script
# Comprehensive deployment automation for production Kubernetes environment

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NAMESPACE="${NAMESPACE:-archangel}"
ENVIRONMENT="${ENVIRONMENT:-production}"
KUBECTL_TIMEOUT="${KUBECTL_TIMEOUT:-300s}"
HELM_TIMEOUT="${HELM_TIMEOUT:-600s}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handling
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Deployment failed with exit code $exit_code"
        log_info "Checking deployment status..."
        kubectl get pods -n "$NAMESPACE" --show-labels || true
        kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10 || true
    fi
    exit $exit_code
}

trap cleanup EXIT

# Validation functions
validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    # Check required tools
    local required_tools=("kubectl" "helm" "docker" "jq" "yq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool '$tool' is not installed"
            exit 1
        fi
    done
    
    # Check Kubernetes connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check cluster version
    local k8s_version
    k8s_version=$(kubectl version --output=json | jq -r '.serverVersion.gitVersion')
    log_info "Kubernetes cluster version: $k8s_version"
    
    # Validate cluster resources
    local nodes_ready
    nodes_ready=$(kubectl get nodes --no-headers | grep -c " Ready ")
    if [ "$nodes_ready" -lt 1 ]; then
        log_error "No ready nodes found in cluster"
        exit 1
    fi
    log_info "Found $nodes_ready ready nodes"
    
    log_success "Prerequisites validation completed"
}

validate_configuration() {
    log_info "Validating deployment configuration..."
    
    # Check required environment variables
    local required_vars=("NAMESPACE" "ENVIRONMENT")
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            log_error "Required environment variable '$var' is not set"
            exit 1
        fi
    done
    
    # Validate Kubernetes manifests
    local manifest_files=(
        "$PROJECT_ROOT/infrastructure/k8s/archangel-namespace.yaml"
        "$PROJECT_ROOT/infrastructure/k8s/production-deployment.yaml"
        "$PROJECT_ROOT/infrastructure/k8s/load-balancing-services.yaml"
        "$PROJECT_ROOT/infrastructure/k8s/persistent-storage-backup.yaml"
        "$PROJECT_ROOT/infrastructure/k8s/production-monitoring.yaml"
    )
    
    for manifest in "${manifest_files[@]}"; do
        if [ ! -f "$manifest" ]; then
            log_error "Required manifest file not found: $manifest"
            exit 1
        fi
        
        # Validate YAML syntax
        if ! yq eval '.' "$manifest" > /dev/null 2>&1; then
            log_error "Invalid YAML syntax in: $manifest"
            exit 1
        fi
    done
    
    log_success "Configuration validation completed"
}

# Deployment functions
create_namespace() {
    log_info "Creating namespace and RBAC configuration..."
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace '$NAMESPACE' already exists"
    else
        kubectl apply -f "$PROJECT_ROOT/infrastructure/k8s/archangel-namespace.yaml"
        log_success "Namespace '$NAMESPACE' created"
    fi
    
    # Wait for namespace to be active
    kubectl wait --for=condition=Active namespace/"$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
}

create_secrets() {
    log_info "Creating production secrets..."
    
    # Create database secrets
    if ! kubectl get secret archangel-secrets -n "$NAMESPACE" &> /dev/null; then
        local db_password
        db_password=$(openssl rand -base64 32)
        
        kubectl create secret generic archangel-secrets \
            --from-literal=database-url="postgresql://admin:${db_password}@postgres-service:5432/archangel" \
            --from-literal=postgres-password="$db_password" \
            -n "$NAMESPACE"
        
        log_success "Database secrets created"
    else
        log_warning "Database secrets already exist"
    fi
    
    # Create backup secrets
    if ! kubectl get secret backup-secrets -n "$NAMESPACE" &> /dev/null; then
        local backup_key
        backup_key=$(openssl rand -base64 32)
        
        kubectl create secret generic backup-secrets \
            --from-literal=encryption-key="$backup_key" \
            -n "$NAMESPACE"
        
        log_success "Backup secrets created"
    else
        log_warning "Backup secrets already exist"
    fi
    
    # Create monitoring secrets
    if ! kubectl get secret alertmanager-secrets -n "$NAMESPACE" &> /dev/null; then
        local grafana_password
        grafana_password=$(openssl rand -base64 16)
        
        kubectl create secret generic alertmanager-secrets \
            --from-literal=smtp-password="changeme" \
            --from-literal=slack-webhook="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK" \
            --from-literal=pagerduty-key="your-pagerduty-integration-key" \
            -n "$NAMESPACE"
        
        kubectl create secret generic grafana-secrets \
            --from-literal=grafana-admin-password="$grafana_password" \
            --from-literal=grafana-secret-key="$(openssl rand -base64 32)" \
            --from-literal=grafana-db-password="$(openssl rand -base64 16)" \
            --from-literal=smtp-password="changeme" \
            -n "$NAMESPACE"
        
        log_success "Monitoring secrets created"
        log_warning "Please update monitoring secrets with actual values:"
        log_warning "  kubectl patch secret alertmanager-secrets -n $NAMESPACE --patch='{\"data\":{\"smtp-password\":\"<base64-encoded-password>\",\"slack-webhook\":\"<base64-encoded-webhook>\"}}'"
    else
        log_warning "Monitoring secrets already exist"
    fi
}

deploy_storage() {
    log_info "Deploying persistent storage and backup systems..."
    
    # Apply storage configurations
    kubectl apply -f "$PROJECT_ROOT/infrastructure/k8s/persistent-storage-backup.yaml"
    
    # Wait for PVCs to be bound
    log_info "Waiting for PVCs to be bound..."
    local pvcs
    pvcs=$(kubectl get pvc -n "$NAMESPACE" -o name 2>/dev/null || echo "")
    
    if [ -n "$pvcs" ]; then
        for pvc in $pvcs; do
            kubectl wait --for=condition=Bound "$pvc" -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT" || {
                log_warning "PVC $pvc did not bind within timeout"
            }
        done
    fi
    
    log_success "Storage deployment completed"
}

deploy_database() {
    log_info "Deploying database services..."
    
    # Deploy PostgreSQL
    kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: $NAMESPACE
  labels:
    app: postgres
    component: database
    environment: $ENVIRONMENT
spec:
  serviceName: postgres-headless
  replicas: 1
  selector:
    matchLabels:
      app: postgres
      component: database
  template:
    metadata:
      labels:
        app: postgres
        component: database
        environment: $ENVIRONMENT
    spec:
      securityContext:
        runAsUser: 70
        runAsGroup: 70
        fsGroup: 70
      containers:
        - name: postgres
          image: postgres:14-alpine
          ports:
            - name: postgres
              containerPort: 5432
          env:
            - name: POSTGRES_DB
              value: archangel
            - name: POSTGRES_USER
              value: admin
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: archangel-secrets
                  key: postgres-password
            - name: PGDATA
              value: /var/lib/postgresql/data/pgdata
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
          livenessProbe:
            exec:
              command:
                - pg_isready
                - -U
                - admin
                - -d
                - archangel
            initialDelaySeconds: 30
            periodSeconds: 30
          readinessProbe:
            exec:
              command:
                - pg_isready
                - -U
                - admin
                - -d
                - archangel
            initialDelaySeconds: 15
            periodSeconds: 10
          volumeMounts:
            - name: postgres-data
              mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
    - metadata:
        name: postgres-data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: archangel-fast-ssd
        resources:
          requests:
            storage: 20Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: $NAMESPACE
  labels:
    app: postgres
    component: database
spec:
  type: ClusterIP
  ports:
    - name: postgres
      port: 5432
      targetPort: 5432
  selector:
    app: postgres
    component: database
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-headless
  namespace: $NAMESPACE
  labels:
    app: postgres
    component: database
spec:
  type: ClusterIP
  clusterIP: None
  ports:
    - name: postgres
      port: 5432
      targetPort: 5432
  selector:
    app: postgres
    component: database
EOF
    
    # Wait for database to be ready
    log_info "Waiting for database to be ready..."
    kubectl wait --for=condition=Ready pod -l app=postgres -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
    
    log_success "Database deployment completed"
}

deploy_core_services() {
    log_info "Deploying core Archangel services..."
    
    # Apply production deployment
    kubectl apply -f "$PROJECT_ROOT/infrastructure/k8s/production-deployment.yaml"
    
    # Wait for deployments to be ready
    log_info "Waiting for core services to be ready..."
    kubectl wait --for=condition=Available deployment/archangel-core-production -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
    kubectl wait --for=condition=Available deployment/archangel-agents-production -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
    
    log_success "Core services deployment completed"
}

deploy_load_balancing() {
    log_info "Deploying load balancing and service discovery..."
    
    # Apply load balancing configuration
    kubectl apply -f "$PROJECT_ROOT/infrastructure/k8s/load-balancing-services.yaml"
    
    # Wait for services to have endpoints
    log_info "Waiting for service endpoints..."
    local services=("archangel-core-service" "archangel-agents-service")
    for service in "${services[@]}"; do
        local retries=0
        local max_retries=30
        
        while [ $retries -lt $max_retries ]; do
            if kubectl get endpoints "$service" -n "$NAMESPACE" -o jsonpath='{.subsets[0].addresses[0].ip}' &> /dev/null; then
                log_success "Service $service has endpoints"
                break
            fi
            
            retries=$((retries + 1))
            sleep 10
        done
        
        if [ $retries -eq $max_retries ]; then
            log_warning "Service $service endpoints not ready within timeout"
        fi
    done
    
    log_success "Load balancing deployment completed"
}

deploy_monitoring() {
    log_info "Deploying monitoring and alerting..."
    
    # Apply monitoring configuration
    kubectl apply -f "$PROJECT_ROOT/infrastructure/k8s/production-monitoring.yaml"
    
    # Wait for monitoring services
    log_info "Waiting for monitoring services..."
    kubectl wait --for=condition=Available deployment/prometheus-production -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT" || {
        log_warning "Prometheus deployment not ready within timeout"
    }
    kubectl wait --for=condition=Available deployment/alertmanager-production -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT" || {
        log_warning "Alertmanager deployment not ready within timeout"
    }
    
    log_success "Monitoring deployment completed"
}

run_health_checks() {
    log_info "Running post-deployment health checks..."
    
    # Check pod status
    local unhealthy_pods
    unhealthy_pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running --no-headers 2>/dev/null | wc -l)
    
    if [ "$unhealthy_pods" -gt 0 ]; then
        log_warning "Found $unhealthy_pods unhealthy pods"
        kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running
    else
        log_success "All pods are running"
    fi
    
    # Check service endpoints
    local services_without_endpoints=0
    local services
    services=$(kubectl get services -n "$NAMESPACE" -o name)
    
    for service in $services; do
        local service_name
        service_name=$(echo "$service" | cut -d'/' -f2)
        
        if ! kubectl get endpoints "$service_name" -n "$NAMESPACE" -o jsonpath='{.subsets[0].addresses[0].ip}' &> /dev/null; then
            log_warning "Service $service_name has no endpoints"
            services_without_endpoints=$((services_without_endpoints + 1))
        fi
    done
    
    if [ "$services_without_endpoints" -eq 0 ]; then
        log_success "All services have endpoints"
    else
        log_warning "$services_without_endpoints services have no endpoints"
    fi
    
    # Test core service health
    log_info "Testing core service health..."
    if kubectl exec -n "$NAMESPACE" deployment/archangel-core-production -- curl -f -s http://localhost:8888/health > /dev/null 2>&1; then
        log_success "Core service health check passed"
    else
        log_warning "Core service health check failed"
    fi
    
    log_success "Health checks completed"
}

run_performance_tests() {
    log_info "Running performance validation tests..."
    
    # Run production deployment tests
    if [ -f "$PROJECT_ROOT/infrastructure/tests/test_production_deployment.py" ]; then
        log_info "Executing production deployment tests..."
        
        # Create test job
        kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: production-tests-$(date +%s)
  namespace: $NAMESPACE
  labels:
    app: archangel
    component: testing
spec:
  template:
    metadata:
      labels:
        app: archangel
        component: testing
    spec:
      restartPolicy: Never
      containers:
        - name: production-tests
          image: python:3.11-slim
          command:
            - /bin/bash
            - -c
            - |
              pip install requests kubernetes psutil pyyaml
              python /tests/test_production_deployment.py
          volumeMounts:
            - name: test-scripts
              mountPath: /tests
              readOnly: true
          resources:
            requests:
              memory: "256Mi"
              cpu: "100m"
            limits:
              memory: "512Mi"
              cpu: "500m"
      volumes:
        - name: test-scripts
          configMap:
            name: production-test-scripts
      serviceAccountName: archangel-service-account
EOF
        
        log_info "Production tests job created. Monitor with: kubectl logs -f job/production-tests-* -n $NAMESPACE"
    else
        log_warning "Production test script not found, skipping performance tests"
    fi
}

generate_deployment_report() {
    log_info "Generating deployment report..."
    
    local report_file="deployment-report-$(date +%Y%m%d-%H%M%S).json"
    
    # Collect deployment information
    local deployment_info
    deployment_info=$(cat <<EOF
{
  "deployment_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "namespace": "$NAMESPACE",
  "environment": "$ENVIRONMENT",
  "cluster_info": $(kubectl cluster-info dump --output-directory=/tmp/cluster-info --quiet 2>/dev/null && echo '{"status": "collected"}' || echo '{"status": "failed"}'),
  "deployments": $(kubectl get deployments -n "$NAMESPACE" -o json | jq '.items[] | {name: .metadata.name, replicas: .spec.replicas, ready_replicas: .status.readyReplicas, conditions: .status.conditions}'),
  "services": $(kubectl get services -n "$NAMESPACE" -o json | jq '.items[] | {name: .metadata.name, type: .spec.type, ports: .spec.ports}'),
  "pods": $(kubectl get pods -n "$NAMESPACE" -o json | jq '.items[] | {name: .metadata.name, phase: .status.phase, ready: .status.conditions[]? | select(.type=="Ready") | .status}'),
  "persistent_volumes": $(kubectl get pvc -n "$NAMESPACE" -o json | jq '.items[] | {name: .metadata.name, phase: .status.phase, capacity: .status.capacity}')
}
EOF
    )
    
    echo "$deployment_info" | jq '.' > "$report_file"
    log_success "Deployment report saved to: $report_file"
}

# Main deployment workflow
main() {
    log_info "Starting Archangel production deployment..."
    log_info "Namespace: $NAMESPACE"
    log_info "Environment: $ENVIRONMENT"
    
    # Pre-deployment validation
    validate_prerequisites
    validate_configuration
    
    # Core deployment steps
    create_namespace
    create_secrets
    deploy_storage
    deploy_database
    deploy_core_services
    deploy_load_balancing
    deploy_monitoring
    
    # Post-deployment validation
    run_health_checks
    run_performance_tests
    generate_deployment_report
    
    log_success "Archangel production deployment completed successfully!"
    log_info "Access the application at: https://archangel.production"
    log_info "Monitor the deployment with: kubectl get pods -n $NAMESPACE -w"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi