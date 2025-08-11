#!/bin/bash

# Deploy Honeypot Infrastructure Script
# This script deploys the complete honeypot and deception technology stack

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root for security reasons"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if user is in docker group
    if ! groups $USER | grep -q docker; then
        error "User $USER is not in the docker group. Please add user to docker group and re-login."
        exit 1
    fi
    
    # Check available disk space (need at least 5GB)
    available_space=$(df / | awk 'NR==2 {print $4}')
    required_space=5242880  # 5GB in KB
    
    if [[ $available_space -lt $required_space ]]; then
        error "Insufficient disk space. Need at least 5GB available."
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    directories=(
        "logs/honeypots"
        "data/honeypots"
        "config/honeypots/threat_intel"
        "shared/honeytokens"
        "confidential/honeytokens"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        log "Created directory: $dir"
    done
    
    success "Directories created successfully"
}

# Generate threat intelligence data
generate_threat_intel() {
    log "Generating threat intelligence data..."
    
    # Create sample malicious IPs (these would normally come from threat feeds)
    cat > config/honeypots/threat_intel/malicious_ips.txt << EOF
# Sample malicious IPs for testing
# In production, these would come from threat intelligence feeds
192.168.1.100
10.0.0.100
172.16.0.100
# Add more IPs as needed
EOF

    # Create sample scanner IPs
    cat > config/honeypots/threat_intel/scanners.txt << EOF
# Known scanner IPs
# These are typically automated scanning tools
192.168.1.200
10.0.0.200
172.16.0.200
EOF

    # Create sample Tor exit nodes
    cat > config/honeypots/threat_intel/tor_exits.txt << EOF
# Sample Tor exit nodes
# In production, these would be updated regularly
192.168.1.300
10.0.0.300
172.16.0.300
EOF

    success "Threat intelligence data generated"
}

# Build custom Docker images
build_images() {
    log "Building custom Docker images..."
    
    # Build decoy services image
    log "Building decoy services image..."
    docker build -t archangel/decoy-services:latest -f config/honeypots/Dockerfile.decoy config/honeypots/
    
    # Build honeytoken distributor image
    log "Building honeytoken distributor image..."
    docker build -t archangel/honeytoken-distributor:latest -f config/honeypots/Dockerfile.honeytokens config/honeypots/
    
    # Build honeypot monitor image
    log "Building honeypot monitor image..."
    docker build -t archangel/honeypot-monitor:latest -f config/honeypots/Dockerfile.monitor config/honeypots/
    
    success "Custom Docker images built successfully"
}

# Update docker-compose.yml with custom images
update_compose_config() {
    log "Updating Docker Compose configuration..."
    
    # Update the docker-compose.yml to use our custom images
    sed -i 's|build:|image: archangel/decoy-services:latest\n    #build:|g' docker-compose.yml
    sed -i 's|dockerfile: Dockerfile.decoy|#dockerfile: Dockerfile.decoy|g' docker-compose.yml
    
    success "Docker Compose configuration updated"
}

# Deploy honeypot infrastructure
deploy_honeypots() {
    log "Deploying honeypot infrastructure..."
    
    # Pull required images
    log "Pulling required Docker images..."
    docker-compose pull cowrie-ssh dionaea-malware glastopf-web
    
    # Start honeypot services
    log "Starting honeypot services..."
    docker-compose up -d cowrie-ssh dionaea-malware glastopf-web decoy-services honeytoken-distributor honeypot-monitor
    
    # Wait for services to start
    log "Waiting for services to start..."
    sleep 30
    
    success "Honeypot infrastructure deployed"
}

# Verify deployment
verify_deployment() {
    log "Verifying honeypot deployment..."
    
    # Check if containers are running
    services=(
        "cowrie-ssh-honeypot"
        "dionaea-malware-honeypot"
        "glastopf-web-honeypot"
        "decoy-services"
        "honeytoken-distributor"
        "honeypot-monitor"
    )
    
    all_running=true
    
    for service in "${services[@]}"; do
        if docker ps --format "table {{.Names}}" | grep -q "$service"; then
            success "✓ $service is running"
        else
            error "✗ $service is not running"
            all_running=false
        fi
    done
    
    if [[ "$all_running" == true ]]; then
        success "All honeypot services are running successfully"
    else
        error "Some honeypot services failed to start"
        return 1
    fi
    
    # Test network connectivity
    log "Testing network connectivity..."
    
    # Test SSH honeypot
    if timeout 5 bash -c "</dev/tcp/localhost/2222"; then
        success "✓ SSH honeypot (port 2222) is accessible"
    else
        warning "⚠ SSH honeypot (port 2222) is not accessible"
    fi
    
    # Test web honeypot
    if timeout 5 bash -c "</dev/tcp/localhost/8080"; then
        success "✓ Web honeypot (port 8080) is accessible"
    else
        warning "⚠ Web honeypot (port 8080) is not accessible"
    fi
    
    # Test fake admin panel
    if timeout 5 bash -c "</dev/tcp/localhost/8081"; then
        success "✓ Fake admin panel (port 8081) is accessible"
    else
        warning "⚠ Fake admin panel (port 8081) is not accessible"
    fi
}

# Generate initial honeytokens
generate_honeytokens() {
    log "Generating initial honeytokens..."
    
    # Execute honeytoken generation inside the container
    docker exec honeytoken-distributor python /opt/honeytokens/honeytokens.py
    
    # Verify honeytokens were created
    if [[ -d "shared/honeytokens" ]] && [[ "$(ls -A shared/honeytokens)" ]]; then
        success "Honeytokens generated successfully"
        log "Honeytokens created in shared/honeytokens/"
        ls -la shared/honeytokens/
    else
        warning "Honeytokens may not have been generated correctly"
    fi
}

# Setup monitoring and alerting
setup_monitoring() {
    log "Setting up honeypot monitoring..."
    
    # Check if monitor is collecting data
    sleep 10  # Wait for some data to be collected
    
    # Check monitor logs
    if docker logs honeypot-monitor 2>&1 | grep -q "Starting honeypot monitoring"; then
        success "Honeypot monitor is running and collecting data"
    else
        warning "Honeypot monitor may not be working correctly"
    fi
    
    # Create monitoring dashboard URL
    log "Monitoring dashboard will be available at:"
    log "  - Kibana: http://localhost:5601"
    log "  - Grafana: http://localhost:3000 (if configured)"
}

# Run tests
run_tests() {
    log "Running honeypot tests..."
    
    # Install test dependencies
    if command -v python3 &> /dev/null; then
        python3 -m pip install --user requests
        
        # Run the honeypot tests
        if python3 infrastructure/tests/test_honeypots.py; then
            success "Honeypot tests passed"
        else
            warning "Some honeypot tests failed - check logs for details"
        fi
    else
        warning "Python3 not available - skipping tests"
    fi
}

# Display deployment summary
show_summary() {
    log "Deployment Summary"
    echo "===================="
    echo
    echo "Honeypot Services Deployed:"
    echo "  • Cowrie SSH Honeypot (port 2222)"
    echo "  • Dionaea Malware Honeypot (ports 21, 80, 135, 445, 1433, 3306, 5060)"
    echo "  • Glastopf Web Honeypot (port 8080)"
    echo "  • Decoy Services (ports 2122, 2223, 8081, 3307)"
    echo "  • Honeytoken Distributor"
    echo "  • Honeypot Monitor & Alerting"
    echo
    echo "Access Points:"
    echo "  • SSH Honeypot: ssh admin@localhost -p 2222"
    echo "  • Web Honeypot: http://localhost:8080"
    echo "  • Fake Admin Panel: http://localhost:8081/admin"
    echo "  • Kibana Dashboard: http://localhost:5601"
    echo
    echo "Log Locations:"
    echo "  • Honeypot Logs: logs/honeypots/"
    echo "  • Container Logs: docker logs <container_name>"
    echo
    echo "Management Commands:"
    echo "  • View status: docker-compose ps"
    echo "  • View logs: docker-compose logs -f <service>"
    echo "  • Stop services: docker-compose down"
    echo "  • Restart services: docker-compose restart"
    echo
    success "Honeypot infrastructure deployment completed successfully!"
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    # Add any cleanup tasks here
}

# Main deployment function
main() {
    log "Starting honeypot infrastructure deployment..."
    
    # Set trap for cleanup on exit
    trap cleanup EXIT
    
    # Run deployment steps
    check_root
    check_prerequisites
    create_directories
    generate_threat_intel
    build_images
    deploy_honeypots
    verify_deployment
    generate_honeytokens
    setup_monitoring
    run_tests
    show_summary
    
    success "Honeypot deployment completed successfully!"
}

# Handle command line arguments
case "${1:-}" in
    "test")
        log "Running honeypot tests only..."
        run_tests
        ;;
    "verify")
        log "Verifying honeypot deployment..."
        verify_deployment
        ;;
    "clean")
        log "Cleaning up honeypot deployment..."
        docker-compose down -v
        docker rmi archangel/decoy-services:latest archangel/honeytoken-distributor:latest archangel/honeypot-monitor:latest 2>/dev/null || true
        success "Cleanup completed"
        ;;
    *)
        main
        ;;
esac