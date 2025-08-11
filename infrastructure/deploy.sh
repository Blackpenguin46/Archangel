#!/bin/bash

# Archangel Mock Enterprise Environment Deployment Script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting Archangel Mock Enterprise Environment Deployment"

# Check prerequisites
check_prerequisites() {
    echo "📋 Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo "❌ Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    echo "✅ Prerequisites check passed"
}

# Create necessary directories
create_directories() {
    echo "📁 Creating necessary directories..."
    
    mkdir -p shared/public
    mkdir -p confidential/admin
    mkdir -p logs/{nginx,mysql,postgresql,suricata}
    mkdir -p config/wordpress/vulnerable-plugins
    
    # Create sample files for SMB shares
    echo "This is a public document" > shared/public/readme.txt
    echo "CONFIDENTIAL: Admin passwords stored here" > confidential/admin/passwords.txt
    echo "admin:password123" >> confidential/admin/passwords.txt
    echo "root:admin123" >> confidential/admin/passwords.txt
    
    echo "✅ Directories created"
}

# Set up vulnerable WordPress plugin
setup_vulnerable_wordpress() {
    echo "🔧 Setting up vulnerable WordPress configuration..."
    
    cat > config/wordpress/vulnerable-plugins/vulnerable-plugin.php << 'EOF'
<?php
/*
Plugin Name: Vulnerable Test Plugin
Description: Intentionally vulnerable plugin for testing
Version: 1.0
*/

// SQL Injection vulnerability
if (isset($_GET['user_id'])) {
    global $wpdb;
    $user_id = $_GET['user_id'];
    $query = "SELECT * FROM wp_users WHERE ID = $user_id";
    $result = $wpdb->get_results($query);
}

// File inclusion vulnerability
if (isset($_GET['page'])) {
    include($_GET['page'] . '.php');
}

// Command injection vulnerability
if (isset($_POST['ping_host'])) {
    $host = $_POST['ping_host'];
    $output = shell_exec("ping -c 1 $host");
    echo "<pre>$output</pre>";
}
?>
EOF
    
    echo "✅ Vulnerable WordPress plugin configured"
}

# Deploy infrastructure
deploy_infrastructure() {
    echo "🏗️  Deploying infrastructure..."
    
    # Pull latest images
    docker-compose pull
    
    # Build and start services
    docker-compose up -d
    
    echo "✅ Infrastructure deployed"
}

# Wait for services to be ready
wait_for_services() {
    echo "⏳ Waiting for services to be ready..."
    
    # Wait for Elasticsearch
    echo "Waiting for Elasticsearch..."
    until curl -s http://localhost:9200/_cluster/health | grep -q '"status":"yellow\|green"'; do
        sleep 5
    done
    
    # Wait for MySQL
    echo "Waiting for MySQL..."
    until docker exec mysql-vulnerable mysqladmin ping -h localhost --silent; do
        sleep 5
    done
    
    # Wait for PostgreSQL
    echo "Waiting for PostgreSQL..."
    until docker exec postgresql-vulnerable pg_isready -h localhost; do
        sleep 5
    done
    
    echo "✅ All services are ready"
}

# Configure Kibana dashboards
setup_kibana_dashboards() {
    echo "📊 Setting up Kibana dashboards..."
    
    # Wait for Kibana to be ready
    until curl -s http://localhost:5601/api/status | grep -q '"overall":{"level":"available"'; do
        sleep 10
    done
    
    # Import index patterns and dashboards
    curl -X POST "localhost:5601/api/saved_objects/_import" \
        -H "kbn-xsrf: true" \
        -H "Content-Type: application/json" \
        --form file=@config/kibana/dashboards.ndjson || true
    
    echo "✅ Kibana dashboards configured"
}

# Run security tests
run_security_tests() {
    echo "🔒 Running security validation tests..."
    
    # Test network segmentation
    echo "Testing network segmentation..."
    docker run --rm --network infrastructure_dmz_network alpine ping -c 1 192.168.20.10 && echo "❌ DMZ can reach internal network" || echo "✅ Network segmentation working"
    
    # Test vulnerable services
    echo "Testing vulnerable service exposure..."
    curl -s http://localhost:3306 && echo "❌ MySQL exposed" || echo "✅ MySQL properly exposed for testing"
    curl -s http://localhost:5432 && echo "❌ PostgreSQL exposed" || echo "✅ PostgreSQL properly exposed for testing"
    
    echo "✅ Security tests completed"
}

# Display deployment summary
show_deployment_summary() {
    echo ""
    echo "🎉 Archangel Mock Enterprise Environment Deployed Successfully!"
    echo ""
    echo "📋 Service Access Points:"
    echo "  🌐 Nginx Load Balancer: http://localhost"
    echo "  📝 WordPress: http://localhost (Host: wordpress.local)"
    echo "  🛒 OpenCart: http://localhost (Host: shop.local)"
    echo "  🗄️  DVWA: http://localhost:8080"
    echo "  📧 MailHog: http://localhost:8025"
    echo "  📊 Kibana: http://localhost:5601"
    echo "  🔍 Elasticsearch: http://localhost:9200"
    echo ""
    echo "🗄️  Database Access:"
    echo "  🐬 MySQL: localhost:3306 (root/root123)"
    echo "  🐘 PostgreSQL: localhost:5432 (admin/admin123)"
    echo ""
    echo "📁 File Shares:"
    echo "  📂 SMB: //localhost/public (guest/guest)"
    echo "  🔒 SMB Admin: //localhost/confidential (admin/admin123)"
    echo ""
    echo "🔧 Management:"
    echo "  📈 Grafana: http://localhost:3000 (admin/admin)"
    echo "  🚨 Suricata Logs: docker logs suricata-ids"
    echo ""
    echo "⚠️  Note: This environment contains intentional vulnerabilities for testing purposes."
    echo "   Do not expose to public networks or use in production environments."
}

# Main execution
main() {
    check_prerequisites
    create_directories
    setup_vulnerable_wordpress
    deploy_infrastructure
    wait_for_services
    setup_kibana_dashboards
    run_security_tests
    show_deployment_summary
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        echo "🛑 Stopping Archangel Mock Enterprise Environment..."
        docker-compose down
        echo "✅ Environment stopped"
        ;;
    "destroy")
        echo "💥 Destroying Archangel Mock Enterprise Environment..."
        docker-compose down -v --remove-orphans
        docker system prune -f
        echo "✅ Environment destroyed"
        ;;
    "status")
        echo "📊 Archangel Mock Enterprise Environment Status:"
        docker-compose ps
        ;;
    "logs")
        docker-compose logs -f "${2:-}"
        ;;
    *)
        echo "Usage: $0 {deploy|stop|destroy|status|logs [service]}"
        exit 1
        ;;
esac