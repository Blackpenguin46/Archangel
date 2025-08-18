#!/bin/bash
"""
Comprehensive Infrastructure Testing Script
Runs all deployment validation and reliability tests
"""

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TESTS_DIR="$PROJECT_ROOT/tests"
REPORTS_DIR="$PROJECT_ROOT/test_reports"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default test configuration
TEST_ENVIRONMENT="${TEST_ENVIRONMENT:-test}"
RELIABILITY_TEST_DURATION="${RELIABILITY_TEST_DURATION:-60}"
LOAD_TEST_USERS="${LOAD_TEST_USERS:-5}"
SKIP_LOAD_TESTS="${SKIP_LOAD_TESTS:-false}"
SKIP_RELIABILITY_TESTS="${SKIP_RELIABILITY_TESTS:-false}"
GENERATE_HTML_REPORT="${GENERATE_HTML_REPORT:-true}"

# Logging
LOG_FILE="$REPORTS_DIR/test_execution_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_info() {
    log "${BLUE}[INFO]${NC} $1"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    log "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE} Archangel Infrastructure Testing Suite${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
    echo -e "Environment: ${GREEN}$TEST_ENVIRONMENT${NC}"
    echo -e "Reliability Duration: ${GREEN}${RELIABILITY_TEST_DURATION}s${NC}"
    echo -e "Load Test Users: ${GREEN}$LOAD_TEST_USERS${NC}"
    echo -e "Reports Directory: ${GREEN}$REPORTS_DIR${NC}"
    echo ""
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    for tool in python3 docker docker-compose; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing_tools+=("$tool")
        fi
    done
    
    # Check optional tools
    for tool in kubectl terraform ansible-playbook; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            log_warning "Optional tool '$tool' not found - some tests may be skipped"
        fi
    done
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        return 1
    fi
    
    # Check Python packages
    python3 -c "
import sys
missing = []
for module in ['docker', 'requests', 'yaml', 'psutil']:
    try:
        __import__(module)
    except ImportError:
        missing.append(module)

if missing:
    print(f'Missing Python packages: {missing}')
    sys.exit(1)
" || {
        log_error "Missing required Python packages. Install with: pip3 install docker requests pyyaml psutil"
        return 1
    }
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running"
        return 1
    fi
    
    log_success "All prerequisites met"
    return 0
}

setup_test_environment() {
    log_info "Setting up test environment..."
    
    # Create reports directory
    mkdir -p "$REPORTS_DIR"
    
    # Set environment variables for tests
    export TEST_ENVIRONMENT="$TEST_ENVIRONMENT"
    export RELIABILITY_TEST_DURATION="$RELIABILITY_TEST_DURATION"
    export LOAD_TEST_USERS="$LOAD_TEST_USERS"
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    
    # Create test configuration file
    cat > "$REPORTS_DIR/test_config.yaml" << EOF
test_environment: $TEST_ENVIRONMENT
services:
  core_api:
    port: 8888
    health_endpoint: "/health"
  prometheus:
    port: 9090
    health_endpoint: "/-/ready"
  grafana:
    port: 3000
    health_endpoint: "/api/health"
  elasticsearch:
    port: 9200
    health_endpoint: "/"
  kibana:
    port: 5601
    health_endpoint: "/"
networks:
  - "dmz"
  - "internal"
  - "management"
  - "deception"
expected_containers: 8
max_startup_time: 300
reliability_test_duration: $RELIABILITY_TEST_DURATION
load_test_users: $LOAD_TEST_USERS
EOF
    
    export TEST_CONFIG="$REPORTS_DIR/test_config.yaml"
    
    log_success "Test environment configured"
}

wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    local services=(
        "http://localhost:8888/health:Core API"
        "http://localhost:9090/-/ready:Prometheus"
        "http://localhost:3000/api/health:Grafana"
        "http://localhost:9200:Elasticsearch"
    )
    
    local max_wait=300
    local start_time=$(date +%s)
    
    for service_info in "${services[@]}"; do
        local url="${service_info%:*}"
        local name="${service_info#*:}"
        local ready=false
        
        log_info "Waiting for $name..."
        
        while [ $(($(date +%s) - start_time)) -lt $max_wait ]; do
            if curl -s -f "$url" >/dev/null 2>&1; then
                log_success "$name is ready"
                ready=true
                break
            fi
            sleep 5
        done
        
        if [ "$ready" = false ]; then
            log_warning "$name not ready after ${max_wait}s - some tests may fail"
        fi
    done
}

run_deployment_validation() {
    log_info "Running deployment validation tests..."
    
    local test_file="$TESTS_DIR/test_deployment_consistency.py"
    local report_file="$REPORTS_DIR/deployment_validation_$(date +%Y%m%d_%H%M%S).xml"
    
    if [ ! -f "$test_file" ]; then
        log_error "Deployment validation test file not found: $test_file"
        return 1
    fi
    
    cd "$PROJECT_ROOT"
    
    if python3 -m pytest "$test_file" \
        --junitxml="$report_file" \
        --verbose \
        --tb=short \
        --capture=no; then
        log_success "Deployment validation tests passed"
        return 0
    else
        log_error "Deployment validation tests failed"
        return 1
    fi
}

run_infrastructure_reliability() {
    if [ "$SKIP_RELIABILITY_TESTS" = "true" ]; then
        log_info "Skipping infrastructure reliability tests (SKIP_RELIABILITY_TESTS=true)"
        return 0
    fi
    
    log_info "Running infrastructure reliability tests..."
    
    local test_file="$TESTS_DIR/test_infrastructure_reliability.py"
    local report_file="$REPORTS_DIR/reliability_tests_$(date +%Y%m%d_%H%M%S).xml"
    
    if [ ! -f "$test_file" ]; then
        log_error "Infrastructure reliability test file not found: $test_file"
        return 1
    fi
    
    cd "$PROJECT_ROOT"
    
    if python3 -m pytest "$test_file" \
        --junitxml="$report_file" \
        --verbose \
        --tb=short \
        --capture=no; then
        log_success "Infrastructure reliability tests passed"
        return 0
    else
        log_warning "Some infrastructure reliability tests failed - check logs for details"
        return 1
    fi
}

run_basic_validation() {
    log_info "Running basic deployment validation..."
    
    local validator_script="$SCRIPT_DIR/validate_deployment.py"
    local report_file="$REPORTS_DIR/basic_validation_$(date +%Y%m%d_%H%M%S).json"
    
    if [ ! -f "$validator_script" ]; then
        log_error "Basic validation script not found: $validator_script"
        return 1
    fi
    
    cd "$PROJECT_ROOT"
    
    if python3 "$validator_script" > "$report_file" 2>&1; then
        log_success "Basic validation passed"
        return 0
    else
        log_error "Basic validation failed - check $report_file for details"
        return 1
    fi
}

run_load_tests() {
    if [ "$SKIP_LOAD_TESTS" = "true" ]; then
        log_info "Skipping load tests (SKIP_LOAD_TESTS=true)"
        return 0
    fi
    
    log_info "Running load tests with $LOAD_TEST_USERS concurrent users..."
    
    # Simple load test using curl and parallel processing
    local target_url="http://localhost:8888/health"
    local requests_per_user=10
    local report_file="$REPORTS_DIR/load_test_$(date +%Y%m%d_%H%M%S).log"
    
    local start_time=$(date +%s.%N)
    local total_requests=$((LOAD_TEST_USERS * requests_per_user))
    local successful_requests=0
    local failed_requests=0
    
    log_info "Sending $total_requests total requests..."
    
    for ((user=1; user<=LOAD_TEST_USERS; user++)); do
        (
            for ((req=1; req<=requests_per_user; req++)); do
                local req_start=$(date +%s.%N)
                if curl -s -f -m 10 "$target_url" >/dev/null 2>&1; then
                    local req_end=$(date +%s.%N)
                    local req_time=$(echo "$req_end - $req_start" | bc -l)
                    echo "User $user Request $req: SUCCESS ${req_time}s" >> "$report_file"
                    echo "SUCCESS"
                else
                    echo "User $user Request $req: FAILED" >> "$report_file"
                    echo "FAILED"
                fi
                sleep 0.1  # Small delay between requests
            done
        ) &
    done
    
    # Wait for all background processes and count results
    wait
    
    if [ -f "$report_file" ]; then
        successful_requests=$(grep -c "SUCCESS" "$report_file" || echo "0")
        failed_requests=$(grep -c "FAILED" "$report_file" || echo "0")
    fi
    
    local end_time=$(date +%s.%N)
    local total_time=$(echo "$end_time - $start_time" | bc -l)
    local success_rate=$(echo "scale=2; $successful_requests * 100 / $total_requests" | bc -l)
    
    log_info "Load test completed in ${total_time}s"
    log_info "Successful requests: $successful_requests/$total_requests (${success_rate}%)"
    
    if (( $(echo "$success_rate >= 95" | bc -l) )); then
        log_success "Load test passed (>= 95% success rate)"
        return 0
    else
        log_warning "Load test showed degraded performance (< 95% success rate)"
        return 1
    fi
}

generate_summary_report() {
    log_info "Generating summary report..."
    
    local summary_file="$REPORTS_DIR/test_summary_$(date +%Y%m%d_%H%M%S).md"
    local html_file="$REPORTS_DIR/test_summary_$(date +%Y%m%d_%H%M%S).html"
    
    # Create markdown summary
    cat > "$summary_file" << EOF
# Archangel Infrastructure Test Summary

**Date:** $(date)
**Environment:** $TEST_ENVIRONMENT
**Test Duration:** ${RELIABILITY_TEST_DURATION}s
**Load Test Users:** $LOAD_TEST_USERS

## Test Results

EOF
    
    # Add results from each test type
    local overall_status="PASS"
    
    # Check deployment validation results
    if ls "$REPORTS_DIR"/deployment_validation_*.xml >/dev/null 2>&1; then
        local latest_deployment=$(ls -t "$REPORTS_DIR"/deployment_validation_*.xml | head -1)
        if grep -q 'failures="0"' "$latest_deployment" 2>/dev/null; then
            echo "### ✅ Deployment Validation: PASSED" >> "$summary_file"
        else
            echo "### ❌ Deployment Validation: FAILED" >> "$summary_file"
            overall_status="FAIL"
        fi
    else
        echo "### ⚠️  Deployment Validation: NOT RUN" >> "$summary_file"
    fi
    
    # Check reliability test results
    if ls "$REPORTS_DIR"/reliability_tests_*.xml >/dev/null 2>&1; then
        local latest_reliability=$(ls -t "$REPORTS_DIR"/reliability_tests_*.xml | head -1)
        if grep -q 'failures="0"' "$latest_reliability" 2>/dev/null; then
            echo "### ✅ Infrastructure Reliability: PASSED" >> "$summary_file"
        else
            echo "### ⚠️  Infrastructure Reliability: PARTIAL" >> "$summary_file"
        fi
    else
        echo "### ⚠️  Infrastructure Reliability: NOT RUN" >> "$summary_file"
    fi
    
    # Check basic validation results
    if ls "$REPORTS_DIR"/basic_validation_*.json >/dev/null 2>&1; then
        echo "### ✅ Basic Validation: COMPLETED" >> "$summary_file"
    else
        echo "### ⚠️  Basic Validation: NOT RUN" >> "$summary_file"
    fi
    
    # Check load test results
    if ls "$REPORTS_DIR"/load_test_*.log >/dev/null 2>&1; then
        local latest_load=$(ls -t "$REPORTS_DIR"/load_test_*.log | head -1)
        local success_count=$(grep -c "SUCCESS" "$latest_load" 2>/dev/null || echo "0")
        local total_requests=$((LOAD_TEST_USERS * 10))
        local success_rate=$(echo "scale=1; $success_count * 100 / $total_requests" | bc -l 2>/dev/null || echo "0")
        echo "### ✅ Load Testing: ${success_rate}% success rate" >> "$summary_file"
    else
        echo "### ⚠️  Load Testing: NOT RUN" >> "$summary_file"
    fi
    
    # Add overall status
    echo "" >> "$summary_file"
    echo "## Overall Status: $overall_status" >> "$summary_file"
    
    # Add file listings
    echo "" >> "$summary_file"
    echo "## Generated Reports" >> "$summary_file"
    echo "" >> "$summary_file"
    for report_file in "$REPORTS_DIR"/*; do
        if [ -f "$report_file" ]; then
            local filename=$(basename "$report_file")
            local filesize=$(du -h "$report_file" | cut -f1)
            echo "- **$filename** ($filesize)" >> "$summary_file"
        fi
    done
    
    log_success "Summary report generated: $summary_file"
    
    # Generate HTML report if requested
    if [ "$GENERATE_HTML_REPORT" = "true" ] && command -v pandoc >/dev/null 2>&1; then
        if pandoc "$summary_file" -o "$html_file" 2>/dev/null; then
            log_success "HTML report generated: $html_file"
        else
            log_warning "Could not generate HTML report (pandoc failed)"
        fi
    elif [ "$GENERATE_HTML_REPORT" = "true" ]; then
        log_warning "HTML report requested but pandoc not available"
    fi
}

cleanup() {
    log_info "Cleaning up test environment..."
    
    # Remove temporary files
    if [ -f "$REPORTS_DIR/test_config.yaml" ]; then
        rm -f "$REPORTS_DIR/test_config.yaml"
    fi
    
    # Compress old reports (older than 7 days)
    find "$REPORTS_DIR" -name "*.xml" -o -name "*.log" -o -name "*.json" | \
        while read -r file; do
            if [ $(find "$file" -mtime +7 -print | wc -l) -gt 0 ]; then
                gzip "$file" 2>/dev/null || true
            fi
        done
    
    log_success "Cleanup completed"
}

main() {
    print_header
    
    # Initialize logging
    mkdir -p "$REPORTS_DIR"
    log_info "Starting infrastructure test suite"
    
    # Trap cleanup on exit
    trap cleanup EXIT
    
    local exit_code=0
    
    # Run test phases
    if ! check_prerequisites; then
        exit_code=1
    elif ! setup_test_environment; then
        exit_code=1
    else
        wait_for_services
        
        # Run basic validation (always)
        if ! run_basic_validation; then
            exit_code=1
        fi
        
        # Run deployment consistency tests
        if ! run_deployment_validation; then
            exit_code=1
        fi
        
        # Run load tests
        if ! run_load_tests; then
            # Load test failure is not critical
            log_warning "Load tests failed but continuing with other tests"
        fi
        
        # Run reliability tests (may take a while)
        if ! run_infrastructure_reliability; then
            # Reliability test failure is not critical
            log_warning "Some reliability tests failed but overall testing continues"
        fi
    fi
    
    # Generate final report
    generate_summary_report
    
    if [ $exit_code -eq 0 ]; then
        log_success "All critical tests passed"
    else
        log_error "Some critical tests failed"
    fi
    
    log_info "Test suite completed with exit code: $exit_code"
    exit $exit_code
}

# Handle command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --environment|-e)
            TEST_ENVIRONMENT="$2"
            shift 2
            ;;
        --duration|-d)
            RELIABILITY_TEST_DURATION="$2"
            shift 2
            ;;
        --users|-u)
            LOAD_TEST_USERS="$2"
            shift 2
            ;;
        --skip-load)
            SKIP_LOAD_TESTS="true"
            shift
            ;;
        --skip-reliability)
            SKIP_RELIABILITY_TESTS="true"
            shift
            ;;
        --no-html)
            GENERATE_HTML_REPORT="false"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -e, --environment ENV    Test environment (default: test)"
            echo "  -d, --duration SECONDS   Reliability test duration (default: 60)"
            echo "  -u, --users COUNT        Load test concurrent users (default: 5)"
            echo "  --skip-load             Skip load testing"
            echo "  --skip-reliability      Skip reliability testing"
            echo "  --no-html               Don't generate HTML reports"
            echo "  -h, --help              Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main