# Scenario Creation Tutorial

This comprehensive tutorial will guide you through creating custom scenarios for the Archangel Autonomous AI Evolution system, from basic concepts to advanced multi-phase scenarios.

## Table of Contents
- [Understanding Scenarios](#understanding-scenarios)
- [Basic Scenario Creation](#basic-scenario-creation)
- [Advanced Scenario Features](#advanced-scenario-features)
- [Testing and Validation](#testing-and-validation)
- [Best Practices](#best-practices)
- [Example Scenarios](#example-scenarios)

## Understanding Scenarios

### What is a Scenario?

A scenario in Archangel defines:
- **Objectives**: What each team is trying to achieve
- **Environment**: The mock enterprise setup and configuration
- **Rules**: Constraints and allowed actions for agents
- **Timeline**: Duration and phase transitions
- **Scoring**: How success is measured

### Scenario Components

```yaml
scenario:
  metadata:
    id: "unique-scenario-id"
    name: "Human-readable scenario name"
    description: "Detailed scenario description"
    version: "1.0.0"
    author: "scenario-creator"
    difficulty: "beginner|intermediate|advanced"
    
  configuration:
    duration: 3600  # Total scenario duration in seconds
    phases: []      # Scenario phases
    environment: {} # Environment configuration
    objectives: {}  # Team objectives
    scoring: {}     # Scoring configuration
    constraints: [] # Global constraints
```

## Basic Scenario Creation

### Step 1: Define Scenario Metadata

Start by creating a new scenario file:

```yaml
# scenarios/my-first-scenario.yml
scenario:
  metadata:
    id: "basic-web-intrusion-001"
    name: "Basic Web Application Intrusion"
    description: |
      A beginner-level scenario where Red Team attempts to compromise
      a vulnerable web application while Blue Team defends it.
    version: "1.0.0"
    author: "tutorial-user"
    difficulty: "beginner"
    tags: ["web", "intrusion", "beginner"]
    estimated_duration: "30 minutes"
```

### Step 2: Configure Environment

Define the mock enterprise environment:

```yaml
  configuration:
    environment:
      network:
        topology: "simple"
        subnets:
          - name: "dmz"
            cidr: "192.168.10.0/24"
            services: ["web-server", "dns-server"]
          - name: "internal"
            cidr: "192.168.20.0/24"
            services: ["database", "file-server"]
      
      services:
        - name: "web-server"
          type: "wordpress"
          version: "5.8.0"  # Vulnerable version
          ip: "192.168.10.10"
          ports: [80, 443]
          vulnerabilities:
            - cve: "CVE-2021-34527"
              severity: "high"
              exploitable: true
        
        - name: "database"
          type: "mysql"
          version: "5.7.0"
          ip: "192.168.20.10"
          ports: [3306]
          credentials:
            - username: "root"
              password: "admin123"  # Weak password
      
      monitoring:
        siem_enabled: true
        ids_enabled: true
        log_level: "info"
        alert_threshold: "medium"
```

### Step 3: Define Objectives

Set clear objectives for both teams:

```yaml
    objectives:
      red_team:
        primary:
          - id: "initial_access"
            description: "Gain initial access to the web server"
            points: 100
            required: true
          - id: "privilege_escalation"
            description: "Escalate privileges on the compromised system"
            points: 150
            required: false
        
        secondary:
          - id: "data_exfiltration"
            description: "Extract sensitive data from the database"
            points: 200
            required: false
      
      blue_team:
        primary:
          - id: "detect_intrusion"
            description: "Detect and alert on intrusion attempts"
            points: 100
            required: true
          - id: "block_attack"
            description: "Successfully block attack attempts"
            points: 150
            required: false
        
        secondary:
          - id: "forensic_analysis"
            description: "Perform forensic analysis of the attack"
            points: 100
            required: false
```

### Step 4: Configure Timeline and Phases

Structure the scenario with phases:

```yaml
    duration: 1800  # 30 minutes total
    
    phases:
      - name: "reconnaissance"
        description: "Red Team performs reconnaissance"
        duration: 300  # 5 minutes
        allowed_agents:
          red_team: ["recon"]
          blue_team: ["soc_analyst", "network_monitor"]
        constraints:
          - type: "stealth_required"
            description: "Red Team must avoid detection"
          - type: "passive_only"
            description: "Only passive reconnaissance allowed"
      
      - name: "exploitation"
        description: "Active exploitation phase"
        duration: 900  # 15 minutes
        allowed_agents:
          red_team: ["recon", "exploit"]
          blue_team: ["soc_analyst", "incident_responder", "firewall_admin"]
        constraints:
          - type: "max_attempts"
            value: 3
            description: "Maximum 3 exploitation attempts"
      
      - name: "post_exploitation"
        description: "Post-exploitation activities"
        duration: 600  # 10 minutes
        allowed_agents:
          red_team: ["exploit", "persistence", "exfiltration"]
          blue_team: ["incident_responder", "forensic_analyst"]
        constraints:
          - type: "containment_active"
            description: "Blue Team containment measures active"
```

### Step 5: Define Scoring System

Configure how success is measured:

```yaml
    scoring:
      red_team:
        base_score: 0
        multipliers:
          stealth_bonus: 1.2  # 20% bonus for avoiding detection
          speed_bonus: 1.1    # 10% bonus for fast completion
        
        actions:
          successful_scan: 10
          vulnerability_found: 25
          successful_exploit: 100
          privilege_escalation: 150
          data_exfiltration: 200
          persistence_established: 100
        
        penalties:
          detected_by_blue: -50
          failed_exploit: -10
          system_crash: -100
      
      blue_team:
        base_score: 0
        multipliers:
          response_speed: 1.3  # 30% bonus for fast response
          accuracy_bonus: 1.1  # 10% bonus for accurate detection
        
        actions:
          alert_generated: 15
          attack_detected: 50
          attack_blocked: 100
          incident_contained: 150
          forensics_completed: 75
        
        penalties:
          false_positive: -5
          missed_detection: -25
          service_downtime: -50
```

## Advanced Scenario Features

### Dynamic Environment Changes

Create scenarios that evolve during execution:

```yaml
    dynamic_events:
      - trigger:
          type: "time_based"
          time: 600  # 10 minutes into scenario
        action:
          type: "service_update"
          target: "web-server"
          change: "patch_vulnerability"
          description: "Security patch applied to web server"
      
      - trigger:
          type: "condition_based"
          condition: "red_team_detected"
        action:
          type: "increase_monitoring"
          description: "Blue Team increases monitoring after detection"
          effects:
            - "reduce_stealth_effectiveness"
            - "increase_alert_sensitivity"
```

### Multi-Stage Scenarios

Create complex scenarios with branching paths:

```yaml
    stages:
      - name: "initial_compromise"
        success_condition: "web_server_compromised"
        failure_condition: "all_attacks_blocked"
        next_stage_success: "lateral_movement"
        next_stage_failure: "containment"
      
      - name: "lateral_movement"
        success_condition: "internal_network_access"
        failure_condition: "contained_by_blue_team"
        next_stage_success: "data_exfiltration"
        next_stage_failure: "incident_response"
      
      - name: "data_exfiltration"
        success_condition: "sensitive_data_extracted"
        failure_condition: "data_loss_prevented"
        next_stage_success: "mission_complete"
        next_stage_failure: "partial_success"
```

### Adaptive Difficulty

Implement scenarios that adjust based on performance:

```yaml
    adaptive_difficulty:
      enabled: true
      adjustment_interval: 300  # 5 minutes
      
      red_team_adjustments:
        - condition: "success_rate < 0.3"
          action: "reduce_blue_team_effectiveness"
          amount: 0.1
        - condition: "success_rate > 0.8"
          action: "increase_blue_team_effectiveness"
          amount: 0.1
      
      blue_team_adjustments:
        - condition: "detection_rate < 0.4"
          action: "increase_alert_sensitivity"
          amount: 0.15
        - condition: "false_positive_rate > 0.2"
          action: "decrease_alert_sensitivity"
          amount: 0.1
```

## Testing and Validation

### Scenario Validation

Use the built-in validation tools:

```bash
# Validate scenario syntax
archangel scenario validate scenarios/my-first-scenario.yml

# Test scenario configuration
archangel scenario test scenarios/my-first-scenario.yml --dry-run

# Run scenario with debug output
archangel scenario run scenarios/my-first-scenario.yml --debug
```

### Automated Testing

Create test cases for your scenarios:

```yaml
# tests/scenario-tests.yml
test_suite:
  name: "Basic Web Intrusion Tests"
  scenario: "basic-web-intrusion-001"
  
  test_cases:
    - name: "Red Team Can Discover Web Server"
      type: "objective_completion"
      team: "red_team"
      objective: "initial_access"
      expected_result: "success"
      timeout: 600
    
    - name: "Blue Team Detects Intrusion"
      type: "objective_completion"
      team: "blue_team"
      objective: "detect_intrusion"
      expected_result: "success"
      timeout: 900
    
    - name: "Scenario Completes Within Time Limit"
      type: "duration_check"
      max_duration: 1800
      expected_result: "success"
```

### Performance Testing

Test scenario performance and resource usage:

```bash
# Run performance test
archangel scenario benchmark scenarios/my-first-scenario.yml \
  --iterations 10 \
  --concurrent-agents 20 \
  --report-file performance-report.json
```

## Best Practices

### 1. Start Simple

Begin with basic scenarios and gradually add complexity:

```yaml
# Good: Simple, focused scenario
objectives:
  red_team:
    - "Compromise web server"
  blue_team:
    - "Detect and block attack"

# Avoid: Overly complex initial scenario
objectives:
  red_team:
    - "Multi-stage attack with 15 different techniques"
  blue_team:
    - "Complex incident response with 10 different tools"
```

### 2. Clear Objectives

Make objectives specific and measurable:

```yaml
# Good: Specific and measurable
- id: "sql_injection_success"
  description: "Successfully execute SQL injection on login form"
  success_criteria: "database_access_gained"
  points: 100

# Avoid: Vague objectives
- id: "hack_website"
  description: "Do something bad to the website"
  points: 50
```

### 3. Balanced Difficulty

Ensure scenarios are challenging but achievable:

```yaml
# Good: Balanced constraints
constraints:
  - type: "max_attempts"
    value: 5
    description: "Reasonable attempt limit"
  - type: "stealth_required"
    threshold: 0.7
    description: "Some stealth required but not perfect"

# Avoid: Impossible constraints
constraints:
  - type: "max_attempts"
    value: 1
    description: "Only one attempt allowed"
  - type: "perfect_stealth"
    description: "Must never be detected"
```

### 4. Realistic Environments

Model real-world environments accurately:

```yaml
# Good: Realistic service configuration
services:
  - name: "web-server"
    type: "apache"
    version: "2.4.41"
    modules: ["mod_php", "mod_ssl"]
    misconfigurations:
      - "directory_listing_enabled"
      - "server_tokens_exposed"

# Avoid: Unrealistic configurations
services:
  - name: "super-vulnerable-server"
    vulnerabilities: ["every-cve-ever"]
```

### 5. Comprehensive Documentation

Document scenarios thoroughly:

```yaml
scenario:
  metadata:
    documentation:
      learning_objectives:
        - "Understand web application reconnaissance"
        - "Practice SQL injection techniques"
        - "Learn incident response procedures"
      
      prerequisites:
        - "Basic understanding of web technologies"
        - "Familiarity with SQL"
      
      resources:
        - url: "https://owasp.org/www-project-top-ten/"
          description: "OWASP Top 10 reference"
        - url: "https://attack.mitre.org/"
          description: "MITRE ATT&CK framework"
```

## Example Scenarios

### Example 1: Phishing Campaign

```yaml
scenario:
  metadata:
    id: "phishing-campaign-001"
    name: "Corporate Phishing Campaign"
    description: "Red Team conducts spear-phishing while Blue Team defends"
    difficulty: "intermediate"
  
  configuration:
    duration: 2700  # 45 minutes
    
    environment:
      services:
        - name: "email-server"
          type: "postfix"
          ip: "192.168.20.5"
        - name: "web-server"
          type: "nginx"
          ip: "192.168.10.10"
      
      synthetic_users:
        count: 50
        profiles:
          - type: "executive"
            count: 5
            susceptibility: 0.3
          - type: "employee"
            count: 40
            susceptibility: 0.6
          - type: "it_admin"
            count: 5
            susceptibility: 0.1
    
    objectives:
      red_team:
        - id: "craft_phishing_email"
          description: "Create convincing phishing email"
          points: 50
        - id: "successful_phish"
          description: "Get user to click malicious link"
          points: 100
        - id: "credential_harvest"
          description: "Harvest user credentials"
          points: 150
      
      blue_team:
        - id: "detect_phishing"
          description: "Identify phishing email"
          points: 75
        - id: "block_malicious_domain"
          description: "Block access to malicious domain"
          points: 100
        - id: "user_education"
          description: "Educate users about the threat"
          points: 50
```

### Example 2: Insider Threat

```yaml
scenario:
  metadata:
    id: "insider-threat-001"
    name: "Malicious Insider Data Theft"
    description: "Simulate insider threat with legitimate access"
    difficulty: "advanced"
  
  configuration:
    duration: 3600  # 60 minutes
    
    environment:
      insider_profile:
        role: "database_administrator"
        access_level: "high"
        motivation: "financial"
        behavior_pattern: "gradual_escalation"
      
      data_classification:
        - type: "public"
          sensitivity: 1
          monitoring: "low"
        - type: "internal"
          sensitivity: 3
          monitoring: "medium"
        - type: "confidential"
          sensitivity: 5
          monitoring: "high"
    
    phases:
      - name: "reconnaissance"
        description: "Insider explores accessible data"
        duration: 900
        constraints:
          - type: "legitimate_access_only"
            description: "Must use legitimate access methods"
      
      - name: "data_collection"
        description: "Systematic data collection"
        duration: 1800
        constraints:
          - type: "avoid_detection"
            threshold: 0.8
      
      - name: "exfiltration"
        description: "Data exfiltration attempt"
        duration: 900
        constraints:
          - type: "covert_channels"
            description: "Must use covert exfiltration methods"
```

### Example 3: Ransomware Simulation

```yaml
scenario:
  metadata:
    id: "ransomware-simulation-001"
    name: "Ransomware Attack Simulation"
    description: "Full ransomware attack lifecycle simulation"
    difficulty: "advanced"
  
  configuration:
    duration: 5400  # 90 minutes
    
    phases:
      - name: "initial_access"
        duration: 900
        objectives:
          red_team: ["gain_foothold"]
          blue_team: ["monitor_endpoints"]
      
      - name: "discovery_and_lateral_movement"
        duration: 1800
        objectives:
          red_team: ["map_network", "escalate_privileges"]
          blue_team: ["detect_lateral_movement"]
      
      - name: "data_exfiltration"
        duration: 1200
        objectives:
          red_team: ["exfiltrate_sensitive_data"]
          blue_team: ["detect_data_exfiltration"]
      
      - name: "encryption_deployment"
        duration: 900
        objectives:
          red_team: ["deploy_ransomware"]
          blue_team: ["prevent_encryption", "isolate_systems"]
      
      - name: "recovery"
        duration: 900
        objectives:
          red_team: ["maintain_persistence"]
          blue_team: ["restore_systems", "eradicate_threat"]
    
    scoring:
      red_team:
        encryption_percentage: 10  # Points per % of systems encrypted
        data_exfiltrated: 5        # Points per GB exfiltrated
        ransom_payment: 1000       # Bonus for successful ransom
      
      blue_team:
        prevention_bonus: 500      # Bonus for preventing encryption
        recovery_speed: 2          # Points per minute faster than baseline
        data_protection: 10        # Points per % of data protected
```

## Scenario Management

### Version Control

Track scenario versions and changes:

```yaml
scenario:
  metadata:
    version: "2.1.0"
    changelog:
      - version: "2.1.0"
        date: "2024-01-15"
        changes:
          - "Added adaptive difficulty"
          - "Improved Blue Team objectives"
      - version: "2.0.0"
        date: "2024-01-01"
        changes:
          - "Major rewrite with new phase system"
          - "Updated scoring mechanism"
```

### Scenario Collections

Organize related scenarios:

```yaml
# collections/web-security-fundamentals.yml
collection:
  name: "Web Security Fundamentals"
  description: "Basic web security scenarios for beginners"
  difficulty: "beginner"
  
  scenarios:
    - id: "basic-web-intrusion-001"
      order: 1
      prerequisites: []
    - id: "sql-injection-basics-001"
      order: 2
      prerequisites: ["basic-web-intrusion-001"]
    - id: "xss-fundamentals-001"
      order: 3
      prerequisites: ["basic-web-intrusion-001"]
  
  learning_path:
    estimated_duration: "4 hours"
    certification: "Web Security Basics"
```

## Troubleshooting Scenarios

### Common Issues

1. **Scenario Won't Start**
   ```bash
   # Check scenario validation
   archangel scenario validate my-scenario.yml
   
   # Check resource availability
   archangel system status
   ```

2. **Agents Not Behaving as Expected**
   ```bash
   # Enable debug logging
   archangel scenario run my-scenario.yml --log-level debug
   
   # Check agent configurations
   archangel agents list --scenario my-scenario
   ```

3. **Performance Issues**
   ```bash
   # Monitor resource usage
   archangel scenario monitor my-scenario.yml
   
   # Reduce complexity
   # - Fewer concurrent agents
   # - Simpler environment
   # - Shorter duration
   ```

### Debug Tools

```bash
# Scenario validation
archangel scenario validate scenarios/my-scenario.yml

# Dry run testing
archangel scenario test scenarios/my-scenario.yml --dry-run

# Interactive debugging
archangel scenario debug scenarios/my-scenario.yml --interactive

# Performance profiling
archangel scenario profile scenarios/my-scenario.yml --output profile.json
```

---

*Next: [Best Practices Guide](best-practices.md) for advanced scenario design patterns*