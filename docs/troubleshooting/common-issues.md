# Common Issues and Solutions

This guide covers the most frequently encountered issues when deploying and operating the Archangel Autonomous AI Evolution system, along with step-by-step solutions.

## Table of Contents
- [Installation and Setup Issues](#installation-and-setup-issues)
- [Agent Communication Problems](#agent-communication-problems)
- [Performance and Resource Issues](#performance-and-resource-issues)
- [LLM Integration Issues](#llm-integration-issues)
- [Database and Storage Problems](#database-and-storage-problems)
- [Network and Connectivity Issues](#network-and-connectivity-issues)
- [Scenario Execution Problems](#scenario-execution-problems)
- [Monitoring and Logging Issues](#monitoring-and-logging-issues)

## Installation and Setup Issues

### Issue: Docker Compose Services Won't Start

**Symptoms:**
- Services fail to start with exit code 1
- "Port already in use" errors
- "Cannot connect to Docker daemon" errors

**Solutions:**

1. **Check Docker daemon status:**
   ```bash
   # Linux/macOS
   sudo systemctl status docker
   
   # Start Docker if stopped
   sudo systemctl start docker
   
   # Windows
   # Restart Docker Desktop application
   ```

2. **Check port conflicts:**
   ```bash
   # Check what's using the ports
   netstat -tulpn | grep :8000
   netstat -tulpn | grep :3000
   netstat -tulpn | grep :9090
   
   # Kill conflicting processes
   sudo kill -9 <PID>
   
   # Or modify ports in docker-compose.yml
   ports:
     - "8001:8000"  # Change host port
   ```

3. **Clean up Docker resources:**
   ```bash
   # Stop all containers
   docker-compose down
   
   # Remove orphaned containers
   docker-compose down --remove-orphans
   
   # Clean up system
   docker system prune -f
   
   # Remove volumes if needed (WARNING: Data loss)
   docker-compose down -v
   ```

4. **Check system resources:**
   ```bash
   # Check available disk space
   df -h
   
   # Check memory usage
   free -h
   
   # Check Docker space usage
   docker system df
   ```

### Issue: Environment Variables Not Loading

**Symptoms:**
- Services start but can't connect to databases
- "API key not found" errors
- Configuration defaults being used instead of custom values

**Solutions:**

1. **Verify .env file exists and is properly formatted:**
   ```bash
   # Check if .env file exists
   ls -la .env
   
   # Verify format (no spaces around =)
   cat .env | grep -E "^[A-Z_]+=.*$"
   
   # Example correct format:
   OPENAI_API_KEY=sk-your-key-here
   POSTGRES_PASSWORD=secure_password
   ```

2. **Check environment variable loading:**
   ```bash
   # Test environment loading
   docker-compose config
   
   # Check specific service environment
   docker-compose exec coordinator env | grep OPENAI
   ```

3. **Common .env file issues:**
   ```bash
   # Wrong: Spaces around equals
   OPENAI_API_KEY = sk-your-key-here
   
   # Wrong: Quotes when not needed
   POSTGRES_PASSWORD="password"
   
   # Correct:
   OPENAI_API_KEY=sk-your-key-here
   POSTGRES_PASSWORD=secure_password
   ```

### Issue: Permission Denied Errors

**Symptoms:**
- "Permission denied" when accessing files
- Services can't write to mounted volumes
- Database initialization fails

**Solutions:**

1. **Fix file permissions:**
   ```bash
   # Make scripts executable
   chmod +x scripts/*.sh
   
   # Fix directory permissions
   sudo chown -R $USER:$USER ./logs
   sudo chown -R $USER:$USER ./data
   
   # Set proper permissions for config files
   chmod 644 config/*.yml
   ```

2. **Docker volume permissions:**
   ```bash
   # Create directories with proper ownership
   mkdir -p logs data config
   sudo chown -R 1000:1000 logs data
   
   # Or use user mapping in docker-compose.yml
   services:
     coordinator:
       user: "${UID}:${GID}"
   ```

## Agent Communication Problems

### Issue: Agents Can't Connect to Coordinator

**Symptoms:**
- Agents show "Connection refused" errors
- Agents start but don't register with coordinator
- No agent activity in logs

**Solutions:**

1. **Check network connectivity:**
   ```bash
   # Test coordinator accessibility
   docker-compose exec red-team-recon curl http://coordinator:8000/health
   
   # Check DNS resolution
   docker-compose exec red-team-recon nslookup coordinator
   
   # Verify network configuration
   docker network ls
   docker network inspect archangel_archangel-network
   ```

2. **Verify coordinator is running:**
   ```bash
   # Check coordinator status
   docker-compose ps coordinator
   
   # Check coordinator logs
   docker-compose logs coordinator
   
   # Test coordinator health endpoint
   curl http://localhost:8000/health
   ```

3. **Check firewall and security groups:**
   ```bash
   # Linux: Check iptables
   sudo iptables -L
   
   # Check if ports are accessible
   telnet localhost 8000
   
   # For cloud deployments, verify security groups allow traffic
   ```

### Issue: Redis Connection Failures

**Symptoms:**
- "Connection to Redis failed" errors
- Agents can't share intelligence
- Message bus not working

**Solutions:**

1. **Verify Redis is running:**
   ```bash
   # Check Redis status
   docker-compose ps redis
   
   # Test Redis connectivity
   docker-compose exec redis redis-cli ping
   
   # Check Redis logs
   docker-compose logs redis
   ```

2. **Check Redis authentication:**
   ```bash
   # Test with password
   docker-compose exec redis redis-cli -a your_redis_password ping
   
   # Verify password in environment
   docker-compose exec coordinator env | grep REDIS
   ```

3. **Redis memory issues:**
   ```bash
   # Check Redis memory usage
   docker-compose exec redis redis-cli info memory
   
   # Increase Redis memory limit
   # In docker-compose.yml:
   redis:
     command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
   ```

### Issue: Message Bus Encryption Errors

**Symptoms:**
- "Encryption key mismatch" errors
- Messages not being delivered
- TLS handshake failures

**Solutions:**

1. **Verify encryption keys:**
   ```bash
   # Check encryption key length (should be 32 characters)
   echo $ENCRYPTION_KEY | wc -c
   
   # Generate new key if needed
   openssl rand -hex 16
   ```

2. **Check TLS certificates:**
   ```bash
   # Verify certificate files exist
   ls -la certs/
   
   # Check certificate validity
   openssl x509 -in certs/server.crt -text -noout
   
   # Regenerate certificates if needed
   ./scripts/generate-certs.sh
   ```

## Performance and Resource Issues

### Issue: High Memory Usage

**Symptoms:**
- System becomes slow or unresponsive
- Out of memory errors
- Containers being killed by OOM killer

**Solutions:**

1. **Monitor memory usage:**
   ```bash
   # Check system memory
   free -h
   
   # Check Docker container memory usage
   docker stats
   
   # Check specific service memory
   docker-compose exec coordinator ps aux
   ```

2. **Optimize memory settings:**
   ```bash
   # Add memory limits to docker-compose.yml
   services:
     coordinator:
       deploy:
         resources:
           limits:
             memory: 2G
           reservations:
             memory: 1G
   ```

3. **Reduce agent concurrency:**
   ```yaml
   # In agent configuration
   agent_config:
     max_concurrent_actions: 2  # Reduce from default 5
     memory_cache_size: 500     # Reduce cache size
     batch_size: 10             # Reduce batch processing
   ```

4. **Optimize vector database:**
   ```bash
   # ChromaDB optimization
   # Reduce embedding dimensions
   # Implement memory cleanup
   docker-compose exec chromadb curl -X POST http://localhost:8000/api/v1/collections/cleanup
   ```

### Issue: Slow Agent Response Times

**Symptoms:**
- Agents take >10 seconds to respond
- Scenario timeouts
- Poor user experience

**Solutions:**

1. **Check LLM API latency:**
   ```bash
   # Test OpenAI API response time
   time curl -X POST https://api.openai.com/v1/chat/completions \
     -H "Authorization: Bearer $OPENAI_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}'
   ```

2. **Optimize LLM settings:**
   ```yaml
   # In agent configuration
   llm_config:
     model: "gpt-3.5-turbo"  # Faster than gpt-4
     max_tokens: 500         # Reduce token limit
     temperature: 0.7        # Reduce for faster inference
     timeout: 10             # Set reasonable timeout
   ```

3. **Implement caching:**
   ```python
   # Enable response caching
   agent_config = {
       "cache_enabled": True,
       "cache_ttl": 300,  # 5 minutes
       "cache_size": 1000
   }
   ```

4. **Use local LLM for faster responses:**
   ```yaml
   # Switch to local model
   llm_config:
     provider: "ollama"
     model: "llama3:8b"
     base_url: "http://ollama:11434"
   ```

### Issue: High CPU Usage

**Symptoms:**
- System CPU usage consistently >80%
- Slow response times
- Fan noise on local machines

**Solutions:**

1. **Identify CPU-intensive processes:**
   ```bash
   # Check system CPU usage
   htop
   
   # Check Docker container CPU usage
   docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
   
   # Profile specific service
   docker-compose exec coordinator py-spy top --pid 1
   ```

2. **Optimize agent processing:**
   ```yaml
   # Reduce processing frequency
   agent_config:
     decision_interval: 10    # Increase from 5 seconds
     batch_processing: true   # Enable batch processing
     parallel_actions: 2      # Reduce parallelism
   ```

3. **Implement CPU limits:**
   ```yaml
   # In docker-compose.yml
   services:
     coordinator:
       deploy:
         resources:
           limits:
             cpus: '1.0'
   ```

## LLM Integration Issues

### Issue: OpenAI API Rate Limits

**Symptoms:**
- "Rate limit exceeded" errors
- Agents stop responding
- 429 HTTP status codes in logs

**Solutions:**

1. **Implement rate limiting:**
   ```python
   # Add rate limiting to LLM calls
   from ratelimit import limits, sleep_and_retry
   
   @sleep_and_retry
   @limits(calls=50, period=60)  # 50 calls per minute
   def call_llm(prompt):
       # LLM call implementation
       pass
   ```

2. **Use exponential backoff:**
   ```python
   import time
   import random
   
   def call_llm_with_backoff(prompt, max_retries=3):
       for attempt in range(max_retries):
           try:
               return call_llm(prompt)
           except RateLimitError:
               wait_time = (2 ** attempt) + random.uniform(0, 1)
               time.sleep(wait_time)
       raise Exception("Max retries exceeded")
   ```

3. **Optimize API usage:**
   ```yaml
   # Reduce API calls
   agent_config:
     cache_responses: true
     batch_requests: true
     reduce_context_size: true
     use_cheaper_model: "gpt-3.5-turbo"
   ```

### Issue: LLM Response Quality Issues

**Symptoms:**
- Agents make poor decisions
- Inconsistent behavior
- Hallucinated information

**Solutions:**

1. **Improve prompt engineering:**
   ```python
   # Better prompt structure
   prompt = f"""
   You are a cybersecurity expert. Your role is {agent_role}.
   
   Context: {context}
   Current situation: {situation}
   Available tools: {tools}
   
   Based on this information, what is your next action?
   Respond in JSON format with 'action', 'reasoning', and 'confidence'.
   """
   ```

2. **Add response validation:**
   ```python
   def validate_llm_response(response):
       required_fields = ['action', 'reasoning', 'confidence']
       if not all(field in response for field in required_fields):
           raise ValueError("Invalid response format")
       
       if response['confidence'] < 0.5:
           # Request human review or retry
           return False
       return True
   ```

3. **Implement fallback logic:**
   ```python
   def get_agent_action(context):
       try:
           llm_response = call_llm(context)
           if validate_llm_response(llm_response):
               return llm_response
       except Exception as e:
           logger.warning(f"LLM call failed: {e}")
       
       # Fallback to rule-based logic
       return get_rule_based_action(context)
   ```

### Issue: Local LLM Performance Problems

**Symptoms:**
- Very slow response times with local models
- High memory usage
- Model loading failures

**Solutions:**

1. **Optimize model selection:**
   ```bash
   # Use smaller, faster models
   ollama pull llama3:8b      # Instead of 70b
   ollama pull codellama:7b   # For code tasks
   ollama pull mistral:7b     # Alternative option
   ```

2. **Configure model parameters:**
   ```yaml
   # Optimize for speed
   ollama_config:
     num_ctx: 2048      # Reduce context window
     num_predict: 256   # Limit response length
     temperature: 0.7   # Balance creativity and consistency
     num_gpu: 1         # Use GPU if available
   ```

3. **Hardware optimization:**
   ```bash
   # Check GPU availability
   nvidia-smi
   
   # Enable GPU support in Docker
   # Add to docker-compose.yml:
   ollama:
     deploy:
       resources:
         reservations:
           devices:
             - driver: nvidia
               count: 1
               capabilities: [gpu]
   ```

## Database and Storage Problems

### Issue: PostgreSQL Connection Failures

**Symptoms:**
- "Connection refused" to PostgreSQL
- Database initialization failures
- Data not persisting

**Solutions:**

1. **Check PostgreSQL status:**
   ```bash
   # Verify PostgreSQL is running
   docker-compose ps postgresql
   
   # Check PostgreSQL logs
   docker-compose logs postgresql
   
   # Test connection
   docker-compose exec postgresql pg_isready -U archangel
   ```

2. **Verify database credentials:**
   ```bash
   # Test connection with credentials
   docker-compose exec postgresql psql -U archangel -d archangel -c "SELECT version();"
   
   # Check environment variables
   docker-compose exec coordinator env | grep POSTGRES
   ```

3. **Database initialization issues:**
   ```bash
   # Recreate database
   docker-compose down
   docker volume rm archangel_postgres-data
   docker-compose up -d postgresql
   
   # Wait for initialization
   docker-compose logs -f postgresql
   ```

### Issue: ChromaDB Vector Store Problems

**Symptoms:**
- Vector similarity search not working
- Memory storage errors
- Slow query performance

**Solutions:**

1. **Check ChromaDB status:**
   ```bash
   # Verify ChromaDB is running
   docker-compose ps chromadb
   
   # Test API endpoint
   curl http://localhost:8000/api/v1/heartbeat
   
   # Check ChromaDB logs
   docker-compose logs chromadb
   ```

2. **Optimize vector storage:**
   ```python
   # Reduce embedding dimensions
   embedding_config = {
       "model": "all-MiniLM-L6-v2",  # Smaller model
       "dimensions": 384,             # Reduced dimensions
       "batch_size": 100             # Optimize batch size
   }
   ```

3. **Clean up vector database:**
   ```bash
   # Reset ChromaDB data
   docker-compose exec chromadb curl -X DELETE http://localhost:8000/api/v1/reset
   
   # Or recreate volume
   docker-compose down
   docker volume rm archangel_chromadb-data
   docker-compose up -d chromadb
   ```

### Issue: Disk Space Issues

**Symptoms:**
- "No space left on device" errors
- Log files growing too large
- Database write failures

**Solutions:**

1. **Check disk usage:**
   ```bash
   # Check overall disk usage
   df -h
   
   # Check Docker space usage
   docker system df
   
   # Check log file sizes
   du -sh logs/
   ```

2. **Clean up disk space:**
   ```bash
   # Clean Docker system
   docker system prune -f
   
   # Remove old images
   docker image prune -a
   
   # Clean up logs
   find logs/ -name "*.log" -mtime +7 -delete
   ```

3. **Implement log rotation:**
   ```yaml
   # In docker-compose.yml
   services:
     coordinator:
       logging:
         driver: "json-file"
         options:
           max-size: "10m"
           max-file: "3"
   ```

## Network and Connectivity Issues

### Issue: Container Network Connectivity

**Symptoms:**
- Services can't reach each other
- DNS resolution failures
- Network timeouts

**Solutions:**

1. **Check Docker networks:**
   ```bash
   # List Docker networks
   docker network ls
   
   # Inspect network configuration
   docker network inspect archangel_archangel-network
   
   # Check container network settings
   docker-compose exec coordinator cat /etc/resolv.conf
   ```

2. **Test network connectivity:**
   ```bash
   # Test DNS resolution
   docker-compose exec coordinator nslookup redis
   docker-compose exec coordinator nslookup postgresql
   
   # Test port connectivity
   docker-compose exec coordinator telnet redis 6379
   docker-compose exec coordinator telnet postgresql 5432
   ```

3. **Fix network issues:**
   ```bash
   # Recreate networks
   docker-compose down
   docker network prune
   docker-compose up -d
   
   # Check for IP conflicts
   docker network inspect bridge
   ```

### Issue: External API Connectivity

**Symptoms:**
- Can't reach OpenAI API
- DNS resolution failures for external services
- SSL/TLS certificate errors

**Solutions:**

1. **Test external connectivity:**
   ```bash
   # Test DNS resolution
   docker-compose exec coordinator nslookup api.openai.com
   
   # Test HTTPS connectivity
   docker-compose exec coordinator curl -I https://api.openai.com
   
   # Check certificate validity
   docker-compose exec coordinator openssl s_client -connect api.openai.com:443
   ```

2. **Configure proxy settings:**
   ```yaml
   # If behind corporate proxy
   services:
     coordinator:
       environment:
         - HTTP_PROXY=http://proxy.company.com:8080
         - HTTPS_PROXY=http://proxy.company.com:8080
         - NO_PROXY=localhost,127.0.0.1,redis,postgresql
   ```

3. **Fix certificate issues:**
   ```bash
   # Update CA certificates
   docker-compose exec coordinator apt-get update && apt-get install -y ca-certificates
   
   # Or add custom certificates
   # Mount certificate directory in docker-compose.yml
   volumes:
     - ./certs:/usr/local/share/ca-certificates
   ```

## Scenario Execution Problems

### Issue: Scenarios Won't Start

**Symptoms:**
- "Scenario validation failed" errors
- Agents not initializing
- Environment setup failures

**Solutions:**

1. **Validate scenario configuration:**
   ```bash
   # Check scenario syntax
   archangel scenario validate scenarios/my-scenario.yml
   
   # Test scenario dry-run
   archangel scenario test scenarios/my-scenario.yml --dry-run
   
   # Check scenario dependencies
   archangel scenario deps scenarios/my-scenario.yml
   ```

2. **Check resource availability:**
   ```bash
   # Verify system resources
   archangel system status
   
   # Check agent availability
   archangel agents list --available
   
   # Verify environment services
   docker-compose ps
   ```

3. **Debug scenario loading:**
   ```bash
   # Enable debug logging
   archangel scenario run scenarios/my-scenario.yml --debug
   
   # Check scenario logs
   tail -f logs/scenario-execution.log
   ```

### Issue: Agents Not Following Scenario Rules

**Symptoms:**
- Agents performing unauthorized actions
- Phase transitions not working
- Constraint violations

**Solutions:**

1. **Check constraint enforcement:**
   ```yaml
   # Verify constraint configuration
   constraints:
     - type: "phase_restriction"
       enforcement: "strict"  # Ensure strict enforcement
       violation_action: "block"
   ```

2. **Review agent behavior:**
   ```bash
   # Check agent decision logs
   grep "constraint_violation" logs/agent-*.log
   
   # Monitor agent actions
   archangel agents monitor --scenario my-scenario
   ```

3. **Update agent prompts:**
   ```python
   # Add constraint awareness to prompts
   prompt += f"""
   IMPORTANT CONSTRAINTS:
   - Current phase: {current_phase}
   - Allowed actions: {allowed_actions}
   - Forbidden actions: {forbidden_actions}
   
   You MUST respect these constraints.
   """
   ```

## Monitoring and Logging Issues

### Issue: Grafana Dashboard Not Loading

**Symptoms:**
- Grafana shows "No data" or blank panels
- Dashboard configuration errors
- Prometheus connection failures

**Solutions:**

1. **Check Grafana status:**
   ```bash
   # Verify Grafana is running
   docker-compose ps grafana
   
   # Check Grafana logs
   docker-compose logs grafana
   
   # Test Grafana web interface
   curl http://localhost:3000/api/health
   ```

2. **Verify Prometheus connection:**
   ```bash
   # Test Prometheus endpoint
   curl http://localhost:9090/api/v1/query?query=up
   
   # Check Prometheus targets
   curl http://localhost:9090/api/v1/targets
   
   # Verify metrics are being collected
   curl http://localhost:8000/metrics
   ```

3. **Fix dashboard configuration:**
   ```bash
   # Reload dashboard configuration
   docker-compose restart grafana
   
   # Check dashboard JSON syntax
   python -m json.tool monitoring/grafana/dashboards/archangel.json
   
   # Import dashboard manually
   # Go to Grafana UI -> Import -> Upload JSON
   ```

### Issue: Missing Log Data

**Symptoms:**
- Log files are empty or missing
- No agent activity logs
- Incomplete audit trails

**Solutions:**

1. **Check logging configuration:**
   ```bash
   # Verify log directories exist
   ls -la logs/
   
   # Check log file permissions
   ls -la logs/*.log
   
   # Verify logging is enabled
   grep -r "LOG_LEVEL" .env docker-compose.yml
   ```

2. **Fix logging issues:**
   ```yaml
   # Ensure proper log configuration
   services:
     coordinator:
       environment:
         - LOG_LEVEL=INFO
         - LOG_FORMAT=json
       volumes:
         - ./logs:/app/logs
   ```

3. **Enable debug logging:**
   ```bash
   # Temporarily increase log level
   export LOG_LEVEL=DEBUG
   docker-compose restart
   
   # Check for log output
   docker-compose logs -f coordinator
   ```

## Quick Diagnostic Commands

### System Health Check
```bash
#!/bin/bash
# Quick system health check

echo "=== Docker Status ==="
docker --version
docker-compose --version
docker system df

echo "=== Service Status ==="
docker-compose ps

echo "=== Resource Usage ==="
docker stats --no-stream

echo "=== Network Status ==="
docker network ls
curl -s http://localhost:8000/health || echo "Coordinator not accessible"

echo "=== Log Summary ==="
docker-compose logs --tail=10 coordinator
```

### Agent Connectivity Test
```bash
#!/bin/bash
# Test agent connectivity

echo "=== Testing Agent Connectivity ==="
for service in coordinator redis postgresql chromadb; do
    echo "Testing $service..."
    docker-compose exec coordinator nc -zv $service 8000 2>/dev/null && echo "✓ $service reachable" || echo "✗ $service unreachable"
done
```

### Performance Monitoring
```bash
#!/bin/bash
# Monitor system performance

echo "=== System Performance ==="
echo "CPU Usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1

echo "Memory Usage:"
free -h | grep Mem | awk '{print $3 "/" $2}'

echo "Disk Usage:"
df -h | grep -E "/$|/var"

echo "Docker Stats:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

---

*For more specific issues, see the [Debugging Guide](debugging.md) or [FAQ](faq.md)*