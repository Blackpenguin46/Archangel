# Archangel AI Security Expert System - Docker Testing Guide 🐳🛡️

This guide provides comprehensive instructions for testing Archangel in a containerized Linux environment using Docker.

## 🚀 Quick Start

### 1. Prerequisites
- Docker installed on your system
- Docker Compose (v2.0+ recommended)
- At least 4GB RAM available for containers
- Optional: Hugging Face account for cloud AI features

### 2. Setup and Build
```bash
# Make setup script executable and run it
chmod +x docker-setup.sh
./docker-setup.sh

# Or build manually
docker build -t archangel-ai:latest .
```

### 3. Start Testing Environment
```bash
# Start containers in background
docker-compose up -d

# Enter the container for testing
docker-compose exec archangel bash
```

## 🧪 Testing Scenarios

### Basic AI Functionality Testing

#### Test 1: Cloud-Based AI Analysis (Lightweight Mode)
```bash
# Inside container - requires HF token in .env
python3 archangel_lightweight.py analyze google.com
python3 archangel_lightweight.py interactive
```

#### Test 2: Local AI Models
```bash
# Test local model capabilities
python3 archangel_ai.py analyze 192.168.1.1
python3 archangel_ai.py interactive
```

#### Test 3: AI Reasoning Demonstrations
```bash
# Basic AI reasoning demo
python3 demo_archangel.py

# Interactive CLI with conversational AI
python3 cli.py

# Complete system demonstration
python3 full_demo.py
```

### Advanced Architecture Testing

#### Test 4: Hybrid Kernel-Userspace Architecture
```bash
# Build kernel module (in container)
cd kernel
make clean && make

# Load kernel module (requires privileged container)
sudo make load

# Test AI-kernel communication
cd /app
python3 hybrid_demo.py

# Check kernel module status
sudo make info
sudo make logs
```

#### Test 5: Tool Integration and Orchestration
```bash
# Test AI-driven tool orchestration
python3 archangel_ai.py interactive
# Then use: analyze <target> and choose 'y' for autonomous execution

# Test unified AI orchestrator
python3 demo_unified_ai.py
```

### Hugging Face Integration Testing

#### Test 6: HF API Integration
```bash
# Set your token and test cloud models
export HF_TOKEN="your_token_here"
python3 archangel_lightweight.py analyze example.com

# Test multiple model fallbacks
python3 -c "
from core.huggingface_orchestrator import HuggingFaceOrchestrator
orchestrator = HuggingFaceOrchestrator()
result = orchestrator.analyze_security_context('test web application', 'reconnaissance')
print(result)
"
```

#### Test 7: SmolAgents Autonomous Operation
```bash
# Test autonomous security agents
python3 -c "
from tools.smolagents_security_tools import SecurityToolsAgent
agent = SecurityToolsAgent()
result = agent.analyze_target('scanme.nmap.org')
print(result)
"
```

## 🔧 Development and Debugging

### Container Management
```bash
# View container logs
docker-compose logs -f archangel

# Restart containers
docker-compose restart

# Stop and remove containers
docker-compose down

# Rebuild after code changes
docker-compose build --no-cache
docker-compose up -d
```

### Inside Container Debugging
```bash
# Enter container shell
docker-compose exec archangel bash

# Check system status
ps aux | grep python
top
df -h

# View kernel module status (if loaded)
lsmod | grep archangel
dmesg | tail -20

# Check Python environment
python3 --version
pip list | grep -E "(transformers|torch|huggingface)"
```

### Volume and Data Management
```bash
# Check Docker volumes
docker volume ls | grep archangel

# Inspect volume contents
docker volume inspect archangel_archangel-hf-cache

# Clear Hugging Face cache (if needed)
docker-compose down
docker volume rm archangel_archangel-hf-cache
docker-compose up -d
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Issue 1: Permission Denied for Kernel Module
```bash
# Solution: Ensure container runs with privileged mode
docker-compose down
# Edit docker-compose.yml to ensure: privileged: true
docker-compose up -d
```

#### Issue 2: Hugging Face Token Not Working
```bash
# Check if token is set correctly
docker-compose exec archangel env | grep HF_TOKEN

# Test token manually
docker-compose exec archangel python3 -c "
import os
from huggingface_hub import login
try:
    login(token=os.getenv('HF_TOKEN'))
    print('✅ Token is valid')
except Exception as e:
    print(f'❌ Token error: {e}')
"
```

#### Issue 3: Memory Issues with Local Models
```bash
# Check available memory
docker-compose exec archangel free -h

# Use lightweight mode instead
python3 archangel_lightweight.py analyze target.com
```

#### Issue 4: Network Connectivity Issues
```bash
# Test network connectivity
docker-compose exec archangel ping -c 3 google.com
docker-compose exec archangel curl -I https://huggingface.co

# Check DNS resolution
docker-compose exec archangel nslookup huggingface.co
```

### Kernel Module Specific Issues

#### Issue 5: Kernel Headers Not Found
```bash
# Check kernel version and headers
docker-compose exec archangel uname -r
docker-compose exec archangel dpkg -l | grep linux-headers

# If headers missing, they should be installed by Dockerfile
# Rebuild container if necessary
docker-compose build --no-cache
```

#### Issue 6: Module Compilation Errors
```bash
# Clean and rebuild kernel module
docker-compose exec archangel bash -c "cd kernel && make clean && make V=1"

# Check for specific errors
docker-compose exec archangel bash -c "cd kernel && make 2>&1 | grep -i error"
```

## 📊 Performance Testing

### Benchmark AI Response Times
```bash
# Test lightweight cloud AI performance
time python3 archangel_lightweight.py analyze google.com

# Test local AI performance
time python3 archangel_ai.py analyze 192.168.1.1

# Benchmark kernel communication
python3 -c "
import time
from core.kernel_interface import KernelInterface
ki = KernelInterface()
start = time.time()
for i in range(100):
    ki.simulate_security_event('test_event')
print(f'100 kernel calls took: {time.time() - start:.3f}s')
"
```

### Memory Usage Monitoring
```bash
# Monitor memory usage during AI operations
docker-compose exec archangel bash -c "
while true; do
    echo '=== Memory Usage ==='
    free -h
    echo '=== Top Processes ==='
    ps aux --sort=-%mem | head -10
    sleep 5
done
"
```

## 🔒 Security Testing

### Test Security Features
```bash
# Test with various target types
python3 archangel_lightweight.py analyze scanme.nmap.org
python3 archangel_lightweight.py analyze testphp.vulnweb.com
python3 archangel_lightweight.py analyze httpbin.org

# Test AI security reasoning
python3 cli.py
# Then ask: "explain how SQL injection works"
# Or: "what should I look for in a web application security assessment?"
```

### Validate Defensive Focus
```bash
# Verify system maintains defensive posture
python3 -c "
from core.real_ai_security_expert import RealAISecurityExpert
expert = RealAISecurityExpert()
# Should provide educational content, not offensive techniques
response = expert.analyze_target('example.com')
print(response)
"
```

## 📈 Advanced Testing Scenarios

### Multi-Container Testing
```bash
# Start with database and cache
docker-compose --profile with-db --profile with-cache up -d

# Test database connectivity
docker-compose exec archangel python3 -c "
import psycopg2
try:
    conn = psycopg2.connect(
        host='archangel-db',
        database='archangel',
        user='archangel',
        password='archangel_secure_password'
    )
    print('✅ Database connection successful')
    conn.close()
except Exception as e:
    print(f'❌ Database connection failed: {e}')
"

# Test Redis cache
docker-compose exec archangel python3 -c "
import redis
try:
    r = redis.Redis(host='archangel-cache', port=6379, db=0)
    r.ping()
    print('✅ Redis connection successful')
except Exception as e:
    print(f'❌ Redis connection failed: {e}')
"
```

### Load Testing
```bash
# Simulate multiple concurrent AI requests
docker-compose exec archangel python3 -c "
import asyncio
import time
from core.real_ai_security_expert import RealAISecurityExpert

async def test_concurrent_analysis():
    expert = RealAISecurityExpert()
    tasks = []
    targets = ['google.com', 'github.com', 'stackoverflow.com']
    
    start_time = time.time()
    for target in targets:
        task = asyncio.create_task(expert.analyze_target(target))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    
    print(f'Analyzed {len(targets)} targets in {end_time - start_time:.2f}s')
    for i, result in enumerate(results):
        print(f'Target {targets[i]}: {len(result)} characters')

asyncio.run(test_concurrent_analysis())
"
```

## 📝 Test Documentation

### Generate Test Reports
```bash
# Create comprehensive test log
docker-compose exec archangel bash -c "
echo '=== Archangel System Test Report ===' > /app/logs/test_report.txt
echo 'Date: $(date)' >> /app/logs/test_report.txt
echo '' >> /app/logs/test_report.txt

echo '=== System Information ===' >> /app/logs/test_report.txt
uname -a >> /app/logs/test_report.txt
python3 --version >> /app/logs/test_report.txt
pip list | grep -E '(transformers|torch|huggingface)' >> /app/logs/test_report.txt
echo '' >> /app/logs/test_report.txt

echo '=== Available Memory ===' >> /app/logs/test_report.txt
free -h >> /app/logs/test_report.txt
echo '' >> /app/logs/test_report.txt

echo '=== Test Results ===' >> /app/logs/test_report.txt
python3 demo_archangel.py >> /app/logs/test_report.txt 2>&1
echo '' >> /app/logs/test_report.txt

cat /app/logs/test_report.txt
"
```

## 🎯 Success Criteria

A successful Docker testing environment should demonstrate:

- ✅ **Basic AI Functionality**: Cloud and local AI models respond correctly
- ✅ **Kernel Integration**: Module compiles, loads, and communicates with userspace
- ✅ **Tool Orchestration**: AI successfully controls security tools
- ✅ **Educational Features**: AI explains reasoning and teaches security concepts
- ✅ **Performance**: Sub-second response times for simple queries
- ✅ **Stability**: System runs without crashes or memory leaks
- ✅ **Security**: Maintains defensive posture in all operations

## 🚀 Next Steps

After successful Docker testing:

1. **Production Deployment**: Adapt containers for production use
2. **CI/CD Integration**: Set up automated testing pipelines
3. **Scaling**: Test with container orchestration (Kubernetes)
4. **Monitoring**: Add comprehensive logging and metrics
5. **Security Hardening**: Implement additional security measures

---

**🛡️ Happy Testing with Archangel AI Security Expert System!**

*Remember: This system is designed for defensive security research and education. Always maintain ethical boundaries in your testing.*