#!/bin/bash

echo "🚀 Starting Archangel AI vs AI Container Environment"
echo "=" * 60

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Create network for AI vs AI combat
echo "🌐 Creating container network..."
docker network create archangel-combat 2>/dev/null || echo "   Network already exists"

# Start Red Team Kali Linux container
echo "🔴 Starting Red Team (Kali Linux)..."
docker run -d \
    --name archangel-red-team \
    --network archangel-combat \
    --privileged \
    --cap-add=NET_ADMIN \
    --cap-add=NET_RAW \
    -v $(pwd)/logs:/logs \
    kalilinux/kali-rolling \
    /bin/bash -c "
        apt update && 
        apt install -y python3 python3-pip nmap curl netcat-openbsd && 
        pip3 install requests psutil --break-system-packages &&
        echo 'Red Team container ready' > /logs/red_team_status.log &&
        tail -f /dev/null
    " || echo "   Red team container already running"

# Start Blue Team Ubuntu SOC container  
echo "🔵 Starting Blue Team (Ubuntu SOC)..."
docker run -d \
    --name archangel-blue-team \
    --network archangel-combat \
    --privileged \
    -v $(pwd)/logs:/logs \
    ubuntu:22.04 \
    /bin/bash -c "
        apt update && 
        apt install -y python3 python3-pip tcpdump iproute2 iptables net-tools &&
        pip3 install requests psutil &&
        echo 'Blue Team container ready' > /logs/blue_team_status.log &&
        tail -f /dev/null
    " || echo "   Blue team container already running"

# Start Target Enterprise Environment
echo "🎯 Starting Target Enterprise Environment..."
docker run -d \
    --name target-enterprise \
    --network archangel-combat \
    -p 8080:80 \
    nginx:alpine || echo "   Target enterprise already running"

echo ""
echo "⏳ Waiting for containers to initialize..."
sleep 15

# Deploy AI agents
echo "🤖 Deploying AI agents..."
mkdir -p logs

# Copy agent scripts to containers
docker exec archangel-red-team mkdir -p /app 2>/dev/null || true
docker exec archangel-blue-team mkdir -p /app 2>/dev/null || true

if [ -f "container_red_team_agent.py" ]; then
    docker cp container_red_team_agent.py archangel-red-team:/app/red_team_agent.py
    echo "   ✅ Red Team AI agent deployed"
else
    echo "   ⚠️ Red Team agent script not found"
fi

if [ -f "container_blue_team_agent.py" ]; then
    docker cp container_blue_team_agent.py archangel-blue-team:/app/blue_team_agent.py  
    echo "   ✅ Blue Team AI agent deployed"
else
    echo "   ⚠️ Blue Team agent script not found"
fi

# Start AI agents
echo "🧠 Starting AI agents..."
docker exec -d archangel-red-team python3 /app/red_team_agent.py 2>/dev/null || echo "   Red team agent starting..."
docker exec -d archangel-blue-team python3 /app/blue_team_agent.py 2>/dev/null || echo "   Blue team agent starting..."

echo ""
echo "✅ Archangel AI vs AI System Started!"
echo ""
echo "📊 Container Status:"
docker ps --filter "name=archangel-" --filter "name=target-enterprise" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""
echo "🌐 Network Information:"
docker network inspect archangel-combat --format '{{range .Containers}}{{.Name}}: {{.IPv4Address}}{{"\n"}}{{end}}'
echo ""
echo "🔗 Access Points:"
echo "   Web Interface: http://localhost:8080"
echo "   Red Team:  docker exec -it archangel-red-team /bin/bash"
echo "   Blue Team: docker exec -it archangel-blue-team /bin/bash"
echo ""
echo "📋 Commands:"
echo "   Status Report: python3 container_status_report.py"
echo "   Integration Test: python3 test_container_integration.py"
echo "   Stop System: docker stop archangel-red-team archangel-blue-team target-enterprise"
echo ""
echo "🎉 Archangel AI vs AI Container Environment Ready!"