#!/bin/bash

echo "🚀 Starting Archangel AI vs AI Container Environment"

# Create network for AI vs AI combat
docker network create archangel-combat 2>/dev/null || echo "Network already exists"

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
    "

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
        apt install -y python3 python3-pip tcpdump netstat iptables fail2ban &&
        pip3 install requests psutil &&
        echo 'Blue Team container ready' > /logs/blue_team_status.log &&
        tail -f /dev/null
    "

# Start Target Enterprise Environment
echo "🎯 Starting Target Enterprise Environment..."
docker run -d \
    --name target-enterprise \
    --network archangel-combat \
    -p 8080:80 \
    nginx:alpine

echo "✅ Container environment started!"
echo ""
echo "📊 Container Status:"
docker ps --filter "name=archangel-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""
echo "🌐 Network Information:"
docker network inspect archangel-combat --format '{{range .Containers}}{{.Name}}: {{.IPv4Address}}{{"\n"}}{{end}}'
echo ""
echo "🔗 Web Interface: http://localhost:8080"
echo "🐳 To access containers:"
echo "   Red Team:  docker exec -it archangel-red-team /bin/bash"
echo "   Blue Team: docker exec -it archangel-blue-team /bin/bash"