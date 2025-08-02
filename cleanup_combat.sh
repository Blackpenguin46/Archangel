#!/bin/bash
# Archangel Combat System Cleanup
# Removes any existing containers and networks

echo "🧹 Cleaning up existing Archangel combat containers..."

# Stop and remove any existing containers
echo "Stopping containers..."
docker stop $(docker ps -q --filter "name=archangel-") 2>/dev/null || true
docker rm $(docker ps -aq --filter "name=archangel-") 2>/dev/null || true

# Remove combat network
echo "Removing combat network..."
docker network rm archangel-combat-net 2>/dev/null || true

# Clean up any dangling volumes
echo "Cleaning up volumes..."
docker volume prune -f 2>/dev/null || true

# Force cleanup any conflicting networks
echo "Cleaning up conflicting networks..."
docker network prune -f 2>/dev/null || true

# Kill any processes using the IP range
echo "Checking for IP conflicts..."
sudo pkill -f "192.168.100" 2>/dev/null || true

echo "✅ Cleanup complete - ready for fresh combat deployment"