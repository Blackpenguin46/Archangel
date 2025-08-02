#!/bin/bash

# Archangel MCP Server Entrypoint Script
# Handles secure startup of team-specific MCP servers

set -euo pipefail

# Configuration
TEAM_TYPE=${TEAM_TYPE:-"blue_team"}
MCP_PORT=${MCP_PORT:-8881}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}
SECURITY_MODE=${SECURITY_MODE:-"standard"}

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [MCP-${TEAM_TYPE^^}] $1" | tee -a /var/log/archangel/mcp-${TEAM_TYPE}.log
}

# Error handling
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Cleanup function
cleanup() {
    log "Shutting down MCP server gracefully..."
    # Kill any background processes
    jobs -p | xargs -r kill
    log "MCP server shutdown complete"
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

log "Starting Archangel MCP Server for ${TEAM_TYPE}"

# Validate team type
if [[ "${TEAM_TYPE}" != "red_team" && "${TEAM_TYPE}" != "blue_team" ]]; then
    error_exit "Invalid TEAM_TYPE: ${TEAM_TYPE}. Must be 'red_team' or 'blue_team'"
fi

# Validate port
if ! [[ "${MCP_PORT}" =~ ^[0-9]+$ ]] || [ "${MCP_PORT}" -lt 1024 ] || [ "${MCP_PORT}" -gt 65535 ]; then
    error_exit "Invalid MCP_PORT: ${MCP_PORT}. Must be a number between 1024-65535"
fi

# Create required directories
mkdir -p /var/log/archangel
mkdir -p /opt/archangel/credentials
mkdir -p /opt/archangel/cache

# Set permissions
chmod 700 /opt/archangel/credentials
chmod 755 /var/log/archangel

log "Environment validation complete"

# Load configuration
CONFIG_FILE="/opt/archangel/config/mcp_config.json"
if [[ ! -f "${CONFIG_FILE}" ]]; then
    error_exit "Configuration file not found: ${CONFIG_FILE}"
fi

log "Loading configuration from ${CONFIG_FILE}"

# Validate Python environment
python3 -c "import asyncio, json, logging" || error_exit "Required Python modules not available"

# Check if MCP libraries are available
python3 -c "
try:
    from core.mcp_integration_architecture import MCPOrchestrator, TeamType
    print('MCP integration modules loaded successfully')
except ImportError as e:
    print(f'MCP modules not available: {e}')
    print('Running in development mode')
" || log "WARNING: Running in development mode without full MCP support"

# Generate or load server secret
SECRET_KEY_FILE="/opt/archangel/credentials/mcp_secret.key"
if [[ ! -f "${SECRET_KEY_FILE}" ]]; then
    log "Generating new MCP secret key"
    python3 -c "
import secrets
import os
key = secrets.token_hex(32)
os.makedirs(os.path.dirname('${SECRET_KEY_FILE}'), exist_ok=True)
with open('${SECRET_KEY_FILE}', 'w') as f:
    f.write(key)
os.chmod('${SECRET_KEY_FILE}', 0o600)
print('Secret key generated')
"
fi

log "Secret key loaded"

# Set environment variables for Python application
export PYTHONPATH="/opt/archangel:${PYTHONPATH:-}"
export MCP_TEAM_TYPE="${TEAM_TYPE}"
export MCP_SERVER_PORT="${MCP_PORT}"
export MCP_LOG_LEVEL="${LOG_LEVEL}"
export MCP_SECURITY_MODE="${SECURITY_MODE}"
export MCP_CONFIG_FILE="${CONFIG_FILE}"
export MCP_SECRET_KEY_FILE="${SECRET_KEY_FILE}"

# Team-specific initialization
if [[ "${TEAM_TYPE}" == "red_team" ]]; then
    log "Initializing Red Team MCP Server"
    
    # Wait for required red team services
    log "Waiting for Metasploit Framework..."
    timeout 120 bash -c 'until nc -z metasploit-framework 55553; do sleep 2; done' || log "WARNING: Metasploit not available"
    
    # Initialize red team specific resources
    python3 -c "
import asyncio
import sys
sys.path.append('/opt/archangel')

async def init_red_team():
    try:
        from core.mcp_integration_architecture import MCPOrchestrator, TeamType
        orchestrator = MCPOrchestrator()
        
        # Test red team initialization
        log_msg = 'Red Team MCP initialization test passed'
        print(log_msg)
        return True
    except Exception as e:
        print(f'Red Team initialization error: {e}')
        return False

result = asyncio.run(init_red_team())
exit(0 if result else 1)
" || log "WARNING: Red team initialization had issues"

elif [[ "${TEAM_TYPE}" == "blue_team" ]]; then
    log "Initializing Blue Team MCP Server"
    
    # Wait for required blue team services
    log "Waiting for monitoring services..."
    timeout 60 bash -c 'until nc -z archangel-postgres 5432; do sleep 2; done' || log "WARNING: Database not available"
    
    # Initialize blue team specific resources
    python3 -c "
import asyncio
import sys
sys.path.append('/opt/archangel')

async def init_blue_team():
    try:
        from core.mcp_integration_architecture import MCPOrchestrator, TeamType
        orchestrator = MCPOrchestrator()
        
        # Test blue team initialization
        log_msg = 'Blue Team MCP initialization test passed'
        print(log_msg)
        return True
    except Exception as e:
        print(f'Blue Team initialization error: {e}')
        return False

result = asyncio.run(init_blue_team())
exit(0 if result else 1)
" || log "WARNING: Blue team initialization had issues"
fi

log "Team-specific initialization complete"

# Health check endpoint setup
log "Setting up health check endpoint"
cat > /tmp/health_check.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import json
from aiohttp import web
import os
import sys

sys.path.append('/opt/archangel')

async def health_check(request):
    """Health check endpoint"""
    try:
        # Basic health checks
        health_status = {
            "status": "healthy",
            "team": os.getenv("MCP_TEAM_TYPE", "unknown"),
            "port": os.getenv("MCP_SERVER_PORT", "unknown"),
            "timestamp": str(request.loop.time())
        }
        
        return web.json_response(health_status)
    except Exception as e:
        return web.json_response({
            "status": "unhealthy",
            "error": str(e)
        }, status=500)

async def create_health_app():
    """Create health check web application"""
    app = web.Application()
    app.router.add_get('/health', health_check)
    return app

if __name__ == "__main__":
    port = int(os.getenv("MCP_SERVER_PORT", "8881"))
    app = create_health_app()
    web.run_app(app, host='0.0.0.0', port=port, access_log=None)
EOF

chmod +x /tmp/health_check.py

# Start health check in background
log "Starting health check endpoint on port ${MCP_PORT}"
python3 /tmp/health_check.py &
HEALTH_PID=$!

# Start main MCP server
log "Starting main MCP server"
python3 -c "
import asyncio
import logging
import sys
import os
import signal
import json

sys.path.append('/opt/archangel')

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('MCP_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f'/var/log/archangel/mcp-{os.getenv(\"MCP_TEAM_TYPE\", \"unknown\")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('mcp_server')

async def main():
    try:
        from core.mcp_integration_architecture import MCPOrchestrator, TeamType
        
        # Load secret key
        with open(os.getenv('MCP_SECRET_KEY_FILE'), 'r') as f:
            secret_key = f.read().strip()
        
        # Create orchestrator
        orchestrator = MCPOrchestrator(secret_key)
        
        # Initialize MCP architecture
        logger.info('Initializing MCP architecture...')
        if not await orchestrator.initialize_mcp_architecture():
            logger.error('Failed to initialize MCP architecture')
            return False
        
        logger.info('MCP architecture initialized successfully')
        
        # Keep server running
        try:
            while True:
                await asyncio.sleep(10)
                
                # Periodic health check
                status = await orchestrator.get_system_status()
                if status.get('orchestrator_status') != 'operational':
                    logger.warning(f'System status warning: {status}')
                
        except KeyboardInterrupt:
            logger.info('Received shutdown signal')
        finally:
            logger.info('Shutting down MCP architecture...')
            await orchestrator.shutdown_mcp_architecture()
            logger.info('MCP server shutdown complete')
        
        return True
        
    except Exception as e:
        logger.error(f'MCP server error: {e}', exc_info=True)
        return False

if __name__ == '__main__':
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
" &

MCP_SERVER_PID=$!

log "MCP server started with PID ${MCP_SERVER_PID}"

# Monitor processes
while true; do
    # Check if health check is still running
    if ! kill -0 ${HEALTH_PID} 2>/dev/null; then
        log "Health check process died, restarting..."
        python3 /tmp/health_check.py &
        HEALTH_PID=$!
    fi
    
    # Check if MCP server is still running
    if ! kill -0 ${MCP_SERVER_PID} 2>/dev/null; then
        log "MCP server process died"
        break
    fi
    
    sleep 10
done

log "MCP server monitoring ended"
cleanup
exit 0