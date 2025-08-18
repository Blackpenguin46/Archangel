"""
Control Plane API for Agent Coordination and Management

This module provides REST and gRPC APIs for managing agents,
coordinating multi-agent activities, and monitoring control plane operations.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4
from dataclasses import asdict
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager

from .control_plane import (
    get_control_plane,
    shutdown_control_plane,
    ControlPlaneOrchestrator,
    DecisionEngine,
    CoordinationManager,
    AgentDecision,
    DecisionType,
    CoordinationRequest
)
from .plane_coordinator import (
    get_plane_coordinator,
    shutdown_plane_coordinator,
    AgentPlaneAdapter
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class AgentRegistrationRequest(BaseModel):
    agent_id: str = Field(..., description="Unique identifier for the agent")
    agent_type: str = Field(..., description="Type of agent (red_team, blue_team, etc.)")
    capabilities: List[str] = Field(default=[], description="List of agent capabilities")
    metadata: Dict[str, Any] = Field(default={}, description="Additional agent metadata")


class AgentRegistrationResponse(BaseModel):
    success: bool
    agent_id: str
    message: str


class DecisionRequest(BaseModel):
    agent_id: str = Field(..., description="ID of the agent making the decision")
    decision_type: str = Field(..., description="Type of decision (tactical, strategic, etc.)")
    context: Dict[str, Any] = Field(..., description="Context for decision making")
    constraints: List[str] = Field(default=[], description="Constraints on the decision")
    timeout_seconds: int = Field(default=30, description="Timeout for decision making")


class DecisionResponse(BaseModel):
    success: bool
    decision_id: Optional[str] = None
    action: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    error_message: Optional[str] = None


class CoordinationRequestModel(BaseModel):
    initiator_agent_id: str = Field(..., description="Agent initiating coordination")
    target_agents: List[str] = Field(..., description="Target agents for coordination")
    coordination_type: str = Field(..., description="Type of coordination required")
    payload: Dict[str, Any] = Field(..., description="Coordination payload")
    priority: int = Field(default=5, description="Priority level (1-10)")
    timeout_seconds: int = Field(default=30, description="Timeout for coordination")


class CoordinationResponse(BaseModel):
    success: bool
    request_id: Optional[str] = None
    message: str


class AgentStatusResponse(BaseModel):
    agent_id: str
    status: str
    metrics: Dict[str, Any]
    last_decision: Optional[Dict[str, Any]] = None


class SystemMetricsResponse(BaseModel):
    control_plane_status: str
    active_agents: int
    total_decisions: int
    decisions_per_second: float
    coordination_success_rate: float
    uptime_seconds: float


class EnvironmentQueryRequest(BaseModel):
    agent_id: str = Field(..., description="Agent making the query")
    query_type: str = Field(..., description="Type of environment query")
    parameters: Dict[str, Any] = Field(default={}, description="Query parameters")


class EnvironmentModificationRequest(BaseModel):
    agent_id: str = Field(..., description="Agent making the modification")
    modification_type: str = Field(..., description="Type of modification")
    parameters: Dict[str, Any] = Field(..., description="Modification parameters")


class EnvironmentResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


# Global state
control_plane: Optional[ControlPlaneOrchestrator] = None
plane_coordinator = None
registered_agents: Dict[str, AgentPlaneAdapter] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global control_plane, plane_coordinator
    
    # Startup
    logger.info("Starting Control Plane API")
    try:
        control_plane = await get_control_plane()
        plane_coordinator = await get_plane_coordinator()
        logger.info("Control Plane API started successfully")
    except Exception as e:
        logger.error(f"Failed to start Control Plane API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Control Plane API")
    try:
        await shutdown_control_plane()
        await shutdown_plane_coordinator()
        logger.info("Control Plane API shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="Archangel Control Plane API",
    description="API for managing autonomous agents and multi-agent coordination",
    version="1.0.0",
    lifespan=lifespan
)


async def get_control_plane_instance() -> ControlPlaneOrchestrator:
    """Dependency to get control plane instance"""
    global control_plane
    if control_plane is None:
        raise HTTPException(status_code=503, detail="Control plane not initialized")
    return control_plane


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/agents/register", response_model=AgentRegistrationResponse)
async def register_agent(
    request: AgentRegistrationRequest,
    cp: ControlPlaneOrchestrator = Depends(get_control_plane_instance)
):
    """Register a new agent with the control plane"""
    try:
        # Register agent with control plane
        decision_engine = cp.register_agent(request.agent_id)
        
        # Initialize decision engine
        await decision_engine.initialize()
        
        # Create plane adapter for the agent
        adapter = AgentPlaneAdapter(request.agent_id)
        await adapter.initialize()
        
        # Store adapter
        global registered_agents
        registered_agents[request.agent_id] = adapter
        
        logger.info(f"Agent {request.agent_id} registered successfully")
        
        return AgentRegistrationResponse(
            success=True,
            agent_id=request.agent_id,
            message="Agent registered successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to register agent {request.agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/agents/{agent_id}")
async def unregister_agent(
    agent_id: str,
    cp: ControlPlaneOrchestrator = Depends(get_control_plane_instance)
):
    """Unregister an agent from the control plane"""
    try:
        # Unregister from control plane
        success = cp.unregister_agent(agent_id)
        
        # Remove adapter
        global registered_agents
        if agent_id in registered_agents:
            del registered_agents[agent_id]
        
        if success:
            logger.info(f"Agent {agent_id} unregistered successfully")
            return {"success": True, "message": "Agent unregistered successfully"}
        else:
            raise HTTPException(status_code=404, detail="Agent not found")
            
    except Exception as e:
        logger.error(f"Failed to unregister agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/{agent_id}/decisions", response_model=DecisionResponse)
async def make_decision(
    agent_id: str,
    request: DecisionRequest,
    cp: ControlPlaneOrchestrator = Depends(get_control_plane_instance)
):
    """Make a decision for a specific agent"""
    try:
        # Get decision engine for agent
        decision_engine = cp.get_agent_decision_engine(agent_id)
        if not decision_engine:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Convert decision type
        try:
            decision_type = DecisionType(request.decision_type.upper())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid decision type: {request.decision_type}")
        
        # Make decision
        decision = await decision_engine.make_decision(
            decision_type=decision_type,
            context=request.context,
            constraints=request.constraints
        )
        
        return DecisionResponse(
            success=True,
            decision_id=decision.decision_id,
            action=decision.action,
            parameters=decision.parameters,
            reasoning=decision.reasoning,
            confidence=decision.confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to make decision for agent {agent_id}: {e}")
        return DecisionResponse(
            success=False,
            error_message=str(e)
        )


@app.get("/agents/{agent_id}/status", response_model=AgentStatusResponse)
async def get_agent_status(
    agent_id: str,
    cp: ControlPlaneOrchestrator = Depends(get_control_plane_instance)
):
    """Get status and metrics for a specific agent"""
    try:
        decision_engine = cp.get_agent_decision_engine(agent_id)
        if not decision_engine:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Get metrics and recent decision
        metrics = decision_engine.get_metrics()
        history = decision_engine.get_decision_history(limit=1)
        last_decision = history[0].to_dict() if history else None
        
        return AgentStatusResponse(
            agent_id=agent_id,
            status=decision_engine.get_status().value,
            metrics=metrics,
            last_decision=last_decision
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/{agent_id}/decisions")
async def get_agent_decisions(
    agent_id: str,
    limit: int = 50,
    cp: ControlPlaneOrchestrator = Depends(get_control_plane_instance)
):
    """Get decision history for a specific agent"""
    try:
        decision_engine = cp.get_agent_decision_engine(agent_id)
        if not decision_engine:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        history = decision_engine.get_decision_history(limit=limit)
        return {
            "agent_id": agent_id,
            "decisions": [decision.to_dict() for decision in history]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get decisions for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/coordination/request", response_model=CoordinationResponse)
async def request_coordination(
    request: CoordinationRequestModel,
    cp: ControlPlaneOrchestrator = Depends(get_control_plane_instance)
):
    """Request coordination between agents"""
    try:
        coordination_manager = cp.get_coordination_manager()
        
        # Create coordination request
        request_id = await coordination_manager.request_coordination(
            initiator_agent_id=request.initiator_agent_id,
            target_agents=request.target_agents,
            coordination_type=request.coordination_type,
            payload=request.payload,
            priority=request.priority,
            timeout=timedelta(seconds=request.timeout_seconds)
        )
        
        return CoordinationResponse(
            success=True,
            request_id=request_id,
            message="Coordination request created successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to create coordination request: {e}")
        return CoordinationResponse(
            success=False,
            message=str(e)
        )


@app.get("/coordination/{request_id}/status")
async def get_coordination_status(
    request_id: str,
    cp: ControlPlaneOrchestrator = Depends(get_control_plane_instance)
):
    """Get status of a coordination request"""
    try:
        coordination_manager = cp.get_coordination_manager()
        status = coordination_manager.get_coordination_status(request_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail="Coordination request not found")
        
        return {
            "request_id": request_id,
            "status": status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get coordination status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents")
async def list_agents(
    cp: ControlPlaneOrchestrator = Depends(get_control_plane_instance)
):
    """List all registered agents"""
    try:
        global registered_agents
        
        agents_info = []
        for agent_id in registered_agents.keys():
            decision_engine = cp.get_agent_decision_engine(agent_id)
            if decision_engine:
                metrics = decision_engine.get_metrics()
                agents_info.append({
                    "agent_id": agent_id,
                    "status": decision_engine.get_status().value,
                    "decisions_made": metrics.get("decisions_made", 0),
                    "average_latency": metrics.get("average_latency", 0.0)
                })
        
        return {
            "agents": agents_info,
            "total_count": len(agents_info)
        }
        
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics(
    cp: ControlPlaneOrchestrator = Depends(get_control_plane_instance)
):
    """Get overall system metrics"""
    try:
        metrics = cp.get_overall_metrics()
        
        return SystemMetricsResponse(
            control_plane_status=cp.status.value,
            active_agents=metrics.active_agents,
            total_decisions=metrics.total_decisions,
            decisions_per_second=metrics.decisions_per_second,
            coordination_success_rate=metrics.coordination_success_rate,
            uptime_seconds=metrics.uptime.total_seconds()
        )
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/environment/query", response_model=EnvironmentResponse)
async def query_environment(request: EnvironmentQueryRequest):
    """Query the environment state through an agent"""
    try:
        global registered_agents
        
        if request.agent_id not in registered_agents:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        adapter = registered_agents[request.agent_id]
        result = await adapter.query_environment(
            request.query_type,
            **request.parameters
        )
        
        return EnvironmentResponse(
            success=True,
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to query environment: {e}")
        return EnvironmentResponse(
            success=False,
            error_message=str(e)
        )


@app.post("/environment/modify", response_model=EnvironmentResponse)
async def modify_environment(request: EnvironmentModificationRequest):
    """Modify the environment state through an agent"""
    try:
        global registered_agents
        
        if request.agent_id not in registered_agents:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        adapter = registered_agents[request.agent_id]
        result = await adapter.modify_environment(
            request.modification_type,
            **request.parameters
        )
        
        return EnvironmentResponse(
            success=True,
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to modify environment: {e}")
        return EnvironmentResponse(
            success=False,
            error_message=str(e)
        )


@app.get("/environment/entities/{entity_id}")
async def get_entity(entity_id: str, agent_id: str):
    """Get a specific entity from the environment"""
    try:
        global registered_agents
        
        if agent_id not in registered_agents:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        adapter = registered_agents[agent_id]
        entity = await adapter.get_entity(entity_id)
        
        if entity is None:
            raise HTTPException(status_code=404, detail="Entity not found")
        
        return entity
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get entity {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/environment/entities")
async def find_entities(
    agent_id: str,
    entity_type: Optional[str] = None,
    limit: int = 100
):
    """Find entities in the environment"""
    try:
        global registered_agents
        
        if agent_id not in registered_agents:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        adapter = registered_agents[agent_id]
        entities = await adapter.find_entities(entity_type=entity_type)
        
        # Apply limit
        if len(entities) > limit:
            entities = entities[:limit]
        
        return {
            "entities": entities,
            "count": len(entities),
            "entity_type": entity_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to find entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time updates
@app.websocket("/ws/events")
async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time event streaming"""
    await websocket.accept()
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "connection_established",
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive and send periodic updates
        while True:
            await asyncio.sleep(5)
            
            # Send system metrics
            if control_plane:
                metrics = control_plane.get_overall_metrics()
                await websocket.send_json({
                    "type": "system_metrics",
                    "data": {
                        "active_agents": metrics.active_agents,
                        "total_decisions": metrics.total_decisions,
                        "decisions_per_second": metrics.decisions_per_second,
                        "coordination_success_rate": metrics.coordination_success_rate
                    },
                    "timestamp": datetime.now().isoformat()
                })
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    return app


def run_api_server(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """Run the API server"""
    uvicorn.run(
        "agents.control_plane_api:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug"
    )


if __name__ == "__main__":
    run_api_server(debug=True)