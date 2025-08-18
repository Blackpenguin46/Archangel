import React, { useRef, useEffect, useState } from 'react';
import { Box, Card, CardContent, Typography, ButtonGroup, Button, Chip } from '@mui/material';
import ForceGraph2D from 'react-force-graph-2d';
import { useSystem } from '../../contexts/SystemContext';
import { useWebSocketEvent } from '../../contexts/WebSocketContext';

interface NetworkNode {
  id: string;
  name: string;
  type: 'agent' | 'scenario' | 'system' | 'target';
  status: 'active' | 'idle' | 'offline' | 'error';
  group: number;
  size: number;
  color: string;
  x?: number;
  y?: number;
  fx?: number;
  fy?: number;
}

interface NetworkLink {
  source: string;
  target: string;
  type: 'communication' | 'attack' | 'defense' | 'monitoring';
  strength: number;
  color: string;
  width: number;
}

interface NetworkData {
  nodes: NetworkNode[];
  links: NetworkLink[];
}

const NetworkTopology: React.FC = () => {
  const { agents, scenarios, systemStatus } = useSystem();
  const [networkData, setNetworkData] = useState<NetworkData>({ nodes: [], links: [] });
  const [viewMode, setViewMode] = useState<'topology' | 'activity' | 'threats'>('topology');
  const [selectedNode, setSelectedNode] = useState<NetworkNode | null>(null);
  const forceRef = useRef<any>();

  // Generate network data from system state
  useEffect(() => {
    const nodes: NetworkNode[] = [];
    const links: NetworkLink[] = [];

    // Add system node (central hub)
    nodes.push({
      id: 'system',
      name: 'Archangel System',
      type: 'system',
      status: systemStatus.overall === 'healthy' ? 'active' : 'error',
      group: 0,
      size: 20,
      color: '#1976d2',
      fx: 0,
      fy: 0,
    });

    // Add agent nodes
    agents.forEach((agent, index) => {
      const angle = (index / agents.length) * 2 * Math.PI;
      const radius = 150;
      
      nodes.push({
        id: agent.id,
        name: agent.name,
        type: 'agent',
        status: agent.status,
        group: agent.type === 'red_team' ? 1 : agent.type === 'blue_team' ? 2 : 3,
        size: 15,
        color: getAgentColor(agent.type, agent.status),
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
      });

      // Link agent to system
      links.push({
        source: 'system',
        target: agent.id,
        type: 'monitoring',
        strength: agent.status === 'active' ? 3 : 1,
        color: agent.status === 'active' ? '#4caf50' : '#666',
        width: agent.status === 'active' ? 2 : 1,
      });
    });

    // Add scenario nodes and connections
    scenarios.forEach((scenario, index) => {
      if (scenario.status === 'running') {
        const angle = (index / scenarios.length) * 2 * Math.PI;
        const radius = 80;
        
        nodes.push({
          id: scenario.id,
          name: scenario.name,
          type: 'scenario',
          status: 'active',
          group: 4,
          size: 12,
          color: '#9c27b0',
          x: Math.cos(angle) * radius,
          y: Math.sin(angle) * radius,
        });

        // Link scenario to system
        links.push({
          source: 'system',
          target: scenario.id,
          type: 'communication',
          strength: 2,
          color: '#9c27b0',
          width: 1.5,
        });

        // Link participating agents to scenario
        scenario.participants.forEach(participantId => {
          const participant = agents.find(a => a.id === participantId);
          if (participant) {
            links.push({
              source: scenario.id,
              target: participantId,
              type: participant.type === 'red_team' ? 'attack' : 'defense',
              strength: 2,
              color: participant.type === 'red_team' ? '#f44336' : '#2196f3',
              width: 2,
            });
          }
        });
      }
    });

    // Add target nodes for red team scenarios
    const redTeamScenarios = scenarios.filter(s => 
      s.status === 'running' && 
      s.participants.some(p => agents.find(a => a.id === p)?.type === 'red_team')
    );

    redTeamScenarios.forEach((scenario, index) => {
      const targetId = `target-${scenario.id}`;
      const angle = ((index + 0.5) / redTeamScenarios.length) * 2 * Math.PI;
      const radius = 220;
      
      nodes.push({
        id: targetId,
        name: `Target System`,
        type: 'target',
        status: 'active',
        group: 5,
        size: 10,
        color: '#ff9800',
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
      });

      // Link red team agents to targets
      scenario.participants.forEach(participantId => {
        const participant = agents.find(a => a.id === participantId);
        if (participant?.type === 'red_team') {
          links.push({
            source: participantId,
            target: targetId,
            type: 'attack',
            strength: 1,
            color: '#f44336',
            width: 1,
          });
        }
      });
    });

    setNetworkData({ nodes, links });
  }, [agents, scenarios, systemStatus]);

  // Subscribe to real-time updates
  useWebSocketEvent('agent_event', (event) => {
    // Update node appearance based on agent activity
    setNetworkData(prev => ({
      ...prev,
      nodes: prev.nodes.map(node => {
        if (node.id === event.agentId && node.type === 'agent') {
          return {
            ...node,
            color: event.event === 'error' ? '#f44336' : 
                   event.event === 'task_completed' ? '#4caf50' : node.color,
            size: event.event === 'task_completed' ? 18 : 15,
          };
        }
        return node;
      }),
    }));

    // Reset size after animation
    if (event.event === 'task_completed') {
      setTimeout(() => {
        setNetworkData(prev => ({
          ...prev,
          nodes: prev.nodes.map(node => 
            node.id === event.agentId ? { ...node, size: 15 } : node
          ),
        }));
      }, 1000);
    }
  });

  const getAgentColor = (type: string, status: string) => {
    if (status === 'error') return '#f44336';
    if (status === 'offline') return '#666';
    
    switch (type) {
      case 'red_team':
        return '#e53935';
      case 'blue_team':
        return '#1976d2';
      case 'purple_team':
        return '#7b1fa2';
      default:
        return '#757575';
    }
  };

  const handleNodeClick = (node: any) => {
    setSelectedNode(node);
  };

  const handleNodeHover = (node: any) => {
    // Optional: Add hover effects
  };

  const paintNode = (node: any, ctx: CanvasRenderingContext2D) => {
    const size = node.size || 8;
    
    // Draw node circle
    ctx.beginPath();
    ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
    ctx.fillStyle = node.color;
    ctx.fill();
    
    // Add status indicator
    if (node.status === 'active') {
      ctx.beginPath();
      ctx.arc(node.x + size * 0.7, node.y - size * 0.7, size * 0.3, 0, 2 * Math.PI);
      ctx.fillStyle = '#4caf50';
      ctx.fill();
    } else if (node.status === 'error') {
      ctx.beginPath();
      ctx.arc(node.x + size * 0.7, node.y - size * 0.7, size * 0.3, 0, 2 * Math.PI);
      ctx.fillStyle = '#f44336';
      ctx.fill();
    }
    
    // Draw node label
    ctx.font = '10px Arial';
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'center';
    ctx.fillText(node.name, node.x, node.y + size + 15);
  };

  const paintLink = (link: any, ctx: CanvasRenderingContext2D) => {
    const width = link.width || 1;
    
    // Draw animated links for active connections
    if (link.type === 'attack' || link.type === 'defense') {
      const gradient = ctx.createLinearGradient(
        link.source.x, link.source.y,
        link.target.x, link.target.y
      );
      gradient.addColorStop(0, link.color);
      gradient.addColorStop(0.5, `${link.color}80`);
      gradient.addColorStop(1, link.color);
      
      ctx.strokeStyle = gradient;
      ctx.lineWidth = width;
      ctx.setLineDash([5, 5]);
      ctx.lineDashOffset = Date.now() * 0.01;
    } else {
      ctx.strokeStyle = link.color;
      ctx.lineWidth = width;
      ctx.setLineDash([]);
    }
    
    ctx.beginPath();
    ctx.moveTo(link.source.x, link.source.y);
    ctx.lineTo(link.target.x, link.target.y);
    ctx.stroke();
  };

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">
            Network Topology
          </Typography>
          
          <ButtonGroup size="small">
            <Button
              variant={viewMode === 'topology' ? 'contained' : 'outlined'}
              onClick={() => setViewMode('topology')}
            >
              Topology
            </Button>
            <Button
              variant={viewMode === 'activity' ? 'contained' : 'outlined'}
              onClick={() => setViewMode('activity')}
            >
              Activity
            </Button>
            <Button
              variant={viewMode === 'threats' ? 'contained' : 'outlined'}
              onClick={() => setViewMode('threats')}
            >
              Threats
            </Button>
          </ButtonGroup>
        </Box>

        {/* Legend */}
        <Box display="flex" gap={1} mb={2} flexWrap="wrap">
          <Chip label="Red Team" size="small" sx={{ bgcolor: '#e53935', color: 'white' }} />
          <Chip label="Blue Team" size="small" sx={{ bgcolor: '#1976d2', color: 'white' }} />
          <Chip label="Purple Team" size="small" sx={{ bgcolor: '#7b1fa2', color: 'white' }} />
          <Chip label="System" size="small" sx={{ bgcolor: '#1976d2', color: 'white' }} />
          <Chip label="Target" size="small" sx={{ bgcolor: '#ff9800', color: 'white' }} />
        </Box>

        <Box className="network-graph-container" position="relative">
          <ForceGraph2D
            ref={forceRef}
            graphData={networkData}
            nodeCanvasObject={paintNode}
            linkCanvasObject={paintLink}
            onNodeClick={handleNodeClick}
            onNodeHover={handleNodeHover}
            width={800}
            height={500}
            backgroundColor="#1a1a2e"
            linkDirectionalParticles={2}
            linkDirectionalParticleSpeed={0.006}
            linkDirectionalParticleWidth={2}
            cooldownTicks={100}
            d3AlphaDecay={0.01}
            d3VelocityDecay={0.3}
            enableZoomPanInteraction={true}
            enableNodeDrag={true}
          />
          
          {selectedNode && (
            <Box
              position="absolute"
              top={16}
              right={16}
              bgcolor="rgba(26, 26, 46, 0.95)"
              border="1px solid #333"
              borderRadius={1}
              p={2}
              minWidth={200}
            >
              <Typography variant="subtitle1" gutterBottom>
                {selectedNode.name}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Type: {selectedNode.type}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Status: {selectedNode.status}
              </Typography>
              {selectedNode.type === 'agent' && (
                <Typography variant="body2" color="text.secondary">
                  Group: {selectedNode.group === 1 ? 'Red Team' : 
                          selectedNode.group === 2 ? 'Blue Team' : 'Purple Team'}
                </Typography>
              )}
            </Box>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default NetworkTopology;