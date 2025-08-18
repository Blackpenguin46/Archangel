import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  IconButton,
  Collapse,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Error as ErrorIcon,
  CheckCircle as SuccessIcon,
  SmartToy as AgentIcon,
  Security as SecurityIcon,
} from '@mui/icons-material';
import { useWebSocketEvent } from '../../../contexts/WebSocketContext';
import moment from 'moment';

interface ActivityItem {
  id: string;
  type: 'agent' | 'scenario' | 'system' | 'alert';
  level: 'info' | 'success' | 'warning' | 'error';
  title: string;
  description: string;
  timestamp: Date;
  details?: any;
}

const ActivityFeed: React.FC = () => {
  const [activities, setActivities] = useState<ActivityItem[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);

  // Subscribe to real-time events
  useWebSocketEvent('system_event', (event) => {
    const activity: ActivityItem = {
      id: `system-${Date.now()}`,
      type: 'system',
      level: event.data.level || 'info',
      title: `System Event: ${event.type}`,
      description: event.data.message || 'System event occurred',
      timestamp: new Date(event.timestamp),
      details: event.data,
    };
    addActivity(activity);
  });

  useWebSocketEvent('agent_event', (event) => {
    const activity: ActivityItem = {
      id: `agent-${Date.now()}`,
      type: 'agent',
      level: event.event === 'error' ? 'error' : 
             event.event === 'task_completed' ? 'success' : 'info',
      title: `Agent ${event.agentId}`,
      description: `${event.event.replace('_', ' ')} - ${event.data.message || ''}`,
      timestamp: new Date(event.timestamp),
      details: event.data,
    };
    addActivity(activity);
  });

  useWebSocketEvent('scenario_event', (event) => {
    const activity: ActivityItem = {
      id: `scenario-${Date.now()}`,
      type: 'scenario',
      level: event.event === 'completed' ? 'success' : 'info',
      title: `Scenario ${event.scenarioId}`,
      description: `${event.event.replace('_', ' ')} - ${event.data.name || ''}`,
      timestamp: new Date(event.timestamp),
      details: event.data,
    };
    addActivity(activity);
  });

  const addActivity = (activity: ActivityItem) => {
    setActivities(prev => [activity, ...prev.slice(0, 49)]); // Keep last 50 activities
  };

  const handleExpandClick = (id: string) => {
    setExpanded(expanded === id ? null : id);
  };

  const getIcon = (activity: ActivityItem) => {
    switch (activity.type) {
      case 'agent':
        return <AgentIcon />;
      case 'scenario':
        switch (activity.level) {
          case 'success':
            return <SuccessIcon />;
          case 'error':
            return <ErrorIcon />;
          default:
            return <PlayIcon />;
        }
      case 'system':
        return <SecurityIcon />;
      case 'alert':
        switch (activity.level) {
          case 'error':
            return <ErrorIcon />;
          case 'warning':
            return <WarningIcon />;
          default:
            return <InfoIcon />;
        }
      default:
        return <InfoIcon />;
    }
  };

  const getIconColor = (level: string) => {
    switch (level) {
      case 'success':
        return '#4caf50';
      case 'warning':
        return '#ff9800';
      case 'error':
        return '#f44336';
      default:
        return '#2196f3';
    }
  };

  // Add some initial mock activities
  useEffect(() => {
    const mockActivities: ActivityItem[] = [
      {
        id: 'mock-1',
        type: 'system',
        level: 'info',
        title: 'System Initialized',
        description: 'Archangel system started successfully',
        timestamp: new Date(Date.now() - 300000), // 5 minutes ago
      },
      {
        id: 'mock-2',
        type: 'agent',
        level: 'success',
        title: 'Agent Red Team Alpha',
        description: 'Task completed successfully',
        timestamp: new Date(Date.now() - 180000), // 3 minutes ago
      },
      {
        id: 'mock-3',
        type: 'scenario',
        level: 'info',
        title: 'Scenario APT-001',
        description: 'Started advanced persistent threat simulation',
        timestamp: new Date(Date.now() - 120000), // 2 minutes ago
      },
    ];

    setActivities(mockActivities);
  }, []);

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Activity Feed
        </Typography>
        
        <List sx={{ maxHeight: 400, overflow: 'auto' }}>
          {activities.map((activity) => (
            <Box key={activity.id}>
              <ListItem
                sx={{
                  borderLeft: `3px solid ${getIconColor(activity.level)}`,
                  borderRadius: '4px',
                  mb: 1,
                  backgroundColor: `${getIconColor(activity.level)}08`,
                  transition: 'background-color 0.2s',
                  '&:hover': {
                    backgroundColor: `${getIconColor(activity.level)}15`,
                  },
                }}
              >
                <ListItemIcon sx={{ color: getIconColor(activity.level) }}>
                  {getIcon(activity)}
                </ListItemIcon>
                
                <ListItemText
                  primary={
                    <Box display="flex" alignItems="center" gap={1}>
                      <Typography variant="subtitle2" noWrap>
                        {activity.title}
                      </Typography>
                      <Chip
                        label={activity.type}
                        size="small"
                        variant="outlined"
                        sx={{ 
                          fontSize: '0.7rem',
                          height: 20,
                          color: getIconColor(activity.level),
                          borderColor: getIconColor(activity.level),
                        }}
                      />
                    </Box>
                  }
                  secondary={
                    <Box>
                      <Typography variant="body2" color="text.secondary" noWrap>
                        {activity.description}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {moment(activity.timestamp).fromNow()}
                      </Typography>
                    </Box>
                  }
                />
                
                {activity.details && (
                  <IconButton
                    size="small"
                    onClick={() => handleExpandClick(activity.id)}
                    sx={{
                      transform: expanded === activity.id ? 'rotate(180deg)' : 'rotate(0deg)',
                      transition: 'transform 0.2s',
                    }}
                  >
                    <ExpandMoreIcon />
                  </IconButton>
                )}
              </ListItem>
              
              {activity.details && (
                <Collapse in={expanded === activity.id}>
                  <Box 
                    sx={{ 
                      ml: 4, 
                      mb: 2, 
                      p: 2, 
                      borderRadius: 1,
                      backgroundColor: 'rgba(255,255,255,0.02)',
                      border: '1px solid rgba(255,255,255,0.1)',
                    }}
                  >
                    <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
                      Details:
                    </Typography>
                    <Box
                      component="pre"
                      sx={{
                        fontSize: '0.75rem',
                        fontFamily: 'monospace',
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                        color: 'text.secondary',
                        maxHeight: 150,
                        overflow: 'auto',
                      }}
                    >
                      {JSON.stringify(activity.details, null, 2)}
                    </Box>
                  </Box>
                </Collapse>
              )}
            </Box>
          ))}
          
          {activities.length === 0 && (
            <ListItem>
              <ListItemText
                primary="No recent activity"
                secondary="System activities will appear here"
                sx={{ textAlign: 'center' }}
              />
            </ListItem>
          )}
        </List>
      </CardContent>
    </Card>
  );
};

export default ActivityFeed;