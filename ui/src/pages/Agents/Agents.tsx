import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  Avatar,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  LinearProgress,
  Tooltip,
  Alert,
  Paper,
  Tab,
  Tabs,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
} from '@mui/material';
import {
  SmartToy as AgentIcon,
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Pause as PauseIcon,
  Settings as SettingsIcon,
  Visibility as ViewIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Add as AddIcon,
  Refresh as RefreshIcon,
  TrendingUp as TrendingUpIcon,
  Speed as SpeedIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';
import { useSystem } from '../../contexts/SystemContext';
import { useWebSocketEvent } from '../../contexts/WebSocketContext';
import NetworkTopology from '../../components/NetworkTopology/NetworkTopology';
import moment from 'moment';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <div hidden={value !== index} style={{ paddingTop: 16 }}>
    {value === index && children}
  </div>
);

const Agents: React.FC = () => {
  const { agents, scenarios, loading, error } = useSystem();
  const [activeTab, setActiveTab] = useState(0);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [isConfigDialogOpen, setIsConfigDialogOpen] = useState(false);
  const [agentActivities, setAgentActivities] = useState<any[]>([]);
  const [agentMetrics, setAgentMetrics] = useState<Record<string, any>>({});

  // Subscribe to real-time agent events
  useWebSocketEvent('agent_event', (event) => {
    const activity = {
      id: Date.now(),
      agentId: event.agentId,
      event: event.event,
      message: event.data.message || '',
      timestamp: new Date(event.timestamp),
      level: event.event === 'error' ? 'error' : 
             event.event === 'task_completed' ? 'success' : 'info',
    };
    
    setAgentActivities(prev => [activity, ...prev.slice(0, 99)]); // Keep last 100 activities
    
    // Update agent metrics
    if (event.data.metrics) {
      setAgentMetrics(prev => ({
        ...prev,
        [event.agentId]: {
          ...prev[event.agentId],
          ...event.data.metrics,
          lastUpdate: new Date(),
        },
      }));
    }
  });

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const getAgentColor = (type: string, status: string) => {
    if (status === 'error') return '#f44336';
    if (status === 'offline') return '#666';
    
    switch (type) {
      case 'red_team': return '#e53935';
      case 'blue_team': return '#1976d2';
      case 'purple_team': return '#7b1fa2';
      default: return '#757575';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <SuccessIcon sx={{ color: '#4caf50' }} />;
      case 'error': return <ErrorIcon sx={{ color: '#f44336' }} />;
      case 'idle': return <PauseIcon sx={{ color: '#ff9800' }} />;
      case 'offline': return <StopIcon sx={{ color: '#666' }} />;
      default: return <WarningIcon sx={{ color: '#ff9800' }} />;
    }
  };

  const handleAgentAction = (agentId: string, action: string) => {
    console.log(`Agent ${agentId}: ${action}`);
    // In a real application, this would make API calls
  };

  const handleConfigureAgent = (agentId: string) => {
    setSelectedAgent(agentId);
    setIsConfigDialogOpen(true);
  };

  const activeAgents = agents.filter(a => a.status === 'active');
  const totalTasks = agents.reduce((sum, agent) => sum + (agent.performance?.tasksCompleted || 0), 0);
  const avgSuccessRate = agents.length > 0 
    ? agents.reduce((sum, agent) => sum + (agent.performance?.successRate || 0), 0) / agents.length 
    : 0;
  const avgResponseTime = agents.length > 0
    ? agents.reduce((sum, agent) => sum + (agent.performance?.responseTime || 0), 0) / agents.length
    : 0;

  return (
    <Box className="page-container fade-in">
      <Box mb={3}>
        <Typography variant="h4" component="h1" gutterBottom>
          Agent Management
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Monitor and manage AI agents, their performance, and configurations
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Agent Statistics */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={2}>
                <AgentIcon sx={{ color: '#1976d2', fontSize: 32 }} />
                <Box>
                  <Typography variant="h4" color="primary">
                    {agents.length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Agents
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={2}>
                <SuccessIcon sx={{ color: '#4caf50', fontSize: 32 }} />
                <Box>
                  <Typography variant="h4" sx={{ color: '#4caf50' }}>
                    {activeAgents.length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Active Agents
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={2}>
                <TrendingUpIcon sx={{ color: '#9c27b0', fontSize: 32 }} />
                <Box>
                  <Typography variant="h4" sx={{ color: '#9c27b0' }}>
                    {Math.round(avgSuccessRate)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Avg Success Rate
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={2}>
                <SpeedIcon sx={{ color: '#ff9800', fontSize: 32 }} />
                <Box>
                  <Typography variant="h4" sx={{ color: '#ff9800' }}>
                    {Math.round(avgResponseTime)}ms
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Avg Response Time
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tabbed Interface */}
      <Paper>
        <Tabs
          value={activeTab}
          onChange={handleTabChange}
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab label="Agent Overview" />
          <Tab label="Network Topology" />
          <Tab label="Activity Feed" />
          <Tab label="Performance Metrics" />
        </Tabs>

        <TabPanel value={activeTab} index={0}>
          {loading && <LinearProgress sx={{ mb: 2 }} />}
          
          <Grid container spacing={3} sx={{ p: 3 }}>
            {agents.map((agent) => (
              <Grid item xs={12} md={6} lg={4} key={agent.id}>
                <Card
                  sx={{
                    border: `1px solid ${getAgentColor(agent.type, agent.status)}30`,
                    transition: 'all 0.2s ease-in-out',
                    '&:hover': {
                      borderColor: getAgentColor(agent.type, agent.status),
                      boxShadow: `0 4px 20px ${getAgentColor(agent.type, agent.status)}30`,
                    },
                  }}
                >
                  <CardContent>
                    {/* Agent Header */}
                    <Box display="flex" alignItems="center" gap={2} mb={2}>
                      <Avatar
                        sx={{
                          bgcolor: getAgentColor(agent.type, agent.status),
                          width: 48,
                          height: 48,
                        }}
                      >
                        <AgentIcon />
                      </Avatar>
                      <Box flex={1}>
                        <Typography variant="h6" noWrap>
                          {agent.name}
                        </Typography>
                        <Box display="flex" alignItems="center" gap={1}>
                          {getStatusIcon(agent.status)}
                          <Typography variant="body2" color="text.secondary">
                            {agent.status}
                          </Typography>
                        </Box>
                      </Box>
                    </Box>

                    {/* Agent Type and Scenario */}
                    <Box display="flex" gap={1} mb={2}>
                      <Chip
                        label={agent.type.replace('_', ' ')}
                        size="small"
                        sx={{
                          bgcolor: `${getAgentColor(agent.type, agent.status)}20`,
                          color: getAgentColor(agent.type, agent.status),
                        }}
                      />
                      {agent.scenario && (
                        <Chip
                          label={`Scenario: ${agent.scenario}`}
                          size="small"
                          variant="outlined"
                        />
                      )}
                    </Box>

                    {/* Performance Metrics */}
                    {agent.performance && (
                      <Box>
                        <Typography variant="subtitle2" gutterBottom>
                          Performance
                        </Typography>
                        <Box display="flex" justifyContent="space-between" mb={1}>
                          <Typography variant="caption">Success Rate</Typography>
                          <Typography variant="caption">
                            {agent.performance.successRate}%
                          </Typography>
                        </Box>
                        <LinearProgress
                          variant="determinate"
                          value={agent.performance.successRate}
                          sx={{ mb: 1, height: 6, borderRadius: 3 }}
                        />
                        
                        <Box display="flex" justify-content="space-between" gap={2}>
                          <Box textAlign="center">
                            <Typography variant="caption" color="text.secondary">
                              Response Time
                            </Typography>
                            <Typography variant="body2">
                              {agent.performance.responseTime}ms
                            </Typography>
                          </Box>
                          <Box textAlign="center">
                            <Typography variant="caption" color="text.secondary">
                              Tasks Completed
                            </Typography>
                            <Typography variant="body2">
                              {agent.performance.tasksCompleted}
                            </Typography>
                          </Box>
                        </Box>
                      </Box>
                    )}

                    {/* Last Activity */}
                    <Box mt={2} pt={2} borderTop="1px solid #333">
                      <Typography variant="caption" color="text.secondary">
                        Last Activity: {moment(agent.lastActivity).fromNow()}
                      </Typography>
                    </Box>
                  </CardContent>

                  <CardActions sx={{ justifyContent: 'space-between' }}>
                    <Box>
                      {agent.status === 'active' ? (
                        <Button
                          size="small"
                          startIcon={<StopIcon />}
                          onClick={() => handleAgentAction(agent.id, 'stop')}
                          sx={{ color: '#f44336' }}
                        >
                          Stop
                        </Button>
                      ) : (
                        <Button
                          size="small"
                          startIcon={<StartIcon />}
                          onClick={() => handleAgentAction(agent.id, 'start')}
                          sx={{ color: '#4caf50' }}
                        >
                          Start
                        </Button>
                      )}
                    </Box>
                    
                    <Box>
                      <Tooltip title="View Details">
                        <IconButton
                          size="small"
                          onClick={() => handleAgentAction(agent.id, 'view')}
                        >
                          <ViewIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Configure">
                        <IconButton
                          size="small"
                          onClick={() => handleConfigureAgent(agent.id)}
                        >
                          <SettingsIcon />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </CardActions>
                </Card>
              </Grid>
            ))}

            {agents.length === 0 && (
              <Grid item xs={12}>
                <Card>
                  <CardContent sx={{ textAlign: 'center', py: 6 }}>
                    <AgentIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                    <Typography variant="h6" color="text.secondary" gutterBottom>
                      No agents found
                    </Typography>
                    <Typography variant="body2" color="text.secondary" mb={3}>
                      Agents will appear here once they are deployed and active.
                    </Typography>
                    <Button
                      variant="contained"
                      startIcon={<RefreshIcon />}
                      onClick={() => window.location.reload()}
                    >
                      Refresh
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
            )}
          </Grid>
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          <Box sx={{ p: 3 }}>
            <NetworkTopology />
          </Box>
        </TabPanel>

        <TabPanel value={activeTab} index={2}>
          <Box sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Real-time Agent Activity
            </Typography>
            <List sx={{ maxHeight: 500, overflow: 'auto' }}>
              {agentActivities.map((activity) => (
                <ListItem
                  key={activity.id}
                  sx={{
                    borderLeft: `3px solid ${
                      activity.level === 'error' ? '#f44336' :
                      activity.level === 'success' ? '#4caf50' : '#2196f3'
                    }`,
                    borderRadius: 1,
                    mb: 1,
                    bgcolor: `${
                      activity.level === 'error' ? '#f44336' :
                      activity.level === 'success' ? '#4caf50' : '#2196f3'
                    }08`,
                  }}
                >
                  <ListItemIcon>
                    {activity.level === 'error' ? <ErrorIcon sx={{ color: '#f44336' }} /> :
                     activity.level === 'success' ? <SuccessIcon sx={{ color: '#4caf50' }} /> :
                     <AgentIcon sx={{ color: '#2196f3' }} />}
                  </ListItemIcon>
                  <ListItemText
                    primary={`Agent ${activity.agentId}: ${activity.event.replace('_', ' ')}`}
                    secondary={
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          {activity.message}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {moment(activity.timestamp).fromNow()}
                        </Typography>
                      </Box>
                    }
                  />
                </ListItem>
              ))}
              
              {agentActivities.length === 0 && (
                <ListItem>
                  <ListItemText
                    primary="No recent activity"
                    secondary="Agent activities will appear here in real-time"
                    sx={{ textAlign: 'center' }}
                  />
                </ListItem>
              )}
            </List>
          </Box>
        </TabPanel>

        <TabPanel value={activeTab} index={3}>
          <Box sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Agent Performance Metrics
            </Typography>
            <Grid container spacing={3}>
              {agents.map((agent) => (
                <Grid item xs={12} md={6} key={agent.id}>
                  <Card>
                    <CardContent>
                      <Typography variant="subtitle1" gutterBottom>
                        {agent.name}
                      </Typography>
                      {agent.performance && (
                        <Box>
                          <Box display="flex" justifyContent="space-between" mb={1}>
                            <Typography variant="body2">Success Rate</Typography>
                            <Typography variant="body2">
                              {agent.performance.successRate}%
                            </Typography>
                          </Box>
                          <LinearProgress
                            variant="determinate"
                            value={agent.performance.successRate}
                            sx={{ mb: 2, height: 8, borderRadius: 4 }}
                          />
                          
                          <Box display="flex" justify-content="space-between" gap={2}>
                            <Box>
                              <Typography variant="caption" color="text.secondary">
                                Response Time
                              </Typography>
                              <Typography variant="h6">
                                {agent.performance.responseTime}ms
                              </Typography>
                            </Box>
                            <Box>
                              <Typography variant="caption" color="text.secondary">
                                Tasks Completed
                              </Typography>
                              <Typography variant="h6">
                                {agent.performance.tasksCompleted}
                              </Typography>
                            </Box>
                          </Box>
                        </Box>
                      )}
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Box>
        </TabPanel>
      </Paper>

      {/* Agent Configuration Dialog */}
      <Dialog
        open={isConfigDialogOpen}
        onClose={() => setIsConfigDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Configure Agent
        </DialogTitle>
        <DialogContent>
          {selectedAgent && (
            <Box sx={{ mt: 2 }}>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Agent Name"
                    defaultValue={agents.find(a => a.id === selectedAgent)?.name}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth>
                    <InputLabel>Agent Type</InputLabel>
                    <Select
                      defaultValue={agents.find(a => a.id === selectedAgent)?.type}
                      label="Agent Type"
                    >
                      <MenuItem value="red_team">Red Team</MenuItem>
                      <MenuItem value="blue_team">Blue Team</MenuItem>
                      <MenuItem value="purple_team">Purple Team</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={<Switch defaultChecked />}
                    label="Auto-start with scenarios"
                  />
                </Grid>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={<Switch defaultChecked />}
                    label="Enable detailed logging"
                  />
                </Grid>
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setIsConfigDialogOpen(false)}>
            Cancel
          </Button>
          <Button variant="contained" onClick={() => setIsConfigDialogOpen(false)}>
            Save Configuration
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Agents;