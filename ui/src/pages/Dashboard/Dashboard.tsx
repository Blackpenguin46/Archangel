import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  CircularProgress,
  Alert,
  Paper,
  Tab,
  Tabs,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  Security as SecurityIcon,
  Speed as SpeedIcon,
  Storage as StorageIcon,
  Group as GroupIcon,
  PlayArrow as PlayArrowIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { useSystem } from '../../contexts/SystemContext';
import { useWebSocket, useWebSocketEvent } from '../../contexts/WebSocketContext';
import SystemStatusCard from './components/SystemStatusCard';
import MetricsChart from './components/MetricsChart';
import ActivityFeed from './components/ActivityFeed';
import ThreatMap from './components/ThreatMap';

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

const Dashboard: React.FC = () => {
  const { systemStatus, agents, scenarios, alerts, loading, error } = useSystem();
  const { connectionStatus } = useWebSocket();
  const [activeTab, setActiveTab] = useState(0);
  const [realtimeMetrics, setRealtimeMetrics] = useState<any[]>([]);
  const [threatData, setThreatData] = useState<any[]>([]);

  // Subscribe to real-time metric updates
  useWebSocketEvent('metric_update', (data) => {
    setRealtimeMetrics(prev => {
      const newMetrics = [...prev, {
        timestamp: new Date(data.timestamp).toLocaleTimeString(),
        ...data.metrics,
      }];
      return newMetrics.slice(-20); // Keep last 20 data points
    });
  });

  // Subscribe to system events for threat mapping
  useWebSocketEvent('system_event', (event) => {
    if (event.type === 'alert') {
      setThreatData(prev => {
        const newThreat = {
          id: Date.now(),
          level: event.data.level,
          source: event.data.source,
          timestamp: new Date(event.timestamp),
          location: event.data.location || { lat: 40.7128, lng: -74.0060 }, // Default to NYC
        };
        return [newThreat, ...prev.slice(0, 49)]; // Keep last 50 threats
      });
    }
  });

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return '#4caf50';
      case 'warning':
        return '#ff9800';
      case 'critical':
        return '#f44336';
      default:
        return '#666';
    }
  };

  const formatNumber = (num: number) => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    }
    if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
  };

  if (loading) {
    return (
      <Box className="flex-center" minHeight="400px">
        <CircularProgress size={60} />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box className="page-container fade-in">
      <Box mb={3}>
        <Typography variant="h4" component="h1" gutterBottom>
          Archangel Dashboard
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Real-time monitoring and control of AI security operations
        </Typography>
      </Box>

      {/* Connection Status */}
      {connectionStatus !== 'connected' && (
        <Alert 
          severity={connectionStatus === 'connecting' ? 'info' : 'warning'} 
          sx={{ mb: 3 }}
        >
          WebSocket {connectionStatus === 'connecting' ? 'connecting...' : 'disconnected'}
          {connectionStatus === 'disconnected' && ' - Some features may not work properly'}
        </Alert>
      )}

      {/* System Overview Cards */}
      <Grid container spacing={3} className="dashboard-grid">
        <Grid item xs={12} sm={6} md={3}>
          <SystemStatusCard
            title="System Status"
            value={systemStatus.overall}
            icon={<SecurityIcon />}
            color={getStatusColor(systemStatus.overall)}
            subtitle="Overall health"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <SystemStatusCard
            title="Active Agents"
            value={systemStatus.activeAgents}
            icon={<GroupIcon />}
            color="#1976d2"
            subtitle={`${agents.filter(a => a.status === 'active').length} running`}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <SystemStatusCard
            title="Running Scenarios"
            value={systemStatus.runningScenarios}
            icon={<PlayArrowIcon />}
            color="#9c27b0"
            subtitle={`${systemStatus.totalScenarios} total`}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <SystemStatusCard
            title="Active Alerts"
            value={systemStatus.activeAlerts}
            icon={<WarningIcon />}
            color="#f44336"
            subtitle={`${alerts.filter(a => !a.acknowledged).length} unacknowledged`}
          />
        </Grid>
      </Grid>

      {/* Tabbed Content */}
      <Paper sx={{ mt: 3 }}>
        <Tabs
          value={activeTab}
          onChange={handleTabChange}
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab label="System Metrics" />
          <Tab label="Agent Activity" />
          <Tab label="Threat Intelligence" />
          <Tab label="Performance Analytics" />
        </Tabs>

        <TabPanel value={activeTab} index={0}>
          <Grid container spacing={3} sx={{ p: 3 }}>
            {/* Real-time Metrics Chart */}
            <Grid item xs={12} lg={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Real-time System Metrics
                  </Typography>
                  <Box height={350}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={realtimeMetrics}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                        <XAxis 
                          dataKey="timestamp" 
                          stroke="#666"
                          fontSize={12}
                        />
                        <YAxis stroke="#666" fontSize={12} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#1a1a2e',
                            border: '1px solid #333',
                            borderRadius: '4px',
                          }}
                        />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="cpu"
                          stroke="#1976d2"
                          strokeWidth={2}
                          name="CPU %"
                        />
                        <Line
                          type="monotone"
                          dataKey="memory"
                          stroke="#9c27b0"
                          strokeWidth={2}
                          name="Memory %"
                        />
                        <Line
                          type="monotone"
                          dataKey="network"
                          stroke="#4caf50"
                          strokeWidth={2}
                          name="Network MB/s"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Resource Usage */}
            <Grid item xs={12} lg={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Resource Usage
                  </Typography>
                  <Box display="flex" flexDirection="column" gap={3}>
                    <Box>
                      <Box display="flex" justifyContent="space-between" mb={1}>
                        <Typography variant="body2">CPU</Typography>
                        <Typography variant="body2">{systemStatus.cpuUsage}%</Typography>
                      </Box>
                      <CircularProgress
                        variant="determinate"
                        value={systemStatus.cpuUsage}
                        size={60}
                        thickness={4}
                        sx={{ color: systemStatus.cpuUsage > 80 ? '#f44336' : '#1976d2' }}
                      />
                    </Box>
                    
                    <Box>
                      <Box display="flex" justifyContent="space-between" mb={1}>
                        <Typography variant="body2">Memory</Typography>
                        <Typography variant="body2">{systemStatus.memoryUsage}%</Typography>
                      </Box>
                      <CircularProgress
                        variant="determinate"
                        value={systemStatus.memoryUsage}
                        size={60}
                        thickness={4}
                        sx={{ color: systemStatus.memoryUsage > 80 ? '#f44336' : '#9c27b0' }}
                      />
                    </Box>
                    
                    <Box>
                      <Box display="flex" justifyContent="space-between" mb={1}>
                        <Typography variant="body2">Network</Typography>
                        <Typography variant="body2">{formatNumber(systemStatus.networkTraffic)} B/s</Typography>
                      </Box>
                      <CircularProgress
                        variant="determinate"
                        value={Math.min(100, (systemStatus.networkTraffic / 10000) * 100)}
                        size={60}
                        thickness={4}
                        sx={{ color: '#4caf50' }}
                      />
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          <Grid container spacing={3} sx={{ p: 3 }}>
            {/* Agent Status Grid */}
            <Grid item xs={12} lg={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Agent Status Overview
                  </Typography>
                  <Grid container spacing={2} className="agent-status-grid">
                    {agents.map((agent) => (
                      <Grid item key={agent.id}>
                        <Card
                          sx={{
                            border: '1px solid',
                            borderColor: agent.status === 'active' ? '#4caf50' : '#666',
                            transition: 'all 0.2s',
                          }}
                        >
                          <CardContent sx={{ p: 2 }}>
                            <Box display="flex" alignItems="center" gap={1} mb={1}>
                              <Box
                                className="status-indicator"
                                sx={{
                                  backgroundColor: agent.status === 'active' ? '#4caf50' : '#666',
                                }}
                              />
                              <Typography variant="subtitle2" noWrap>
                                {agent.name}
                              </Typography>
                            </Box>
                            <Chip
                              label={agent.type.replace('_', ' ')}
                              size="small"
                              color={
                                agent.type === 'red_team' ? 'error' :
                                agent.type === 'blue_team' ? 'primary' : 'secondary'
                              }
                              sx={{ mb: 1 }}
                            />
                            <Typography variant="caption" display="block" color="text.secondary">
                              Success Rate: {agent.performance.successRate}%
                            </Typography>
                            <Typography variant="caption" display="block" color="text.secondary">
                              Response Time: {agent.performance.responseTime}ms
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                </CardContent>
              </Card>
            </Grid>

            {/* Activity Feed */}
            <Grid item xs={12} lg={4}>
              <ActivityFeed />
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={activeTab} index={2}>
          <Grid container spacing={3} sx={{ p: 3 }}>
            {/* Threat Map */}
            <Grid item xs={12}>
              <ThreatMap threats={threatData} />
            </Grid>
            
            {/* Alert Distribution */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Alert Distribution
                  </Typography>
                  <Box height={300}>
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={[
                            { name: 'Critical', value: alerts.filter(a => a.level === 'critical').length, color: '#f44336' },
                            { name: 'Error', value: alerts.filter(a => a.level === 'error').length, color: '#ff9800' },
                            { name: 'Warning', value: alerts.filter(a => a.level === 'warning').length, color: '#ffeb3b' },
                            { name: 'Info', value: alerts.filter(a => a.level === 'info').length, color: '#2196f3' },
                          ]}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={120}
                          dataKey="value"
                        >
                          {[
                            { name: 'Critical', value: alerts.filter(a => a.level === 'critical').length, color: '#f44336' },
                            { name: 'Error', value: alerts.filter(a => a.level === 'error').length, color: '#ff9800' },
                            { name: 'Warning', value: alerts.filter(a => a.level === 'warning').length, color: '#ffeb3b' },
                            { name: 'Info', value: alerts.filter(a => a.level === 'info').length, color: '#2196f3' },
                          ].map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#1a1a2e',
                            border: '1px solid #333',
                          }}
                        />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Recent Threats */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Recent Threats
                  </Typography>
                  <Box maxHeight={300} overflow="auto">
                    {threatData.slice(0, 10).map((threat) => (
                      <Box
                        key={threat.id}
                        display="flex"
                        alignItems="center"
                        gap={2}
                        py={1}
                        borderBottom="1px solid #333"
                      >
                        <Chip
                          label={threat.level}
                          size="small"
                          className={`threat-${threat.level}`}
                        />
                        <Box flex={1}>
                          <Typography variant="body2">
                            {threat.source}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {threat.timestamp.toLocaleTimeString()}
                          </Typography>
                        </Box>
                      </Box>
                    ))}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={activeTab} index={3}>
          <Grid container spacing={3} sx={{ p: 3 }}>
            {/* Performance Metrics */}
            <Grid item xs={12}>
              <MetricsChart />
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>
    </Box>
  );
};

export default Dashboard;