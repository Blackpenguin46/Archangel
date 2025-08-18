import React from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Chip,
  Avatar,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Paper,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Pause as PauseIcon,
  CheckCircle as CompleteIcon,
  Schedule as ScheduleIcon,
  Group as GroupIcon,
  Assignment as AssignmentIcon,
  TrendingUp as TrendingUpIcon,
  Security as SecurityIcon,
} from '@mui/icons-material';
import moment from 'moment';

interface Agent {
  id: string;
  name: string;
  type: 'red_team' | 'blue_team' | 'purple_team';
  status: 'active' | 'idle' | 'offline' | 'error';
}

interface Scenario {
  id: string;
  name: string;
  description: string;
  type: 'training' | 'assessment' | 'live';
  status: 'running' | 'paused' | 'completed' | 'failed';
  participants: string[];
  startTime?: Date;
  endTime?: Date;
  progress: number;
}

interface ScenarioDetailsProps {
  scenario: Scenario;
  agents: Agent[];
}

const ScenarioDetails: React.FC<ScenarioDetailsProps> = ({ scenario, agents }) => {
  const participantAgents = agents.filter(agent => 
    scenario.participants.includes(agent.id)
  );

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return '#4caf50';
      case 'paused': return '#ff9800';
      case 'completed': return '#2196f3';
      case 'failed': return '#f44336';
      default: return '#757575';
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'training': return '#1976d2';
      case 'assessment': return '#9c27b0';
      case 'live': return '#f44336';
      default: return '#757575';
    }
  };

  const getAgentColor = (type: string) => {
    switch (type) {
      case 'red_team': return '#e53935';
      case 'blue_team': return '#1976d2';
      case 'purple_team': return '#7b1fa2';
      default: return '#757575';
    }
  };

  const getStatusIcon = () => {
    switch (scenario.status) {
      case 'running':
        return <PlayIcon sx={{ color: getStatusColor(scenario.status) }} />;
      case 'paused':
        return <PauseIcon sx={{ color: getStatusColor(scenario.status) }} />;
      case 'completed':
        return <CompleteIcon sx={{ color: getStatusColor(scenario.status) }} />;
      case 'failed':
        return <StopIcon sx={{ color: getStatusColor(scenario.status) }} />;
      default:
        return <StopIcon sx={{ color: getStatusColor(scenario.status) }} />;
    }
  };

  const formatDuration = () => {
    if (!scenario.startTime) return 'Not started';
    
    const end = scenario.endTime || new Date();
    const duration = moment.duration(moment(end).diff(moment(scenario.startTime)));
    
    if (duration.asHours() >= 1) {
      return `${Math.floor(duration.asHours())}h ${duration.minutes()}m`;
    }
    return `${duration.minutes()}m ${duration.seconds()}s`;
  };

  // Mock scenario events for timeline
  const scenarioEvents = [
    {
      id: 1,
      title: 'Scenario Initialized',
      description: 'Environment setup and agent deployment completed',
      timestamp: scenario.startTime ? moment(scenario.startTime).subtract(5, 'minutes') : moment(),
      type: 'system',
      status: 'completed',
    },
    {
      id: 2,
      title: 'Reconnaissance Phase Started',
      description: 'Red team agents began network discovery',
      timestamp: scenario.startTime || moment(),
      type: 'red_team',
      status: 'completed',
    },
    {
      id: 3,
      title: 'Initial Detection',
      description: 'Blue team detected suspicious network activity',
      timestamp: scenario.startTime ? moment(scenario.startTime).add(15, 'minutes') : moment(),
      type: 'blue_team',
      status: scenario.progress > 30 ? 'completed' : 'in_progress',
    },
    {
      id: 4,
      title: 'Exploitation Attempt',
      description: 'Red team attempted to exploit discovered vulnerabilities',
      timestamp: scenario.startTime ? moment(scenario.startTime).add(30, 'minutes') : moment(),
      type: 'red_team',
      status: scenario.progress > 60 ? 'completed' : scenario.progress > 30 ? 'in_progress' : 'pending',
    },
    {
      id: 5,
      title: 'Incident Response',
      description: 'Blue team initiated containment procedures',
      timestamp: scenario.startTime ? moment(scenario.startTime).add(45, 'minutes') : moment(),
      type: 'blue_team',
      status: scenario.progress > 80 ? 'completed' : scenario.progress > 60 ? 'in_progress' : 'pending',
    },
  ];

  const getEventColor = (type: string) => {
    switch (type) {
      case 'red_team': return '#e53935';
      case 'blue_team': return '#1976d2';
      case 'system': return '#4caf50';
      default: return '#757575';
    }
  };

  const getEventIcon = (type: string, status: string) => {
    if (status === 'completed') {
      return <CompleteIcon sx={{ color: '#4caf50' }} />;
    }
    if (status === 'in_progress') {
      return <PlayIcon sx={{ color: '#ff9800' }} />;
    }
    
    switch (type) {
      case 'red_team': return <SecurityIcon sx={{ color: '#e53935' }} />;
      case 'blue_team': return <SecurityIcon sx={{ color: '#1976d2' }} />;
      case 'system': return <AssignmentIcon sx={{ color: '#4caf50' }} />;
      default: return <ScheduleIcon sx={{ color: '#757575' }} />;
    }
  };

  return (
    <Box>
      <Grid container spacing={3}>
        {/* Scenario Overview */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={2} mb={3}>
                {getStatusIcon()}
                <Box flex={1}>
                  <Typography variant="h5" gutterBottom>
                    {scenario.name}
                  </Typography>
                  <Typography variant="body1" color="text.secondary" paragraph>
                    {scenario.description}
                  </Typography>
                </Box>
              </Box>

              <Box display="flex" gap={1} mb={3}>
                <Chip
                  label={scenario.status.charAt(0).toUpperCase() + scenario.status.slice(1)}
                  sx={{
                    backgroundColor: `${getStatusColor(scenario.status)}20`,
                    color: getStatusColor(scenario.status),
                    border: `1px solid ${getStatusColor(scenario.status)}`,
                  }}
                />
                <Chip
                  label={scenario.type.charAt(0).toUpperCase() + scenario.type.slice(1)}
                  sx={{
                    backgroundColor: `${getTypeColor(scenario.type)}20`,
                    color: getTypeColor(scenario.type),
                    border: `1px solid ${getTypeColor(scenario.type)}`,
                  }}
                />
              </Box>

              {/* Progress */}
              {scenario.status === 'running' && (
                <Box mb={3}>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                    <Typography variant="subtitle2">
                      Scenario Progress
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {scenario.progress}%
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={scenario.progress}
                    sx={{
                      height: 8,
                      borderRadius: 4,
                      backgroundColor: 'rgba(255, 255, 255, 0.1)',
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: getStatusColor(scenario.status),
                      },
                    }}
                  />
                </Box>
              )}

              {/* Timing Information */}
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <Box display="flex" alignItems="center" gap={1}>
                    <ScheduleIcon sx={{ color: 'text.secondary', fontSize: 20 }} />
                    <Box>
                      <Typography variant="caption" color="text.secondary" display="block">
                        Start Time
                      </Typography>
                      <Typography variant="body2">
                        {scenario.startTime 
                          ? moment(scenario.startTime).format('MMM DD, YYYY HH:mm')
                          : 'Not started'
                        }
                      </Typography>
                    </Box>
                  </Box>
                </Grid>
                
                <Grid item xs={12} sm={6}>
                  <Box display="flex" alignItems="center" gap={1}>
                    <TrendingUpIcon sx={{ color: 'text.secondary', fontSize: 20 }} />
                    <Box>
                      <Typography variant="caption" color="text.secondary" display="block">
                        Duration
                      </Typography>
                      <Typography variant="body2">
                        {formatDuration()}
                      </Typography>
                    </Box>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Participants */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={1} mb={2}>
                <GroupIcon sx={{ color: 'text.secondary' }} />
                <Typography variant="h6">
                  Participants ({participantAgents.length})
                </Typography>
              </Box>
              
              <List dense>
                {participantAgents.map((agent) => (
                  <ListItem key={agent.id} sx={{ px: 0 }}>
                    <ListItemIcon>
                      <Avatar
                        sx={{
                          width: 32,
                          height: 32,
                          backgroundColor: getAgentColor(agent.type),
                          fontSize: '0.75rem',
                        }}
                      >
                        {agent.name.charAt(0)}
                      </Avatar>
                    </ListItemIcon>
                    <ListItemText
                      primary={agent.name}
                      secondary={
                        <Box display="flex" alignItems="center" gap={1}>
                          <Chip
                            label={agent.type.replace('_', ' ')}
                            size="small"
                            sx={{
                              fontSize: '0.7rem',
                              height: 20,
                              backgroundColor: `${getAgentColor(agent.type)}20`,
                              color: getAgentColor(agent.type),
                            }}
                          />
                          <Chip
                            label={agent.status}
                            size="small"
                            sx={{
                              fontSize: '0.7rem',
                              height: 20,
                              backgroundColor: agent.status === 'active' ? '#4caf5020' : '#66666620',
                              color: agent.status === 'active' ? '#4caf50' : '#666',
                            }}
                          />
                        </Box>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Scenario Timeline */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Scenario Timeline
              </Typography>
              
              <Box>
                {scenarioEvents.map((event, index) => (
                  <Box key={event.id} display="flex" gap={2} mb={3}>
                    <Box
                      sx={{
                        width: 40,
                        height: 40,
                        borderRadius: '50%',
                        backgroundColor: getEventColor(event.type),
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        flexShrink: 0,
                      }}
                    >
                      {getEventIcon(event.type, event.status)}
                    </Box>
                    <Box flex={1}>
                      <Paper
                        elevation={0}
                        sx={{
                          p: 2,
                          backgroundColor: `${getEventColor(event.type)}10`,
                          border: `1px solid ${getEventColor(event.type)}30`,
                        }}
                      >
                        <Typography variant="subtitle2" gutterBottom>
                          {event.title}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" paragraph>
                          {event.description}
                        </Typography>
                        <Box display="flex" justifyContent="space-between" alignItems="center">
                          <Chip
                            label={event.type.replace('_', ' ')}
                            size="small"
                            sx={{
                              backgroundColor: `${getEventColor(event.type)}20`,
                              color: getEventColor(event.type),
                            }}
                          />
                          <Typography variant="caption" color="text.secondary">
                            {event.timestamp.format('HH:mm:ss')}
                          </Typography>
                        </Box>
                      </Paper>
                    </Box>
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ScenarioDetails;