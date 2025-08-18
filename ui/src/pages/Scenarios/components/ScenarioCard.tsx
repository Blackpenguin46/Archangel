import React from 'react';
import {
  Card,
  CardContent,
  CardActions,
  Typography,
  Box,
  Chip,
  IconButton,
  Tooltip,
  LinearProgress,
  Avatar,
  AvatarGroup,
  Button,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Pause as PauseIcon,
  MoreVert as MoreVertIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Visibility as ViewIcon,
  Schedule as ScheduleIcon,
  Group as GroupIcon,
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

interface ScenarioCardProps {
  scenario: Scenario;
  agents: Agent[];
  onStart: (scenarioId: string) => void;
  onStop: (scenarioId: string) => void;
  onEdit: (scenarioId: string) => void;
  onView: (scenarioId: string) => void;
  onDelete: (scenarioId: string) => void;
}

const ScenarioCard: React.FC<ScenarioCardProps> = ({
  scenario,
  agents,
  onStart,
  onStop,
  onEdit,
  onView,
  onDelete,
}) => {
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);

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
        return <StopIcon sx={{ color: getStatusColor(scenario.status) }} />;
      default:
        return <StopIcon sx={{ color: getStatusColor(scenario.status) }} />;
    }
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleAction = (action: string) => {
    handleMenuClose();
    switch (action) {
      case 'view':
        onView(scenario.id);
        break;
      case 'edit':
        onEdit(scenario.id);
        break;
      case 'delete':
        onDelete(scenario.id);
        break;
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

  return (
    <Card 
      className="scenario-card"
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        border: `1px solid ${getStatusColor(scenario.status)}30`,
        transition: 'all 0.2s ease-in-out',
        '&:hover': {
          borderColor: getStatusColor(scenario.status),
          boxShadow: `0 4px 20px ${getStatusColor(scenario.status)}30`,
        },
      }}
    >
      <CardContent sx={{ flex: 1 }}>
        {/* Header */}
        <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
          <Box flex={1}>
            <Typography variant="h6" component="h3" gutterBottom noWrap>
              {scenario.name}
            </Typography>
            <Typography 
              variant="body2" 
              color="text.secondary" 
              sx={{ 
                display: '-webkit-box',
                WebkitLineClamp: 2,
                WebkitBoxOrient: 'vertical',
                overflow: 'hidden',
              }}
            >
              {scenario.description}
            </Typography>
          </Box>
          
          <IconButton
            size="small"
            onClick={handleMenuOpen}
            sx={{ ml: 1 }}
          >
            <MoreVertIcon />
          </IconButton>
        </Box>

        {/* Status and Type Chips */}
        <Box display="flex" gap={1} mb={2}>
          <Chip
            icon={getStatusIcon()}
            label={scenario.status.charAt(0).toUpperCase() + scenario.status.slice(1)}
            size="small"
            sx={{
              backgroundColor: `${getStatusColor(scenario.status)}20`,
              color: getStatusColor(scenario.status),
              border: `1px solid ${getStatusColor(scenario.status)}`,
            }}
          />
          <Chip
            label={scenario.type.charAt(0).toUpperCase() + scenario.type.slice(1)}
            size="small"
            sx={{
              backgroundColor: `${getTypeColor(scenario.type)}20`,
              color: getTypeColor(scenario.type),
              border: `1px solid ${getTypeColor(scenario.type)}`,
            }}
          />
        </Box>

        {/* Progress Bar */}
        {scenario.status === 'running' && (
          <Box mb={2}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
              <Typography variant="caption" color="text.secondary">
                Progress
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {scenario.progress}%
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={scenario.progress}
              sx={{
                height: 6,
                borderRadius: 3,
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: getStatusColor(scenario.status),
                },
              }}
            />
          </Box>
        )}

        {/* Participants */}
        <Box mb={2}>
          <Box display="flex" alignItems="center" gap={1} mb={1}>
            <GroupIcon sx={{ fontSize: 16, color: 'text.secondary' }} />
            <Typography variant="caption" color="text.secondary">
              Participants ({participantAgents.length})
            </Typography>
          </Box>
          <AvatarGroup max={4} sx={{ justifyContent: 'flex-start' }}>
            {participantAgents.map((agent) => (
              <Tooltip key={agent.id} title={`${agent.name} (${agent.type.replace('_', ' ')})`}>
                <Avatar
                  sx={{
                    width: 32,
                    height: 32,
                    backgroundColor: getAgentColor(agent.type),
                    fontSize: '0.75rem',
                    border: `2px solid ${agent.status === 'active' ? '#4caf50' : '#666'}`,
                  }}
                >
                  {agent.name.charAt(0)}
                </Avatar>
              </Tooltip>
            ))}
          </AvatarGroup>
        </Box>

        {/* Timing Info */}
        <Box display="flex" alignItems="center" gap={1}>
          <ScheduleIcon sx={{ fontSize: 16, color: 'text.secondary' }} />
          <Typography variant="caption" color="text.secondary">
            {scenario.startTime ? (
              <>
                Started {moment(scenario.startTime).fromNow()} • {formatDuration()}
              </>
            ) : (
              'Not started'
            )}
          </Typography>
        </Box>
      </CardContent>

      {/* Actions */}
      <CardActions sx={{ justifyContent: 'space-between', pt: 0 }}>
        <Box>
          {scenario.status === 'running' ? (
            <Button
              size="small"
              startIcon={<StopIcon />}
              onClick={() => onStop(scenario.id)}
              sx={{ color: '#f44336' }}
            >
              Stop
            </Button>
          ) : (
            <Button
              size="small"
              startIcon={<PlayIcon />}
              onClick={() => onStart(scenario.id)}
              sx={{ color: '#4caf50' }}
              disabled={scenario.status === 'completed'}
            >
              Start
            </Button>
          )}
        </Box>
        
        <Button
          size="small"
          startIcon={<ViewIcon />}
          onClick={() => onView(scenario.id)}
        >
          View
        </Button>
      </CardActions>

      {/* Context Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
        PaperProps={{
          elevation: 0,
          sx: {
            overflow: 'visible',
            filter: 'drop-shadow(0px 2px 8px rgba(0,0,0,0.32))',
            mt: 1.5,
          },
        }}
      >
        <MenuItem onClick={() => handleAction('view')}>
          <ListItemIcon>
            <ViewIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>View Details</ListItemText>
        </MenuItem>
        
        <MenuItem 
          onClick={() => handleAction('edit')}
          disabled={scenario.status === 'running'}
        >
          <ListItemIcon>
            <EditIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Edit</ListItemText>
        </MenuItem>
        
        <MenuItem 
          onClick={() => handleAction('delete')}
          disabled={scenario.status === 'running'}
          sx={{ color: 'error.main' }}
        >
          <ListItemIcon>
            <DeleteIcon fontSize="small" sx={{ color: 'error.main' }} />
          </ListItemIcon>
          <ListItemText>Delete</ListItemText>
        </MenuItem>
      </Menu>
    </Card>
  );
};

export default ScenarioCard;