import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  IconButton,
  Tooltip,
  Alert,
  LinearProgress,
  Fab,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Pause as PauseIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Add as AddIcon,
  Settings as SettingsIcon,
  Visibility as ViewIcon,
  Assignment as AssignmentIcon,
} from '@mui/icons-material';
import { useSystem } from '../../contexts/SystemContext';
import { useWebSocketEvent } from '../../contexts/WebSocketContext';
import ScenarioCard from './components/ScenarioCard';
import ScenarioEditor from './components/ScenarioEditor';
import ScenarioDetails from './components/ScenarioDetails';

interface NewScenario {
  name: string;
  description: string;
  type: 'training' | 'assessment' | 'live';
  participants: string[];
  configuration: any;
}

const Scenarios: React.FC = () => {
  const { scenarios, agents, startScenario, stopScenario, loading } = useSystem();
  const [selectedScenario, setSelectedScenario] = useState<string | null>(null);
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [isDetailsDialogOpen, setIsDetailsDialogOpen] = useState(false);
  const [filterType, setFilterType] = useState<'all' | 'training' | 'assessment' | 'live'>('all');
  const [filterStatus, setFilterStatus] = useState<'all' | 'running' | 'paused' | 'completed'>('all');

  const [newScenario, setNewScenario] = useState<NewScenario>({
    name: '',
    description: '',
    type: 'training',
    participants: [],
    configuration: {},
  });

  // Subscribe to scenario updates
  useWebSocketEvent('scenario_event', (event) => {
    // Handle real-time scenario updates
    console.log('Scenario event:', event);
  });

  const filteredScenarios = scenarios.filter(scenario => {
    const typeMatch = filterType === 'all' || scenario.type === filterType;
    const statusMatch = filterStatus === 'all' || scenario.status === filterStatus;
    return typeMatch && statusMatch;
  });

  const handleStartScenario = async (scenarioId: string) => {
    try {
      await startScenario(scenarioId);
    } catch (error) {
      console.error('Failed to start scenario:', error);
    }
  };

  const handleStopScenario = async (scenarioId: string) => {
    try {
      await stopScenario(scenarioId);
    } catch (error) {
      console.error('Failed to stop scenario:', error);
    }
  };

  const handleCreateScenario = () => {
    setIsCreateDialogOpen(true);
  };

  const handleEditScenario = (scenarioId: string) => {
    const scenario = scenarios.find(s => s.id === scenarioId);
    if (scenario) {
      setSelectedScenario(scenarioId);
      setNewScenario({
        name: scenario.name,
        description: scenario.description,
        type: scenario.type,
        participants: scenario.participants,
        configuration: {},
      });
      setIsEditDialogOpen(true);
    }
  };

  const handleViewScenario = (scenarioId: string) => {
    setSelectedScenario(scenarioId);
    setIsDetailsDialogOpen(true);
  };

  const handleSaveScenario = async () => {
    try {
      // In a real application, this would make an API call
      console.log('Saving scenario:', newScenario);
      
      // Reset form
      setNewScenario({
        name: '',
        description: '',
        type: 'training',
        participants: [],
        configuration: {},
      });
      
      setIsCreateDialogOpen(false);
      setIsEditDialogOpen(false);
    } catch (error) {
      console.error('Failed to save scenario:', error);
    }
  };

  const handleDeleteScenario = async (scenarioId: string) => {
    if (window.confirm('Are you sure you want to delete this scenario?')) {
      try {
        // In a real application, this would make an API call
        console.log('Deleting scenario:', scenarioId);
      } catch (error) {
        console.error('Failed to delete scenario:', error);
      }
    }
  };

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

  return (
    <Box className="page-container fade-in">
      <Box mb={3}>
        <Typography variant="h4" component="h1" gutterBottom>
          Scenario Management
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Create, configure, and manage AI security training scenarios
        </Typography>
      </Box>

      {/* Filters and Actions */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box display="flex" gap={2}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Type</InputLabel>
            <Select
              value={filterType}
              label="Type"
              onChange={(e) => setFilterType(e.target.value as any)}
            >
              <MenuItem value="all">All Types</MenuItem>
              <MenuItem value="training">Training</MenuItem>
              <MenuItem value="assessment">Assessment</MenuItem>
              <MenuItem value="live">Live</MenuItem>
            </Select>
          </FormControl>

          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Status</InputLabel>
            <Select
              value={filterStatus}
              label="Status"
              onChange={(e) => setFilterStatus(e.target.value as any)}
            >
              <MenuItem value="all">All Status</MenuItem>
              <MenuItem value="running">Running</MenuItem>
              <MenuItem value="paused">Paused</MenuItem>
              <MenuItem value="completed">Completed</MenuItem>
            </Select>
          </FormControl>
        </Box>

        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={handleCreateScenario}
        >
          Create Scenario
        </Button>
      </Box>

      {/* Statistics Cards */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={2}>
                <AssignmentIcon sx={{ color: '#1976d2', fontSize: 32 }} />
                <Box>
                  <Typography variant="h4" color="primary">
                    {scenarios.length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Scenarios
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
                <PlayIcon sx={{ color: '#4caf50', fontSize: 32 }} />
                <Box>
                  <Typography variant="h4" sx={{ color: '#4caf50' }}>
                    {scenarios.filter(s => s.status === 'running').length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Running
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
                <PauseIcon sx={{ color: '#ff9800', fontSize: 32 }} />
                <Box>
                  <Typography variant="h4" sx={{ color: '#ff9800' }}>
                    {scenarios.filter(s => s.status === 'paused').length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Paused
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
                <StopIcon sx={{ color: '#2196f3', fontSize: 32 }} />
                <Box>
                  <Typography variant="h4" sx={{ color: '#2196f3' }}>
                    {scenarios.filter(s => s.status === 'completed').length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Completed
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {loading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Scenarios Grid */}
      <Grid container spacing={3}>
        {filteredScenarios.map((scenario) => (
          <Grid item xs={12} md={6} lg={4} key={scenario.id}>
            <ScenarioCard
              scenario={scenario}
              agents={agents}
              onStart={handleStartScenario}
              onStop={handleStopScenario}
              onEdit={handleEditScenario}
              onView={handleViewScenario}
              onDelete={handleDeleteScenario}
            />
          </Grid>
        ))}

        {filteredScenarios.length === 0 && (
          <Grid item xs={12}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 6 }}>
                <AssignmentIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  No scenarios found
                </Typography>
                <Typography variant="body2" color="text.secondary" mb={3}>
                  {filterType !== 'all' || filterStatus !== 'all' 
                    ? 'Try adjusting your filters or create a new scenario.'
                    : 'Get started by creating your first scenario.'
                  }
                </Typography>
                <Button
                  variant="contained"
                  startIcon={<AddIcon />}
                  onClick={handleCreateScenario}
                >
                  Create Scenario
                </Button>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>

      {/* Create/Edit Scenario Dialog */}
      <Dialog
        open={isCreateDialogOpen || isEditDialogOpen}
        onClose={() => {
          setIsCreateDialogOpen(false);
          setIsEditDialogOpen(false);
        }}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {isEditDialogOpen ? 'Edit Scenario' : 'Create New Scenario'}
        </DialogTitle>
        <DialogContent>
          <ScenarioEditor
            scenario={newScenario}
            agents={agents}
            onChange={setNewScenario}
          />
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => {
              setIsCreateDialogOpen(false);
              setIsEditDialogOpen(false);
            }}
          >
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={handleSaveScenario}
            disabled={!newScenario.name || !newScenario.description}
          >
            {isEditDialogOpen ? 'Update' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Scenario Details Dialog */}
      <Dialog
        open={isDetailsDialogOpen}
        onClose={() => setIsDetailsDialogOpen(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          Scenario Details
        </DialogTitle>
        <DialogContent>
          {selectedScenario && (
            <ScenarioDetails
              scenario={scenarios.find(s => s.id === selectedScenario)!}
              agents={agents}
            />
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setIsDetailsDialogOpen(false)}>
            Close
          </Button>
        </DialogActions>
      </Dialog>

      {/* Floating Action Button for Quick Create */}
      <Fab
        color="primary"
        sx={{ position: 'fixed', bottom: 24, right: 24 }}
        onClick={handleCreateScenario}
      >
        <AddIcon />
      </Fab>
    </Box>
  );
};

export default Scenarios;