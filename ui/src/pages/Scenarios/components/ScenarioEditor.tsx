import React from 'react';
import {
  Box,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  OutlinedInput,
  Typography,
  Grid,
  Card,
  CardContent,
  Switch,
  FormControlLabel,
  Slider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
} from '@mui/icons-material';

interface Agent {
  id: string;
  name: string;
  type: 'red_team' | 'blue_team' | 'purple_team';
  status: 'active' | 'idle' | 'offline' | 'error';
}

interface NewScenario {
  name: string;
  description: string;
  type: 'training' | 'assessment' | 'live';
  participants: string[];
  configuration: {
    duration?: number;
    complexity?: number;
    autoStart?: boolean;
    notifications?: boolean;
    saveResults?: boolean;
    recordLogs?: boolean;
    networkTopology?: string;
    targetSystems?: string[];
    attackVectors?: string[];
    defenseStrategies?: string[];
  };
}

interface ScenarioEditorProps {
  scenario: NewScenario;
  agents: Agent[];
  onChange: (scenario: NewScenario) => void;
}

const ScenarioEditor: React.FC<ScenarioEditorProps> = ({
  scenario,
  agents,
  onChange,
}) => {
  const handleFieldChange = (field: keyof NewScenario, value: any) => {
    onChange({
      ...scenario,
      [field]: value,
    });
  };

  const handleConfigChange = (field: string, value: any) => {
    onChange({
      ...scenario,
      configuration: {
        ...scenario.configuration,
        [field]: value,
      },
    });
  };

  const availableAgents = agents.filter(agent => agent.status !== 'offline');

  const getAgentColor = (type: string) => {
    switch (type) {
      case 'red_team': return '#e53935';
      case 'blue_team': return '#1976d2';
      case 'purple_team': return '#7b1fa2';
      default: return '#757575';
    }
  };

  const networkTopologies = [
    'Simple Network',
    'Enterprise Network',
    'Cloud Infrastructure',
    'Hybrid Environment',
    'IoT Network',
    'Critical Infrastructure',
  ];

  const attackVectors = [
    'Phishing',
    'Malware',
    'SQL Injection',
    'Cross-Site Scripting',
    'Privilege Escalation',
    'Lateral Movement',
    'Data Exfiltration',
    'Denial of Service',
    'Social Engineering',
    'Zero-Day Exploit',
  ];

  const defenseStrategies = [
    'Network Monitoring',
    'Endpoint Detection',
    'User Behavior Analytics',
    'Threat Intelligence',
    'Incident Response',
    'Vulnerability Management',
    'Access Control',
    'Data Loss Prevention',
    'Security Awareness',
    'Backup and Recovery',
  ];

  return (
    <Box sx={{ mt: 2 }}>
      <Grid container spacing={3}>
        {/* Basic Information */}
        <Grid item xs={12}>
          <Typography variant="h6" gutterBottom>
            Basic Information
          </Typography>
        </Grid>

        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Scenario Name"
            value={scenario.name}
            onChange={(e) => handleFieldChange('name', e.target.value)}
            required
          />
        </Grid>

        <Grid item xs={12} sm={6}>
          <FormControl fullWidth required>
            <InputLabel>Type</InputLabel>
            <Select
              value={scenario.type}
              label="Type"
              onChange={(e) => handleFieldChange('type', e.target.value)}
            >
              <MenuItem value="training">Training</MenuItem>
              <MenuItem value="assessment">Assessment</MenuItem>
              <MenuItem value="live">Live Exercise</MenuItem>
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12}>
          <TextField
            fullWidth
            label="Description"
            value={scenario.description}
            onChange={(e) => handleFieldChange('description', e.target.value)}
            multiline
            rows={3}
            required
          />
        </Grid>

        {/* Participants */}
        <Grid item xs={12}>
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
            Participants
          </Typography>
        </Grid>

        <Grid item xs={12}>
          <FormControl fullWidth>
            <InputLabel>Select Agents</InputLabel>
            <Select
              multiple
              value={scenario.participants}
              onChange={(e) => handleFieldChange('participants', e.target.value)}
              input={<OutlinedInput label="Select Agents" />}
              renderValue={(selected) => (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {(selected as string[]).map((agentId) => {
                    const agent = agents.find(a => a.id === agentId);
                    return agent ? (
                      <Chip
                        key={agentId}
                        label={agent.name}
                        size="small"
                        sx={{
                          backgroundColor: `${getAgentColor(agent.type)}20`,
                          color: getAgentColor(agent.type),
                          border: `1px solid ${getAgentColor(agent.type)}`,
                        }}
                      />
                    ) : null;
                  })}
                </Box>
              )}
            >
              {availableAgents.map((agent) => (
                <MenuItem key={agent.id} value={agent.id}>
                  <Box display="flex" alignItems="center" gap={1}>
                    <Box
                      width={12}
                      height={12}
                      borderRadius="50%"
                      bgcolor={getAgentColor(agent.type)}
                    />
                    {agent.name} ({agent.type.replace('_', ' ')})
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        {/* Configuration */}
        <Grid item xs={12}>
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
            Configuration
          </Typography>
        </Grid>

        <Grid item xs={12}>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography>Basic Settings</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <Typography gutterBottom>
                    Duration (minutes): {scenario.configuration.duration || 60}
                  </Typography>
                  <Slider
                    value={scenario.configuration.duration || 60}
                    onChange={(_, value) => handleConfigChange('duration', value)}
                    min={15}
                    max={480}
                    step={15}
                    marks={[
                      { value: 15, label: '15m' },
                      { value: 60, label: '1h' },
                      { value: 240, label: '4h' },
                      { value: 480, label: '8h' },
                    ]}
                    valueLabelDisplay="auto"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <Typography gutterBottom>
                    Complexity: {scenario.configuration.complexity || 3}
                  </Typography>
                  <Slider
                    value={scenario.configuration.complexity || 3}
                    onChange={(_, value) => handleConfigChange('complexity', value)}
                    min={1}
                    max={5}
                    step={1}
                    marks={[
                      { value: 1, label: 'Basic' },
                      { value: 3, label: 'Medium' },
                      { value: 5, label: 'Expert' },
                    ]}
                    valueLabelDisplay="auto"
                  />
                </Grid>

                <Grid item xs={12}>
                  <Box display="flex" flexDirection="column" gap={1}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={scenario.configuration.autoStart || false}
                          onChange={(e) => handleConfigChange('autoStart', e.target.checked)}
                        />
                      }
                      label="Auto-start scenario"
                    />
                    <FormControlLabel
                      control={
                        <Switch
                          checked={scenario.configuration.notifications || true}
                          onChange={(e) => handleConfigChange('notifications', e.target.checked)}
                        />
                      }
                      label="Send notifications"
                    />
                    <FormControlLabel
                      control={
                        <Switch
                          checked={scenario.configuration.saveResults || true}
                          onChange={(e) => handleConfigChange('saveResults', e.target.checked)}
                        />
                      }
                      label="Save results"
                    />
                    <FormControlLabel
                      control={
                        <Switch
                          checked={scenario.configuration.recordLogs || true}
                          onChange={(e) => handleConfigChange('recordLogs', e.target.checked)}
                        />
                      }
                      label="Record detailed logs"
                    />
                  </Box>
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>

          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography>Network Topology</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <FormControl fullWidth>
                <InputLabel>Network Type</InputLabel>
                <Select
                  value={scenario.configuration.networkTopology || ''}
                  label="Network Type"
                  onChange={(e) => handleConfigChange('networkTopology', e.target.value)}
                >
                  {networkTopologies.map((topology) => (
                    <MenuItem key={topology} value={topology}>
                      {topology}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </AccordionDetails>
          </Accordion>

          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography>Attack Vectors</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <FormControl fullWidth>
                <InputLabel>Select Attack Vectors</InputLabel>
                <Select
                  multiple
                  value={scenario.configuration.attackVectors || []}
                  onChange={(e) => handleConfigChange('attackVectors', e.target.value)}
                  input={<OutlinedInput label="Select Attack Vectors" />}
                  renderValue={(selected) => (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {(selected as string[]).map((vector) => (
                        <Chip
                          key={vector}
                          label={vector}
                          size="small"
                          sx={{ backgroundColor: '#f4433620', color: '#f44336' }}
                        />
                      ))}
                    </Box>
                  )}
                >
                  {attackVectors.map((vector) => (
                    <MenuItem key={vector} value={vector}>
                      {vector}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </AccordionDetails>
          </Accordion>

          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography>Defense Strategies</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <FormControl fullWidth>
                <InputLabel>Select Defense Strategies</InputLabel>
                <Select
                  multiple
                  value={scenario.configuration.defenseStrategies || []}
                  onChange={(e) => handleConfigChange('defenseStrategies', e.target.value)}
                  input={<OutlinedInput label="Select Defense Strategies" />}
                  renderValue={(selected) => (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {(selected as string[]).map((strategy) => (
                        <Chip
                          key={strategy}
                          label={strategy}
                          size="small"
                          sx={{ backgroundColor: '#1976d220', color: '#1976d2' }}
                        />
                      ))}
                    </Box>
                  )}
                >
                  {defenseStrategies.map((strategy) => (
                    <MenuItem key={strategy} value={strategy}>
                      {strategy}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </AccordionDetails>
          </Accordion>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ScenarioEditor;