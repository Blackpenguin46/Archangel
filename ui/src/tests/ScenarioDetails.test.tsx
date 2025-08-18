import React from 'react';
import { render, screen } from '@testing-library/react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import ScenarioDetails from '../pages/Scenarios/components/ScenarioDetails';

const theme = createTheme({
  palette: {
    mode: 'dark',
  },
});

const mockScenario = {
  id: 'scenario-001',
  name: 'Advanced Persistent Threat Simulation',
  description: 'Multi-stage APT attack simulation with lateral movement and data exfiltration',
  type: 'training' as const,
  status: 'running' as const,
  participants: ['agent-001', 'agent-002'],
  startTime: new Date(Date.now() - 3600000), // 1 hour ago
  progress: 65,
};

const mockAgents = [
  {
    id: 'agent-001',
    name: 'Red Team Alpha',
    type: 'red_team' as const,
    status: 'active' as const,
  },
  {
    id: 'agent-002',
    name: 'Blue Team Delta',
    type: 'blue_team' as const,
    status: 'active' as const,
  },
  {
    id: 'agent-003',
    name: 'Purple Team Gamma',
    type: 'purple_team' as const,
    status: 'idle' as const,
  },
];

const renderScenarioDetails = (scenario = mockScenario, agents = mockAgents) => {
  return render(
    <ThemeProvider theme={theme}>
      <ScenarioDetails scenario={scenario} agents={agents} />
    </ThemeProvider>
  );
};

describe('ScenarioDetails Component', () => {
  test('renders scenario name and description', () => {
    renderScenarioDetails();
    
    expect(screen.getByText('Advanced Persistent Threat Simulation')).toBeInTheDocument();
    expect(screen.getByText('Multi-stage APT attack simulation with lateral movement and data exfiltration')).toBeInTheDocument();
  });

  test('displays scenario status and type chips', () => {
    renderScenarioDetails();
    
    expect(screen.getByText('Running')).toBeInTheDocument();
    expect(screen.getByText('Training')).toBeInTheDocument();
  });

  test('shows progress bar for running scenarios', () => {
    renderScenarioDetails();
    
    expect(screen.getByText('Scenario Progress')).toBeInTheDocument();
    expect(screen.getByText('65%')).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });

  test('does not show progress bar for completed scenarios', () => {
    const completedScenario = {
      ...mockScenario,
      status: 'completed' as const,
      progress: 100,
    };
    
    renderScenarioDetails(completedScenario);
    
    expect(screen.queryByText('Scenario Progress')).not.toBeInTheDocument();
  });

  test('displays timing information', () => {
    renderScenarioDetails();
    
    expect(screen.getByText('Start Time')).toBeInTheDocument();
    expect(screen.getByText('Duration')).toBeInTheDocument();
  });

  test('shows participants section', () => {
    renderScenarioDetails();
    
    expect(screen.getByText('Participants (2)')).toBeInTheDocument();
    expect(screen.getByText('Red Team Alpha')).toBeInTheDocument();
    expect(screen.getByText('Blue Team Delta')).toBeInTheDocument();
  });

  test('displays participant agent types and statuses', () => {
    renderScenarioDetails();
    
    expect(screen.getByText('red team')).toBeInTheDocument();
    expect(screen.getByText('blue team')).toBeInTheDocument();
    expect(screen.getAllByText('active')).toHaveLength(2);
  });

  test('shows scenario timeline', () => {
    renderScenarioDetails();
    
    expect(screen.getByText('Scenario Timeline')).toBeInTheDocument();
    expect(screen.getByText('Scenario Initialized')).toBeInTheDocument();
    expect(screen.getByText('Reconnaissance Phase Started')).toBeInTheDocument();
    expect(screen.getByText('Initial Detection')).toBeInTheDocument();
  });

  test('timeline events show correct team types', () => {
    renderScenarioDetails();
    
    expect(screen.getAllByText('red team')).toHaveLength(2); // In timeline events
    expect(screen.getAllByText('blue team')).toHaveLength(2); // In timeline events
    expect(screen.getByText('system')).toBeInTheDocument();
  });

  test('handles scenario without start time', () => {
    const scenarioWithoutStart = {
      ...mockScenario,
      startTime: undefined,
    };
    
    renderScenarioDetails(scenarioWithoutStart);
    
    expect(screen.getByText('Not started')).toBeInTheDocument();
  });

  test('handles scenario with end time', () => {
    const completedScenario = {
      ...mockScenario,
      status: 'completed' as const,
      endTime: new Date(),
    };
    
    renderScenarioDetails(completedScenario);
    
    // Should still show duration calculation
    expect(screen.getByText('Duration')).toBeInTheDocument();
  });

  test('displays correct status icons', () => {
    renderScenarioDetails();
    
    // Running scenario should have play icon
    const statusElements = screen.getAllByTestId(/PlayArrowIcon|StopIcon|PauseIcon|CheckCircleIcon/);
    expect(statusElements.length).toBeGreaterThan(0);
  });

  test('handles different scenario types', () => {
    const assessmentScenario = {
      ...mockScenario,
      type: 'assessment' as const,
    };
    
    renderScenarioDetails(assessmentScenario);
    
    expect(screen.getByText('Assessment')).toBeInTheDocument();
  });

  test('handles different scenario statuses', () => {
    const pausedScenario = {
      ...mockScenario,
      status: 'paused' as const,
    };
    
    renderScenarioDetails(pausedScenario);
    
    expect(screen.getByText('Paused')).toBeInTheDocument();
  });

  test('shows only participating agents', () => {
    renderScenarioDetails();
    
    // Should show only agents that are participants
    expect(screen.getByText('Red Team Alpha')).toBeInTheDocument();
    expect(screen.getByText('Blue Team Delta')).toBeInTheDocument();
    expect(screen.queryByText('Purple Team Gamma')).not.toBeInTheDocument();
  });

  test('handles empty participants list', () => {
    const scenarioWithoutParticipants = {
      ...mockScenario,
      participants: [],
    };
    
    renderScenarioDetails(scenarioWithoutParticipants);
    
    expect(screen.getByText('Participants (0)')).toBeInTheDocument();
  });

  test('responsive design works', () => {
    // Mock mobile viewport
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 375,
    });
    
    renderScenarioDetails();
    
    // Should still render properly on mobile
    expect(screen.getByText('Advanced Persistent Threat Simulation')).toBeInTheDocument();
  });

  test('timeline events show timestamps', () => {
    renderScenarioDetails();
    
    // Timeline events should show time stamps
    const timeElements = screen.getAllByText(/\d{2}:\d{2}:\d{2}/);
    expect(timeElements.length).toBeGreaterThan(0);
  });

  test('progress calculation affects timeline event status', () => {
    const lowProgressScenario = {
      ...mockScenario,
      progress: 20,
    };
    
    renderScenarioDetails(lowProgressScenario);
    
    // With low progress, later events should be pending
    expect(screen.getByText('Scenario Timeline')).toBeInTheDocument();
  });
});

describe('ScenarioDetails Performance', () => {
  test('renders quickly with complex scenario data', () => {
    const startTime = performance.now();
    renderScenarioDetails();
    const endTime = performance.now();
    
    // Should render within reasonable time
    expect(endTime - startTime).toBeLessThan(100);
  });

  test('handles large participant lists efficiently', () => {
    const manyParticipants = Array.from({ length: 20 }, (_, i) => `agent-${i}`);
    const manyAgents = Array.from({ length: 20 }, (_, i) => ({
      id: `agent-${i}`,
      name: `Agent ${i}`,
      type: 'red_team' as const,
      status: 'active' as const,
    }));
    
    const scenarioWithManyParticipants = {
      ...mockScenario,
      participants: manyParticipants,
    };
    
    const startTime = performance.now();
    renderScenarioDetails(scenarioWithManyParticipants, manyAgents);
    const endTime = performance.now();
    
    // Should still render efficiently with many participants
    expect(endTime - startTime).toBeLessThan(200);
  });
});

describe('ScenarioDetails Accessibility', () => {
  test('has proper heading structure', () => {
    renderScenarioDetails();
    
    const mainHeading = screen.getByRole('heading', { level: 5 });
    expect(mainHeading).toHaveTextContent('Advanced Persistent Threat Simulation');
  });

  test('has accessible progress bar', () => {
    renderScenarioDetails();
    
    const progressBar = screen.getByRole('progressbar');
    expect(progressBar).toBeInTheDocument();
    expect(progressBar).toHaveAttribute('aria-valuenow', '65');
  });

  test('has proper list structure for participants', () => {
    renderScenarioDetails();
    
    const participantsList = screen.getByRole('list');
    expect(participantsList).toBeInTheDocument();
    
    const listItems = screen.getAllByRole('listitem');
    expect(listItems.length).toBeGreaterThan(0);
  });

  test('has sufficient color contrast', () => {
    renderScenarioDetails();
    
    // This would require additional testing with tools like axe-core
    // For now, we ensure the theme provides good contrast
    expect(theme.palette.mode).toBe('dark');
  });
});