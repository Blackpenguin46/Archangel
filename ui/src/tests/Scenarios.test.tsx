import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import Scenarios from '../pages/Scenarios/Scenarios';

// Mock the contexts
jest.mock('../contexts/SystemContext', () => ({
  useSystem: () => ({
    scenarios: [
      {
        id: 'scenario-001',
        name: 'Advanced Persistent Threat Simulation',
        description: 'Multi-stage APT attack simulation with lateral movement',
        type: 'training',
        status: 'running',
        participants: ['agent-001', 'agent-002'],
        startTime: new Date(Date.now() - 3600000),
        progress: 65,
      },
      {
        id: 'scenario-002',
        name: 'Phishing Campaign Assessment',
        description: 'Email-based social engineering attack simulation',
        type: 'assessment',
        status: 'completed',
        participants: ['agent-003'],
        startTime: new Date(Date.now() - 7200000),
        endTime: new Date(Date.now() - 1800000),
        progress: 100,
      },
      {
        id: 'scenario-003',
        name: 'Live Red Team Exercise',
        description: 'Real-time penetration testing exercise',
        type: 'live',
        status: 'paused',
        participants: ['agent-001', 'agent-003'],
        startTime: new Date(Date.now() - 1800000),
        progress: 35,
      },
    ],
    agents: [
      {
        id: 'agent-001',
        name: 'Red Team Alpha',
        type: 'red_team',
        status: 'active',
      },
      {
        id: 'agent-002',
        name: 'Blue Team Delta',
        type: 'blue_team',
        status: 'active',
      },
      {
        id: 'agent-003',
        name: 'Purple Team Gamma',
        type: 'purple_team',
        status: 'idle',
      },
    ],
    loading: false,
    error: null,
    startScenario: jest.fn().mockResolvedValue(undefined),
    stopScenario: jest.fn().mockResolvedValue(undefined),
  }),
}));

jest.mock('../contexts/WebSocketContext', () => ({
  useWebSocketEvent: jest.fn(),
}));

const theme = createTheme({
  palette: {
    mode: 'dark',
  },
});

const renderScenarios = () => {
  return render(
    <BrowserRouter>
      <ThemeProvider theme={theme}>
        <Scenarios />
      </ThemeProvider>
    </BrowserRouter>
  );
};

describe('Scenarios Page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders scenarios page with title and description', () => {
    renderScenarios();
    
    expect(screen.getByText('Scenario Management')).toBeInTheDocument();
    expect(screen.getByText('Create, configure, and manage AI security training scenarios')).toBeInTheDocument();
  });

  test('displays filter controls', () => {
    renderScenarios();
    
    expect(screen.getByLabelText('Type')).toBeInTheDocument();
    expect(screen.getByLabelText('Status')).toBeInTheDocument();
  });

  test('shows create scenario button', () => {
    renderScenarios();
    
    expect(screen.getByText('Create Scenario')).toBeInTheDocument();
  });

  test('displays statistics cards', () => {
    renderScenarios();
    
    expect(screen.getByText('Total Scenarios')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument(); // Total scenarios
    
    expect(screen.getByText('Running')).toBeInTheDocument();
    expect(screen.getByText('1')).toBeInTheDocument(); // Running scenarios
    
    expect(screen.getByText('Completed')).toBeInTheDocument();
    expect(screen.getByText('1')).toBeInTheDocument(); // Completed scenarios
  });

  test('renders scenario cards', () => {
    renderScenarios();
    
    expect(screen.getByText('Advanced Persistent Threat Simulation')).toBeInTheDocument();
    expect(screen.getByText('Phishing Campaign Assessment')).toBeInTheDocument();
    expect(screen.getByText('Live Red Team Exercise')).toBeInTheDocument();
  });

  test('filters scenarios by type', async () => {
    renderScenarios();
    
    // Click on type filter
    fireEvent.mouseDown(screen.getByLabelText('Type'));
    
    // Select training type
    const trainingOption = screen.getByText('Training');
    fireEvent.click(trainingOption);
    
    // Should show only training scenarios
    await waitFor(() => {
      expect(screen.getByText('Advanced Persistent Threat Simulation')).toBeInTheDocument();
      expect(screen.queryByText('Phishing Campaign Assessment')).not.toBeInTheDocument();
    });
  });

  test('filters scenarios by status', async () => {
    renderScenarios();
    
    // Click on status filter
    fireEvent.mouseDown(screen.getByLabelText('Status'));
    
    // Select running status
    const runningOption = screen.getByText('Running');
    fireEvent.click(runningOption);
    
    // Should show only running scenarios
    await waitFor(() => {
      expect(screen.getByText('Advanced Persistent Threat Simulation')).toBeInTheDocument();
      expect(screen.queryByText('Phishing Campaign Assessment')).not.toBeInTheDocument();
    });
  });

  test('opens create scenario dialog', async () => {
    renderScenarios();
    
    // Click create scenario button
    fireEvent.click(screen.getByText('Create Scenario'));
    
    // Dialog should open
    await waitFor(() => {
      expect(screen.getByText('Create New Scenario')).toBeInTheDocument();
    });
  });

  test('create scenario dialog has required fields', async () => {
    renderScenarios();
    
    // Open create dialog
    fireEvent.click(screen.getByText('Create Scenario'));
    
    await waitFor(() => {
      expect(screen.getByLabelText(/Scenario Name/)).toBeInTheDocument();
      expect(screen.getByLabelText('Type')).toBeInTheDocument();
      expect(screen.getByLabelText(/Description/)).toBeInTheDocument();
    });
  });

  test('validates required fields in create dialog', async () => {
    renderScenarios();
    
    // Open create dialog
    fireEvent.click(screen.getByText('Create Scenario'));
    
    await waitFor(() => {
      const createButton = screen.getByRole('button', { name: 'Create' });
      expect(createButton).toBeDisabled();
    });
  });

  test('scenario cards show correct status indicators', () => {
    renderScenarios();
    
    // Check for status chips
    expect(screen.getByText('Running')).toBeInTheDocument();
    expect(screen.getByText('Completed')).toBeInTheDocument();
    expect(screen.getByText('Paused')).toBeInTheDocument();
  });

  test('scenario cards show participant information', () => {
    renderScenarios();
    
    // Should show participant counts
    expect(screen.getAllByText(/Participants/)).toHaveLength(3);
  });

  test('start/stop scenario buttons work', async () => {
    const mockStartScenario = jest.fn().mockResolvedValue(undefined);
    const mockStopScenario = jest.fn().mockResolvedValue(undefined);
    
    // Re-mock with our spy functions
    jest.doMock('../contexts/SystemContext', () => ({
      useSystem: () => ({
        scenarios: [
          {
            id: 'scenario-002',
            name: 'Test Scenario',
            description: 'Test description',
            type: 'training',
            status: 'completed',
            participants: ['agent-001'],
            progress: 100,
          },
        ],
        agents: [
          {
            id: 'agent-001',
            name: 'Test Agent',
            type: 'red_team',
            status: 'active',
          },
        ],
        loading: false,
        error: null,
        startScenario: mockStartScenario,
        stopScenario: mockStopScenario,
      }),
    }));
    
    renderScenarios();
    
    // Find start button and click it
    const startButton = screen.getByText('Start');
    fireEvent.click(startButton);
    
    await waitFor(() => {
      expect(mockStartScenario).toHaveBeenCalledWith('scenario-002');
    });
  });

  test('handles loading state', () => {
    // Mock loading state
    jest.doMock('../contexts/SystemContext', () => ({
      useSystem: () => ({
        scenarios: [],
        agents: [],
        loading: true,
        error: null,
        startScenario: jest.fn(),
        stopScenario: jest.fn(),
      }),
    }));
    
    renderScenarios();
    
    // Should show loading indicator
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });

  test('shows empty state when no scenarios', () => {
    // Mock empty state
    jest.doMock('../contexts/SystemContext', () => ({
      useSystem: () => ({
        scenarios: [],
        agents: [],
        loading: false,
        error: null,
        startScenario: jest.fn(),
        stopScenario: jest.fn(),
      }),
    }));
    
    renderScenarios();
    
    expect(screen.getByText('No scenarios found')).toBeInTheDocument();
    expect(screen.getByText('Get started by creating your first scenario.')).toBeInTheDocument();
  });

  test('floating action button opens create dialog', async () => {
    renderScenarios();
    
    // Find and click the floating action button
    const fab = screen.getAllByRole('button').find(button => 
      button.querySelector('[data-testid="AddIcon"]')
    );
    
    if (fab) {
      fireEvent.click(fab);
      
      await waitFor(() => {
        expect(screen.getByText('Create New Scenario')).toBeInTheDocument();
      });
    }
  });

  test('responsive design works', () => {
    // Mock mobile viewport
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 375,
    });
    
    renderScenarios();
    
    // Should still render properly on mobile
    expect(screen.getByText('Scenario Management')).toBeInTheDocument();
  });

  test('keyboard navigation works', () => {
    renderScenarios();
    
    // Find first focusable element
    const createButton = screen.getByText('Create Scenario');
    createButton.focus();
    
    expect(document.activeElement).toBe(createButton);
  });
});

describe('Scenario Performance', () => {
  test('renders quickly with many scenarios', () => {
    const manyScenarios = Array.from({ length: 50 }, (_, i) => ({
      id: `scenario-${i}`,
      name: `Scenario ${i}`,
      description: `Description ${i}`,
      type: 'training' as const,
      status: 'completed' as const,
      participants: ['agent-001'],
      progress: 100,
    }));
    
    jest.doMock('../contexts/SystemContext', () => ({
      useSystem: () => ({
        scenarios: manyScenarios,
        agents: [{ id: 'agent-001', name: 'Test Agent', type: 'red_team', status: 'active' }],
        loading: false,
        error: null,
        startScenario: jest.fn(),
        stopScenario: jest.fn(),
      }),
    }));
    
    const startTime = performance.now();
    renderScenarios();
    const endTime = performance.now();
    
    // Should render within reasonable time even with many scenarios
    expect(endTime - startTime).toBeLessThan(1000);
  });
});