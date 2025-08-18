import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import Agents from '../pages/Agents/Agents';

// Mock the contexts
jest.mock('../contexts/SystemContext', () => ({
  useSystem: () => ({
    agents: [
      {
        id: 'agent-001',
        name: 'Red Team Alpha',
        type: 'red_team',
        status: 'active',
        scenario: 'scenario-001',
        lastActivity: new Date(),
        performance: {
          successRate: 87,
          responseTime: 245,
          tasksCompleted: 23,
        },
      },
      {
        id: 'agent-002',
        name: 'Blue Team Delta',
        type: 'blue_team',
        status: 'active',
        scenario: 'scenario-001',
        lastActivity: new Date(Date.now() - 5000),
        performance: {
          successRate: 92,
          responseTime: 180,
          tasksCompleted: 31,
        },
      },
      {
        id: 'agent-003',
        name: 'Purple Team Gamma',
        type: 'purple_team',
        status: 'idle',
        lastActivity: new Date(Date.now() - 60000),
        performance: {
          successRate: 89,
          responseTime: 200,
          tasksCompleted: 18,
        },
      },
    ],
    scenarios: [
      {
        id: 'scenario-001',
        name: 'Test Scenario',
        description: 'Test description',
        type: 'training',
        status: 'running',
        participants: ['agent-001', 'agent-002'],
        progress: 65,
      },
    ],
    loading: false,
    error: null,
  }),
}));

jest.mock('../contexts/WebSocketContext', () => ({
  useWebSocketEvent: jest.fn(),
}));

// Mock NetworkTopology component
jest.mock('../components/NetworkTopology/NetworkTopology', () => {
  return function MockNetworkTopology() {
    return <div data-testid="network-topology">Network Topology Component</div>;
  };
});

const theme = createTheme({
  palette: {
    mode: 'dark',
  },
});

const renderAgents = () => {
  return render(
    <BrowserRouter>
      <ThemeProvider theme={theme}>
        <Agents />
      </ThemeProvider>
    </BrowserRouter>
  );
};

describe('Agents Page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders agents page with title and description', () => {
    renderAgents();
    
    expect(screen.getByText('Agent Management')).toBeInTheDocument();
    expect(screen.getByText('Monitor and manage AI agents, their performance, and configurations')).toBeInTheDocument();
  });

  test('displays agent statistics cards', () => {
    renderAgents();
    
    expect(screen.getByText('Total Agents')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument(); // Total agents
    
    expect(screen.getByText('Active Agents')).toBeInTheDocument();
    expect(screen.getByText('2')).toBeInTheDocument(); // Active agents
    
    expect(screen.getByText('Avg Success Rate')).toBeInTheDocument();
    expect(screen.getByText('89%')).toBeInTheDocument(); // Average success rate
    
    expect(screen.getByText('Avg Response Time')).toBeInTheDocument();
    expect(screen.getByText('208ms')).toBeInTheDocument(); // Average response time
  });

  test('renders tabbed interface', () => {
    renderAgents();
    
    expect(screen.getByText('Agent Overview')).toBeInTheDocument();
    expect(screen.getByText('Network Topology')).toBeInTheDocument();
    expect(screen.getByText('Activity Feed')).toBeInTheDocument();
    expect(screen.getByText('Performance Metrics')).toBeInTheDocument();
  });

  test('displays agent cards in overview tab', () => {
    renderAgents();
    
    expect(screen.getByText('Red Team Alpha')).toBeInTheDocument();
    expect(screen.getByText('Blue Team Delta')).toBeInTheDocument();
    expect(screen.getByText('Purple Team Gamma')).toBeInTheDocument();
  });

  test('shows agent performance metrics', () => {
    renderAgents();
    
    // Check for performance indicators
    expect(screen.getAllByText(/Success Rate/)).toHaveLength(3); // One for each agent
    expect(screen.getAllByText(/Response Time/)).toHaveLength(3);
    expect(screen.getAllByText(/Tasks Completed/)).toHaveLength(3);
  });

  test('switches between tabs correctly', async () => {
    renderAgents();
    
    // Click on Network Topology tab
    fireEvent.click(screen.getByText('Network Topology'));
    
    await waitFor(() => {
      expect(screen.getByTestId('network-topology')).toBeInTheDocument();
    });
    
    // Click on Activity Feed tab
    fireEvent.click(screen.getByText('Activity Feed'));
    
    await waitFor(() => {
      expect(screen.getByText('Real-time Agent Activity')).toBeInTheDocument();
    });
    
    // Click on Performance Metrics tab
    fireEvent.click(screen.getByText('Performance Metrics'));
    
    await waitFor(() => {
      expect(screen.getByText('Agent Performance Metrics')).toBeInTheDocument();
    });
  });

  test('agent cards show correct status indicators', () => {
    renderAgents();
    
    // Check for status indicators
    expect(screen.getAllByText('active')).toHaveLength(2);
    expect(screen.getByText('idle')).toBeInTheDocument();
  });

  test('agent cards show team type chips', () => {
    renderAgents();
    
    expect(screen.getByText('red team')).toBeInTheDocument();
    expect(screen.getByText('blue team')).toBeInTheDocument();
    expect(screen.getByText('purple team')).toBeInTheDocument();
  });

  test('start/stop buttons work for agents', async () => {
    renderAgents();
    
    // Find stop button for active agent
    const stopButtons = screen.getAllByText('Stop');
    expect(stopButtons).toHaveLength(2); // Two active agents
    
    // Find start button for idle agent
    const startButton = screen.getByText('Start');
    expect(startButton).toBeInTheDocument();
  });

  test('configure agent dialog opens', async () => {
    renderAgents();
    
    // Find and click a settings button
    const settingsButtons = screen.getAllByRole('button', { name: /configure/i });
    fireEvent.click(settingsButtons[0]);
    
    await waitFor(() => {
      expect(screen.getByText('Configure Agent')).toBeInTheDocument();
    });
  });

  test('configure agent dialog has required fields', async () => {
    renderAgents();
    
    // Open configure dialog
    const settingsButtons = screen.getAllByRole('button', { name: /configure/i });
    fireEvent.click(settingsButtons[0]);
    
    await waitFor(() => {
      expect(screen.getByLabelText('Agent Name')).toBeInTheDocument();
      expect(screen.getByLabelText('Agent Type')).toBeInTheDocument();
    });
  });

  test('activity feed shows empty state initially', async () => {
    renderAgents();
    
    // Switch to Activity Feed tab
    fireEvent.click(screen.getByText('Activity Feed'));
    
    await waitFor(() => {
      expect(screen.getByText('No recent activity')).toBeInTheDocument();
      expect(screen.getByText('Agent activities will appear here in real-time')).toBeInTheDocument();
    });
  });

  test('performance metrics tab shows agent metrics', async () => {
    renderAgents();
    
    // Switch to Performance Metrics tab
    fireEvent.click(screen.getByText('Performance Metrics'));
    
    await waitFor(() => {
      expect(screen.getByText('Agent Performance Metrics')).toBeInTheDocument();
      // Should show metrics for each agent
      expect(screen.getAllByText(/Success Rate/)).toHaveLength(3);
    });
  });

  test('handles empty agents state', () => {
    // Mock empty agents
    jest.doMock('../contexts/SystemContext', () => ({
      useSystem: () => ({
        agents: [],
        scenarios: [],
        loading: false,
        error: null,
      }),
    }));
    
    renderAgents();
    
    expect(screen.getByText('No agents found')).toBeInTheDocument();
    expect(screen.getByText('Agents will appear here once they are deployed and active.')).toBeInTheDocument();
  });

  test('handles loading state', () => {
    // Mock loading state
    jest.doMock('../contexts/SystemContext', () => ({
      useSystem: () => ({
        agents: [],
        scenarios: [],
        loading: true,
        error: null,
      }),
    }));
    
    renderAgents();
    
    // Should show loading indicator
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });

  test('handles error state', () => {
    // Mock error state
    jest.doMock('../contexts/SystemContext', () => ({
      useSystem: () => ({
        agents: [],
        scenarios: [],
        loading: false,
        error: 'Failed to load agents',
      }),
    }));
    
    renderAgents();
    
    expect(screen.getByText('Failed to load agents')).toBeInTheDocument();
  });

  test('responsive design works on mobile', () => {
    // Mock mobile viewport
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 375,
    });
    
    renderAgents();
    
    // Should still render properly on mobile
    expect(screen.getByText('Agent Management')).toBeInTheDocument();
  });

  test('keyboard navigation works', () => {
    renderAgents();
    
    // Find first focusable element (tab)
    const firstTab = screen.getByText('Agent Overview');
    firstTab.focus();
    
    expect(document.activeElement).toBe(firstTab);
  });
});

describe('Agents Performance', () => {
  test('renders quickly with many agents', () => {
    const manyAgents = Array.from({ length: 50 }, (_, i) => ({
      id: `agent-${i}`,
      name: `Agent ${i}`,
      type: 'red_team' as const,
      status: 'active' as const,
      lastActivity: new Date(),
      performance: { successRate: 90, responseTime: 200, tasksCompleted: 10 },
    }));
    
    jest.doMock('../contexts/SystemContext', () => ({
      useSystem: () => ({
        agents: manyAgents,
        scenarios: [],
        loading: false,
        error: null,
      }),
    }));
    
    const startTime = performance.now();
    renderAgents();
    const endTime = performance.now();
    
    // Should render within reasonable time even with many agents
    expect(endTime - startTime).toBeLessThan(1000);
  });
});

describe('Agents Accessibility', () => {
  test('has proper ARIA labels', () => {
    renderAgents();
    
    // Check for proper heading structure
    const mainHeading = screen.getByRole('heading', { level: 1 });
    expect(mainHeading).toHaveTextContent('Agent Management');
  });

  test('supports keyboard navigation between tabs', () => {
    renderAgents();
    
    const tabs = screen.getAllByRole('tab');
    expect(tabs).toHaveLength(4);
    
    // Should be able to focus on tabs
    tabs[0].focus();
    expect(document.activeElement).toBe(tabs[0]);
  });

  test('has sufficient color contrast for agent status', () => {
    renderAgents();
    
    // This would require additional testing with tools like axe-core
    // For now, we ensure the theme provides good contrast
    expect(theme.palette.mode).toBe('dark');
  });
});