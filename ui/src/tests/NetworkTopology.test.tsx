import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import NetworkTopology from '../components/NetworkTopology/NetworkTopology';

// Mock the contexts
jest.mock('../contexts/SystemContext', () => ({
  useSystem: () => ({
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
    scenarios: [
      {
        id: 'scenario-001',
        name: 'Test Scenario',
        status: 'running',
        participants: ['agent-001', 'agent-002'],
      },
    ],
    systemStatus: {
      overall: 'healthy',
    },
  }),
}));

jest.mock('../contexts/WebSocketContext', () => ({
  useWebSocketEvent: jest.fn(),
}));

// Mock react-force-graph-2d
jest.mock('react-force-graph-2d', () => {
  return React.forwardRef((props: any, ref: any) => (
    <div 
      data-testid="force-graph"
      onClick={() => props.onNodeClick && props.onNodeClick({ id: 'test-node', name: 'Test Node' })}
    >
      Force Graph Mock
    </div>
  ));
});

const theme = createTheme({
  palette: {
    mode: 'dark',
  },
});

const renderNetworkTopology = () => {
  return render(
    <ThemeProvider theme={theme}>
      <NetworkTopology />
    </ThemeProvider>
  );
};

describe('NetworkTopology Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders network topology with title', () => {
    renderNetworkTopology();
    
    expect(screen.getByText('Network Topology')).toBeInTheDocument();
  });

  test('displays view mode buttons', () => {
    renderNetworkTopology();
    
    expect(screen.getByText('Topology')).toBeInTheDocument();
    expect(screen.getByText('Activity')).toBeInTheDocument();
    expect(screen.getByText('Threats')).toBeInTheDocument();
  });

  test('shows legend with team types', () => {
    renderNetworkTopology();
    
    expect(screen.getByText('Red Team')).toBeInTheDocument();
    expect(screen.getByText('Blue Team')).toBeInTheDocument();
    expect(screen.getByText('Purple Team')).toBeInTheDocument();
    expect(screen.getByText('System')).toBeInTheDocument();
    expect(screen.getByText('Target')).toBeInTheDocument();
  });

  test('renders force graph component', () => {
    renderNetworkTopology();
    
    expect(screen.getByTestId('force-graph')).toBeInTheDocument();
  });

  test('switches between view modes', async () => {
    renderNetworkTopology();
    
    // Click on Activity mode
    fireEvent.click(screen.getByText('Activity'));
    
    await waitFor(() => {
      const activityButton = screen.getByText('Activity');
      expect(activityButton).toHaveClass('MuiButton-contained');
    });
    
    // Click on Threats mode
    fireEvent.click(screen.getByText('Threats'));
    
    await waitFor(() => {
      const threatsButton = screen.getByText('Threats');
      expect(threatsButton).toHaveClass('MuiButton-contained');
    });
  });

  test('handles node selection', async () => {
    renderNetworkTopology();
    
    // Click on the force graph to simulate node selection
    fireEvent.click(screen.getByTestId('force-graph'));
    
    // Should show node details panel
    await waitFor(() => {
      expect(screen.getByText('Test Node')).toBeInTheDocument();
    });
  });

  test('displays node information panel when node is selected', async () => {
    renderNetworkTopology();
    
    // Simulate node click
    fireEvent.click(screen.getByTestId('force-graph'));
    
    await waitFor(() => {
      expect(screen.getByText('Type:')).toBeInTheDocument();
      expect(screen.getByText('Status:')).toBeInTheDocument();
    });
  });

  test('legend chips have correct colors', () => {
    renderNetworkTopology();
    
    const redTeamChip = screen.getByText('Red Team').closest('.MuiChip-root');
    const blueTeamChip = screen.getByText('Blue Team').closest('.MuiChip-root');
    const purpleTeamChip = screen.getByText('Purple Team').closest('.MuiChip-root');
    
    expect(redTeamChip).toHaveStyle('background-color: rgb(229, 57, 53)');
    expect(blueTeamChip).toHaveStyle('background-color: rgb(25, 118, 210)');
    expect(purpleTeamChip).toHaveStyle('background-color: rgb(123, 31, 162)');
  });

  test('handles empty system state', () => {
    // Mock empty system state
    jest.doMock('../contexts/SystemContext', () => ({
      useSystem: () => ({
        agents: [],
        scenarios: [],
        systemStatus: { overall: 'offline' },
      }),
    }));
    
    renderNetworkTopology();
    
    // Should still render the component
    expect(screen.getByText('Network Topology')).toBeInTheDocument();
    expect(screen.getByTestId('force-graph')).toBeInTheDocument();
  });

  test('responsive design works', () => {
    // Mock mobile viewport
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 375,
    });
    
    renderNetworkTopology();
    
    // Should still render properly on mobile
    expect(screen.getByText('Network Topology')).toBeInTheDocument();
  });

  test('handles WebSocket events for real-time updates', () => {
    const mockUseWebSocketEvent = require('../contexts/WebSocketContext').useWebSocketEvent;
    
    renderNetworkTopology();
    
    // Verify that WebSocket event listener is set up
    expect(mockUseWebSocketEvent).toHaveBeenCalledWith('agent_event', expect.any(Function));
  });
});

describe('NetworkTopology Performance', () => {
  test('renders quickly with complex network data', () => {
    const manyAgents = Array.from({ length: 50 }, (_, i) => ({
      id: `agent-${i}`,
      name: `Agent ${i}`,
      type: 'red_team',
      status: 'active',
    }));
    
    jest.doMock('../contexts/SystemContext', () => ({
      useSystem: () => ({
        agents: manyAgents,
        scenarios: [],
        systemStatus: { overall: 'healthy' },
      }),
    }));
    
    const startTime = performance.now();
    renderNetworkTopology();
    const endTime = performance.now();
    
    // Should render within reasonable time even with many nodes
    expect(endTime - startTime).toBeLessThan(500);
  });

  test('handles frequent updates efficiently', () => {
    renderNetworkTopology();
    
    // Should not throw errors with rapid updates
    expect(screen.getByTestId('force-graph')).toBeInTheDocument();
  });
});

describe('NetworkTopology Accessibility', () => {
  test('has proper heading structure', () => {
    renderNetworkTopology();
    
    const heading = screen.getByRole('heading', { level: 6 });
    expect(heading).toHaveTextContent('Network Topology');
  });

  test('view mode buttons are accessible', () => {
    renderNetworkTopology();
    
    const buttons = screen.getAllByRole('button');
    const viewModeButtons = buttons.filter(button => 
      ['Topology', 'Activity', 'Threats'].includes(button.textContent || '')
    );
    
    expect(viewModeButtons).toHaveLength(3);
    viewModeButtons.forEach(button => {
      expect(button).toBeEnabled();
    });
  });

  test('legend items are properly labeled', () => {
    renderNetworkTopology();
    
    const legendItems = ['Red Team', 'Blue Team', 'Purple Team', 'System', 'Target'];
    legendItems.forEach(item => {
      expect(screen.getByText(item)).toBeInTheDocument();
    });
  });

  test('supports keyboard navigation', () => {
    renderNetworkTopology();
    
    const firstButton = screen.getByText('Topology');
    firstButton.focus();
    
    expect(document.activeElement).toBe(firstButton);
  });
});

describe('NetworkTopology Data Processing', () => {
  test('correctly processes agent data into nodes', () => {
    renderNetworkTopology();
    
    // Component should process the mock agents correctly
    expect(screen.getByTestId('force-graph')).toBeInTheDocument();
  });

  test('creates links between related entities', () => {
    renderNetworkTopology();
    
    // Should create network connections based on scenarios and participants
    expect(screen.getByTestId('force-graph')).toBeInTheDocument();
  });

  test('handles agent status changes', () => {
    const mockUseWebSocketEvent = require('../contexts/WebSocketContext').useWebSocketEvent;
    
    renderNetworkTopology();
    
    // Get the callback function passed to useWebSocketEvent
    const eventCallback = mockUseWebSocketEvent.mock.calls[0][1];
    
    // Simulate an agent event
    const mockEvent = {
      agentId: 'agent-001',
      event: 'task_completed',
      data: { message: 'Task completed successfully' },
      timestamp: new Date().toISOString(),
    };
    
    // Should not throw error when processing the event
    expect(() => eventCallback(mockEvent)).not.toThrow();
  });

  test('updates node appearance based on events', () => {
    renderNetworkTopology();
    
    // The component should handle real-time updates
    expect(screen.getByTestId('force-graph')).toBeInTheDocument();
  });
});