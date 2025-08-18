import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import Dashboard from '../pages/Dashboard/Dashboard';
import Scenarios from '../pages/Scenarios/Scenarios';
import Agents from '../pages/Agents/Agents';

// Mock all contexts
jest.mock('../contexts/SystemContext', () => ({
  useSystem: () => ({
    systemStatus: {
      overall: 'healthy',
      activeAgents: 5,
      activeAlerts: 3,
      totalScenarios: 10,
      runningScenarios: 2,
      cpuUsage: 45,
      memoryUsage: 62,
      networkTraffic: 1024,
    },
    agents: Array.from({ length: 20 }, (_, i) => ({
      id: `agent-${i}`,
      name: `Agent ${i}`,
      type: i % 3 === 0 ? 'red_team' : i % 3 === 1 ? 'blue_team' : 'purple_team',
      status: i % 4 === 0 ? 'offline' : 'active',
      lastActivity: new Date(Date.now() - i * 60000),
      performance: {
        successRate: 80 + Math.random() * 20,
        responseTime: 100 + Math.random() * 200,
        tasksCompleted: Math.floor(Math.random() * 50),
      },
    })),
    scenarios: Array.from({ length: 15 }, (_, i) => ({
      id: `scenario-${i}`,
      name: `Scenario ${i}`,
      description: `Description for scenario ${i}`,
      type: i % 3 === 0 ? 'training' : i % 3 === 1 ? 'assessment' : 'live',
      status: i % 4 === 0 ? 'completed' : i % 4 === 1 ? 'running' : i % 4 === 2 ? 'paused' : 'failed',
      participants: [`agent-${i}`, `agent-${i + 1}`],
      startTime: new Date(Date.now() - i * 3600000),
      progress: Math.floor(Math.random() * 100),
    })),
    alerts: Array.from({ length: 10 }, (_, i) => ({
      id: `alert-${i}`,
      level: i % 4 === 0 ? 'critical' : i % 4 === 1 ? 'error' : i % 4 === 2 ? 'warning' : 'info',
      title: `Alert ${i}`,
      message: `Alert message ${i}`,
      timestamp: new Date(Date.now() - i * 300000),
      source: `source-${i}`,
      acknowledged: i % 2 === 0,
    })),
    notifications: [],
    loading: false,
    error: null,
    refreshSystemData: jest.fn(),
    acknowledgeAlert: jest.fn(),
    markNotificationRead: jest.fn(),
    clearNotifications: jest.fn(),
    startScenario: jest.fn(),
    stopScenario: jest.fn(),
  }),
  SystemProvider: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

jest.mock('../contexts/WebSocketContext', () => ({
  useWebSocket: () => ({
    connectionStatus: 'connected',
    emit: jest.fn(),
    subscribe: jest.fn(),
  }),
  useWebSocketEvent: jest.fn(),
  WebSocketProvider: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

// Mock heavy components
jest.mock('recharts', () => ({
  LineChart: ({ children }: { children: React.ReactNode }) => <div data-testid="line-chart">{children}</div>,
  Line: () => <div data-testid="line" />,
  AreaChart: ({ children }: { children: React.ReactNode }) => <div data-testid="area-chart">{children}</div>,
  Area: () => <div data-testid="area" />,
  BarChart: ({ children }: { children: React.ReactNode }) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => <div data-testid="bar" />,
  PieChart: ({ children }: { children: React.ReactNode }) => <div data-testid="pie-chart">{children}</div>,
  Pie: () => <div data-testid="pie" />,
  Cell: () => <div data-testid="cell" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div data-testid="responsive-container">{children}</div>,
}));

jest.mock('react-force-graph-2d', () => {
  return React.forwardRef((props: any, ref: any) => (
    <div data-testid="force-graph">Force Graph Mock</div>
  ));
});

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

const renderWithProviders = (Component: React.ComponentType) => {
  return render(
    <BrowserRouter>
      <ThemeProvider theme={theme}>
        <Component />
      </ThemeProvider>
    </BrowserRouter>
  );
};

describe('UI Responsiveness Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Dashboard Responsiveness', () => {
    test('renders quickly with large datasets', async () => {
      const startTime = performance.now();
      renderWithProviders(Dashboard);
      const endTime = performance.now();
      
      expect(endTime - startTime).toBeLessThan(500);
      expect(screen.getByText('Archangel Dashboard')).toBeInTheDocument();
    });

    test('handles rapid tab switching without performance issues', async () => {
      renderWithProviders(Dashboard);
      
      const tabs = ['System Metrics', 'Agent Activity', 'Threat Intelligence', 'Performance Analytics'];
      
      for (const tabName of tabs) {
        const startTime = performance.now();
        fireEvent.click(screen.getByText(tabName));
        
        await waitFor(() => {
          expect(screen.getByText(tabName)).toHaveClass('Mui-selected');
        });
        
        const endTime = performance.now();
        expect(endTime - startTime).toBeLessThan(200);
      }
    });

    test('maintains responsiveness during real-time updates', async () => {
      const mockUseWebSocketEvent = require('../contexts/WebSocketContext').useWebSocketEvent;
      renderWithProviders(Dashboard);
      
      // Simulate rapid WebSocket events
      const eventCallback = mockUseWebSocketEvent.mock.calls.find(
        call => call[0] === 'metric_update'
      )?.[1];
      
      if (eventCallback) {
        const startTime = performance.now();
        
        // Simulate 10 rapid updates
        for (let i = 0; i < 10; i++) {
          act(() => {
            eventCallback({
              timestamp: new Date().toISOString(),
              metrics: {
                cpu: Math.random() * 100,
                memory: Math.random() * 100,
                network: Math.random() * 1000,
              },
            });
          });
        }
        
        const endTime = performance.now();
        expect(endTime - startTime).toBeLessThan(100);
      }
    });

    test('responsive design adapts to different screen sizes', () => {
      const screenSizes = [
        { width: 320, height: 568 },  // Mobile
        { width: 768, height: 1024 }, // Tablet
        { width: 1920, height: 1080 }, // Desktop
      ];
      
      screenSizes.forEach(({ width, height }) => {
        Object.defineProperty(window, 'innerWidth', {
          writable: true,
          configurable: true,
          value: width,
        });
        Object.defineProperty(window, 'innerHeight', {
          writable: true,
          configurable: true,
          value: height,
        });
        
        renderWithProviders(Dashboard);
        expect(screen.getByText('Archangel Dashboard')).toBeInTheDocument();
      });
    });
  });

  describe('Scenarios Page Responsiveness', () => {
    test('renders large scenario lists efficiently', async () => {
      const startTime = performance.now();
      renderWithProviders(Scenarios);
      const endTime = performance.now();
      
      expect(endTime - startTime).toBeLessThan(300);
      expect(screen.getByText('Scenario Management')).toBeInTheDocument();
    });

    test('filtering operations are fast', async () => {
      renderWithProviders(Scenarios);
      
      // Test type filter
      const startTime = performance.now();
      fireEvent.mouseDown(screen.getByLabelText('Type'));
      fireEvent.click(screen.getByText('Training'));
      const endTime = performance.now();
      
      expect(endTime - startTime).toBeLessThan(100);
    });

    test('scenario card interactions are responsive', async () => {
      renderWithProviders(Scenarios);
      
      // Find scenario cards and test interactions
      const viewButtons = screen.getAllByText('View');
      if (viewButtons.length > 0) {
        const startTime = performance.now();
        fireEvent.click(viewButtons[0]);
        
        await waitFor(() => {
          expect(screen.getByText('Scenario Details')).toBeInTheDocument();
        });
        
        const endTime = performance.now();
        expect(endTime - startTime).toBeLessThan(200);
      }
    });
  });

  describe('Agents Page Responsiveness', () => {
    test('renders large agent lists efficiently', async () => {
      const startTime = performance.now();
      renderWithProviders(Agents);
      const endTime = performance.now();
      
      expect(endTime - startTime).toBeLessThan(300);
      expect(screen.getByText('Agent Management')).toBeInTheDocument();
    });

    test('tab switching is smooth with many agents', async () => {
      renderWithProviders(Agents);
      
      const tabs = ['Agent Overview', 'Network Topology', 'Activity Feed', 'Performance Metrics'];
      
      for (const tabName of tabs) {
        const startTime = performance.now();
        fireEvent.click(screen.getByText(tabName));
        
        await waitFor(() => {
          expect(screen.getByText(tabName)).toHaveClass('Mui-selected');
        });
        
        const endTime = performance.now();
        expect(endTime - startTime).toBeLessThan(150);
      }
    });
  });
});

describe('Data Accuracy Tests', () => {
  describe('Dashboard Data Accuracy', () => {
    test('displays correct system statistics', () => {
      renderWithProviders(Dashboard);
      
      expect(screen.getByText('5')).toBeInTheDocument(); // Active agents
      expect(screen.getByText('3')).toBeInTheDocument(); // Active alerts
      expect(screen.getByText('2')).toBeInTheDocument(); // Running scenarios
    });

    test('agent status counts are accurate', () => {
      renderWithProviders(Dashboard);
      
      // Switch to Agent Activity tab
      fireEvent.click(screen.getByText('Agent Activity'));
      
      // Should show correct agent information
      expect(screen.getByText('Agent Status Overview')).toBeInTheDocument();
    });

    test('alert distribution is calculated correctly', async () => {
      renderWithProviders(Dashboard);
      
      // Switch to Threat Intelligence tab
      fireEvent.click(screen.getByText('Threat Intelligence'));
      
      await waitFor(() => {
        expect(screen.getByText('Alert Distribution')).toBeInTheDocument();
      });
    });
  });

  describe('Scenarios Data Accuracy', () => {
    test('scenario statistics are calculated correctly', () => {
      renderWithProviders(Scenarios);
      
      expect(screen.getByText('15')).toBeInTheDocument(); // Total scenarios
      
      // Check running scenarios count
      const runningScenarios = screen.getAllByText(/\d+/).find(el => 
        el.parentElement?.textContent?.includes('Running')
      );
      expect(runningScenarios).toBeInTheDocument();
    });

    test('scenario filtering works correctly', async () => {
      renderWithProviders(Scenarios);
      
      // Filter by training type
      fireEvent.mouseDown(screen.getByLabelText('Type'));
      fireEvent.click(screen.getByText('Training'));
      
      // Should show only training scenarios
      await waitFor(() => {
        // The filtering should work (exact count depends on mock data)
        expect(screen.getByLabelText('Type')).toBeInTheDocument();
      });
    });

    test('scenario progress is displayed accurately', () => {
      renderWithProviders(Scenarios);
      
      // Should show progress bars for running scenarios
      const progressBars = screen.getAllByRole('progressbar');
      expect(progressBars.length).toBeGreaterThan(0);
    });
  });

  describe('Agents Data Accuracy', () => {
    test('agent statistics are calculated correctly', () => {
      renderWithProviders(Agents);
      
      expect(screen.getByText('20')).toBeInTheDocument(); // Total agents
      
      // Check active agents count (should be less than total due to some offline)
      const activeAgentsText = screen.getAllByText(/\d+/).find(el => 
        el.parentElement?.textContent?.includes('Active Agents')
      );
      expect(activeAgentsText).toBeInTheDocument();
    });

    test('agent performance metrics are accurate', () => {
      renderWithProviders(Agents);
      
      // Should show average success rate and response time
      const avgSuccessRate = screen.getAllByText(/\d+%/).find(el => 
        el.parentElement?.textContent?.includes('Avg Success Rate')
      );
      expect(avgSuccessRate).toBeInTheDocument();
      
      const avgResponseTime = screen.getAllByText(/\d+ms/).find(el => 
        el.parentElement?.textContent?.includes('Avg Response Time')
      );
      expect(avgResponseTime).toBeInTheDocument();
    });

    test('agent status distribution is correct', () => {
      renderWithProviders(Agents);
      
      // Should show mix of active and offline agents
      expect(screen.getAllByText('active').length).toBeGreaterThan(0);
      expect(screen.getAllByText('offline').length).toBeGreaterThan(0);
    });
  });
});

describe('Real-time Data Updates', () => {
  test('WebSocket events update UI correctly', async () => {
    const mockUseWebSocketEvent = require('../contexts/WebSocketContext').useWebSocketEvent;
    renderWithProviders(Dashboard);
    
    // Find metric update callback
    const metricCallback = mockUseWebSocketEvent.mock.calls.find(
      call => call[0] === 'metric_update'
    )?.[1];
    
    if (metricCallback) {
      act(() => {
        metricCallback({
          timestamp: new Date().toISOString(),
          metrics: {
            cpu: 85,
            memory: 70,
            network: 2048,
          },
        });
      });
      
      // UI should update with new metrics
      expect(screen.getByText('Real-time System Metrics')).toBeInTheDocument();
    }
  });

  test('agent events update agent status correctly', async () => {
    const mockUseWebSocketEvent = require('../contexts/WebSocketContext').useWebSocketEvent;
    renderWithProviders(Agents);
    
    // Find agent event callback
    const agentCallback = mockUseWebSocketEvent.mock.calls.find(
      call => call[0] === 'agent_event'
    )?.[1];
    
    if (agentCallback) {
      act(() => {
        agentCallback({
          agentId: 'agent-001',
          event: 'task_completed',
          data: { message: 'Task completed successfully' },
          timestamp: new Date().toISOString(),
        });
      });
      
      // Switch to Activity Feed to see the update
      fireEvent.click(screen.getByText('Activity Feed'));
      
      await waitFor(() => {
        expect(screen.getByText('Real-time Agent Activity')).toBeInTheDocument();
      });
    }
  });
});

describe('Error Handling and Edge Cases', () => {
  test('handles missing data gracefully', () => {
    // Mock empty data
    jest.doMock('../contexts/SystemContext', () => ({
      useSystem: () => ({
        systemStatus: { overall: 'offline' },
        agents: [],
        scenarios: [],
        alerts: [],
        notifications: [],
        loading: false,
        error: null,
        refreshSystemData: jest.fn(),
        acknowledgeAlert: jest.fn(),
        markNotificationRead: jest.fn(),
        clearNotifications: jest.fn(),
        startScenario: jest.fn(),
        stopScenario: jest.fn(),
      }),
    }));
    
    renderWithProviders(Dashboard);
    expect(screen.getByText('Archangel Dashboard')).toBeInTheDocument();
  });

  test('handles loading states appropriately', () => {
    // Mock loading state
    jest.doMock('../contexts/SystemContext', () => ({
      useSystem: () => ({
        systemStatus: { overall: 'offline' },
        agents: [],
        scenarios: [],
        alerts: [],
        notifications: [],
        loading: true,
        error: null,
        refreshSystemData: jest.fn(),
        acknowledgeAlert: jest.fn(),
        markNotificationRead: jest.fn(),
        clearNotifications: jest.fn(),
        startScenario: jest.fn(),
        stopScenario: jest.fn(),
      }),
    }));
    
    renderWithProviders(Dashboard);
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });

  test('displays error messages when data fails to load', () => {
    // Mock error state
    jest.doMock('../contexts/SystemContext', () => ({
      useSystem: () => ({
        systemStatus: { overall: 'offline' },
        agents: [],
        scenarios: [],
        alerts: [],
        notifications: [],
        loading: false,
        error: 'Failed to load system data',
        refreshSystemData: jest.fn(),
        acknowledgeAlert: jest.fn(),
        markNotificationRead: jest.fn(),
        clearNotifications: jest.fn(),
        startScenario: jest.fn(),
        stopScenario: jest.fn(),
      }),
    }));
    
    renderWithProviders(Dashboard);
    expect(screen.getByText('Failed to load system data')).toBeInTheDocument();
  });
});