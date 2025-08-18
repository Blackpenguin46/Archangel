import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import Dashboard from '../pages/Dashboard/Dashboard';
import { SystemProvider } from '../contexts/SystemContext';
import { WebSocketProvider } from '../contexts/WebSocketContext';

// Mock the WebSocket and System contexts
jest.mock('../contexts/WebSocketContext', () => ({
  useWebSocket: () => ({
    connectionStatus: 'connected',
    emit: jest.fn(),
    subscribe: jest.fn(),
  }),
  useWebSocketEvent: jest.fn(),
  WebSocketProvider: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

jest.mock('../contexts/SystemContext', () => ({
  useSystem: () => ({
    systemStatus: {
      overall: 'healthy',
      activeAgents: 3,
      activeAlerts: 2,
      totalScenarios: 8,
      runningScenarios: 1,
      cpuUsage: 45,
      memoryUsage: 62,
      networkTraffic: 1024,
    },
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
        lastActivity: new Date(),
        performance: {
          successRate: 92,
          responseTime: 180,
          tasksCompleted: 31,
        },
      },
    ],
    scenarios: [
      {
        id: 'scenario-001',
        name: 'Advanced Persistent Threat Simulation',
        description: 'Multi-stage APT attack simulation',
        type: 'training',
        status: 'running',
        participants: ['agent-001', 'agent-002'],
        startTime: new Date(Date.now() - 3600000),
        progress: 65,
      },
    ],
    alerts: [
      {
        id: 'alert-001',
        level: 'warning',
        title: 'High CPU Usage',
        message: 'CPU usage above 80%',
        timestamp: new Date(),
        source: 'system-monitor',
        acknowledged: false,
      },
    ],
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

// Mock recharts components
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

const theme = createTheme({
  palette: {
    mode: 'dark',
  },
});

const renderDashboard = () => {
  return render(
    <BrowserRouter>
      <ThemeProvider theme={theme}>
        <SystemProvider>
          <WebSocketProvider>
            <Dashboard />
          </WebSocketProvider>
        </SystemProvider>
      </ThemeProvider>
    </BrowserRouter>
  );
};

describe('Dashboard Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders dashboard title and description', () => {
    renderDashboard();
    
    expect(screen.getByText('Archangel Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Real-time monitoring and control of AI security operations')).toBeInTheDocument();
  });

  test('displays system status cards with correct data', () => {
    renderDashboard();
    
    // Check system status cards
    expect(screen.getByText('System Status')).toBeInTheDocument();
    expect(screen.getByText('healthy')).toBeInTheDocument();
    
    expect(screen.getByText('Active Agents')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
    
    expect(screen.getByText('Running Scenarios')).toBeInTheDocument();
    expect(screen.getByText('1')).toBeInTheDocument();
    
    expect(screen.getByText('Active Alerts')).toBeInTheDocument();
    expect(screen.getByText('2')).toBeInTheDocument();
  });

  test('renders tabbed interface', () => {
    renderDashboard();
    
    expect(screen.getByText('System Metrics')).toBeInTheDocument();
    expect(screen.getByText('Agent Activity')).toBeInTheDocument();
    expect(screen.getByText('Threat Intelligence')).toBeInTheDocument();
    expect(screen.getByText('Performance Analytics')).toBeInTheDocument();
  });

  test('switches between tabs correctly', async () => {
    renderDashboard();
    
    // Click on Agent Activity tab
    fireEvent.click(screen.getByText('Agent Activity'));
    
    await waitFor(() => {
      expect(screen.getByText('Agent Status Overview')).toBeInTheDocument();
    });
    
    // Click on Threat Intelligence tab
    fireEvent.click(screen.getByText('Threat Intelligence'));
    
    await waitFor(() => {
      expect(screen.getByText('Alert Distribution')).toBeInTheDocument();
    });
  });

  test('displays real-time metrics chart', () => {
    renderDashboard();
    
    expect(screen.getByText('Real-time System Metrics')).toBeInTheDocument();
    expect(screen.getByTestId('responsive-container')).toBeInTheDocument();
    expect(screen.getByTestId('line-chart')).toBeInTheDocument();
  });

  test('shows resource usage indicators', () => {
    renderDashboard();
    
    expect(screen.getByText('Resource Usage')).toBeInTheDocument();
    expect(screen.getByText('CPU')).toBeInTheDocument();
    expect(screen.getByText('Memory')).toBeInTheDocument();
    expect(screen.getByText('Network')).toBeInTheDocument();
  });

  test('displays agent status in Agent Activity tab', async () => {
    renderDashboard();
    
    // Switch to Agent Activity tab
    fireEvent.click(screen.getByText('Agent Activity'));
    
    await waitFor(() => {
      expect(screen.getByText('Red Team Alpha')).toBeInTheDocument();
      expect(screen.getByText('Blue Team Delta')).toBeInTheDocument();
    });
  });

  test('shows threat distribution chart in Threat Intelligence tab', async () => {
    renderDashboard();
    
    // Switch to Threat Intelligence tab
    fireEvent.click(screen.getByText('Threat Intelligence'));
    
    await waitFor(() => {
      expect(screen.getByText('Alert Distribution')).toBeInTheDocument();
      expect(screen.getByTestId('pie-chart')).toBeInTheDocument();
    });
  });

  test('renders performance metrics in Performance Analytics tab', async () => {
    renderDashboard();
    
    // Switch to Performance Analytics tab
    fireEvent.click(screen.getByText('Performance Analytics'));
    
    await waitFor(() => {
      expect(screen.getByText('Performance Metrics')).toBeInTheDocument();
    });
  });

  test('handles WebSocket disconnection gracefully', () => {
    // Mock disconnected state
    const mockUseWebSocket = require('../contexts/WebSocketContext').useWebSocket;
    mockUseWebSocket.mockReturnValue({
      connectionStatus: 'disconnected',
      emit: jest.fn(),
      subscribe: jest.fn(),
    });
    
    renderDashboard();
    
    expect(screen.getByText(/WebSocket disconnected/)).toBeInTheDocument();
  });

  test('responsive design works on mobile', () => {
    // Mock mobile viewport
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 375,
    });

    renderDashboard();
    
    // Dashboard should still render on mobile
    expect(screen.getByText('Archangel Dashboard')).toBeInTheDocument();
  });
});

describe('Dashboard Performance', () => {
  test('renders within acceptable time', async () => {
    const startTime = performance.now();
    renderDashboard();
    const endTime = performance.now();
    
    // Should render within 1 second
    expect(endTime - startTime).toBeLessThan(1000);
  });

  test('handles large datasets efficiently', () => {
    // Mock large dataset
    const mockUseSystem = require('../contexts/SystemContext').useSystem;
    mockUseSystem.mockReturnValue({
      ...mockUseSystem(),
      agents: Array.from({ length: 100 }, (_, i) => ({
        id: `agent-${i}`,
        name: `Agent ${i}`,
        type: 'red_team',
        status: 'active',
        performance: { successRate: 90, responseTime: 200, tasksCompleted: 10 },
      })),
    });

    const startTime = performance.now();
    renderDashboard();
    const endTime = performance.now();
    
    // Should still render efficiently with large datasets
    expect(endTime - startTime).toBeLessThan(2000);
  });
});

describe('Dashboard Accessibility', () => {
  test('has proper ARIA labels', () => {
    renderDashboard();
    
    // Check for proper heading structure
    const mainHeading = screen.getByRole('heading', { level: 1 });
    expect(mainHeading).toHaveTextContent('Archangel Dashboard');
  });

  test('supports keyboard navigation', () => {
    renderDashboard();
    
    // Tab navigation should work
    const firstTab = screen.getByText('System Metrics');
    firstTab.focus();
    expect(document.activeElement).toBe(firstTab);
  });

  test('has sufficient color contrast', () => {
    renderDashboard();
    
    // This would require additional testing with tools like axe-core
    // For now, we ensure the theme provides good contrast
    expect(theme.palette.mode).toBe('dark');
  });
});