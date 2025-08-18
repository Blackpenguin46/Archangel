import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { SystemProvider, useSystem } from '../contexts/SystemContext';
import axios from 'axios';

// Mock axios
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

// Test component that uses the system context
const TestComponent: React.FC = () => {
  const {
    systemStatus,
    agents,
    scenarios,
    alerts,
    notifications,
    loading,
    error,
    refreshSystemData,
    acknowledgeAlert,
    startScenario,
    stopScenario,
  } = useSystem();

  return (
    <div>
      <div data-testid="system-status">{systemStatus.overall}</div>
      <div data-testid="agents-count">{agents.length}</div>
      <div data-testid="scenarios-count">{scenarios.length}</div>
      <div data-testid="alerts-count">{alerts.length}</div>
      <div data-testid="loading">{loading.toString()}</div>
      <div data-testid="error">{error || 'no-error'}</div>
      
      <button onClick={refreshSystemData} data-testid="refresh-btn">
        Refresh
      </button>
      <button onClick={() => acknowledgeAlert('alert-1')} data-testid="ack-btn">
        Acknowledge Alert
      </button>
      <button onClick={() => startScenario('scenario-1')} data-testid="start-btn">
        Start Scenario
      </button>
      <button onClick={() => stopScenario('scenario-1')} data-testid="stop-btn">
        Stop Scenario
      </button>
    </div>
  );
};

const renderWithProvider = () => {
  return render(
    <SystemProvider>
      <TestComponent />
    </SystemProvider>
  );
};

describe('SystemContext', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Mock successful API responses
    mockedAxios.get.mockImplementation((url) => {
      if (url === '/api/system/status') {
        return Promise.resolve({
          data: {
            overall: 'healthy',
            activeAgents: 3,
            activeAlerts: 1,
            totalScenarios: 5,
            runningScenarios: 2,
            cpuUsage: 45,
            memoryUsage: 62,
            networkTraffic: 1024,
          },
        });
      }
      if (url === '/api/agents') {
        return Promise.resolve({
          data: [
            {
              id: 'agent-1',
              name: 'Test Agent',
              type: 'red_team',
              status: 'active',
              performance: { successRate: 90, responseTime: 200, tasksCompleted: 10 },
            },
          ],
        });
      }
      if (url === '/api/scenarios') {
        return Promise.resolve({
          data: [
            {
              id: 'scenario-1',
              name: 'Test Scenario',
              type: 'training',
              status: 'running',
              participants: ['agent-1'],
              progress: 50,
            },
          ],
        });
      }
      return Promise.reject(new Error('Unknown URL'));
    });

    mockedAxios.post.mockResolvedValue({ data: { success: true } });
  });

  test('provides initial state correctly', async () => {
    renderWithProvider();

    // Should start with loading state
    expect(screen.getByTestId('loading')).toHaveTextContent('true');

    // Wait for data to load
    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('false');
    });

    // Check that data was loaded
    expect(screen.getByTestId('system-status')).toHaveTextContent('healthy');
    expect(screen.getByTestId('agents-count')).toHaveTextContent('1');
    expect(screen.getByTestId('scenarios-count')).toHaveTextContent('1');
    expect(screen.getByTestId('error')).toHaveTextContent('no-error');
  });

  test('handles API errors gracefully', async () => {
    mockedAxios.get.mockRejectedValue(new Error('API Error'));

    renderWithProvider();

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('false');
    });

    // Should show error and fallback to mock data
    expect(screen.getByTestId('error')).toHaveTextContent('Failed to fetch system data');
    // Mock data should still be loaded
    expect(screen.getByTestId('system-status')).toHaveTextContent('healthy');
  });

  test('refreshSystemData function works', async () => {
    renderWithProvider();

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('false');
    });

    // Click refresh button
    fireEvent.click(screen.getByTestId('refresh-btn'));

    // Should show loading state again
    expect(screen.getByTestId('loading')).toHaveTextContent('true');

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('false');
    });

    // API should have been called again
    expect(mockedAxios.get).toHaveBeenCalledWith('/api/system/status');
  });

  test('acknowledgeAlert function works', async () => {
    renderWithProvider();

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('false');
    });

    // Add an alert to the context first
    // This would normally be done through WebSocket events
    fireEvent.click(screen.getByTestId('ack-btn'));

    // Should not throw any errors
    expect(screen.getByTestId('error')).toHaveTextContent('no-error');
  });

  test('startScenario function makes API call', async () => {
    renderWithProvider();

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('false');
    });

    fireEvent.click(screen.getByTestId('start-btn'));

    await waitFor(() => {
      expect(mockedAxios.post).toHaveBeenCalledWith('/api/scenarios/scenario-1/start');
    });
  });

  test('stopScenario function makes API call', async () => {
    renderWithProvider();

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('false');
    });

    fireEvent.click(screen.getByTestId('stop-btn'));

    await waitFor(() => {
      expect(mockedAxios.post).toHaveBeenCalledWith('/api/scenarios/scenario-1/stop');
    });
  });

  test('handles notification management', async () => {
    const { rerender } = renderWithProvider();

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('false');
    });

    // The context should handle notifications properly
    expect(screen.getByTestId('error')).toHaveTextContent('no-error');
  });

  test('context throws error when used outside provider', () => {
    // Temporarily mock console.error to avoid test output pollution
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

    expect(() => {
      render(<TestComponent />);
    }).toThrow('useSystem must be used within a SystemProvider');

    consoleSpy.mockRestore();
  });

  test('handles concurrent API calls', async () => {
    renderWithProvider();

    // Make multiple concurrent calls
    fireEvent.click(screen.getByTestId('refresh-btn'));
    fireEvent.click(screen.getByTestId('refresh-btn'));
    fireEvent.click(screen.getByTestId('refresh-btn'));

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('false');
    });

    // Should handle concurrent calls gracefully
    expect(screen.getByTestId('error')).toHaveTextContent('no-error');
  });

  test('periodic refresh works', async () => {
    jest.useFakeTimers();
    
    renderWithProvider();

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('false');
    });

    // Clear the initial API calls
    mockedAxios.get.mockClear();

    // Fast forward 30 seconds (the refresh interval)
    jest.advanceTimersByTime(30000);

    await waitFor(() => {
      expect(mockedAxios.get).toHaveBeenCalledWith('/api/system/status');
    });

    jest.useRealTimers();
  });
});