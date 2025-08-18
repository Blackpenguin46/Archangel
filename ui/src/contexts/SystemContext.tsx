import React, { createContext, useContext, useReducer, useEffect } from 'react';
import axios from 'axios';

// Types for system state
interface SystemStatus {
  overall: 'healthy' | 'warning' | 'critical' | 'offline';
  activeAgents: number;
  activeAlerts: number;
  totalScenarios: number;
  runningScenarios: number;
  cpuUsage: number;
  memoryUsage: number;
  networkTraffic: number;
}

interface Agent {
  id: string;
  name: string;
  type: 'red_team' | 'blue_team' | 'purple_team';
  status: 'active' | 'idle' | 'offline' | 'error';
  scenario?: string;
  lastActivity: Date;
  performance: {
    successRate: number;
    responseTime: number;
    tasksCompleted: number;
  };
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

interface Alert {
  id: string;
  level: 'info' | 'warning' | 'error' | 'critical';
  title: string;
  message: string;
  timestamp: Date;
  source: string;
  acknowledged: boolean;
}

interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: Date;
  read: boolean;
}

interface SystemState {
  systemStatus: SystemStatus;
  agents: Agent[];
  scenarios: Scenario[];
  alerts: Alert[];
  notifications: Notification[];
  loading: boolean;
  error: string | null;
}

// Action types
type SystemAction =
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'UPDATE_SYSTEM_STATUS'; payload: SystemStatus }
  | { type: 'UPDATE_AGENTS'; payload: Agent[] }
  | { type: 'UPDATE_SCENARIOS'; payload: Scenario[] }
  | { type: 'ADD_ALERT'; payload: Alert }
  | { type: 'ACKNOWLEDGE_ALERT'; payload: string }
  | { type: 'ADD_NOTIFICATION'; payload: Notification }
  | { type: 'MARK_NOTIFICATION_READ'; payload: string }
  | { type: 'CLEAR_NOTIFICATIONS' };

// Initial state
const initialState: SystemState = {
  systemStatus: {
    overall: 'offline',
    activeAgents: 0,
    activeAlerts: 0,
    totalScenarios: 0,
    runningScenarios: 0,
    cpuUsage: 0,
    memoryUsage: 0,
    networkTraffic: 0,
  },
  agents: [],
  scenarios: [],
  alerts: [],
  notifications: [],
  loading: true,
  error: null,
};

// Reducer
const systemReducer = (state: SystemState, action: SystemAction): SystemState => {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, loading: action.payload };
    
    case 'SET_ERROR':
      return { ...state, error: action.payload, loading: false };
    
    case 'UPDATE_SYSTEM_STATUS':
      return { ...state, systemStatus: action.payload };
    
    case 'UPDATE_AGENTS':
      return { ...state, agents: action.payload };
    
    case 'UPDATE_SCENARIOS':
      return { ...state, scenarios: action.payload };
    
    case 'ADD_ALERT':
      return {
        ...state,
        alerts: [action.payload, ...state.alerts],
        systemStatus: {
          ...state.systemStatus,
          activeAlerts: state.systemStatus.activeAlerts + 1,
        },
      };
    
    case 'ACKNOWLEDGE_ALERT':
      return {
        ...state,
        alerts: state.alerts.map(alert =>
          alert.id === action.payload
            ? { ...alert, acknowledged: true }
            : alert
        ),
        systemStatus: {
          ...state.systemStatus,
          activeAlerts: Math.max(0, state.systemStatus.activeAlerts - 1),
        },
      };
    
    case 'ADD_NOTIFICATION':
      return {
        ...state,
        notifications: [action.payload, ...state.notifications.slice(0, 49)], // Keep last 50
      };
    
    case 'MARK_NOTIFICATION_READ':
      return {
        ...state,
        notifications: state.notifications.map(notification =>
          notification.id === action.payload
            ? { ...notification, read: true }
            : notification
        ),
      };
    
    case 'CLEAR_NOTIFICATIONS':
      return {
        ...state,
        notifications: [],
      };
    
    default:
      return state;
  }
};

// Context
interface SystemContextType extends SystemState {
  refreshSystemData: () => Promise<void>;
  acknowledgeAlert: (alertId: string) => void;
  markNotificationRead: (notificationId: string) => void;
  clearNotifications: () => void;
  startScenario: (scenarioId: string) => Promise<void>;
  stopScenario: (scenarioId: string) => Promise<void>;
}

const SystemContext = createContext<SystemContextType | undefined>(undefined);

// Provider component
interface SystemProviderProps {
  children: React.ReactNode;
}

export const SystemProvider: React.FC<SystemProviderProps> = ({ children }) => {
  const [state, dispatch] = useReducer(systemReducer, initialState);

  // API functions
  const refreshSystemData = async () => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      
      // Fetch system status
      const statusResponse = await axios.get('/api/system/status');
      dispatch({ type: 'UPDATE_SYSTEM_STATUS', payload: statusResponse.data });
      
      // Fetch agents
      const agentsResponse = await axios.get('/api/agents');
      dispatch({ type: 'UPDATE_AGENTS', payload: agentsResponse.data });
      
      // Fetch scenarios
      const scenariosResponse = await axios.get('/api/scenarios');
      dispatch({ type: 'UPDATE_SCENARIOS', payload: scenariosResponse.data });
      
      dispatch({ type: 'SET_ERROR', payload: null });
    } catch (error) {
      console.error('Failed to refresh system data:', error);
      dispatch({ type: 'SET_ERROR', payload: 'Failed to fetch system data' });
      
      // Use mock data in case of API failure
      dispatch({ type: 'UPDATE_SYSTEM_STATUS', payload: getMockSystemStatus() });
      dispatch({ type: 'UPDATE_AGENTS', payload: getMockAgents() });
      dispatch({ type: 'UPDATE_SCENARIOS', payload: getMockScenarios() });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  };

  const acknowledgeAlert = (alertId: string) => {
    dispatch({ type: 'ACKNOWLEDGE_ALERT', payload: alertId });
  };

  const markNotificationRead = (notificationId: string) => {
    dispatch({ type: 'MARK_NOTIFICATION_READ', payload: notificationId });
  };

  const clearNotifications = () => {
    dispatch({ type: 'CLEAR_NOTIFICATIONS' });
  };

  const startScenario = async (scenarioId: string) => {
    try {
      await axios.post(`/api/scenarios/${scenarioId}/start`);
      await refreshSystemData();
      
      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: Date.now().toString(),
          type: 'success',
          title: 'Scenario Started',
          message: `Scenario ${scenarioId} has been started successfully.`,
          timestamp: new Date(),
          read: false,
        },
      });
    } catch (error) {
      console.error('Failed to start scenario:', error);
      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: Date.now().toString(),
          type: 'error',
          title: 'Failed to Start Scenario',
          message: `Failed to start scenario ${scenarioId}.`,
          timestamp: new Date(),
          read: false,
        },
      });
    }
  };

  const stopScenario = async (scenarioId: string) => {
    try {
      await axios.post(`/api/scenarios/${scenarioId}/stop`);
      await refreshSystemData();
      
      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: Date.now().toString(),
          type: 'info',
          title: 'Scenario Stopped',
          message: `Scenario ${scenarioId} has been stopped.`,
          timestamp: new Date(),
          read: false,
        },
      });
    } catch (error) {
      console.error('Failed to stop scenario:', error);
      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: Date.now().toString(),
          type: 'error',
          title: 'Failed to Stop Scenario',
          message: `Failed to stop scenario ${scenarioId}.`,
          timestamp: new Date(),
          read: false,
        },
      });
    }
  };

  // Initialize system data
  useEffect(() => {
    refreshSystemData();
    
    // Set up periodic refresh
    const interval = setInterval(refreshSystemData, 30000); // Every 30 seconds
    
    return () => clearInterval(interval);
  }, []);

  const contextValue: SystemContextType = {
    ...state,
    refreshSystemData,
    acknowledgeAlert,
    markNotificationRead,
    clearNotifications,
    startScenario,
    stopScenario,
  };

  return (
    <SystemContext.Provider value={contextValue}>
      {children}
    </SystemContext.Provider>
  );
};

// Hook to use system context
export const useSystem = (): SystemContextType => {
  const context = useContext(SystemContext);
  if (context === undefined) {
    throw new Error('useSystem must be used within a SystemProvider');
  }
  return context;
};

// Mock data functions for development/fallback
const getMockSystemStatus = (): SystemStatus => ({
  overall: 'healthy',
  activeAgents: 3,
  activeAlerts: 2,
  totalScenarios: 8,
  runningScenarios: 1,
  cpuUsage: 45,
  memoryUsage: 62,
  networkTraffic: 1024,
});

const getMockAgents = (): Agent[] => [
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
];

const getMockScenarios = (): Scenario[] => [
  {
    id: 'scenario-001',
    name: 'Advanced Persistent Threat Simulation',
    description: 'Multi-stage APT attack simulation with lateral movement',
    type: 'training',
    status: 'running',
    participants: ['agent-001', 'agent-002'],
    startTime: new Date(Date.now() - 3600000), // 1 hour ago
    progress: 65,
  },
  {
    id: 'scenario-002',
    name: 'Phishing Campaign Assessment',
    description: 'Email-based social engineering attack simulation',
    type: 'assessment',
    status: 'completed',
    participants: ['agent-003'],
    startTime: new Date(Date.now() - 7200000), // 2 hours ago
    endTime: new Date(Date.now() - 1800000), // 30 minutes ago
    progress: 100,
  },
];