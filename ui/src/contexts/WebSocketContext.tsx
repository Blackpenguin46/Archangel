import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';
import toast from 'react-hot-toast';

// Types for WebSocket events
interface SystemEvent {
  type: 'system_status' | 'agent_update' | 'scenario_update' | 'alert' | 'log' | 'metric_update';
  data: any;
  timestamp: string;
}

interface AgentEvent {
  agentId: string;
  event: 'started' | 'stopped' | 'task_completed' | 'error' | 'status_change';
  data: any;
  timestamp: string;
}

interface ScenarioEvent {
  scenarioId: string;
  event: 'started' | 'stopped' | 'paused' | 'progress_update' | 'completed';
  data: any;
  timestamp: string;
}

interface LogEvent {
  level: 'debug' | 'info' | 'warning' | 'error' | 'critical';
  component: string;
  message: string;
  timestamp: string;
  metadata?: any;
}

interface MetricUpdate {
  metrics: {
    cpu: number;
    memory: number;
    network: number;
    disk: number;
    activeConnections: number;
    throughput: number;
  };
  timestamp: string;
}

// Connection states
type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

interface WebSocketState {
  socket: Socket | null;
  connectionStatus: ConnectionStatus;
  lastActivity: Date | null;
  reconnectAttempts: number;
  isReconnecting: boolean;
}

// Context interface
interface WebSocketContextType extends WebSocketState {
  connect: () => void;
  disconnect: () => void;
  emit: (event: string, data: any) => void;
  subscribe: (event: string, callback: (data: any) => void) => () => void;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

// Provider component
interface WebSocketProviderProps {
  children: React.ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const [state, setState] = useState<WebSocketState>({
    socket: null,
    connectionStatus: 'disconnected',
    lastActivity: null,
    reconnectAttempts: 0,
    isReconnecting: false,
  });

  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const eventListenersRef = useRef<Map<string, Set<(data: any) => void>>>(new Map());

  // Configuration
  const WEBSOCKET_URL = process.env.REACT_APP_WEBSOCKET_URL || 'ws://localhost:8888';
  const MAX_RECONNECT_ATTEMPTS = 10;
  const RECONNECT_INTERVAL = 5000; // 5 seconds
  const HEARTBEAT_INTERVAL = 30000; // 30 seconds

  const connect = () => {
    if (state.socket?.connected) {
      return;
    }

    setState(prev => ({ ...prev, connectionStatus: 'connecting' }));

    try {
      const socket = io(WEBSOCKET_URL, {
        transports: ['websocket', 'polling'],
        timeout: 10000,
        forceNew: true,
        reconnection: false, // We handle reconnection manually
      });

      // Connection events
      socket.on('connect', () => {
        console.log('WebSocket connected');
        setState(prev => ({
          ...prev,
          connectionStatus: 'connected',
          lastActivity: new Date(),
          reconnectAttempts: 0,
          isReconnecting: false,
        }));

        toast.success('Connected to Archangel System', {
          duration: 2000,
          icon: '🔗',
        });

        // Start heartbeat
        startHeartbeat(socket);
      });

      socket.on('disconnect', (reason) => {
        console.log('WebSocket disconnected:', reason);
        setState(prev => ({
          ...prev,
          connectionStatus: 'disconnected',
          lastActivity: new Date(),
        }));

        stopHeartbeat();

        // Attempt reconnection if not manually disconnected
        if (reason !== 'io client disconnect') {
          handleReconnection();
        }
      });

      socket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error);
        setState(prev => ({
          ...prev,
          connectionStatus: 'error',
          lastActivity: new Date(),
        }));

        handleReconnection();
      });

      // System events
      socket.on('system_event', (event: SystemEvent) => {
        setState(prev => ({ ...prev, lastActivity: new Date() }));
        notifyListeners('system_event', event);
        
        // Handle specific system events
        if (event.type === 'alert' && event.data.level === 'critical') {
          toast.error(`Critical Alert: ${event.data.message}`, {
            duration: 8000,
            icon: '🚨',
          });
        }
      });

      // Agent events
      socket.on('agent_event', (event: AgentEvent) => {
        setState(prev => ({ ...prev, lastActivity: new Date() }));
        notifyListeners('agent_event', event);
        
        if (event.event === 'error') {
          toast.error(`Agent ${event.agentId}: ${event.data.message}`, {
            duration: 5000,
            icon: '🤖',
          });
        }
      });

      // Scenario events
      socket.on('scenario_event', (event: ScenarioEvent) => {
        setState(prev => ({ ...prev, lastActivity: new Date() }));
        notifyListeners('scenario_event', event);
        
        if (event.event === 'completed') {
          toast.success(`Scenario completed: ${event.data.name}`, {
            duration: 4000,
            icon: '🎯',
          });
        }
      });

      // Log events
      socket.on('log_event', (event: LogEvent) => {
        setState(prev => ({ ...prev, lastActivity: new Date() }));
        notifyListeners('log_event', event);
        
        // Show critical/error logs as notifications
        if (event.level === 'critical' || event.level === 'error') {
          toast.error(`${event.component}: ${event.message}`, {
            duration: 6000,
            icon: event.level === 'critical' ? '💥' : '❌',
          });
        }
      });

      // Metric updates
      socket.on('metric_update', (event: MetricUpdate) => {
        setState(prev => ({ ...prev, lastActivity: new Date() }));
        notifyListeners('metric_update', event);
      });

      // Heartbeat response
      socket.on('pong', () => {
        setState(prev => ({ ...prev, lastActivity: new Date() }));
      });

      setState(prev => ({ ...prev, socket }));

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setState(prev => ({ ...prev, connectionStatus: 'error' }));
      handleReconnection();
    }
  };

  const disconnect = () => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    stopHeartbeat();

    if (state.socket) {
      state.socket.disconnect();
      setState(prev => ({
        ...prev,
        socket: null,
        connectionStatus: 'disconnected',
        isReconnecting: false,
      }));
    }
  };

  const emit = (event: string, data: any) => {
    if (state.socket?.connected) {
      state.socket.emit(event, data);
    } else {
      console.warn('Cannot emit event: WebSocket not connected');
    }
  };

  const subscribe = (event: string, callback: (data: any) => void) => {
    if (!eventListenersRef.current.has(event)) {
      eventListenersRef.current.set(event, new Set());
    }
    eventListenersRef.current.get(event)!.add(callback);

    // Return unsubscribe function
    return () => {
      const listeners = eventListenersRef.current.get(event);
      if (listeners) {
        listeners.delete(callback);
        if (listeners.size === 0) {
          eventListenersRef.current.delete(event);
        }
      }
    };
  };

  const notifyListeners = (event: string, data: any) => {
    const listeners = eventListenersRef.current.get(event);
    if (listeners) {
      listeners.forEach(callback => callback(data));
    }
  };

  const handleReconnection = () => {
    if (state.isReconnecting || state.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      return;
    }

    setState(prev => ({
      ...prev,
      isReconnecting: true,
      reconnectAttempts: prev.reconnectAttempts + 1,
    }));

    const delay = Math.min(RECONNECT_INTERVAL * Math.pow(2, state.reconnectAttempts), 30000);
    
    reconnectTimeoutRef.current = setTimeout(() => {
      console.log(`Attempting to reconnect (${state.reconnectAttempts + 1}/${MAX_RECONNECT_ATTEMPTS})...`);
      
      if (state.socket) {
        state.socket.disconnect();
      }
      
      connect();
    }, delay);
  };

  const startHeartbeat = (socket: Socket) => {
    heartbeatIntervalRef.current = setInterval(() => {
      if (socket.connected) {
        socket.emit('ping');
      }
    }, HEARTBEAT_INTERVAL);
  };

  const stopHeartbeat = () => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = null;
    }
  };

  // Initialize connection on mount
  useEffect(() => {
    connect();

    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, []);

  // Handle browser visibility changes
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible' && state.connectionStatus === 'disconnected') {
        connect();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, [state.connectionStatus]);

  // Handle network online/offline events
  useEffect(() => {
    const handleOnline = () => {
      if (state.connectionStatus === 'disconnected') {
        connect();
      }
    };

    const handleOffline = () => {
      toast.error('Network connection lost', {
        duration: 3000,
        icon: '📡',
      });
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [state.connectionStatus]);

  const contextValue: WebSocketContextType = {
    ...state,
    connect,
    disconnect,
    emit,
    subscribe,
  };

  return (
    <WebSocketContext.Provider value={contextValue}>
      {children}
    </WebSocketContext.Provider>
  );
};

// Hook to use WebSocket context
export const useWebSocket = (): WebSocketContextType => {
  const context = useContext(WebSocketContext);
  if (context === undefined) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

// Hook for subscribing to specific events
export const useWebSocketEvent = (event: string, callback: (data: any) => void) => {
  const { subscribe } = useWebSocket();

  useEffect(() => {
    const unsubscribe = subscribe(event, callback);
    return unsubscribe;
  }, [event, callback, subscribe]);
};

// Hook for subscribing to multiple events
export const useWebSocketEvents = (eventCallbacks: Record<string, (data: any) => void>) => {
  const { subscribe } = useWebSocket();

  useEffect(() => {
    const unsubscribeFunctions = Object.entries(eventCallbacks).map(([event, callback]) =>
      subscribe(event, callback)
    );

    return () => {
      unsubscribeFunctions.forEach(unsubscribe => unsubscribe());
    };
  }, [eventCallbacks, subscribe]);
};