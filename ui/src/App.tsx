import React, { useState, useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Box, CircularProgress, Typography } from '@mui/material';
import Layout from './components/Layout/Layout';
import Dashboard from './pages/Dashboard/Dashboard';
import Monitoring from './pages/Monitoring/Monitoring';
import Scenarios from './pages/Scenarios/Scenarios';
import Agents from './pages/Agents/Agents';
import Logs from './pages/Logs/Logs';
import Settings from './pages/Settings/Settings';
import { WebSocketProvider } from './contexts/WebSocketContext';
import { SystemProvider } from './contexts/SystemContext';
import ErrorBoundary from './components/ErrorBoundary/ErrorBoundary';
import './App.css';

function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Initialize application
    const initializeApp = async () => {
      try {
        // Simulate initialization time
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Check if backend is available
        const healthCheck = await fetch('/api/health');
        if (!healthCheck.ok) {
          throw new Error('Backend service unavailable');
        }
        
        setIsLoading(false);
      } catch (err) {
        console.error('App initialization failed:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
        setIsLoading(false);
      }
    };

    initializeApp();
  }, []);

  if (isLoading) {
    return (
      <Box className="full-height flex-center">
        <Box className="text-center">
          <CircularProgress size={60} sx={{ color: '#1976d2', mb: 2 }} />
          <Typography variant="h6" color="primary">
            Initializing Archangel System
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Loading AI Security Expert System<span className="loading-dots">...</span>
          </Typography>
        </Box>
      </Box>
    );
  }

  if (error) {
    return (
      <Box className="full-height flex-center">
        <Box className="text-center">
          <Typography variant="h5" color="error" gutterBottom>
            System Initialization Failed
          </Typography>
          <Typography variant="body1" color="text.secondary">
            {error}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
            Please check the backend service and refresh the page.
          </Typography>
        </Box>
      </Box>
    );
  }

  return (
    <ErrorBoundary>
      <SystemProvider>
        <WebSocketProvider>
          <Layout>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/monitoring" element={<Monitoring />} />
              <Route path="/scenarios" element={<Scenarios />} />
              <Route path="/agents" element={<Agents />} />
              <Route path="/logs" element={<Logs />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </Layout>
        </WebSocketProvider>
      </SystemProvider>
    </ErrorBoundary>
  );
}

export default App;