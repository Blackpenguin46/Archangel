import React, { Component, ErrorInfo, ReactNode } from 'react';
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  Error as ErrorIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  BugReport as BugReportIcon,
} from '@mui/icons-material';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
  errorId: string;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    errorId: '',
  };

  public static getDerivedStateFromError(error: Error): State {
    // Update state so the next render will show the fallback UI
    return {
      hasError: true,
      error,
      errorId: Date.now().toString(36),
    };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log error to console and potentially to monitoring service
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    this.setState({
      error,
      errorInfo,
    });

    // In a real application, you would send this to your error reporting service
    this.reportError(error, errorInfo);
  }

  private reportError = (error: Error, errorInfo: ErrorInfo) => {
    const errorReport = {
      message: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href,
      errorId: this.state.errorId,
    };

    // Log to console for now (in production, send to error tracking service)
    console.error('Error Report:', errorReport);
    
    // You could send to services like Sentry, LogRocket, etc.
    // Sentry.captureException(error, { contexts: { react: errorInfo } });
  };

  private handleReload = () => {
    window.location.reload();
  };

  private handleRetry = () => {
    this.setState({
      hasError: false,
      error: undefined,
      errorInfo: undefined,
    });
  };

  private copyErrorToClipboard = () => {
    const errorText = `
Error ID: ${this.state.errorId}
Message: ${this.state.error?.message}
Stack: ${this.state.error?.stack}
Component Stack: ${this.state.errorInfo?.componentStack}
Timestamp: ${new Date().toISOString()}
URL: ${window.location.href}
User Agent: ${navigator.userAgent}
    `.trim();

    navigator.clipboard.writeText(errorText).then(() => {
      // You could show a toast notification here
      console.log('Error details copied to clipboard');
    });
  };

  public render() {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default error UI
      return (
        <Box
          className="full-height flex-center"
          sx={{
            background: 'linear-gradient(135deg, #0c1426 0%, #1a1a2e 50%, #16213e 100%)',
            p: 3,
          }}
        >
          <Card sx={{ maxWidth: 800, width: '100%' }}>
            <CardContent sx={{ p: 4 }}>
              <Box display="flex" alignItems="center" gap={2} mb={3}>
                <ErrorIcon sx={{ fontSize: 48, color: 'error.main' }} />
                <Box>
                  <Typography variant="h4" color="error" gutterBottom>
                    Oops! Something went wrong
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    The Archangel interface encountered an unexpected error.
                  </Typography>
                </Box>
              </Box>

              <Alert severity="error" sx={{ mb: 3 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Error ID: {this.state.errorId}
                </Typography>
                <Typography variant="body2">
                  {this.state.error?.message || 'An unknown error occurred'}
                </Typography>
              </Alert>

              <Box display="flex" gap={2} mb={3}>
                <Button
                  variant="contained"
                  startIcon={<RefreshIcon />}
                  onClick={this.handleRetry}
                  color="primary"
                >
                  Try Again
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<RefreshIcon />}
                  onClick={this.handleReload}
                >
                  Reload Page
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<BugReportIcon />}
                  onClick={this.copyErrorToClipboard}
                >
                  Copy Error Details
                </Button>
              </Box>

              {process.env.NODE_ENV === 'development' && this.state.error && (
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="subtitle1">
                      Error Details (Development Mode)
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Box>
                      <Typography variant="subtitle2" gutterBottom>
                        Error Message:
                      </Typography>
                      <Box
                        component="pre"
                        sx={{
                          p: 2,
                          backgroundColor: 'rgba(255, 255, 255, 0.05)',
                          borderRadius: 1,
                          overflow: 'auto',
                          fontSize: '0.875rem',
                          fontFamily: 'monospace',
                          mb: 2,
                        }}
                      >
                        {this.state.error.message}
                      </Box>

                      <Typography variant="subtitle2" gutterBottom>
                        Stack Trace:
                      </Typography>
                      <Box
                        component="pre"
                        sx={{
                          p: 2,
                          backgroundColor: 'rgba(255, 255, 255, 0.05)',
                          borderRadius: 1,
                          overflow: 'auto',
                          fontSize: '0.75rem',
                          fontFamily: 'monospace',
                          mb: 2,
                          maxHeight: 200,
                        }}
                      >
                        {this.state.error.stack}
                      </Box>

                      {this.state.errorInfo && (
                        <>
                          <Typography variant="subtitle2" gutterBottom>
                            Component Stack:
                          </Typography>
                          <Box
                            component="pre"
                            sx={{
                              p: 2,
                              backgroundColor: 'rgba(255, 255, 255, 0.05)',
                              borderRadius: 1,
                              overflow: 'auto',
                              fontSize: '0.75rem',
                              fontFamily: 'monospace',
                              maxHeight: 200,
                            }}
                          >
                            {this.state.errorInfo.componentStack}
                          </Box>
                        </>
                      )}
                    </Box>
                  </AccordionDetails>
                </Accordion>
              )}

              <Box mt={3} p={2} bgcolor="rgba(255, 255, 255, 0.02)" borderRadius={1}>
                <Typography variant="body2" color="text.secondary">
                  <strong>Troubleshooting Tips:</strong>
                </Typography>
                <Typography variant="body2" color="text.secondary" component="ul" sx={{ mt: 1, pl: 2 }}>
                  <li>Try refreshing the page or clearing your browser cache</li>
                  <li>Check if the Archangel backend service is running</li>
                  <li>Verify your network connection</li>
                  <li>If the problem persists, contact your system administrator</li>
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Box>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;