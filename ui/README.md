# Archangel UI - Real-time Security Monitoring Dashboard

A comprehensive React-based user interface for monitoring and managing the Archangel autonomous AI security system. This dashboard provides real-time visualization of agent activities, scenario management, network topology, and system performance metrics.

## 🚀 Features

### Real-time System Monitoring
- **Live Dashboard**: Real-time system metrics with WebSocket updates
- **Agent Activity Visualization**: Monitor AI agent performance and status
- **Network Topology Display**: Interactive network graph showing agent relationships
- **Threat Intelligence Map**: Global threat visualization with real-time updates

### Scenario Management
- **Scenario Creation**: Intuitive interface for creating and configuring security scenarios
- **Progress Tracking**: Real-time scenario progress monitoring with detailed timelines
- **Participant Management**: Assign and manage AI agents for scenarios
- **Configuration Tools**: Advanced scenario configuration with attack vectors and defense strategies

### Agent Management
- **Agent Overview**: Comprehensive agent status and performance monitoring
- **Performance Metrics**: Success rates, response times, and task completion tracking
- **Activity Feed**: Real-time agent activity logs and event notifications
- **Configuration Interface**: Agent settings and behavior configuration

### Advanced Visualization
- **Interactive Charts**: Performance metrics with multiple chart types (line, area, bar)
- **Network Graphs**: Force-directed graph visualization of system topology
- **Real-time Updates**: Live data updates via WebSocket connections
- **Responsive Design**: Mobile-first design that works on all screen sizes

## 🏗️ Architecture

### Component Structure
```
ui/src/
├── components/           # Reusable UI components
│   ├── ErrorBoundary/   # Error handling wrapper
│   ├── Layout/          # Main application layout
│   └── NetworkTopology/ # Network visualization component
├── contexts/            # React context providers
│   ├── SystemContext.tsx    # System state management
│   └── WebSocketContext.tsx # Real-time communication
├── pages/               # Main application pages
│   ├── Dashboard/       # Main monitoring dashboard
│   ├── Scenarios/       # Scenario management interface
│   ├── Agents/          # Agent management interface
│   ├── Monitoring/      # System monitoring tools
│   ├── Logs/           # Log viewing interface
│   └── Settings/       # Application settings
└── tests/              # Comprehensive test suite
```

### Technology Stack
- **React 18**: Modern React with hooks and concurrent features
- **Material-UI v5**: Comprehensive component library with dark theme
- **TypeScript**: Type-safe development with full IntelliSense
- **Socket.IO**: Real-time WebSocket communication
- **Recharts**: Advanced charting and data visualization
- **React Force Graph**: Interactive network topology visualization
- **Axios**: HTTP client for API communication
- **Moment.js**: Date and time manipulation

## 🚦 Getting Started

### Prerequisites
- Node.js 16+ and npm 8+
- Backend Archangel system running on port 8888

### Installation
```bash
# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build
```

### Environment Configuration
Create a `.env` file in the ui directory:
```env
REACT_APP_API_URL=http://localhost:8888
REACT_APP_WEBSOCKET_URL=ws://localhost:8888
REACT_APP_VERSION=1.0.0
```

## 🧪 Testing

### Test Suite Overview
The UI includes a comprehensive test suite covering:
- **Unit Tests**: Individual component functionality
- **Integration Tests**: Component interaction and data flow
- **Performance Tests**: UI responsiveness and rendering speed
- **Accessibility Tests**: WCAG compliance and keyboard navigation
- **Real-time Tests**: WebSocket event handling and live updates

### Running Tests
```bash
# Run all tests with coverage
npm run test:all

# Run comprehensive UI test suite
npm run test:ui

# Run performance-specific tests
npm run test:performance

# Run tests in watch mode
npm test
```

### Test Coverage Requirements
- **Minimum Coverage**: 80% for all components
- **Performance Benchmarks**: 
  - Component render time < 100ms
  - Test execution time < 5s per file
  - Memory usage < 512MB during testing

## 📊 Performance Monitoring

### Real-time Metrics
The dashboard displays live system metrics including:
- **CPU Usage**: Real-time processor utilization
- **Memory Usage**: RAM consumption and availability
- **Network Traffic**: Bandwidth utilization and throughput
- **Agent Performance**: Success rates and response times
- **System Health**: Overall system status and alerts

### WebSocket Events
The UI subscribes to various real-time events:
```typescript
// System events
'system_event'    // System status changes
'metric_update'   // Performance metrics
'alert'          // Security alerts

// Agent events  
'agent_event'     // Agent status changes
'task_completed'  // Task completion notifications
'error'          // Agent error events

// Scenario events
'scenario_event'  // Scenario lifecycle events
'progress_update' // Scenario progress changes
```

## 🎨 UI Components

### Dashboard Components
- **SystemStatusCard**: Displays key system metrics with trend indicators
- **MetricsChart**: Configurable charts for performance data visualization
- **ActivityFeed**: Real-time activity log with filtering and search
- **ThreatMap**: Global threat visualization with interactive map

### Scenario Components
- **ScenarioCard**: Individual scenario display with actions and status
- **ScenarioEditor**: Advanced scenario configuration interface
- **ScenarioDetails**: Detailed scenario information with timeline
- **ParticipantManager**: Agent assignment and management

### Agent Components
- **AgentCard**: Individual agent status and performance display
- **AgentConfiguration**: Agent settings and behavior configuration
- **PerformanceMetrics**: Agent performance tracking and analytics
- **NetworkTopology**: Interactive agent relationship visualization

## 🔧 Configuration

### Theme Customization
The UI uses Material-UI's theming system with a dark theme optimized for security operations:

```typescript
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#1976d2' },
    secondary: { main: '#9c27b0' },
    error: { main: '#f44336' },
    warning: { main: '#ff9800' },
    success: { main: '#4caf50' },
  },
});
```

### WebSocket Configuration
WebSocket connection settings can be configured:
```typescript
const WEBSOCKET_CONFIG = {
  url: process.env.REACT_APP_WEBSOCKET_URL,
  reconnectInterval: 5000,
  maxReconnectAttempts: 10,
  heartbeatInterval: 30000,
};
```

## 📱 Responsive Design

The UI is designed mobile-first with responsive breakpoints:
- **Mobile**: 320px - 767px (Single column layout)
- **Tablet**: 768px - 1023px (Two column layout)
- **Desktop**: 1024px+ (Multi-column layout with sidebars)

### Accessibility Features
- **WCAG 2.1 AA Compliance**: Full accessibility standard compliance
- **Keyboard Navigation**: Complete keyboard-only navigation support
- **Screen Reader Support**: ARIA labels and semantic HTML structure
- **High Contrast**: Sufficient color contrast ratios for all text
- **Focus Management**: Proper focus handling for dynamic content

## 🔒 Security Considerations

### Data Protection
- **No Sensitive Data**: UI never stores sensitive security data locally
- **Secure Communication**: All API calls use HTTPS in production
- **Input Validation**: Client-side validation for all user inputs
- **XSS Protection**: Proper sanitization of dynamic content

### Authentication Integration
The UI is designed to integrate with authentication systems:
```typescript
// Future authentication integration points
interface AuthContext {
  user: User | null;
  login: (credentials: Credentials) => Promise<void>;
  logout: () => void;
  hasPermission: (permission: string) => boolean;
}
```

## 🚀 Deployment

### Production Build
```bash
# Create optimized production build
npm run build

# Serve static files (example with serve)
npx serve -s build -l 3000
```

### Docker Deployment
```dockerfile
FROM node:16-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Environment Variables
Production environment variables:
```env
REACT_APP_API_URL=https://api.archangel.security
REACT_APP_WEBSOCKET_URL=wss://ws.archangel.security
REACT_APP_VERSION=1.0.0
REACT_APP_ENVIRONMENT=production
```

## 🐛 Troubleshooting

### Common Issues

**WebSocket Connection Failed**
```bash
# Check backend service
curl http://localhost:8888/api/health

# Verify WebSocket endpoint
wscat -c ws://localhost:8888
```

**Performance Issues**
```bash
# Run performance tests
npm run test:performance

# Check bundle size
npm run build -- --analyze
```

**Test Failures**
```bash
# Run tests with verbose output
npm test -- --verbose

# Check test coverage
npm run test:all
```

## 📈 Monitoring and Analytics

### Performance Metrics
The UI tracks various performance metrics:
- **Component Render Times**: Track slow-rendering components
- **API Response Times**: Monitor backend communication latency
- **WebSocket Latency**: Measure real-time update delays
- **Memory Usage**: Track memory leaks and optimization opportunities

### Error Tracking
Comprehensive error handling and reporting:
```typescript
// Error boundary for component crashes
<ErrorBoundary>
  <Dashboard />
</ErrorBoundary>

// WebSocket error handling
socket.on('error', (error) => {
  console.error('WebSocket error:', error);
  // Implement retry logic
});
```

## 🤝 Contributing

### Development Guidelines
1. **Code Style**: Follow TypeScript and React best practices
2. **Testing**: Maintain 80%+ test coverage for all new components
3. **Performance**: Ensure components render in <100ms
4. **Accessibility**: Test with screen readers and keyboard navigation
5. **Documentation**: Update README and component documentation

### Pull Request Process
1. Create feature branch from `main`
2. Implement changes with comprehensive tests
3. Run full test suite: `npm run test:ui`
4. Update documentation as needed
5. Submit PR with detailed description

## 📄 License

This project is part of the Archangel AI Security System. See the main project LICENSE file for details.

## 🆘 Support

For technical support and questions:
- **Documentation**: Check the main Archangel project README
- **Issues**: Create GitHub issues for bugs and feature requests
- **Performance**: Use the built-in performance monitoring tools
- **Testing**: Run the comprehensive test suite for debugging

---

**Built with ❤️ for cybersecurity professionals and AI researchers**