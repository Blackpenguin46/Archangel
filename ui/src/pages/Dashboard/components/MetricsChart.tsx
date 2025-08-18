import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  FormControl,
  Select,
  MenuItem,
  SelectChangeEvent,
} from '@mui/material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { useWebSocketEvent } from '../../../contexts/WebSocketContext';
import moment from 'moment';

interface MetricData {
  timestamp: string;
  cpu: number;
  memory: number;
  network: number;
  disk: number;
  responseTime: number;
  throughput: number;
  errorRate: number;
  activeConnections: number;
}

const MetricsChart: React.FC = () => {
  const [timeRange, setTimeRange] = useState<'1h' | '6h' | '24h' | '7d'>('1h');
  const [chartType, setChartType] = useState<'line' | 'area' | 'bar'>('line');
  const [metricsData, setMetricsData] = useState<MetricData[]>([]);

  // Subscribe to real-time metric updates
  useWebSocketEvent('metric_update', (data) => {
    const newMetric: MetricData = {
      timestamp: moment(data.timestamp).format('HH:mm:ss'),
      cpu: data.metrics.cpu || 0,
      memory: data.metrics.memory || 0,
      network: (data.metrics.network || 0) / 1024 / 1024, // Convert to MB
      disk: data.metrics.disk || 0,
      responseTime: data.metrics.responseTime || 0,
      throughput: data.metrics.throughput || 0,
      errorRate: data.metrics.errorRate || 0,
      activeConnections: data.metrics.activeConnections || 0,
    };

    setMetricsData(prev => {
      const maxPoints = getMaxDataPoints(timeRange);
      return [...prev, newMetric].slice(-maxPoints);
    });
  });

  // Generate initial mock data
  useEffect(() => {
    const generateMockData = () => {
      const data: MetricData[] = [];
      const points = getMaxDataPoints(timeRange);
      const now = moment();
      
      for (let i = points - 1; i >= 0; i--) {
        const timestamp = now.clone().subtract(i * getTimeInterval(timeRange), 'minutes');
        data.push({
          timestamp: timestamp.format('HH:mm:ss'),
          cpu: Math.random() * 80 + 10,
          memory: Math.random() * 70 + 20,
          network: Math.random() * 100 + 10,
          disk: Math.random() * 60 + 10,
          responseTime: Math.random() * 500 + 50,
          throughput: Math.random() * 1000 + 100,
          errorRate: Math.random() * 5,
          activeConnections: Math.floor(Math.random() * 100) + 10,
        });
      }
      
      setMetricsData(data);
    };

    generateMockData();
  }, [timeRange]);

  const getMaxDataPoints = (range: string): number => {
    switch (range) {
      case '1h': return 60;
      case '6h': return 72;
      case '24h': return 48;
      case '7d': return 168;
      default: return 60;
    }
  };

  const getTimeInterval = (range: string): number => {
    switch (range) {
      case '1h': return 1;
      case '6h': return 5;
      case '24h': return 30;
      case '7d': return 60;
      default: return 1;
    }
  };

  const handleTimeRangeChange = (event: SelectChangeEvent) => {
    setTimeRange(event.target.value as '1h' | '6h' | '24h' | '7d');
  };

  const handleChartTypeChange = (event: SelectChangeEvent) => {
    setChartType(event.target.value as 'line' | 'area' | 'bar');
  };

  const renderChart = () => {
    const commonProps = {
      data: metricsData,
      margin: { top: 20, right: 30, left: 20, bottom: 5 },
    };

    const commonAxisProps = {
      stroke: '#666',
      fontSize: 12,
    };

    switch (chartType) {
      case 'area':
        return (
          <AreaChart {...commonProps}>
            <defs>
              <linearGradient id="cpuGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#1976d2" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#1976d2" stopOpacity={0.1} />
              </linearGradient>
              <linearGradient id="memoryGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#9c27b0" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#9c27b0" stopOpacity={0.1} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="timestamp" {...commonAxisProps} />
            <YAxis {...commonAxisProps} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1a1a2e',
                border: '1px solid #333',
                borderRadius: '4px',
              }}
            />
            <Legend />
            <Area
              type="monotone"
              dataKey="cpu"
              stroke="#1976d2"
              fill="url(#cpuGradient)"
              name="CPU %"
            />
            <Area
              type="monotone"
              dataKey="memory"
              stroke="#9c27b0"
              fill="url(#memoryGradient)"
              name="Memory %"
            />
          </AreaChart>
        );

      case 'bar':
        return (
          <BarChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="timestamp" {...commonAxisProps} />
            <YAxis {...commonAxisProps} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1a1a2e',
                border: '1px solid #333',
                borderRadius: '4px',
              }}
            />
            <Legend />
            <Bar dataKey="cpu" fill="#1976d2" name="CPU %" />
            <Bar dataKey="memory" fill="#9c27b0" name="Memory %" />
            <Bar dataKey="network" fill="#4caf50" name="Network MB/s" />
          </BarChart>
        );

      default: // line
        return (
          <LineChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="timestamp" {...commonAxisProps} />
            <YAxis {...commonAxisProps} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1a1a2e',
                border: '1px solid #333',
                borderRadius: '4px',
              }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="cpu"
              stroke="#1976d2"
              strokeWidth={2}
              name="CPU %"
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="memory"
              stroke="#9c27b0"
              strokeWidth={2}
              name="Memory %"
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="network"
              stroke="#4caf50"
              strokeWidth={2}
              name="Network MB/s"
              dot={false}
            />
          </LineChart>
        );
    }
  };

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Typography variant="h6">
            Performance Metrics
          </Typography>
          
          <Box display="flex" gap={2}>
            <FormControl size="small" sx={{ minWidth: 80 }}>
              <Select
                value={timeRange}
                onChange={handleTimeRangeChange}
                variant="outlined"
              >
                <MenuItem value="1h">1h</MenuItem>
                <MenuItem value="6h">6h</MenuItem>
                <MenuItem value="24h">24h</MenuItem>
                <MenuItem value="7d">7d</MenuItem>
              </Select>
            </FormControl>
            
            <FormControl size="small" sx={{ minWidth: 80 }}>
              <Select
                value={chartType}
                onChange={handleChartTypeChange}
                variant="outlined"
              >
                <MenuItem value="line">Line</MenuItem>
                <MenuItem value="area">Area</MenuItem>
                <MenuItem value="bar">Bar</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </Box>

        <Box height={400}>
          <ResponsiveContainer width="100%" height="100%">
            {renderChart()}
          </ResponsiveContainer>
        </Box>

        {/* Performance Summary */}
        <Box display="flex" justifyContent="space-around" mt={3} pt={2} borderTop="1px solid #333">
          <Box textAlign="center">
            <Typography variant="h6" color="primary">
              {metricsData.length > 0 ? 
                Math.round(metricsData[metricsData.length - 1]?.cpu || 0) : 0}%
            </Typography>
            <Typography variant="caption" color="text.secondary">
              CPU Usage
            </Typography>
          </Box>
          
          <Box textAlign="center">
            <Typography variant="h6" color="secondary">
              {metricsData.length > 0 ? 
                Math.round(metricsData[metricsData.length - 1]?.memory || 0) : 0}%
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Memory Usage
            </Typography>
          </Box>
          
          <Box textAlign="center">
            <Typography variant="h6" sx={{ color: '#4caf50' }}>
              {metricsData.length > 0 ? 
                Math.round(metricsData[metricsData.length - 1]?.network || 0) : 0} MB/s
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Network I/O
            </Typography>
          </Box>
          
          <Box textAlign="center">
            <Typography variant="h6" sx={{ color: '#ff9800' }}>
              {metricsData.length > 0 ? 
                Math.round(metricsData[metricsData.length - 1]?.responseTime || 0) : 0}ms
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Response Time
            </Typography>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default MetricsChart;