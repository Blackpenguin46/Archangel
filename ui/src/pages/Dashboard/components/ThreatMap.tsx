import React, { useState, useRef, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  IconButton,
  Tooltip,
  Button,
  ButtonGroup,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Fullscreen as FullscreenIcon,
  FilterList as FilterIcon,
} from '@mui/icons-material';

interface ThreatLocation {
  lat: number;
  lng: number;
}

interface ThreatData {
  id: string | number;
  level: 'low' | 'medium' | 'high' | 'critical';
  source: string;
  timestamp: Date;
  location: ThreatLocation;
  description?: string;
}

interface ThreatMapProps {
  threats: ThreatData[];
}

const ThreatMap: React.FC<ThreatMapProps> = ({ threats }) => {
  const [filter, setFilter] = useState<'all' | 'critical' | 'high' | 'medium' | 'low'>('all');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const mapRef = useRef<HTMLDivElement>(null);

  const filteredThreats = threats.filter(threat => 
    filter === 'all' || threat.level === filter
  );

  const getThreatColor = (level: string) => {
    switch (level) {
      case 'critical': return '#f44336';
      case 'high': return '#ff9800';
      case 'medium': return '#ffeb3b';
      case 'low': return '#4caf50';
      default: return '#757575';
    }
  };

  const getThreatSize = (level: string) => {
    switch (level) {
      case 'critical': return 16;
      case 'high': return 12;
      case 'medium': return 8;
      case 'low': return 6;
      default: return 8;
    }
  };

  const handleFilterChange = (newFilter: typeof filter) => {
    setFilter(newFilter);
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement && mapRef.current) {
      mapRef.current.requestFullscreen();
      setIsFullscreen(true);
    } else if (document.fullscreenElement) {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  // Listen for fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  // Generate world map visualization using SVG
  const renderWorldMap = () => {
    return (
      <svg
        width="100%"
        height="100%"
        viewBox="0 0 800 400"
        style={{ background: '#0a0a0a' }}
      >
        {/* Simple world outline */}
        <g fill="none" stroke="#333" strokeWidth="1">
          {/* Continents outline (simplified) */}
          <path d="M150 100 L250 120 L350 100 L450 120 L550 100 L650 120 L750 100" />
          <path d="M100 200 L200 180 L300 200 L400 180 L500 200 L600 180 L700 200" />
          <path d="M120 300 L220 280 L320 300 L420 280 L520 300 L620 280 L720 300" />
        </g>
        
        {/* Grid lines */}
        <g stroke="#222" strokeWidth="0.5" opacity="0.5">
          {/* Longitude lines */}
          {Array.from({ length: 9 }, (_, i) => (
            <line key={`lon-${i}`} x1={i * 100} y1="0" x2={i * 100} y2="400" />
          ))}
          {/* Latitude lines */}
          {Array.from({ length: 5 }, (_, i) => (
            <line key={`lat-${i}`} x1="0" y1={i * 100} x2="800" y2={i * 100} />
          ))}
        </g>
        
        {/* Threat markers */}
        {filteredThreats.map((threat, index) => {
          // Convert lat/lng to SVG coordinates (simplified projection)
          const x = ((threat.location.lng + 180) / 360) * 800;
          const y = ((90 - threat.location.lat) / 180) * 400;
          const size = getThreatSize(threat.level);
          const color = getThreatColor(threat.level);
          
          return (
            <g key={threat.id}>
              {/* Pulse animation for critical threats */}
              {threat.level === 'critical' && (
                <circle
                  cx={x}
                  cy={y}
                  r={size * 2}
                  fill={color}
                  opacity="0.3"
                >
                  <animate
                    attributeName="r"
                    values={`${size};${size * 3};${size}`}
                    dur="2s"
                    repeatCount="indefinite"
                  />
                  <animate
                    attributeName="opacity"
                    values="0.3;0.1;0.3"
                    dur="2s"
                    repeatCount="indefinite"
                  />
                </circle>
              )}
              
              {/* Main threat marker */}
              <circle
                cx={x}
                cy={y}
                r={size}
                fill={color}
                stroke="#fff"
                strokeWidth="1"
                style={{ cursor: 'pointer' }}
              >
                <title>
                  {`${threat.source} - ${threat.level} threat at ${threat.timestamp.toLocaleString()}`}
                </title>
              </circle>
              
              {/* Threat level indicator */}
              <text
                x={x}
                y={y + 2}
                textAnchor="middle"
                fontSize="8"
                fill="#fff"
                fontWeight="bold"
              >
                {threat.level.charAt(0).toUpperCase()}
              </text>
            </g>
          );
        })}
        
        {/* Legend */}
        <g transform="translate(20, 350)">
          <rect x="0" y="0" width="200" height="45" fill="rgba(0,0,0,0.7)" stroke="#333" rx="5" />
          <text x="10" y="15" fill="#fff" fontSize="12" fontWeight="bold">Threat Levels</text>
          
          <circle cx="20" cy="28" r="6" fill="#f44336" />
          <text x="30" y="32" fill="#fff" fontSize="10">Critical</text>
          
          <circle cx="70" cy="28" r="5" fill="#ff9800" />
          <text x="78" y="32" fill="#fff" fontSize="10">High</text>
          
          <circle cx="110" cy="28" r="4" fill="#ffeb3b" />
          <text x="118" y="32" fill="#fff" fontSize="10">Med</text>
          
          <circle cx="145" cy="28" r="3" fill="#4caf50" />
          <text x="152" y="32" fill="#fff" fontSize="10">Low</text>
        </g>
      </svg>
    );
  };

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">
            Global Threat Map
          </Typography>
          
          <Box display="flex" gap={1}>
            <ButtonGroup size="small" variant="outlined">
              <Button
                variant={filter === 'all' ? 'contained' : 'outlined'}
                onClick={() => handleFilterChange('all')}
              >
                All
              </Button>
              <Button
                variant={filter === 'critical' ? 'contained' : 'outlined'}
                onClick={() => handleFilterChange('critical')}
                sx={{ color: '#f44336', borderColor: '#f44336' }}
              >
                Critical
              </Button>
              <Button
                variant={filter === 'high' ? 'contained' : 'outlined'}
                onClick={() => handleFilterChange('high')}
                sx={{ color: '#ff9800', borderColor: '#ff9800' }}
              >
                High
              </Button>
            </ButtonGroup>
            
            <Tooltip title="Refresh">
              <IconButton size="small">
                <RefreshIcon />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Fullscreen">
              <IconButton size="small" onClick={toggleFullscreen}>
                <FullscreenIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Stats Summary */}
        <Box display="flex" gap={1} mb={2}>
          <Chip
            label={`${threats.filter(t => t.level === 'critical').length} Critical`}
            size="small"
            sx={{ bgcolor: '#f44336', color: 'white' }}
          />
          <Chip
            label={`${threats.filter(t => t.level === 'high').length} High`}
            size="small"
            sx={{ bgcolor: '#ff9800', color: 'white' }}
          />
          <Chip
            label={`${threats.filter(t => t.level === 'medium').length} Medium`}
            size="small"
            sx={{ bgcolor: '#ffeb3b', color: 'black' }}
          />
          <Chip
            label={`${threats.filter(t => t.level === 'low').length} Low`}
            size="small"
            sx={{ bgcolor: '#4caf50', color: 'white' }}
          />
        </Box>

        {/* Map Container */}
        <Box
          ref={mapRef}
          sx={{
            height: isFullscreen ? '100vh' : 400,
            width: '100%',
            border: '1px solid #333',
            borderRadius: 1,
            overflow: 'hidden',
            position: 'relative',
            bgcolor: '#0a0a0a',
          }}
        >
          {renderWorldMap()}
          
          {/* Active threats counter */}
          <Box
            position="absolute"
            top={16}
            right={16}
            bgcolor="rgba(26, 26, 46, 0.9)"
            p={1}
            borderRadius={1}
            border="1px solid #333"
          >
            <Typography variant="body2" color="primary">
              {filteredThreats.length} Active Threats
            </Typography>
          </Box>
        </Box>

        {/* Recent Threats List */}
        <Box mt={2} maxHeight={150} overflow="auto">
          <Typography variant="subtitle2" gutterBottom>
            Recent Threats ({Math.min(5, filteredThreats.length)})
          </Typography>
          {filteredThreats.slice(0, 5).map((threat, index) => (
            <Box
              key={threat.id}
              display="flex"
              alignItems="center"
              gap={2}
              py={0.5}
              borderBottom={index < 4 ? '1px solid #333' : 'none'}
            >
              <Box
                width={12}
                height={12}
                borderRadius="50%"
                bgcolor={getThreatColor(threat.level)}
              />
              <Box flex={1}>
                <Typography variant="body2" noWrap>
                  {threat.source}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {threat.timestamp.toLocaleTimeString()}
                </Typography>
              </Box>
              <Chip
                label={threat.level}
                size="small"
                sx={{
                  bgcolor: getThreatColor(threat.level),
                  color: threat.level === 'medium' ? 'black' : 'white',
                  fontSize: '0.7rem',
                }}
              />
            </Box>
          ))}
        </Box>
      </CardContent>
    </Card>
  );
};

export default ThreatMap;