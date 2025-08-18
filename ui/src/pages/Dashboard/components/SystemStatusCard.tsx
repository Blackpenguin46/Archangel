import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  useTheme,
} from '@mui/material';

interface SystemStatusCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color: string;
  subtitle?: string;
  trend?: {
    value: number;
    isPositive: boolean;
  };
}

const SystemStatusCard: React.FC<SystemStatusCardProps> = ({
  title,
  value,
  icon,
  color,
  subtitle,
  trend,
}) => {
  const theme = useTheme();

  return (
    <Card 
      className="metric-card"
      sx={{
        background: `linear-gradient(135deg, ${color}15 0%, ${color}05 100%)`,
        border: `1px solid ${color}30`,
        transition: 'all 0.2s ease-in-out',
        '&:hover': {
          borderColor: color,
          boxShadow: `0 4px 20px ${color}30`,
        },
      }}
    >
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: 48,
              height: 48,
              borderRadius: '12px',
              backgroundColor: `${color}20`,
              color: color,
            }}
          >
            {icon}
          </Box>
          {trend && (
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                color: trend.isPositive ? theme.palette.success.main : theme.palette.error.main,
                fontSize: '0.875rem',
                fontWeight: 500,
              }}
            >
              {trend.isPositive ? '↗' : '↘'} {Math.abs(trend.value)}%
            </Box>
          )}
        </Box>
        
        <Typography
          variant="h4"
          component="div"
          sx={{
            fontWeight: 700,
            color: color,
            mb: 0.5,
          }}
        >
          {typeof value === 'string' ? value : value.toLocaleString()}
        </Typography>
        
        <Typography
          variant="h6"
          component="div"
          color="text.primary"
          sx={{ mb: subtitle ? 0.5 : 0 }}
        >
          {title}
        </Typography>
        
        {subtitle && (
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ fontSize: '0.75rem' }}
          >
            {subtitle}
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default SystemStatusCard;