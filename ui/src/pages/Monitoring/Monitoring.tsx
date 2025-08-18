import React from 'react';
import { Box, Typography, Grid } from '@mui/material';
import NetworkTopology from '../../components/NetworkTopology/NetworkTopology';

const Monitoring: React.FC = () => {
  return (
    <Box className="page-container fade-in">
      <Box mb={3}>
        <Typography variant="h4" component="h1" gutterBottom>
          System Monitoring
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Real-time monitoring of agent activity and system performance
        </Typography>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <NetworkTopology />
        </Grid>
      </Grid>
    </Box>
  );
};

export default Monitoring;