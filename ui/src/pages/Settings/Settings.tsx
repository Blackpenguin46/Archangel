import React from 'react';
import { Box, Typography, Alert } from '@mui/material';

const Settings: React.FC = () => {
  return (
    <Box className="page-container fade-in">
      <Box mb={3}>
        <Typography variant="h4" component="h1" gutterBottom>
          System Settings
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Configure system preferences and security policies
        </Typography>
      </Box>

      <Alert severity="info">
        Settings interface is under development. This will include system configuration, 
        user management, security policies, and integration settings.
      </Alert>
    </Box>
  );
};

export default Settings;