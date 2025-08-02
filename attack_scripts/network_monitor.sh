#!/bin/bash
# Network monitoring for red vs blue combat
LOGFILE="/var/log/attacks/network_monitor.log"
echo "$(date): Starting network monitoring" >> $LOGFILE

# Monitor network traffic between red and blue teams
tcpdump -i any -w /var/log/attacks/traffic_$(date +%s).pcap \
  "host 192.168.100.10 or host 192.168.100.20" &

# Monitor connection attempts
netstat -tuln >> $LOGFILE
ss -tuln >> $LOGFILE

echo "$(date): Network monitoring active" >> $LOGFILE
