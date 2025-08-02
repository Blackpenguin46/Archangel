#!/bin/bash
# Attack detection for blue team defense
LOGFILE="/var/log/defenses/attack_detection.log"
echo "$(date): Starting attack detection" >> $LOGFILE

# Monitor for port scans
tail -f /var/log/auth.log | while read line; do
    if echo "$line" | grep -q "Failed password"; then
        echo "$(date): ATTACK DETECTED - Brute force attempt: $line" >> $LOGFILE
    fi
done &

# Monitor for unusual network connections
netstat -tuln | awk '{print $4}' | sort | uniq -c | sort -nr >> $LOGFILE

echo "$(date): Attack detection active" >> $LOGFILE
