#!/bin/bash
# Defense action monitoring
LOGFILE="/var/log/defenses/defense_actions.log"
echo "$(date): Defense monitoring started" >> $LOGFILE

# Monitor iptables for blocking actions
iptables -L -n -v >> $LOGFILE

# Monitor fail2ban status
fail2ban-client status >> $LOGFILE

echo "$(date): Defense monitoring active" >> $LOGFILE
