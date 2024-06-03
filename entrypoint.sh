#!/bin/bash
echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
echo "root:123" | chpasswd
service ssh start
cd /app
/bin/bash
# for oneAPI(if need)
source /opt/intel/oneapi/setvars.sh
