#!/bin/bash -l
# dcgmi dmon -e 1009,1010 -i 0,1,2,3 -d 160 > pcie_throughput.log & DCGMI_PID=$!
sh /home/binkma/bm_ds/train_llama3/start_train_test.sh &
sh /home/binkma/bm_ds/train_opt/start_train_test.sh &