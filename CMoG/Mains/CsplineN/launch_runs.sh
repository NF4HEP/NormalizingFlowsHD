#!/bin/bash

# Declare an array of visible devices you'd like to use
declare -a visible_devices_list=("0" "1" "2")

# Declare an array of run numbers
declare -a run_numbers=("9")

# Loop through arrays and 
launch jobs
for run_num in "${run_numbers[@]}"
do
    for vis_dev in "${visible_devices_list[@]}"
    do
        echo "Launching run RUN_NUMBER=$run_num on device VISIBLE_DEVICES=$vis_dev"
        echo "Script string: nohup python c_Main_CsplineN.py --visible_devices $vis_dev >> ../../Outputs/CsplineN/run_gpu_${vis_dev}_replica_${run_num}.out 2>&1 &"

        nohup python c_Main_CsplineN.py --visible_devices $vis_dev >> ../../Outputs/CsplineN/run_gpu_${vis_dev}_replica_${run_num}.out 2>&1 &

        # Capture the PID of the last background job
        pid=$!
        echo "Launched with PID $pid"

        sleep 10  # Adding sleep to prevent possible race conditions
    done
done
