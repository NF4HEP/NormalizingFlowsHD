#!/bin/bash

# Define alias for the long absolute path
cmog_path="/leonardo/home/userexternal/rtorre00/git/GitHub/NormalizingFlows/NF4HEP/NormalizingFlowsHD/CMoG"

# Monitoring nvidia-smi and top
(
while true; do
    nvidia-smi > ${cmog_path}/Outputs/MsplineN/lenorado_nvidia-smi.out
    top -n 1 -b > ${cmog_path}/Outputs/MsplineN/leonardo_top.out
    sleep 5
done
) &

# Declare an array of visible devices you'd like to use
declare -a visible_devices_list=("01" "12" "23" "30")

# Declare an array of run numbers
declare -a run_numbers=("10")

# Loop through arrays and launch jobs
for run_num in "${run_numbers[@]}"
do
    for vis_dev in "${visible_devices_list[@]}"
    do
        echo "Launching run RUN_NUMBER=$run_num on device VISIBLE_DEVICES=$vis_dev"
        echo "Script string: nohup python ${cmog_path}/Mains/MsplineN/c_Main_MsplineN_leonardo.py --visible_devices $vis_dev >> ${cmog_path}/Outputs/MsplineN/leonardo_run_gpu_${vis_dev}_replica_${run_num}.out 2>&1 &"

        python ${cmog_path}/Mains/MsplineN/c_Main_MsplineN_leonardo.py --visible_devices $vis_dev >> ${cmog_path}/Outputs/MsplineN/leonardo_run_gpu_${vis_dev}_replica_${run_num}.out 2>&1 &

        # Capture the PID of the last background job
        pid=$!
        echo "Launched with PID $pid"

        sleep 3  # Adding sleep to prevent possible race conditions
    done
done

echo "Everything launched!"