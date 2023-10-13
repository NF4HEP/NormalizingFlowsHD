#!/bin/bash

echo "Script string: python c_Main_MsplineN_leonardo.py --visible_devices 0,1 >> /leonardo/home/userexternal/rtorre00/git/GitHub/NormalizingFlows/NF4HEP/NormalizingFlowsHD/CMoG/Outputs/MsplineN/run_leonardo_devices_01_run_0.out 2>&1 &"
python c_Main_MsplineN_leonardo.py --visible_devices 0,1 >> /leonardo/home/userexternal/rtorre00/git/GitHub/NormalizingFlows/NF4HEP/NormalizingFlowsHD/CMoG/Outputs/MsplineN/run_leonardo_devices_01_run_0.out 2>&1 &
pid=$!
echo "Launched with PID $pid"

sleep 3

echo "Script string: python c_Main_MsplineN_leonardo.py --visible_devices 2,3 >> /leonardo/home/userexternal/rtorre00/git/GitHub/NormalizingFlows/NF4HEP/NormalizingFlowsHD/CMoG/Outputs/MsplineN/run_leonardo_devices_01_run_0.out 2>&1 &"
python c_Main_MsplineN_leonardo.py --visible_devices 2,3 >> /leonardo/home/userexternal/rtorre00/git/GitHub/NormalizingFlows/NF4HEP/NormalizingFlowsHD/CMoG/Outputs/MsplineN/run_leonardo_devices_01_run_0.out 2>&1 &
pid=$!
echo "Launched with PID $pid"

echo "Two runs launched on leonardo"