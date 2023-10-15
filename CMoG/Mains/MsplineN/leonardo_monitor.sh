#!/bin/bash
cmog_path="/leonardo/home/userexternal/rtorre00/git/GitHub/NormalizingFlows/NF4HEP/NormalizingFlowsHD/CMoG"
while true; do
    nvidia-smi > ${cmog_path}/Outputs/MsplineN/leonardo_nvidia-smi.out
    top -n 1 -b > ${cmog_path}/Outputs/MsplineN/leonardo_top.out
    sleep 5
done
