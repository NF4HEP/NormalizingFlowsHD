#!/bin/bash

#echo "Loading Python environment..."
#ource .bashrc
#conda activate tf2_9_source

# Define alias for the long absolute path
cmog_path="/leonardo/home/userexternal/rtorre00/git/GitHub/NormalizingFlows/NF4HEP/NormalizingFlowsHD/CMoG"

# Declare an array of visible devices you'd like to use
#declare -a visible_devices_list=("01")
declare -a visible_devices_list=("01" "01")

# Declare an array of run numbers
#declare -a run_numbers=("10" "11" "12" "13")
declare -a run_numbers=("10" "11")

# Loop through arrays and launch jobs
for run_num in "${run_numbers[@]}"
do
    for vis_dev in "${visible_devices_list[@]}"
    do
        echo "Launching run RUN_NUMBER=$run_num on device VISIBLE_DEVICES=$vis_dev"

        # Launch the job with srun
        salloc -A inf23_test_2 -p boost_usr_prod --gres=gpu:2
        conda activate tf2_9_source
        srun python ${cmog_path}/Mains/MsplineN/c_Main_MsplineN_leonardo.py --visible_devices $vis_dev >> ${cmog_path}/Outputs/MsplineN/leonardo_run_gpu_${vis_dev}_replica_${run_num}.out 2>&1 &

        # Capture the PID of the last background job
        echo "Launched"

    done
done

# Start monitoring task
#echo "Starting monitoring task..."
#srun $cmog_path/Mains/MsplineN/leonardo_monitor.sh &

echo "Everything launched!"

salloc -A inf23_test_2 -p boost_usr_prod --time=02:00:00 --gpus=2
salloc -A inf23_test_2 -p boost_usr_prod --time=00:10:00 --gpus=2

salloc -A inf23_test_2 -p g100_usr_interactive --time=00:10:00 --gpus=2



conda activate tf2_9_source
srun python /leonardo/home/userexternal/rtorre00/git/GitHub/NormalizingFlows/NF4HEP/NormalizingFlowsHD/CMoG/Mains/MsplineN/c_Main_MsplineN_leonardo.py --visible_devices 01 >> /leonardo/home/userexternal/rtorre00/git/GitHub/NormalizingFlows/NF4HEP/NormalizingFlowsHD/CMoG/Outputs/MsplineN/leonardo_run_gpus_0-1_replica_1.out 2>&1 &

srun python -c "import tensorflow as tf; print('Built with CUDA:', tf.test.is_built_with_cuda()); print('Built with GPU support:', tf.test.is_built_with_gpu_support()); print('CUDA Version: ', tf.sysconfig.get_build_info()['cuda_version']);print('cuDNN Version: ', tf.sysconfig.get_build_info()['cudnn_version'])"

srun python -c "import tensorflow as tf; print(tf.compat.v1.ConfigProto().gpu_options.allocator_type)"


import tensorflow as tf
print("CUDA Version: ", tf.sysconfig.get_build_info()['cuda_version'])

import tensorflow as tf
print("cuDNN Version: ", tf.sysconfig.get_build_info()['cudnn_version'])