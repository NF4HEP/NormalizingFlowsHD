#!/bin/bash
#SBATCH --job-name=my_4gpu_job
#SBATCH -A inf23_test_2
#SBATCH -p boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4  # or however many CPUs you want to request
#SBATCH --time=24:00:00  # walltime limit (HH:MM:SS)
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt

# Run your script, which will internally use srun to launch job steps
./leonardo_launch_runs_srun.sh