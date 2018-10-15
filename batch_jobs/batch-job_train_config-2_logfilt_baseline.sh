#!/bin/bash

#SBATCH -J jsulpico_train_config-2_logfilt_baseline
#SBATCH -N 2
#SBATCH -p quickq
#SBATCH -o jsulpico_results_train_config-2_logfilt_baseline.out
#SBATCH -e jsulpico_errors_train_config-2_logfilt_baseline.err
#SBATCH --gres=gpu:1
module load shared tensorflow openmpi3/gcc/64/3.0.0
srun --gres=gpu:1 ~/superduperthesis/src/batch-job_train_config-2_logfilt_baseline.py gpu 1000