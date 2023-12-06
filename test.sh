#!/bin/bash
#SBATCH -A m4341
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

#export SLURM_CPU_BIND="cores"
profile() {
    srun nsys profile -s none --trace=cuda,nvtx,osrt,cusparse --force-overwrite true -o results/$1.nsys-rep ./profile data/$1/$1.rb > results/$1.txt
}

profile GD97_b
profile Hamrle1
profile micromass_10NN
profile umistfacesnorm_10NN