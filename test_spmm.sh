#!/bin/bash
#SBATCH -A m4341
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"
srun ./spmm_profile data/GD97_b/GD97_b.rb                            > spmm_results/GD97_b.txt
srun ./spmm_profile data/Hamrle1/Hamrle1.rb                          > spmm_results/Hamrle1.txt
srun ./spmm_profile data/micromass_10NN/micromass_10NN.rb            > spmm_results/micromass_10NN.txt
srun ./spmm_profile data/umistfacesnorm_10NN/umistfacesnorm_10NN.rb  > spmm_results/umistfacesnorm_10NN.txt


