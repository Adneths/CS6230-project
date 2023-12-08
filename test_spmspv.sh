#!/bin/bash
#SBATCH -A m4341
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"
srun ./spmspv_profile data/GD97_b/GD97_b.rb                            > results/GD97_b.txt
srun ./spmspv_profile data/Hamrle1/Hamrle1.rb                          > results/Hamrle1.txt
srun ./spmspv_profile data/micromass_10NN/micromass_10NN.rb            > results/micromass_10NN.txt
srun ./spmspv_profile data/umistfacesnorm_10NN/umistfacesnorm_10NN.rb  > results/umistfacesnorm_10NN.txt
srun ./spmspv_profile data/ex36/ex36.rb                                > results/ex36.txt
srun ./spmspv_profile data/lock1074/lock1074.rb                        > results/lock1074.txt
srun ./spmspv_profile data/ia2010/ia2010.rb                            > results/ia2010.txt
