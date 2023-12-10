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
    srun nsys profile -s none --trace=cuda,nvtx,osrt,cusparse --force-overwrite true -o results/$3$1.nsys-rep ./spgemm_p data/$1/$1.rb $2 > results/$3$1.txt
}
run() {
    ./spgemm data/$1/$1.rb $2 > results/$3$1.txt
}

DATASETS=$(ls ./data | grep -Po "^[a-zA-Z0-9\_]+" | sort | uniq | grep -P "(GD|Ha)")
DACC=0
SACC=0

for dataset in ${DATASETS[@]}; do
    #run $dataset 0 dacc_
    DACC=$(echo $DACC+$(grep -Po "Cuda Result matches CuSparse Result" results/dacc_$dataset.txt | wc -l) | bc)

    #run $dataset 1 sacc_
    SACC=$(echo $SACC+$(grep -Po "Cuda Result matches CuSparse Result" results/sacc_$dataset.txt | wc -l) | bc)
done
echo $DACC $SACC ${#DATASETS[@]}
if [[ $DACC -eq ${#DATASETS[@]} ]]; then
    echo "All dense accumulator results match"
else
    echo "Not all dense accumulator results match"
fi
if [[ $SACC -eq ${#DATASETS[@]} ]]; then
    echo "All sparse accumulator results match"
else
    echo "Not all sparse accumulator results match"
fi

#profile GD97_b
#profile Hamrle1
#profile micromass_10NN
#profile umistfacesnorm_10NN
#profile netscience
#run GD97_b 0 dacc_
#run GD97_b 1 sacc_