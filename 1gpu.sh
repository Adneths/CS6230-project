#!/bin/bash
#SBATCH -A m4341
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 0:10:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"

if [ "$#" -lt "1" ]; then
    echo "Usage: $0 <run|profile> <filter>"
    exit -1
fi

if [[ "$1" != "run" && "$1" != "profile" ]]; then
    echo "Usage: $0 <run|profile> <filter>"
    exit -1
fi

PROGRAM=$1
FILTER=''
if [ "$#" -ge "2" ]; then
    FILTER=$2
fi

profile() {
    srun nsys profile -s none --trace=cuda,nvtx,osrt,cusparse --force-overwrite true -o $RESULTS_PATH/$3$1.nsys-rep ./spgemm_p data/$1/$1.rb $2 > $RESULTS_PATH/$3$1.txt
}
run() {
    srun ./spgemm data/$1/$1.rb $2 > $RESULTS_PATH/$3$1.txt
}

RESULTS_PATH='results/1gpu'
mkdir -p $RESULTS_PATH
DATASETS=$(ls ./data | grep -Po "^[a-zA-Z0-9\_]+" | sort | uniq | grep -P "$FILTER")
LEN=$(echo ${DATASETS[@]} | wc -w)
DACC=0
SACC=0

for dataset in ${DATASETS[@]}; do
    $PROGRAM $dataset 0 dacc_
    DACC=$(echo $DACC+$(grep -Po "Cuda Result matches CuSparse Result" $RESULTS_PATH/dacc_$dataset.txt | wc -l) | bc)

    $PROGRAM $dataset 1 sacc_
    SACC=$(echo $SACC+$(grep -Po "Cuda Result matches CuSparse Result" $RESULTS_PATH/sacc_$dataset.txt | wc -l) | bc)
done

rm -f $0.out
echo ""  > $0.out
if [[ $DACC -eq $LEN ]]; then
    echo "All dense accumulator results match" >> $0.out
else
    echo "Not all dense accumulator results match" >> $0.out
fi
if [[ $SACC -eq $LEN ]]; then
    echo "All sparse accumulator results match" >> $0.out
else
    echo "Not all sparse accumulator results match" >> $0.out
fi
