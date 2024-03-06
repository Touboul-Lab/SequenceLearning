#!/bin/bash
set -x

# Activate conda environment

if [ -z ${OAR_JOB_ID+x} ]
then
    export PATH="/scratch/gvignoud/miniconda3/bin:$PATH"
    SCRIPT_DIR='/home/gvignoud'
elif [ -z ${SLURM_JOB_ID+x} ]
then
    export PATH="/home/gvignoud/miniconda3/bin:$PATH"
    SCRIPT_DIR='/home/rioc/gvignoud'
fi
source activate py37

ARGS_JOB_NAME=$1
ARGS_LIST_NAME=( $2 )
ARGS_LIST=( "${@:3}" )
ARGS_SAVE='True'
ARGS_RANDOM_SEED='0'

ARGS_LINE="--name=$ARGS_JOB_NAME/${ARGS_LIST[0]} "
for ((i=1;i<${#ARGS_LIST_NAME[@]};++i)); do
    name_params="${ARGS_LIST_NAME[$i]}"
    name_params_without="$(echo $name_params)"
    value="${ARGS_LIST[$i]}"
    PARAMS="--${name_params_without}=${value} "
    ARGS_LINE=$ARGS_LINE$PARAMS
done

python $SCRIPT_DIR/numeric_networks/Articles/Sequential_Learning_Striatum/Code/main_pattern_single.py \
    $ARGS_LINE \
    --save $ARGS_SAVE \
    --random_seed $ARGS_RANDOM_SEED

source deactivate