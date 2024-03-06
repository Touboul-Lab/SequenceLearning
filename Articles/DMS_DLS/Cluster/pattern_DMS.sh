#!/bin/bash
set -x

# Activate conda environment

if [ $USER == 'uai89wz' ]
then
    SCRIPT_DIR=$WORK
    DIRECTORY_PROJECT=$WORK/results_DMS
    module load anaconda-py3/
    source activate $WORK/.conda/envs/py37
else
  DIRECTORY_PROJECT=/scratch/gvignoud/results_DMS
  if [ -z ${OAR_JOB_ID+x} ]
  then
      export PATH="/scratch/gvignoud/miniconda3/bin:$PATH"
      SCRIPT_DIR='/scratch/gvignoud'
  elif [ -z ${SLURM_JOB_ID+x} ]
  then
      export PATH="/home/gvignoud/miniconda3/bin:$PATH"
      SCRIPT_DIR='/home/rioc/gvignoud'
  fi
  sleep $[ ( $RANDOM % 100 )  + 1 ]s
  source activate py37
fi

ARGS_JOB_NAME=$1
ARGS_LIST_NAME=( $2 )
ARGS_LIST=( "${@:3}" )

ARGS_LINE=" --name=$ARGS_JOB_NAME/${ARGS_LIST[0]} "
for ((i=1;i<${#ARGS_LIST_NAME[@]};++i)); do
    name_params="${ARGS_LIST_NAME[$i]}"
    name_params_without="$(echo $name_params)"
    value="${ARGS_LIST[$i]}"
    PARAMS="--${name_params_without}=${value} "
    ARGS_LINE=$ARGS_LINE$PARAMS
done

python $SCRIPT_DIR/numeric_networks/Articles/DMS_DLS/Code/main_DMS.py \
    $ARGS_LINE \
    --save \
    --save_dir $DIRECTORY_PROJECT \

conda deactivate