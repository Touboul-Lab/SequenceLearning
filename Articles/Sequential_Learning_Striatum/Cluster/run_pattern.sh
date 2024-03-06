#!/bin/bash

NAME_PROJECT=$1
RERUN=$2

if [ $HOSTNAME == 'cleps' ]
then
    SCRIPT_DIR='/scratch/gvignoud'
    DIRECTORY_PROJECT=/scratch/gvignoud/pattern
elif [ $HOSTNAME == 'rioc' ]
then
    SCRIPT_DIR='/home/rioc/gvignoud'
    DIRECTORY_PROJECT=/scratch/gvignoud/pattern
elif [[ $HOSTNAME =~ ^jean-zay[0-9]$ ]]
then
    SCRIPT_DIR=$WORK
    DIRECTORY_PROJECT=$WORK/pattern
fi

RUN_TIME_PROJECT=$(date '+%d/%m/%Y %H:%M:%S')

if [ $RERUN == 'FALSE' ]
then
  if [ -d $DIRECTORY_PROJECT/$NAME_PROJECT ]
  then
    rm -r $DIRECTORY_PROJECT/$NAME_PROJECT
  fi

  if [ -e $DIRECTORY_PROJECT/$NAME_PROJECT.zip ]
  then
    rm $DIRECTORY_PROJECT/$NAME_PROJECT.zip
  fi
  mkdir -p $DIRECTORY_PROJECT/$NAME_PROJECT
fi

PARAMS_TXT="$SCRIPT_DIR/numeric_networks/Articles/Sequential_Learning_Striatum/Cluster/params_pattern_$NAME_PROJECT.txt"

NUM_SECTION=$(($(fgrep -o $ $PARAMS_TXT | wc -l)))
NUM_LINES_ARRAY=$(($(sed -n '=' $PARAMS_TXT | wc -l)-3*$NUM_SECTION-3))
DURATION=$(awk 'NR==1 {print $2}' $PARAMS_TXT)
NODE=$(awk 'NR==2 {print $2}' $PARAMS_TXT)
CORES=$(awk 'NR==3 {print $2}' $PARAMS_TXT)

if [ $RERUN == 'FALSE' ]
then
  cp -r $PARAMS_TXT $DIRECTORY_PROJECT/$NAME_PROJECT
fi

cd $SCRIPT_DIR/numeric_networks/

echo -e "Script run on $RUN_TIME_PROJECT" >> $DIRECTORY_PROJECT/$NAME_PROJECT/info.txt
echo -e "\n\ngit info\n\n" >> $DIRECTORY_PROJECT/$NAME_PROJECT/info.txt
git status >> $DIRECTORY_PROJECT/$NAME_PROJECT/info.txt
git log -n 1 >> $DIRECTORY_PROJECT/$NAME_PROJECT/info.txt

cd ..
if [ $HOSTNAME == 'cleps' ]
then
    sbatch  --chdir=$DIRECTORY_PROJECT/$NAME_PROJECT --job-name=$NAME_PROJECT --array 1-$NUM_LINES_ARRAY --time="$DURATION:00:00" --partition=cpu_homogen --nodes=$NODE --cpus-per-task=$CORES $SCRIPT_DIR/numeric_networks/Articles/Sequential_Learning_Striatum/Cluster/Pattern.batch
elif [ $HOSTNAME == 'rioc' ]
then
    oarsub -S $SCRIPT_DIR/numeric_networks/Articles/Sequential_Learning_Striatum/Cluster/Pattern.batch -l "/core=$CORES,walltime=$DURATION:00:00" --directory $DIRECTORY_PROJECT/$NAME_PROJECT --project $NAME_PROJECT --array $NUM_LINES_ARRAY
elif [[ $HOSTNAME =~ ^jean-zay[0-9]$ ]]
then
  QOSduration='qos_cpu-dev'
  if [ $DURATION -gt 2 ]
  then
    QOSduration='qos_cpu-t3'
  fi
  if [ $DURATION -gt 20 ]
  then
    QOSduration='qos_cpu-t4'
  fi
  sbatch --account cvc@cpu --chdir=$DIRECTORY_PROJECT/$NAME_PROJECT --job-name=$NAME_PROJECT --array 1-$NUM_LINES_ARRAY --time="$DURATION:00:00" --partition=cpu_p1 --qos=$QOSduration --nodes=$NODE --cpus-per-task=$CORES --hint=nomultithread $SCRIPT_DIR/numeric_networks/Articles/Sequential_Learning_Striatum/Cluster/Pattern.batch
fi