#!/bin/bash

DIRECTORY_PROJECT=/scratch/gvignoud/results_single
NAME_PROJECT=$1

if [ $HOSTNAME == 'cleps.CLEPS.eth.cluster' ]
then
    SCRIPT_DIR='/home/gvignoud'
elif [ $HOSTNAME == 'rioc' ]
then
    SCRIPT_DIR='/home/rioc/gvignoud'
fi

RUN_TIME_PROJECT=$(date '+%d/%m/%Y %H:%M:%S')

mkdir -p $DIRECTORY_PROJECT/$NAME_PROJECT

PARAMS_TXT="$SCRIPT_DIR/numeric_networks/Articles/Sequential_Learning_Striatum/Cluster/single/params_pattern_single_$NAME_PROJECT.txt"
NUM_LINES_ARRAY=$(($(sed -n '=' $PARAMS_TXT | wc -l)-1))

cp -r $PARAMS_TXT $DIRECTORY_PROJECT/$NAME_PROJECT
cd $DIRECTORY_PROJECT/$NAME_PROJECT

cd ~/numeric_networks/

echo -e "Script run on $RUN_TIME_PROJECT" >> $DIRECTORY_PROJECT/$NAME_PROJECT/info.txt
echo -e "\n\ngit info\n\n" >> $DIRECTORY_PROJECT/$NAME_PROJECT/info.txt
git status >> $DIRECTORY_PROJECT/$NAME_PROJECT/info.txt
git log -n 1 >> $DIRECTORY_PROJECT/$NAME_PROJECT/info.txt

cd ..

if [ $HOSTNAME == 'cleps.CLEPS.eth.cluster' ]
then
    sbatch  --chdir=$DIRECTORY_PROJECT/$NAME_PROJECT --job-name=$NAME_PROJECT --array 1-$NUM_LINES_ARRAY /home/gvignoud/numeric_networks/Articles/Sequential_Learning_Striatum/Cluster/single/Pattern_Single.batch
elif [ $HOSTNAME == 'rioc' ]
then
    oarsub -S /home/rioc/gvignoud/numeric_networks/Articles/Sequential_Learning_Striatum/Cluster/single/Pattern_Single.batch --directory $DIRECTORY_PROJECT/$NAME_PROJECT --project $NAME_PROJECT --array $NUM_LINES_ARRAY
fi