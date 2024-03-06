#!/bin/bash

NAME_PROJECT=$1

if [ $HOSTNAME == 'cleps' ]
then
    SCRIPT_DIR='/scratch/gvignoud'
    DIRECTORY_PROJECT=/scratch/gvignoud/results_DMS
elif [ $HOSTNAME == 'rioc' ]
then
    SCRIPT_DIR='/home/rioc/gvignoud'
    DIRECTORY_PROJECT=/scratch/gvignoud/results_DMS
elif [[ $HOSTNAME =~ ^jean-zay[0-9]$ ]]
then
    SCRIPT_DIR=$WORK
    DIRECTORY_PROJECT=$WORK/results_DMS
fi

ZIP_TIME=$(date '+%d/%m/%Y %H:%M:%S')

echo -e "\n\nScript zipped on $ZIP_TIME" >> $DIRECTORY_PROJECT/$NAME_PROJECT/info.txt

cd $DIRECTORY_PROJECT/$NAME_PROJECT

if [ $HOSTNAME == 'rioc' ]
then
    INPUT_OAR_JOBS=$DIRECTORY_PROJECT/$NAME_PROJECT/OAR_JOB_NAMES.txt
    echo -e "\n\nOAR info\n\n" >> $DIRECTORY_PROJECT/$NAME_PROJECT/info.txt
    while IFS= read -r line
    do
      echo -e "$line\n\n" >> $DIRECTORY_PROJECT/$NAME_PROJECT/info.txt
      oarstat -j "$line" -f >> $DIRECTORY_PROJECT/$NAME_PROJECT/info.txt
    done < "$INPUT_OAR_JOBS"
fi

cd $DIRECTORY_PROJECT/

zip -rm $NAME_PROJECT.zip $NAME_PROJECT/

cd