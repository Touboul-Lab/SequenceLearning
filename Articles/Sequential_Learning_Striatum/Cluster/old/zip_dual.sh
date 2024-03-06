#!/bin/bash

DIRECTORY_PROJECT=/scratch/gvignoud/results_dual
NAME_PROJECT=$1

ZIP_TIME=$(date '+%d/%m/%Y %H:%M:%S')

echo -e "\n\nScript zipped on $ZIP_TIME" >> $DIRECTORY_PROJECT/$NAME_PROJECT/info.txt

cd $DIRECTORY_PROJECT/$NAME_PROJECT

INPUT_OAR_JOBS=$DIRECTORY_PROJECT/$NAME_PROJECT/OAR_JOB_NAMES.txt

echo -e "\n\nOAR info\n\n" >> $DIRECTORY_PROJECT/$NAME_PROJECT/info.txt

while IFS= read -r line
do
  echo -e "$line\n\n" >> $DIRECTORY_PROJECT/$NAME_PROJECT/info.txt
  oarstat -j "$line" -f >> $DIRECTORY_PROJECT/$NAME_PROJECT/info.txt
done < "$INPUT_OAR_JOBS"

cd $DIRECTORY_PROJECT/

zip -rm $NAME_PROJECT.zip $NAME_PROJECT/

cd
