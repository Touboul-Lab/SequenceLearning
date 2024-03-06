#!/bin/bash

RUN_TIME_PROJECT=$(date '+%d/%m/%Y %H:%M:%S')
DIRECTORY_PROJECT='../Models'

echo -e "Script run on $RUN_TIME_PROJECT" >> info.txt
echo -e "\n\ngit info\n\n" >> info.txt
git status >> info.txt
git log -n 1 >> info.txt

if [ -d $DIRECTORY_PROJECT/ ]
  then
    git rm -rf $DIRECTORY_PROJECT/
fi

mkdir $DIRECTORY_PROJECT/

mv info.txt $DIRECTORY_PROJECT/info.txt

python AP/main_AP.py
python AP/main_export_AP.py
