#!/bin/bash
DIRECTORY_PROJECT=~/Documents/numeric_networks/Articles/Sequential_Learning_Striatum/Figures/simu
NAME_PROJECT=$1

RUN_TIME_PROJECT=$(date '+%d/%m/%Y %H:%M:%S')

rm -r $DIRECTORY_PROJECT/$NAME_PROJECT

mkdir -p $DIRECTORY_PROJECT/$NAME_PROJECT

PARAMS_TXT="$DIRECTORY_PROJECT/params_figure_pattern_$NAME_PROJECT.txt"

cp -r $PARAMS_TXT $DIRECTORY_PROJECT/$NAME_PROJECT

echo -e "Script run on $RUN_TIME_PROJECT" >> $DIRECTORY_PROJECT/$NAME_PROJECT/info.txt
echo -e "\n\ngit info\n\n" >> $DIRECTORY_PROJECT/$NAME_PROJECT/info.txt
git status >> $DIRECTORY_PROJECT/$NAME_PROJECT/info.txt
git log -n 1 >> $DIRECTORY_PROJECT/$NAME_PROJECT/info.txt

NUM_SECTION=( $(grep -n '\$Figure' $PARAMS_TXT | cut -d: -f 1) )

TEXT=()
while IFS= read -r line; do
  TEXT+=("$line")
done < $PARAMS_TXT
NUM_SECTION+=("$((${#TEXT[@]}+3))")

for ((i=0;i<$((${#NUM_SECTION[@]}-1));++i)); do
  line1=${NUM_SECTION[$i]}
  line2=$((${NUM_SECTION[$(($i+1))]}-2))
  NAME_SECTION=$(sed "${line1}!d" $PARAMS_TXT)
  NAME_SECTION=${NAME_SECTION:1}
  mkdir -p $DIRECTORY_PROJECT/$NAME_PROJECT/$NAME_SECTION
  TEXT_FIGURE=("${TEXT[@]:$line1:$(($line2-$line1+1))}")
  FOR_ARGS_NAME=${TEXT_FIGURE[1]}
  FOR_ARGS_VALUE=${TEXT_FIGURE[2]}

  ARGS_PLOT=("${TEXT_FIGURE[@]:5:$((${#TEXT_FIGURE[@]}-6))}")
  for j in "${ARGS_PLOT[@]}"
  do
    echo $j
  done >$DIRECTORY_PROJECT/$NAME_PROJECT/$NAME_SECTION/info_figures.txt

  python ~/Documents/numeric_networks/Articles/Sequential_Learning_Striatum/Code/main_figure.py --name_project $NAME_PROJECT --name_figure $NAME_SECTION --for_args_name "$FOR_ARGS_NAME" --for_args_value "$FOR_ARGS_VALUE" --args_plot $DIRECTORY_PROJECT/$NAME_PROJECT/$NAME_SECTION/info_figures.txt
done