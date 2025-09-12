#!/bin/bash

# we are skipping config 10 for now
#configs=("01" "02" "03" "04" "05" "06" "07" "08" "09")
configs=("10" "06")

STATE="$1"

if [[ $# -eq 0 ]] ; then
	echo 'no state provided'
	exit 1
fi

for conf in "${configs[@]}"
do
  sbatch --job-name=ag-irr-${STATE}${conf} submit_state_conf.sh "$STATE" "$conf"
  #sbatch --job-name="ag-irr-${STATE// /}_${conf}" submit_state_conf.sh "$STATE" "$conf"
done
