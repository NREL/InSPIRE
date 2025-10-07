#!/bin/bash

states=(
  "alabama" "arizona" "arkansas" "california" "colorado" "connecticut"
  "delaware" "florida" "georgia" "idaho" "illinois" "indiana"
  "iowa" "kansas" "kentucky" "louisiana" "maine" "maryland"
  "massachusetts" "michigan" "minnesota" "mississippi" "missouri" "montana"
  "nebraska" "nevada" "new_hampshire" "new_jersey" "new_mexico" "new_york"
  "north_carolina" "north_dakota" "ohio" "oklahoma" "oregon" "pennsylvania"
  "rhode_island" "south_carolina" "south_dakota" "tennessee" "texas" "utah"
  "vermont" "virginia" "washington" "west_virginia" "wisconsin" "wyoming"
)

for state in "${states[@]}"
do
  sbatch --job-name=agrivolt-irr-${state} submit_state.sh "$state"
done