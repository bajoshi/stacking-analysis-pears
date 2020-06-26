#!/bin/bash 

echo "Starting bash script to do template mods at: $(date)"
echo ""
echo "This script will call the Python script -- $1"
echo "that saves each modified template to a numpy array."
echo "This bash script expects the name of the Python script as its first argument."

ncores=5
total_templates=10000
total_par_runs=$((total_templates/ncores))

echo ""
echo "Cores allowed: $ncores"
echo "Total templates: $total_templates"
echo "Total parallel batches: $total_par_runs"
echo ""

counter=0
for (( i = 0; i < $total_par_runs; i++ )); do
    for (( j = i; j < i+$ncores; j++ )); do
        python $1 $counter &
        ((counter++))
    done
    wait
done

echo ""
echo "Finished modifying all templates."
echo "Finished at: $(date)"
echo ""
